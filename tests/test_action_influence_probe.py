"""Controlled causal-mask *policy sensitivity* tests; no environmental causality."""
import unittest

import torch
from torch import nn

from research.dependency_decoder.action_influence_probe import legal_action_kl_probe
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer


class OnlyFirstToThirdToyDecoder(nn.Module):
    """One known dependency: first decoded action changes third's logits."""
    def __init__(self, n_actions=3):
        super().__init__()
        self.a = n_actions
        self.forward_calls = []

    def forward(self, shifted, rep, obs):
        self.forward_calls.append(shifted.shape[0])
        b, n = shifted.shape[:2]
        logits = shifted.new_zeros(b, n, self.a)
        if n >= 3:
            logits[:, 2] = 3 * shifted[:, 1, 1:]
        return logits


class TestActionInfluenceProbe(unittest.TestCase):
    @staticmethod
    def _inputs(n_batch=2):
        rep = torch.zeros(n_batch, 3, 3)
        obs = torch.zeros(n_batch, 3, 4)
        action = torch.zeros(n_batch, 3, 1, dtype=torch.long)
        masks = torch.ones(n_batch, 3, 3)
        return rep, obs, action, masks

    def test_only_known_conditional_edge_and_two_forward_passes(self):
        decoder = OnlyFirstToThirdToyDecoder().eval()
        rep, obs, actions, legal = self._inputs()
        scores = legal_action_kl_probe(decoder, rep, obs, actions, legal)
        self.assertEqual(scores.shape, (2, 3, 3))
        self.assertEqual(decoder.forward_calls, [2, 4])
        self.assertTrue(bool((scores[:, 0, 2] > 0.25).all()))
        masked = scores.clone()
        masked[:, 0, 2] = 0
        self.assertTrue(torch.equal(masked, torch.zeros_like(masked)))

    def test_original_agent_ids_restored_on_both_axes(self):
        decoder = OnlyFirstToThirdToyDecoder().eval()
        rep, obs, actions, legal = self._inputs()
        order = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.long)
        scores = legal_action_kl_probe(
            decoder, rep, obs, actions, legal, agent_order=order)
        # Edges are [original first, original third] in each batch.
        self.assertGreater(scores[0, 2, 1].item(), 0.2)
        self.assertGreater(scores[1, 1, 0].item(), 0.2)
        scores[0, 2, 1] = 0
        scores[1, 1, 0] = 0
        self.assertFalse(bool(scores.any()))

    def test_legal_actions_and_eval_mode_are_required(self):
        decoder = OnlyFirstToThirdToyDecoder().eval()
        rep, obs, actions, legal = self._inputs(n_batch=1)
        # No legal alternative for the first agent: no measurable contrast.
        legal[:, 0] = torch.tensor([1., 0., 0.])
        scores = legal_action_kl_probe(decoder, rep, obs, actions, legal)
        self.assertFalse(bool(scores.any()))
        legal[:, 0] = 1
        legal[:, 2] = torch.tensor([1., 0., 0.])
        scores = legal_action_kl_probe(decoder, rep, obs, actions, legal)
        self.assertFalse(bool(scores.any()))  # last agent's legal action is fixed
        decoder.train()
        with self.assertRaises(ValueError):
            legal_action_kl_probe(decoder, rep, obs, actions, legal)
        decoder.eval()
        legal[:, 2] = 1
        with self.assertRaises(ValueError):
            legal_action_kl_probe(
                decoder, rep, obs, actions, legal,
                agent_order=torch.tensor([[0, 0, 1]]))

    def test_real_mat_masked_decoder_interface(self):
        # Ensure the probe works on the actual MAT Decoder, not only a toy.
        torch.manual_seed(710)
        model = MultiAgentTransformer(
            state_dim=9, obs_dim=7, action_dim=4, n_agent=3,
            n_block=1, n_embd=16, n_head=1,
            action_type="Discrete").eval()
        rep = torch.randn(2, 3, 16)
        obs = torch.randn(2, 3, 7)
        actions = torch.tensor([[[0], [1], [2]], [[2], [1], [0]]])
        legal = torch.ones(2, 3, 4)
        order = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.long)
        scores = legal_action_kl_probe(
            model.decoder, rep, obs, actions, legal, agent_order=order)
        self.assertEqual(scores.shape, (2, 3, 3))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((scores >= 0).all())
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(3).expand_as(order))
        impossible = ranks.unsqueeze(2) >= ranks.unsqueeze(1)
        self.assertTrue(torch.equal(
            scores.masked_select(impossible),
            torch.zeros_like(scores).masked_select(impossible)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_probe(self):
        decoder = OnlyFirstToThirdToyDecoder().cuda().eval()
        rep, obs, action, legal = (v.cuda() for v in self._inputs())
        scores = legal_action_kl_probe(decoder, rep, obs, action, legal)
        self.assertEqual(scores.device.type, "cuda")
        self.assertGreater(scores[:, 0, 2].min().item(), 0.2)


if __name__ == "__main__":
    unittest.main()
