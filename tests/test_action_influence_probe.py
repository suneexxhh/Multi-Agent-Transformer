"""Controlled causal-mask *policy sensitivity* tests; no environmental causality."""
import unittest

import torch
from torch import nn

from research.dependency_decoder.action_influence_probe import legal_action_kl_probe


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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_probe(self):
        decoder = OnlyFirstToThirdToyDecoder().cuda().eval()
        rep, obs, action, legal = (v.cuda() for v in self._inputs())
        scores = legal_action_kl_probe(decoder, rep, obs, action, legal)
        self.assertEqual(scores.device.type, "cuda")
        self.assertGreater(scores[:, 0, 2].min().item(), 0.2)


if __name__ == "__main__":
    unittest.main()
