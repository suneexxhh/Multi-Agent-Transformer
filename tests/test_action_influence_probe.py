"""Controlled causal-mask *policy sensitivity* tests; no environmental causality."""
import unittest
import math

import torch
from torch import nn

from research.dependency_decoder.action_influence_probe import (
    legal_action_kl_probe, combine_order_probes, conservative_precedence_targets,
)
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
        self.assertEqual(decoder.forward_calls, [2, 8])
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
        scores, valid = legal_action_kl_probe(
            decoder, rep, obs, actions, legal, return_valid=True)
        self.assertFalse(bool(scores.any()))  # last agent's legal action is fixed
        self.assertFalse(valid[0, 0, 2].item())  # unknown, NOT measured-zero
        self.assertFalse(valid[0, 1, 2].item())
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

    def test_all_legal_alternatives_are_averaged_and_chunked(self):
        # Decoder responds ONLY to alternative action 2, not action 1.
        # Choosing the first legal alternative would miss the dependency.
        class ActionTwoOnlyDecoder(nn.Module):
            def forward(self, shifted, rep, obs):
                logits = shifted.new_zeros(shifted.shape[0], 3, 3)
                logits[:, 2, 1] = 5.0 * shifted[:, 1, 3]
                return logits

        decoder = ActionTwoOnlyDecoder().eval()
        rep, obs, actions, legal = self._inputs(n_batch=1)
        averaged, valid = legal_action_kl_probe(
            decoder, rep, obs, actions, legal, return_valid=True,
            max_counterfactual_batch=1)
        self.assertGreater(averaged[0, 0, 2].item(), 0.05)
        self.assertTrue(valid[0, 0, 2].item())
        self.assertFalse(valid[0, 2, 0].item())
        self.assertFalse(valid[0, 0, 0].item())

        # Manually exclude action 2 as an alternative: measured zero,
        # but the pair remains identifiable (unlike an untested pair).
        legal[:, 0, 2] = 0
        no_effect, no_effect_valid = legal_action_kl_probe(
            decoder, rep, obs, actions, legal, return_valid=True)
        self.assertLess(no_effect[0, 0, 2].item(), 1e-7)
        self.assertTrue(no_effect_valid[0, 0, 2].item())
        # Lack of any predecessor alternative is UNKNOWN, not a zero edge.
        legal[:, 0, 1] = 0
        untested, unknown = legal_action_kl_probe(
            decoder, rep, obs, actions, legal, return_valid=True)
        self.assertEqual(untested[0, 0, 2].item(), 0)
        self.assertFalse(unknown[0, 0, 2].item())

    def test_order_coverage_and_conservative_targets(self):
        # Only orientation observed in a single order; opposite
        # orientation must NOT be trained as a negative pair.
        first = torch.tensor([[[0., 0.7, 0.4],
                               [0., 0., 0.2],
                               [0., 0., 0.]]])
        valid_first = torch.tensor([[[False, True, True],
                                    [False, False, True],
                                    [False, False, False]]])
        scores, counts = combine_order_probes(
            first.unsqueeze(0), valid_first.unsqueeze(0))
        labels, trainmask = conservative_precedence_targets(scores, counts)
        self.assertFalse(trainmask.any().item())
        # Measured reverse orientation in a second order context.
        second = torch.tensor([[[0., 0., 0.],
                                [0.1, 0., 0.],
                                [0.3, 0., 0.]]])
        valid_second = torch.tensor([[[False, False, False],
                                      [True, False, False],
                                      [True, False, False]]])
        s, c = combine_order_probes(
            torch.stack((first[0], second[0])).unsqueeze(1),
            torch.stack((valid_first[0], valid_second[0])).unsqueeze(1))
        y, mask = conservative_precedence_targets(s, c, margin=0.05)
        self.assertTrue(mask[0, 0, 1].item())
        self.assertEqual(y[0, 0, 1].item(), 1.)
        self.assertEqual(y[0, 1, 0].item(), 0.)
        self.assertTrue(mask[0, 0, 2].item())
        self.assertFalse(mask[0, 1, 2].item())
        self.assertFalse(mask[0].diagonal().any().item())
        with self.assertRaises(ValueError):
            conservative_precedence_targets(s, c, margin=-0.1)

    def test_unpermuted_batchwise_coverage_and_memory_cap(self):
        decoder = OnlyFirstToThirdToyDecoder().eval()
        rep, obs, actions, legal = self._inputs(n_batch=2)
        scores, valid = legal_action_kl_probe(
            decoder, rep, obs, actions, legal, return_valid=True,
            max_counterfactual_batch=3)
        self.assertEqual(scores.shape, (2, 3, 3))
        self.assertEqual(valid.shape, scores.shape)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(decoder.forward_calls, [2, 3, 3, 2])
        self.assertTrue(valid[:, 0, 2].all().item())
        self.assertTrue((scores[:, 0, 2] > 0.25).all().item())

    def test_sub_float32_precision_kl_has_analytic_value(self):
        # A one-logit perturbation of 1e-4 has binary KL
        # log(cosh(1e-4/2)) ~ 1.25e-9. Pure float32
        # log_softmax subtraction can report spurious zero/negative KL.
        class TinyDecoder(nn.Module):
            def forward(self, shifted, rep, obs):
                b, n, dim = shifted.shape
                self_dim = 2
                if n != 2 or dim != 3:
                    raise ValueError("expected 2 agents, 2 legal actions")
                logits = shifted.new_zeros((b, n, self_dim))
                logits[:, 1, 1] = 1e-4 * shifted[:, 1, 2]
                return logits

        decoder = TinyDecoder().eval()
        rep = torch.zeros(1, 2, 3, dtype=torch.float32)
        obs = torch.zeros(1, 2, 4, dtype=torch.float32)
        actions = torch.zeros(1, 2, 1, dtype=torch.long)
        legal = torch.ones(1, 2, 2)
        scores, valid = legal_action_kl_probe(
            decoder, rep, obs, actions, legal, return_valid=True)
        expected = math.log(math.cosh(0.0001 / 2))
        self.assertTrue(valid[0, 0, 1].item())
        self.assertGreater(scores[0, 0, 1].item(), 1e-9)
        self.assertAlmostEqual(scores[0, 0, 1].item(),
                               expected, delta=4e-12)
        self.assertTrue(torch.isfinite(scores).all())
        legal[:, 1] = torch.tensor([1., 0.])
        masked, coverage = legal_action_kl_probe(
            decoder, rep, obs, actions, legal, return_valid=True)
        self.assertTrue(torch.isfinite(masked).all())
        self.assertFalse(coverage[0, 0, 1].item())
        self.assertEqual(masked[0, 0, 1].item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_probe(self):
        decoder = OnlyFirstToThirdToyDecoder().cuda().eval()
        rep, obs, action, legal = (v.cuda() for v in self._inputs())
        scores = legal_action_kl_probe(decoder, rep, obs, action, legal)
        self.assertEqual(scores.device.type, "cuda")
        self.assertGreater(scores[:, 0, 2].min().item(), 0.2)


if __name__ == "__main__":
    unittest.main()
