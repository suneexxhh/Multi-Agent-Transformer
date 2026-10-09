"""Expectations over action histories: weights, coverage, and PPO safety."""
import unittest

import torch

from research.dependency_decoder.action_influence_probe import (
    legal_action_kl_probe,
)
from research.dependency_decoder.nonlinear_history_oracle import (
    TwoParentInteractionDecoder,
)
from research.dependency_decoder.expected_action_dependency import (
    probe_action_histories, expected_dependency, reliable_precedence_targets,
)


def setup(batch=1):
    ids = torch.eye(3).unsqueeze(0).expand(batch, -1, -1).clone()
    obs = torch.zeros(batch, 3, 2)
    legal = torch.ones(batch, 3, 3)
    return ids, obs, legal


class TestExpectedActionDependency(unittest.TestCase):
    def test_weighted_expectation_tracks_other_parent_activation(self):
        """Known p(a1=2) changes expected 0->2 sensitivity but not observation."""
        rep, obs, legal = setup(batch=3)
        decoder = TwoParentInteractionDecoder("and").eval()
        histories = torch.zeros(2, 3, 3, 1, dtype=torch.long)
        histories[1, :, 1] = 2  # on only in history 1
        scores, valid = probe_action_histories(
            decoder, rep, obs, histories, legal, max_context_batch=2,
            max_counterfactual_batch=2)
        self.assertEqual(scores.shape, (2, 3, 3, 3))
        self.assertTrue(valid[:, :, 0, 2].all().item())
        self.assertEqual(scores[0, :, 0, 2].abs().sum().item(), 0)
        self.assertGreater(scores[1, :, 0, 2].min().item(), 0.2)

        probabilities = torch.tensor([0.10, 0.50, 0.90])
        weights = torch.stack([1 - probabilities, probabilities])
        mean, coverage, ess = expected_dependency(scores, valid, weights)
        oracle_on = scores[1, :, 0, 2]
        self.assertTrue(torch.allclose(mean[:, 0, 2],
                                       probabilities * oracle_on, atol=1e-6))
        self.assertTrue(torch.allclose(coverage[:, 0, 2],
                                       torch.ones(3)))
        theoretical_ess = 1 / (probabilities.square() +
                               (1-probabilities).square())
        self.assertTrue(torch.allclose(ess[:, 0, 2],
                                       theoretical_ess, atol=1e-5))
        self.assertEqual(mean[:, 1, 2].abs().sum().item(), 0)
        self.assertTrue(torch.all(mean[:, 0, 2].diff() > 0))
        print("EXPECTED_AND p_a1_active=(0.1,0.5,0.9) "
              "weighted_dependence_monotone=PASS", flush=True)

    def test_coverage_cannot_treat_unmeasured_contexts_as_zero(self):
        rep, obs, legal = setup()
        decoder = TwoParentInteractionDecoder("and").eval()
        histories = torch.zeros(2, 1, 3, 1, dtype=torch.long)
        histories[:, :, 1] = 2  # 0->2 is active in both
        legal_by_history = legal.unsqueeze(0).repeat(2, 1, 1, 1)
        legal_by_history[0, :, 2] = torch.tensor([1., 0., 0.])
        scores, valid = probe_action_histories(
            decoder, rep, obs, histories, legal_by_history,
            max_context_batch=1)
        self.assertFalse(valid[0, 0, 0, 2].item())
        self.assertTrue(valid[1, 0, 0, 2].item())
        weights = torch.tensor([[0.75], [0.25]])
        mean, coverage, ess = expected_dependency(scores, valid, weights)
        self.assertTrue(torch.allclose(mean[0, 0, 2], scores[1, 0, 0, 2]))
        self.assertAlmostEqual(coverage[0, 0, 2].item(), 0.25, places=5)
        self.assertAlmostEqual(ess[0, 0, 2].item(), 1.0, places=5)
        self.assertGreater(mean[0, 0, 2].item(), 0.2)
        self.assertFalse(valid[0, 0, 0, 2].item())

    def test_opposite_direction_requires_independent_order_coverage(self):
        rep, obs, legal = setup()
        decoder = TwoParentInteractionDecoder("and").eval()
        h = 6
        histories = torch.zeros(h, 1, 3, 1, dtype=torch.long)
        histories[:, :, 1] = 2
        ascending = torch.tensor([0, 1, 2])
        descending = ascending.flip(0)
        orders = torch.stack([ascending] * 3 + [descending] * 3).unsqueeze(1)
        scores, valid = probe_action_histories(
            decoder, rep, obs, histories, legal, orders,
            max_context_batch=2)
        self.assertTrue(valid[:3, :, 0, 2].all().item())
        self.assertTrue(valid[3:, :, 2, 0].all().item())
        avg, coverage, ess = expected_dependency(scores, valid)
        self.assertAlmostEqual(coverage[0, 0, 2].item(), 0.5)
        self.assertAlmostEqual(coverage[0, 2, 0].item(), 0.5)
        self.assertAlmostEqual(ess[0, 0, 2].item(), 3.0)
        label, mask = reliable_precedence_targets(
            avg, coverage, ess, min_coverage=0.5, min_ess=2, margin=0.05)
        self.assertTrue(mask[0, 0, 2].item())
        self.assertTrue(mask[0, 2, 0].item())
        self.assertEqual(label[0, 0, 2].item(), 1)
        self.assertEqual(label[0, 2, 0].item(), 0)
        _, strict = reliable_precedence_targets(
            avg, coverage, ess, min_coverage=0.6, min_ess=2)
        self.assertFalse(strict.any().item())
        _, few = reliable_precedence_targets(
            avg, coverage, ess, min_coverage=0.5, min_ess=4)
        self.assertFalse(few.any().item())
        # The present test does not justify comparing conditional
        # factorizations from distinct MAT policies.
        print("EXPECTED_AND bidirectional_observation="
              "0.5 ess=3 conservative_labels=PASS", flush=True)

    def test_context_batching_matches_individual_calls(self):
        rep, obs, legal = setup(batch=2)
        decoder = TwoParentInteractionDecoder("xor").eval()
        torch.manual_seed(9501)
        histories = (2 * torch.randint(0, 2, (4, 2, 3, 1))).long()
        histories[:, :, 2] = 0
        a, av = probe_action_histories(
            decoder, rep, obs, histories, legal,
            max_context_batch=1, max_counterfactual_batch=3)
        b, bv = probe_action_histories(
            decoder, rep, obs, histories, legal,
            max_context_batch=8, max_counterfactual_batch=128)
        self.assertTrue(torch.equal(av, bv))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        # Aggregation must be invariant to scaling all weights.
        weights = torch.rand(4, 2)
        estimate = expected_dependency(a, av, weights)
        scaled = expected_dependency(a, av, weights * 9)
        for lhs, rhs in zip(estimate, scaled):
            self.assertTrue(torch.allclose(lhs, rhs, atol=1e-6))

    def test_all_missing_is_unknown_and_reject_bad_input(self):
        scores = torch.zeros(3, 2, 3, 3)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mean, coverage, ess = expected_dependency(scores, mask)
        self.assertFalse(coverage.any().item())
        self.assertFalse(ess.any().item())
        self.assertFalse(mean.any().item())
        with self.assertRaises(ValueError):
            expected_dependency(scores, mask, torch.zeros(3, 2))
        with self.assertRaises(ValueError):
            expected_dependency(scores, mask, torch.tensor(
                [[1., -1.], [2., 2.], [1., 1.]]))
        with self.assertRaises(ValueError):
            expected_dependency(scores, mask, torch.ones(3, 3))
        with self.assertRaises(ValueError):
            reliable_precedence_targets(mean, coverage, ess, min_ess=0)
        rep, obs, legal = setup()
        histories = torch.zeros(2, 1, 3, 1, dtype=torch.long)
        with self.assertRaises(ValueError):
            probe_action_histories(TwoParentInteractionDecoder().eval(),
                                   rep, obs, histories, legal,
                                   max_context_batch=0)


if __name__ == "__main__":
    unittest.main()
