"""Nonlinear conditional action interactions and counterfactual observability."""
import unittest

import torch

from mat.algorithms.mat.algorithm.dependency_ordering import PairwisePrecedenceScorer
from research.dependency_decoder.action_influence_probe import legal_action_kl_probe
from research.dependency_decoder.nonlinear_history_oracle import (
    TwoParentInteractionDecoder, max_sensitivity_across_histories,
)


def fixture(batch=2):
    ids = torch.eye(3).unsqueeze(0).expand(batch, -1, -1).clone()
    observations = torch.zeros(batch, 3, 2)
    legal = torch.ones(batch, 3, 3)
    return ids, observations, legal


class TestNonlinearHistoryOracle(unittest.TestCase):
    def test_and_requires_joint_predecessor_context(self):
        """A real conditional dependency can have zero KL in one context."""
        decoder = TwoParentInteractionDecoder("and").eval()
        ids, obs, legal = fixture()
        contexts = [
            torch.tensor([0, 0, 0]),
            torch.tensor([2, 0, 0]),
            torch.tensor([2, 2, 0]),
        ]
        results, validity = [], []
        for history in contexts:
            actions = history.expand(2, -1).unsqueeze(-1).clone()
            values, mask = legal_action_kl_probe(
                decoder, ids, obs, actions, legal, return_valid=True)
            results.append(values)
            validity.append(mask)
        # With (0,0), toggling either parent alone cannot activate AND.
        self.assertEqual(results[0][0, 0, 2].item(), 0.)
        self.assertEqual(results[0][0, 1, 2].item(), 0.)
        self.assertTrue(validity[0][:, 0, 2].all().item())
        self.assertTrue(validity[0][:, 1, 2].all().item())
        # With (2,0), parent 1 matters and parent 0 does not.
        self.assertLess(results[1][:, 0, 2].max().item(), 1e-7)
        self.assertGreater(results[1][:, 1, 2].min().item(), 0.1)
        # With (2,2), changing EITHER active parent breaks conjunction.
        self.assertGreater(results[2][:, 0, 2].min().item(), 0.1)
        self.assertGreater(results[2][:, 1, 2].min().item(), 0.1)
        maximum, counts = max_sensitivity_across_histories(
            torch.stack(results), torch.stack(validity))
        self.assertEqual(counts[0, 0, 2].item(), 3)
        self.assertGreater(maximum[0, 0, 2].item(), 0.1)
        self.assertGreater(maximum[0, 1, 2].item(), 0.1)
        # Both are known parent edges, yet a single all-zero rollout
        # makes the sensitivity test miss BOTH of them.
        print("NONLINEAR_AND zero_history_recall=0.0 "
              "varied_history_recall=1.0 expected_edges=2", flush=True)

    def test_xor_changes_sign_but_kl_only_detects_difference(self):
        """Sensitivity KL is nonnegative, cannot infer effect direction."""
        decoder = TwoParentInteractionDecoder("xor").eval()
        ids, obs, legal = fixture(batch=1)
        results = []
        for baseline in ((0, 0, 0), (2, 0, 0), (2, 2, 0)):
            actions = torch.tensor(baseline, dtype=torch.long).view(1, 3, 1)
            scores, mask = legal_action_kl_probe(
                decoder, ids, obs, actions, legal, return_valid=True)
            self.assertTrue(mask[0, 0, 2].item())
            self.assertTrue(mask[0, 1, 2].item())
            self.assertTrue((scores >= 0).all().item())
            results.append(scores)
        self.assertTrue(all(x[0, 0, 2].item() > 0.05 for x in results))
        self.assertTrue(all(x[0, 1, 2].item() > 0.05 for x in results))
        self.assertEqual(results[0][0, 2, 0].item(), 0)

    def test_order_and_legal_action_support_limit_identification(self):
        decoder = TwoParentInteractionDecoder("and").eval()
        ids, obs, legal = fixture(batch=1)
        actions = torch.tensor([[[2], [2], [0]]])
        reverse = torch.tensor([[2, 1, 0]], dtype=torch.long)
        scores, valid = legal_action_kl_probe(
            decoder, ids, obs, actions, legal, reverse, return_valid=True)
        self.assertEqual(scores[0, 0, 2].item(), 0.)
        self.assertFalse(valid[0, 0, 2].item())
        self.assertFalse(valid[0, 1, 2].item())
        # Restrict action 2 for agent 0. A different legal action exists
        # (action 1), but can never activate the AND gate from baseline 0.
        restricted = legal.clone()
        restricted[:, 0, 2] = 0
        a0 = torch.zeros_like(actions)
        kl, valid = legal_action_kl_probe(
            decoder, ids, obs, a0, restricted, return_valid=True)
        self.assertTrue(valid[0, 0, 2].item())
        self.assertEqual(kl[0, 0, 2].item(), 0.)
        # A target with only one legal action is explicitly unknown.
        target_restricted = legal.clone()
        target_restricted[:, 2] = torch.tensor([1., 0., 0.])
        scores, valid = legal_action_kl_probe(
            decoder, ids, obs, actions, target_restricted,
            return_valid=True)
        self.assertFalse(valid[0, 0, 2].item())
        self.assertFalse(valid[0, 1, 2].item())

    def test_deterministic_action_masks_skip_all_decoder_calls(self):
        decoder = TwoParentInteractionDecoder("and").eval()
        ids, obs, _ = fixture(batch=1)
        legal = torch.tensor([[[1., 0., 0.]]]).expand(1, 3, 3)
        acts = torch.zeros(1, 3, 1, dtype=torch.long)
        scores, valid = legal_action_kl_probe(
            decoder, ids, obs, acts, legal, return_valid=True)
        self.assertFalse(valid.any().item())
        self.assertEqual(scores.sum().item(), 0.)
        self.assertEqual(decoder.forward_calls, [])

    def test_observation_only_precedence_cannot_express_action_context(self):
        """Same observations, different action histories, different KL."""
        torch.manual_seed(9401)
        scorer = PairwisePrecedenceScorer(obs_dim=2, hidden_dim=2)
        decoder = TwoParentInteractionDecoder("and").eval()
        ids, obs, legal = fixture(batch=1)
        p_before = scorer(obs).detach().clone()
        a0 = torch.zeros(1, 3, 1, dtype=torch.long)
        a2 = torch.tensor([[[2], [2], [0]]])
        c0 = legal_action_kl_probe(decoder, ids, obs, a0, legal)
        c2 = legal_action_kl_probe(decoder, ids, obs, a2, legal)
        p_after = scorer(obs)
        self.assertTrue(torch.equal(p_before, p_after))
        self.assertEqual(c0[0, 0, 2].item(), 0.)
        self.assertGreater(c2[0, 0, 2].item(), 0.1)
        # A scorer conditioned ONLY on fixed observations is incapable
        # of representing this history-dependent dependency change.
        self.assertTrue(torch.allclose(p_before, torch.full_like(p_before, 0.5)))


if __name__ == "__main__":
    unittest.main()
