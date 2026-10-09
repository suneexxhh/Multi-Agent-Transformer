"""Controlled identification test: policy dependence versus learned precedence.

No claims about SMAC, optimal joint return or environment causality.
The toy decoder has exactly one fixed directed action dependency (0 -> 2).
It is evaluated under BOTH original-agent orderings, since a single
autoregressive order cannot measure reverse conditional sensitivity.
"""
import unittest

import torch
import torch.nn as nn

from mat.algorithms.mat.algorithm.dependency_ordering import PairwisePrecedenceScorer
from research.dependency_decoder.action_influence_probe import (
    legal_action_kl_probe,
    combine_order_probes,
    conservative_precedence_targets,
)


class KnownZeroToTwoDecoder(nn.Module):
    """Only the preceding action of original agent 0 can affect agent 2."""

    def forward(self, shifted_actions, obs_rep, obs):
        b, n, width = shifted_actions.shape
        if width != 4 or obs_rep.shape != (b, n, 3):
            raise ValueError("3 discrete actions and one-hot agent IDs expected")
        # At position k, shifted_actions[k] encodes the PREVIOUS
        # position's action. This is the original MAT shift convention.
        shifted_agent0 = torch.nn.functional.pad(
            obs_rep[:, :-1, 0], (1, 0))
        predecessor_action2 = shifted_actions[:, :, 3]
        influence = (shifted_agent0 * predecessor_action2).cumsum(dim=1)
        logits = shifted_actions.new_zeros((b, n, 3))
        logits[:, :, 1] = 5.0 * influence * obs_rep[:, :, 2]
        return logits


class TestDirectionalToyOracle(unittest.TestCase):

    def test_oracle_labels_and_scoring_network_fit(self):
        torch.manual_seed(730)
        decoder = KnownZeroToTwoDecoder().eval()
        batch = 4
        agent_ids = torch.eye(3).unsqueeze(0).expand(batch, 3, 3).clone()
        actions = torch.zeros(batch, 3, 1, dtype=torch.long)
        legal = torch.ones(batch, 3, 3)
        order_forward = torch.tensor([[0, 1, 2]]).expand(batch, 3)
        order_reverse = torch.tensor([[2, 1, 0]]).expand(batch, 3)

        forward_score, forward_valid = legal_action_kl_probe(
            decoder, agent_ids, agent_ids, actions, legal,
            agent_order=order_forward, return_valid=True)
        reverse_score, reverse_valid = legal_action_kl_probe(
            decoder, agent_ids, agent_ids, actions, legal,
            agent_order=order_reverse, return_valid=True)

        # A direction unmeasured under one ordering is NOT a zero edge.
        self.assertFalse(forward_valid[:, 2, 0].any().item())
        self.assertFalse(reverse_valid[:, 0, 2].any().item())
        self.assertTrue((forward_score[:, 0, 2] > 0.4).all().item())
        self.assertTrue((reverse_score[:, 2, 0] == 0).all().item())

        mean_score, count = combine_order_probes(
            torch.stack((forward_score, reverse_score)),
            torch.stack((forward_valid, reverse_valid)))
        target, valid_pairs = conservative_precedence_targets(
            mean_score, count, margin=0.1)
        expected = torch.zeros_like(valid_pairs)
        expected[:, 0, 2] = True
        expected[:, 2, 0] = True
        self.assertTrue(torch.equal(valid_pairs, expected))
        self.assertTrue(torch.all(target[:, 0, 2] == 1))
        self.assertTrue(torch.all(target[:, 2, 0] == 0))

        scorer = PairwisePrecedenceScorer(obs_dim=3, hidden_dim=2)
        optimizer = torch.optim.Adam(scorer.parameters(), lr=0.04)
        before = scorer.supervised_loss(agent_ids, target, valid_pairs).item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = scorer.supervised_loss(agent_ids, target, valid_pairs)
            loss.backward()
            optimizer.step()
        final = scorer.supervised_loss(agent_ids, target, valid_pairs).item()
        self.assertLess(final, 0.35)
        self.assertLess(final, before * 0.6)
        with torch.no_grad():
            preference = scorer(agent_ids)
            self.assertTrue((preference[:, 0, 2] > 0.8).all().item())
            # Nothing here establishes the 0-vs-1 or 1-vs-2 order.
            # The test intentionally checks ONLY the externally
            # identified edge 0 -> 2.


if __name__ == "__main__":
    unittest.main()
