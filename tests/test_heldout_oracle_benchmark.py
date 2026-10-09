"""Held-out toy graph validation: recovery, false positives and transfer.

This tests a known synthetic ACTION-CONDITIONAL decoder structure, not
causality in StarCraft II or learned coordination effectiveness.
"""
import unittest

import torch

from mat.algorithms.mat.algorithm.dependency_ordering import PairwisePrecedenceScorer
from research.dependency_decoder.heldout_oracle_benchmark import (
    make_priority_graph, probe_known_graphs, direction_metrics,
    calibrate_scorer_threshold, scorer_graph_metrics,
)
from research.dependency_decoder.action_influence_probe import (
    conservative_precedence_targets,
)


class TestHeldoutOracleBenchmark(unittest.TestCase):

    def test_unseen_dags_have_perfect_conditional_probe_recovery(self):
        """Hold out graphs across 3, 4, and 5 agents with varied edges."""
        torch.manual_seed(8301)
        for n in (3, 4, 5):
            with self.subTest(n=n):
                traits = torch.rand(16, n, 1) * 2 - 1
                graph = make_priority_graph(traits, min_gap=0.55)
                self.assertTrue(graph.any().item())
                scores, counts = probe_known_graphs(graph, chunk=17)
                stats = direction_metrics(scores, counts, graph, margin=0.05)
                self.assertEqual(stats["coverage"], 1.0)
                self.assertEqual(stats["fp"], 0)
                self.assertEqual(stats["fn"], 0)
                self.assertEqual(stats["tp"], int(graph.sum()))
                self.assertEqual(stats["false_positive_rate"], 0)
                self.assertEqual(stats["recall"], 1.0)

    def test_counterfactual_action_support_can_hide_true_edges(self):
        """A mathematically identifiable contrast may still be powerless.

        The toy environment offers action 1 but action 2 is the only
        action which activates an edge; remove action 2 from legal set.
        This demonstrates that zero conditional KL is NOT no true edge.
        """
        traits = torch.tensor([[[1.0], [-0.3], [-1.0]]])
        graph = make_priority_graph(traits)
        scores, counts = probe_known_graphs(graph)
        baseline = direction_metrics(scores, counts, graph, 0.05)
        self.assertGreater(baseline["tp"], 0)
        legal = torch.tensor([[[1., 1., 0.]]]).expand(1, 3, 3)
        restricted_scores, restricted_counts = probe_known_graphs(graph, legal=legal)
        limited = direction_metrics(restricted_scores, restricted_counts, graph, 0.05)
        self.assertEqual(limited["coverage"], 1.0)
        self.assertEqual(limited["tp"], 0)
        self.assertEqual(limited["fn"], int(graph.sum()))
        self.assertEqual(limited["fp"], 0)

    def test_calibration_and_unseen_agent_count(self):
        """Train ONLY from synthetic supervised oracle labels on N=4.

        Calibration uses an independent N=4 split; the final metrics
        use N=5 previously unseen graphs with independently drawn traits.
        The scorer receives traits, not graph adjacency or true labels.
        """
        torch.manual_seed(8302)
        train_traits = torch.rand(72, 4, 1) * 2 - 1
        cal_traits = torch.rand(48, 4, 1) * 2 - 1
        test_traits = torch.rand(56, 5, 1) * 2 - 1
        train_graph = make_priority_graph(train_traits)
        cal_graph = make_priority_graph(cal_traits)
        test_graph = make_priority_graph(test_traits)

        train_scores, train_counts = probe_known_graphs(train_graph, chunk=64)
        target, observed = conservative_precedence_targets(
            train_scores, train_counts, margin=0.05)
        self.assertEqual(int(observed.sum()), 2*int(train_graph.sum()))
        scorer = PairwisePrecedenceScorer(obs_dim=1, hidden_dim=2)
        opt = torch.optim.Adam(scorer.parameters(), lr=0.045)
        for _ in range(90):
            opt.zero_grad()
            loss = scorer.supervised_loss(train_traits, target, observed)
            loss.backward()
            opt.step()

        with torch.no_grad():
            validation = scorer(cal_traits)
            threshold = calibrate_scorer_threshold(
                validation, cal_graph, max_false_positive_rate=0.05)
            val_stats = scorer_graph_metrics(validation, cal_graph, threshold)
            heldout = scorer(test_traits)
            test_stats = scorer_graph_metrics(heldout, test_graph, threshold)
        self.assertLessEqual(val_stats["false_positive_rate"], 0.05)
        self.assertLess(test_stats["false_positive_rate"], 0.12)
        self.assertGreater(test_stats["recall"], 0.70)
        self.assertGreater(test_stats["precision"], 0.75)
        self.assertGreater(threshold, 0.5)
        print(
            "HELDOUT_SCORER n_train=4 n_test=5 "
            f"threshold={threshold:.5f} "
            f"cal_FPR={val_stats['false_positive_rate']:.5f} "
            f"test_FPR={test_stats['false_positive_rate']:.5f} "
            f"test_precision={test_stats['precision']:.5f} "
            f"test_recall={test_stats['recall']:.5f} "
            f"test_TP={test_stats['tp']} test_FP={test_stats['fp']}",
            flush=True)


    def test_rule_reversal_is_explicit_negative_control(self):
        # A ranker trained on larger-trait-first cannot be expected to
        # generalize to the OPPOSITE structural rule; correct held-out
        # scores on the original rule do not establish causal discovery.
        torch.manual_seed(8303)
        traits = torch.rand(20, 5, 1) * 2 - 1
        reversed_graph = make_priority_graph(-traits)
        scorer = PairwisePrecedenceScorer(obs_dim=1, hidden_dim=1)
        with torch.no_grad():
            scorer.query.weight.fill_(1)
            scorer.query.bias.zero_()
            scorer.key.weight.zero_()
            scorer.key.bias.fill_(1)
            p = scorer(traits)
            wrong_rule = scorer_graph_metrics(p, reversed_graph, 0.55)
        self.assertTrue(reversed_graph.any().item())
        self.assertEqual(wrong_rule["recall"], 0.0)
        self.assertGreater(wrong_rule["false_positive_rate"], 0.0)
        print(
            "OOD_REVERSED_RULE negative_control "
            f"recall={wrong_rule['recall']:.5f} "
            f"FPR={wrong_rule['false_positive_rate']:.5f}",
            flush=True)

    def test_metrics_distinguish_false_alarm_from_abstention(self):
        traits = torch.tensor([[[1.], [0.], [-1.]]])
        graph = make_priority_graph(traits)
        scores, counts = probe_known_graphs(graph)
        measured = direction_metrics(scores, counts, graph)
        self.assertEqual(measured["fp"], 0)
        # Inject a spurious directional KL in a known nonedge 1 -> 0.
        corrupt = scores.clone()
        corrupt[:, 1, 0] = 30
        bad = direction_metrics(corrupt, counts, graph)
        self.assertGreater(bad["fp"], 0)
        # Remove all measurements; unknowns are NOT negative samples.
        unseen = direction_metrics(scores, torch.zeros_like(counts), graph)
        self.assertEqual(unseen["coverage"], 0)
        self.assertEqual(unseen["fp"], 0)
        self.assertEqual(unseen["tn"], 0)
        self.assertEqual(unseen["tp"], 0)


if __name__ == "__main__":
    unittest.main()
