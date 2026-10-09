"""Matched-context policy sensitivity checks; no SMAC process or PPO."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from research.dependency_decoder.snapshot_capture import MultiContextCapture
from research.dependency_decoder.paired_checkpoint_diagnostic import (
    compare_frozen_checkpoints,
)


class TestPairedCheckpointDiagnostic(unittest.TestCase):
    @staticmethod
    def fixtures(path):
        torch.manual_seed(9931)
        model = MultiAgentTransformer(
            state_dim=37, obs_dim=7, action_dim=4, n_agent=3,
            n_block=1, n_embd=8, n_head=1,
            action_type="Discrete").eval()
        obs = np.random.default_rng(9932).normal(size=(1, 3, 7)).astype(np.float32)
        legal = np.ones((1, 3, 4), dtype=np.float32)
        order = np.array([[2, 0, 1]], dtype=np.int64)
        collector = MultiContextCapture((0, 10, 20, 30))
        for step in (0, 10, 20, 30):
            collector.observe(step, obs + step * 0.01, legal, order)
        snapshot, early = collector.save(model, path, "MAT_paired_early", 1)
        with torch.no_grad():
            # Force a meaningful actor-parameter shift, while preserving
            # architecture, action support, and original agent IDs.
            for p in model.decoder.parameters():
                p.add_(0.05 * torch.randn_like(p))
        # Save a second checkpoint with the same test-context interface.
        stage2 = MultiContextCapture((0, 10, 20, 30))
        for step in (0, 10, 20, 30):
            stage2.observe(step, obs, legal, order)
        _, late = stage2.save(model, path, "MAT_paired_late", 10)
        return snapshot, early, late

    def test_same_weights_produce_zero_matched_dependency_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, early, _ = self.fixtures(tmp)
            stats = compare_frozen_checkpoints(
                early, early, snapshot, 1, 8, 1,
                histories=5, max_context_batch=2,
                max_counterfactual_batch=8, seed=9933)
            self.assertEqual(stats["observations"], 4)
            self.assertEqual(stats["measurable_pairs"], 12)
            self.assertEqual(stats["max_absolute_parameter_shift"], 0)
            self.assertAlmostEqual(stats["max_absolute_joint_logp_shift"], 0)
            self.assertLess(stats["sampled_vs_replay_max_error"], 2e-5)
            self.assertAlmostEqual(stats["mean_absolute_matched_pair_kl_change"], 0)
            self.assertEqual(stats["contexts_with_adequate_ess"], 4)
            self.assertTrue(all(abs(v - 5) < 1e-5 for v in
                                stats["importance_ess_by_context"]))
            self.assertEqual(len(stats["per_context_mean_absolute_kl_change"]), 4)

    def test_parameter_shift_on_exact_same_context_is_detectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, early, late = self.fixtures(tmp)
            stats = compare_frozen_checkpoints(
                early, late, snapshot, 1, 8, 1,
                histories=6, max_context_batch=2,
                max_counterfactual_batch=8, seed=9934)
            self.assertGreater(stats["max_absolute_parameter_shift"], 0)
            self.assertGreater(stats["max_absolute_joint_logp_shift"], 0)
            self.assertGreaterEqual(
                stats["mean_absolute_matched_pair_kl_change"], 0)
            self.assertLess(stats["sampled_vs_replay_max_error"], 2e-5)
            self.assertEqual(stats["measurable_pairs"], 12)
            self.assertEqual(len(stats["importance_ess_by_context"]), 4)

    def test_low_importance_ess_abstains_from_snis_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, early, late = self.fixtures(tmp)
            stats = compare_frozen_checkpoints(
                early, late, snapshot, 1, 8, 1,
                histories=4, min_importance_ess=5.0,
                max_context_batch=2, max_counterfactual_batch=8)
            self.assertEqual(stats["contexts_with_adequate_ess"], 0)
            self.assertEqual(stats["snis_measurable_pairs_with_adequate_ess"], 0)
            self.assertIsNone(stats["snis_late_kl_mean"])
            self.assertIsNotNone(stats["matched_late_mean_kl"])
            with self.assertRaises((RuntimeError, ValueError)):
                compare_frozen_checkpoints(
                    early, late, snapshot, 2, 8, 1,
                    histories=3)


if __name__ == "__main__":
    unittest.main()
