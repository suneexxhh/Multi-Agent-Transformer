"""Opt-in MAT snapshot capture tests; no StarCraftII environment required."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from research.dependency_decoder.checkpoint_diagnostic import run_diagnostic
from research.dependency_decoder.snapshot_capture import (
    save_snapshot_and_checkpoint, MultiContextCapture,
)


class TestMATSnapshotCapture(unittest.TestCase):
    def setup_snapshot(self):
        model = MultiAgentTransformer(
            state_dim=37, obs_dim=7, action_dim=4, n_agent=3,
            n_block=1, n_embd=8, n_head=1,
            action_type="Discrete").eval()
        obs = np.random.default_rng(9801).normal(size=(1, 3, 7)).astype(np.float32)
        legal = np.ones((1, 3, 4), dtype=np.float32)
        legal[0, 0, 2] = 0
        order = np.array([[2, 0, 1]], dtype=np.int64)
        return model, obs, legal, order

    def test_saved_pair_is_consumable_by_read_only_checkpoint_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            model, obs, legal, order = self.setup_snapshot()
            snapshot, checkpoint = save_snapshot_and_checkpoint(
                model, obs, legal, order, directory, "MAT_obs_norm_seed1_trial")
            self.assertTrue(Path(snapshot).exists())
            self.assertTrue(Path(checkpoint).exists())
            with np.load(snapshot, allow_pickle=False) as result:
                self.assertTrue(np.array_equal(result["obs"], obs))
                self.assertTrue(np.array_equal(result["available_actions"], legal))
                self.assertTrue(np.array_equal(result["agent_order"], order))
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(weights[key], tensor.cpu()))
            report = run_diagnostic(
                checkpoint, snapshot, 1, 8, 1, num_histories=4,
                max_context_batch=2, max_counterfactual_batch=4)
            self.assertLess(report["max_sampling_likelihood_error"], 2e-5)
            self.assertEqual(report["bidirectionally_observable_pairs"], 0)
            self.assertGreater(report["observable_directed_pairs"], 0)
            with self.assertRaises(FileExistsError):
                save_snapshot_and_checkpoint(model, obs, legal, order,
                                             directory, "MAT_obs_norm_seed1_trial")

    def test_two_frozen_stages_with_four_contexts_each(self):
        with tempfile.TemporaryDirectory() as directory:
            model, obs, legal, order = self.setup_snapshot()
            first_weight = next(model.parameters()).detach().clone()
            for episode in (1, 10):
                collector = MultiContextCapture(steps=(0, 10, 20, 30))
                with self.assertRaises(ValueError):
                    collector.save(model, directory, "MAT_early", episode)
                for step in (0, 10, 20, 30):
                    current = obs + step * 0.0003 + episode * 0.0005
                    self.assertTrue(collector.observe(step, current, legal, order))
                    if step == 0:
                        # The collector owns copies, not mutable rollout views.
                        current[:, :, :] = -999
                self.assertTrue(collector.complete)
                with self.assertRaises(ValueError):
                    collector.observe(0, obs, legal, order)
                prefix = f"MAT_obs_norm_seed1_ep{episode:04d}"
                snapshot, checkpoint = collector.save(
                    model, directory, prefix, episode)
                with np.load(snapshot, allow_pickle=False) as source:
                    self.assertEqual(source["obs"].shape, (4, 3, 7))
                    self.assertEqual(source["agent_order"].shape, (4, 3))
                    self.assertEqual(source["context_step"].tolist(), [0, 10, 20, 30])
                    self.assertEqual(source["context_episode"].tolist(), [episode]*4)
                    self.assertFalse((source["obs"] == -999).any())
                stats = run_diagnostic(
                    checkpoint, snapshot, 1, 8, 1,
                    num_histories=3, max_context_batch=2,
                    max_counterfactual_batch=4, seed=1)
                self.assertEqual(stats["snapshot_contexts"], 4)
                self.assertEqual(stats["context_step"], [0, 10, 20, 30])
                self.assertEqual(stats["context_episode"], [episode]*4)
                self.assertEqual(len(stats["mean_kl_by_context"]), 4)
                self.assertEqual(len(stats["observable_pairs_by_context"]), 4)
                self.assertLess(stats["max_sampling_likelihood_error"], 2e-5)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(0.001)
            self.assertFalse(torch.equal(first_weight, next(model.parameters())))
            self.assertEqual(len(list(Path(directory).glob("*_snapshot.npz"))), 2)
            self.assertEqual(len(list(Path(directory).glob("*_transformer.pt"))), 2)

    def test_multicontext_guardrails(self):
        for steps in ((), (0, 0), (10, 0), (-1,), tuple(range(17))):
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    MultiContextCapture(steps)
        collector = MultiContextCapture((0, 1))
        model, obs, legal, order = self.setup_snapshot()
        collector.observe(0, obs, legal, order)
        self.assertFalse(collector.complete)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                collector.save(model, directory, "MAT_incomplete", 1)

    def test_fail_closed_if_model_training_or_agent_order_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            model, obs, legal, order = self.setup_snapshot()
            model.train()
            with self.assertRaises(ValueError):
                save_snapshot_and_checkpoint(
                    model, obs, legal, order, directory, "MAT_debug")
            model.eval()
            invalid = np.array([[0, 0, 1]], dtype=np.int64)
            with self.assertRaises(ValueError):
                save_snapshot_and_checkpoint(
                    model, obs, legal, invalid, directory, "MAT_debug")
            with self.assertRaises(ValueError):
                save_snapshot_and_checkpoint(
                    model, obs, legal, order, directory, "../unsafe")
            self.assertFalse(list(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
