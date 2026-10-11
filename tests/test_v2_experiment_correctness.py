"""CPU-only correctness: paired MAT initialization, strict frozen manifest.

Do not load StarCraftII, use CUDA, or launch training here.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.config import get_config
from research.dependency_decoder.checkpoint_manifest import (
    build_training_manifest, load_training_manifest, evaluation_model_flags,
)
from research.dependency_decoder.evaluate_frozen_smac import evaluate_frozen_checkpoint


def new_model(mode, attention=True):
    return MultiAgentTransformer(
        state_dim=9, obs_dim=7, action_dim=4, n_agent=4,
        n_block=2, n_embd=16, n_head=1,
        agent_order_mode=mode, order_hidden_dim=16, order_pair_rank=8,
        order_attention_bias=attention)


class TestPairedMATInitialization(unittest.TestCase):
    def test_same_seed_same_encoder_decoder_and_global_rng(self):
        reference = None
        for mode in ("identity", "obs_norm", "learned", "dependency"):
            for attention in ((True, False) if mode == "dependency" else (True,)):
                with self.subTest(mode=mode, bias=attention):
                    torch.manual_seed(114)
                    net = new_model(mode, attention)
                    state = {key: val.detach().clone()
                             for key, val in net.state_dict().items()
                             if key.startswith(("encoder.", "decoder."))}
                    next_random = torch.rand(5)
                    if reference is None:
                        reference = (state, next_random)
                    else:
                        self.assertEqual(set(state), set(reference[0]))
                        for key, value in state.items():
                            self.assertTrue(torch.equal(value, reference[0][key]), key)
                        self.assertTrue(torch.equal(next_random, reference[1]))

    def test_attention_ablation_preserves_same_orders_and_backbone(self):
        torch.manual_seed(221)
        with_attention = new_model("dependency", attention=True)
        torch.manual_seed(221)
        without_attention = new_model("dependency", attention=False)
        for name, weight in with_attention.state_dict().items():
            self.assertTrue(torch.equal(weight, without_attention.state_dict()[name]), name)
        self.assertTrue(with_attention.order_attention_bias)
        self.assertFalse(without_attention.order_attention_bias)
        obs = torch.randn(2, 4, 7)
        o1, lp1 = with_attention.order_policy.sample(obs, deterministic=True)
        o2, lp2 = without_attention.order_policy.sample(obs, deterministic=True)
        self.assertTrue(torch.equal(o1, o2))
        self.assertTrue(torch.equal(lp1, lp2))

    def test_frozen_strict_load_rejects_incompatible_architecture(self):
        torch.manual_seed(442)
        identity = new_model("identity")
        dependency = new_model("dependency")
        with self.assertRaises(RuntimeError):
            dependency.load_state_dict(identity.state_dict(), strict=True)


class TestFrozenRunManifest(unittest.TestCase):
    def arguments(self, mode="dependency", attention=True):
        config = get_config().parse_args([])
        config.env_name = "StarCraft2"
        config.algorithm_name = "mat"
        config.map_name = "3s5z"
        config.agent_order_mode = mode
        config.store_agent_orders = mode in ("learned", "dependency")
        config.disable_order_attention_bias = not attention
        config.n_block = 2
        config.n_embd = 64
        config.n_head = 1
        config.seed = 1
        config.num_env_steps = 1000000
        config.episode_length = 100
        config.save_interval = 500
        return config

    def test_manifest_roundtrip_and_exact_evaluator_flags(self):
        for mode in ("identity", "learned", "dependency"):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                args = self.arguments(mode, attention=False)
                data = build_training_manifest(args, "e" * 40)
                file = Path(tmp) / "mat_decoder_manifest.json"
                file.write_text(json.dumps(data), encoding="utf-8")
                loaded = load_training_manifest(file, "3s5z", mode)
                self.assertEqual(loaded["training_source_sha"], "e" * 40)
                self.assertEqual(loaded["config"]["save_interval"], 500)
                self.assertFalse(loaded["resumable_training_state"])
                flags = evaluation_model_flags(loaded)
                self.assertIn("--disable_order_attention_bias", flags)
                self.assertEqual(flags[flags.index("--n_block") + 1], "2")
                self.assertEqual("--store_agent_orders" in flags,
                                 mode in ("learned", "dependency"))

    def test_reject_map_mode_and_dirty_source_mismatch(self):
        with TemporaryDirectory() as tmp:
            data = build_training_manifest(self.arguments(), "c" * 40)
            file = Path(tmp) / "manifest.json"
            file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_training_manifest(file, "6h_vs_8z", "dependency")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_training_manifest(file, "3s5z", "identity")
            data["source_dirty"] = True
            file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                load_training_manifest(file, "3s5z", "dependency")

    def test_reject_invalid_or_missing_manifest_before_loading_smac(self):
        # Preflight errors must occur without importing StarCraftII or torch
        # checkpoint bytes (the dummy weights are intentionally invalid).
        with TemporaryDirectory() as tmp:
            weights = Path(tmp) / "dummy.pt"
            weights.write_bytes(b"dummy")
            with self.assertRaisesRegex(ValueError, "requires --manifest"):
                evaluate_frozen_checkpoint(
                    weights, "3s5z", agent_order_mode="dependency", gpu=False)
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "schema"):
                evaluate_frozen_checkpoint(
                    weights, "3s5z", agent_order_mode="dependency", gpu=False,
                    manifest_path=manifest)


if __name__ == "__main__":
    unittest.main()
