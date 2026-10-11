"""V3 prefix dependency fail-closed launcher and CLI smoke tests (CPU only)."""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "mat/scripts/train_smac_research.sh"


def dry(map_name="3s5z", **overrides):
    env = dict(os.environ)
    env.update(
        DRY_RUN="1", AGENT_ORDER_MODE="dependency_prefix",
        STORE_AGENT_ORDERS="1", N_ROLLOUT_THREADS="2",
        N_TRAINING_THREADS="2", NUM_ENV_STEPS="4000",
        PPO_EPOCH="10", CLIP_PARAM="0.05", USE_EVAL="0",
        ORDER_PREFIX_SCALE="1.0", DISABLE_ORDER_ATTENTION_BIAS="0")
    env.update({k: str(v) for k, v in overrides.items()})
    return subprocess.run(
        ["bash", str(SCRIPT), map_name, "0", "1"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)


class TestPrefixDependencyLauncher(unittest.TestCase):
    def test_prototype_flags_only_dry_run(self):
        result = dry()
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--agent_order_mode dependency_prefix",
                     "--store_agent_orders", "--order_pair_rank 16",
                     "--order_graph_scale 0.5",
                     "--order_prefix_scale 1.0",
                     "--num_env_steps 4000",
                     "DRY_RUN=1: command printed; no training started."):
            self.assertIn(flag, result.stdout)

    def test_nobias_ablation_flag(self):
        result = dry(DISABLE_ORDER_ATTENTION_BIAS="1", ORDER_PREFIX_SCALE="0.8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--disable_order_attention_bias", result.stdout)
        self.assertIn("--order_prefix_scale 0.8", result.stdout)

    def test_v31_independent_switches_and_invalid_combinations(self):
        aligned_no_bias = dry(DISABLE_ORDER_ATTENTION_BIAS="1",
                              ORDER_INFLUENCE_ALIGN_COEF="0.2")
        self.assertEqual(aligned_no_bias.returncode, 0, aligned_no_bias.stderr)
        self.assertIn("--order_influence_align_coef 0.2", aligned_no_bias.stdout)
        self.assertIn("--disable_order_attention_bias", aligned_no_bias.stdout)
        self.assertNotIn("--adaptive_order_attention_bias", aligned_no_bias.stdout)
        full = dry(ADAPTIVE_ORDER_ATTENTION_BIAS="1",
                   ORDER_INFLUENCE_ALIGN_COEF="0.2",
                   ORDER_ATTENTION_SCALE_MAX="1.5",
                   ORDER_INFLUENCE_EFFECT_SCALE="0.00002",
                   ORDER_INFLUENCE_MIN_EFFECT="0.0000002",
                   ORDER_INFLUENCE_MAX_ALTERNATIVES="3")
        self.assertEqual(full.returncode, 0, full.stderr)
        for flag in ("--adaptive_order_attention_bias", "--order_attention_scale_max 1.5",
                     "--order_influence_align_coef 0.2",
                     "--order_influence_effect_scale 0.00002",
                     "--order_influence_min_effect 0.0000002",
                     "--order_influence_max_alternatives 3"):
            self.assertIn(flag, full.stdout)
        invalid = dry(ADAPTIVE_ORDER_ATTENTION_BIAS="1",
                      DISABLE_ORDER_ATTENTION_BIAS="1")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("adaptive bias requires", invalid.stderr)
        invalid = dry(ORDER_INFLUENCE_MAX_ALTERNATIVES="0")
        self.assertNotEqual(invalid.returncode, 0)

    def test_reject_unapproved_maps_and_missing_saved_orders(self):
        for name in ("3m", "corridor", "bogus"):
            with self.subTest(map=name):
                result = dry(name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("outside the nine approved maps", result.stderr)
        result = dry(STORE_AGENT_ORDERS="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires STORE_AGENT_ORDERS=1", result.stderr)

    def test_teacher_cache_is_opt_in_and_requires_alignment(self):
        ordinary = dry(ORDER_INFLUENCE_ALIGN_COEF="0.2")
        self.assertEqual(ordinary.returncode, 0, ordinary.stderr)
        self.assertNotIn("--cache_order_influence_teacher", ordinary.stdout)
        cached = dry(ORDER_INFLUENCE_ALIGN_COEF="0.2",
                     CACHE_ORDER_INFLUENCE_TEACHER="1")
        self.assertEqual(cached.returncode, 0, cached.stderr)
        self.assertIn("--cache_order_influence_teacher", cached.stdout)
        invalid = dry(CACHE_ORDER_INFLUENCE_TEACHER="1")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("teacher cache requires", invalid.stderr)

    def test_approved_superhard_and_external_root(self):
        result = dry("6h_vs_8z",
                     MAT_DECODER_RESULTS_ROOT="/data/sandboxed-v3-results",
                     MAT_DECODER_LOG_DIR="/data/sandboxed-v3-logs")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--run_dir /data/sandboxed-v3-results", result.stdout)
        self.assertIn("results_root=/data/sandboxed-v3-results", result.stdout)


if __name__ == "__main__":
    unittest.main()
