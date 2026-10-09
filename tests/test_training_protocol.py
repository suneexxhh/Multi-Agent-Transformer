"""SMAC launch-protocol regression: source code budget != pilot budget.

Dry-run only: absolutely no StarCraft II, GPU, Conda, or training.
The historical MAT-MSA map table in train_smac_research.sh is a
PROVISIONAL reference, not a verified copy of every original user script.
"""
import os
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "mat" / "scripts" / "train_smac_research.sh"


def dry_run(map_name="3m", **overrides):
    env = os.environ.copy()
    for key in (
        "NUM_ENV_STEPS", "PPO_EPOCH", "CLIP_PARAM", "N_ROLLOUT_THREADS",
        "N_TRAINING_THREADS", "USE_EVAL", "AGENT_ORDER_MODE",
    ):
        env.pop(key, None)
    env.update(DRY_RUN="1", **{k: str(v) for k, v in overrides.items()})
    return subprocess.run(
        ["bash", str(LAUNCHER), map_name, "0", "1"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=15)


class TestHistoricalMATMSAReferenceLauncher(unittest.TestCase):
    def test_all_explicit_map_defaults_do_not_silently_become_pilots(self):
        # Reference table comes from the provisional MAT-MSA branch
        # launcher; its origin must still be verified against MSA+MAT
        # original training scripts for publication-quality experiments.
        reference = {
            "3m": (5_000_000, 15, "0.2"),
            "3s5z": (5_000_000, 10, "0.05"),
            "5m_vs_6m": (5_000_000, 10, "0.05"),
            "10m_vs_11m": (5_000_000, 10, "0.05"),
            "6h_vs_8z": (10_000_000, 15, "0.05"),
            "MMM2": (10_000_000, 5, "0.05"),
            "3s5z_vs_3s6z": (20_000_000, 5, "0.05"),
            "27m_vs_30m": (10_000_000, 5, "0.2"),
        }
        for name, (steps, ppo, clip) in reference.items():
            with self.subTest(map=name):
                proc = dry_run(name)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(
                    "training_protocol_class=historical_reference_candidate "
                    "historical_reference_candidate_is_unverified=1",
                    proc.stdout)
                self.assertIn(
                    f"reference_map_steps={steps} "
                    f"reference_ppo_epoch={ppo} reference_clip={clip}",
                    proc.stdout)
                self.assertIn(
                    f"steps={steps} ppo_epoch={ppo} clip={clip}",
                    proc.stdout)
                effective = steps // 3200 * 3200
                self.assertIn(
                    f"requested_env_steps={steps} effective_env_steps={effective} "
                    "rollout_batch_steps=3200",
                    proc.stdout)
                self.assertIn(f"--num_env_steps {steps}", proc.stdout)
                self.assertIn(f"--ppo_epoch {ppo}", proc.stdout)
                self.assertIn(f"--clip_param {clip}", proc.stdout)
                if effective != steps:
                    self.assertIn("WARNING: requested env steps rounded down", proc.stdout)

    def test_explicit_100k_ppo5_one_rollout_is_always_pilot(self):
        proc = dry_run("3m", NUM_ENV_STEPS=100000, PPO_EPOCH=5,
                       N_ROLLOUT_THREADS=1, N_TRAINING_THREADS=2,
                       USE_EVAL=0, AGENT_ORDER_MODE="obs_norm")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("training_protocol_class=pilot_or_nonreference", proc.stdout)
        self.assertIn("requested_env_steps=100000 effective_env_steps=100000",
                      proc.stdout)
        self.assertIn("reference_map_steps=5000000 reference_ppo_epoch=15",
                      proc.stdout)
        self.assertIn("--n_rollout_threads 1", proc.stdout)

    def test_rollout_batch_rounding_is_exposed_for_nondivisible_budgets(self):
        proc = dry_run("3m", NUM_ENV_STEPS=500000, N_ROLLOUT_THREADS=32)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("requested_env_steps=500000 effective_env_steps=499200",
                      proc.stdout)
        self.assertIn("training_protocol_class=pilot_or_nonreference",
                      proc.stdout)

    def test_invalid_training_budgets_fail_before_smac_launch(self):
        cases = [
            {"NUM_ENV_STEPS": "0"},
            {"NUM_ENV_STEPS": "-100"},
            {"NUM_ENV_STEPS": "2.5e6"},
            {"NUM_ENV_STEPS": "99"},
            {"PPO_EPOCH": "0"},
            {"PPO_EPOCH": "-1"},
            {"PPO_EPOCH": "five"},
            {"CLIP_PARAM": "0"},
            {"CLIP_PARAM": "0.0"},
            {"CLIP_PARAM": "-0.05"},
            {"CLIP_PARAM": "1.1"},
            {"CLIP_PARAM": "not_a_number"},
        ]
        for overrides in cases:
            with self.subTest(kwargs=overrides):
                proc = dry_run("3m", **overrides)
                self.assertNotEqual(proc.returncode, 0)
                self.assertNotIn("Launching SMAC training", proc.stdout)


if __name__ == "__main__":
    unittest.main()
