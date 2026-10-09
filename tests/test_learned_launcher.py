"""Fail-closed CPU launcher regression tests; never calls GPU/SMAC."""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "mat" / "scripts" / "train_smac_research.sh"


def run_launcher(map_name, **overrides):
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "AGENT_ORDER_MODE": "learned",
                "STORE_AGENT_ORDERS": "1", "NUM_ENV_STEPS": "100000",
                "N_ROLLOUT_THREADS": "1", "USE_EVAL": "0",
                "PPO_EPOCH": "10", "CLIP_PARAM": "0.05"})
    env.update({k: str(v) for k, v in overrides.items()})
    return subprocess.run(["bash", str(SCRIPT), map_name, "0", "1"],
                          cwd=ROOT, env=env, text=True, capture_output=True,
                          timeout=15)


class TestLearnedLauncher(unittest.TestCase):
    def test_approved_map_dry_run_only(self):
        p = run_launcher("3s5z")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--agent_order_mode learned", p.stdout)
        self.assertIn("--store_agent_orders", p.stdout)
        self.assertIn("--order_hidden_dim 64", p.stdout)
        self.assertIn("--order_temperature 1.0", p.stdout)
        self.assertIn("--order_loss_coef 0.1", p.stdout)
        self.assertIn("DRY_RUN=1: command printed; no training started.", p.stdout)

    def test_unapproved_new_training_maps_fail_closed(self):
        for name in ["3m", "corridor", "MMM", "5m_vs_6m_fake"]:
            with self.subTest(name=name):
                p = run_launcher(name)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("outside the nine approved maps", p.stderr)

    def test_missing_saved_order_rejected(self):
        p = run_launcher("3s5z", STORE_AGENT_ORDERS="0")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("requires STORE_AGENT_ORDERS=1", p.stderr)


if __name__ == "__main__":
    unittest.main()
