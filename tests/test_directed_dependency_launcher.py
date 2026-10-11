"""Dependency-order launcher flag and map safety (CPU subprocess, dry run)."""
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "mat/scripts/train_smac_research.sh"


def run_dry(map_name, **updates):
    env = os.environ.copy()
    env.update(DRY_RUN="1", AGENT_ORDER_MODE="dependency",
               STORE_AGENT_ORDERS="1", NUM_ENV_STEPS="1000000",
               PPO_EPOCH="10", CLIP_PARAM="0.05", N_ROLLOUT_THREADS="1",
               N_TRAINING_THREADS="2", USE_EVAL="0")
    env.update({k: str(v) for k,v in updates.items()})
    return subprocess.run(["bash", str(SCRIPT), map_name, "0", "1"], cwd=ROOT,
                          env=env, text=True, capture_output=True, timeout=20)


class TestDirectedDependencyLauncher(unittest.TestCase):
    def test_hard_map_dry_only(self):
        proc = run_dry("3s5z")
        self.assertEqual(proc.returncode,0,proc.stderr)
        for arg in ("--agent_order_mode dependency","--store_agent_orders",
                    "--order_pair_rank 16","--order_graph_scale 0.5",
                    "--order_hidden_dim 64","--order_loss_coef 0.1"):
            self.assertIn(arg,proc.stdout)
        self.assertIn("DRY_RUN=1: command printed; no training started.",proc.stdout)

    def test_superhard_map_is_whitelisted(self):
        proc=run_dry("6h_vs_8z",PPO_EPOCH="15")
        self.assertEqual(proc.returncode,0,proc.stderr)

    def test_old_and_unapproved_maps_rejected(self):
        for name in ("3m","corridor","fake_map"):
            with self.subTest(name=name):
                proc=run_dry(name)
                self.assertNotEqual(proc.returncode,0)
                self.assertIn("outside the nine approved maps",proc.stderr)

    def test_results_root_and_log_dir_can_be_external(self):
        proc = run_dry("3s5z",
                       MAT_DECODER_RESULTS_ROOT="/tmp/mat-decoder-persistent-runs",
                       MAT_DECODER_LOG_DIR="/tmp/mat-decoder-persistent-logs")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--run_dir /tmp/mat-decoder-persistent-runs", proc.stdout)
        self.assertIn("results_root=/tmp/mat-decoder-persistent-runs", proc.stdout)
        self.assertIn("log_file=/tmp/mat-decoder-persistent-logs/", proc.stdout)

    def test_missing_saved_order_refused(self):
        proc=run_dry("3s5z",STORE_AGENT_ORDERS="0")
        self.assertNotEqual(proc.returncode,0)
        self.assertIn("requires STORE_AGENT_ORDERS=1",proc.stderr)


if __name__=="__main__":
    unittest.main()
