"""A single real SMAC pilot must not produce supervised precedence labels."""
import json
import tempfile
import unittest
from pathlib import Path

from research.dependency_decoder.stage_quality_review import (
    review_stage_summary, main,
)


class TestStageQualityReview(unittest.TestCase):
    @staticmethod
    def example():
        return {
            "map":"3m", "seed":1, "no_eval_win_rate":True,
            "num_env_steps":100000,
            "stages":[
                {"episode_index":0,"train_steps_at_capture":0,
                 "observable_pairs":9,"total_directed_pairs":24,
                 "context_pair_counts":[3,3,0,3],
                 "mean_KL":2e-13,"mean_TV":2e-7},
                {"episode_index":900,"train_steps_at_capture":90000,
                 "observable_pairs":7,"total_directed_pairs":24,
                 "context_pair_counts":[0,3,1,3],
                 "mean_KL":1e-5,"mean_TV":0.002},
            ],
        }

    def test_do_not_confuse_tiny_kl_or_stronger_signal_with_labels(self):
        out = review_stage_summary(self.example())
        self.assertFalse(out["allowed_for_precedence_label_training"])
        self.assertTrue(out["stage_differences_confounded_by_observation_change"])
        self.assertAlmostEqual(out["raw_last_minus_first_mean_TV"], 0.002-2e-7)
        self.assertAlmostEqual(out["stages"][0]["observable_fraction"], 9/24)
        self.assertAlmostEqual(out["stages"][1]["observable_fraction"], 7/24)
        self.assertEqual(out["stages"][0]["completely_unidentifiable_contexts"], 1)
        self.assertIsNone(out["stages"][0]["tv_over_external_noise_floor"])
        self.assertIn("Only one documented experiment seed", out["reasons"])

    def test_external_noise_floor_does_not_make_false_causal_claim(self):
        out = review_stage_summary(self.example(), noise_floor_tv=1e-6)
        self.assertAlmostEqual(out["stages"][1]["tv_over_external_noise_floor"], 2000)
        self.assertFalse(out["allowed_for_precedence_label_training"])
        self.assertTrue(any("not a causal" in v for v in out["reasons"]))

    def test_null_coverage_is_missing_not_zero_kl(self):
        src = self.example()
        src["stages"][0]["observable_pairs"] = 0
        src["stages"][0]["context_pair_counts"] = [0,0,0,0]
        src["stages"][0]["mean_KL"] = None
        src["stages"][0]["mean_TV"] = None
        out = review_stage_summary(src)
        self.assertIsNone(out["raw_last_minus_first_mean_TV"])
        self.assertEqual(out["stages"][0]["observable_fraction"], 0)

    def test_invalid_data_are_rejected_and_cli_is_json(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "pilot.json"
            output = Path(d) / "review.json"
            src.write_text(json.dumps(self.example()), encoding="utf-8")
            review = main([str(src), "--output", str(output)])
            self.assertEqual(review, json.loads(output.read_text()))
            self.assertFalse(review["allowed_for_precedence_label_training"])
        for mutated in ("negative", "fake_measured_zero", "mismatched_count",
                        "duplicate_episode", "invalid_noise"):
            sample = self.example()
            noise = None
            if mutated == "negative":
                sample["stages"][0]["mean_TV"] = -0.1
            elif mutated == "fake_measured_zero":
                sample["stages"][0].update(
                    observable_pairs=0, context_pair_counts=[0,0,0,0])
            elif mutated == "mismatched_count":
                sample["stages"][0]["context_pair_counts"] = [3,0,0,3]
            elif mutated == "duplicate_episode":
                sample["stages"][1]["episode_index"] = 0
            else:
                noise = 0
            with self.subTest(case=mutated):
                with self.assertRaises(ValueError):
                    review_stage_summary(sample, noise)


if __name__ == "__main__":
    unittest.main()
