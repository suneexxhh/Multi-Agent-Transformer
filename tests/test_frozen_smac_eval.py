"""Evaluation provenance and log parser tests; never starts StarCraft II."""
import unittest

from research.dependency_decoder.evaluate_frozen_smac import (
    parse_native_eval_win_rate,
    evaluate_frozen_checkpoint,
)


class TestFrozenSmacEval(unittest.TestCase):
    def test_parse_true_native_integer_win_counts(self):
        for wins in (0, 1, 9, 16, 32):
            with self.subTest(wins=wins):
                log = "Starting evaluation\neval win rate is {}.\n".format(
                    wins / 32)
                result, rate = parse_native_eval_win_rate(log, 32)
                self.assertEqual(result, wins)
                self.assertAlmostEqual(rate, wins / 32)

    def test_reject_ambiguous_or_invalid_metrics(self):
        for output, count in (
            ("No win rate", 32),
            ("eval win rate is 0.2.\n", 32),
            ("eval win rate is 1.3.\n", 32),
            ("eval win rate is 0.5.\neval win rate is 0.5.", 32),
            ("eval win rate is 0.5.", 0),
        ):
            with self.subTest(log=output, episodes=count):
                with self.assertRaises(ValueError):
                    parse_native_eval_win_rate(output, count)

    def test_missing_checkpoint_fails_without_importing_smac(self):
        # The helper validates inputs before opening an environment.
        # Avoid actually starting StarCraft II in GitHub CPU CI.
        with self.assertRaises(FileNotFoundError):
            evaluate_frozen_checkpoint("/file/that/does/not/exist.pt")
        with self.assertRaises(ValueError):
            evaluate_frozen_checkpoint("/file/that/does/not/exist.pt",
                                       agent_order_mode="unknown")
        with self.assertRaises(FileNotFoundError):
            evaluate_frozen_checkpoint("/file/that/does/not/exist.pt",
                                       agent_order_mode="identity")


if __name__ == "__main__":
    unittest.main()
