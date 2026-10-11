"""CPU-only experiment manager checks; launcher calls are DRY_RUN=1."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research.dependency_decoder import prepare_three_seed_experiments as plan
from research.dependency_decoder.checkpoint_manifest import (
    evaluation_model_flags, load_training_manifest,
)


class TestThreeSeedExperiments(unittest.TestCase):
    def test_all_five_modes_have_paired_dry_run_and_strict_eval_config(self):
        with TemporaryDirectory() as tmp:
            for mode in plan.MODES:
                with self.subTest(mode=mode):
                    run = Path(tmp) / mode
                    output, args = plan.dry_run(run, mode, 2)
                    self.assertIn('DRY_RUN=1: command printed; no training started.', output)
                    manifest = plan.expected_manifest(args, 'a' * 40)
                    path = Path(tmp) / f'{mode}.json'
                    path.write_text(json.dumps(manifest))
                    loaded = load_training_manifest(path, '3s5z', plan.MODES[mode][0])
                    flags = evaluation_model_flags(loaded)
                    self.assertEqual('--disable_order_attention_bias' in flags,
                                     mode == 'dependency-NoBias')
                    self.assertEqual(loaded['config']['num_env_steps'], 2_000_000)
                    self.assertEqual(loaded['config']['save_interval'], 100)
                    self.assertEqual(loaded['config']['eval_episodes'], 32)

    def test_six_primary_nine_ablation_and_fail_closed_existing_suite(self):
        with TemporaryDirectory() as tmp:
            suite = Path(tmp) / 'suite'
            original = plan.git_output
            def clean_git(*args):
                if args[0] == 'status':
                    return ''
                return original(*args)
            with patch.object(plan, 'git_output', side_effect=clean_git):
                runs = plan.prepare(suite)
            self.assertEqual(len(runs), 15)
            self.assertEqual(sum(x['role'] == 'primary' for x in runs), 6)
            self.assertEqual(sum(x['role'] == 'ablation' for x in runs), 9)
            self.assertEqual({x['training_source_sha'] for x in runs},
                             {plan.git_output('rev-parse', 'HEAD')})
            for item in runs:
                paths = item['paths']
                self.assertTrue(Path(paths['run_directory'], 'logs/dry_run.txt').is_file())
                self.assertFalse(Path(paths['native_run']).exists())
                self.assertEqual(item['effective_env_steps'], 2_000_000)
                self.assertEqual(item['frozen_evaluation_command'][-2], '--output')
                self.assertEqual(item['frozen_evaluation_command'][-1],
                                 paths['evaluation_result'])
            self.assertEqual(len(plan.index(suite)), 15)
            with self.assertRaises(FileExistsError):
                plan.prepare(suite)


if __name__ == '__main__':
    unittest.main()
