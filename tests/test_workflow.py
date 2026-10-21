"""Synthetic artifact/orchestration checks; never train a CNN or access held-out data."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from cnn_experiment.config import Config
from cnn_experiment.final_evaluation import evaluate_final_test
from cnn_experiment.workflow import create_session, package_session, output_inventory, diagnostic_stage
from cnn_experiment.evaluation import comparison_table
from cnn_experiment.training import RunProgress
from cnn_experiment.evaluation import write_epoch_diagnostics
from cnn_experiment.evaluation import evaluate_predictions


class NotebookWorkflowTests(unittest.TestCase):
    def test_diagnostic_failure_is_saved_and_reraised(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'synthetic loader failure'):
                with diagnostic_stage(folder, 'loader'):
                    raise RuntimeError('synthetic loader failure')
            saved = list(Path(folder).glob('loader_failure_*.txt'))
            self.assertEqual(len(saved), 1)
            self.assertIn('synthetic loader failure', saved[0].read_text())

    def test_recipe_is_preserved(self):
        cfg = Config().as_dict()
        expected = {'train_per_class': 4000, 'warmup_epochs': 2, 'head_lr': 1e-4,
                    'backbone_lr': 1e-5, 'weight_decay': 1e-4, 'dropout': .3,
                    'flip_probability': .5, 'finetune_final_stage': True, 'threshold': .5,
                    'topk_fraction': .05, 'patience': 3, 'amp': True}
        self.assertEqual({key: cfg[key] for key in expected}, expected)

    def test_sessions_never_reuse_previous_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = create_session(folder), create_session(folder)
            self.assertNotEqual(first, second)
            (first/'last.pt').write_bytes(b'original')
            self.assertEqual((first/'last.pt').read_bytes(), b'original')

    def test_backup_includes_checkpoints_and_verifiable_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            session = create_session(Path(folder)/'runs')
            (session/'resnet50').mkdir()
            (session/'resnet50/last.pt').write_bytes(b'synthetic checkpoint')
            (session/'history.csv').write_text('epoch,loss\n1,0.4\n')
            before = output_inventory(session)
            backup = package_session(session, Path(folder)/'backups')
            self.assertEqual(before, output_inventory(session))
            archive = Path(backup['archive'])
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), backup['sha256'])
            with zipfile.ZipFile(archive) as z:
                self.assertIn(session.name+'/resnet50/last.pt', z.namelist())
                self.assertIsNone(z.testzip())
            self.assertNotEqual(backup['archive'], package_session(session, Path(folder)/'backups')['archive'])
            with self.assertRaises(ValueError):
                package_session(session, session/'recursive_backup')

    def test_comparison_enriches_metrics_without_changing_winner(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            runs = {}
            for name, auc in [('resnet50', .8), ('efficientnet_b0', .82)]:
                run = root/name; run.mkdir(); runs[name] = run
                (run/'best.pt').write_bytes(b'synthetic frozen checkpoint')
                (run/'training_summary.json').write_text(json.dumps({'status': 'trained', 'best_epoch': 3,
                    'best_balanced_dev_loss': .4}))
                metrics = {'roc_auc': auc, 'pr_auc': .7, 'precision': .8, 'recall': .6, 'f1': .6857}
                (run/'best_dev_metrics.json').write_text(json.dumps({'unit': metrics, 'video': metrics,
                    'losses': {'loss': .3, 'real_class_loss': .2, 'fake_class_loss': .6, 'balanced_loss': .4}}))
            rows = comparison_table(runs, root/'comparison')
            self.assertIn('dev_video_f1', rows[0])
            self.assertEqual(json.loads((root/'comparison/comparison.json').read_text())['selected_cnn'], 'efficientnet_b0')
            with self.assertRaises(FileExistsError):
                comparison_table(runs, root/'comparison')

    def test_current_epoch_plots_exist_even_when_not_best(self):
        with tempfile.TemporaryDirectory() as folder:
            records = [{'label': y, 'source_file': str(i), 'source_family': str(i),
                        'video_label': y, 'metric_eligible': True} for i, y in enumerate([0, 1])]
            predictions = [{'record_index': i, 'probability': p, 'target': y}
                           for i, (y, p) in enumerate([(0, .2), (1, .8)])]
            report = evaluate_predictions(records, predictions, {'aggregation': 'topk', 'topk_fraction': .05,
                'prediction_threshold': .5, 'video_threshold': .5})
            history = [{'epoch': 2, 'is_best': False, 'train': {'numerics': {
                'gradient_norm_max_finite': 2., 'amp_scale_end': 128., 'skipped_steps': 1}}}]
            paths = write_epoch_diagnostics(folder, history, records, predictions, report)
            self.assertTrue((Path(folder)/'epoch_diagnostics/epoch_002/dev_metrics.json').is_file())
            self.assertTrue(paths[1].is_file()); self.assertTrue(paths[2].is_file())

    def test_predeclared_frame_pr_rule_can_choose_a_different_model(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); runs = {}
            for name, video_auc, pr in [('resnet50', .8, .6), ('efficientnet_b0', .9, .4)]:
                run = root/name; run.mkdir(); runs[name] = run
                (run/'best.pt').write_bytes(b'synthetic')
                (run/'training_summary.json').write_text(json.dumps({'status':'trained','best_epoch':1,'best_balanced_dev_loss':.4}))
                unit = {'roc_auc':.8,'pr_auc':pr,'precision':.7,'recall':.5,'f1':.58}
                (run/'best_dev_metrics.json').write_text(json.dumps({'unit':unit,'video':dict(unit,roc_auc=video_auc),
                    'losses':{'loss':.3,'real_class_loss':.2,'fake_class_loss':.6,'balanced_loss':.4}}))
            comparison_table(runs, root/'video')
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                comparison_table(runs, root/'mismatched', selection_metric='frame_pr_auc')
            for run in runs.values():
                summary = json.loads((run/'training_summary.json').read_text())
                summary['checkpoint_selection'] = 'frame_pr_auc'
                (run/'training_summary.json').write_text(json.dumps(summary))
            comparison_table(runs, root/'frame', selection_metric='frame_pr_auc')
            video = json.loads((root/'video/comparison.json').read_text())
            frame = json.loads((root/'frame/comparison.json').read_text())
            self.assertEqual(video['selected_cnn'],'efficientnet_b0')
            self.assertEqual(frame['selected_cnn'],'resnet50')
            self.assertEqual(frame['selection_metric'],'frame_pr_auc')

    def test_compact_stdout_preserves_full_diagnostics_on_disk(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as stdout:
            with RunProgress(folder, 'resnet50', 0) as progress:
                progress.stage('train_fetch_batch', announce=True, candidate_ids=[987654321], sources=['train/secret-decoy.mp4'])
                status = json.loads((Path(folder)/'run_status.json').read_text())
                self.assertEqual(status['candidate_ids'], [987654321])
            self.assertNotIn('987654321', stdout.getvalue())
            self.assertNotIn('secret-decoy', stdout.getvalue())
            self.assertIn('candidate_count', stdout.getvalue())

    def test_final_evaluation_guard_precedes_all_file_access(self):
        with patch('builtins.open', side_effect=AssertionError('Unexpected data access')), \
                patch.object(Path, 'read_text', side_effect=AssertionError('Unexpected metadata access')):
            for approval in (None, '', 'approved upload list'):
                with self.assertRaises(PermissionError):
                    evaluate_final_test('unused', 'unused', 'unused', approval=approval)


if __name__ == '__main__':
    unittest.main()
