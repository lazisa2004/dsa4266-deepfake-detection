"""Synthetic read-only reporting tests: no video, trained weights, training or TEST."""
import base64
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch
import zipfile

from cnn_experiment.evaluation import (binary_metrics, comparison_table,
    metrics_with_f2, write_history_metrics)
from cnn_experiment.reporting import (SavedRun, GRAPH_ORDER, epoch_table, table_html,
    comparison_rows, display_completed_run, export_epoch_report)


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=')


def fixture(folder):
    folder = Path(folder); folder.mkdir(exist_ok=True)
    labels = [0, 0, 1, 1]
    rows = []
    for epoch, scores in enumerate(([.7, .1, .2, .8], [.1, .2, .8, .9], [.6, .6, .2, .8]), 1):
        unit = binary_metrics(labels, scores)
        unit.update(loss=.5/epoch, balanced_loss=.6/epoch,
                    real_class_loss=.4/epoch, fake_class_loss=.8/epoch)
        video = binary_metrics(labels, scores)
        # Exercise legacy F2 reconstruction from each epoch's actual confusion.
        video.pop('f2')
        rows.append({'epoch': epoch, 'train': {'loss': .4/epoch}, 'dev': unit,
                     'dev_video': video, 'learning_rates': {'head': .0001, 'final_stage': .000015},
                     'is_best': epoch < 3})
    write_history_metrics(folder, rows)
    config = {'model': 'resnet50', 'threshold': .5}
    summary = {'best_epoch': 2, 'status': 'completed', 'checkpoint_selection': 'video_roc_auc',
               'best_balanced_dev_loss': .3}
    metrics = {'unit': rows[1]['dev'], 'video': rows[1]['dev_video'],
               'losses': {key: rows[1]['dev'][key] for key in
                          ('loss', 'balanced_loss', 'real_class_loss', 'fake_class_loss')}}
    for filename, value in [('config.json', config), ('training_summary.json', summary), ('best_dev_metrics.json', metrics)]:
        (folder/filename).write_text(json.dumps(value), encoding='utf-8')
    for name in GRAPH_ORDER: (folder/name).write_bytes(PNG)
    (folder/'best.pt').write_bytes(b'not a deserializable checkpoint')
    return rows


class ReportingTests(unittest.TestCase):
    def test_f2_formula_actual_video_predictions_and_zeros(self):
        for scores in ([.7, .1, .2, .8], [.1, .2, .8, .9], [.1, .6, .2, .8]):
            actual = binary_metrics([0, 0, 1, 1], scores)
            p, r = actual['precision'], actual['recall']
            self.assertAlmostEqual(actual['f2'], 5*p*r/(4*p+r))
            legacy = dict(actual); legacy.pop('f2')
            original = copy.deepcopy(legacy)
            self.assertEqual(metrics_with_f2(legacy)['f2'], actual['f2'])
            self.assertEqual(legacy, original)
            self.assertEqual(metrics_with_f2({'precision': p, 'recall': r})['f2'], actual['f2'])
        self.assertEqual(binary_metrics([0, 1], [.1, .2])['f2'], 0.)
        self.assertNotIn('f2', metrics_with_f2({}))

    def test_history_writes_copy_and_per_epoch_video_f2(self):
        with tempfile.TemporaryDirectory() as temp:
            original = fixture(temp)
            self.assertNotIn('f2', original[0]['dev_video'])
            history = [json.loads(line) for line in (Path(temp)/'history.jsonl').read_text().splitlines()]
            self.assertEqual([r['dev_video']['f2'] for r in history], [.5, 1., 5/11])
            csv_rows = list(csv.DictReader(io.StringIO((Path(temp)/'history.csv').read_text())))
            self.assertEqual([float(r['dev_video.f2']) for r in csv_rows], [.5, 1., 5/11])

    def test_complete_table_and_selected_epoch_highlight(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(temp)
            rows, selected, kind = epoch_table(temp)
            self.assertEqual((len(rows), len(rows[0]), selected, kind), (3, 19, 2, 'frame'))
            self.assertEqual([r['Video F2'] for r in rows], [.5, 1., 5/11])
            self.assertEqual([r['Epoch'] for r in rows if r['Best Checkpoint'] == 'Selected best'], [2])
            html = table_html(rows)
            self.assertEqual(html.count('background:#fff0b3'), 1)
            self.assertIn('1.5000e-05', html); self.assertIn('0.1667', html)
            self.assertIn('overflow-x:auto', html)

    def test_csv_clip_metrics_and_unknown_fields_remain_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'old_history.csv'
            path.write_text('epoch,dev.pr_auc,dev_video.precision,dev_video.recall,dev_video.threshold,is_best,learning_rates.temporal_head\n1,0.33,0.5,0.25,0.5,True,0.0001\n2,0.4,,,,False,0.0001\n')
            rows, best, _ = epoch_table(path, kind='clip')
            self.assertEqual(best, 1); self.assertEqual(rows[0]['Clip PR-AUC'], .33)
            self.assertAlmostEqual(rows[0]['Video F2'], 5*.5*.25/(4*.5+.25))
            self.assertIsNone(rows[1]['Video F2']); self.assertIsNone(rows[0]['Backbone LR'])
            self.assertEqual(rows[0]['Classifier/Head LR'], .0001)
            self.assertIn('N/A', table_html(rows)); self.assertNotIn('Frame PR-AUC', rows[0])

    def test_non_half_threshold_metrics_are_not_misreported(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(temp)
            path = Path(temp)/'history.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows: row['dev_video']['threshold'] = .8
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (Path(temp)/'best_dev_metrics.json').unlink()
            (Path(temp)/'best_dev_video_predictions.json').write_text(json.dumps([
                {'target': y, 'probability': p} for y, p in zip([0, 0, 1, 1], [.1, .2, .8, .9])]))
            table, _, _ = epoch_table(temp)
            self.assertIsNone(table[0]['Video F2'])
            self.assertEqual(table[1]['Video F2'], 1.)  # saved best probabilities only
            self.assertIsNone(table[2]['Video F2'])
            self.assertIsNotNone(table[0]['Video ROC-AUC'])

    def test_saved_zip_equivalent_and_never_reads_weights_or_test(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/'run'; fixture(run)
            archive = Path(temp)/'completed.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for file in run.iterdir(): z.write(file, 'model/'+file.name)
                z.writestr('final_test/predictions.csv', 'DO NOT READ')
            original_read = zipfile.ZipFile.read
            reads = []
            def read(z, name, *args, **kwargs):
                reads.append(name)
                return original_read(z, name, *args, **kwargs)
            with patch.object(zipfile.ZipFile, 'read', read), patch('torch.load', side_effect=AssertionError('no weights')):
                self.assertEqual(epoch_table(run), epoch_table(archive))
            self.assertTrue(reads)
            self.assertFalse(any('test' in name or name.endswith('.pt') for name in reads))
            with self.assertRaises(PermissionError): SavedRun(archive).read('best.pt')
            with self.assertRaises(PermissionError): SavedRun(Path(temp)/'final_test')

    def test_display_one_table_then_original_graphs_in_order(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(temp)
            names = []
            original = SavedRun.read
            def read(run, name):
                if name in GRAPH_ORDER: names.append(name)
                return original(run, name)
            # Validate presentation without requiring a local Jupyter installation.
            from unittest.mock import Mock
            fake = ModuleType('IPython.display')
            fake.display = Mock()
            class Artifact:
                def __init__(self, data): self.data = data
            fake.HTML = fake.Image = Artifact
            with patch.object(SavedRun, 'read', read), patch.dict('sys.modules', {'IPython.display': fake}):
                display_completed_run(temp)
            calls = [c.args[0] for c in fake.display.call_args_list]
            self.assertEqual(len(calls), 5)
            self.assertIn('<table', calls[0].data)
            self.assertEqual(names, list(GRAPH_ORDER))
            self.assertTrue(all(c.data == PNG for c in calls[1:]))

    def test_exports_new_files_without_modifying_saved_run(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/'run'; fixture(run)
            before = {p.name: p.read_bytes() for p in run.iterdir()}
            destination = Path(temp)/'report'
            export_epoch_report(run, destination)
            self.assertEqual(before, {p.name: p.read_bytes() for p in run.iterdir()})
            value = json.loads((destination/'epoch_metrics.json').read_text())
            self.assertEqual(value['rows'][2]['Video F2'], 5/11)
            with self.assertRaises(FileExistsError): export_epoch_report(run, destination)

    def test_six_model_comparison_native_level_pending_and_column_order(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(temp)
            temporal = Path(temp)/'temporal'; fixture(temporal)
            (temporal/'config.json').write_text(json.dumps({'model': 'resnet50_lstm', 'threshold': .5}))
            rows = comparison_rows({'resnet50': temp, 'resnet50_lstm': temporal})
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]['Status'], 'Completed')
            self.assertEqual(rows[0]['Best Epoch'], 2)
            self.assertEqual(rows[0]['Video F2'], 1.)
            self.assertEqual(rows[1]['Level'], 'Clip')
            self.assertTrue(all(r['Status'] == 'Pending' for r in rows[2:]))
            fields = list(rows[0]); start = fields.index('Video F1')
            self.assertEqual(fields[start:start+3], ['Video F1', 'Video F2', 'DEV Loss'])

    def test_model_comparison_rejects_wrong_saved_architecture(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(temp)
            with self.assertRaisesRegex(ValueError, 'cannot be reported as'):
                comparison_rows({'resnet50_lstm': temp})

    def test_existing_selection_rule_unchanged_and_legacy_best_f2_derived(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/'run'; fixture(run)
            output = Path(temp)/'comparison'
            rows = comparison_table({'resnet50': run}, output)
            self.assertEqual(rows[0]['dev_video_f2'], 1.)
            comparison = json.loads((output/'comparison.json').read_text())
            self.assertEqual(comparison['selection_metric'], 'video_roc_auc')
            self.assertEqual(comparison['selected_cnn'], 'resnet50')
            fields = list(rows[0]); start = fields.index('dev_video_f1')
            self.assertEqual(fields[start:start+3], ['dev_video_f1', 'dev_video_f2', 'dev_loss'])

    def test_notebook_training_reports_once_after_training_and_all_cells_compile(self):
        root = Path(__file__).resolve().parents[1]
        for relative, indices in [('experiment1_cnn_v2_vscode_kaggle.ipynb', (15, 17, 31)),
                                  ('notebooks/experiment1_face_cnn.ipynb', (12, 14, 16))]:
            path = root/relative
            if not path.exists(): continue  # main notebook deliberately local-only
            notebook = json.loads(path.read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code': compile(''.join(cell['source']), relative, 'exec')
            for index in indices:
                source = ''.join(notebook['cells'][index]['source'])
                self.assertEqual(source.count('display_completed_run('), 1)
                call = min(source.find(name) for name in ('train(', 'train_experiment(', 'train_temporal(') if name in source)
                self.assertLess(call, source.find('display_completed_run('))
            self.assertNotIn('display_completed_run', (root/'cnn_experiment/training.py').read_text())


if __name__ == '__main__': unittest.main()
