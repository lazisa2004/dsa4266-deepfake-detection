"""Bounded CPU mathematical/architecture tests; synthetic data, never held-out files."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cnn_experiment.gradcam import (GradCAM, ExplanationData, guard_split,
    load_explanation_data, load_model, select_examples, target_layer,
    top5_contributors, save_explanations)
from cnn_experiment.config import Config
from cnn_experiment.models import StagedCNN
from cnn_experiment.final_evaluation import FINAL_TEST_APPROVAL


class ToyCNN(nn.Module):
    def __init__(self, sign=1):
        super().__init__()
        self.features = nn.Conv2d(3, 1, 1, bias=False)
        with torch.no_grad():
            self.features.weight.fill_(1)
        self.sign = sign

    def forward(self, x):
        return self.sign * self.features(x).mean((2, 3))


def fixture():
    records, predictions = [], []
    for i, (target, score) in enumerate([(1, .9), (0, .1), (0, .95), (1, .05), (1, .6), (0, .4)]):
        records.append({'id': i, 'source_file': 'dev/example.mp4', 'source_family': 'dev/original.mp4',
                        'timestamps_exact': [str(i)], 'label': target, 'video_label': 1})
        predictions.append({'record_index': i, 'target': target, 'probability': score})
    return ExplanationData('dev', records, predictions, .5, .5, 'checkpoint', 'predictions')


class GradCAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_formula_and_no_parameter_mutation(self):
        model = ToyCNN().train()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = torch.ones_like(parameter)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        gradients = [p.grad.clone() for p in model.parameters()]
        image = torch.arange(48).view(1, 3, 4, 4).float()/48
        expected = image.sum(1)[0].numpy()
        expected /= expected.max()
        with torch.no_grad():
            result = GradCAM(model, model.features)(image)
        np.testing.assert_allclose(result['heatmaps'][0], expected, atol=1e-6)
        self.assertTrue(model.training)
        self.assertEqual(len(model.features._forward_hooks), 0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        for parameter, gradient in zip(model.parameters(), gradients):
            self.assertTrue(torch.equal(parameter.grad, gradient))
            self.assertFalse(parameter.requires_grad)

    def test_negative_fake_evidence_gives_zero_not_nan(self):
        model = ToyCNN(-1)
        result = GradCAM(model, model.features)(torch.ones(1, 3, 4, 4))
        self.assertTrue(result['zero_maps'][0])
        self.assertTrue(np.isfinite(result['heatmaps']).all())
        self.assertEqual(float(result['heatmaps'].max()), 0)

    def test_hooks_removed_on_failure_and_nonfinite(self):
        model = ToyCNN()
        with self.assertRaises(FloatingPointError):
            GradCAM(model, model.features)(torch.full((1, 3, 4, 4), float('nan')))
        self.assertEqual(len(model.features._forward_hooks), 0)
        with patch.object(model, 'forward', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                GradCAM(model, model.features)(torch.ones(1, 3, 4, 4))
        self.assertEqual(len(model.features._forward_hooks), 0)

    def test_both_real_architectures_frozen_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            from cnn_experiment.data import PROTOCOL
            for name in ('resnet50', 'efficientnet_b0'):
                config = Config(model=name)
                model = StagedCNN(config, '.', pretrained=False).eval()
                path = Path(folder)/'fixture.pt'
                torch.save({'protocol': PROTOCOL, 'config': config.as_dict(), 'model': model.state_dict()}, path)
                loaded, actual, _ = load_model(path)
                self.assertEqual(actual.model, name)
                self.assertEqual(list(loaded.state_dict()), list(model.state_dict()))
                for key, value in loaded.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]))
                del model
                for parameter in loaded.parameters():
                    parameter.requires_grad_(False)
                layer, _ = target_layer(loaded, name)
                image = torch.rand(1, 3, 224, 224)
                with torch.no_grad():
                    expected = loaded(image).sigmoid().flatten().numpy()
                result = GradCAM(loaded, layer)(image)
                self.assertEqual(result['heatmaps'].shape, (1, 224, 224))
                self.assertEqual(result['activation_shape'][-2:], [7, 7])
                np.testing.assert_allclose(result['probabilities'], expected, atol=1e-6)
                self.assertTrue(np.isfinite(result['heatmaps']).all())
                self.assertEqual(len(layer._forward_hooks), 0)

    def test_seeded_categories_and_threshold(self):
        data = fixture()
        for cell, expected in [('tp', {0, 4}), ('tn', {1, 5}), ('fp', {2}), ('fn', {3})]:
            self.assertEqual(set(select_examples(data, cell, 10)), expected)
            self.assertEqual(select_examples(data, cell, 1), select_examples(data, cell, 1))
        self.assertEqual(select_examples(data, 'high_confidence_correct', 1), [0])
        self.assertEqual(set(select_examples(data, 'high_confidence_incorrect', 2)), {2, 3})
        data.frame_threshold = .92
        self.assertEqual(set(select_examples(data, 'fn', 10)), {0, 3, 4})

    def test_top5_ceil_mean_not_max_and_tie_order(self):
        data = fixture()
        data.records = [dict(data.records[0], id=i, timestamps_exact=[str(i)]) for i in range(41)]
        data.predictions = [{'record_index': i, 'probability': i/100} for i in range(41)]
        report = top5_contributors(data, 'dev/example.mp4')
        self.assertEqual(report['k'], 3)
        self.assertAlmostEqual(report['probability'], .39)
        self.assertNotEqual(report['probability'], .40)
        self.assertEqual([p['record_index'] for p in report['contributors']], [40, 39, 38])
        data.predictions[-2]['probability'] = .4
        report = top5_contributors(data, 'dev/example.mp4')
        self.assertEqual([p['record_index'] for p in report['contributors'][:2]], [39, 40])

    def test_test_gate_precedes_all_io(self):
        with patch('cnn_experiment.gradcam.digest', side_effect=AssertionError('I/O before gate')), \
             patch('cnn_experiment.gradcam.torch.load', side_effect=AssertionError('I/O before gate')):
            with self.assertRaises(PermissionError):
                load_explanation_data('missing', 'missing', split='test')
            with self.assertRaises(PermissionError):
                load_model('missing', split='test')
            data = fixture(); data.split = 'test'
            for function in (lambda: select_examples(data), lambda: top5_contributors(data, 'test/a.mp4'),
                             lambda: save_explanations(data, 'missing', 'missing', 'missing', [0])):
                with self.assertRaises(PermissionError):
                    function()
        with self.assertRaises(ValueError):
            guard_split('train')

    def test_dev_csv_joins_exact_index_and_rejects_label_changes(self):
        from cnn_experiment.data import PROTOCOL
        data = fixture()
        for row in data.records:
            row.update(split='dev', kind='frame', indices=[row['id']])
        class Schedule:
            def __init__(self, path):
                pass
            def records(self, split, kind):
                return data.records
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            predictions = folder/'predictions.csv'
            fields = ['candidate_id', 'source_file', 'timestamp_exact', 'label', 'probability']
            rows = [dict(zip(fields, [r['id'], r['source_file'], r['timestamps_exact'][0], r['label'], p['probability']]))
                    for r, p in zip(data.records, data.predictions)]
            def write_rows():
                with predictions.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(reversed(rows))
            write_rows()
            state = {'protocol': PROTOCOL, 'index_sha256': 'fixture_hash', 'config': {'threshold': .5}}
            with patch('cnn_experiment.gradcam.DevSchedule', Schedule), \
                 patch('cnn_experiment.gradcam.digest', return_value='fixture_hash'), \
                 patch('cnn_experiment.gradcam.torch.load', return_value=state):
                loaded = load_explanation_data('fixture.pt', predictions, index_path='fixture.sqlite')
                self.assertEqual(loaded.predictions, data.predictions)
                rows[0]['label'] = 0
                write_rows()
                with self.assertRaisesRegex(ValueError, 'labels/candidate IDs'):
                    load_explanation_data('fixture.pt', predictions, index_path='fixture.sqlite')

    def test_authorized_interface_with_synthetic_artifacts_only(self):
        # No held-out files or model execution: fixture CSVs validate the future interface.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/'frozen_policy.json').write_text(json.dumps({'approval': FINAL_TEST_APPROVAL,
                'checkpoint_sha256': 'fixture_hash', 'topk_fraction': .05, 'fps': 10, 'threshold': .7}))
            (folder/'test_metrics.json').write_text(json.dumps({'scheduled_predictions': 2}))
            (folder/'test_video_predictions.json').write_text(json.dumps([{'source_file': 'test/SYNTHETIC.mp4',
                'source_family': 'test/SYNTHETIC.mp4', 'target': 1, 'probability': .8, 'samples': 2}]))
            path = folder/'test_frame_predictions.csv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle); writer.writerow(['source_file', 'timestamp_exact', 'label', 'probability'])
                writer.writerow(['test/SYNTHETIC.mp4', '0', 1, .8])
                writer.writerow(['test/SYNTHETIC.mp4', '1/10', 0, .2])
            with patch('cnn_experiment.gradcam.digest', return_value='fixture_hash'):
                loaded = load_explanation_data('fixture.pt', path, split='test',
                    frozen_policy_path=folder/'frozen_policy.json', approval=FINAL_TEST_APPROVAL)
                self.assertEqual(loaded.frame_threshold, .7)
                self.assertEqual(select_examples(loaded, 'tp', 1), [0])
                self.assertEqual(top5_contributors(loaded, 'test/SYNTHETIC.mp4')['k'], 1)
                (folder/'test_metrics.json').unlink()
                with self.assertRaises(FileNotFoundError):
                    load_explanation_data('fixture.pt', path, split='test',
                        frozen_policy_path=folder/'frozen_policy.json', approval=FINAL_TEST_APPROVAL)


if __name__ == '__main__':
    unittest.main()
