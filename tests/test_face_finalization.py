"""Mask-plan provenance, restored assets and reporting; synthetic CPU fixtures only."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from cnn_experiment.backup import package_model_run, restore_local_zip
from cnn_experiment.config import experiment_config
from cnn_experiment.reporting import comparison_rows, GRAPH_ORDER
from cnn_face_experiment.crops import BoxCache
from cnn_face_experiment.preprocessing import MaskConfig, MaskReader, FaceMaskPipeline, masked_frame
from cnn_face_experiment.experiments import EXPERIMENTS, experiment_configurations, train_experiment
from tests import test_face_preprocessing as preproc_fixtures
from tests.test_face_preprocessing import landmarks, crop, record
from tests.test_reporting import fixture


class FaceFinalizationTests(unittest.TestCase):
    def pipeline(self, path): return preproc_fixtures.FacePreprocessingTests().pipeline(path)

    def test_exact_mask_policy_and_cache_identity_rejects_policy_change(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.pipeline(temp)
            self.assertEqual(pipeline.mask_config, MaskConfig(.06, 4, (128, 128, 128), .5, .5, 5, 'full_frame'))
            self.assertEqual(pipeline.identity['mask'], json.loads(json.dumps(asdict(MaskConfig()))))
            for field in ('kind','version','mask','crop','detector_sha256','landmarker_sha256','mediapipe','opencv','index_sha256','pool_sha256'):
                self.assertIn(field, pipeline.identity)
            changed = copy.deepcopy(pipeline.identity); changed['mask']['feather_pixels'] = 6
            with self.assertRaisesRegex(ValueError, 'policy'):
                BoxCache(pipeline.cache_path, changed).connection()

    def test_train_dev_share_identical_mask_policy_and_tensor_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.pipeline(temp)
            train = record(); dev = dict(record(), split='dev', source_file='dev/synthetic.mkv')
            image = np.random.default_rng(3).integers(0, 255, (224, 224, 3), dtype=np.uint8)
            _, _, metadata = masked_frame(image, landmarks(), crop())
            cache = BoxCache(pipeline.cache_path, pipeline.identity)
            for row in (train, dev): cache.put(row, metadata)
            cache.close()
            cfg = replace(experiment_config('resnet50', 'fixed20k_lr15e6'), workers=0, flip_probability=0.)
            with patch('cnn_experiment.data.FrameReader.read', return_value=[[image]]):
                train_data = pipeline.dataset([train], temp, cfg, training=True)
                dev_data = pipeline.dataset([dev], temp, cfg, training=False)
                try:
                    self.assertTrue(torch.equal(train_data[0][0], dev_data[0][0]))
                    self.assertEqual(train_data.reader.boxes.policy, dev_data.reader.boxes.policy)
                    self.assertEqual(dev_data.flip_probability, 0.)
                finally:
                    train_data.reader.close(); dev_data.reader.close()

    def test_snapshot_clipping_fallback_counts_and_rolling_final_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pipeline = self.pipeline(root)
            image = np.zeros((224,224,3), np.uint8)
            cache = BoxCache(pipeline.cache_path, pipeline.identity)
            _, _, success = masked_frame(image, landmarks(), crop())
            _, _, fallback = masked_frame(image, None, crop())
            warning = dict(success, facial_regions_clipped=True)
            for i, metadata in enumerate((success, fallback, warning)):
                row = dict(record(), split='dev', source_file='dev/synthetic.mkv', indices=[i], timestamps_exact=[str(i)])
                cache.put(row, metadata)
            cache.close()
            run = root/'run'; fixture(run)  # synthetic histories and deliberately fake checkpoint bytes
            (run/'last.pt').write_bytes(b'synthetic last')
            (run/'input_pipeline.json').write_text(json.dumps(pipeline.identity))
            pipeline.snapshot(run)
            report = json.loads((run/'face_detection_report.json').read_text())
            self.assertEqual(report['fallback_count'], 1)
            self.assertEqual(report['by_split']['dev']['clipping_warning_count'], 1)
            self.assertEqual(report['by_split']['dev']['clipping_check_unknown'], 1)
            self.assertEqual(report['dev_frames_excluded'], 0)
            before = (run/'best.pt').read_bytes()
            first = package_model_run(run, 'resnet50', 1)
            second = package_model_run(run, 'resnet50', 2)
            final = package_model_run(run, 'resnet50', 2, final=True)
            self.assertEqual(first['archive'], second['archive']); self.assertNotEqual(second['archive'], final['archive'])
            with zipfile.ZipFile(final['archive']) as archive:
                self.assertTrue({'best.pt','last.pt','history.csv','input_pipeline.json','face_boxes.sqlite',
                    'face_detection_report.json','face_detector.onnx','face_landmarker.task'} <= set(archive.namelist()))
                self.assertIsNone(archive.testzip())
            restored = restore_local_zip(final['archive'], root/'restored')
            self.assertEqual((restored/'face_landmarker.task').read_bytes(), pipeline.landmark_weights.read_bytes())
            restored_cache = BoxCache(restored/'face_boxes.sqlite', pipeline.identity)
            self.assertEqual(restored_cache.get(dict(record(), split='dev', source_file='dev/synthetic.mkv')), success)
            restored_cache.close()
            self.assertEqual((run/'best.pt').read_bytes(), before)

    def test_successful_masks_are_not_reported_as_fallback_cohort(self):
        from cnn_face_experiment.reporting import failure_cohorts
        with tempfile.TemporaryDirectory() as temp:
            rows = [dict(record(), id=i, split='dev', source_file='dev/synthetic.mkv', indices=[i], timestamps_exact=[str(i)], label=i%2) for i in range(4)]
            cache = BoxCache(Path(temp)/'face_boxes.sqlite', {'kind':'face_mask'})
            for i, row in enumerate(rows): cache.put(row, {'status': 'masked' if i < 2 else 'fallback_no_landmarks'})
            cache.close()
            predictions = [{'record_index':i, 'probability':.8 if r['label'] else .1} for i,r in enumerate(rows)]
            result = failure_cohorts(temp, rows, predictions, predictions)
            self.assertEqual(result['cohorts']['detected']['n'], 2)
            self.assertEqual(result['cohorts']['fallback']['n'], 2)
            self.assertIn('face_mask', result['cohorts']['detected'])

    def test_six_rows_have_mask_identities_and_exclude_crop_as_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask = root/'mask'; fixture(mask)
            (mask/'config.json').write_text(json.dumps({'model':'resnet50','head_lr':1e-4,'threshold':.5}))
            (mask/'input_pipeline.json').write_text(json.dumps({'kind':'face_mask'}))
            rows = comparison_rows({'resnet50_face_crop':mask})
            self.assertEqual(len(rows), 6)
            selected = next(r for r in rows if r['Model'] == 'Face-masked ResNet50')
            self.assertEqual(selected['Input'], 'face_mask'); self.assertEqual(selected['Status'], 'Completed')
            (mask/'input_pipeline.json').write_text(json.dumps({'kind':'face_crop'}))
            rows = comparison_rows({'resnet50_face_crop':mask})
            selected = next(r for r in rows if r['Model'] == 'Face-masked ResNet50')
            self.assertTrue(selected['Status'].startswith('Pending')); self.assertIsNone(selected['Video F2'])
            self.assertEqual([r['Level'] for r in rows].count('Clip'), 1)

    def test_full_frame_baseline_cannot_fill_the_new_head_lr_control(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture(temp)
            (Path(temp)/'config.json').write_text(json.dumps({'model':'efficientnet_b0','head_lr':1e-4,'threshold':.5}))
            row = comparison_rows({'efficientnet_b0_head_lr_2e4':temp})[-1]
            self.assertTrue(row['Status'].startswith('Pending')); self.assertIsNone(row['Video F2'])
            (Path(temp)/'config.json').write_text(json.dumps({'model':'efficientnet_b0','head_lr':2e-4,'threshold':.5}))
            self.assertEqual(comparison_rows({'efficientnet_b0_head_lr_2e4':temp})[-1]['Status'], 'Completed')

    def test_only_three_new_recipes_unchanged_hyperparameters_and_imagenet_source(self):
        configs = experiment_configurations()
        self.assertEqual(len(configs), 3)
        for name, config in configs.items():
            baseline = experiment_config(EXPERIMENTS[name][0], 'fixed20k_lr15e6')
            different = {key for key,value in config.as_dict().items() if baseline.as_dict()[key] != value}
            self.assertEqual(different, {'head_lr'} if name.endswith('2e4') else set())
        with tempfile.TemporaryDirectory() as temp, patch('cnn_face_experiment.experiments.existing_pool', return_value='pool'), \
             patch('cnn_face_experiment.experiments.digest', side_effect=lambda p:'index' if str(p)=='index' else 'pool'), \
             patch('cnn_experiment.training.train') as train:
            from types import SimpleNamespace
            pipeline = SimpleNamespace(identity={'kind':'face_mask','index_sha256':'index','pool_sha256':'pool'})
            for name in configs:
                train_experiment(name, 'index', 'pool', 'videos', 'imagenet_weights', Path(temp)/name/'run', pipeline)
            self.assertEqual(train.call_count, 3)
            for call in train.call_args_list:
                self.assertEqual(call.args[3], 'imagenet_weights')
                self.assertIsNone(call.args[5])  # no completed checkpoint used for initialization
            self.assertIs(train.call_args_list[0].kwargs['input_pipeline'], pipeline)
            self.assertIs(train.call_args_list[1].kwargs['input_pipeline'], pipeline)
            self.assertIsNone(train.call_args_list[2].kwargs['input_pipeline'])
            plan = json.loads((Path(temp)/'resnet50_face_crop/experiment_plans.json').read_text())
            self.assertEqual(plan['resnet50']['preprocessing_mode'], 'face_mask')

    def test_checkpoint_input_identity_and_assets_committed_with_mocked_compute(self):
        from cnn_experiment import training
        from tests.test_fixed_pool import PoolTests
        original = training._train
        identity = {'kind':'face_mask','mask':asdict(MaskConfig()),'landmarker_sha256':'fixture-task','detector_sha256':'fixture-yunet'}
        class Pipeline:
            def __init__(self): self.identity = identity
            def make_loader(self, *args, **kwargs): return training.make_loader(*args, **kwargs)
            def snapshot(self, path):
                for name,data in [('face_landmarker.task',b'synthetic task'),('face_detector.onnx',b'synthetic yunet'),('face_detection_report.json',b'{"fallback_count":0,"clipping_warning_count":0}')]:
                    (Path(path)/name).write_bytes(data)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs, input_pipeline=Pipeline())
            for name in ('last.pt','best.pt'):
                state = torch.load(Path(result)/name, map_location='cpu', weights_only=True)
                self.assertEqual(state['input_pipeline'], identity)
            rolling = next((Path(result).parent/'downloads').glob('*latest*.zip'))
            with zipfile.ZipFile(rolling) as archive: self.assertIn('face_landmarker.task', archive.namelist())
            return result
        with patch.object(training, '_train', side_effect=wrapped):
            PoolTests('test_shared_training_engine_commits_pool_identity_and_final_artifacts').test_shared_training_engine_commits_pool_identity_and_final_artifacts()

    def test_notebook_default_mask_cache_restore_and_no_additional_training_calls(self):
        root = Path(__file__).resolve().parents[1]
        path = root/'notebooks/experiment1_face_cnn.ipynb'
        if not path.is_file(): self.skipTest('Local-only notebook not in source package')
        notebook = json.loads(path.read_text(encoding='utf-8'))
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        self.assertIn("PREPROCESSING_MODE = 'face_mask'", source)
        self.assertIn('MASK_CACHE_OVERRIDE', source)
        self.assertEqual(source.count('= train_experiment('), 3)
        self.assertEqual(GRAPH_ORDER, ('training_curves.png','best_dev_confusion.png','best_dev_roc_pr.png','dev_f2_reporting_confusion.png'))


if __name__ == '__main__': unittest.main()
