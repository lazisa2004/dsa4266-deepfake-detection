"""Synthetic/metadata-only face-experiment regression tests; no held-out reads."""
from dataclasses import asdict, replace
import hashlib
from fractions import Fraction
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from cnn_face_experiment.crops import (CropConfig, BoxCache, FacePipeline, FaceReader,
    choose_primary, crop_rectangle, frame_key)
from cnn_face_experiment.experiments import (experiment_configurations, output_paths, existing_pool)
from cnn_experiment.config import experiment_config
from cnn_experiment.data import LazyDataset, make_loader
from cnn_experiment.models import StagedCNN, make_optimizer
from cnn_experiment.pool import FixedPoolSampler, DevSchedule


def record(i=0):
    return {'id': i, 'split': 'dev', 'kind': 'frame', 'source_file': 'dev/synthetic.mp4',
            'source_family': 'dev/synthetic.mp4', 'indices': [i], 'timestamps_exact': [str(Fraction(i, 25))],
            'label': i % 2, 'video_label': 1, 'metric_eligible': True}


def face(x, y, w, h, score):
    return [x, y, w, h]+[0]*10+[score]


class FaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def pipeline(self, folder):
        with patch('cnn_face_experiment.crops.ensure_detector'):
            return FacePipeline(Path(folder)/'fixture.onnx', Path(folder)/'boxes.sqlite', 'index_fixture', 'pool_fixture')

    def test_crop_bounds_margin_and_square(self):
        self.assertEqual(crop_rectangle([70, 60, 80, 100], 224, 224, .3), [45, 45, 175, 175])
        for box in ([0, 0, 60, 90], [195, 200, 50, 50], [-20, -10, 60, 90], [0, 0, 1000, 1000]):
            x1, y1, x2, y2 = crop_rectangle(box, 224, 224)
            self.assertTrue(0 <= x1 < x2 <= 224 and 0 <= y1 < y2 <= 224)
            self.assertEqual(x2-x1, y2-y1)
        with self.assertRaises(ValueError):
            crop_rectangle([0, 0, 0, 20], 224, 224)
        with self.assertRaises(ValueError):
            CropConfig(margin=.8).validate()

    def test_primary_selection_ties_confidence_and_fallback(self):
        cfg = CropConfig()
        faces = [face(10, 20, 20, 20, .99), face(60, 60, 80, 80, .85), face(0, 0, 100, 100, .1)]
        result = choose_primary(faces, 224, 224, cfg)
        self.assertEqual(result['bbox_xywh'], [60, 60, 80, 80])
        self.assertEqual(result, choose_primary(list(reversed(faces)), 224, 224, cfg))
        fallback = choose_primary(None, 224, 224, cfg)
        self.assertEqual(fallback['status'], 'fallback_no_face'); self.assertEqual(fallback['crop_xyxy'], [0, 0, 224, 224])

    def test_cache_reuse_policy_and_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cache.sqlite'; cache = BoxCache(path, {'policy': 1})
            value = choose_primary(None, 224, 224, CropConfig())
            cache.put(record(), value)
            changed_label = dict(record(), label=1)
            self.assertEqual(cache.get(changed_label), value)  # labels never select detector/crop
            snapshot = Path(folder)/'snapshot.sqlite'; cache.snapshot(snapshot); cache.close()
            restored = BoxCache(snapshot, {'policy': 1})
            self.assertEqual(restored.get(record()), value); restored.close()
            bad = BoxCache(path, {'policy': 2})
            with self.assertRaises(ValueError): bad.connection()
            bad.close()

    def test_pipeline_snapshot_is_portable_and_preserves_live_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); pipeline = self.pipeline(folder)
            pipeline.weights.write_bytes(b'synthetic detector fixture')
            cache = BoxCache(pipeline.cache_path, pipeline.identity)
            value = choose_primary(None, 224, 224, CropConfig()); cache.put(record(), value)
            with self.assertRaisesRegex(ValueError, 'differ from the live'):
                cache.snapshot(pipeline.cache_path)
            cache.close()
            pipeline.snapshot(root/'run')
            self.assertTrue((root/'run/face_detector_LICENSE.txt').is_file())
            restored = BoxCache(root/'run/face_boxes.sqlite', pipeline.identity)
            self.assertEqual(restored.get(record()), value); restored.close()
            live = BoxCache(pipeline.cache_path, pipeline.identity)
            self.assertEqual(live.get(record()), value); live.close()

    def test_failure_cohorts_match_frame_ids_and_keep_both_classes(self):
        from cnn_face_experiment.reporting import failure_cohorts
        rows = [record(i) for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            cache = BoxCache(Path(folder)/'face_boxes.sqlite', {'fixture': True})
            for i, row in enumerate(rows):
                value = choose_primary(None, 224, 224, CropConfig())
                if i < 2: value.update(status='detected', confidence=.9)
                cache.put(row, value)
            cache.close()
            original = [{'record_index': i, 'probability': .2 if r['label'] == 0 else .8} for i, r in enumerate(rows)]
            result = failure_cohorts(folder, rows, original, original)
            self.assertEqual(result['dev_frames_excluded'], 0)
            for cohort in result['cohorts'].values():
                self.assertEqual((cohort['n'], cohort['real'], cohort['fake']), (2, 1, 1))
                self.assertEqual(cohort['original'], cohort['face_crop'])

    def test_rgb_normalization_labels_and_fallback_identical(self):
        image = np.zeros((224, 224, 3), np.uint8); image[..., 0] = 255
        rows = [record(), record(1)]
        originals = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as folder:
            pipeline = self.pipeline(folder)
            metadata = choose_primary(None, 224, 224, CropConfig())
            cache = BoxCache(pipeline.cache_path, pipeline.identity)
            for row in rows: cache.put(row, metadata)
            cache.close()
            with patch('cnn_experiment.data.FrameReader.read', return_value=[[image] for _ in rows]):
                plain = LazyDataset(rows, folder).__getitems__([0, 1])
                cropped = pipeline.dataset(rows, folder, experiment_config('resnet50', 'fixed20k_lr15e6')).__getitems__([0, 1])
            for left, right in zip(plain, cropped):
                self.assertTrue(torch.equal(left[0], right[0])); self.assertTrue(torch.equal(left[1], right[1]))
                self.assertEqual(right[0].shape, (3, 224, 224))
                self.assertAlmostEqual(float(right[0][0, 0, 0]), (1-.485)/.229, places=5)
            self.assertEqual(rows, originals)

    def test_crop_applied_before_existing_flip_and_normalization(self):
        import cv2
        image = np.arange(224*224*3, dtype=np.uint8).reshape(224, 224, 3)
        row = dict(record(), split='train', source_file='train/synthetic.mp4', source_family='train/synthetic.mp4')
        with tempfile.TemporaryDirectory() as folder:
            pipeline = self.pipeline(folder)
            cached = {'status': 'detected', 'confidence': .99, 'bbox_xywh': [30, 40, 50, 60],
                      'crop_xyxy': [20, 30, 120, 130], 'faces_detected': 1}
            cache = BoxCache(pipeline.cache_path, pipeline.identity); cache.put(row, cached); cache.close()
            cfg = replace(experiment_config('resnet50', 'fixed20k_lr15e6'), flip_probability=1.)
            with patch('cnn_experiment.data.FrameReader.read', return_value=[[image]]):
                actual, label, _ = pipeline.dataset([row], folder, cfg, training=True)[0]
            expected = cv2.resize(image[30:130, 20:120], (224, 224), interpolation=cv2.INTER_LINEAR)[:, ::-1].copy()
            expected = torch.from_numpy(expected).permute(2, 0, 1).float()/255
            expected = (expected-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
            self.assertTrue(torch.equal(actual, expected)); self.assertEqual(float(label), row['label'])

    def test_all_three_configs_and_optimizer_groups(self):
        before = experiment_config('efficientnet_b0', 'fixed20k_lr15e6').as_dict()
        configs = experiment_configurations()
        for name, cfg in configs.items():
            original = experiment_config(cfg.model, 'fixed20k_lr15e6').as_dict()
            changes = {k for k, v in cfg.as_dict().items() if v != original[k]}
            self.assertEqual(changes, {'head_lr'} if name.endswith('2e4') else set())
            model = StagedCNN(cfg, '.', pretrained=False)
            optimizer = make_optimizer(model, cfg)
            self.assertEqual({g['name']: g['lr'] for g in optimizer.param_groups}, {'head': cfg.head_lr, 'final_stage': 1.5e-5})
            model.set_phase(True); model.eval()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.rand(2, 3, 224, 224)).squeeze(1), torch.tensor([0., 1.]))
            loss.backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
        self.assertEqual(before, experiment_config('efficientnet_b0', 'fixed20k_lr15e6').as_dict())

    def test_same_sampler_ids_every_epoch(self):
        from tests.test_fixed_pool import pool_fixture
        with tempfile.TemporaryDirectory() as folder:
            pool = pool_fixture(Path(folder)/'train_pool.json')
            samplers = [FixedPoolSampler(pool, cfg.train_per_class, cfg.seed) for cfg in experiment_configurations().values()]
            for epoch in range(10):
                selected = [sampler.select(epoch)[0] for sampler in samplers]
                self.assertTrue(all(ids == selected[0] for ids in selected[1:])); self.assertEqual(len(set(selected[0])), 8000)

    def test_missing_manifest_and_isolated_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError): existing_pool(Path(folder)/'missing.json')
            paths = output_paths(folder)
            self.assertEqual(len(set(p.parent for p in paths.values())), 3)
            self.assertTrue(all(p.name == 'run' for p in paths.values()))

    def test_reject_test_before_any_decoder_access(self):
        row = dict(record(), split='test', source_file='test/SYNTHETIC.mp4')
        with self.assertRaises(ValueError): frame_key(row)

    def test_spawned_loader_with_synthetic_video(self):
        import av
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'dev').mkdir()
            with av.open(str(root/'dev/synthetic.mp4'), 'w') as out:
                stream = out.add_stream('mpeg4', rate=25); stream.width = stream.height = 224; stream.pix_fmt = 'yuv420p'
                for i in range(4):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((224, 224, 3), 30+i, np.uint8), format='rgb24')): out.mux(packet)
                for packet in stream.encode(): out.mux(packet)
            pipeline = self.pipeline(folder); rows = [record(i) for i in range(4)]
            cache = BoxCache(pipeline.cache_path, pipeline.identity)
            for row in rows: cache.put(row, choose_primary(None, 224, 224, CropConfig()))
            cache.close()
            cfg = replace(experiment_config('resnet50', 'fixed20k_lr15e6'), workers=0, batch_size=2)
            single = torch.cat([x for x, _, _ in pipeline.make_loader(rows, root, cfg)])
            spawned = torch.cat([x for x, _, _ in pipeline.make_loader(rows, root, replace(cfg, workers=2))])
            self.assertTrue(torch.equal(single, spawned))

    def test_gradcam_refuses_face_checkpoint_without_crop_pipeline(self):
        from cnn_experiment.gradcam import save_explanations, ExplanationData
        model = torch.nn.Identity(); model.explanation_preprocessing = {'kind': 'face_crop'}
        data = ExplanationData('dev', [record()], [{'record_index': 0, 'probability': .5}], .5, .5, 'fixture', 'fixture')
        with patch('cnn_experiment.gradcam.digest', return_value='fixture'), \
             patch('cnn_experiment.gradcam.load_model', return_value=(model, experiment_config('resnet50', 'fixed20k_lr15e6'), None)), \
             patch('cnn_experiment.gradcam.target_layer', return_value=(model, 'fixture')):
            with self.assertRaisesRegex(ValueError, 'exact saved input pipeline'):
                save_explanations(data, 'fixture', 'fixture', 'fixture', [0])

    def test_engine_metadata_snapshots_and_backups_with_mock_compute(self):
        from cnn_experiment import training
        from tests.test_fixed_pool import PoolTests
        original = training._train
        observed = []
        class Pipeline:
            identity = {'kind': 'face_crop', 'fixture': True}
            def make_loader(self, *args, **kwargs): return training.make_loader(*args, **kwargs)
            def snapshot(self, out): (Path(out)/'face_detection_report.json').write_text('{"fixture": true}')
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs, input_pipeline=Pipeline())
            state = torch.load(Path(result)/'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(state['input_pipeline'], Pipeline.identity)
            self.assertTrue((Path(result)/'input_pipeline.json').is_file())
            import zipfile
            rolling = next((Path(result).parent/'downloads').glob('*latest*.zip'))
            with zipfile.ZipFile(rolling) as z:
                self.assertIn('input_pipeline.json', z.namelist()); self.assertIn('face_detection_report.json', z.namelist())
            observed.append(True)
            return result
        with patch.object(training, '_train', side_effect=wrapped):
            PoolTests('test_shared_training_engine_commits_pool_identity_and_final_artifacts').test_shared_training_engine_commits_pool_identity_and_final_artifacts()
        self.assertEqual(observed, [True])


class PoolResolutionTests(unittest.TestCase):
    """Small synthetic manifest fixtures; no production pool build or training."""
    def setUp(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__(); self.addCleanup(quiet.__exit__, None, None, None)
        from cnn_face_experiment import experiments
        self.module = experiments
        self.payload = json.dumps({'policy': asdict(experiments.PoolConfig()), 'fixture': True}).encode()
        self.expected = hashlib.sha256(self.payload).hexdigest()
        self.hash_patch = patch.object(experiments, 'EXPECTED_TRAIN_POOL_SHA256', self.expected)
        self.hash_patch.start(); self.addCleanup(self.hash_patch.stop)
        validation = patch.object(experiments, 'FixedPool', return_value=SimpleNamespace(content={'policy': asdict(experiments.PoolConfig())}))
        self.validated = validation.start(); self.addCleanup(validation.stop)

    def test_matching_existing_manifest_avoids_zip_and_build(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(self.module, 'build_pool') as build:
            root = Path(folder); existing = root/'train_pool.json'; existing.write_bytes(self.payload)
            result = self.module.resolve_train_pool(root, root/'new', existing_paths=[existing], completed_run_zips=[root/'DO_NOT_OPEN.zip'])
            self.assertEqual(result, existing); self.assertFalse((root/'new').exists()); build.assert_not_called()
            self.validated.assert_called_once_with(existing, root)

    def test_zip_reads_only_exact_train_manifest_not_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(self.module, 'build_pool') as build:
            root = Path(folder); saved = root/'resnet50_final_fixture.zip'
            with zipfile.ZipFile(saved, 'w') as z:
                z.writestr('run/train_pool.json', self.payload)
                z.writestr('run/best.pt', b'never read or extracted')
            original_read = zipfile.ZipFile.read
            reads = []
            def guarded_read(archive, member, *args, **kwargs):
                name = member.filename if isinstance(member, zipfile.ZipInfo) else member
                reads.append(name)
                self.assertTrue(name.endswith('train_pool.json'))
                return original_read(archive, member, *args, **kwargs)
            with patch.object(zipfile.ZipFile, 'read', autospec=True, side_effect=guarded_read):
                result = self.module.resolve_train_pool(root, root/'pool', completed_run_zips=[saved])
            self.assertEqual(result.read_bytes(), self.payload); self.assertEqual(reads, ['run/train_pool.json'])
            self.assertEqual([p.name for p in (root/'pool').iterdir()], ['train_pool.json']); build.assert_not_called()

    def test_original_builder_defaults_are_used_then_hash_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def builder(data, destination, policy):
                self.assertEqual(data, root); self.assertEqual(policy, self.module.PoolConfig())
                self.assertEqual((policy.seed, policy.per_class, policy.fps), (4266, 10000, 10))
                destination.mkdir(); result = destination/'train_pool.json'; result.write_bytes(self.payload); return result
            with patch.object(self.module, 'build_pool', side_effect=builder) as build:
                result = self.module.resolve_train_pool(root, root/'pool')
            self.assertEqual(hashlib.sha256(result.read_bytes()).hexdigest(), self.expected); build.assert_called_once()

    def test_wrong_existing_or_override_hash_blocks_without_rebuild(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(self.module, 'build_pool') as build:
            root = Path(folder); bad = root/'train_pool.json'; bad.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                self.module.resolve_train_pool(root, root/'pool', override=bad)
            with self.assertRaisesRegex(ValueError, 'Training blocked'):
                self.module.resolve_train_pool(root, root/'pool', existing_paths=[bad])
            build.assert_not_called(); self.assertEqual(bad.read_bytes(), b'wrong')

    def test_bad_zip_hash_blocks_without_build_or_weight_extraction(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(self.module, 'build_pool') as build:
            root = Path(folder); saved = root/'efficientnet_b0_final_fixture.zip'
            with zipfile.ZipFile(saved, 'w') as z: z.writestr('train_pool.json', b'wrong')
            with self.assertRaisesRegex(ValueError, 'Training blocked'):
                self.module.resolve_train_pool(root, root/'pool', completed_run_zips=[saved])
            self.assertFalse((root/'pool').exists()); build.assert_not_called()

    def test_generated_mismatch_and_partial_build_block(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def bad_builder(data, destination, policy):
                destination.mkdir(); result = destination/'train_pool.json'; result.write_bytes(b'different generated pool'); return result
            with patch.object(self.module, 'build_pool', side_effect=bad_builder):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    self.module.resolve_train_pool(root, root/'pool')
            partial = root/'interrupted'; partial.mkdir(); (partial/'audit.json').write_text('preserved')
            with patch.object(self.module, 'build_pool') as build:
                with self.assertRaisesRegex(FileExistsError, 'Incomplete pool build'):
                    self.module.resolve_train_pool(root, partial)
                build.assert_not_called()
            self.assertEqual((partial/'audit.json').read_text(), 'preserved')

    def test_unsafe_zip_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); saved = root/'fixture.zip'
            with zipfile.ZipFile(saved, 'w') as z: z.writestr('../train_pool.json', self.payload)
            with self.assertRaisesRegex(ValueError, 'Unsafe/oversized'):
                self.module.resolve_train_pool(root, root/'pool', completed_run_zips=[saved])
            self.assertFalse((root/'pool').exists())

    def test_discovery_prunes_video_split_folders_and_skips_lstm(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'input').mkdir(); (root/'test').mkdir()
            for name in ('resnet50_final_fixture.zip', 'efficientnet_b0_latest_completed_epoch.zip', 'resnet50_lstm_final_fixture.zip', 'face_cnn_source_fixture.zip'):
                (root/'input'/name).write_bytes(b'fixture directory entry only')
            (root/'test/resnet50_final_DO_NOT_INSPECT.zip').write_bytes(b'not opened')
            self.assertEqual([p.name for p in self.module.attached_pool_zips(root)],
                ['efficientnet_b0_latest_completed_epoch.zip', 'resnet50_final_fixture.zip'])


if __name__ == '__main__':
    unittest.main()
