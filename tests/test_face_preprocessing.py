"""Focused synthetic preprocessing tests; no CNN training, weights or TEST input."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cnn_experiment.config import Config
from cnn_experiment.data import FrameReader
from cnn_face_experiment.crops import CropConfig, FacePipeline, crop_rectangle
from cnn_face_experiment.preprocessing import (FACE_OVAL, PREPROCESSING_MODES, MaskConfig,
    FaceMaskPipeline, MaskReader, face_alpha, masked_frame, make_preprocessing,
    preprocessing_previews, select_landmarks)
from cnn_face_experiment.preprocessing import ensure_opencv_runtime, ensure_landmark_runtime


def landmarks():
    points = np.full((478, 2), 112., dtype=np.float32)
    angles = np.arange(len(FACE_OVAL))*2*np.pi/len(FACE_OVAL)-np.pi/2
    points[list(FACE_OVAL)] = np.stack((112+50*np.cos(angles), 112+70*np.sin(angles)), axis=1)
    return points


def crop():
    return {'status': 'detected', 'confidence': .95, 'bbox_xywh': [60, 40, 104, 145],
            'crop_xyxy': crop_rectangle([60, 40, 104, 145], 224, 224, .30), 'faces_detected': 1}


def record(label=0):
    return {'id': 1, 'source_file': 'train/synthetic.mkv', 'source_family': 'train/family',
            'kind': 'frame', 'split': 'train', 'indices': [0], 'timestamps_exact': ['0'],
            'label': label, 'video_label': label, 'metric_eligible': True}


class FacePreprocessingTests(unittest.TestCase):
    def test_single_contrib_provider_and_landmark_dependencies_are_satisfied(self):
        runtime = ensure_landmark_runtime(install=False)
        self.assertEqual(runtime['opencv-contrib-python'], '4.13.0.92')
        self.assertEqual(runtime['cv2'], '4.13.0')
        self.assertEqual(runtime['mediapipe'], '0.10.32')

    def test_conflicting_provider_rejected_without_install(self):
        versions = {'opencv-contrib-python': '4.13.0.92', 'opencv-python-headless': '4.13.0.92'}
        with patch('cnn_face_experiment.preprocessing._opencv_distributions', return_value=versions), \
             patch('subprocess.run') as run:
            with self.assertRaises(RuntimeError): ensure_opencv_runtime(install=False)
            run.assert_not_called()

    def test_loaded_cv2_cannot_be_replaced_in_running_kernel(self):
        with patch('cnn_face_experiment.preprocessing._opencv_distributions', return_value={'opencv-python-headless':'4.13.0.92'}), \
             patch.dict(sys.modules, {'cv2': object()}), patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Restart the remote kernel'):
                ensure_opencv_runtime(install=True)
            run.assert_not_called()

    def test_wheel_download_failure_does_not_uninstall_existing_provider(self):
        with patch('cnn_face_experiment.preprocessing._opencv_distributions', return_value={'opencv-python-headless':'4.13.0.92'}), \
             patch.dict(sys.modules), patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'pip download')) as run:
            sys.modules.pop('cv2', None)
            with self.assertRaises(subprocess.CalledProcessError): ensure_opencv_runtime(install=True)
            self.assertEqual(run.call_count, 1)
            self.assertIn('download', run.call_args.args[0])
            self.assertNotIn('uninstall', run.call_args.args[0])

    def pipeline(self, folder):
        path = Path(folder)
        detector, task = path/'detector.onnx', path/'landmarker.task'
        detector.write_bytes(b'synthetic detector'); task.write_bytes(b'synthetic task')
        with patch('cnn_face_experiment.preprocessing.ensure_detector'), patch('cnn_face_experiment.preprocessing.ensure_landmarker'):
            return FaceMaskPipeline(detector, task, path/'mask.sqlite', 'index_fixture', 'pool_fixture')

    def test_exactly_two_modes_and_crop_object_identity_unchanged(self):
        self.assertEqual(PREPROCESSING_MODES, ('crop_30', 'face_mask'))
        with tempfile.TemporaryDirectory() as folder, patch('cnn_face_experiment.crops.ensure_detector'):
            old = FacePipeline('fixture.onnx', Path(folder)/'old.sqlite', 'index', 'pool', CropConfig(margin=.30))
            new = make_preprocessing('crop_30', 'fixture.onnx', Path(folder)/'new.sqlite', 'index', 'pool')
            self.assertIs(type(new), FacePipeline)
            self.assertEqual(old.identity, new.identity)
            self.assertEqual(old.config, new.config)
        with self.assertRaises(ValueError): make_preprocessing('tight_crop', None, None, None, None)
        with self.assertRaises(ValueError): make_preprocessing('face_mask', None, None, None, None)

    def test_scale_pixels_full_face_and_neutral_background(self):
        original = np.random.default_rng(4266).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        result, alpha, details = masked_frame(original, landmarks(), crop())
        self.assertEqual(result.shape, original.shape); self.assertEqual(result.dtype, original.dtype)
        np.testing.assert_array_equal(result[alpha == 1], original[alpha == 1])
        self.assertTrue(np.all(result[alpha == 0] == 128))
        self.assertTrue(all(details['regions_fully_preserved'].values()))
        self.assertFalse(details['facial_regions_clipped'])
        self.assertLess(details['retained_area_fraction'], .7)
        self.assertEqual(details['spatial_scale'], 1.)
        np.testing.assert_array_equal(result, masked_frame(original, landmarks(), crop())[0])

    def test_outward_feather_does_not_fade_facial_landmarks(self):
        alpha, margin = face_alpha(landmarks())
        self.assertGreater(margin, 0)
        self.assertGreater(np.sum((alpha > 0) & (alpha < 1)), 100)
        self.assertLess(max(np.abs(np.diff(alpha, axis=0)).max(), np.abs(np.diff(alpha, axis=1)).max()), .5)
        for x, y in landmarks()[list(FACE_OVAL)]: self.assertEqual(alpha[round(y), round(x)], 1.)

    def test_missing_and_invalid_landmarks_log_full_frame_fallback(self):
        original = np.full((224, 224, 3), 77, np.uint8)
        for points, reason in ((None, 'no_landmarks'), (np.zeros((5, 2)), 'invalid_landmarks')):
            output, alpha, metadata = masked_frame(original, points, crop())
            np.testing.assert_array_equal(output, original)
            self.assertTrue(np.all(alpha == 1)); self.assertEqual(metadata['fallback'], 'full_frame')
            self.assertEqual(metadata['status'], 'fallback_'+reason)
            self.assertIsNone(metadata['facial_regions_clipped'])

    def test_clipping_diagnostics_distinguish_crop_from_full_frame_mask(self):
        narrow = dict(crop(), crop_xyxy=[90, 90, 134, 134])
        _, _, metadata = masked_frame(np.zeros((224, 224, 3), np.uint8), landmarks(), narrow)
        self.assertTrue(metadata['crop_facial_regions_clipped'])
        self.assertFalse(metadata['facial_regions_clipped'])
        points = landmarks(); points[10] = [112, -5]
        _, _, metadata = masked_frame(np.zeros((224, 224, 3), np.uint8), points, crop())
        self.assertTrue(metadata['facial_regions_clipped'])
        self.assertTrue(metadata['face_touches_source_edge'])

    def test_landmark_primary_match_no_labels_or_order_dependency(self):
        points = landmarks(); other = points+[100, 100]
        chosen, reason = select_landmarks([other, points], crop()['bbox_xywh'])
        self.assertIsNone(reason); np.testing.assert_array_equal(chosen, points)
        np.testing.assert_array_equal(select_landmarks([points, other], crop()['bbox_xywh'])[0], points)
        self.assertEqual(select_landmarks([], crop()['bbox_xywh'])[1], 'no_landmarks')
        self.assertEqual(select_landmarks([points], None)[1], 'no_face')

    def test_cache_reuse_label_independence_and_pickling(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = self.pipeline(folder); reader = MaskReader(folder, pipeline)
            image = np.full((224, 224, 3), 42, np.uint8)
            with patch('cnn_face_experiment.preprocessing.YuNet') as detector, patch('cnn_face_experiment.preprocessing.DenseLandmarker') as dense:
                detector.return_value.detect.return_value = crop()
                dense.return_value.detect.return_value = [landmarks()]
                first, _, metadata = reader.transform(record(0), image)
                second, _, reused = reader.transform(record(1), image)
                self.assertEqual(dense.return_value.detect.call_count, 1)
                np.testing.assert_array_equal(first, second); self.assertEqual(metadata, reused)
                cloned = pickle.loads(pickle.dumps(reader))
                self.assertIsNone(cloned.detector); self.assertIsNone(cloned.landmarker)
                np.testing.assert_array_equal(cloned.transform(record(), image)[0], first)
                cloned.close()
            reader.close()
            self.assertEqual(pipeline.identity['kind'], 'face_mask')
            self.assertNotEqual(pipeline.identity.get('kind'), 'face_crop')

    def test_model_runtime_errors_raise_instead_of_becoming_missing_landmarks(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = self.pipeline(folder); reader = MaskReader(folder, pipeline)
            with patch('cnn_face_experiment.preprocessing.YuNet') as detector, patch('cnn_face_experiment.preprocessing.DenseLandmarker') as dense:
                detector.return_value.detect.return_value = crop()
                dense.return_value.detect.side_effect = RuntimeError('broken backend')
                with self.assertRaisesRegex(RuntimeError, 'broken backend'):
                    reader.transform(record(), np.zeros((224, 224, 3), np.uint8))
                self.assertIsNone(reader.boxes.get(record()))
            reader.close()

    def test_no_landmarks_fallback_is_persisted_not_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            reader = MaskReader(folder, self.pipeline(folder))
            with patch('cnn_face_experiment.preprocessing.YuNet') as detector, patch('cnn_face_experiment.preprocessing.DenseLandmarker') as dense:
                detector.return_value.detect.return_value = crop()
                dense.return_value.detect.return_value = []
                image = np.full((224, 224, 3), 50, np.uint8)
                np.testing.assert_array_equal(reader.transform(record(), image)[0], image)
                self.assertEqual(reader.boxes.get(record())['status'], 'fallback_no_landmarks')
            reader.close()

    def test_test_guard_and_train_only_preview_guard_before_decoding(self):
        with tempfile.TemporaryDirectory() as folder:
            reader = MaskReader(folder, self.pipeline(folder))
            forbidden = dict(record(), split='test', source_file='test/synthetic.mkv')
            with patch.object(reader, 'original') as decode:
                with self.assertRaises(ValueError): reader.read([forbidden])
                decode.assert_not_called()
            with self.assertRaises(ValueError): preprocessing_previews([dict(record(), split='dev')], folder, None, None, Path(folder)/'preview')
            self.assertFalse((Path(folder)/'preview').exists())
            reader.close()

    def test_shape_policy_never_resizes_or_creates_central_face_only_mask(self):
        with self.assertRaises(ValueError): masked_frame(np.zeros((112, 112, 3), np.uint8), landmarks(), crop())
        with self.assertRaises(ValueError): MaskConfig(feather_pixels=0).validate()
        with self.assertRaises(ValueError): MaskConfig(fallback='central_crop').validate()

    def test_spawned_cached_mask_loader_matches_single_worker(self):
        import av
        import torch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'train').mkdir()
            image = np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8)
            with av.open(str(root/'train/synthetic.mkv'), 'w') as container:
                stream = container.add_stream('ffv1', rate=10); stream.width = stream.height = 224; stream.pix_fmt = 'bgr0'
                frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            decoded = FrameReader(root).read([record()])[0][0]
            pipeline = self.pipeline(root); reader = MaskReader(root, pipeline)
            with patch('cnn_face_experiment.preprocessing.YuNet') as detector, patch('cnn_face_experiment.preprocessing.DenseLandmarker') as dense:
                detector.return_value.detect.return_value = crop(); dense.return_value.detect.return_value = [landmarks()]
                reader.transform(record(), decoded)
            reader.close()
            cfg = replace(Config(), workers=0, batch_size=1, cache_mib=1)
            one = next(iter(pipeline.make_loader([record()], root, cfg)))[0]
            spawned = next(iter(pipeline.make_loader([record()], root, replace(cfg, workers=1))))[0]
            self.assertTrue(torch.equal(one, spawned))

    def test_notebook_has_two_modes_and_preserves_existing_training_report_cells(self):
        path = Path(__file__).resolve().parents[1]/'notebooks/experiment1_face_cnn.ipynb'
        if not path.is_file(): return
        notebook = json.loads(path.read_text(encoding='utf-8'))
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        self.assertIn("PREPROCESSING_MODE = 'face_mask'", source)
        self.assertIn("record['split'] == 'train'", source)
        self.assertIn('preprocessing_previews(MATCHED_TRAIN_RECORDS', source)
        self.assertIn('PIPELINE = CROP_PIPELINE if', source)
        self.assertGreaterEqual(len(notebook['cells']), 23)
        self.assertEqual([cell['id'] for cell in notebook['cells'][:23]],
                         ['face-cnn-%02d' % i for i in range(23)])
        self.assertTrue(all(cell['id'].startswith('roc-de-') for cell in notebook['cells'][23:]))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code': compile(''.join(cell['source']), str(path), 'exec')


if __name__ == '__main__': unittest.main()
