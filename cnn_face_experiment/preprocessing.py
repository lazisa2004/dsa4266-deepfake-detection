"""Two label-independent face inputs: unchanged crop_30, or soft full-frame masking.

Only TRAIN/DEV frame records are accepted. No CNN inference, sampling or TEST.
The original crops.py implementation is intentionally not modified.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np

from cnn_experiment.checkpoints import digest
from .crops import (BoxCache, CropConfig, FaceDataset, FacePipeline, FaceReader,
                    YuNet, ensure_detector, frame_key)

PREPROCESSING_MODES = ('crop_30', 'face_mask')
MASK_VERSION = 'full_face_oval_outer_feather_v1'
LANDMARK_VERSION = '0.10.32'
LANDMARK_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
                'face_landmarker/float16/1/face_landmarker.task')
LANDMARK_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
LANDMARK_PACKAGES = ('mediapipe==0.10.32', 'absl-py==2.3.1', 'flatbuffers==25.9.23',
                     'sounddevice==0.5.3', 'cffi==1.17.1', 'pycparser==2.22')
OPENCV_PACKAGE = 'opencv-contrib-python'
OPENCV_VERSION = '4.13.0.92'
OPENCV_DISTRIBUTIONS = ('opencv-python', 'opencv-python-headless',
                        'opencv-contrib-python', 'opencv-contrib-python-headless')
# Ordered official MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL.
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
REGIONS = {
    'forehead': (10, 67, 109, 338, 297, 103, 332),
    'eyebrows': (70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276),
    'eyes': (33, 133, 159, 145, 263, 362, 386, 374),
    'nose': (1, 4, 5, 6, 98, 327), 'cheeks': (234, 93, 132, 454, 323, 361),
    'mouth': (61, 291, 0, 17, 13, 14, 78, 308),
    'jaw': (172, 136, 150, 176, 152, 400, 378, 365, 397), 'chin': (152, 148, 377)}


@dataclass(frozen=True)
class MaskConfig:
    margin_fraction: float = .06  # outward radius / largest oval dimension
    feather_pixels: int = 4       # OUTSIDE the fully preserved face + margin
    neutral_rgb: tuple = (128, 128, 128)
    min_detection_confidence: float = .5
    min_presence_confidence: float = .5
    max_faces: int = 5
    fallback: str = 'full_frame'

    def validate(self):
        if not 0 < self.margin_fraction <= .15 or not 2 <= self.feather_pixels <= 12:
            raise ValueError('Use a small outward margin and 2-12px feather')
        if len(self.neutral_rgb) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in self.neutral_rgb):
            raise ValueError('neutral_rgb must contain three integer RGB values')
        if self.fallback != 'full_frame' or not 0 < self.min_detection_confidence < 1 or not 0 < self.min_presence_confidence < 1 or not 1 <= self.max_faces <= 10:
            raise ValueError('Invalid landmark policy; only explicit full-frame fallback is supported')
        return self


def ensure_landmarker(path, *, download=False):
    """Versioned public model, SHA-pinned; existing files are never overwritten."""
    path = Path(path)
    if path.is_file():
        if digest(path) != LANDMARK_SHA256:
            raise ValueError('Face landmarker SHA256 mismatch')
        return path
    if not download:
        raise FileNotFoundError('Attach face_landmarker_float16_v1.task or allow its explicit setup download')
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.task.download')
    with urllib.request.urlopen(LANDMARK_URL, timeout=90) as source, temporary.open('xb') as target:
        shutil.copyfileobj(source, target)
    if digest(temporary) != LANDMARK_SHA256:
        raise ValueError('Landmarker download hash mismatch; partial preserved')
    temporary.rename(path)
    return path


def _opencv_distributions():
    import importlib.metadata
    versions = {}
    for name in OPENCV_DISTRIBUTIONS:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return versions


def ensure_opencv_runtime(*, install=False):
    """Use exactly one cv2 provider satisfying MediaPipe, at the audited version.

    Download a compatible wheel before removing overlapping distributions. Never
    replace native cv2 inside a process that has already imported it. NumPy,
    torch, torchvision, detector settings and mask policy are left unchanged.
    """
    import importlib
    import importlib.metadata
    import subprocess
    import sys
    import tempfile
    versions = _opencv_distributions()
    if versions != {OPENCV_PACKAGE: OPENCV_VERSION}:
        if 'cv2' in sys.modules:
            raise RuntimeError('OpenCV providers differ from the audited setup and cv2 is already loaded. '
                               'Restart the remote kernel and run setup first; no packages were changed.')
        if not install:
            raise RuntimeError('Use only opencv-contrib-python==4.13.0.92; run '
                               'ensure_opencv_runtime(install=True) in a fresh kernel before importing cv2.')
        if int(np.__version__.split('.')[0]) < 2:
            raise RuntimeError('OpenCV 4.13 requires NumPy >=2. Preserve the experiment environment; '
                               'review dependencies before setup. No packages were changed.')
        with tempfile.TemporaryDirectory(prefix='dsa4266_opencv_') as folder:
            subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                            '--only-binary=:all:', '--dest', folder,
                            OPENCV_PACKAGE+'=='+OPENCV_VERSION], check=True)
            wheels = list(Path(folder).glob('*.whl'))
            if len(wheels) != 1:
                raise RuntimeError('Expected one compatible OpenCV wheel; existing packages preserved')
            # Uninstall all existing providers first: they share the same cv2 files.
            if versions:
                subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', *versions], check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(wheels[0])], check=True)
        importlib.invalidate_caches()
        if _opencv_distributions() != {OPENCV_PACKAGE: OPENCV_VERSION}:
            raise RuntimeError('OpenCV provider verification failed; stop and repair setup before training')
    import cv2
    if cv2.__version__ != '4.13.0' or not hasattr(cv2, 'FaceDetectorYN'):
        raise RuntimeError('Pinned OpenCV/YuNet import check failed; restart and repair setup before training')
    return {OPENCV_PACKAGE: OPENCV_VERSION, 'cv2': cv2.__version__}


def ensure_landmark_runtime(*, install=False):
    """Add pinned landmarks without changing existing NumPy/PyTorch packages.

    The single contrib provider satisfies MediaPipe's declared OpenCV dependency.
    Supporting packages already installed are preserved; missing ones are explicit.
    """
    import importlib.metadata
    import subprocess
    import sys
    runtime = ensure_opencv_runtime(install=install)
    missing = []
    for spec in LANDMARK_PACKAGES:
        name, expected = spec.split('==')
        try: actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: actual = None
        if actual is None or name == 'mediapipe' and actual != expected: missing.append(spec)
    if missing:
        if not install: raise RuntimeError('Landmark packages unavailable; run ensure_landmark_runtime(install=True) in setup')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir', *missing], check=True)
    if _opencv_distributions() != {OPENCV_PACKAGE: OPENCV_VERSION}:
        raise RuntimeError('OpenCV providers changed during landmark setup; stop before training')
    return dict(runtime, **{spec.split('==')[0]: importlib.metadata.version(spec.split('==')[0])
                            for spec in LANDMARK_PACKAGES})


class DenseLandmarker:
    """Process-local CPU IMAGE task: no tracking state or order-dependent smoothing."""
    def __init__(self, weights, config):
        import mediapipe as mp
        if mp.__version__ != LANDMARK_VERSION:
            raise RuntimeError('Use mediapipe=='+LANDMARK_VERSION+' with the existing pinned OpenCV (see setup)')
        ensure_landmarker(weights)
        self.mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(weights), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=config.max_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_presence_confidence,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False)
        self.task = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, image):
        result = self.task.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                data=np.ascontiguousarray(image)))
        h, w = image.shape[:2]
        return [np.array([(p.x*w, p.y*h) for p in face], dtype=np.float32) for face in result.face_landmarks]

    def close(self):
        if self.task is not None:
            self.task.close(); self.task = None


def _rgb(image):
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError('Masking requires the existing native 224x224 uint8 RGB frame; no resizing or warping is allowed')


def _valid_landmarks(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite dense face landmarks in original-frame pixel coordinates')
    return points


def select_landmarks(faces, bbox):
    """Match dense landmarks to the unchanged YuNet primary face, never to a label."""
    if bbox is None:
        return None, 'no_face'
    x, y, w, h = bbox
    choices = []
    for face in faces:
        try:
            points = _valid_landmarks(face)
        except ValueError:
            continue
        lower, upper = points[list(FACE_OVAL)].min(0), points[list(FACE_OVAL)].max(0)
        intersection = np.maximum(0, np.minimum(upper, [x+w, y+h])-np.maximum(lower, [x, y])).prod()
        area = np.maximum(0, upper-lower).prod()
        union = area+w*h-intersection
        iou = float(intersection/union) if union > 0 else 0.
        distance = float(np.square((lower+upper)/2-[x+w/2, y+h/2]).sum())
        choices.append(((-iou, distance, tuple(lower), tuple(upper), tuple(points.ravel())), points))
    if not choices:
        return None, 'no_landmarks' if not faces else 'invalid_landmarks'
    score, points = min(choices, key=lambda pair: pair[0])
    return (points, None) if -score[0] >= .1 else (None, 'unmatched_landmarks')


def face_alpha(points, config=MaskConfig()):
    """Filled full oval + small margin. Feather extends OUTWARD; interior stays 1."""
    import cv2
    config.validate(); points = _valid_landmarks(points)
    # Include all surface landmarks as well as the full oval to protect eyebrows,
    # lips and nose in pose variations; never form a mouth-only/central mask.
    hull = cv2.convexHull(np.rint(points[:468]).astype(np.int32))
    span = np.ptp(points[list(FACE_OVAL)], axis=0)
    if min(span) < 8 or cv2.contourArea(hull) < 64:
        raise ValueError('Degenerate facial oval')
    margin = max(1, math.ceil(float(max(span))*config.margin_fraction))
    core = np.zeros((224, 224), np.uint8); cv2.fillConvexPoly(core, hull, 1)
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1)))
    outside_distance = cv2.distanceTransform(1-core, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    t = np.clip(outside_distance/config.feather_pixels, 0, 1)
    alpha = ((1+np.cos(np.pi*t))*.5).astype(np.float32); alpha[core == 1] = 1.
    return alpha, margin


def clipping_diagnostics(points, alpha, crop_bounds):
    """Geometric landmark coverage only; cannot certify hidden/unobserved anatomy."""
    points = _valid_landmarks(points)
    x1, y1, x2, y2 = crop_bounds
    regions, crop_regions = {}, {}
    for name, ids in REGIONS.items():
        coordinates = points[list(ids)]
        inside = (coordinates[:, 0] >= 0) & (coordinates[:, 0] < 224) & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < 224)
        pixels = np.clip(np.rint(coordinates).astype(int), 0, 223)
        regions[name] = bool(inside.all() and np.all(alpha[pixels[:, 1], pixels[:, 0]] == 1))
        crop_regions[name] = not bool(((coordinates[:, 0] >= x1) & (coordinates[:, 0] < x2) &
                                      (coordinates[:, 1] >= y1) & (coordinates[:, 1] < y2)).all())
    return {'regions_fully_preserved': regions, 'facial_regions_clipped': not all(regions.values()),
            'crop_regions_clipped': crop_regions, 'crop_facial_regions_clipped': any(crop_regions.values()),
            'face_touches_source_edge': bool(((points[:468] <= 1) | (points[:468] >= 223)).any()),
            'coverage_check': 'landmark geometry, not an anatomical guarantee or occlusion detector'}


def masked_frame(image, points, crop_metadata, config=MaskConfig(), *, failure=None):
    """RGB-only masking; exact original interior pixels, no resize/repositioning."""
    _rgb(image); config.validate()
    details = {'mode': 'face_mask', 'output_size': [224, 224], 'spatial_scale': 1.,
               'yunet_status': crop_metadata['status'], 'confidence': crop_metadata['confidence'],
               'crop_xyxy': crop_metadata['crop_xyxy'], 'landmarks_detected': points is not None,
               'facial_regions_clipped': None, 'crop_facial_regions_clipped': None}
    if points is not None:
        try:
            alpha, margin = face_alpha(points, config)
            details.update(clipping_diagnostics(points, alpha, crop_metadata['crop_xyxy']))
        except ValueError:
            failure = 'invalid_landmarks'
    else:
        failure = failure or 'no_landmarks'
    if failure:
        details.update(status='fallback_'+failure, fallback='full_frame', retained_area_fraction=1.,
                       fully_preserved_area_fraction=1., nonneutral_support_fraction=1., margin_pixels=None,
                       oval_points_px=None)
        return image.copy(), np.ones((224, 224), np.float32), details
    blend = image.astype(np.float32)*alpha[..., None]+np.array(config.neutral_rgb, np.float32)*(1-alpha[..., None])
    output = np.rint(blend).astype(np.uint8)
    details.update(status='masked', fallback=None, retained_area_fraction=float(alpha.mean()),
        fully_preserved_area_fraction=float((alpha == 1).mean()), nonneutral_support_fraction=float((alpha > 0).mean()),
        margin_pixels=margin, oval_points_px=np.asarray(points).tolist(),
        margin_or_feather_touches_edge=bool((alpha[0] > 0).any() or (alpha[-1] > 0).any() or (alpha[:, 0] > 0).any() or (alpha[:, -1] > 0).any()))
    return np.ascontiguousarray(output), alpha, details


class MaskReader(FaceReader):
    def __init__(self, root, pipeline, cache_mib=0):
        super().__init__(root, pipeline.weights, pipeline.cache_path, pipeline.identity, pipeline.config, cache_mib)
        self.landmark_weights, self.mask_config = str(pipeline.landmark_weights), pipeline.mask_config
        self.landmarker = None

    def transform(self, record, image):
        frame_key(record); _rgb(image)
        metadata = self.boxes.get(record)
        if metadata is None:
            if self.detector is None: self.detector = YuNet(self.weights, self.config)
            crop = self.detector.detect(image)
            points, reason = None, 'no_face'
            if crop['bbox_xywh'] is not None:
                if self.landmarker is None: self.landmarker = DenseLandmarker(self.landmark_weights, self.mask_config)
                points, reason = select_landmarks(self.landmarker.detect(image), crop['bbox_xywh'])
            output, alpha, metadata = masked_frame(image, points, crop, self.mask_config, failure=reason)
            self.boxes.put(record, metadata)  # fallback reason recorded per source frame, never dropped
        else:
            points = metadata['oval_points_px']
            crop = {'status': metadata['yunet_status'], 'confidence': metadata['confidence'], 'crop_xyxy': metadata['crop_xyxy']}
            reason = metadata['status'].removeprefix('fallback_') if metadata['fallback'] else None
            output, alpha, _ = masked_frame(image, points, crop, self.mask_config, failure=reason)
        return output, alpha, metadata

    def read(self, records):
        for record in records: frame_key(record)  # guard BEFORE any decode
        originals = self.original(records)
        return [[self.transform(record, image[0])[0]] for record, image in zip(records, originals)]

    def close(self):
        self.boxes.close()
        if self.landmarker is not None: self.landmarker.close(); self.landmarker = None

    def __getstate__(self):
        from collections import OrderedDict
        state = dict(self.__dict__)
        state.update(detector=None, landmarker=None, cache=OrderedDict(), bytes=0)
        return state


class MaskDataset(FaceDataset):
    def __init__(self, records, data_root, pipeline, seed=4266, epoch=0, flip_probability=0., cache_mib=0):
        super().__init__(records, data_root, pipeline, seed, epoch, flip_probability, cache_mib)
        self.reader = MaskReader(data_root, pipeline, cache_mib)


class FaceMaskPipeline(FacePipeline):
    def __init__(self, weights, landmark_weights, cache_path, index_sha256, pool_sha256, mask_config=MaskConfig()):
        import cv2
        self.weights, self.landmark_weights, self.cache_path = Path(weights), Path(landmark_weights), Path(cache_path)
        self.config, self.mask_config = CropConfig(margin=.30).validate(), mask_config.validate()
        ensure_detector(self.weights); ensure_landmarker(self.landmark_weights)
        self.identity = {'kind': 'face_mask', 'version': MASK_VERSION, 'mask': asdict(mask_config),
            'crop': asdict(self.config), 'detector_sha256': digest(self.weights), 'landmarker_sha256': LANDMARK_SHA256,
            'mediapipe': LANDMARK_VERSION, 'opencv': cv2.__version__,
            'index_sha256': index_sha256, 'pool_sha256': pool_sha256}
        # JSON-normalize tuples so checkpoint/JSON/cache identities compare identically.
        self.identity = json.loads(json.dumps(self.identity))
        cache = BoxCache(self.cache_path, self.identity); cache.connection(); cache.close()

    def dataset(self, records, data_root, config, training=False, epoch=0, cache_mib=None):
        return MaskDataset(records, data_root, self, config.seed, epoch,
            config.flip_probability if training else 0., config.cache_mib if cache_mib is None else cache_mib)

    def snapshot(self, out):
        # Same backup machinery, separate cache identity and additional small landmark model.
        super().snapshot(out)
        out = Path(out)
        cache = BoxCache(self.cache_path, self.identity)
        try:
            cached = [(split, json.loads(payload)) for split, payload in cache.connection().execute('SELECT split,metadata FROM boxes')]
        finally:
            cache.close()
        rows = [row for _, row in cached]
        def outcomes(values):
            return {'frames': len(values), 'status_counts': dict(Counter(row['status'] for row in values)),
                'fallback_count': sum(row['fallback'] is not None for row in values),
                'clipping_warning_count': sum(row.get('facial_regions_clipped') is True for row in values),
                'clipping_check_unknown': sum(row.get('facial_regions_clipped') is None for row in values)}
        report = {'mode': 'face_mask', 'input_pipeline': self.identity,
            'status_counts': dict(Counter(row['status'] for row in rows)),
            'cached_unique_frames': len(rows), 'dev_frames_excluded': 0,
            'mean_retained_area_fraction': float(np.mean([row['retained_area_fraction'] for row in rows])) if rows else None,
            'fallback_count': sum(row['fallback'] is not None for row in rows),
            'clipping_warning_count': sum(row.get('facial_regions_clipped') is True for row in rows),
            'clipping_check_unknown': sum(row.get('facial_regions_clipped') is None for row in rows),
            'by_split': {split: outcomes([row for s, row in cached if s == split]) for split in ('train', 'dev')},
            'coverage_note': 'Landmark geometry warnings; hidden anatomy is not certifiable. No DEV frames excluded.', 'test_used': False}
        (out/'face_detection_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if not (out/'face_landmarker.task').exists(): shutil.copy2(self.landmark_weights, out/'face_landmarker.task')


def make_preprocessing(mode, weights, cache_path, index_sha256, pool_sha256, *, landmark_weights=None):
    """Exactly two modes. crop_30 returns the ORIGINAL object/config/identity unchanged."""
    if mode not in PREPROCESSING_MODES:
        raise ValueError('PREPROCESSING_MODE must be crop_30 or face_mask; no tighter-crop option')
    if mode == 'crop_30':
        return FacePipeline(weights, cache_path, index_sha256, pool_sha256, CropConfig(margin=.30))
    if landmark_weights is None: raise ValueError('face_mask requires the pinned dense landmark model')
    return FaceMaskPipeline(weights, landmark_weights, cache_path, index_sha256, pool_sha256)


def preprocessing_previews(records, data_root, crop_pipeline, mask_pipeline, output):
    """Same bounded TRAIN identities in three columns, original crop pixels unchanged."""
    if not 1 <= len(records) <= 16 or any(record['split'] != 'train' for record in records):
        raise ValueError('Choose 1-16 existing TRAIN frame records for the matched preview')
    for record in records: frame_key(record)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    crop_reader = FaceReader(data_root, crop_pipeline.weights, crop_pipeline.cache_path,
                             crop_pipeline.identity, crop_pipeline.config, cache_mib=4)
    mask_reader = MaskReader(data_root, mask_pipeline, cache_mib=4)
    report = []
    try:
        originals, cropped = crop_reader.original(records), crop_reader.read(records)
        for i, (record, original, crop) in enumerate(zip(records, originals, cropped)):
            masked, alpha, metadata = mask_reader.transform(record, original[0])
            crop_metadata = crop_reader.boxes.get(record)
            x1, y1, x2, y2 = crop_metadata['crop_xyxy']
            crop_area = (x2-x1)*(y2-y1)/(224*224)
            fig, axes = plt.subplots(1, 3, figsize=(11, 4))
            try:
                for ax, image, title in zip(axes, (original[0], crop[0], masked),
                    ('Original full frame', f'crop_30: {crop_area:.1%} retained',
                     f"face_mask: {metadata['retained_area_fraction']:.1%} alpha-weighted area")):
                    ax.imshow(image); ax.set_title(title, fontsize=10); ax.axis('off')
                fig.suptitle(f"TRAIN {record['source_file']} @ {record['timestamps_exact'][0]}s\n"
                    f"YuNet: {crop_metadata['status']} | mask: {metadata['status']} | "
                    f"face regions clipped (crop / mask): {metadata['crop_facial_regions_clipped']} / {metadata['facial_regions_clipped']}", fontsize=8)
                fig.tight_layout(); fig.savefig(output/f'preview_{i:02d}.png', dpi=140)
            finally: plt.close(fig)
            report.append({'frame_key': frame_key(record), 'crop_retained_area_fraction': crop_area, **metadata})
    finally:
        crop_reader.boxes.close(); mask_reader.close()
    (output/'previews.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fields = ['frame_key', 'crop_retained_area_fraction', 'retained_area_fraction', 'fully_preserved_area_fraction',
              'nonneutral_support_fraction', 'yunet_status', 'landmarks_detected', 'status', 'fallback',
              'crop_facial_regions_clipped', 'facial_regions_clipped', 'face_touches_source_edge']
    with (output/'previews.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(report)
    summary = {'train_frames': len(report), 'status_counts': dict(Counter(row['status'] for row in report)),
        'fallbacks': sum(row['fallback'] is not None for row in report),
        'diagnostic_note': 'Landmark checks cannot certify occluded or unobserved facial anatomy. Missing checks are null/unknown.',
        'sampling_changed': False, 'test_used': False}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return report
