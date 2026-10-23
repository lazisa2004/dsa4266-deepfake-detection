"""Deterministic CPU YuNet, resumable bounding-box metadata, and lazy RGB crops."""
from dataclasses import asdict, dataclass
from contextlib import closing
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_experiment.checkpoints import digest
from cnn_experiment.data import FrameReader, LazyDataset, stable_seed

DETECTOR_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
DETECTOR_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
                'models/face_detection_yunet/face_detection_yunet_2023mar.onnx')
CROP_VERSION = 'yunet2023mar_primary_area_square_margin_total_v2'


@dataclass(frozen=True)
class CropConfig:
    margin: float = .30  # TOTAL expansion: 30% larger side, i.e. 15% padding per side
    score_threshold: float = .8
    nms_threshold: float = .3
    top_k: int = 5000
    fallback: str = 'full_frame'
    output_size: int = 224
    interpolation: str = 'linear'

    def validate(self):
        if not .25 <= self.margin <= .35:
            raise ValueError('Face margin must be 25%-35% total expansion (12.5%-17.5% per side)')
        if not 0 < self.score_threshold < 1 or not 0 < self.nms_threshold < 1 or self.top_k < 1:
            raise ValueError('Invalid detector thresholds')
        if (self.fallback, self.output_size, self.interpolation) != ('full_frame', 224, 'linear'):
            raise ValueError('This protocol fixes full-frame fallback, 224 RGB and bilinear resize')
        return self


def ensure_detector(path, *, download=False):
    """Explicit small public-weight download, pinned by SHA; never a training download."""
    path = Path(path)
    if path.is_file():
        if digest(path) != DETECTOR_SHA256:
            raise ValueError('YuNet checkpoint SHA256 differs; do not use an unreviewed detector')
        return path
    if not download:
        raise FileNotFoundError('Attach face_detection_yunet_2023mar.onnx or explicitly run ensure_detector(download=True) in setup')
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+'.download')
    with urllib.request.urlopen(DETECTOR_URL, timeout=90) as source, partial.open('xb') as target:
        import shutil
        shutil.copyfileobj(source, target)
    if digest(partial) != DETECTOR_SHA256:
        raise ValueError('Downloaded detector failed SHA256; partial file preserved for diagnosis')
    partial.rename(path)
    return path


def crop_rectangle(box, width, height, margin=.3):
    """Square, shift inside image (no black padding); right/bottom exclusive."""
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
        raise ValueError('Invalid face bounding box')
    side = min(min(width, height), max(1, math.ceil(max(w, h)*(1+margin))))
    left = min(max(0, math.floor(x+w/2-side/2)), width-side)
    top = min(max(0, math.floor(y+h/2-side/2)), height-side)
    return [int(left), int(top), int(left+side), int(top+side)]


def choose_primary(faces, width, height, config):
    """Largest valid face, then confidence, centre distance, coordinates. No labels."""
    candidates = []
    if faces is not None:
        for f in faces:
            f = np.asarray(f).tolist()
            if len(f) < 15 or not all(math.isfinite(float(v)) for v in f):
                continue
            x, y, w, h, score = *map(float, f[:4]), float(f[14])
            if score < config.score_threshold or min(w, h) <= 0 or x+w <= 0 or y+h <= 0 or x >= width or y >= height:
                continue
            area = max(0, min(width, x+w)-max(0, x))*max(0, min(height, y+h)-max(0, y))
            distance = (x+w/2-width/2)**2+(y+h/2-height/2)**2
            candidates.append(((-area, -score, distance, x, y, w, h), f))
    if not candidates:
        return {'status': 'fallback_no_face', 'confidence': None, 'bbox_xywh': None,
                'crop_xyxy': [0, 0, width, height], 'faces_detected': 0}
    _, face = min(candidates, key=lambda pair: pair[0])
    return {'status': 'detected', 'confidence': float(face[14]), 'bbox_xywh': list(map(float, face[:4])),
            'crop_xyxy': crop_rectangle(face[:4], width, height, config.margin), 'faces_detected': len(candidates)}


class YuNet:
    def __init__(self, weights, config):
        import cv2
        if cv2.__version__ != '4.13.0':
            raise RuntimeError('Use the pinned opencv-contrib-python==4.13.0.92 for reproducible face detection')
        ensure_detector(weights)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        self.config = config.validate()
        self.detector = cv2.FaceDetectorYN.create(str(weights), '', (224, 224), config.score_threshold,
            config.nms_threshold, config.top_k, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)

    def detect(self, rgb):
        import cv2
        h, w = rgb.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return choose_primary(faces, w, h, self.config)


def frame_key(record):
    if record['split'] not in ('train', 'dev') or record['kind'] != 'frame' or len(record['indices']) != 1:
        raise ValueError('Face experiments expose existing TRAIN/DEV CNN frames only')
    if not record['source_file'].startswith(record['split']+'/') or '..' in Path(record['source_file']).parts:
        raise ValueError('Invalid source identity')
    return json.dumps([record['source_file'], int(record['indices'][0]), str(Fraction(record['timestamps_exact'][0]))], separators=(',', ':'))


class BoxCache:
    """Process-local SQLite connections, transactional per-frame writes, WAL for spawn."""
    def __init__(self, path, policy):
        self.path, self.policy = Path(path), policy
        self.db = None
        self.pid = None

    def connection(self):
        if self.db is None or self.pid != os.getpid():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path, timeout=90)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS boxes (frame_key TEXT PRIMARY KEY, split TEXT, source_file TEXT, source_index INTEGER, timestamp_exact TEXT, metadata TEXT NOT NULL)')
            value = json.dumps(self.policy, sort_keys=True)
            self.db.execute('INSERT OR IGNORE INTO info VALUES (?,?)', ('policy', value)); self.db.commit()
            if self.db.execute("SELECT value FROM info WHERE key='policy'").fetchone()[0] != value:
                self.close()
                raise ValueError('Face cache policy/detector/dataset changed; use a separate cache')
            self.pid = os.getpid()
        return self.db

    def get(self, record):
        row = self.connection().execute('SELECT metadata FROM boxes WHERE frame_key=?', (frame_key(record),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, record, metadata):
        db = self.connection()
        db.execute('INSERT OR IGNORE INTO boxes VALUES (?,?,?,?,?,?)', (frame_key(record), record['split'],
            record['source_file'], record['indices'][0], str(Fraction(record['timestamps_exact'][0])), json.dumps(metadata, sort_keys=True)))
        db.commit()

    def snapshot(self, output):
        """SQLite backup API includes committed WAL rows; copying the live DB is unsafe."""
        output = Path(output); temporary = output.with_suffix('.snapshot.tmp')
        if output.resolve() == self.path.resolve():
            raise ValueError('Snapshot destination must differ from the live face-cache database')
        with closing(sqlite3.connect(temporary)) as target:
            self.connection().backup(target)
        temporary.replace(output)

    def summary(self):
        from collections import Counter
        counts, confidences, areas = Counter(), [], []
        for split, payload in self.connection().execute('SELECT split,metadata FROM boxes'):
            metadata = json.loads(payload); counts[(split, metadata['status'])] += 1
            x1, y1, x2, y2 = metadata['crop_xyxy']; areas.append((x2-x1)*(y2-y1)/(224*224))
            if metadata['confidence'] is not None:
                confidences.append(metadata['confidence'])
        return {'counts': {f'{s}_{status}': n for (s, status), n in sorted(counts.items())},
                'cached_unique_frames': sum(counts.values()), 'confidence_mean_detected': sum(confidences)/len(confidences) if confidences else None,
                'mean_retained_area_fraction': sum(areas)/len(areas) if areas else None,
                'full_frame_inputs_including_fallback': sum(a == 1 for a in areas),
                'dev_frames_excluded': 0, 'test_used': False}

    def close(self):
        if self.db is not None:
            self.db.close(); self.db = None; self.pid = None

    def __getstate__(self):
        return {'path': self.path, 'policy': self.policy, 'db': None, 'pid': None}

    def __del__(self):
        if getattr(self, 'db', None) is not None:
            self.close()


class FaceReader(FrameReader):
    def __init__(self, root, weights, cache_path, policy, config, cache_mib=0):
        super().__init__(root, cache_mib)
        self.weights, self.config = str(weights), config
        self.boxes = BoxCache(cache_path, policy)
        self.detector = None

    def metadata(self, record, image):
        value = self.boxes.get(record)
        if value is None:
            if self.detector is None:
                self.detector = YuNet(self.weights, self.config)
            value = self.detector.detect(image)  # runtime/model errors raise; no silent fallback for broken setup
            self.boxes.put(record, value)
        return value

    def original(self, records):
        return super().read(records)

    def read(self, records):
        # Validate split before decode. Never pass labels to the detector.
        for record in records:
            frame_key(record)
        decoded = self.original(records)
        import cv2
        result = []
        for record, images in zip(records, decoded):
            image = images[0]
            metadata = self.metadata(record, image)
            x1, y1, x2, y2 = metadata['crop_xyxy']
            if not 0 <= x1 < x2 <= image.shape[1] or not 0 <= y1 < y2 <= image.shape[0]:
                raise ValueError('Cached face crop escaped frame bounds')
            crop = image[y1:y2, x1:x2]
            if crop.shape[:2] != (224, 224):
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            result.append([np.ascontiguousarray(crop)])
        return result


class FaceDataset(LazyDataset):
    def __init__(self, records, data_root, pipeline, seed=4266, epoch=0, flip_probability=0., cache_mib=0):
        super().__init__(records, data_root, seed, epoch, flip_probability, cache_mib)
        self.reader = FaceReader(data_root, pipeline.weights, pipeline.cache_path,
                                 pipeline.identity, pipeline.config, cache_mib)


class FacePipeline:
    def __init__(self, weights, cache_path, index_sha256, pool_sha256, config=CropConfig()):
        import cv2
        self.weights, self.cache_path = Path(weights), Path(cache_path)
        self.config = config.validate()
        ensure_detector(self.weights)
        self.identity = {'kind': 'face_crop', 'version': CROP_VERSION, 'crop': asdict(config),
                         'detector_sha256': DETECTOR_SHA256, 'opencv': cv2.__version__,
                         'index_sha256': index_sha256, 'pool_sha256': pool_sha256}
        cache = BoxCache(self.cache_path, self.identity); cache.connection(); cache.close()

    def dataset(self, records, data_root, config, training=False, epoch=0, cache_mib=None):
        return FaceDataset(records, data_root, self, config.seed, epoch,
            config.flip_probability if training else 0., config.cache_mib if cache_mib is None else cache_mib)

    def make_loader(self, records, data_root, config, training=False, epoch=0):
        dataset = self.dataset(records, data_root, config, training, epoch)
        kwargs = {'num_workers': config.workers, 'pin_memory': True,
                  'generator': torch.Generator().manual_seed(stable_seed(config.seed, epoch, training) % 2**63)}
        if config.workers:
            kwargs.update(prefetch_factor=config.prefetch_factor, persistent_workers=False,
                          multiprocessing_context='spawn', timeout=config.loader_timeout_seconds)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    def snapshot(self, out):
        import shutil
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        cache = BoxCache(self.cache_path, self.identity)
        try:
            cache.snapshot(out/'face_boxes.sqlite')
            (out/'face_detection_report.json').write_text(json.dumps(cache.summary(), indent=2), encoding='utf-8')
        finally:
            cache.close()
        if not (out/'face_detector.onnx').is_file():
            shutil.copy2(self.weights, out/'face_detector.onnx')
        if not (out/'face_detector_LICENSE.txt').is_file():
            shutil.copy2(Path(__file__).with_name('YUNET_LICENSE.txt'), out/'face_detector_LICENSE.txt')


def crop_previews(records, data_root, pipeline, output):
    """Bounded label-independent preview set, BEFORE full preprocessing/training."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    reader = FaceReader(data_root, pipeline.weights, pipeline.cache_path, pipeline.identity, pipeline.config, cache_mib=4)
    originals = reader.original(records)
    crops = reader.read(records)
    report = []
    for i, (record, original, cropped) in enumerate(zip(records, originals, crops)):
        metadata = reader.boxes.get(record)
        fig, axes = plt.subplots(1, 2, figsize=(7, 4))
        try:
            axes[0].imshow(original[0]); axes[1].imshow(cropped[0])
            x1, y1, x2, y2 = metadata['crop_xyxy']
            axes[0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime'))
            axes[0].set_title('Original RGB + crop bounds'); axes[1].set_title('CNN input: face / explicit fallback')
            for ax in axes: ax.axis('off')
            fig.suptitle(f"{record['source_file']} @ {float(Fraction(record['timestamps_exact'][0])):.3f}s\n{metadata['status']}; confidence={metadata['confidence']}", fontsize=9)
            fig.tight_layout(); fig.savefig(output/f'preview_{i:02d}.png', dpi=130)
        finally:
            plt.close(fig)
        report.append({'frame_key': frame_key(record), **metadata})
    reader.boxes.close()
    (output/'previews.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
