"""Candidate rules/index, rotating sampling, bounded decoding and spawn-safe loading."""
from collections import Counter, OrderedDict, defaultdict, deque
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import random
import sqlite3
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PROTOCOL = 'cnn_v2_pts10_clip16_stride12_one_fake'


EXPECTED = {
    ('train', 'frame'): (315058, 302240, 12818),
    ('dev', 'frame'): (41746, 40115, 1631),
    ('train', 'clip'): (26817, 23026, 3791),
    ('dev', 'clip'): (3551, 3075, 476),
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def merge_periods(periods):
    merged = []
    for a, b in sorted((Fraction(str(a)), Fraction(str(b))) for a, b in periods):
        if a < 0 or b <= a:
            raise ValueError('Invalid interval')
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def choose_indices(times, fps=10):
    if not times or fps <= 0:
        raise ValueError('Positive FPS and nonempty PTS required')
    result, cursor, target = [], 0, times[0]
    while target <= times[-1]:
        while cursor + 1 < len(times) and abs(times[cursor + 1] - target) < abs(times[cursor] - target):
            cursor += 1
        if not result or result[-1] != cursor:
            result.append(cursor)
        target += Fraction(1, fps)
    return result


def clip_grid(n, length=16, stride=12):
    if n < length or not 1 <= stride <= length:
        raise ValueError('Full consecutive clip required; no padding')
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] != n - length:
        starts.append(n - length)
    return starts


def packet_timeline(path, count):
    import av
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Exactly one video stream required')
        origin = Fraction(container.start_time, av.time_base) if container.start_time is not None else Fraction(0)
        times = sorted(Fraction(p.pts * p.time_base) - origin for p in container.demux(container.streams.video[0])
                       if p.pts is not None and p.time_base is not None)
    if len(times) != count or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f'Packet/frame PTS mismatch: {path}')
    return times


def train_dev_rows(root):
    # The dataset has one shared JSON. Only split names are used to discard
    # held-out rows; no TEST annotation, video path or statistics are processed.
    rows = [r for r in json.loads((Path(root) / 'metadata.json').read_text(encoding='utf-8'))
            if r['split'] in ('train', 'dev')]
    by_file = {r['file']: r for r in rows}
    if len(by_file) != len(rows) or Counter(r['split'] for r in rows) != {'train': 4000, 'dev': 500}:
        raise ValueError('Expected 4,000 TRAIN and 500 DEV videos')
    for r in rows:
        original = by_file.get(r['original'] or r['file'])
        if (r['file'].split('/')[0] != r['split'] or original is None or
            original['split'] != r['split'] or original['modify_video'] or original['modify_audio']):
            raise ValueError('Invalid or cross-split source family')
        if type(r['modify_video']) is not bool or type(r['modify_audio']) is not bool:
            raise ValueError('Expected boolean modality flags')
        merge_periods(r['fake_periods'])
    return sorted(rows, key=lambda r: r['file'])


def rows_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_index(data_root, output):
    root = Path(data_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new metadata output: {output}')
    rows = train_dev_rows(root)
    output.mkdir(parents=True)
    db = sqlite3.connect(output / 'candidates.sqlite')
    db.executescript('''
      CREATE TABLE info(key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE videos(id INTEGER PRIMARY KEY, split TEXT, source_file TEXT UNIQUE,
        source_family TEXT, video_label INTEGER, modify_audio INTEGER,
        periods TEXT, pts TEXT, sampled_indices TEXT, source_bytes INTEGER);
      CREATE TABLE frames(id INTEGER PRIMARY KEY, video_id INTEGER, position INTEGER,
        source_index INTEGER, timestamp_s REAL, label INTEGER,
        UNIQUE(video_id, position));
      CREATE TABLE clips(id INTEGER PRIMARY KEY, video_id INTEGER, position INTEGER,
        start_s REAL, end_s REAL, fake_count INTEGER, label INTEGER,
        onset INTEGER, offset INTEGER, UNIQUE(video_id, position));
    ''')
    try:
        with db:
            db.execute('INSERT INTO info VALUES (?,?)', ('protocol', PROTOCOL))
            db.execute('INSERT INTO info VALUES (?,?)', ('train_dev_metadata_sha256', rows_hash(rows)))
            db.execute('INSERT INTO info VALUES (?,?)', ('policy', json.dumps({
                'fps': 10, 'clip_length': 16, 'clip_stride': 12, 'min_fake_frames': 1,
                'labels': 'modify_video and merged [start,end)', 'boundary_exclusion': False,
                'phase': 'first PTS; nearest; earlier-source tie; stop at last PTS'})))
            for video_id, row in enumerate(rows, 1):
                path = (root / row['file']).resolve()
                if root not in path.parents:
                    raise ValueError('Source path escaped data root')
                times = packet_timeline(path, row['video_frames'])
                indices = choose_indices(times)
                periods = merge_periods(row['fake_periods']) if row['modify_video'] else []
                labels = [int(row['modify_video'] and any(a <= times[i] < b for a, b in periods)) for i in indices]
                db.execute('INSERT INTO videos VALUES (?,?,?,?,?,?,?,?,?,?)', (
                    video_id, row['split'], row['file'], row['original'] or row['file'], int(row['modify_video']),
                    int(row['modify_audio']), json.dumps([[str(a), str(b)] for a, b in periods]),
                    json.dumps(list(map(str, times))), json.dumps(indices), path.stat().st_size))
                db.executemany('INSERT INTO frames(video_id,position,source_index,timestamp_s,label) VALUES (?,?,?,?,?)',
                               [(video_id, p, i, float(times[i]), labels[p]) for p, i in enumerate(indices)])
                clips = []
                for p in clip_grid(len(indices)):
                    ys = labels[p:p+16]
                    clips.append((video_id, p, float(times[indices[p]]), float(times[indices[p+15]]),
                                  sum(ys), int(any(ys)), int(any(a == 0 and b == 1 for a, b in zip(ys, ys[1:]))),
                                  int(any(a == 1 and b == 0 for a, b in zip(ys, ys[1:])))))
                db.executemany('INSERT INTO clips(video_id,position,start_s,end_s,fake_count,label,onset,offset) VALUES (?,?,?,?,?,?,?,?)', clips)
                if video_id % 500 == 0:
                    print(f'Indexed {video_id}/{len(rows)} TRAIN/DEV packet timelines', flush=True)
            db.executescript('CREATE INDEX frame_video ON frames(video_id); CREATE INDEX clip_video ON clips(video_id);')
            report = validate_index(db)
            db.execute('INSERT INTO info VALUES (?,?)', ('complete', 'true'))
        write_json(output / 'index_report.json', report)
    finally:
        db.close()
    print(json.dumps(report, indent=2), flush=True)
    return output / 'candidates.sqlite'


def validate_index(db):
    report = {'protocol': PROTOCOL, 'counts': {}, 'test_used': False}
    if set(r[0] for r in db.execute('SELECT DISTINCT split FROM videos')) != {'train', 'dev'}:
        raise ValueError('Only TRAIN/DEV allowed')
    if db.execute('SELECT COUNT(*) FROM videos WHERE source_file NOT LIKE split || \'/%\' OR source_family NOT LIKE split || \'/%\'').fetchone()[0]:
        raise ValueError('Split leakage')
    for (split, kind), expected in EXPECTED.items():
        table = 'frames' if kind == 'frame' else 'clips'
        n, fake = db.execute(f'SELECT COUNT(*),SUM(c.label) FROM {table} c JOIN videos v ON v.id=c.video_id WHERE v.split=?', (split,)).fetchone()
        if (n, n-fake, fake) != expected:
            raise ValueError(f'{split}/{kind}: {(n,n-fake,fake)} != audited {expected}')
        report['counts'][f'{split}_{kind}'] = {'total': n, 'real': n-fake, 'fake': fake}
    return report


class Index:
    def __init__(self, path, data_root=None):
        self.path = Path(path).resolve()
        self.db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.info = dict(self.db.execute('SELECT key,value FROM info'))
        if self.info.get('protocol') != PROTOCOL or self.info.get('complete') != 'true':
            raise ValueError('Incomplete or incompatible candidate index')
        self.report = validate_index(self.db)
        if data_root is not None and rows_hash(train_dev_rows(data_root)) != self.info['train_dev_metadata_sha256']:
            raise ValueError('Reattached TRAIN/DEV metadata differs')
        self.videos = {}
        for row in self.db.execute('SELECT * FROM videos'):
            v = dict(row)
            v['pts'] = list(map(Fraction, json.loads(v['pts'])))
            v['sampled_indices'] = json.loads(v['sampled_indices'])
            self.videos[v['id']] = v

    def candidates(self, split, kind):
        if split not in ('train', 'dev') or kind not in ('frame', 'clip'):
            raise ValueError('Only TRAIN/DEV frame/clip candidates allowed')
        table = 'frames' if kind == 'frame' else 'clips'
        return [dict(r) for r in self.db.execute(f'SELECT c.* FROM {table} c JOIN videos v ON v.id=c.video_id WHERE v.split=? ORDER BY c.id', (split,))]

    def records(self, split, kind, ids=None):
        table = 'frames' if kind == 'frame' else 'clips'
        if split not in ('train', 'dev') or kind not in ('frame', 'clip'):
            raise ValueError('Only TRAIN/DEV allowed')
        if ids is None:
            selected = self.candidates(split, kind)
        else:
            selected = []
            for start in range(0, len(ids), 500):
                chunk = ids[start:start+500]
                selected.extend(dict(r) for r in self.db.execute(f'SELECT * FROM {table} WHERE id IN ({",".join("?" for _ in chunk)})', chunk))
            by_id = {r['id']: r for r in selected}
            selected = [by_id[i] for i in ids]
        result = []
        for row in selected:
            v = self.videos[row['video_id']]
            if v['split'] != split:
                raise ValueError('Selected candidate belongs to another split')
            p = row['position']; length = 1 if kind == 'frame' else 16
            indices = v['sampled_indices'][p:p+length]
            result.append(row | {'split': split, 'kind': kind, 'source_file': v['source_file'],
                'source_family': v['source_family'], 'video_label': v['video_label'],
                'indices': indices, 'timestamps_exact': list(map(str, (v['pts'][i] for i in indices))),
                'metric_eligible': True})
        return result

    def close(self):
        self.db.close()


def stable_seed(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


class EpochSampler:
    def __init__(self, index, kind='frame', per_class=4000, seed=4266):
        self.kind, self.per_class, self.seed = kind, per_class, seed
        self.groups = {0: defaultdict(lambda: defaultdict(list)), 1: defaultdict(lambda: defaultdict(list))}
        self.rows = {}
        self.videos = index.videos
        for row in index.candidates('train', kind):
            cid, vid, label = row['id'], row['video_id'], row['label']
            self.groups[label][self.videos[vid]['source_family']][vid].append(cid)
            self.rows[cid] = (vid, label, row.get('fake_count', label), row.get('onset', 0), row.get('offset', 0))
        self.totals = {label: sum(len(ids) for vs in self.groups[label].values() for ids in vs.values()) for label in (0, 1)}
        if min(self.totals.values()) < per_class or per_class < 1:
            raise ValueError('Insufficient distinct candidates for balanced epoch')
        self.last_used, self.uses, self.previous = {}, Counter(), set()
        self.epoch = -1

    def state_dict(self):
        return {'kind': self.kind, 'per_class': self.per_class, 'seed': self.seed, 'epoch': self.epoch,
                'last_used': dict(self.last_used), 'uses': dict(self.uses), 'previous': sorted(self.previous)}

    def load_state_dict(self, state):
        if (state['kind'], state['per_class'], state['seed']) != (self.kind, self.per_class, self.seed):
            raise ValueError('Sampler resume policy mismatch')
        self.epoch = state['epoch']
        self.last_used = {int(k): v for k, v in state['last_used'].items()}
        self.uses = Counter({int(k): v for k, v in state['uses'].items()})
        self.previous = set(state['previous'])

    def select(self, epoch):
        # A standalone request for epoch N replays only small metadata selections.
        # This makes seed + epoch sufficient even without a saved sampler state.
        if epoch < 0 or epoch <= self.epoch:
            raise ValueError('Request successive epochs or create a fresh sampler to replay')
        if epoch > self.epoch + 1:
            for earlier in range(self.epoch + 1, epoch):
                self.select(earlier)
        rng = random.Random(stable_seed(self.seed, self.kind, epoch))
        chosen = {0: [], 1: []}
        for label in (0, 1):
            queues = {}
            for fam, videos in self.groups[label].items():
                queues[fam] = {}
                for video, ids in videos.items():
                    tie = {i: rng.random() for i in ids}
                    queues[fam][video] = deque(sorted(ids, key=lambda i: (self.last_used.get(i, -1), tie[i])))
            counts = Counter()
            families = sorted(queues)
            while len(chosen[label]) < self.per_class:
                rng.shuffle(families)
                made_progress = False
                for fam in families:
                    available = [v for v, q in queues[fam].items() if q]
                    if not available:
                        continue
                    # Family sweeps ensure broad family coverage. Within each
                    # family, spread requests among videos before reusing one.
                    rng.shuffle(available)
                    video = min(available, key=lambda v: (counts[v], self.last_used.get(queues[fam][v][0], -1)))
                    chosen[label].append(queues[fam][video].popleft())
                    counts[video] += 1
                    made_progress = True
                    if len(chosen[label]) == self.per_class:
                        break
                if not made_progress:
                    raise ValueError('Candidate pools exhausted')
        # Interleave classes: every even-size minibatch is balanced, including
        # the final smaller batch. Weights continue normally across epochs.
        rng.shuffle(chosen[0]); rng.shuffle(chosen[1])
        fake_by_video = defaultdict(deque)
        for fake in chosen[1]:
            fake_by_video[self.rows[fake][0]].append(fake)
        pairs, unmatched_real = [], []
        for real in chosen[0]:
            matches = fake_by_video[self.rows[real][0]]
            if matches:
                pairs.append([real, matches.popleft()])
            else:
                unmatched_real.append(real)
        unmatched_fake = [i for queue in fake_by_video.values() for i in queue]
        rng.shuffle(unmatched_fake)
        pairs.extend([real, fake] for real, fake in zip(unmatched_real, unmatched_fake))
        rng.shuffle(pairs)
        ids = []
        for pair in pairs:
            rng.shuffle(pair); ids.extend(pair)
        if len(ids) != len(set(ids)):
            raise AssertionError('Exact duplicate within epoch')
        reused = sum(i in self.uses for i in ids)
        previous_overlap = len(set(ids) & self.previous)
        for i in ids:
            self.last_used[i] = epoch; self.uses[i] += 1
        self.epoch, self.previous = epoch, set(ids)
        vc = Counter(self.rows[i][0] for i in ids)
        fc = Counter(self.videos[v]['source_family'] for v in (self.rows[i][0] for i in ids))
        diagnostic = {'epoch': epoch + 1, 'kind': self.kind, 'samples': len(ids),
            'real': len(chosen[0]), 'fake': len(chosen[1]), 'unique_samples': len(set(ids)),
            'duplicate_rate': 0.0, 'unique_videos': len(vc), 'unique_families': len(fc),
            'max_samples_one_video': max(vc.values()), 'max_samples_one_family': max(fc.values()),
            'previous_epoch_overlap': previous_overlap, 'previous_epoch_overlap_rate': previous_overlap / len(ids),
            'previously_seen_samples': reused, 'reuse_rate': reused / len(ids),
            'cumulative_unique': len(self.uses), 'cumulative_coverage': len(self.uses) / len(self.rows),
            'class_cumulative_coverage': {str(label): sum(self.rows[i][1] == label for i in self.uses) / self.totals[label] for label in (0, 1)}}
        if self.kind == 'clip':
            diagnostic.update(fake_frame_counts=dict(sorted(Counter(self.rows[i][2] for i in ids).items())),
                              onset_clips=sum(self.rows[i][3] for i in ids),
                              offset_clips=sum(self.rows[i][4] for i in ids),
                              both_transition_clips=sum(self.rows[i][3] and self.rows[i][4] for i in ids))
        return ids, diagnostic


class FrameReader:
    def __init__(self, data_root, cache_mib=0):
        self.root = Path(data_root).resolve()
        self.limit = int(cache_mib * 1024 * 1024)
        self.cache, self.bytes = OrderedDict(), 0
        self.opens = self.hits = self.decoded = 0

    def _put(self, key, value):
        if not self.limit or value.nbytes > self.limit:
            return
        while self.cache and self.bytes + value.nbytes > self.limit:
            _, old = self.cache.popitem(last=False); self.bytes -= old.nbytes
        self.cache[key] = value; self.bytes += value.nbytes

    def read(self, records):
        import av
        requests = defaultdict(dict)
        found = {}
        for row in records:
            if row['split'] not in ('train', 'dev') or row['source_file'].split('/')[0] != row['split']:
                raise ValueError('Only TRAIN/DEV sources may be decoded')
            for i, stamp in zip(row['indices'], row['timestamps_exact']):
                key = (row['source_file'], i)
                if key in self.cache:
                    found[key] = self.cache[key]; self.cache.move_to_end(key); self.hits += 1
                else:
                    expected = Fraction(stamp)
                    if i in requests[row['source_file']] and requests[row['source_file']][i] != expected:
                        raise ValueError('Conflicting timestamp reference')
                    requests[row['source_file']][i] = expected
        for source, needed in requests.items():
            if not needed:
                continue
            path = (self.root / source).resolve()
            if self.root not in path.parents:
                raise ValueError('Source path escaped data root')
            self.opens += 1
            try:
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    origin = Fraction(container.start_time, av.time_base) if container.start_time is not None else Fraction(0)
                    for i, frame in enumerate(container.decode(stream)):
                        self.decoded += 1
                        if i in needed:
                            if frame.is_corrupt or frame.pts is None or Fraction(frame.pts * frame.time_base) - origin != needed[i]:
                                raise ValueError(f'Decoded PTS mismatch: {source}/{i}')
                            if (frame.width, frame.height) != (224, 224):
                                raise ValueError('Expected native 224x224 source; no silent resize')
                            value = frame.to_ndarray(format='rgb24')
                            found[(source, i)] = value
                            self._put((source, i), value)
                        if i >= max(needed):
                            break
                if any((source, i) not in found for i in needed):
                    raise ValueError('Missing requested source frame')
            except Exception as error:
                raise RuntimeError(f'Video decode failed: {source}; requested source indices {sorted(needed)}') from error
        return [[found[(r['source_file'], i)] for i in r['indices']] for r in records]


def memory_mib():
    """Current process RSS, stdlib on Windows/Linux (not GPU memory)."""
    import os
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info.restype = wintypes.BOOL
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not info(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.WorkingSetSize / 2**20
    path = '/proc/self/statm'
    try:
        return int(Path(path).read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**20
    except (OSError, ValueError):
        return None


def benchmark_decode(records, data_root, cache_options=(0, 64), batch_size=32):
    """CPU-only pixel loading, not a model/dataloader GPU benchmark."""
    reports = []
    for budget in cache_options:
        reader = FrameReader(data_root, budget)
        for pass_name in ('cold_selected_pass', 'repeat_same_subset_upper_bound'):
            before = (reader.opens, reader.hits, reader.decoded)
            start = time.perf_counter(); peak = memory_mib()
            for p in range(0, len(records), batch_size):
                images = reader.read(records[p:p+batch_size])
                del images
                rss = memory_mib()
                if rss is not None:
                    peak = max(peak or 0, rss)
            seconds = time.perf_counter() - start
            frames = sum(len(r['indices']) for r in records)
            reports.append({'cache_mib_per_process': budget, 'pass': pass_name, 'samples': len(records),
                            'requested_frames': frames, 'seconds': seconds, 'frames_per_second': frames / seconds,
                            'process_rss_mib_sampled_peak': peak, 'video_opens': reader.opens-before[0],
                            'cache_hits': reader.hits-before[1], 'source_frames_decoded': reader.decoded-before[2],
                            'resident_cache_mib': reader.bytes / 2**20})
    return reports


class LazyDataset(Dataset):
    def __init__(self, records, data_root, seed=4266, epoch=0, flip_probability=0., cache_mib=0):
        if not records or len({r['split'] for r in records}) != 1:
            raise ValueError('Expected one nonempty split')
        self.records = records
        self.seed, self.epoch = seed, epoch
        self.flip_probability = flip_probability if records[0]['split'] == 'train' else 0.
        if records[0]['split'] != 'train' and flip_probability:
            raise ValueError('DEV augmentation is prohibited')
        self.reader = FrameReader(data_root, cache_mib)

    def __len__(self):
        return len(self.records)

    def __getitems__(self, indices):
        records = [self.records[i] for i in indices]
        try:
            decoded = self.reader.read(records)
        except Exception as error:
            raise RuntimeError(f"Batch decode failed; candidate IDs {[r['id'] for r in records]}; "
                               f"sources {sorted({r['source_file'] for r in records})}") from error
        result = []
        for idx, row, images in zip(indices, records, decoded):
            pixels = np.stack(images)
            flip = stable_seed(self.seed, self.epoch, row['kind'], row['id'], 'flip') / 2**64 < self.flip_probability
            if flip:
                pixels = pixels[:, :, ::-1, :].copy()  # same flip for all 16 frames
            tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().div_(255.)
            tensor.sub_(torch.tensor((.485, .456, .406)).view(1, 3, 1, 1))
            tensor.div_(torch.tensor((.229, .224, .225)).view(1, 3, 1, 1))
            result.append((tensor[0] if row['kind'] == 'frame' else tensor,
                           torch.tensor(float(row['label'])), idx))
        return result

    def __getitem__(self, index):
        return self.__getitems__([index])[0]


def make_loader(records, data_root, config, training=False, epoch=0, cache_mib=None):
    dataset = LazyDataset(records, data_root, config.seed, epoch,
                          config.flip_probability if training else 0.,
                          config.cache_mib if cache_mib is None else cache_mib)
    kwargs = {'num_workers': config.workers, 'pin_memory': True,
              'generator': torch.Generator().manual_seed(stable_seed(config.seed, epoch, training) % 2**63)}
    if config.workers:
        # Importable dataset/decoder classes are safe to pickle in notebook workers.
        # Never fork a parent that already initialized CUDA / decoder threads.
        kwargs.update(prefetch_factor=config.prefetch_factor, persistent_workers=False,
                      multiprocessing_context='spawn', timeout=config.loader_timeout_seconds)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
