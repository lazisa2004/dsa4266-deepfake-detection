"""Notebook input validation, bounded diagnostics, fresh sessions and output preservation."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile
import torch
from torch import nn
from cnn_experiment.data import Index, write_json, EpochSampler, make_loader, memory_mib
from cnn_experiment.models import StagedCNN, make_optimizer
from cnn_experiment.training import RunProgress, GradientGuard, require_cuda, seed_everything


def input_files(name, base='/kaggle/input'):
    """Bounded mount discovery; never enumerate TRAIN/DEV/TEST video folders."""
    base = Path(base)
    result = []
    if not base.is_dir():
        return result
    for root, dirs, files in os.walk(base):
        relative = Path(root).relative_to(base)
        dirs[:] = [d for d in dirs if d.lower() not in ('train', 'dev', 'test') and not d.startswith('.')]
        if len(relative.parts) >= 6 or 'metadata.json' in files:
            dirs[:] = []
        if name in files:
            result.append(Path(root) / name)
    return sorted(result)


def unique_path(candidates, description):
    if len(candidates) != 1:
        raise ValueError(f'Set an explicit {description} path. Discovered: {list(map(str, candidates))}')
    return candidates[0]


def discover_roots(data_override=None, weights_override=None):
    data = Path(data_override) if data_override else unique_path(
        [p.parent for p in input_files('metadata.json') if (p.parent/'train').is_dir() and (p.parent/'dev').is_dir()], 'DATA')
    weights = Path(weights_override) if weights_override else unique_path(
        [p.parent for p in input_files('resnet50-11ad3fa6.pth')
         if (p.parent/'efficientnet_b0_rwightman-7f5810bc.pth').is_file()], 'WEIGHTS')
    if not (data/'metadata.json').is_file() or not (data/'train').is_dir() or not (data/'dev').is_dir():
        raise ValueError('DATA must contain metadata.json and TRAIN/DEV videos')
    for name in ('resnet50-11ad3fa6.pth', 'efficientnet_b0_rwightman-7f5810bc.pth'):
        if not (weights/name).is_file():
            raise FileNotFoundError(weights/name)
    print('Mounted DATA:', data, '\nMounted WEIGHTS:', weights)
    return data, weights


@contextmanager
def diagnostic_stage(folder, name):
    """Save a preflight failure without suppressing the original exception."""
    import traceback
    folder = Path(folder)
    try:
        yield
    except BaseException:
        path = folder/(name+'_failure_'+uuid.uuid4().hex[:8]+'.txt')
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path.write_text(traceback.format_exc(), encoding='utf-8')
            print('Diagnostic failure log:', path, flush=True)
        except OSError as error:
            print('Could not save diagnostic failure log:', type(error).__name__, flush=True)
        raise


def discover_attached_inputs(data=None, weights=None, index=None, input_base='/kaggle/input'):
    """Require existing datasets, using bounded marker discovery only."""
    if data is None:
        data = unique_path([p.parent for p in input_files('metadata.json', input_base)
                            if (p.parent/'train').is_dir() and (p.parent/'dev').is_dir()], 'DATA_OVERRIDE')
    if weights is None:
        weights = unique_path([p.parent for p in input_files('resnet50-11ad3fa6.pth', input_base)
                               if (p.parent/'efficientnet_b0_rwightman-7f5810bc.pth').is_file()], 'WEIGHTS_OVERRIDE')
    if index is None:
        index = unique_path(input_files('candidates.sqlite', input_base), 'INDEX_OVERRIDE')
    index = Path(index)
    if index.name != 'candidates.sqlite' or not index.is_file():
        raise FileNotFoundError('Attach the existing candidates.sqlite dataset; this workflow never rebuilds it.')
    data, weights = discover_roots(data, weights)
    return {'data': data, 'weights': weights, 'index': index}


def validate_candidates(index_path, data_root):
    """Verify the audited 10-FPS index and source metadata without demux/decode."""
    from cnn_experiment.data import Index
    index = Index(index_path, data_root)
    try:
        policy = json.loads(index.info['policy'])
        required = {'fps': 10, 'clip_length': 16, 'clip_stride': 12,
                    'min_fake_frames': 1, 'boundary_exclusion': False}
        if any(policy.get(key) != value for key, value in required.items()):
            raise ValueError('The attached index preprocessing policy differs from the audited CNN v2 policy.')
        counts = Counter(v['split'] for v in index.videos.values())
        if counts != {'train': 4000, 'dev': 500}:
            raise ValueError('Expected 4,000 TRAIN videos and 500 DEV videos.')
        return index.report | {'videos': dict(counts), 'policy': policy,
                              'families': {split: len({v['source_family'] for v in index.videos.values()
                                                      if v['split'] == split}) for split in ('train', 'dev')}}
    finally:
        index.close()


def verify_epoch_sampling(index_path, data_root, config):
    """Verify two balanced rotating CNN epochs; return compact diagnostics only."""
    from cnn_experiment.data import Index
    from cnn_experiment.data import EpochSampler
    index = Index(index_path, data_root)
    try:
        if config.train_per_class != 4000:
            raise ValueError('The notebook requires 4,000 examples per class per CNN epoch.')
        sampler = EpochSampler(index, per_class=config.train_per_class, seed=config.seed)
        reports, previous = [], None
        for epoch in range(2):
            ids, report = sampler.select(epoch)
            if len(ids) != 8000 or len(set(ids)) != 8000 or sum(sampler.rows[i][1] for i in ids) != 4000:
                raise AssertionError('Epoch must have 8,000 unique samples, balanced 4,000/4,000.')
            if previous == ids:
                raise AssertionError('Epoch sampling did not rotate.')
            records = index.records('train', 'frame', ids)
            for record in records:
                video = index.videos[record['video_id']]
                timestamp = video['pts'][record['source_index']]
                periods = json.loads(video['periods'])
                expected = int(any(Fraction(a) <= timestamp < Fraction(b) for a, b in periods))
                if record['label'] != expected:
                    raise AssertionError('Selected label does not match the finalized visual interval rule.')
            reports.append(report)
            previous = ids
        # These disposable sampler instances never become the training sampler.
        return reports
    finally:
        index.close()


def create_session(base='/kaggle/working/experiment1_cnn_v2'):
    """Create a new output directory; never reuse a previous experiment."""
    name = 'session_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC') + '_' + uuid.uuid4().hex[:8]
    path = Path(base)/name
    path.mkdir(parents=True, exist_ok=False)
    return path


def output_inventory(session):
    """List outputs, not video inputs or cloned code."""
    root = Path(session).resolve()
    return [{'file': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size}
            for path in sorted(root.rglob('*')) if path.is_file() and not path.is_symlink()]


def package_session(session, destination='/kaggle/working/experiment1_cnn_v2_backups'):
    """Create a new checkpoint-inclusive ZIP, verify CRCs, record SHA256.

    Originals are never moved, removed or overwritten. Copy/download the returned
    ZIP outside Kaggle before expiry; creating a ZIP on ephemeral disk is not
    itself an external backup.
    """
    root, destination = Path(session).resolve(), Path(destination).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if destination == root or root in destination.parents:
        raise ValueError('The backup destination must be outside the session directory.')
    paths = sorted(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Session symlinks are prohibited: backups must not include external datasets or credentials.')
    files = [p for p in paths if p.is_file()]
    if not files:
        raise ValueError('There are no session output files to preserve.')
    destination.mkdir(parents=True, exist_ok=True)
    total = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(destination).free < total + 64*2**20:
        raise OSError('Insufficient disk space for a conservative full session ZIP estimate.')
    filename = root.name + '_' + uuid.uuid4().hex[:8] + '.zip'
    archive_path = destination/filename
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in files:
            archive.write(path, (Path(root.name)/path.relative_to(root)).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise OSError('Backup CRC verification failed; preserve originals and create a new backup.')
    digest = hashlib.sha256()
    with archive_path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            digest.update(chunk)
    result = {'archive': str(archive_path), 'bytes': archive_path.stat().st_size,
              'sha256': digest.hexdigest(), 'files': len(files),
              'external_backup_required': True}
    archive_path.with_suffix('.manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def benchmark_loader(index_path, data_root, config, samples=512):
    """Kaggle loader benchmark; a subset is timing evidence, not training."""
    index = Index(index_path, data_root)
    ids, _ = EpochSampler(index, per_class=config.train_per_class, seed=config.seed).select(0)
    records = index.records('train', 'frame', ids[:samples])
    dev = index.records('dev', 'frame')
    # Distributed contiguous minibatches approximate the actual fixed DEV order.
    step = max(1, len(dev) // max(1, samples // config.batch_size))
    dev_subset = [r for start in range(0, len(dev), step) for r in dev[start:start+config.batch_size]][:samples]
    reports = []
    for split, subset in (('train', records), ('dev', dev_subset)):
        for budget in (0, 64):
            loader = make_loader(subset, data_root, config, training=split == 'train', cache_mib=budget)
            start = time.perf_counter(); seen = 0
            for images, _, _ in loader:
                seen += len(images)
            seconds = time.perf_counter() - start
            reports.append({'split': split, 'cache_mib_per_worker': budget, 'workers': config.workers, 'samples': seen,
                'seconds': seconds, 'frames_per_second': seen / seconds, 'parent_rss_mib': memory_mib(),
                'worker_cache_upper_bound_mib': budget * max(1, config.workers),
                'note': 'RSS excludes worker processes. Cold one-pass subset; no model execution.'})
            del loader; gc.collect()
    index.close(); print(json.dumps(reports, indent=2))
    return reports


def probe_gpu(index_path, data_root, weights_root, config):
    """One disposable forward/backward, zero optimizer steps, on CUDA only."""
    require_cuda(); seed_everything(config.seed)
    index = Index(index_path, data_root)
    ids, _ = EpochSampler(index, per_class=config.train_per_class, seed=config.seed).select(0)
    records = index.records('train', 'frame', ids[:config.batch_size])
    images, targets, _ = next(iter(make_loader(records, data_root, replace(config, workers=0), training=True)))
    model = StagedCNN(config, weights_root).cuda(); model.set_phase(True); model.train()
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
    guard = GradientGuard(config); guard.begin_epoch(0)
    events = []
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
    with torch.autocast('cuda', enabled=config.amp):
        loss = nn.functional.binary_cross_entropy_with_logits(model(images.cuda()).squeeze(1), targets.cuda())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Non-finite probe loss')
    scaler.scale(loss).backward()
    guard.finish_step(optimizer, scaler, epoch=0, batch=1, emit=events.append, execute_step=False)
    torch.cuda.synchronize()
    report = {'model': config.model, 'gpu': torch.cuda.get_device_name(0), 'batch_size': len(images),
              'forward_backward_seconds': time.perf_counter()-start, 'optimizer_steps': 0,
              'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
              'total_gpu_mib': torch.cuda.get_device_properties(0).total_memory/2**20,
              'trainable': model.trainable_report(),
              'numerical_probe': events[-1],
              'note': 'Optimizer state is not allocated in this probe; actual training needs additional memory.'}
    del model, optimizer, scaler, guard, images, targets, loss; index.close(); gc.collect(); torch.cuda.empty_cache()
    return report


def diagnose_training_start(config, index_path, data_root, weights_root, output, max_batches=2):
    """Compare main-process vs spawned loading AFTER CUDA model initialization.

    Uses exact first-epoch candidate ordering and actual AMP backward/unscale/
    clipping, but deliberately performs ZERO optimizer steps on a disposable
    model. This cannot diagnose later corrupt samples or full-run OOM by itself.
    """
    config.validate(); require_cuda()
    if not 1 <= max_batches <= 4:
        raise ValueError('Diagnostic is capped at four batches per loader mode')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    with RunProgress(out, config.model, config.heartbeat_seconds) as progress:
        seed_everything(config.seed)
        progress.stage('diagnostic_index', announce=True)
        index = Index(index_path, data_root)
        ids, _ = EpochSampler(index, per_class=config.train_per_class, seed=config.seed).select(0)
        records = index.records('train', 'frame', ids[:max_batches*config.batch_size])
        index.close()
        progress.stage('diagnostic_cuda_model', announce=True)
        model = StagedCNN(config, weights_root).cuda(); model.set_phase(False); model.train()
        optimizer = make_optimizer(model, config)
        reports = []
        for workers in dict.fromkeys((0, config.workers)):
            mode = replace(config, workers=workers)
            progress.stage('diagnostic_iterator_start', workers=workers, batch=0, announce=True)
            loader = make_loader(records, data_root, mode, training=True, epoch=0)
            iterator = iter(loader)
            scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
            guard = GradientGuard(config); guard.begin_epoch(0)
            for batch in range(1, len(loader)+1):
                progress.stage('diagnostic_fetch_batch', workers=workers, batch=batch, announce=True)
                images, targets, _ = next(iterator)
                progress.stage('diagnostic_transfer', announce=True)
                images, targets = images.cuda(), targets.cuda()
                optimizer.zero_grad(set_to_none=True)
                progress.stage('diagnostic_forward_loss', announce=True)
                with torch.autocast('cuda', enabled=config.amp):
                    loss = nn.functional.binary_cross_entropy_with_logits(model(images).squeeze(1), targets)
                progress.stage('diagnostic_amp_backward_unscale_clip', announce=True)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite diagnostic loss')
                scaler.scale(loss).backward()
                events = []
                guard.finish_step(optimizer, scaler, epoch=0, batch=batch, emit=events.append, execute_step=False)
                progress.numerics(events[-1])
                torch.cuda.synchronize()
                reports.append({'workers': workers, 'batch': batch, 'loss': float(loss.detach()),
                                'numerics': events[-1], 'optimizer_steps': 0})
                progress.note(f'Diagnostic workers={workers} batch={batch}/{len(loader)} finished; '
                              f'skipped overflow={events[-1]["skipped_step"]}; optimizer steps=0')
            progress.stage('diagnostic_iterator_shutdown', announce=True)
            del iterator, loader
        write_json(out/'diagnostic_results.json', reports)
        del model, optimizer, scaler, images, targets, loss
        gc.collect(); torch.cuda.empty_cache()
        progress.note(f'Bounded diagnostic completed: {out}')
    return out
