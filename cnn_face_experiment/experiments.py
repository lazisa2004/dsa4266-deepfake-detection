"""Controlled recipes and preflight, delegating the actual training loop unchanged."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import uuid
import zipfile

import torch

from cnn_experiment.checkpoints import digest
from cnn_experiment.config import experiment_config
from cnn_experiment.data import make_loader, write_json
from cnn_experiment.models import StagedCNN, make_optimizer
from cnn_experiment.pool import FixedPool, FixedPoolSampler, DevSchedule, PoolConfig, build_pool
from .crops import BoxCache, FacePipeline, CropConfig, crop_previews, frame_key

EXPERIMENTS = {
    'resnet50_face_crop': ('resnet50', True, 1e-4),
    'efficientnet_b0_face_crop': ('efficientnet_b0', True, 1e-4),
    'efficientnet_b0_head_lr_2e4': ('efficientnet_b0', False, 2e-4),
}
EXPECTED_TRAIN_POOL_SHA256 = '8bcb5c9845f92b43e36213f22e410bcecedf983f7db87250f02a9968868eea92'


def experiment_configurations():
    return {name: replace(experiment_config(model, 'fixed20k_lr15e6'), head_lr=head_lr).validate()
            for name, (model, cropped, head_lr) in EXPERIMENTS.items()}


def output_paths(session):
    # A distinct parent per experiment also isolates rolling ZIP names for both EfficientNets.
    return {name: Path(session)/name/'run' for name in EXPERIMENTS}


def existing_pool(path):
    """Fail closed for an absent or different manifest, including explicit overrides."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Verified TRAIN pool not found; use resolve_train_pool to reuse or reproduce it automatically')
    actual = digest(path)
    if actual != EXPECTED_TRAIN_POOL_SHA256:
        raise ValueError(f'TRAIN pool SHA256 mismatch: expected {EXPECTED_TRAIN_POOL_SHA256}, got {actual}. Training is blocked; no sampling rule is relaxed.')
    return path


def _verify_pool(path, data_root):
    path = existing_pool(path)
    pool = FixedPool(path, data_root)
    if pool.content['policy'] != asdict(PoolConfig()):
        raise ValueError('TRAIN pool must use the exact original PoolConfig and seed4266')
    print(f'Original TRAIN pool verified: {EXPECTED_TRAIN_POOL_SHA256}', flush=True)
    return path


def attached_pool_zips(base='/kaggle/input'):
    """Bounded completed-CNN/pool ZIP discovery; never enter video split folders."""
    result = []
    base = Path(base)
    if not base.is_dir():
        return result
    for folder, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d.lower() not in ('train', 'dev', 'test') and not d.startswith('.')]
        if len(Path(folder).relative_to(base).parts) >= 6 or 'metadata.json' in files:
            dirs[:] = []
        for name in files:
            cnn_zip = name.startswith(('resnet50_', 'efficientnet_b0_')) and ('final' in name or 'latest_completed_epoch' in name)
            if name.endswith('.zip') and (cnn_zip or name == 'fixed_cnn_pool_manifest.zip') and 'lstm' not in name.lower():
                result.append(Path(folder)/name)
    return sorted(result)


def resolve_train_pool(data_root, output_root, *, existing_paths=(), completed_run_zips=(), override=None):
    """Reuse exact metadata, otherwise invoke the ORIGINAL deterministic builder.

    No SQLite/index regeneration, model work or TEST access. Only train_pool.json
    is read from a ZIP; weights, video data and prediction files are never extracted.
    Every branch requires the historical byte-for-byte SHA, including generated pools.
    """
    output_root = Path(output_root)
    if override is not None:
        return _verify_pool(override, data_root)
    candidates = [Path(p) for p in existing_paths if Path(p).is_file()]
    local = output_root/'train_pool.json'
    if local.is_file():
        candidates.append(local)
    mismatches = []
    for path in sorted(set(candidates)):
        if digest(path) == EXPECTED_TRAIN_POOL_SHA256:
            print('Reusing existing original TRAIN manifest:', path, flush=True)
            return _verify_pool(path, data_root)
        mismatches.append(str(path))
    archive_mismatches = []
    for archive_path in sorted({Path(p) for p in completed_run_zips}):
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted((info for info in archive.infolist() if PurePosixPath(info.filename).name == 'train_pool.json'), key=lambda info: info.filename)
            for member in members:
                parts = PurePosixPath(member.filename)
                if parts.is_absolute() or '..' in parts.parts or '\\' in member.filename or member.file_size > 32*2**20:
                    raise ValueError('Unsafe/oversized TRAIN manifest in completed-run ZIP; nothing extracted')
                payload = archive.read(member)  # small metadata only; ZipFile verifies member CRC
                if hashlib.sha256(payload).hexdigest() != EXPECTED_TRAIN_POOL_SHA256:
                    archive_mismatches.append(f'{archive_path.name}:{member.filename}')
                    continue
                if json.loads(payload)['policy'] != asdict(PoolConfig()):
                    raise ValueError('Saved TRAIN pool policy differs from original')
                output_root.mkdir(parents=True, exist_ok=True)
                # Never overwrite an existing manifest or extract arbitrary archive paths.
                if local.exists():
                    return _verify_pool(local, data_root)
                temporary = output_root/('train_pool_'+uuid.uuid4().hex+'.partial')
                with temporary.open('xb') as handle:
                    handle.write(payload)
                temporary.rename(local)
                print('Reused ONLY train_pool.json from completed-run ZIP:', archive_path.name, flush=True)
                return _verify_pool(local, data_root)
    if mismatches or archive_mismatches:
        raise ValueError('Existing TRAIN pool artifacts do not match the original saved hash. Training blocked; '
                         'no replacement pool generated. Check POOL_OVERRIDE/attached runs. Sources: '+str(mismatches+archive_mismatches))
    if output_root.exists():
        raise FileExistsError('Incomplete pool build directory preserved. Use a new session/output path; no partial manifest is accepted.')
    print('No original manifest found; reproducing it with cnn_experiment.pool.build_pool and PoolConfig(seed=4266). '
          'Reads TRAIN timelines and verifies selected frames; no model training or DEV/TEST rebuild.', flush=True)
    generated = build_pool(data_root, output_root, PoolConfig())
    return _verify_pool(generated, data_root)


def verify_inputs(index_path, pool_path, data_root):
    pool = FixedPool(existing_pool(pool_path), data_root)
    dev = DevSchedule(index_path, data_root)
    try:
        records = dev.records('dev', 'frame')
        samplers = [FixedPoolSampler(pool, 4000, 4266) for _ in range(3)]
        reports = []
        for epoch in range(10):
            selections = [s.select(epoch) for s in samplers]
            ids, report = selections[0]
            if any(ids != other[0] for other in selections[1:]) or len(ids) != len(set(ids)) or len(ids) != 8000:
                raise AssertionError('Controlled experiments must use identical balanced epoch IDs')
            if sum(r['label'] for r in pool.records(ids)) != 4000:
                raise AssertionError('Class balance changed')
            reports.append({k: report[k] for k in ('epoch', 'unique_samples', 'unique_families', 'duplicate_rate')})
        return {'dev': dev.report, 'index_sha256': digest(index_path), 'pool_sha256': pool.sha256,
                'train_pool': pool.report(), 'epoch_checks': reports, 'test_used': False}, records
    finally:
        dev.close()


def preview_records(dev_records, pool_path, count_each=6):
    # Evenly distributed ordinal positions, independent of labels/manipulation intervals.
    pool = FixedPool(pool_path)
    train = pool.records()
    def spaced(records):
        return [records[i*len(records)//count_each] for i in range(count_each)]
    return spaced(train)+spaced(dev_records)


def bounded_preflight(configs, records, data_root, weights_root, pipeline, output, device='cuda', *, face_inputs=None):
    """Finite forward/backward checks with NO optimizer/scaler steps or production weights."""
    from cnn_experiment.training import seed_everything
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if len(records) > 16 or not records:
        raise ValueError('Use 1-16 already-selected DEV/TRAIN frame records for preflight')
    results = {}
    for name, config in configs.items():
        seed_everything(config.seed)
        cfg = replace(config, workers=0, batch_size=2)
        cropped = EXPERIMENTS[name][1] if face_inputs is None else face_inputs[name]
        loader_factory = pipeline.make_loader if cropped else make_loader
        start = time.perf_counter()
        loader = loader_factory(records, data_root, cfg)
        x, y, _ = next(iter(loader))
        if x.shape[1:] != (3, 224, 224) or not bool(torch.isfinite(x).all()):
            raise ValueError('Invalid normalized CNN inputs')
        model = StagedCNN(cfg, weights_root, pretrained=True).to(device)
        optimizer = make_optimizer(model, cfg)
        before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        phases = []
        for active in (False, True):
            model.set_phase(active); model.eval(); model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda' and cfg.amp):
                logits = model(x.to(device)).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(device))
            # A single unscaled smoke backward: no updates, no training loop.
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError('Non-finite/absent gradients in bounded preflight')
            phases.append({'phase': model.phase, 'loss': float(loss.detach()),
                           'gradient_norm': float(torch.linalg.vector_norm(torch.stack([g.double().norm() for g in grads])))})
        if any(not torch.equal(before[key], value) for key, value in model.state_dict().items()):
            raise AssertionError('Preflight changed model weights/BN buffers')
        results[name] = {'input_shape': list(x.shape), 'phases': phases, 'optimizer_steps': 0,
            'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
            'seconds': time.perf_counter()-start, 'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None}
        del model, optimizer, before, loader, x, y, logits, loss, grads
    # Spawned-worker identity, detector/caches must be pickle-safe.
    cfg = replace(next(iter(configs.values())), batch_size=2, workers=0)
    expected = torch.cat([x for x, _, _ in pipeline.make_loader(records[:4], data_root, cfg)])
    spawned = torch.cat([x for x, _, _ in pipeline.make_loader(records[:4], data_root, replace(cfg, workers=2))])
    if not torch.equal(expected, spawned):
        raise AssertionError('Spawned face loader changed input tensors')
    results['spawned_worker_identity'] = True
    write_json(output/'preflight.json', results)
    return results


def train_experiment(name, index_path, pool_path, data_root, weights_root, output, pipeline,
                     *, resume_from=None, provenance=None):
    from cnn_experiment.training import train
    if name not in EXPERIMENTS:
        raise ValueError('Unknown controlled experiment')
    config = experiment_configurations()[name]
    model_name, cropped, _ = EXPERIMENTS[name]
    output = Path(output)
    if output.exists():
        raise FileExistsError('Existing outputs preserved; choose a NEW session/run for resume')
    pool_hash = digest(existing_pool(pool_path))
    if (pipeline.identity['index_sha256'], pipeline.identity['pool_sha256']) != (digest(index_path), pool_hash):
        raise ValueError('Pipeline/index/TRAIN-pool identity mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    controlled = {'experiment': name, 'model': model_name, 'config': config.as_dict(),
        'input_pipeline': pipeline.identity if cropped else None,
        'preprocessing_mode': ('face_mask' if pipeline.identity['kind'] == 'face_mask' else 'crop_30') if cropped else 'full_frame',
        'changed_factor': ('face mask only' if pipeline.identity['kind'] == 'face_mask' else 'face crop only') if cropped else 'head_lr only: 1e-4 -> 2e-4',
        'initialization': 'same ImageNet checkpoint and seed; fresh independent model unless explicit same-experiment resume',
        'source': provenance, 'index_sha256': digest(index_path), 'pool_sha256': pool_hash, 'test_used': False}
    # Metadata created before train() makes the run dir; engine snapshots it into every ZIP.
    write_json(output.parent/'experiment_plans.json', {model_name: controlled})
    if provenance is not None:
        write_json(output.parent/'git_provenance.json', provenance)
    result = train(config, index_path, data_root, weights_root, output, resume_from,
        pool_path=pool_path, input_pipeline=pipeline if cropped else None)
    return result
