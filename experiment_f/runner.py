"""Standalone Experiment F. No face package imports, downloads, or work on import."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import zipfile

from cnn_experiment.config import experiment_config

EXPERIMENT_ID = 'resnet50_full_frame_head5e5_roc_f'
EXPECTED_POOL_SHA256 = '8bcb5c9845f92b43e36213f22e410bcecedf983f7db87250f02a9968868eea92'
EXPECTED_INDEX_SHA256 = 'ac3cd8c440f0bb1e53cf955bcc6a2596e9ee1bc40e508f254c82e10bde13cfcb'


def experiment_f_config():
    return replace(experiment_config('resnet50', 'fixed20k_lr15e6'),
                   head_lr=5e-5, backbone_lr=1.5e-5, epochs=15, patience=5,
                   checkpoint_selection='frame_roc_auc').validate()


def create_f_output(base='/kaggle/working/experiment1_f'):
    """Create a NEW F-only session; run/ and downloads/ remain siblings."""
    from cnn_experiment.workflow import create_session
    return create_session(Path(base)/EXPERIMENT_ID)/'run'


def _check_output(output):
    output = Path(output).resolve()
    if EXPERIMENT_ID not in output.parts or output.name != 'run':
        raise ValueError('Use create_f_output: F requires its own named session/run directory')
    if output.exists() or any((output.parent/name).exists() for name in
                             ('experiment_plans.json', 'git_provenance.json')):
        raise FileExistsError('Existing run/plans preserved; create a NEW F session, including for resume')
    return output


def validate_f_inputs(index_path, pool_path, data_root):
    """Verify existing metadata only. Never rebuild an index/pool or scan TEST."""
    from cnn_experiment.checkpoints import digest
    from cnn_experiment.pool import FixedPool, DevSchedule
    if digest(index_path) != EXPECTED_INDEX_SHA256:
        raise ValueError('F requires the unchanged approved TRAIN/DEV index; no replacement is generated')
    if digest(pool_path) != EXPECTED_POOL_SHA256:
        raise ValueError('F requires the original saved fixed20k TRAIN pool; no replacement is generated')
    pool = FixedPool(pool_path, data_root)
    dev = DevSchedule(index_path, data_root)
    try:
        return {'experiment': EXPERIMENT_ID, 'input': 'full_frame',
                'index_sha256': EXPECTED_INDEX_SHA256, 'pool_sha256': pool.sha256,
                'data_root': str(Path(data_root).resolve()),
                'train_pool': pool.report(), 'dev': dict(dev.report), 'test_used': False}
    finally:
        dev.close()


def preflight_f(index_path, pool_path, data_root, weights_root, output):
    """Explicit FUTURE preflight: original full-frame loader, zero optimizer steps.

    Do not invoke during another model's active training. Writes only a new F
    diagnostic folder and operates on disposable pretrained models.
    """
    from cnn_experiment.data import write_json
    from cnn_experiment.workflow import (diagnostic_stage, verify_epoch_sampling,
                                         diagnose_training_start, probe_gpu)
    output = _check_output(output)
    diagnostic = output.parent/'preflight'
    diagnostic.mkdir(parents=True, exist_ok=False)
    with diagnostic_stage(diagnostic, 'full_frame_f'):
        inputs = validate_f_inputs(index_path, pool_path, data_root)
        config = experiment_f_config()
        sampling = verify_epoch_sampling(index_path, data_root, config, pool_path=pool_path)
        loading_path = diagnose_training_start(config, index_path, data_root, weights_root,
                                                diagnostic/'loader_smoke', max_batches=1,
                                                pool_path=pool_path)
        loading = json.loads((Path(loading_path)/'diagnostic_results.json').read_text(encoding='utf-8'))
        gpu = probe_gpu(index_path, data_root, weights_root, config, pool_path=pool_path)
        if not loading or any(row.get('optimizer_steps') != 0 for row in loading) or gpu.get('optimizer_steps') != 0:
            raise ValueError('F preflight must finish with zero optimizer steps')
        proof = {'experiment': EXPERIMENT_ID, 'input': 'full_frame',
                 'config': config.as_dict(), 'inputs': inputs,
                 'weights_root': str(Path(weights_root).resolve()),
                 'output': str(output), 'sampling': sampling,
                 'loading': loading, 'gpu': gpu, 'optimizer_steps': 0, 'test_used': False}
        write_json(diagnostic/'preflight.json', proof)
        return proof


def _saved_f(source):
    """Read F identity from saved metadata only, never deserialize trained weights."""
    from cnn_experiment.reporting import SavedRun
    saved = SavedRun(source)  # Enforces the existing TRAIN/DEV reporting gate.
    if saved.single_history:
        raise ValueError('F identity needs a run folder or verified model ZIP, not a lone history file')
    if saved.archive:
        with zipfile.ZipFile(source) as archive:
            name = saved.prefix+'experiment_plan.json'
            if archive.getinfo(name).file_size > 1024**2:
                raise ValueError('Oversized experiment identity')
            plan = json.loads(archive.read(name))
    else:
        path = Path(source)/'experiment_plan.json'
        if path.stat().st_size > 1024**2:
            raise ValueError('Oversized experiment identity')
        plan = json.loads(path.read_text(encoding='utf-8'))
    config = saved.json('config.json') or {}
    summary = saved.json('training_summary.json') or {}
    required = experiment_f_config().as_dict()
    identity_keys = ('head_lr', 'backbone_lr', 'epochs', 'patience', 'warmup_epochs',
                     'batch_size', 'dropout', 'training_pool', 'threshold', 'topk_fraction')
    if (plan.get('experiment') != EXPERIMENT_ID or plan.get('preprocessing_mode') != 'full_frame' or
        plan.get('input_pipeline') is not None or saved.json('input_pipeline.json') is not None or
        config.get('model') != 'resnet50' or config.get('checkpoint_selection') != 'frame_roc_auc' or
        any(config.get(key) != required[key] for key in identity_keys) or
        summary.get('checkpoint_selection', 'frame_roc_auc') != 'frame_roc_auc'):
        raise ValueError('Not an independently labelled full-frame F run; masked/E/other results cannot be relabelled')
    return saved


def train_f(index_path, pool_path, data_root, weights_root, output, *, preflight,
            resume_from=None, provenance=None, display=True):
    """Explicit training entry point; core full-frame route is input_pipeline=None."""
    from cnn_experiment.training import train
    output = _check_output(output)
    config = experiment_f_config()
    inputs = validate_f_inputs(index_path, pool_path, data_root)
    if (preflight.get('experiment') != EXPERIMENT_ID or preflight.get('input') != 'full_frame' or
        preflight.get('config') != config.as_dict() or preflight.get('inputs') != inputs or
        preflight.get('output') != str(output) or preflight.get('optimizer_steps') != 0 or
        preflight.get('weights_root') != str(Path(weights_root).resolve())):
        raise ValueError('Complete the matching F-only preflight; D/E or stale receipts are not accepted')
    if resume_from is not None:
        resume_from = Path(resume_from)
        if resume_from.name != 'last.pt' or not resume_from.is_file() or not (resume_from.parent/'best.pt').is_file():
            raise ValueError('Resume F with last.pt and its companion best.pt in a restored F run')
        _saved_f(resume_from.parent)
    plan = {'experiment': EXPERIMENT_ID, 'model': 'resnet50', 'preprocessing_mode': 'full_frame',
            'input_pipeline': None, 'config': config.as_dict(), 'checkpoint_selection': 'frame_roc_auc',
            'early_stopping_monitor': 'DEV frame ROC-AUC; earliest strict maximum; patience5',
            'scheduler_monitor': 'DEV video ROC-AUC (unchanged)',
            'initialization': 'Independent original ImageNet weights unless explicit F-only resume',
            'index_sha256': inputs['index_sha256'], 'pool_sha256': inputs['pool_sha256'],
            'source': provenance, 'test_used': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive writes: never overwrite existing experiment plans or provenance.
    for name, value in (('experiment_plans.json', {'resnet50': plan}),
                        ('git_provenance.json', provenance)):
        with (output.parent/name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
    result = train(config, index_path, data_root, weights_root, output, resume_from,
                   pool_path=pool_path, input_pipeline=None)
    if display:
        display_f_run(result)
    return result


def display_f_run(source):
    from cnn_experiment.reporting import display_completed_run
    _saved_f(source)
    print('Experiment F | ResNet50 FULL FRAME | selected by DEV FRAME ROC-AUC | threshold0.5')
    return display_completed_run(source, name='F - ResNet50 full frame, head LR5e-5', kind='frame')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'pool', 'data', 'weights'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--output-base', type=Path, default=Path('/kaggle/working/experiment1_f'))
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--train', action='store_true', help='Explicitly train AFTER F preflight passes; otherwise preflight only')
    args = parser.parse_args(argv)
    output = create_f_output(args.output_base)
    print('Independent F output:', output, flush=True)
    proof = preflight_f(args.index, args.pool, args.data, args.weights, output)
    if args.train:
        train_f(args.index, args.pool, args.data, args.weights, output,
                preflight=proof, resume_from=args.resume_from)
    else:
        print('F preflight complete; no experiment training requested.', flush=True)


if __name__ == '__main__':
    main()
