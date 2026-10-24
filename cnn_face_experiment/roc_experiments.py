"""Additive D/E recipes. Existing A/B/C recipes and six-model reports are unchanged."""
from dataclasses import replace
from pathlib import Path

from cnn_experiment.checkpoints import digest
from cnn_experiment.config import selection_label
from cnn_experiment.data import write_json
from .experiments import bounded_preflight, existing_pool, experiment_configurations


ROC_EXPERIMENTS = {
    'efficientnet_b0_face_mask_roc_d': {
        'letter': 'D', 'baseline': 'efficientnet_b0_face_crop',
        'head_lr': 1e-4, 'backbone_lr': 5e-5},
    'resnet50_face_mask_roc_e': {
        'letter': 'E', 'baseline': 'resnet50_face_crop',
        'head_lr': 5e-5, 'backbone_lr': 1.5e-5},
}


def roc_experiment_configurations():
    baseline = experiment_configurations()
    return {name: replace(baseline[recipe['baseline']], head_lr=recipe['head_lr'],
                          backbone_lr=recipe['backbone_lr'], epochs=15, patience=5,
                          checkpoint_selection='frame_roc_auc').validate()
            for name, recipe in ROC_EXPERIMENTS.items()}


def roc_output_paths(session):
    return {name: Path(session)/name/'run' for name in ROC_EXPERIMENTS}


def validate_roc_pipeline(pipeline):
    if pipeline.identity.get('kind') != 'face_mask':
        raise ValueError('Experiments D/E require the unchanged face_mask pipeline, not crop_30/full frame')


def roc_bounded_preflight(records, data_root, weights_root, pipeline, output, device='cuda'):
    validate_roc_pipeline(pipeline)
    configs = roc_experiment_configurations()
    return bounded_preflight(configs, records, data_root, weights_root, pipeline, output,
                             device=device, face_inputs={name: True for name in configs})


def train_roc_experiment(name, index_path, pool_path, data_root, weights_root,
                         output, pipeline, *, resume_from=None, provenance=None):
    """Explicit training call only; no completed model or previous selection is reused."""
    from cnn_experiment.training import train
    if name not in ROC_EXPERIMENTS:
        raise ValueError('Unknown D/E experiment')
    validate_roc_pipeline(pipeline)
    config = roc_experiment_configurations()[name]
    output = Path(output)
    if output.exists():
        raise FileExistsError('Existing experiment preserved; choose a NEW run directory')
    pool_hash = digest(existing_pool(pool_path))
    if (pipeline.identity['index_sha256'], pipeline.identity['pool_sha256']) != (digest(index_path), pool_hash):
        raise ValueError('Pipeline/index/TRAIN-pool identity mismatch')
    recipe = ROC_EXPERIMENTS[name]
    plan = {'experiment': name, 'letter': recipe['letter'], 'model': config.model,
            'config': config.as_dict(), 'preprocessing_mode': 'face_mask',
            'input_pipeline': pipeline.identity, 'checkpoint_selection': 'frame_roc_auc',
            'selection_label': selection_label('frame_roc_auc'),
            'early_stopping_monitor': 'DEV frame ROC-AUC; earliest strict maximum; patience5',
            'scheduler_monitor': 'DEV video ROC-AUC (unchanged)',
            'comparison_note': 'New selection/budget/LR recipe; previous PR-selected checkpoints remain unchanged.',
            'initialization': 'Independent original ImageNet weights, unless explicit same-experiment resume',
            'source': provenance, 'index_sha256': digest(index_path),
            'pool_sha256': pool_hash, 'threshold': .5, 'test_used': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.parent/'experiment_plans.json', {config.model: plan})
    if provenance is not None:
        write_json(output.parent/'git_provenance.json', provenance)
    return train(config, index_path, data_root, weights_root, output, resume_from,
                 pool_path=pool_path, input_pipeline=pipeline)


def display_roc_completed_run(source, *, name):
    """Same cumulative table/yellow highlight/graph order; never reselect old runs."""
    from cnn_experiment.reporting import SavedRun, display_completed_run
    saved = SavedRun(source)
    config = saved.json('config.json') or {}
    summary = saved.json('training_summary.json') or {}
    if (config.get('checkpoint_selection') != 'frame_roc_auc' or
        summary.get('checkpoint_selection', 'frame_roc_auc') != 'frame_roc_auc'):
        raise ValueError('This D/E report requires recorded DEV frame ROC-AUC selection')
    print(name+' | selected by DEV FRAME ROC-AUC | classification threshold 0.5')
    return display_completed_run(source, name=name, kind='frame')
