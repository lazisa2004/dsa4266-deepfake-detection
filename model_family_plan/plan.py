"""Explicit new IDs and duplicate analysis. Nothing trains or scans data on import."""
from dataclasses import replace

from cnn_experiment.config import experiment_config

RECIPES = {
    '1a': {'model': 'resnet50', 'input': 'full_frame', 'head_lr': 1e-4, 'backbone_lr': 4.5e-6},
    '1b': {'model': 'resnet50', 'input': 'face_mask', 'head_lr': 1e-4, 'backbone_lr': 4.5e-6},
    '2a': {'model': 'efficientnet_b0', 'input': 'full_frame', 'head_lr': 2e-4, 'backbone_lr': 1.5e-5},
    '2b': {'model': 'efficientnet_b0', 'input': 'face_mask', 'head_lr': 2e-4, 'backbone_lr': 1.5e-5},
    '3a': {'model': 'resnet50_lstm', 'input': 'full_frame_clips', 'head_lr': 3e-5, 'backbone_lr': None},
}
# This is a review/display order, never an execution loop.
REVIEW_ORDER = ('1a', '1b', '3a', '2a', '2b')


def configuration(experiment_id):
    if experiment_id not in RECIPES:
        raise ValueError('Use an explicit new ID: 1a, 1b, 2a, 2b or 3a')
    recipe = RECIPES[experiment_id]
    if experiment_id == '3a':
        from cnn_experiment.temporal import TemporalConfig
        return replace(TemporalConfig(), head_lr=3e-5, epochs=15, patience=3,
                       checkpoint_selection='clip_roc_auc').validate()
    return replace(experiment_config(recipe['model'], 'fixed20k_lr15e6'),
                   head_lr=recipe['head_lr'], backbone_lr=recipe['backbone_lr'],
                   epochs=15, patience=3, checkpoint_selection='frame_roc_auc').validate()


def declared_plan():
    return {name: {'experiment_id': name, **RECIPES[name],
                   'config': configuration(name).as_dict(),
                   'checkpoint_monitor': 'DEV clip ROC-AUC' if name == '3a' else 'DEV frame ROC-AUC',
                   'scheduler_monitor': 'DEV clip PR-AUC (original)' if name == '3a' else 'DEV video ROC-AUC (original)',
                   'execution': ('FIRST; explicit training cell only' if name == '1a' else
                                 'after ResNet review; explicit training cell only' if name == '2a' else
                                 'OPTIONAL; explicit training cell only'),
                   'initialization': 'explicit original trained full-frame ResNet50; frozen' if name == '3a' else 'independent ImageNet',
                   'test_used': False} for name in REVIEW_ORDER}


def duplicate_audit():
    """Compare actual definitions; never map new IDs onto old outputs."""
    from cnn_face_experiment.experiments import experiment_configurations, EXPERIMENTS
    from cnn_face_experiment.roc_experiments import roc_experiment_configurations
    from experiment_f.runner import experiment_f_config
    from cnn_experiment.temporal import TemporalConfig
    historical = {name: (cfg, 'face_mask' if EXPERIMENTS[name][1] else 'full_frame')
                  for name, cfg in experiment_configurations().items()}
    historical.update({name: (cfg, 'face_mask') for name, cfg in roc_experiment_configurations().items()})
    historical['F'] = (experiment_f_config(), 'full_frame')
    historical['original_resnet50_lstm'] = (TemporalConfig(), 'full_frame_clips')
    result = []
    for new_id in REVIEW_ORDER:
        new = configuration(new_id)
        for old_id, (old, mode) in historical.items():
            if new.model != old.model or RECIPES[new_id]['input'] != mode:
                continue
            differences = {k: {'historical': old.as_dict().get(k), 'new': v}
                           for k, v in new.as_dict().items() if old.as_dict().get(k) != v}
            result.append({'new_id': new_id, 'historical_id': old_id, 'input': mode,
                           'identical_configuration': not differences, 'differences': differences})
    c = next(row for row in result if row['new_id'] == '2a' and row['historical_id'] == 'efficientnet_b0_head_lr_2e4')
    if set(c['differences']) != {'epochs', 'checkpoint_selection'}:
        raise ValueError('Historical C no longer matches the reviewed reference; inspect before using 2a')
    if any(row['identical_configuration'] for row in result):
        raise ValueError('A new recipe exactly duplicates a historical recipe; inspect before scheduling')
    return result
