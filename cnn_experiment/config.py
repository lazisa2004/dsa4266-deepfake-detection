"""Central configuration for the v2 CNN experiment."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass
class Config:
    model: str = 'resnet50'
    seed: int = 4266
    batch_size: int = 32
    epochs: int = 10
    warmup_epochs: int = 2
    patience: int = 3
    head_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    dropout: float = .3
    flip_probability: float = .5
    finetune_final_stage: bool = True
    train_per_class: int = 4000
    clip_per_class: int = 2000
    workers: int = 2
    prefetch_factor: int = 2
    cache_mib: int = 0
    amp: bool = True
    threshold: float = .5
    topk_fraction: float = .05
    scheduler_patience: int = 1
    loader_timeout_seconds: float = 120.
    train_progress_every: int = 50
    dev_progress_every: int = 200
    heartbeat_seconds: float = 30.
    max_consecutive_gradient_skips: int = 5
    max_gradient_skips_per_epoch: int = 20
    checkpoint_selection: str = 'video_roc_auc'
    training_pool: str = 'legacy_full_index'

    def validate(self):
        if self.training_pool not in ('legacy_full_index','fixed20k'):
            raise ValueError('Unknown CNN training-pool protocol')
        if self.training_pool=='fixed20k' and self.train_per_class!=4000:
            raise ValueError('Fixed-pool CNN epochs require 4,000 per class')
        if self.checkpoint_selection not in ('video_roc_auc', 'frame_pr_auc', 'frame_roc_auc'):
            raise ValueError('Checkpoint selection must be video_roc_auc, frame_pr_auc or frame_roc_auc')
        if self.model not in ('resnet50', 'efficientnet_b0'):
            raise ValueError('This stage trains CNNs only')
        if not 1 <= self.warmup_epochs < self.epochs:
            raise ValueError('Use at least one head epoch and one later epoch')
        if min(self.batch_size, self.patience, self.train_per_class,
               self.clip_per_class, self.prefetch_factor) < 1 or self.batch_size % 2:
            raise ValueError('Positive counts and an even batch size required')
        if self.workers < 0 or self.cache_mib < 0 or self.scheduler_patience < 0:
            raise ValueError('Invalid loader or scheduler configuration')
        if type(self.amp) is not bool or min(self.max_consecutive_gradient_skips, self.max_gradient_skips_per_epoch) < 1:
            raise ValueError('AMP must be boolean; gradient skip limits must be positive')
        if (not math.isfinite(self.loader_timeout_seconds) or self.loader_timeout_seconds < 0 or
            not math.isfinite(self.heartbeat_seconds) or self.heartbeat_seconds < 0 or
            min(self.train_progress_every, self.dev_progress_every) < 1):
            raise ValueError('Invalid progress interval or worker timeout')
        for x in (self.head_lr, self.backbone_lr):
            if not math.isfinite(x) or x <= 0:
                raise ValueError('Learning rates must be finite and positive')
        if self.backbone_lr > self.head_lr:
            raise ValueError('Backbone LR must not exceed head LR')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid weight decay')
        if not 0 <= self.dropout < 1 or not 0 <= self.flip_probability <= 1:
            raise ValueError('Invalid dropout or augmentation probability')
        if not 0 <= self.threshold <= 1 or self.topk_fraction != .05:
            raise ValueError('This version fixes Top-5% aggregation')
        return self

    def as_dict(self):
        return asdict(self)


# Recipes are explicit; fixed20k profiles change pool and selection together.
EXPERIMENTS = {
    'baseline': {},
    'resnet_weight_decay_3e-4': {'weight_decay': 3e-4},
    'fixed20k_lr1e5': {'training_pool':'fixed20k','checkpoint_selection':'frame_pr_auc'},
    'fixed20k_lr15e6': {'training_pool':'fixed20k','checkpoint_selection':'frame_pr_auc','backbone_lr':1.5e-5},
}

HISTORICAL_BASELINE = {
    'resnet50': {'best_epoch': 9, 'dev_frame_roc_auc': .8871688331974448,
        'dev_frame_pr_auc': .37357527800828777, 'dev_video_roc_auc': .8315839999999995,
        'epoch10_train_roc_auc': .9699439687499969,
        'history_sha256': 'd09442a71d8b516f14e6023bcd9d398e1e2fc8e07b23b6fb1b0a32aebe80773b'},
    'efficientnet_b0': {'best_epoch': 10, 'dev_frame_roc_auc': .8063480889132856,
        'dev_frame_pr_auc': .21249941960362337, 'dev_video_roc_auc': .739199999999999,
        'epoch10_train_roc_auc': .8370912500000004,
        'history_sha256': '8623c953cdd08890b2ebd2ec229c6ae0ea6fa8fb31f7d805725e8727af268d6e'},
}


def experiment_config(model, profile='baseline', amp=True, checkpoint_selection=None, warmup_epochs=2):
    """Explicit opt-in experiment; baseline defaults and checkpoint schema unchanged."""
    if profile not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment profile: {profile}')
    if profile.startswith('resnet_') and model != 'resnet50' or profile.startswith('efficientnet_') and model != 'efficientnet_b0':
        raise ValueError('Experiment profile does not match the model')
    config=replace(Config(model=model,amp=amp,warmup_epochs=warmup_epochs),**EXPERIMENTS[profile])
    if checkpoint_selection is not None:config=replace(config,checkpoint_selection=checkpoint_selection)
    return config.validate()


def selection_label(selection):
    return {'video_roc_auc': 'DEV video ROC-AUC, fixed Top-5%',
            'frame_pr_auc': 'DEV frame PR-AUC (trapezoidal)',
            'frame_roc_auc': 'DEV frame ROC-AUC',
            'clip_roc_auc': 'DEV clip ROC-AUC',
            'clip_pr_auc':'DEV clip PR-AUC (trapezoidal)'}[selection]


def checkpoint_score(report, selection):
    selection_label(selection)
    value = (report['video']['roc_auc'] if selection == 'video_roc_auc' else
             report['unit']['roc_auc'] if selection in ('frame_roc_auc', 'clip_roc_auc') else report['unit']['pr_auc'])
    if value is None or not math.isfinite(value):
        raise ValueError('Checkpoint-selection score is undefined/non-finite')
    return value


def experiment_plan(config, profile):
    expected = experiment_config(config.model, profile, config.amp, config.checkpoint_selection,config.warmup_epochs)
    if config.as_dict() != expected.as_dict():
        raise ValueError('Config differs from the named one-factor experiment')
    return {'profile': profile, 'changed_factor': EXPERIMENTS[profile], 'config': config.as_dict(),
        'checkpoint_selection': 'maximum '+selection_label(config.checkpoint_selection)+'; earliest strict maximum',
        'selection_change_from_historical': config.checkpoint_selection != 'video_roc_auc',
        'comparison_type': 'new combined pool/selection/LR configuration; not an isolated LR ablation' if profile=='fixed20k_lr15e6' else 'declared controlled configuration',
        'early_stopping_monitor': selection_label(config.checkpoint_selection),
        'scheduler_monitor': 'DEV video ROC-AUC', 'cross_model_selection': selection_label(config.checkpoint_selection)+'; balanced DEV loss tie-break',
        'primary_analyses': ['DEV frame PR-AUC (trapezoidal)', 'DEV frame ROC-AUC'],
        'secondary_analysis': 'DEV video ROC-AUC', 'threshold': .5,
        'threshold_analysis': 'DEV only; descriptive F1/F2/FPR, does not change training or final TEST policy',
        'historical_baseline': HISTORICAL_BASELINE[config.model], 'historical_weights_available': False,
        'comparison_note': 'Use same seed, precision and selection rule for one-factor comparisons. A different checkpoint-selection rule is a separately declared factor; TEST remains isolated.'}
