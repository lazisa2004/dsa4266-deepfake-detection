"""Central configuration for the v2 CNN experiment."""
from dataclasses import asdict, dataclass
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

    def validate(self):
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
