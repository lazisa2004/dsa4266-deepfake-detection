"""Synthetic tensors only: AMP overflow/skip and checkpoint resume, no CNN training."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from cnn_experiment.config import Config
from cnn_experiment.data import PROTOCOL
from cnn_experiment.training import GradientGuard, NumericalInstabilityError
from cnn_experiment.checkpoints import copy_resume_boundary, inspect_resume_checkpoint, load_resume_bundle, restore_training_state, rng_state
from cnn_experiment.data import EpochSampler
from tests.test_data import FixtureIndex


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


class GradientTests(unittest.TestCase):
    def setup_step(self, **config):
        cfg = Config(**config)
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        opt = torch.optim.AdamW([p], lr=cfg.head_lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler('cpu', enabled=cfg.amp, init_scale=8., growth_interval=100)
        guard = GradientGuard(cfg); guard.begin_epoch(0)
        return p, opt, scaler, guard

    def backward(self, p, opt, scaler):
        opt.zero_grad(set_to_none=True)
        scaler.scale(p.sum()).backward()

    def test_amp_inf_skips_weights_and_moments_backs_off_then_recovers(self):
        p, opt, scaler, guard = self.setup_step()
        events = []
        self.backward(p, opt, scaler)
        guard.finish_step(opt, scaler, epoch=0, batch=1, emit=events.append)
        before, moments = p.detach().clone(), copy.deepcopy(opt.state_dict())
        self.backward(p, opt, scaler); p.grad[0] = float('inf')
        event = guard.finish_step(opt, scaler, epoch=0, batch=2, emit=events.append)
        self.assertTrue(torch.equal(p, before)); self.assertTrue(equal_tree(opt.state_dict(), moments))
        self.assertEqual(event['amp_scale_before'], 8.); self.assertEqual(event['amp_scale_after'], 4.)
        self.assertTrue(event['skipped_step']); self.assertFalse(event['optimizer_step_applied'])
        self.assertIsNone(p.grad); self.assertEqual(event['epoch'], 1)
        self.backward(p, opt, scaler)
        recovered = guard.finish_step(opt, scaler, epoch=0, batch=3, emit=events.append)
        self.assertFalse(torch.equal(p, before)); self.assertEqual(recovered['consecutive_skips'], 0)
        self.assertEqual(guard.state['optimizer_steps'], 2)
        json.dumps(events, allow_nan=False)

    def test_amp_nan_skips_safely(self):
        p, opt, scaler, guard = self.setup_step()
        self.backward(p, opt, scaler); p.grad[0] = float('nan')
        before = p.detach().clone()
        event = guard.finish_step(opt, scaler, epoch=0, batch=1, emit=lambda e: None)
        self.assertTrue(torch.equal(p, before)); self.assertEqual(opt.state, {})
        self.assertEqual(event['gradient_norm_status'], 'nan'); self.assertEqual(scaler.get_scale(), 4.)

    def test_finite_large_gradients_have_safe_fp64_norm_and_clip(self):
        p, opt, scaler, guard = self.setup_step(amp=False)
        p.grad = torch.tensor([1e30, 1e30])
        self.assertFalse(bool(torch.isfinite(torch.linalg.vector_norm(p.grad))))
        event = guard.finish_step(opt, scaler, epoch=0, batch=1, emit=lambda e: None)
        self.assertTrue(event['optimizer_step_applied']); self.assertFalse(event['skipped_step'])
        self.assertTrue(bool(torch.isfinite(p).all())); self.assertLessEqual(float(p.grad.norm()), 5.00001)

    def test_fp32_nonfinite_stops_without_update(self):
        p, opt, scaler, guard = self.setup_step(amp=False)
        p.grad = torch.tensor([float('inf'), 1.]); before = p.detach().clone(); events = []
        with self.assertRaisesRegex(NumericalInstabilityError, 'FP32'):
            guard.finish_step(opt, scaler, epoch=0, batch=1, emit=events.append)
        self.assertTrue(torch.equal(p, before)); self.assertEqual(opt.state, {}); self.assertIsNone(p.grad)
        self.assertTrue(events[-1]['skipped_step'])

    def test_repeated_overflows_stop_after_safe_scaler_updates(self):
        p, opt, scaler, guard = self.setup_step(max_consecutive_gradient_skips=2)
        before = p.detach().clone(); events = []
        for batch in (1, 2):
            self.backward(p, opt, scaler); p.grad[0] = float('inf')
            if batch == 1:
                guard.finish_step(opt, scaler, epoch=0, batch=batch, emit=events.append)
            else:
                with self.assertRaisesRegex(NumericalInstabilityError, 'Repeated AMP'):
                    guard.finish_step(opt, scaler, epoch=0, batch=batch, emit=events.append)
        self.assertTrue(torch.equal(p, before)); self.assertEqual(scaler.get_scale(), 2.)
        self.assertEqual(guard.state['skipped_steps'], 2)
        restored = GradientGuard(guard.config, guard.state_dict())
        self.assertEqual(restored.state, guard.state)

    def test_diagnostic_does_not_update_finite_gradients(self):
        p, opt, scaler, guard = self.setup_step()
        before = p.detach().clone(); self.backward(p, opt, scaler)
        event = guard.finish_step(opt, scaler, epoch=0, batch=1, emit=lambda e: None, execute_step=False)
        self.assertTrue(torch.equal(p, before)); self.assertEqual(opt.state, {})
        self.assertFalse(event['optimizer_step_applied'])


class ResumeTests(unittest.TestCase):
    def fixture(self, folder):
        config = Config(train_per_class=24)
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=config.head_lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=.5, patience=1)
        scaler = torch.amp.GradScaler('cpu', init_scale=8., growth_interval=100)
        sampler = EpochSampler(FixtureIndex(), per_class=24, seed=config.seed)
        guard = GradientGuard(config)
        history = []
        for epoch in range(3):
            # A few scalar-parameter checks, no CNN, images or dataset training.
            guard.begin_epoch(epoch); opt.zero_grad(set_to_none=True)
            scaler.scale(sum(p.sum() for p in model.parameters())).backward()
            guard.finish_step(opt, scaler, epoch=epoch, batch=1, emit=lambda e: None)
            sampler.select(epoch)
            score = (.7, .8, .79)[epoch]
            if epoch >= 2: scheduler.step(score)
            history.append({'epoch': epoch+1, 'dev': {'roc_auc': (.7, .8, .828)[epoch]}, 'train': {'n': 48}})
            saved_config = config.as_dict()
            saved_config.pop('max_consecutive_gradient_skips'); saved_config.pop('max_gradient_skips_per_epoch')
            payload = {'protocol': PROTOCOL, 'config': saved_config, 'index_sha256': 'fixture-index',
                'epoch': epoch, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'sampler': sampler.state_dict(),
                'rng': rng_state(), 'history': copy.deepcopy(history), 'best': .7 if epoch == 0 else .8,
                'best_epoch': 1 if epoch == 0 else 2, 'stale': 1 if epoch == 2 else 0,
                'best_balanced_dev_loss': .4}
            if epoch == 1: torch.save(payload, folder/'best.pt')
            if epoch == 2: torch.save(payload, folder/'last.pt')
        return config, payload, sampler

    def restore(self, bundle, config):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=config.head_lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=.5, patience=1)
        scaler = torch.amp.GradScaler('cpu', enabled=config.amp, init_scale=1024.)
        sampler = EpochSampler(FixtureIndex(), per_class=24, seed=config.seed)
        restore_training_state(bundle, model, opt, scheduler, scaler, sampler)
        return model, opt, scheduler, scaler, sampler

    def test_epoch4_resume_restores_all_state_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config, payload, sampler = self.fixture(root)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.pt')}
            expected_random = (random.random(), float(np.random.rand()), torch.rand(3))
            expected_ids, expected_diag = sampler.select(3)
            bundle = load_resume_bundle(root/'last.pt', config, 'fixture-index', expected_completed_epoch=3)
            model, opt, scheduler, scaler, restored = self.restore(bundle, config)
            self.assertEqual(bundle['report']['next_epoch'], 4)
            self.assertEqual(bundle['state']['stale'], 1); self.assertEqual(bundle['state']['best_epoch'], 2)
            self.assertTrue(equal_tree(model.state_dict(), payload['model']))
            self.assertTrue(equal_tree(opt.state_dict(), payload['optimizer']))
            self.assertTrue(equal_tree(scheduler.state_dict(), payload['scheduler']))
            self.assertTrue(equal_tree(scaler.state_dict(), payload['scaler']))
            self.assertEqual(restored.select(3), (expected_ids, expected_diag))
            self.assertEqual(random.random(), expected_random[0]); self.assertEqual(float(np.random.rand()), expected_random[1])
            self.assertTrue(torch.equal(torch.rand(3), expected_random[2]))
            copy_resume_boundary(bundle, root/'new_run')
            for name, value in hashes.items():
                self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), value)
                self.assertEqual((root/name).read_bytes(), (root/'new_run'/name).read_bytes())
            with self.assertRaises(FileExistsError): copy_resume_boundary(bundle, root/'new_run')

    def test_amp_to_fp32_resume_keeps_weights_optimizer_rng_sampler(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config, payload, sampler = self.fixture(root)
            fp32 = replace(config, amp=False)
            bundle = load_resume_bundle(root/'last.pt', fp32, 'fixture-index')
            model, opt, scheduler, scaler, restored = self.restore(bundle, fp32)
            self.assertFalse(scaler.is_enabled()); self.assertEqual(scaler.state_dict(), {})
            self.assertTrue(equal_tree(model.state_dict(), payload['model']))
            self.assertTrue(equal_tree(opt.state_dict(), payload['optimizer']))
            self.assertEqual(restored.state_dict(), payload['sampler'])
            self.assertEqual(bundle['report']['saved_amp_scale'], 8.)
            self.assertTrue(bundle['report']['precision_changed'])
            self.assertEqual(bundle['state']['scaler'], payload['scaler'])

    def test_wrong_epoch_index_or_learning_rate_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config, _, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'Expected completed Epoch 4'):
                inspect_resume_checkpoint(root/'last.pt', config, 'fixture-index', 4)
            with self.assertRaisesRegex(ValueError, 'exact saved candidate index'):
                load_resume_bundle(root/'last.pt', config, 'wrong-index')
            with self.assertRaisesRegex(ValueError, 'head_lr'):
                load_resume_bundle(root/'last.pt', replace(config, head_lr=2e-4), 'fixture-index')

    def test_nonfinite_saved_optimizer_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config, payload, _ = self.fixture(root)
            next(iter(payload['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
            torch.save(payload, root/'last.pt')
            with self.assertRaisesRegex(ValueError, 'non-finite tensor: last.optimizer'):
                load_resume_bundle(root/'last.pt', config, 'fixture-index')

    def test_subsequent_fp32_resume_preserves_numerical_counters(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config, payload, _ = self.fixture(root)
            payload['inactive_amp_scaler_state'] = copy.deepcopy(payload['scaler'])
            payload['scaler'] = {}; payload['config']['amp'] = False
            payload['numerics'] = {'optimizer_steps': 3, 'skipped_steps': 2, 'consecutive_skips': 0,
                'max_consecutive_skips': 1, 'epoch': 2, 'epoch_optimizer_steps': 1, 'epoch_skipped_steps': 0}
            torch.save(payload, root/'last.pt')
            fp32 = replace(config, amp=False)
            bundle = load_resume_bundle(root/'last.pt', fp32, 'fixture-index')
            guard = GradientGuard(fp32, bundle['state']['numerics']); guard.begin_epoch(3)
            self.assertEqual(guard.state['optimizer_steps'], 3); self.assertEqual(guard.state['skipped_steps'], 2)
            self.assertEqual(guard.state['epoch_optimizer_steps'], 0)
            self.assertEqual(bundle['state']['inactive_amp_scaler_state']['scale'], 8.)
            self.assertFalse(self.restore(bundle, fp32)[3].is_enabled())
            with self.assertRaisesRegex(ValueError, 'not FP32 -> AMP'):
                load_resume_bundle(root/'last.pt', config, 'fixture-index')


if __name__ == '__main__':
    unittest.main()
