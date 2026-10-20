"""Lightweight progress/persistence tests; no CNN training or optimizer steps."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from cnn_experiment.config import Config
from cnn_experiment.data import LazyDataset, make_loader
from cnn_experiment.data import Index
from cnn_experiment.training import RunProgress, should_log_batch
from cnn_experiment.training import commit_epoch, run_epoch


class FakeDataset:
    records = [{'id': i, 'source_file': 'dev/synthetic.mp4'} for i in range(6)]

    def __len__(self):
        return 6


class FakeLoader:
    dataset = FakeDataset()
    num_workers = 0

    def __len__(self):
        return 3

    def __iter__(self):
        for i in range(3):
            yield torch.zeros(2, 1), torch.tensor([0., 1.]), torch.tensor([2*i, 2*i+1])


class ConstantLogits:
    def train(self, training):
        assert not training  # tests execute evaluation only, never local training

    def __call__(self, images):
        return torch.tensor([[-1.], [1.]])


def entry(epoch=1, improved=True):
    metrics = {'loss': .4, 'roc_auc': .9, 'n': 6, 'balanced_loss': .5,
               'real_class_loss': .3, 'fake_class_loss': .7,
               'precision': .8, 'recall': .7, 'f1': .75}
    return {'epoch': epoch, 'train': metrics, 'dev': metrics, 'dev_video': metrics,
            'learning_rates': {'head': 1e-4, 'final_stage': 1e-5}, 'phase': 'head',
            'train_seconds': 1., 'dev_seconds': 2., 'is_best': improved, 'stale_epochs': epoch-1}


class ProgressTests(unittest.TestCase):
    def test_heartbeat_reports_a_stalled_stage(self):
        heard = threading.Event()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            progress = RunProgress(folder, 'resnet50', .02)
            original = progress.note
            def note(message):
                original(message)
                if 'heartbeat' in message:
                    heard.set()
            progress.note = note
            with progress:
                progress.stage('train_fetch_batch', epoch=1, batch=1)
                self.assertTrue(heard.wait(2), 'Heartbeat did not report blocked main thread')

    def test_intervals_first_periodic_last(self):
        self.assertEqual([n for n in range(1, 251) if should_log_batch(n, 250, 50)], [1, 50, 100, 150, 200, 250])
        self.assertEqual([n for n in range(1, 1306) if should_log_batch(n, 1305, 200)], [1, 200, 400, 600, 800, 1000, 1200, 1305])

    def test_exception_and_interrupt_are_logged_and_reraised(self):
        for error in (RuntimeError('decode failure'), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(type(error)):
                    with RunProgress(folder, 'resnet50', 0) as progress:
                        progress.stage('train_fetch_batch', epoch=1, batch=1)
                        raise error
                state = json.loads((Path(folder)/'run_status.json').read_text())
                self.assertEqual(state['failed_at']['stage'], 'train_fetch_batch')
                self.assertTrue((Path(folder)/'failure_traceback.txt').is_file())

    def test_loader_uses_spawn_and_timeout_only_for_workers(self):
        records = [{'id': 1, 'split': 'train'}]
        with patch('cnn_experiment.data.DataLoader') as constructor:
            make_loader(records, '.', Config(workers=2))
            kwargs = constructor.call_args.kwargs
            self.assertEqual(kwargs['multiprocessing_context'], 'spawn')
            self.assertEqual(kwargs['timeout'], 120.)
            make_loader(records, '.', Config(workers=0))
            kwargs = constructor.call_args.kwargs
            self.assertNotIn('timeout', kwargs)
            self.assertNotIn('multiprocessing_context', kwargs)

    def test_decode_failure_keeps_candidate_and_video_context(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = LazyDataset([{'id': 23, 'split': 'train', 'source_file': 'train/missing.mp4',
                                    'indices': [0], 'timestamps_exact': ['0']}], folder)
            with self.assertRaisesRegex(RuntimeError, r'candidate IDs \[23\]') as error:
                dataset[0]
            self.assertIn('train/missing.mp4', str(error.exception.__cause__))
            self.assertIsNotNone(error.exception.__cause__.__cause__)

    @unittest.skipUnless(Path('outputs/experiment1_v2_20261019_metadata/candidates.sqlite').is_file()
                         and Path('LAV-DF-5000/train').is_dir(), 'Local TRAIN video/index fixture not attached')
    def test_real_video_main_and_spawned_loading_match(self):
        index = Index('outputs/experiment1_v2_20261019_metadata/candidates.sqlite', 'LAV-DF-5000')
        ids = [r[0] for r in index.db.execute("SELECT f.id FROM frames f JOIN videos v ON f.video_id=v.id WHERE v.split='train' ORDER BY f.id LIMIT 8")]
        records = index.records('train', 'frame', ids); index.close()
        loaded = []
        for workers in (0, 2):
            loader = make_loader(records, 'LAV-DF-5000', Config(batch_size=4, workers=workers), training=True)
            loaded.append(torch.cat([images for images, _, _ in loader]))
        self.assertEqual(loaded[0].shape, (8, 3, 224, 224))
        self.assertTrue(torch.equal(*loaded))

    def test_history_and_last_survive_plot_error(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as stdout:
            out = Path(folder); record = entry()
            with patch('cnn_experiment.training.plot_history', side_effect=RuntimeError('plot failed')):
                with self.assertRaisesRegex(RuntimeError, 'plot failed'):
                    with RunProgress(out, 'resnet50', 0) as progress:
                        commit_epoch(out, [record], record, {'best_epoch': 1, 'epoch': 0},
                                     True, [], [], {}, Config(), progress)
            for name in ('history.csv', 'history.jsonl', 'last.pt', 'best.pt', 'failure_traceback.txt'):
                self.assertTrue((out/name).is_file(), name)
            state = json.loads((out/'run_status.json').read_text())
            self.assertEqual(state['failed_at']['stage'], 'plot_history')
            self.assertIn('Epoch 1/10 completed', stdout.getvalue())
            self.assertIn('Checkpoint saved:', stdout.getvalue())

    def test_each_epoch_commits_and_retains_best(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as stdout:
            out = Path(folder); history = []
            with patch('cnn_experiment.training.plot_history'), patch('cnn_experiment.training.prediction_artifacts') as predictions, \
                    patch('cnn_experiment.training.write_epoch_diagnostics', return_value=[]) as diagnostics:
                with RunProgress(out, 'efficientnet_b0', 0) as progress:
                    for epoch in (1, 2):
                        row = entry(epoch, epoch == 1); history.append(row)
                        commit_epoch(out, history, row, {'epoch': epoch, 'best_epoch': 1},
                                     epoch == 1, [], [], {}, Config(model='efficientnet_b0'), progress)
                self.assertEqual(predictions.call_count, 1)
                self.assertEqual(diagnostics.call_count, 2)
            self.assertEqual(len((out/'history.jsonl').read_text().splitlines()), 2)
            self.assertEqual(torch.load(out/'best.pt', weights_only=True)['epoch'], 1)
            self.assertEqual(torch.load(out/'last.pt', weights_only=True)['epoch'], 2)
            self.assertIn('EfficientNet-B0 | Epoch 2/10 completed', stdout.getvalue())

    def test_eval_loop_live_messages_without_gpu_or_training(self):
        # Constant synthetic logits exercise metrics/logging only. CUDA transfers
        # are bypassed; no CNN, backward pass or optimizer exists in this test.
        def cpu_transfer(tensor, *args, **kwargs):
            return tensor
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as stdout:
            with patch.object(torch.Tensor, 'to', cpu_transfer), patch('cnn_experiment.training.torch.autocast', return_value=contextlib.nullcontext()):
                with RunProgress(folder, 'resnet50', 0) as progress:
                    metrics, predictions, _ = run_epoch(ConstantLogits(), FakeLoader(), Config(batch_size=2, dev_progress_every=2), progress=progress)
            self.assertEqual(metrics['n'], 6)
            self.assertEqual(len(predictions), 6)
            for n in (1, 2, 3):
                self.assertIn(f'DEV evaluation: Batch {n}/3', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
