"""Offline persistent-storage contract tests. Never call Kaggle or train a model."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import os
import types

from cnn_experiment.backup import BackupError, NewSessionRequired, PrivateBackup, retrieve, verify_archive
from cnn_experiment.config import Config, EXPERIMENTS, experiment_config, experiment_plan
from cnn_experiment.training import train
from cnn_experiment.evaluation import dev_threshold_analysis
from cnn_experiment.backup import package_model_run, create_download_test, confirm_download


class MemoryRemote:
    """Remote bytes survive replacement of the client and simulated runtime."""
    owner = 'leezile'

    def __init__(self):
        self.versions = []; self.private = True; self.fail_upload = False; self.corrupt = False

    def info(self, handle):
        return {'private': self.private, 'version': len(self.versions)} if self.versions else None

    def upload(self, handle, folder, create, note):
        if self.fail_upload:
            raise BackupError('Synthetic network failure')
        if create != (not self.versions):
            raise AssertionError('Unexpected create/version request')
        names = {p.name for p in Path(folder).iterdir()}
        assert names == {'snapshot.zip', 'manifest.json', 'dataset-metadata.json'}
        self.versions.append({p.name: p.read_bytes() for p in Path(folder).iterdir()})

    def download(self, handle, version, filename, folder):
        p = Path(folder)/filename
        content = self.versions[version-1][filename]
        p.write_bytes(content+b'corrupt' if self.corrupt and filename == 'snapshot.zip' else content)
        return p


class BackupTests(unittest.TestCase):
    def manager(self, remote, folder, runtime):
        return PrivateBackup(remote, 'leezile/dsa4266-cnn-backups', folder,
            {'resnet50': Config().as_dict()}, 'indexhash', {'commit': 'synthetic'},
            runtime=runtime, poll_seconds=0, timeout_seconds=0)

    def ready(self, remote, folder):
        first = self.manager(remote, Path(folder)/'session1', 'container1')
        with self.assertRaises(NewSessionRequired):
            first.preflight()
        self.assertFalse(first.ready)
        return self.manager(remote, Path(folder)/'session2', 'container2').preflight()

    def artifacts(self, folder):
        out = Path(folder)/'run'; out.mkdir()
        for name in ('last.pt', 'best.pt', 'history.csv', 'history.jsonl', 'config.json', 'numerics.jsonl'):
            (out/name).write_text('synthetic '+name)
        for name in ('candidates.sqlite', 'video.mp4', 'kaggle.json', 'pretrained.pth', 'extra.ipynb'):
            (out/name).write_text('must never upload')
        return out

    def test_probe_requires_different_runtime_not_kernel_rerun(self):
        remote = MemoryRemote()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            first = self.manager(remote, folder, 'container1')
            with self.assertRaises(NewSessionRequired): first.preflight()
            with self.assertRaises(NewSessionRequired): self.manager(remote, folder, 'container1').preflight()
            new = self.manager(remote, Path(folder)/'new', 'container2').preflight()
            self.assertTrue(new.ready); self.assertEqual(len(remote.versions), 2)

    def test_epoch_recoverable_from_new_client_whitelist_and_old_version_preserved(self):
        remote = MemoryRemote()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            manager = self.ready(remote, folder); out = self.artifacts(folder)
            receipt = manager.save_run(out, Config(), 1)
            self.assertEqual(receipt['version'], 3)
            old = remote.versions[2]['snapshot.zip']
            (out/'last.pt').write_bytes(b'epoch2')
            manager.save_run(out, Config(), 2)
            new_client = MemoryRemote(); new_client.versions = remote.versions
            manifest = retrieve(new_client, manager.handle, 3, Path(folder)/'restored')
            self.assertEqual((Path(folder)/'restored/last.pt').read_text(), 'synthetic last.pt')
            self.assertEqual(remote.versions[2]['snapshot.zip'], old)
            self.assertEqual(set(manifest['files']) & {'candidates.sqlite', 'video.mp4', 'kaggle.json', 'pretrained.pth', 'extra.ipynb'}, set())
            with self.assertRaises(FileExistsError): retrieve(new_client, manager.handle, 3, Path(folder)/'restored')

    def test_upload_failure_preserves_previous_versions_and_local_epoch(self):
        remote = MemoryRemote()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            manager = self.ready(remote, folder); out = self.artifacts(folder)
            manager.save_run(out, Config(), 1); before = list(remote.versions)
            remote.fail_upload = True
            with self.assertRaises(BackupError): manager.save_run(out, Config(), 2)
            self.assertEqual(before, remote.versions); self.assertTrue((out/'last.pt').is_file())

    def test_public_dataset_rejected_before_upload(self):
        remote = MemoryRemote()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            manager = self.ready(remote, folder); out = self.artifacts(folder)
            remote.private = False; n = len(remote.versions)
            with self.assertRaises(BackupError): manager.save_run(out, Config(), 1)
            self.assertEqual(n, len(remote.versions))

    def test_corrupt_download_and_wrong_plan_block_training(self):
        remote = MemoryRemote()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            manager = self.ready(remote, folder); out = self.artifacts(folder)
            manager.save_run(out, Config(), 1); remote.corrupt = True
            with self.assertRaises(BackupError): retrieve(remote, manager.handle, 3, Path(folder)/'corrupt')
            self.assertFalse((Path(folder)/'corrupt').exists())
            with self.assertRaises(BackupError): manager.require_ready(replace(Config(), dropout=.4), 'indexhash')
            with self.assertRaises(BackupError): manager.require_ready(Config(), 'otherhash')

    def test_default_training_has_no_remote_auth_or_index_backup_blocker(self):
        with patch('cnn_experiment.training.require_cuda', side_effect=RuntimeError('CUDA guard reached')), \
             patch('cnn_experiment.backup.KaggleBackend', side_effect=AssertionError('Authentication attempted')), \
             patch('cnn_experiment.training.digest', side_effect=AssertionError('Backup index access')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA guard reached'):
                train(Config(), 'unused', 'unused', 'unused', 'unused')

    def test_restore_rejects_path_traversal_before_writing(self):
        from cnn_experiment.backup import SCHEMA, sha256
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z: z.writestr('../escape', b'bad')
            manifest = {'schema': SCHEMA, 'archive_sha256': sha256(archive),
                        'files': {'../escape': hashlib.sha256(b'bad').hexdigest()}}
            with self.assertRaises(BackupError): verify_archive(archive, manifest)


class ExperimentTests(unittest.TestCase):
    def test_checkpoint_selector_changes_score_without_changing_scheduler_lr_recipe(self):
        from cnn_experiment.config import checkpoint_score
        report_a = {'video': {'roc_auc': .9}, 'unit': {'pr_auc': .2}}
        report_b = {'video': {'roc_auc': .8}, 'unit': {'pr_auc': .4}}
        self.assertGreater(checkpoint_score(report_a, 'video_roc_auc'), checkpoint_score(report_b, 'video_roc_auc'))
        self.assertLess(checkpoint_score(report_a, 'frame_pr_auc'), checkpoint_score(report_b, 'frame_pr_auc'))
        cfg = experiment_config('resnet50', checkpoint_selection='frame_pr_auc')
        self.assertEqual(cfg.head_lr, Config().head_lr); self.assertEqual(cfg.backbone_lr, Config().backbone_lr)
        plan = experiment_plan(cfg, 'baseline')
        self.assertTrue(plan['selection_change_from_historical'])
        self.assertEqual(plan['scheduler_monitor'], 'DEV video ROC-AUC')
        self.assertIn('frame PR-AUC', plan['checkpoint_selection'])
        with self.assertRaises(ValueError): replace(Config(), checkpoint_selection='bad').validate()

    def test_final_plot_helpers_without_training(self):
        from cnn_experiment.evaluation import plot_history, write_final_numerics
        from tests.test_training_progress import entry
        import matplotlib.pyplot as plt
        created = []; original = plt.subplots
        def capture(*args, **kwargs):
            fig, axes = original(*args, **kwargs); created.append(fig)
            return fig, axes
        with tempfile.TemporaryDirectory() as folder, patch.object(plt, 'subplots', side_effect=capture) as figures:
            row = entry(); row['dev']['pr_auc'] = .4
            row['train']['numerics'] = {'gradient_norm_max_finite': 3., 'amp_scale_end': 128., 'skipped_steps': 1}
            plot_history(folder, [row]); write_final_numerics(folder, [row])
            self.assertTrue((Path(folder)/'training_curves.png').is_file())
            self.assertTrue((Path(folder)/'numerical_curves.png').is_file())
            self.assertEqual(figures.call_count, 2)
            for fig in created:
                for ax in fig.axes:
                    self.assertEqual(ax.get_ylim()[0], 0.)
                    self.assertEqual(ax.get_xlim()[0], 0.)

    def test_one_factor_recipes_and_unchanged_baseline(self):
        for profile, changes in EXPERIMENTS.items():
            model = 'efficientnet_b0' if profile.startswith('efficientnet') else 'resnet50'
            config = experiment_config(model, profile)
            baseline = Config(model=model).as_dict()
            delta = {k: v for k, v in config.as_dict().items() if baseline[k] != v}
            self.assertEqual(delta, changes)
            self.assertIn('DEV frame PR-AUC' if profile.startswith('fixed20k') else 'DEV video ROC-AUC', experiment_plan(config, profile)['checkpoint_selection'])
        self.assertEqual(experiment_config('resnet50').as_dict(), Config().as_dict())
        with self.assertRaises(ValueError): experiment_config('resnet50', 'efficientnet_20_epochs')

    def test_threshold_sweep_ties_matches_brute_force_and_preserves_auc(self):
        from cnn_experiment.evaluation import binary_metrics
        y = [0, 1, 1, 0, 0, 1]; p = [.1, .4, .4, .8, 1., 1.]
        result = dev_threshold_analysis(y, p)
        rows = [binary_metrics(y, p, t) for t in sorted(set(p))]
        for metric in ('f1', 'f2'):
            self.assertAlmostEqual(result['max_'+metric][metric], max(r[metric] for r in rows))
        self.assertEqual(result['reference']['fp'], 2)
        for row in result['fpr_tradeoffs']:
            self.assertLessEqual(row['operating_point']['false_positive_rate'], row['max_fpr'])
        first, last = binary_metrics(y, p, .1), binary_metrics(y, p, .9)
        self.assertEqual(first['roc_auc'], last['roc_auc']); self.assertEqual(first['pr_auc'], last['pr_auc'])


class ManualZipTests(unittest.TestCase):
    def test_tiny_zip_confirmation_requires_code_from_own_session(self):
        import re
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder)/'session'; session.mkdir()
            report = create_download_test(session)
            self.assertLess(report['bytes'], 2048)
            with zipfile.ZipFile(report['archive']) as z:
                code = re.search(r': ([A-F0-9]{8})', z.read('READ_ME.txt').decode()).group(1)
            with self.assertRaises(ValueError): confirm_download(session, report, 'wrong')
            self.assertFalse((session/'manual_download_verified.json').exists())
            with self.assertRaises(ValueError): confirm_download(Path(folder)/'another', report, code)
            self.assertTrue(confirm_download(session, report, code)['confirmed'])

    def test_rolling_and_separate_final_archives_include_weights_results_and_exclude_inputs(self):
        from cnn_experiment.backup import sha256
        with tempfile.TemporaryDirectory() as folder:
            out = BackupTests().artifacts(folder)
            for name in ('training_curves.png', 'best_dev_predictions.csv', 'best_dev_metrics.json'):
                (out/name).write_bytes(b'final artifact')
            first = package_model_run(out, 'resnet50', 1)
            final = package_model_run(out, 'resnet50', 1, final=True)
            with zipfile.ZipFile(final['archive']) as z:
                names = set(z.namelist())
                self.assertTrue({'best.pt','last.pt','history.csv','config.json','training_curves.png',
                                 'best_dev_predictions.csv','best_dev_metrics.json','backup_manifest.json'} <= names)
                self.assertFalse({'candidates.sqlite','kaggle.json','video.mp4','pretrained.pth'} & names)
                manifest = json.loads(z.read('backup_manifest.json'))
                import hashlib
                for name, expected in manifest['files'].items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), expected)
            final_bytes = Path(final['archive']).read_bytes()
            (out/'last.pt').write_bytes(b'epoch2')
            second = package_model_run(out, 'resnet50', 2)
            self.assertEqual(first['archive'], second['archive'])
            self.assertEqual(final_bytes, Path(final['archive']).read_bytes())
            self.assertEqual(sha256(second['archive']), second['sha256'])
            with zipfile.ZipFile(second['archive']) as z:
                self.assertEqual(z.read('last.pt'), b'epoch2')
                self.assertEqual(json.loads(z.read('backup_manifest.json'))['completed_epoch'], 2)
            eff = package_model_run(out, 'efficientnet_b0', 2, final=True)
            self.assertNotEqual(eff['archive'], final['archive'])

    def test_zip_failure_preserves_previous_rolling_zip_and_original_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            out = BackupTests().artifacts(folder)
            receipt = package_model_run(out, 'resnet50', 1)
            old = Path(receipt['archive']).read_bytes(); source = (out/'last.pt').read_bytes()
            with patch.object(zipfile.ZipFile, 'testzip', return_value='bad-member'):
                with self.assertRaises(OSError): package_model_run(out, 'resnet50', 2)
            self.assertEqual(old, Path(receipt['archive']).read_bytes())
            self.assertEqual(source, (out/'last.pt').read_bytes())

    def test_epoch_commit_runs_without_private_backup_and_creates_zip(self):
        from cnn_experiment.training import RunProgress, commit_epoch
        from tests.test_training_progress import entry
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch('cnn_experiment.training.prediction_artifacts'), \
             patch('cnn_experiment.backup.KaggleBackend', side_effect=AssertionError('Authentication attempted')):
            out = BackupTests().artifacts(folder); row = entry()
            with RunProgress(out, 'resnet50', 0) as progress:
                commit_epoch(out, [row], row, {'best_epoch':1}, True, [], [], {}, Config(), progress)
            self.assertTrue((out.parent/'downloads/resnet50_latest_completed_epoch.zip').is_file())


@unittest.skipUnless(__import__('importlib').util.find_spec('requests'), 'Optional remote adapter HTTP dependency absent')
class BackendSecurityTests(unittest.TestCase):
    def test_secret_is_separate_owner_checked_environment_restored_errors_sanitized(self):
        from cnn_experiment.backup import KaggleBackend
        sentinel = 'synthetic-kaggle-secret-never-show'
        from unittest.mock import Mock
        api = Mock(); api.CONFIG_NAME_USER = 'username'; api.CONFIG_NAME_TOKEN = 'token'
        api.get_config_value.side_effect = lambda k: {'username': 'different_account', 'token': sentinel}[k]
        module = types.ModuleType('kaggle.api.kaggle_api_extended'); module.KaggleApi = Mock(return_value=api)
        kaggle_package = types.ModuleType('kaggle'); kaggle_package.__version__ = '2.2.4'
        sdk_package = types.ModuleType('kagglesdk'); sdk_package.__version__ = '0.1.37'
        api_package = types.ModuleType('kaggle.api')
        secrets = types.ModuleType('kaggle_secrets'); secrets.UserSecretsClient = Mock()
        client = secrets.UserSecretsClient.return_value; client.get_secret.return_value = sentinel
        output = io.StringIO()
        with patch.dict('sys.modules', {'kaggle': kaggle_package, 'kagglesdk': sdk_package,
             'kaggle.api': api_package, 'kaggle.api.kaggle_api_extended': module, 'kaggle_secrets': secrets}), \
             patch('importlib.metadata.version', side_effect=lambda k: {'kaggle': '2.2.4', 'kagglesdk': '0.1.37'}[k]), \
             patch.dict(os.environ, {'KAGGLE_API_TOKEN': 'existing-synthetic-value'}), contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(BackupError, 'account does not match'): KaggleBackend()
            self.assertEqual(os.environ['KAGGLE_API_TOKEN'], 'existing-synthetic-value')
            client.get_secret.assert_called_once_with('KAGGLE_API_TOKEN')
            api.dataset_create_new.assert_not_called()
            api.authenticate.side_effect = RuntimeError(sentinel)
            with self.assertRaises(BackupError) as error: KaggleBackend()
            self.assertNotIn(sentinel, str(error.exception))
            self.assertTrue(error.exception.__suppress_context__)
        self.assertNotIn(sentinel, output.getvalue())

    def test_private_create_versions_and_pinned_uncached_download_arguments(self):
        from cnn_experiment.backup import KaggleBackend
        from unittest.mock import Mock
        backend = object.__new__(KaggleBackend); backend.api = Mock()
        backend.api.dataset_create_new.return_value = types.SimpleNamespace(error='', status='ok')
        backend.api.dataset_create_version.return_value = types.SimpleNamespace(error='', status='ok')
        with tempfile.TemporaryDirectory() as folder:
            backend.upload('leezile/dsa4266-cnn-backups', folder, True, 'probe')
            self.assertIs(backend.api.dataset_create_new.call_args.kwargs['public'], False)
            backend.upload('leezile/dsa4266-cnn-backups', folder, False, 'epoch')
            self.assertIs(backend.api.dataset_create_version.call_args.kwargs['delete_old_versions'], False)
            (Path(folder)/'snapshot.zip').write_bytes(b'synthetic')
            backend.download('leezile/dsa4266-cnn-backups', 7, 'snapshot.zip', folder)
            args = backend.api.dataset_download_file.call_args
            self.assertEqual(args.args[:2], ('leezile/dsa4266-cnn-backups/7', 'snapshot.zip'))
            self.assertTrue(args.kwargs['force']); self.assertEqual(args.kwargs['path'], folder)
            self.assertNotIn('unzip', args.kwargs)  # Not in the released 2.2.4 API.

    def test_large_file_transport_wrapper_is_unwrapped_once_without_extracting_checkpoint_archive(self):
        from cnn_experiment.backup import KaggleBackend
        from unittest.mock import Mock
        backend = object.__new__(KaggleBackend); backend.api = Mock()
        with tempfile.TemporaryDirectory() as folder:
            inner = io.BytesIO()
            with zipfile.ZipFile(inner, 'w') as z: z.writestr('last.pt', b'synthetic')
            payload = inner.getvalue()
            def download(*args, **kwargs):
                with zipfile.ZipFile(Path(folder)/'snapshot.zip.zip', 'w') as z:
                    z.writestr('snapshot.zip', payload)
            backend.api.dataset_download_file.side_effect = download
            path = backend.download('leezile/dsa4266-cnn-backups', 7, 'snapshot.zip', folder)
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse((Path(folder)/'last.pt').exists())
            self.assertFalse((Path(folder)/'snapshot.zip.zip').exists())


if __name__ == '__main__':
    unittest.main()
