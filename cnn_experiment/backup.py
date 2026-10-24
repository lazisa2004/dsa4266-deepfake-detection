"""Private, versioned Kaggle backups. No model/data imports or training on import."""
import contextlib
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
import time
import uuid
import zipfile
import secrets


SCHEMA = 'dsa4266-private-checkpoint-v1'
REQUIRED = ('best.pt', 'last.pt', 'history.csv', 'history.jsonl', 'config.json')
OPTIONAL = ('numerics.jsonl', 'progress.jsonl', 'run_status.json', 'runtime_environment.json',
            'resume_state.json', 'trainable_head.json', 'trainable_final_stage.json',
            'experiment_plan.json', 'training_summary.json', 'generalization_assessment.json',
            'best_dev_metrics.json', 'best_dev_predictions.csv', 'best_dev_curves.json',
            'best_dev_video_predictions.json', 'dev_threshold_analysis.json',
            'training_curves.png', 'numerical_curves.png', 'best_dev_confusion.png', 'best_dev_roc_pr.png',
            'train_pool.json','clip_manifest.json','feature_source.json','dev_f2_calibration.json','dev_f2_reporting_confusion.png',
            'input_pipeline.json','face_boxes.sqlite','face_detection_report.json','face_detector.onnx','face_detector_LICENSE.txt',
            'face_landmarker.task','video_dev_f2_calibration.json','controlled_experiment.json',
            'epoch_metrics.csv','epoch_metrics.json','epoch_metrics.html')


def create_download_test(session):
    """Tiny ZIP with a code to read AFTER downloading/extracting on the laptop."""
    folder = Path(session)/'downloads'; folder.mkdir(parents=True, exist_ok=True)
    archive = folder/('download_test_'+uuid.uuid4().hex[:8]+'.zip')
    code = secrets.token_hex(4).upper()
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('READ_ME.txt', 'DSA4266 download test\n\nDownload this ZIP to your Windows laptop.\n'
            'Right-click it, select Extract All, and open this file locally.\n'
            f'Enter this verification code in notebook section 4b: {code}\n'
            'This is a dummy text file, not model weights.\n')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise OSError('Download-test ZIP failed CRC verification')
    return {'archive': str(archive), 'sha256': sha256(archive), 'bytes': archive.stat().st_size,
            'verification_sha256': hashlib.sha256(code.encode()).hexdigest()}


def confirm_download(session, report, code):
    """User attestation that the downloaded file opened; never infer a download."""
    if Path(report['archive']).resolve().parent != (Path(session)/'downloads').resolve():
        raise ValueError('Download confirmation belongs to a different session; create/download this session\'s test ZIP.')
    actual = hashlib.sha256(str(code).strip().upper().encode()).hexdigest()
    if not secrets.compare_digest(actual, report['verification_sha256']):
        raise ValueError('Code does not match. Download/extract the test ZIP on Windows and read READ_ME.txt.')
    if sha256(report['archive']) != report['sha256']:
        raise ValueError('The test ZIP changed; create and download a new test.')
    proof = {'confirmed': True, 'method': 'user-entered code from downloaded/extracted ZIP',
             'test_archive': report['archive'], 'sha256': report['sha256']}
    path = Path(session)/'manual_download_verified.json'
    path.write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
    return proof


def restore_local_zip(archive,destination):
    """Verify every hash and extract flat run files into a NEW directory only."""
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Restore destination exists; originals preserved')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any('/' in name or '\\' in name or name in ('.','..') for name in names):
            raise ValueError('Expected a flat manual checkpoint archive')
        manifest=json.loads(z.read('backup_manifest.json'))
        if manifest.get('schema')!='dsa4266-manual-zip-v1' or set(names)!=set(manifest['files'])|{'backup_manifest.json'}:
            raise ValueError('Invalid manual backup manifest')
        if not set(REQUIRED)<=set(manifest['files']):raise ValueError('Incomplete checkpoint archive')
        for name,expected in manifest['files'].items():
            h=hashlib.sha256()
            with z.open(name) as member:
                for chunk in iter(lambda:member.read(2**20),b''):h.update(chunk)
            if h.hexdigest()!=expected:raise ValueError('Checkpoint archive hash mismatch')
        if z.testzip() is not None:raise ValueError('Checkpoint archive CRC mismatch')
        destination.mkdir(parents=True,exist_ok=False)
        for name in names:
            with z.open(name) as source,(destination/name).open('xb') as target:shutil.copyfileobj(source,target)
    return destination


def package_model_run(out, model, epoch, *, final=False, snapshot_lock=None):
    """Bounded rolling epoch ZIP; immutable final ZIP. No remote authentication.

    Flat members make best.pt/last.pt directly reusable after extraction. Only
    generated artifacts are eligible. Existing unrelated archives are preserved.
    """
    out = Path(out)
    if out.is_symlink() or not out.is_dir() or model not in ('resnet50', 'efficientnet_b0','resnet50_lstm'):
        raise ValueError('Expected a real CNN output directory')
    for name in REQUIRED:
        if not (out/name).is_file():
            raise FileNotFoundError(f'Cannot package incomplete run: {name}')
    files = {name: out/name for name in REQUIRED+OPTIONAL if (out/name).is_file()}
    files.update({p.name: p for p in out.glob('sampling_epoch_[0-9][0-9][0-9].json')})
    for name in ('git_provenance.json', 'planned_configs.json', 'experiment_plans.json',
                 'historical_baseline.json', 'manual_download_verified.json'):
        if (out.parent/name).is_file():
            files['session_'+name] = out.parent/name
    if any(p.is_symlink() for p in files.values()):
        raise ValueError('Symlink artifacts cannot be packaged')
    destination = out.parent/'downloads'; destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < sum(p.stat().st_size for p in files.values())+64*2**20:
        raise OSError('Not enough free space to create a verified model ZIP; checkpoints remain untouched.')
    filename = (model+'_final_'+uuid.uuid4().hex[:8]+'.zip' if final else model+'_latest_completed_epoch.zip')
    archive = destination/filename; receipt_path = archive.with_suffix('.manifest.json')
    if archive.exists():
        if final or not receipt_path.is_file():
            raise FileExistsError('Existing archive preserved; it is not this run\'s rolling ZIP.')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('source_run') != str(out.resolve()) or receipt.get('model') != model:
            raise FileExistsError('Existing archive belongs to another run; preserved.')
    partial = destination/('.partial_'+uuid.uuid4().hex+'.zip')
    try:
        with (snapshot_lock or contextlib.nullcontext()), zipfile.ZipFile(partial, 'x', zipfile.ZIP_STORED) as z:
            for name, path in sorted(files.items()):
                z.write(path, name)
        hashes = {}
        with zipfile.ZipFile(partial) as z:
            for name in z.namelist():
                h = hashlib.sha256()
                with z.open(name) as member:
                    for block in iter(lambda: member.read(2**20), b''):
                        h.update(block)
                hashes[name] = h.hexdigest()
        manifest = {'schema': 'dsa4266-manual-zip-v1', 'model': model, 'completed_epoch': epoch,
                    'kind': 'final' if final else 'epoch', 'files': hashes}
        with zipfile.ZipFile(partial, 'a') as z:
            z.writestr('backup_manifest.json', json.dumps(manifest, indent=2)+'\n')
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:
                raise OSError('Model ZIP CRC verification failed; previous rolling ZIP retained.')
        receipt = {'archive': str(archive), 'bytes': partial.stat().st_size, 'sha256': sha256(partial),
            'source_run': str(out.resolve()), 'model': model, 'completed_epoch': epoch,
            'kind': manifest['kind'], 'external_download_required': True}
        partial.replace(archive)  # Only the known rolling artifact can be replaced.
        temp_receipt = receipt_path.with_suffix('.json.tmp')
        temp_receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        temp_receipt.replace(receipt_path)
    finally:
        partial.unlink(missing_ok=True)
    return receipt


class BackupError(RuntimeError):
    pass


class NewSessionRequired(BackupError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def runtime_identity():
    """Container identity stays stable across Python kernel restarts."""
    parts = [socket.gethostname()]
    for name in ('/proc/1/stat', '/proc/sys/kernel/random/boot_id'):
        if Path(name).is_file():
            value = Path(name).read_text()
            # Only PID 1 start time is stable; CPU counters change continuously.
            parts.append(value.split(') ', 1)[1].split()[19] if name.endswith('stat') else value.strip())
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()


@contextlib.contextmanager
def quiet_sdk():
    """Discard SDK output and raw HTTP errors; never echo credentials/signed URLs."""
    from requests.sessions import Session
    original_send = Session.send
    def bounded_send(session, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (15, 180)
        return original_send(session, request, **kwargs)
    old = logging.root.manager.disable
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        logging.disable(logging.CRITICAL)
        Session.send = bounded_send
        try:
            yield
        finally:
            Session.send = original_send
            logging.disable(old)


class KaggleBackend:
    """Pinned official client; authenticate and verify account before any write."""
    def __init__(self, owner='leezile', secret_name='KAGGLE_API_TOKEN'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
            raise BackupError('Invalid Kaggle owner. Use the account username, not its display name.')
        try:
            from importlib.metadata import version
            if version('kaggle') != '2.2.4' or version('kagglesdk') != '0.1.37':
                raise BackupError('Install kaggle==2.2.4 and kagglesdk==0.1.37 before backup setup.')
            previous = os.environ.get('KAGGLE_API_TOKEN')
            previous_runtime_token_path = os.environ.get('KAGGLE_API_V1_TOKEN')
            token = None
            try:
                # Existing supported API-token environment/file credentials are tried
                # when no enabled Secret exists. Notebook input access is not proof
                # of dataset upload permission.
                with quiet_sdk():
                    try:
                        from kaggle_secrets import UserSecretsClient
                        token = UserSecretsClient().get_secret(secret_name)
                    except Exception:
                        pass
                    if token:
                        os.environ['KAGGLE_API_TOKEN'] = token
                        # The official SDK normally prioritizes its injected
                        # runtime-token file. An explicit enabled Secret must be
                        # able to replace unavailable/insufficient runtime auth.
                        os.environ.pop('KAGGLE_API_V1_TOKEN', None)
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    import sys
                    for package, pinned in (('kaggle', '2.2.4'), ('kagglesdk', '0.1.37')):
                        loaded = sys.modules.get(package)
                        if loaded is not None and getattr(loaded, '__version__', pinned) != pinned:
                            raise BackupError('A stale Kaggle client is loaded. Restart the Python kernel after installing the pinned clients.')
                    self.api = KaggleApi()
                    self.api.authenticate()
                    authenticated = self.api.get_config_value(self.api.CONFIG_NAME_USER)
                    access_token = self.api.get_config_value(self.api.CONFIG_NAME_TOKEN)
                if not access_token or not authenticated:
                    raise BackupError('A supported Kaggle API access token is required.')
                if authenticated.lower() != owner.lower():
                    raise BackupError('Authenticated Kaggle account does not match the requested backup owner; nothing created.')
                self.owner = authenticated
                self._runtime_credentials = not bool(token)
            finally:
                if previous is None:
                    os.environ.pop('KAGGLE_API_TOKEN', None)
                else:
                    os.environ['KAGGLE_API_TOKEN'] = previous
                if previous_runtime_token_path is not None:
                    os.environ['KAGGLE_API_V1_TOKEN'] = previous_runtime_token_path
                token = None
        except BackupError:
            raise
        except (Exception, SystemExit):
            raise BackupError('Kaggle authentication unavailable. Enable a Kaggle API token Secret named '
                              'KAGGLE_API_TOKEN (separate from Github Token), then rerun setup. No training permitted.') from None

    def _refresh_credentials(self):
        # Runtime-managed token files may rotate during a long session. Reload
        # supported credentials rather than retaining a stale session token.
        if getattr(self, '_runtime_credentials', False):
            self.api.authenticate()
            owner = self.api.get_config_value(self.api.CONFIG_NAME_USER)
            if not owner or owner.lower() != self.owner.lower():
                raise BackupError('Authenticated backup account changed. No upload permitted.')

    def info(self, handle):
        try:
            with quiet_sdk():
                self._refresh_credentials()
                from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
                request = ApiGetDatasetRequest()
                request.owner_slug, request.dataset_slug = handle.split('/')
                with self.api.build_kaggle_client() as client:
                    response = client.datasets.dataset_api_client.get_dataset(request)
            return {'private': response.is_private, 'version': response.current_version_number}
        except BackupError:
            raise
        except (Exception, SystemExit) as error:
            if getattr(getattr(error, 'response', None), 'status_code', None) == 404:
                return None
            raise BackupError('Cannot verify private dataset metadata. Check account permissions and Internet.') from None

    def upload(self, handle, folder, create, note):
        try:
            with quiet_sdk():
                self._refresh_credentials()
                result = (self.api.dataset_create_new(str(folder), public=False, quiet=True,
                           convert_to_csv=False) if create else
                          self.api.dataset_create_version(str(folder), note, quiet=True,
                           convert_to_csv=False, delete_old_versions=False))
            if result.error or str(result.status).lower() == 'error':
                raise BackupError('Kaggle rejected the private backup upload; prior versions retained.')
        except BackupError:
            raise
        except (Exception, SystemExit):
            raise BackupError('Private backup upload failed; training stopped and prior versions retained.') from None

    def download(self, handle, version, filename, folder):
        try:
            with quiet_sdk():
                self._refresh_credentials()
                # A version-pinned HTTP download into a NEW directory, outside
                # kagglehub/input caches. The RELEASED 2.2.4 single-file API has
                # no unzip argument (unlike newer development source).
                self.api.dataset_download_file(f'{handle}/{version}', filename, path=str(folder),
                                              force=True, quiet=True)
            path = Path(folder)/filename
            wrapper = Path(folder)/(filename+'.zip')
            if not path.exists() and wrapper.is_file():
                # Kaggle may wrap a large single file in an extra ZIP. Unwrap
                # exactly the requested member; never unzip snapshot.zip itself.
                with zipfile.ZipFile(wrapper) as z:
                    if z.namelist() != [filename]:
                        raise BackupError('Unexpected members in Kaggle transport wrapper.')
                    with z.open(filename) as source, path.open('xb') as target:
                        shutil.copyfileobj(source, target, length=2**20)
                wrapper.unlink()
            if not path.is_file():
                raise BackupError('Version-pinned backup download did not produce the requested file.')
            return path
        except BackupError:
            raise
        except (Exception, SystemExit):
            raise BackupError('Independent remote backup retrieval failed; no unprotected training permitted.') from None


def verify_archive(archive, manifest):
    if manifest.get('schema') != SCHEMA or sha256(archive) != manifest.get('archive_sha256'):
        raise BackupError('Backup archive/schema verification failed.')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest['files']):
            raise BackupError('Unexpected/duplicate files in backup archive.')
        for name, expected in manifest['files'].items():
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
                raise BackupError('Unsafe backup member path.')
            h = hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda: f.read(2**20), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise BackupError('Backup member hash verification failed.')


def retrieve(backend, handle, version, destination):
    """Restore into a new directory. Verify all bytes BEFORE writing any members."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Restore destination exists; preserved. Choose a new folder.')
    info = backend.info(handle)
    if not info or info['private'] is not True:
        raise BackupError('Refusing retrieval from an unverified/public dataset.')
    with tempfile.TemporaryDirectory(prefix='dsa4266_remote_read_') as temp:
        folder = Path(temp)
        manifest = json.loads(backend.download(handle, version, 'manifest.json', folder).read_text())
        if manifest.get('handle') != handle or manifest.get('version') != version:
            raise BackupError('Remote manifest does not match the requested dataset version.')
        archive = backend.download(handle, version, 'snapshot.zip', folder)
        verify_archive(archive, manifest)
        destination.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive) as z:
            z.extractall(destination)  # flat names validated above
        (destination/'backup_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


class PrivateBackup:
    """One private dataset; each immutable version is one model's epoch boundary.

    Earlier versions (including the other model) are NEVER deleted. Version 1
    is a dummy probe. A different Kaggle runtime must recover it before training.
    """
    def __init__(self, backend, handle, session, plans, index_sha256, provenance,
                 runtime=None, poll_seconds=10, timeout_seconds=1200):
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9-]{6,50}', handle) or handle.split('/')[0].lower() != backend.owner.lower():
            raise BackupError('Backup handle must belong to the authenticated owner.')
        self.backend, self.handle, self.session = backend, handle, Path(session)
        # Freeze the reviewed plan; mutating a notebook dictionary cannot bypass
        # a required new preflight after changing experiment settings.
        self.plans = json.loads(json.dumps(plans))
        self.index_sha256 = index_sha256
        self.provenance = json.loads(json.dumps(provenance))
        self.runtime = runtime or runtime_identity()
        self.poll_seconds, self.timeout_seconds = poll_seconds, timeout_seconds
        self.ready = False

    def _publish(self, files, metadata, create=False, snapshot_lock=None):
        before = self.backend.info(self.handle)
        if create and before is not None:
            raise BackupError('Dataset already exists; refusing to replace it.')
        if not create and (not before or before['private'] is not True):
            raise BackupError('Backup dataset is unavailable or public. No upload/training permitted.')
        expected_version = 1 if create else before['version']+1
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='dsa4266_private_upload_') as temp:
            folder = Path(temp); archive = folder/'snapshot.zip'
            with (snapshot_lock or contextlib.nullcontext()), zipfile.ZipFile(archive, 'x', zipfile.ZIP_STORED) as z:
                for name, path in files.items():
                    if Path(name).name != name or Path(path).is_symlink():
                        raise BackupError('Unsafe backup source.')
                    z.write(path, name)
            # Hash the snapshot bytes, not live log files which heartbeats may append.
            with zipfile.ZipFile(archive) as z:
                hashes = {}
                for name in z.namelist():
                    h = hashlib.sha256()
                    with z.open(name) as f:
                        for block in iter(lambda: f.read(2**20), b''):
                            h.update(block)
                    hashes[name] = h.hexdigest()
            manifest = metadata | {'schema': SCHEMA, 'files': hashes, 'archive_sha256': sha256(archive),
                                   'handle': self.handle, 'version': expected_version}
            (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            (folder/'dataset-metadata.json').write_text(json.dumps({'id': self.handle,
                'title': 'DSA4266 CNN private checkpoint backups', 'licenses': [{'name': 'other'}],
                'description': 'Private research checkpoints. All prior versions retained. No original data.'}))
            self.backend.upload(self.handle, folder, create, f"{metadata['kind']} {metadata.get('model', '')} epoch {metadata.get('epoch', '')}")
            deadline = time.monotonic()+self.timeout_seconds
            while True:
                info = self.backend.info(self.handle)
                if info and info['private'] is not True:
                    raise BackupError('Remote privacy verification failed. Training stopped.')
                if info and info['version'] > expected_version:
                    raise BackupError('Concurrent backup writer detected; stop other notebooks.')
                if info and info['version'] == expected_version:
                    break
                if time.monotonic() >= deadline:
                    raise BackupError('Timed out waiting for a new private dataset version. Training stopped.')
                time.sleep(self.poll_seconds)
            # Processing may expose the version before downloads become available.
            while True:
                try:
                    with tempfile.TemporaryDirectory(prefix='dsa4266_verify_') as recovered:
                        downloaded = retrieve(self.backend, self.handle, expected_version, Path(recovered)/'restored')
                    if downloaded != manifest:
                        raise BackupError('Retrieved manifest differs from the uploaded snapshot.')
                    after = self.backend.info(self.handle)
                    if not after or after['private'] is not True or after['version'] != expected_version:
                        raise BackupError('Dataset privacy/version changed during backup verification.')
                    break
                except BackupError:
                    if time.monotonic() >= deadline:
                        raise BackupError('Remote backup could not be recovered/verified before the deadline. Training stopped.') from None
                    time.sleep(self.poll_seconds)
            receipt = {'handle': self.handle, 'version': expected_version, 'private': True,
                       'archive_sha256': manifest['archive_sha256'], 'bytes': archive.stat().st_size,
                       'seconds': time.monotonic()-started, 'independent_download_verified': True,
                       'model': metadata.get('model'), 'epoch': metadata.get('epoch'), 'kind': metadata['kind']}
        receipts = self.session/'backup_receipts'; receipts.mkdir(parents=True, exist_ok=True)
        with (receipts/f'version_{expected_version:06}.json').open('x') as f:
            json.dump(receipt, f, indent=2)
        print(f"Private backup verified: {self.handle} version {expected_version} | "
              f"{receipt['bytes']/2**20:.1f} MiB | {receipt['seconds']:.1f}s", flush=True)
        return receipt

    def preflight(self):
        self.ready = False
        info = self.backend.info(self.handle)
        if info is None:
            with tempfile.TemporaryDirectory() as temp:
                probe = Path(temp)/'recovery_probe.json'
                probe.write_text(json.dumps({'schema': SCHEMA, 'origin_runtime': self.runtime,
                                            'nonce': uuid.uuid4().hex}))
                self._publish({'recovery_probe.json': probe}, {'kind': 'dummy_recovery_probe'}, create=True)
            raise NewSessionRequired('Private dummy backup uploaded and read back. Stop this Kaggle session; '
                'start a NEW Kaggle session with the same Secrets/inputs and rerun preflight. Training is blocked '
                'until that separate runtime recovers version 1.')
        with tempfile.TemporaryDirectory() as temp:
            restored = Path(temp)/'probe'
            manifest = retrieve(self.backend, self.handle, 1, restored)
            if manifest.get('kind') != 'dummy_recovery_probe':
                raise BackupError('Dataset version 1 is not this workflow\'s dummy probe; preserve it and choose another slug.')
            probe = json.loads((restored/'recovery_probe.json').read_text())
        if probe['origin_runtime'] == self.runtime:
            raise NewSessionRequired('Dummy recovery must be verified from a NEW Kaggle runtime, not a restarted Python kernel. Training blocked.')
        # Prove current-session write AND read permission, without replacing the
        # old probe. This becomes another retained, private dataset version.
        with tempfile.TemporaryDirectory() as temp:
            attestation = Path(temp)/'recovery_attestation.json'
            attestation.write_text(json.dumps({'probe_version': 1, 'different_runtime': True,
                'probe_sha256': manifest['archive_sha256'], 'runtime': self.runtime}))
            self._publish({'recovery_attestation.json': attestation}, {'kind': 'new_session_recovery_verified'})
        self.ready = True
        self.session.mkdir(parents=True, exist_ok=True)
        (self.session/'persistent_backup_preflight.json').write_text(json.dumps({
            'handle': self.handle, 'private': True, 'recovered_version': 1, 'different_runtime': True,
            'probe_archive_sha256': manifest['archive_sha256'], 'plans': self.plans,
            'index_sha256': self.index_sha256, 'provenance': self.provenance}, indent=2)+'\n')
        print(f'Persistent backup ready: authenticated owner {self.backend.owner}; PRIVATE {self.handle}; '
              'dummy recovered and verified in a different Kaggle runtime.', flush=True)
        return self

    def require_ready(self, config, index_sha256):
        if not self.ready or self.index_sha256 != index_sha256 or self.plans.get(config.model) != config.as_dict():
            raise BackupError('Persistent backup preflight missing or experiment/index changed. Rerun preflight; no training permitted.')
        try:
            info = self.backend.info(self.handle)
            if not info or info['private'] is not True:
                raise BackupError('Private persistent backup is no longer available. Training blocked.')
        except BaseException:
            self.ready = False
            raise

    def save_run(self, out, config, epoch, kind='completed_epoch', snapshot_lock=None):
        out = Path(out)
        self.require_ready(config, self.index_sha256)
        for name in REQUIRED:
            if not (out/name).is_file():
                self.ready = False
                raise BackupError(f'Required checkpoint artifact missing: {name}. Training stopped.')
        names = REQUIRED+OPTIONAL
        if kind == 'completed_epoch':
            # Final plots and best-prediction artifacts are generated separately;
            # avoid a previous best's prediction files accompanying a new best.pt.
            names = tuple(n for n in names if not n.startswith('best_dev_') and not n.endswith('.png'))
        files = {name: out/name for name in names if (out/name).is_file()}
        # Include compact, bounded preflight reports, not videos/SQLite or arbitrary trees.
        for name in ('persistent_backup_preflight.json', 'git_provenance.json', 'planned_configs.json'):
            if (self.session/name).is_file():
                files['session_'+name] = self.session/name
        for name in ('candidate_validation.json', 'index_provenance.json', 'sampling_checks.json',
                     'environment.json', 'lightweight_tests.json', 'preflight_complete.json',
                     'cuda_overflow_check.json', 'loader_benchmark.json'):
            path = self.session/'diagnostics'/name
            if path.is_file():
                files['preflight_'+name] = path
        try:
            return self._publish(files, {'kind': kind, 'model': config.model, 'epoch': epoch,
                'config': config.as_dict(), 'index_sha256': self.index_sha256, 'provenance': self.provenance},
                snapshot_lock=snapshot_lock)
        except BaseException:
            self.ready = False
            raise
