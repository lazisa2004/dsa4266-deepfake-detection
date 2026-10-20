"""Private checkout security tests with mocked Git/Secrets; no network/training."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from cnn_experiment.github import checkout_private_repo, REPOSITORY_URL
import json


class GitHubBootstrapTests(unittest.TestCase):
    def exercise(self, fail_fetch=False):
        sentinel = 'synthetic-secret-never-persist'
        sha = 'a' * 40
        calls, helpers, child_envs = [], [], []
        client = types.SimpleNamespace(get_secret=lambda name: sentinel)
        module = types.SimpleNamespace(UserSecretsClient=lambda: client)

        def fake_run(args, *, env, capture_output, text, timeout):
            calls.append(list(args)); child_envs.append(env)
            self.assertTrue(capture_output)
            self.assertNotIn(sentinel, ' '.join(args))
            self.assertEqual(env['GIT_CONFIG_GLOBAL'], os.devnull)
            self.assertNotIn('GIT_TRACE', env)
            self.assertIn('credential.helper=', args)
            if 'fetch' in args:
                helper = Path(env['GIT_ASKPASS']); helpers.append(helper)
                self.assertEqual(env['DSA4266_GITHUB_TOKEN'], sentinel)
                self.assertNotIn(sentinel, helper.read_text())
                # Deliberately make failure output sensitive to test suppression.
                return subprocess.CompletedProcess(args, int(fail_fetch), '', sentinel)
            self.assertNotIn('DSA4266_GITHUB_TOKEN', env)
            stdout = sha if 'rev-parse' in args else REPOSITORY_URL if 'get-url' in args else ''
            return subprocess.CompletedProcess(args, 0, stdout, '')

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict('sys.modules', {'kaggle_secrets': module}), \
                patch.dict(os.environ, {'GIT_TRACE': '1'}), \
                patch('cnn_experiment.github.subprocess.run', side_effect=fake_run), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            if fail_fetch:
                with self.assertRaises(RuntimeError) as caught:
                    checkout_private_repo(Path(folder)/'fresh', sha)
                self.assertNotIn(sentinel, str(caught.exception))
            else:
                result = checkout_private_repo(Path(folder)/'fresh', sha)
                self.assertEqual(result['commit'], sha)
                self.assertEqual(result['repository'], REPOSITORY_URL)
                self.assertNotIn(sentinel, str(result))
        self.assertTrue(helpers)
        self.assertTrue(all(not p.exists() for p in helpers))
        self.assertTrue(all('DSA4266_GITHUB_TOKEN' not in env for env in child_envs))
        self.assertNotIn(sentinel, output.getvalue())

    def test_token_never_persisted_or_printed(self):
        self.exercise()

    def test_failure_output_suppressed_and_token_cleaned(self):
        self.exercise(fail_fetch=True)

    def test_ref_and_existing_folder_rejected_before_git(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('cnn_experiment.github.subprocess.run') as run:
            with self.assertRaises(FileExistsError):
                checkout_private_repo(folder)
            with self.assertRaises(ValueError):
                checkout_private_repo(Path(folder)/'fresh', '--upload-pack=bad')
            run.assert_not_called()

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'experiment1_cnn_v2_vscode_kaggle.ipynb').is_file(),
                         'The main notebook is intentionally local-only')
    def test_local_notebook_embeds_bootstrap_and_defaults_fresh(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/'experiment1_cnn_v2_vscode_kaggle.ipynb').read_text(encoding='utf-8'))
        first = next(c['source'] for c in nb['cells'] if c['cell_type'] == 'code')
        self.assertTrue(first.startswith((root/'cnn_experiment/github.py').read_text(encoding='utf-8')))
        self.assertIn('checkout_private_repo(CODE_ROOT, GITHUB_REF)', first)
        source = '\n'.join(c['source'] for c in nb['cells'])
        self.assertIn("write_json(SESSION/'git_provenance.json', GIT_PROVENANCE)", source)
        self.assertNotIn('CODE_ZIP_OVERRIDE', source)
        self.assertNotIn('find_code_dirs', source)
        self.assertIn('RESUME_FROM = {}', source)
        self.assertNotIn('RUN_TRAINING =', source)
        self.assertNotIn('RUN_START_DIAGNOSTIC =', source)
        self.assertIn('TEST_APPROVAL = None', source)
        self.assertNotIn('EXPECTED_COMPLETED_EPOCH', source)
        self.assertTrue(all(not c.get('outputs') for c in nb['cells']))


if __name__ == '__main__':
    unittest.main()
