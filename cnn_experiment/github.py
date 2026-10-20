"""Embedded into the Kaggle notebook before the repository is available.

Only standard-library imports are needed until Kaggle Secrets is read.
Credentials never become a Git URL, command argument, config value or file.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPOSITORY_URL = 'https://github.com/lazisa2004/dsa4266-deepfake-detection.git'


def checkout_private_repo(destination, ref='main', secret_name='GITHUB_TOKEN'):
    """Fetch a fresh detached checkout; return only non-secret provenance.

    Does not overwrite an existing directory or use persistent credential helpers.
    The token exists only in memory and the environment of the fetch child.
    Captured Git output is deliberately never printed, including on failure.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new code directory; existing files are preserved.')
    if not isinstance(ref, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*', ref):
        raise ValueError('GITHUB_REF must be a branch, tag or full commit SHA.')
    # Ignore inherited Git tracing/config overrides and persistent credential stores.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('GIT_') and k not in ('SSH_ASKPASS', 'DSA4266_GITHUB_TOKEN')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    git = ['git', '-c', 'credential.helper=', '-c', 'http.extraHeader=',
           '-c', 'core.hooksPath=' + os.devnull]

    def run(arguments, child_env=env):
        try:
            result = subprocess.run(git + arguments, env=child_env,
                                    capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError('Git checkout could not run or timed out. Check Kaggle Internet and Git availability.') from None
        if result.returncode:
            error = result.stderr.lower()
            if any(term in error for term in ('could not resolve host', 'failed to connect', 'network is unreachable', 'timed out')):
                raise RuntimeError('Cannot reach GitHub. Enable Kaggle Internet and retry in a fresh kernel; older code will not be used.')
            raise RuntimeError('Private Git checkout failed. Check Internet, GITHUB_REF and the enabled read-only Kaggle secret.')
        return result.stdout.strip()

    run(['init', '--quiet', str(destination)])
    run(['-C', str(destination), 'remote', 'add', 'origin', REPOSITORY_URL])
    token = None
    fetch_env = dict(env)
    try:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret(secret_name)
        except Exception:
            raise RuntimeError('Enable the GITHUB_TOKEN secret for this Kaggle notebook. Never paste a token into a cell.') from None
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError('The Kaggle GitHub secret is empty.')
        # The temporary executable contains no secret. Git consumes its stdout.
        with tempfile.TemporaryDirectory(prefix='dsa4266_git_askpass_') as folder:
            helper = Path(folder)/'askpass.py'
            helper.write_text(
                '#!/usr/bin/env python3\n'
                'import os, sys\n'
                'prompt = sys.argv[1].lower() if len(sys.argv) > 1 else ""\n'
                'if "github.com" not in prompt: sys.exit(1)\n'
                'if "username" in prompt: print("x-access-token")\n'
                'elif "password" in prompt: print(os.environ["DSA4266_GITHUB_TOKEN"])\n'
                'else: sys.exit(1)\n', encoding='utf-8')
            helper.chmod(0o700)
            fetch_env.update(GIT_ASKPASS=str(helper), DSA4266_GITHUB_TOKEN=token)
            run(['-C', str(destination), 'fetch', '--quiet', '--depth=1', 'origin', ref], fetch_env)
    finally:
        fetch_env.pop('DSA4266_GITHUB_TOKEN', None)
        fetch_env.pop('GIT_ASKPASS', None)
        token = None
    run(['-C', str(destination), 'checkout', '--quiet', '--detach', 'FETCH_HEAD'])
    commit = run(['-C', str(destination), 'rev-parse', 'HEAD'])
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise RuntimeError('Git returned an invalid commit identifier.')
    if run(['-C', str(destination), 'remote', 'get-url', 'origin']) != REPOSITORY_URL:
        raise RuntimeError('Unexpected repository remote URL.')
    return {'repository': REPOSITORY_URL, 'requested_ref': ref,
            'commit': commit, 'code_root': str(destination)}
