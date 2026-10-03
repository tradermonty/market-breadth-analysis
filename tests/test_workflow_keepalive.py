import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'keep_workflow_active.sh'


def git(repo, *args, env=None):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True, env=env).strip()


@pytest.mark.parametrize('age_days,expected_commits', [(0, 1), (29, 1), (30, 2), (61, 2)])
def test_01_keepalive_only_pushes_after_thirty_days(tmp_path, age_days, expected_commits):
    remote = tmp_path / 'remote.git'
    repo = tmp_path / 'checkout'
    git(tmp_path, 'init', '--bare', str(remote))
    git(tmp_path, 'clone', str(remote), str(repo))
    git(repo, 'checkout', '-b', 'main')
    git(repo, 'config', 'user.name', 'Test')
    git(repo, 'config', 'user.email', 'test@example.com')
    (repo / 'data.txt').write_text('unchanged\n')
    git(repo, 'add', 'data.txt')
    old_timestamp = int(time.time()) - age_days * 86400 - 5
    env = {**os.environ, 'GIT_AUTHOR_DATE': f'@{old_timestamp} +0000', 'GIT_COMMITTER_DATE': f'@{old_timestamp} +0000'}
    git(repo, 'commit', '-m', 'Initial data', env=env)
    git(repo, 'push', 'origin', 'main')
    initial_tree = git(repo, 'rev-parse', 'HEAD^{tree}')
    (repo / 'data.txt').write_text('generated report must stay local\n')
    git(repo, 'add', 'data.txt')

    subprocess.run(['bash', str(SCRIPT), 'main'], cwd=repo, check=True)

    assert int(git(remote, 'rev-list', '--count', 'main')) == expected_commits
    assert git(remote, 'rev-parse', 'main^{tree}') == initial_tree
    # Repeated runs must not create another commit.
    subprocess.run(['bash', str(SCRIPT), 'main'], cwd=repo, check=True)
    assert int(git(remote, 'rev-list', '--count', 'main')) == expected_commits
