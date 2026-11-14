"""Reproducible installs must reject stale inputs and credential-triggered tests."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_01_core_has_no_optional_runtime_packages():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert project['project']['requires-python'] == '>=3.11,<3.12'
    core = '\n'.join(project['project']['dependencies']).lower()
    assert not any(name in core for name in ('kaleido', 'pyqt5', 'alpaca', 'pytest'))
    assert set(project['project']['optional-dependencies']) == {'png', 'trading', 'windows-qt'}


def test_02_tool_groups_match_existing_direct_pins():
    groups = tomllib.loads((ROOT / 'pyproject.toml').read_text())['dependency-groups']
    for group in ('lint', 'security'):
        pins = {
            line.strip()
            for line in (ROOT / f'requirements-{group}-tools.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')
        }
        assert set(groups[group]) == pins


def run_check(path):
    env = dict(os.environ, UV_OFFLINE='true')
    return subprocess.run(
        [sys.executable, str(ROOT / 'scripts/check_dependencies.py'), '--project-dir', str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.fixture
def dependency_copy(tmp_path):
    for filename in ('pyproject.toml', 'uv.lock', 'requirements.txt', 'requirements-analysis.txt', '.python-version'):
        shutil.copyfile(ROOT / filename, tmp_path / filename)
    return tmp_path


def test_03_committed_lock_and_export_are_current():
    result = run_check(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('filename', ['requirements.txt', 'requirements-analysis.txt'])
def test_04_corrupt_export_is_rejected(dependency_copy, filename):
    (dependency_copy / filename).write_text('requests==0.0.0\n')
    result = run_check(dependency_copy)
    assert result.returncode != 0
    assert f'{filename} is stale' in result.stdout + result.stderr


def test_05_new_dependency_invalidates_lock(dependency_copy):
    path = dependency_copy / 'pyproject.toml'
    path.write_text(path.read_text().replace('dependencies = [', 'dependencies = ["not-a-real-lock-probe==0.0.0",', 1))
    result = run_check(dependency_copy)
    assert result.returncode != 0


def test_06_skip_api_tests_with_present_key():
    env = dict(os.environ, SKIP_API_TESTS='true', FMP_API_KEY='dummy-api-key-for-test')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', str(ROOT / 'tests/test_sp500_fetch.py'), '-q', '-rs'],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 skipped' in result.stdout


def test_07_noninteractive_backtest_does_not_select_tk(monkeypatch):
    from unittest.mock import Mock

    from backtest import visualization

    use = Mock()
    monkeypatch.setattr(visualization.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(visualization.matplotlib, 'use', use)
    visualization.setup_matplotlib_backend(headless=True)
    use.assert_called_once_with('Agg')


def test_08_ci_environment_uses_available_contexts():
    import yaml

    ci = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    assert 'runner.' not in str(ci.get('env', {}))
    for job in ci['jobs'].values():
        assert 'runner.' not in str(job.get('env', {}))
        steps = job['steps']
        configure = next(i for i, step in enumerate(steps) if 'GITHUB_ENV' in step.get('run', ''))
        sync = next(i for i, step in enumerate(steps) if 'uv sync' in step.get('run', ''))
        assert configure < sync
