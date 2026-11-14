"""A real invalid boundary call must fail the incremental mypy gate."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_invalid_contract_call_is_rejected(tmp_path):
    pytest.importorskip('mypy')
    probe = tmp_path / 'invalid_contract.py'
    probe.write_text('from feed_contract import validate_csv_content\nvalidate_csv_content(42, [])\n')
    result = subprocess.run(
        [sys.executable, '-m', 'mypy', '--config-file', str(ROOT / 'mypy-critical.ini'), str(probe)],
        cwd=ROOT,
        env=dict(os.environ, MYPYPATH=str(ROOT)),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert '[arg-type]' in result.stdout


def test_absent_optional_distribution_is_explicit_metadata():
    from scripts.run_critical_validation import installed_version

    assert installed_version('not-a-real-breadth-runtime-package') == 'not installed'
