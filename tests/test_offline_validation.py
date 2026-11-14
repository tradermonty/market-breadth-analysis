"""The API-free launcher must detect caught attempts and Python child processes."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def probe(tmp_path, code):
    log = tmp_path / 'attempts.log'
    env = dict(os.environ, BREADTH_OFFLINE_LOG=str(log))
    proc = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/run_offline.py'), '--log', str(log), '--', sys.executable, '-c', code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if os.environ.get('BREADTH_PROBE_EVIDENCE'):
        evidence = Path(os.environ['BREADTH_PROBE_EVIDENCE'])
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f'network-{uuid.uuid4().hex}.json').write_text(
            json.dumps(
                {
                    'returncode': proc.returncode,
                    'stdout': proc.stdout,
                    'stderr': proc.stderr,
                    'events': log.read_text() if log.exists() else None,
                },
                indent=2,
            )
        )
    return proc, log


def test_01_healthy_python_child_passes(tmp_path):
    result, log = probe(tmp_path, 'print("healthy")')
    assert result.returncode == 0, result.stderr
    assert not log.read_text()


@pytest.mark.parametrize(
    'call,event',
    [
        ('socket.getaddrinfo("example.invalid", 443)', 'socket.getaddrinfo'),
        ('socket.getnameinfo(("127.0.0.1", 9), 0)', 'socket.getnameinfo'),
    ],
)
def test_02_caught_dns_attempt_fails_launcher(tmp_path, call, event):
    result, log = probe(tmp_path, f'import socket\ntry: {call}\nexcept OSError: pass')
    assert result.returncode != 0
    assert event in log.read_text()


def test_03_explicit_child_environment_cannot_drop_guard(tmp_path):
    code = (
        'import os, subprocess, sys\n'
        'subprocess.run([sys.executable, "-c", '
        '"import socket; socket.socket().connect((\\"127.0.0.1\\", 9))"], '
        'env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": "/tmp"})\n'
    )
    result, log = probe(tmp_path, code)
    assert result.returncode != 0
    assert 'socket.connect' in log.read_text()


def test_04_collection_attempt_fails_even_if_caught(tmp_path):
    test = tmp_path / 'test_probe.py'
    test.write_text(
        'import socket\ntry: socket.gethostbyname("example.invalid")\nexcept OSError: pass\ndef test_ok(): pass\n'
    )
    result, log = probe(tmp_path, f'import pytest; raise SystemExit(pytest.main([{str(test)!r}, "-q"]))')
    assert result.returncode != 0
    assert 'socket.gethostbyname' in log.read_text()


def test_05_child_only_coverage_marker_executes(tmp_path):
    result, log = probe(
        tmp_path,
        'from validation.coverage_child_probe import child_marker\nassert child_marker(1) == 2\nassert child_marker(-1) == -2',
    )
    assert result.returncode == 0, result.stderr
    assert not log.read_text()


def test_06_direct_pytest_default_child_environment_is_guarded(tmp_path):
    # -S deliberately starts a pristine process, then loads the real conftest.
    # Its normally-started child must inherit the bootstrap without env=...
    log = tmp_path / 'direct-attempts.log'
    code = (
        f'import sys; sys.path[:] = {sys.path!r}\n'
        'import importlib.util, os, subprocess\n'
        f'os.environ["PYTHONPATH"] = {str(ROOT)!r}\n'
        f'spec = importlib.util.spec_from_file_location("guard_conftest", {str(ROOT / "tests/conftest.py")!r})\n'
        'module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n'
        'subprocess.run([sys.executable, "-c", "import socket\\ntry: socket.getaddrinfo(\'localhost\', 9)\\nexcept OSError: pass"])\n'
    )
    env = dict(os.environ, SKIP_API_TESTS='true', BREADTH_OFFLINE_LOG=str(log))
    result = subprocess.run([sys.executable, '-S', '-c', code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'socket.getaddrinfo' in log.read_text()
