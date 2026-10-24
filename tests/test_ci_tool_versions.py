"""CI ↔ pre-commit tool-version parity and gate-failure regression tests.

Implements the parity contract of Issue #15:
- requirements-lint-tools.txt / requirements-security-tools.txt are the
  documented source of truth for CI tool versions; their pins must mirror
  the .pre-commit-config.yaml revs (leading 'v' stripped).
- Deliberately invalid lint/format/security fixtures must fail their gates.
- The repository itself must pass the real gate commands under the pinned
  tools. This is a local-depth check: tool availability in the executing
  environment decides skips. In CI, the equivalent verification lives in
  the lint and security jobs (which run this file with their own pinned
  tool set installed; each job skipping the assertions that need tools
  it does not install).

Deterministic and API-free; no network access.
"""

import re
import subprocess
import sys
import tempfile
import typing
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_FILE = REPO_ROOT / 'requirements-lint-tools.txt'
SECURITY_FILE = REPO_ROOT / 'requirements-security-tools.txt'


def pin_version(pin: str) -> str:
    return pin.split('==', 1)[1]


# tool -> (pin expr, pre-commit repo marker + rev)
EXPECTED_PINS = {
    'ruff': ('ruff==0.9.6', ('ruff-pre-commit', '0.9.6')),
    'codespell': ('codespell==2.3.0', ('codespell-project/codespell', '2.3.0')),
    'bandit': ('bandit[toml]==1.8.3', ('bandit', '1.8.3')),
    'detect-secrets': ('detect-secrets==1.5.0', ('Yelp/detect-secrets', '1.5.0')),
    'mypy': ('mypy==1.14.1', ('mirrors-mypy', '1.14.1')),
}

EXACT_PIN_RE = re.compile(r'^[A-Za-z0-9_.-]+(\[[a-z,\-]+\])?==\d+(\.\d+)+$')


def _pin_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]


class ToolGateTestCase(unittest.TestCase):
    """Helpers for running real tools with skip/parity guards."""

    # Always invoke tools against the EXECUTING interpreter (sys.executable)
    # so a stray pipx/homebrew install on PATH cannot satisfy the parity
    # check with a different version than the environment under test.
    TOOL_MODULES: typing.ClassVar[dict[str, str]] = {
        'ruff': 'ruff',
        'codespell': 'codespell_lib',
        'bandit': 'bandit',
        'detect-secrets': 'detect_secrets',  # pragma: allowlist secret (identifier, not a value)
        'mypy': 'mypy',
    }

    def _skip_if_tool_missing(self, tool: str):
        module = self.TOOL_MODULES[tool]
        proc = self._run_tool([sys.executable, '-m', module, '--version'])
        if proc.returncode != 0:
            self.skipTest(f'{tool} not installed for {sys.executable}')

    @staticmethod
    def _run_tool(command):
        return subprocess.run(command, capture_output=True, text=True, timeout=300)

    def _run_module(self, tool: str, *args):
        # Always through sys.executable -m: a PATH binary must never run the
        # repo gates; the executing interpreter is the parity-checked surface.
        module = self.TOOL_MODULES[tool]
        return self._run_tool([sys.executable, '-m', module, *args])

    def _assert_tool_version_matches_pin(self, tool: str, pin: str):
        proc = self._run_module(tool, '--version')
        version = (proc.stdout + proc.stderr).strip()
        self.assertIn(pin_version(pin), version, f'{tool} version mismatch: {version!r} vs pin {pin!r}')

    def _require_result(self, tool: str):
        pin = EXPECTED_PINS[tool][0]
        self._skip_if_tool_missing(tool)
        self._assert_tool_version_matches_pin(tool, pin)
        return pin


class TestPinFiles(unittest.TestCase):
    def test_01_pin_files_exact_only(self):
        self.assertTrue(LINT_FILE.exists(), 'requirements-lint-tools.txt missing')
        self.assertTrue(SECURITY_FILE.exists(), 'requirements-security-tools.txt missing')
        for path in (LINT_FILE, SECURITY_FILE):
            for line in _pin_lines(path):
                self.assertRegex(line, EXACT_PIN_RE, f'non-exact/invalid pin line: {line!r}')

    def test_02_pins_match_pre_commit_revs(self):
        config = (REPO_ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8')
        pins = {
            line.split('==')[0].split('[')[0]: line for path in (LINT_FILE, SECURITY_FILE) for line in _pin_lines(path)
        }
        for tool, (pin, (marker, version)) in EXPECTED_PINS.items():
            for chunk in config.split('repo:'):
                if marker in chunk:
                    match = re.search(r'rev:\s*v?([0-9.]+)', chunk)
                    self.assertIsNotNone(match, f'{tool}: rev not found in pre-commit config')
                    self.assertEqual(match.group(1), version, f'{tool}: pre-commit rev != expected')
                    self.assertEqual(pins[tool], pin, f'{tool}: pin file entry != expected')
                    break
            else:
                self.fail(f'{tool}: pre-commit repo marker {marker!r} not found')

    def test_07_ci_wiring_contract(self):
        ci = (REPO_ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
        lint = ci.split('lint:')[1].split('security:')[0]
        security = ci.split('security:')[1].split('test:')[0]
        test_job = ci.split('test:')[1]
        self.assertIn('pip install -r requirements-lint-tools.txt', lint)
        self.assertIn('pip install -r requirements-security-tools.txt', security)
        for name in ('requirements-lint-tools.txt', 'requirements-security-tools.txt'):
            self.assertNotIn(name, test_job, f'test job must not install {name}')
            self.assertIn(f"'{name}'", ci, f'{name} missing from push.paths')

    def test_08_pin_contract_documented(self):
        for path in (LINT_FILE, SECURITY_FILE):
            header = path.read_text(encoding='utf-8').lower()
            self.assertIn('source of truth', header, path.name)
            self.assertIn('update procedure', header, path.name)


class TestRealGates(ToolGateTestCase):
    """Local-depth: run the real gate commands; skip if the tool is absent."""

    def test_03_invalid_lint_fixture_fails_ruff_check(self):
        self._require_result('ruff')
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / 'bad.py'
            fixture.write_text('import os\n')
            proc = self._run_tool(
                [sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'F401', str(fixture)]
            )
            captured = proc.stdout + proc.stderr
            self.assertNotEqual(proc.returncode, 0, 'invalid lint fixture unexpectedly passed')
            self.assertIn('F401', captured)

    def test_04_invalid_format_fixture_fails_ruff_format(self):
        self._require_result('ruff')
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / 'bad.py'
            fixture.write_text('x=1\n')
            proc = self._run_tool(
                [sys.executable, '-m', 'ruff', 'format', '--check', '--line-length', '120', str(fixture)]
            )
            captured = proc.stdout + proc.stderr
            self.assertNotEqual(proc.returncode, 0, 'invalid format fixture unexpectedly passed')
            self.assertIn('would be reformatted', captured)

    def test_05_insecure_fixture_fails_bandit(self):
        self._require_result('bandit')
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / 'insecure.py'
            fixture.write_text('password = "dummy-password-value"\n')  # pragma: allowlist secret (test dummy)
            proc = self._run_tool(
                [sys.executable, '-m', 'bandit', '-c', str(REPO_ROOT / 'pyproject.toml'), str(fixture)]
            )
            captured = proc.stdout + proc.stderr
            self.assertNotEqual(proc.returncode, 0, 'insecure fixture unexpectedly passed')
            self.assertIn('B105', captured)

    def test_06_repo_passes_pinned_lint_tools(self):
        self._require_result('ruff')
        self._require_result('codespell')
        for tool, args in (
            ('ruff', ('check', str(REPO_ROOT))),
            ('ruff', ('format', '--check', str(REPO_ROOT))),
            ('codespell', (str(REPO_ROOT),)),
        ):
            proc = self._run_module(tool, *args)
            self.assertEqual(proc.returncode, 0, f'{tool} {args} failed: {proc.stdout}{proc.stderr}')

    def test_09_repo_passes_pinned_bandit(self):
        self._require_result('bandit')
        proc = self._run_module(
            'bandit',
            '-c',
            str(REPO_ROOT / 'pyproject.toml'),
            '-r',
            str(REPO_ROOT),
            '--exclude',
            'archive,venv311,tests',
        )
        self.assertEqual(proc.returncode, 0, f'bandit repo scan failed: {proc.stdout}{proc.stderr}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
