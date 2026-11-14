"""Produce fresh API-free branch/JUnit evidence and enforce the critical policy."""

import argparse
import importlib.metadata
import json
import shutil
import subprocess  # nosec B404: fixed interpreter and argument lists, no shell
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.critical_coverage import digest, evaluate, source_fingerprint  # noqa: E402
from scripts.offline_guard import offline_environment  # noqa: E402


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'


def run(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    log = output / 'network-attempts.log'
    log.touch()
    env = offline_environment(log)
    env.update(
        BREADTH_SOURCE_ROOT=str(ROOT),
        BREADTH_COVERAGE_DATA=str(output / '.coverage'),
        BREADTH_PROBE_EVIDENCE=str(output / 'fault-probes'),
    )
    config = str(ROOT / '.coveragerc-critical')
    inputs = source_fingerprint(ROOT)
    command = [sys.executable, '-m', 'coverage']
    with (output / 'tests.log').open('w') as stream:
        result = subprocess.run(  # nosec B603
            [
                *command,
                'run',
                f'--rcfile={config}',
                '-m',
                'pytest',
                'tests/',
                '-q',
                '--tb=short',
                f'--junitxml={output / "junit.xml"}',
            ],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    for args in (
        ['combine'],
        ['json', '-o', str(output / 'coverage.json')],
        ['html', '-d', str(output / 'html')],
    ):
        # Diagnostics are attempted even after failing tests; missing reports fail the gate.
        subprocess.run([*command, args[0], f'--rcfile={config}', *args[1:]], cwd=ROOT, env=env)  # nosec B603
    manifest = {
        'run_id': run_id,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'tools': {name: importlib.metadata.version(name) for name in ('pytest', 'coverage', 'mypy')},
        'source_files': inputs,
        'pytest_exit_code': result.returncode,
    }
    git = shutil.which('git')
    if git:
        revision = subprocess.run(  # nosec B603: fixed read-only git command
            [git, 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True
        )
        manifest['source_revision'] = revision.stdout.strip() if revision.returncode == 0 else 'unavailable'
    else:
        manifest['source_revision'] = 'unavailable'
    manifest['runtime_versions'] = {
        name: installed_version(name)
        for name in ('numpy', 'pandas', 'scipy', 'plotly', 'exchange-calendars', 'alpaca-trade-api')
    }
    for name, filename in (('coverage', 'coverage.json'), ('junit', 'junit.xml')):
        if (output / filename).exists():
            manifest[f'{name}_sha256'] = digest(output / filename)
    (output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    errors = evaluate(output, expected_run=run_id)
    gaps = {}
    if (output / 'coverage.json').exists():
        measured = json.loads((output / 'coverage.json').read_text())
        policy = json.loads((ROOT / 'validation/critical_policy.json').read_text())
        for name in policy['files']:
            info = measured.get('files', {}).get(name, {})
            gaps[name] = {key: info.get(key) for key in ('summary', 'missing_lines', 'missing_branches')}
    (output / 'critical-gaps.json').write_text(json.dumps(gaps, indent=2) + '\n')
    (output / 'verdict.json').write_text(
        json.dumps({'run_id': run_id, 'pass': not errors, 'errors': errors}, indent=2) + '\n'
    )
    print('\n'.join(errors) if errors else 'PASS: fresh critical-path validation')
    print(f'Evidence: {output}')
    return bool(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    args = parser.parse_args()
    return run(args.output_dir)


if __name__ == '__main__':
    raise SystemExit(main())
