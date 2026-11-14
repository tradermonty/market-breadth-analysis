"""Fail closed on stale branch coverage, missing critical tests or attempted network."""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET  # nosec B405: local pytest-generated JUnit, not external XML
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(root):
    paths = list(root.glob('*.py'))
    for folder in ('backtest', 'trade', 'scripts', 'tests', 'validation'):
        paths.extend((root / folder).rglob('*.py'))
    paths.extend((root / 'tests/fixtures').rglob('*'))
    paths.extend(root.glob('requirements*.txt'))
    paths.extend((root / '.github/workflows').glob('*.yml'))
    paths.extend(
        root / name
        for name in (
            '.coveragerc-critical',
            'validation/critical_policy.json',
            'pyproject.toml',
            'uv.lock',
            '.python-version',
            '.pre-commit-config.yaml',
            'mypy-critical.ini',
            'requirements-validation-tools.txt',
            '.github/workflows/critical-validation.yml',
        )
    )
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(set(paths))
        if p.is_file() and p.suffix not in ('.pyc',) and '__pycache__' not in p.parts
    }


def evaluate(report, *, expected_run, root=ROOT):
    errors = []
    try:
        manifest = json.loads((report / 'run.json').read_text())
        coverage = json.loads((report / 'coverage.json').read_text())
        policy = json.loads((root / 'validation/critical_policy.json').read_text())
        if not expected_run or manifest['run_id'] != expected_run:
            errors.append('stale or missing run identity')
        if manifest['source_files'] != source_fingerprint(root):
            errors.append('source/test/config identity changed since measurement')
        for name in ('coverage', 'junit'):
            filename = 'coverage.json' if name == 'coverage' else 'junit.xml'
            if manifest[f'{name}_sha256'] != digest(report / filename):
                errors.append(f'{filename} digest mismatch')
        if manifest['pytest_exit_code'] != 0:
            errors.append('pytest did not succeed')
        if (report / 'network-attempts.log').read_text():
            errors.append('network attempted, even if its exception was caught')
        if coverage['meta']['branch_coverage'] is not True or coverage['meta']['version'] != '7.17.3':
            errors.append('branch-enabled coverage 7.17.3 is required')
        files = coverage['files']
        if not policy['files'] or not policy['required_tests']:
            raise ValueError('policy cannot omit all modules or required tests')
        for filename, floors in policy['files'].items():
            data = files.get(filename)
            if not isinstance(data, dict):
                errors.append(f'missing measured file: {filename}')
                continue
            summary = data['summary']
            for kind, numerator, denominator in (
                ('line', 'covered_lines', 'num_statements'),
                ('branch', 'covered_branches', 'num_branches'),
            ):
                covered, total = summary[numerator], summary[denominator]
                floor = floors[kind]
                if not (type(covered) is int and type(total) is int and 0 <= covered <= total and total > 0):
                    raise ValueError(f'invalid measurement counts for {filename}')
                if not isinstance(floor, float | int) or not math.isfinite(floor) or not 0 < floor <= 100:
                    raise ValueError(f'invalid floor for {filename}')
                actual = 100 * covered / total
                if actual < floor:
                    errors.append(f'{filename} {kind}: {actual:.2f}% below {floor}%')
        junit = ET.parse(report / 'junit.xml')  # nosec B314: same local generated artifact
        cases = list(junit.iter('testcase'))
        for selector in policy['required_tests']:
            matching = [c for c in cases if f'{c.get("classname")}::{c.get("name", "").split("[")[0]}' == selector]
            if not matching or any(
                any(c.find(tag) is not None for tag in ('skipped', 'failure', 'error')) for c in matching
            ):
                errors.append(f'critical test absent, failed or skipped: {selector}')
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        errors.append(f'missing or malformed validation evidence: {exc}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    errors = evaluate(args.report, expected_run=args.run_id)
    print('\n'.join(errors) if errors else 'PASS: fresh coverage, mandatory tests and API-free evidence')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
