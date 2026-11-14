"""Failure injection for current-run coverage and mandatory critical test evidence."""

import json
import os
from pathlib import Path

import pytest


def test_01_empty_coverage_cannot_pass(tmp_path):
    from scripts.critical_coverage import evaluate

    (tmp_path / 'coverage.json').write_text('{}')
    errors = evaluate(tmp_path, expected_run='current-run', root=tmp_path)
    assert errors


@pytest.fixture
def bundle(tmp_path):
    from scripts.critical_coverage import digest, source_fingerprint

    root = tmp_path / 'source'
    root.mkdir()
    (root / 'sample.py').write_text('x = 1\n')
    (root / 'validation').mkdir()
    policy = {'files': {'sample.py': {'line': 50, 'branch': 50}}, 'required_tests': ['tests.test_sample::test_ok']}
    (root / 'validation/critical_policy.json').write_text(json.dumps(policy))
    report = tmp_path / 'report'
    report.mkdir()
    coverage = {
        'meta': {'branch_coverage': True, 'version': '7.17.3'},
        'files': {
            'sample.py': {
                'summary': {'num_statements': 2, 'covered_lines': 2, 'num_branches': 2, 'covered_branches': 2},
                'missing_lines': [],
                'missing_branches': [],
            }
        },
    }
    (report / 'coverage.json').write_text(json.dumps(coverage))
    (report / 'junit.xml').write_text(
        '<testsuites><testsuite><testcase classname="tests.test_sample" name="test_ok"/></testsuite></testsuites>'
    )
    (report / 'network-attempts.log').write_text('')
    manifest = {
        'run_id': 'current-run',
        'source_files': source_fingerprint(root),
        'pytest_exit_code': 0,
        'coverage_sha256': digest(report / 'coverage.json'),
        'junit_sha256': digest(report / 'junit.xml'),
    }
    (report / 'run.json').write_text(json.dumps(manifest))
    return root, report


def failures(bundle):
    from scripts.critical_coverage import evaluate

    root, report = bundle
    return evaluate(report, expected_run='current-run', root=root)


def test_02_complete_current_evidence_passes(bundle):
    assert failures(bundle) == []


@pytest.mark.parametrize(
    'fault',
    [
        'stale_run',
        'changed_source',
        'changed_policy',
        'changed_config',
        'changed_lock',
        'changed_python',
        'missing_file',
        'line_drop',
        'branch_drop',
        'no_branch',
        'skipped',
        'missing_test',
        'caught_network',
    ],
)
def test_03_faults_cannot_produce_pass(bundle, fault):
    from scripts.critical_coverage import digest

    root, report = bundle
    manifest = json.loads((report / 'run.json').read_text())
    data = json.loads((report / 'coverage.json').read_text())
    if fault == 'stale_run':
        manifest['run_id'] = 'previous-run'
    elif fault == 'changed_source':
        (root / 'sample.py').write_text('x = 2\n')
    elif fault == 'changed_policy':
        path = root / 'validation/critical_policy.json'
        changed = json.loads(path.read_text())
        changed['files']['sample.py']['line'] = 1
        path.write_text(json.dumps(changed))
    elif fault == 'changed_config':
        (root / 'pyproject.toml').write_text('[tool.pytest.ini_options]\naddopts = "--ignore=tests"\n')
    elif fault == 'changed_lock':
        (root / 'uv.lock').write_text('version = 1\n')
    elif fault == 'changed_python':
        (root / '.python-version').write_text('3.12\n')
    elif fault == 'missing_file':
        data['files'] = {}
    elif fault == 'line_drop':
        data['files']['sample.py']['summary']['covered_lines'] = 0
    elif fault == 'branch_drop':
        data['files']['sample.py']['summary']['covered_branches'] = 0
    elif fault == 'no_branch':
        data['meta']['branch_coverage'] = False
    elif fault == 'skipped':
        (report / 'junit.xml').write_text(
            '<testsuites><testsuite><testcase classname="tests.test_sample" name="test_ok"><skipped/></testcase></testsuite></testsuites>'
        )
    elif fault == 'missing_test':
        (report / 'junit.xml').write_text('<testsuites/>')
    elif fault == 'caught_network':
        (report / 'network-attempts.log').write_text('socket.connect\n')
    (report / 'coverage.json').write_text(json.dumps(data))
    manifest['coverage_sha256'] = digest(report / 'coverage.json')
    manifest['junit_sha256'] = digest(report / 'junit.xml')
    (report / 'run.json').write_text(json.dumps(manifest))
    errors = failures(bundle)
    if os.environ.get('BREADTH_PROBE_EVIDENCE'):
        destination = Path(os.environ['BREADTH_PROBE_EVIDENCE'])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f'gate-{fault}.json').write_text(json.dumps({'fault': fault, 'errors': errors}, indent=2))
    assert errors
