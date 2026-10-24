"""API-free publication boundary regressions."""

import csv
import json
from datetime import datetime, timezone

import pytest

from scripts import publish_market_breadth as pub

NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
START = datetime(2026, 10, 17, 11, tzinfo=timezone.utc)
DAY = '2026-10-16'


def fixture_outputs(path):
    path.mkdir(exist_ok=True)
    (path / 'index.html').write_text('<html>index</html>')
    (path / 'market_breadth.html').write_text('<html>Plotly.newPlot()</html>')
    row = {'Date': DAY, 'S&P500_Price': '5000'}
    for suffix, smooth, trend, markers in [
        ('', '200MA', 'Breadth_200MA_Trend', ['Bearish_Signal', 'Is_Peak', 'Is_Trough', 'Is_Trough_8MA_Below_04']),
        ('_50', '50MA', 'Breadth_50_MA_Trend', ['Bearish_Signal_50', 'Is_Peak_50', 'Is_Trough_50']),
    ]:
        for name in ['Raw', smooth, '8MA']:
            row[f'Breadth{suffix}_Index_{name}'] = '0.5'
        row[trend] = '0'
        row.update(dict.fromkeys(markers, 'False'))
    for period in (200, 50):
        row.update(
            {
                f'Eligible_Count_{period}': '10',
                f'Missing_Count_{period}': '0',
                f'Above_Count_{period}': '5',
                f'Coverage_{period}': '1',
            }
        )
    write_rows(path, [row])
    summary = dict.fromkeys(pub.SUMMARY_KEYS, '0')
    summary.update({'Analysis Period Start': DAY, 'Analysis Period End': DAY, 'Total Trading Days': '1'})
    with (path / 'market_breadth_summary.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Metric', 'Value'])
        writer.writerows(summary.items())
    metadata = {
        'selected_date': DAY,
        'expected_date': DAY,
        'latest_completed_date': DAY,
        'confidence': 'observed',
        'publication_eligible': True,
        'calendar_name': 'XNYS',
        'calendar_version': '4.13.2',
        'generated_at': NOW.isoformat(),
        'source': 'fmp_spy_daily',
    }
    (path / 'market_date_resolution.json').write_text(json.dumps(metadata))
    return row


def write_rows(path, rows):
    with (path / 'market_breadth_data.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket

    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('network forbidden'))
    monkeypatch.setattr(pub, '_calendar_bounds', lambda *_: (DAY, DAY))


def validate(path):
    return pub.validate_outputs(path, started=START, now=NOW, min_coverage=0.9, short_ma=8, grace=30)


def test_valid_without_png(tmp_path):
    fixture_outputs(tmp_path)
    result = validate(tmp_path)
    assert result['latest_market_date'] == DAY
    assert 'market_breadth_data.csv' in result['sha256']


@pytest.mark.parametrize(
    'field,value',
    [
        ('Date', '2026-10-15'),
        ('Date', '20261016'),
        ('S&P500_Price', 'nan'),
        ('Breadth_Index_Raw', '1.1'),
        ('Breadth_Index_8MA', 'nan'),
        ('Breadth_200MA_Trend', '2'),
        ('Is_Peak', 'yes'),
        ('Coverage_50', '.8'),
        ('Missing_Count_200', '-1'),
        ('Above_Count_200', '11'),
        ('Eligible_Count_50', '1.5'),
        ('Breadth_50_Index_Raw', '.4'),
    ],
)
def test_invalid_observations(tmp_path, field, value):
    row = fixture_outputs(tmp_path)
    row[field] = value
    write_rows(tmp_path, [row])
    with pytest.raises(pub.PublicationError):
        validate(tmp_path)


@pytest.mark.parametrize(
    'kind', ['missing', 'empty', 'symlink', 'duplicate', 'summary', 'stale', 'unverified', 'future']
)
def test_invalid_artifacts(tmp_path, kind):
    row = fixture_outputs(tmp_path)
    chart = tmp_path / 'market_breadth.html'
    if kind == 'missing':
        chart.unlink()
    elif kind == 'empty':
        chart.write_text('')
    elif kind == 'symlink':
        chart.unlink()
        chart.symlink_to(tmp_path / 'index.html')
    elif kind == 'duplicate':
        write_rows(tmp_path, [row, row])
    elif kind == 'summary':
        p = tmp_path / 'market_breadth_summary.csv'
        p.write_text(p.read_text().replace('Total Trading Days,1', 'Total Trading Days,2'))
    else:
        p = tmp_path / 'market_date_resolution.json'
        data = json.loads(p.read_text())
        data.update(
            {
                'stale': {'expected_date': '2026-10-15'},
                'unverified': {'publication_eligible': 'true'},
                'future': {'generated_at': '2026-10-18T00:00:00+00:00'},
            }[kind]
        )
        p.write_text(json.dumps(data))
    with pytest.raises(pub.PublicationError):
        validate(tmp_path)


def test_crossed_grace_boundary(tmp_path, monkeypatch):
    fixture_outputs(tmp_path)
    monkeypatch.setattr(pub, '_calendar_bounds', lambda *_: ('2026-10-19', '2026-10-19'))
    with pytest.raises(pub.PublicationError):
        validate(tmp_path)


def test_runner_success_and_previous_preserved(tmp_path):
    previous = tmp_path / 'previous'
    fixture_outputs(previous)
    before = (previous / 'market_breadth_data.csv').read_bytes()
    dest = tmp_path / 'new'
    assert (
        pub.publish_run(
            dest, analyze=lambda p: (fixture_outputs(p), None)[1], clock=lambda: NOW, source_revision='a' * 40
        )
        == 0
    )
    report = json.loads((dest / 'validation.json').read_text())
    assert report['status'] == 'PASS' and report['source_revision'] == 'a' * 40
    assert (dest / '.nojekyll').exists()
    assert (previous / 'market_breadth_data.csv').read_bytes() == before
    assert pub.publish_run(dest, analyze=lambda _: pytest.fail('must not execute')) == 1
    assert json.loads((dest / 'validation.json').read_text()) == report


@pytest.mark.parametrize('raises', [False, True])
def test_swallowed_or_raised_errors_redacted(tmp_path, monkeypatch, capsys, raises):
    monkeypatch.setenv('FMP_API_KEY', 'dummy-private-environment-value')

    def bad(_):
        import sys

        print('dummy-private-environment-value https://example/?apikey=query-secret')
        print('Authorization: Bearer header-secret', file=sys.stderr)
        if raises:
            raise ValueError('https://example/?token=exception-secret')

    assert pub.publish_run(tmp_path / 'run', analyze=bad, clock=lambda: NOW, source_revision='a' * 40) == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err + (tmp_path / 'run' / 'validation.json').read_text()
    for secret in ['dummy-private-environment-value', 'query-secret', 'header-secret', 'exception-secret']:
        assert secret not in text
    assert json.loads((tmp_path / 'run' / 'validation.json').read_text())['status'] == 'FAIL'


def test_subprocess_timeout_redacted(monkeypatch, capsys, tmp_path):
    import subprocess

    def timeout(*_, **__):
        raise subprocess.TimeoutExpired(
            ['python'], 1, output=b'?apikey=timeout-secret', stderr=b'Authorization: Bearer timeout-header'
        )

    monkeypatch.setattr(pub.subprocess, 'run', timeout)
    with pytest.raises(pub.PublicationError):
        pub._analysis_process(tmp_path, 8, 30)
    captured = capsys.readouterr()
    assert 'timeout-secret' not in captured.out and 'timeout-header' not in captured.err


def test_missing_prerequisites_fail_before_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pub, '_preflight', lambda: (_ for _ in ()).throw(pub.PublicationError('PREREQUISITE', 'missing'))
    )
    assert pub.publish_run(tmp_path / 'run', clock=lambda: NOW, source_revision='a' * 40) == 1
    assert json.loads((tmp_path / 'run' / 'validation.json').read_text())['reason_code'] == 'PREREQUISITE'


def test_schema_and_order_failures(tmp_path):
    row = fixture_outputs(tmp_path)
    del row['Coverage_200']
    write_rows(tmp_path, [row])
    with pytest.raises(pub.PublicationError):
        validate(tmp_path)
    row = fixture_outputs(tmp_path)
    earlier = dict(row, Date='2026-10-15')
    write_rows(tmp_path, [row, earlier])
    with pytest.raises(pub.PublicationError):
        validate(tmp_path)


def test_legitimate_warmup_and_low_latest_coverage(tmp_path):
    row = fixture_outputs(tmp_path)
    earlier = dict(row, Date='2026-10-15', Breadth_Index_200MA='', Breadth_50_Index_50MA='')
    for period, prefix in [(200, 'Breadth'), (50, 'Breadth_50')]:
        earlier.update(
            {
                f'{prefix}_Index_Raw': '',
                f'{prefix}_Index_8MA': '',
                f'Eligible_Count_{period}': '0',
                f'Missing_Count_{period}': '10',
                f'Above_Count_{period}': '0',
                f'Coverage_{period}': '0',
            }
        )
    write_rows(tmp_path, [earlier, row])
    summary = tmp_path / 'market_breadth_summary.csv'
    summary.write_text(
        summary.read_text()
        .replace('Total Trading Days,1', 'Total Trading Days,2')
        .replace('Analysis Period Start,2026-10-16', 'Analysis Period Start,2026-10-15')
    )
    validate(tmp_path)
    row.update(Eligible_Count_200='8', Missing_Count_200='2', Above_Count_200='4', Coverage_200='.8')
    write_rows(tmp_path, [earlier, row])
    with pytest.raises(pub.PublicationError, match='coverage below'):
        validate(tmp_path)


def test_dynamic_short_ma(tmp_path):
    fixture_outputs(tmp_path)
    for name in ['market_breadth_data.csv', 'market_breadth_summary.csv']:
        file = tmp_path / name
        file.write_text(file.read_text().replace('8MA', '10MA'))
    pub.validate_outputs(tmp_path, started=START, now=NOW, min_coverage=0.9, short_ma=10, grace=30)


def test_subprocess_logs_redacted(monkeypatch, tmp_path, capsys):
    import subprocess

    monkeypatch.setattr(
        pub.subprocess,
        'run',
        lambda *_, **__: subprocess.CompletedProcess(
            [], 0, stdout='?apikey=swallowed-secret', stderr='Authorization: Bearer logging-secret'
        ),
    )
    pub._analysis_process(tmp_path, 8, 30)
    captured = capsys.readouterr()
    assert 'swallowed-secret' not in captured.out and 'logging-secret' not in captured.err


def test_cli_failure_is_nonzero(tmp_path):
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(pub.ROOT / 'scripts/publish_market_breadth.py'), '--output-dir', str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, 'FMP_API_KEY': '', 'DOTENV_DISABLED': '1'},
        check=False,
    )
    assert result.returncode == 1
    assert 'OUTPUT_EXISTS' in result.stderr


def test_workflow_publishes_only_validated_fresh_directory():
    import yaml

    workflow = yaml.safe_load((pub.ROOT / '.github/workflows/daily-market-breadth.yml').read_text())
    steps = workflow['jobs']['build']['steps']
    generation = next(i for i, step in enumerate(steps) if 'publish_market_breadth.py' in step.get('run', ''))
    uploads = [(i, step) for i, step in enumerate(steps) if 'upload-pages-artifact' in step.get('uses', '')]
    assert len(uploads) == 1
    index, upload = uploads[0]
    assert generation < index and 'if' not in upload
    assert upload['with']['path'] == '${{ runner.temp }}/market-breadth-${{ github.run_id }}-${{ github.run_attempt }}'
    diagnostic = next(step for step in steps if step.get('if') == 'always()')
    assert diagnostic['uses'].startswith('actions/upload-artifact@')
    assert diagnostic['with']['path'] == upload['with']['path'] + '/validation.json'
    assert workflow['jobs']['deploy']['needs'] == 'build'
    assert 'needs' not in workflow['jobs']['keepalive']


def test_real_producer_contract_and_restored_paths(tmp_path, monkeypatch):
    """Runs when the prerequisite PRs are present; never fetches provider data."""
    import sys

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    import market_breadth as engine

    if not hasattr(engine, 'resolve_market_end_date'):
        pytest.skip('Requires exact prerequisite Issue #8 producer integration')
    from market_dates import resolve_market_date

    dates = pd.bdate_range(end=DAY, periods=650)
    values = np.arange(650)
    stocks = pd.DataFrame(
        {'A': 100 + values * 0.1 + 5 * np.sin(values / 20), 'B': 100 + values * 0.1 - 5 * np.sin(values / 20)},
        index=dates,
    )
    resolution = resolve_market_date(lambda *_: [{'date': DAY, 'close': 5000}], now=NOW, source='fmp_spy_daily')
    monkeypatch.setattr(engine, 'resolve_market_end_date', lambda *_, **__: resolution)
    monkeypatch.setattr(pub, '_calendar_bounds', lambda *_: (DAY, DAY))
    monkeypatch.setattr(engine, 'get_sp500_tickers_from_fmp', lambda: ['A', 'B'])
    monkeypatch.setattr(engine, 'get_sp500_price_data', lambda *_: pd.Series(5000.0, index=dates))
    monkeypatch.setattr(engine, 'get_multiple_stock_data', lambda *_: stocks)
    monkeypatch.setattr(go.Figure, 'write_image', lambda *_args, **_kwargs: None)
    original = (engine.reports_dir, engine.plot_breadth_and_sp500_with_peaks, sys.argv)
    dest = tmp_path / 'real'
    assert (
        pub.publish_run(
            dest, analyze=lambda path: pub._run_analysis(path, 8, 30), clock=lambda: NOW, source_revision='a' * 40
        )
        == 0
    )
    assert (engine.reports_dir, engine.plot_breadth_and_sp500_with_peaks, sys.argv) == original
    report = json.loads((dest / 'validation.json').read_text())
    assert report['status'] == 'PASS' and (dest / 'market_breadth.html').stat().st_size > 10000
    assert not (dest / 'market_breadth.png').exists()
