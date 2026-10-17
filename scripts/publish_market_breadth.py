"""Generate isolated, validated Pages artifacts; never publish leftover analysis files."""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess  # nosec B404: fixed argument lists, no shell, captured output
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from secret_redaction import redact, register_secret  # noqa: E402

SUMMARY_KEYS = (
    'Average Peaks (200MA)',
    'Average Troughs (8MA < 0.4)',
    'Total Peaks Count',
    'Total Troughs Count',
    'Total Troughs (8MA < 0.4) Count',
    'Analysis Period Start',
    'Analysis Period End',
    'Total Trading Days',
    'Average Peaks (50MA)',
    'Average Troughs 50 (< 0.3)',
    'Total Peaks 50 Count',
    'Total Troughs 50 Count',
)
STATIC_ASSETS = ('index.html', 'downtrend_forecast.html', 'downtrend_forecast_high_price.html')
REQUIRED = (
    'index.html',
    'market_breadth.html',
    'market_breadth_data.csv',
    'market_breadth_summary.csv',
    'market_date_resolution.json',
)


class PublicationError(ValueError):
    """An unsuccessful publication with a stable diagnostic category."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise PublicationError(code, message)


def _day(value):
    require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value), 'DATE', 'Invalid ISO date')
    return date.fromisoformat(value)


def _number(value, *, nan=False):
    value = float('nan') if nan and value == '' else float(value)
    require(math.isfinite(value) or (nan and math.isnan(value)), 'VALUES', 'Nonfinite numeric value')
    return value


def _rows(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), 'SCHEMA', 'Duplicate CSV columns')
        rows = list(reader)
        require(
            rows and all(None not in row and None not in row.values() for row in rows),
            'SCHEMA',
            'Empty or malformed CSV',
        )
        return rows


def _calendar_bounds(now, grace):
    from market_dates import resolve_market_date

    result = resolve_market_date(lambda *_: [], now=now, grace_minutes=grace)
    require(
        result.expected_date and result.latest_completed_date and result.calendar_name == 'XNYS',
        'CALENDAR',
        'Calendar bounds unavailable',
    )
    return result.expected_date, result.latest_completed_date


def _validate_outputs(path, *, started, now, min_coverage=0.9, short_ma=8, grace=30):
    """Validate actual generated CSV and date provenance against current offline sessions."""
    require(math.isfinite(min_coverage) and 0 < min_coverage <= 1, 'CONFIG', 'Invalid minimum coverage')
    for name in REQUIRED:
        file = path / name
        require(
            not file.is_symlink() and file.is_file() and file.stat().st_size > 0,
            'ARTIFACT',
            f'Missing, empty or unsafe artifact: {name}',
        )
    for name in ('index.html', 'market_breadth.html'):
        html = (path / name).read_text()
        require('<html' in html.lower() and '</html>' in html.lower(), 'HTML', f'Invalid HTML: {name}')
        if name == 'market_breadth.html':
            require('plotly' in html.lower(), 'HTML', 'Missing Plotly chart')
    rows = _rows(path / 'market_breadth_data.csv')
    days = [_day(row['Date']) for row in rows]
    require(days == sorted(set(days)), 'DATE', 'Dates must be sorted and unique')
    markers = [
        'Bearish_Signal',
        'Is_Peak',
        'Is_Trough',
        f'Is_Trough_{short_ma}MA_Below_04',
        'Bearish_Signal_50',
        'Is_Peak_50',
        'Is_Trough_50',
    ]
    first_eligible = {
        period: next((i for i, r in enumerate(rows) if _number(r[f'Eligible_Count_{period}']) > 0), len(rows))
        for period in (200, 50)
    }
    for index, row in enumerate(rows):
        require(_number(row['S&P500_Price']) > 0, 'VALUES', 'Index price must be positive')
        for marker in markers:
            require(row[marker] in ('True', 'False'), 'VALUES', f'Invalid boolean: {marker}')
        for period, prefix, trend in [
            (200, 'Breadth', 'Breadth_200MA_Trend'),
            (50, 'Breadth_50', 'Breadth_50_MA_Trend'),
        ]:
            require(_number(row[trend]) in (-1, 0, 1), 'VALUES', 'Invalid trend')
            for label in ('Raw', f'{period}MA', f'{short_ma}MA'):
                value = _number(row[f'{prefix}_Index_{label}'], nan=True)
                require(math.isnan(value) or 0 <= value <= 1, 'VALUES', 'Breadth outside [0,1]')
                if label != 'Raw':
                    require(not math.isnan(value) or index < first_eligible[period], 'VALUES', 'NaN outside MA warmup')
                    if index == len(rows) - 1:
                        require(math.isfinite(value), 'VALUES', 'Latest smoothed breadth unavailable')
            counts = [_number(row[f'{label}_Count_{period}']) for label in ('Eligible', 'Missing', 'Above')]
            require(all(v >= 0 and v.is_integer() for v in counts), 'COVERAGE', 'Invalid constituent counts')
            eligible, missing, above = counts
            require(eligible + missing > 0 and above <= eligible, 'COVERAGE', 'Invalid coverage denominator/count')
            coverage = _number(row[f'Coverage_{period}'])
            require(
                math.isclose(coverage, eligible / (eligible + missing), abs_tol=1e-8),
                'COVERAGE',
                'Inconsistent coverage',
            )
            raw = _number(row[f'{prefix}_Index_Raw'], nan=True)
            require(
                (eligible == 0 and math.isnan(raw))
                or (eligible > 0 and math.isclose(raw, above / eligible, abs_tol=1e-8)),
                'COVERAGE',
                'Inconsistent raw breadth',
            )
            if index == len(rows) - 1:
                require(coverage >= min_coverage, 'COVERAGE', 'Latest coverage below threshold')
    summary_rows = _rows(path / 'market_breadth_summary.csv')
    summary = {row['Metric']: row['Value'] for row in summary_rows}
    require(
        len(summary) == len(summary_rows)
        and {key.replace('8MA', f'{short_ma}MA') for key in SUMMARY_KEYS} <= summary.keys(),
        'SUMMARY',
        'Missing/duplicate summary metrics',
    )
    require(
        summary['Analysis Period Start'] == days[0].isoformat()
        and summary['Analysis Period End'] == days[-1].isoformat(),
        'SUMMARY',
        'Summary period mismatch',
    )
    counts = {
        'Total Trading Days': len(rows),
        'Total Peaks Count': sum(r['Is_Peak'] == 'True' for r in rows),
        'Total Troughs Count': sum(r['Is_Trough'] == 'True' for r in rows),
        f'Total Troughs ({short_ma}MA < 0.4) Count': sum(r[f'Is_Trough_{short_ma}MA_Below_04'] == 'True' for r in rows),
        'Total Peaks 50 Count': sum(r['Is_Peak_50'] == 'True' for r in rows),
        'Total Troughs 50 Count': sum(r['Is_Trough_50'] == 'True' for r in rows),
    }
    for key, count in counts.items():
        require(_number(summary[key]) == count, 'SUMMARY', f'Summary count mismatch: {key}')
    for original_key in SUMMARY_KEYS:
        key = original_key.replace('8MA', f'{short_ma}MA')
        if key.startswith('Average'):
            if summary[key] == 'N/A' and key in ('Average Peaks (50MA)', 'Average Troughs 50 (< 0.3)'):
                continue
            require(0 <= _number(summary[key]) <= 1, 'SUMMARY', 'Invalid summary average')
    metadata = json.loads((path / 'market_date_resolution.json').read_text())
    expected, completed = _calendar_bounds(now, grace)
    selected = _day(metadata['selected_date'])
    require(
        metadata['confidence'] == 'observed' and metadata['publication_eligible'] is True,
        'PROVENANCE',
        'Publication requires observed, eligible data',
    )
    require(
        metadata['calendar_name'] == 'XNYS'
        and metadata['calendar_version'] not in ('', 'unavailable')
        and metadata['source'] == 'fmp_spy_daily',
        'PROVENANCE',
        'Invalid calendar/source provenance',
    )
    require(
        metadata['expected_date'] == expected
        and metadata['latest_completed_date'] == completed
        and _day(expected) <= selected <= _day(completed)
        and days[-1] == selected,
        'FRESHNESS',
        'Latest data/provenance does not match current market bounds',
    )
    generated = datetime.fromisoformat(metadata['generated_at'])
    require(
        generated.tzinfo is not None and generated.utcoffset().total_seconds() == 0 and started <= generated <= now,
        'PROVENANCE',
        'Metadata was not generated within this run in UTC',
    )
    return {
        'expected_market_date': expected,
        'latest_market_date': days[-1].isoformat(),
        'sha256': {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in REQUIRED},
    }


def validate_outputs(path, **kwargs):
    try:
        return _validate_outputs(path, **kwargs)
    except PublicationError:
        raise
    except Exception as exc:
        raise PublicationError('INVALID_OUTPUT', redact(str(exc))) from None


def _preflight():
    import market_breadth as engine

    require(
        hasattr(engine, 'resolve_market_end_date') and hasattr(engine, 'compute_breadth_coverage'),
        'PREREQUISITE',
        'Producer requires Issue #7 coverage and Issue #8 market-date changes',
    )
    import market_dates  # noqa: F401


def _run_analysis(output_dir, short_ma, grace):
    import market_breadth as engine

    original_dir, original_plot, original_argv = engine.reports_dir, engine.plot_breadth_and_sp500_with_peaks, sys.argv

    def plot(*args, **kwargs):
        kwargs['output_dir'] = str(output_dir)
        return original_plot(*args, **kwargs)

    try:
        engine.reports_dir = output_dir
        engine.plot_breadth_and_sp500_with_peaks = plot
        sys.argv = [
            'market_breadth.py',
            '--include_50ma',
            '--short_ma',
            str(short_ma),
            '--data-arrival-grace-minutes',
            str(grace),
        ]
        result = engine.main()
        require(result in (None, 0), 'ANALYSIS', 'Analysis returned a failure status')
    finally:
        engine.reports_dir, engine.plot_breadth_and_sp500_with_peaks, sys.argv = (
            original_dir,
            original_plot,
            original_argv,
        )


def _emit(value, stream):
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    if value:
        print(redact(value), file=stream, end='')


def _analysis_process(path, short_ma, grace):
    bootstrap = (
        'from pathlib import Path; import sys; from scripts.publish_market_breadth import _run_analysis; '
        '_run_analysis(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))'
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', bootstrap, str(path), str(short_ma), str(grace)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=720,
            check=False,
        )  # nosec B603
    except subprocess.TimeoutExpired as exc:
        _emit(exc.stdout, sys.stdout)
        _emit(exc.stderr, sys.stderr)
        raise PublicationError('ANALYSIS_TIMEOUT', 'Analysis timed out') from None
    _emit(result.stdout, sys.stdout)
    _emit(result.stderr, sys.stderr)
    require(result.returncode == 0, 'ANALYSIS', 'Analysis process failed')


def _revision():
    git = shutil.which('git')
    require(git is not None, 'SOURCE', 'Git executable unavailable')
    output = subprocess.check_output([git, 'rev-parse', '--verify', 'HEAD'], cwd=ROOT, text=True)  # nosec B603
    return output.strip()


def publish_run(path, *, analyze=None, clock=None, source_revision=None, min_coverage=0.9, short_ma=8, grace=30):
    """Return nonzero and a redacted diagnostic on any generation/validation failure."""
    register_secret(os.environ.get('FMP_API_KEY'))
    clock = clock or (lambda: datetime.now(timezone.utc))
    path = Path(path).absolute()
    started = clock()
    report = {'status': 'FAIL', 'reason_code': 'UNKNOWN', 'run_started_at': started.isoformat()}
    created = False
    out, err = io.StringIO(), io.StringIO()
    try:
        path.mkdir(parents=True, exist_ok=False)
        created = True
        revision = source_revision or _revision()
        require(re.fullmatch('[0-9a-f]{40}', revision), 'SOURCE', 'Invalid Git revision')
        report['source_revision'] = revision
        require(
            short_ma > 0
            and math.isfinite(grace)
            and grace >= 0
            and math.isfinite(min_coverage)
            and 0 < min_coverage <= 1,
            'CONFIG',
            'Invalid publication configuration',
        )
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if analyze is None:
                _preflight()
                register_secret(os.environ.get('FMP_API_KEY'))
                _analysis_process(path, short_ma, grace)
            else:
                result = analyze(path)
                require(result in (None, 0), 'ANALYSIS', 'Analysis returned failure')
        for name in STATIC_ASSETS:
            source = ROOT / 'reports' / name
            require(source.is_file() and not source.is_symlink(), 'STATIC', f'Missing static asset: {name}')
            shutil.copyfile(source, path / name)
        report.update(
            validate_outputs(
                path, started=started, now=clock(), min_coverage=min_coverage, short_ma=short_ma, grace=grace
            )
        )
        (path / '.nojekyll').touch()
        report.update(status='PASS', reason_code='OK', message='All publication checks passed')
    except Exception as exc:
        report.update(
            reason_code=getattr(exc, 'code', 'OUTPUT_EXISTS' if isinstance(exc, FileExistsError) else 'INVALID_OUTPUT'),
            message=redact(str(exc)),
        )
    finally:
        _emit(out.getvalue(), sys.stdout)
        _emit(err.getvalue(), sys.stderr)
    report['generated_at'] = clock().isoformat()
    payload = redact(json.dumps(report, indent=2)) + '\n'
    if created:
        (path / 'validation.json').write_text(payload)
    else:
        print(payload, file=sys.stderr)
    return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--min-coverage', type=float, default=0.9)
    parser.add_argument('--short-ma', type=int, default=8)
    parser.add_argument('--data-arrival-grace-minutes', type=float, default=30)
    args = parser.parse_args()
    return publish_run(
        args.output_dir, min_coverage=args.min_coverage, short_ma=args.short_ma, grace=args.data_arrival_grace_minutes
    )


if __name__ == '__main__':
    raise SystemExit(main())
