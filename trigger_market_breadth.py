#!/usr/bin/env python3
"""
Market Breadth data fetcher and workflow trigger.

Checks freshness of GitHub Pages CSV data by parsing the actual market
dates in the content (not just HTTP Last-Modified). If stale, triggers
the daily-market-breadth workflow via GitHub API.

Usage:
    python trigger_market_breadth.py                  # Auto: fetch if fresh, trigger if stale
    python trigger_market_breadth.py --trigger-only   # Force trigger workflow
    python trigger_market_breadth.py --fetch-only     # Fetch CSV only (no trigger)
    python trigger_market_breadth.py --max-age 6      # Last-Modified ceiling in hours
"""

import argparse
import csv
import io
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from market_dates import resolve_market_date
from secret_redaction import redact, register_secret

load_dotenv()

BASE_URL = 'https://tradermonty.github.io/market-breadth-analysis'
DATA_CSV_URL = f'{BASE_URL}/market_breadth_data.csv'
SUMMARY_CSV_URL = f'{BASE_URL}/market_breadth_summary.csv'

GITHUB_API_URL = (
    'https://api.github.com/repos/tradermonty/market-breadth-analysis'
    '/actions/workflows/daily-market-breadth.yml/dispatches'
)

DATE_COLUMN = 'Date'
SUMMARY_END_DATE_KEY = 'Analysis Period End'


@dataclass(frozen=True)
class ContentValidation:
    is_valid: bool
    latest_data_date: str | None
    summary_end_date: str | None
    row_count: int
    error: str | None


@dataclass(frozen=True)
class FreshnessResult:
    is_fresh: bool
    latest_data_date: str | None
    expected_date: str | None
    reason: str
    resolution: object = None


def _get_github_token():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise OSError('GITHUB_TOKEN not set. Add it to .env or export it.')
    register_secret(token)
    return token


def _parse_last_modified(resp):
    """Extract Last-Modified from a response as aware datetime, or None."""
    lm = resp.headers.get('Last-Modified')
    if not lm:
        return None
    from email.utils import parsedate_to_datetime

    return parsedate_to_datetime(lm)


def validate_csv_content(csv_text, summary_text):
    """Validate CSV content and extract the latest data date.

    Checks for HTML error responses, empty content, missing date columns,
    and consistency between data CSV and summary CSV.
    """
    if not csv_text or not csv_text.strip():
        return ContentValidation(False, None, None, 0, 'Empty CSV content')

    stripped = csv_text.strip()
    if stripped.startswith(('<!DOCTYPE', '<html', '<HTML')):
        return ContentValidation(False, None, None, 0, 'Response is HTML, not CSV')

    if not summary_text or not summary_text.strip():
        return ContentValidation(False, None, None, 0, 'Empty summary CSV content')

    summary_stripped = summary_text.strip()
    if summary_stripped.startswith(('<!DOCTYPE', '<html')):
        return ContentValidation(False, None, None, 0, 'Summary response is HTML, not CSV')

    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        if DATE_COLUMN not in (reader.fieldnames or []):
            return ContentValidation(False, None, None, 0, f'Missing {DATE_COLUMN!r} column in data CSV')

        dates = []
        row_count = 0
        for row in reader:
            row_count += 1
            date_val = row.get(DATE_COLUMN, '').strip()
            if date_val:
                dates.append(date_val)
    except Exception as e:
        return ContentValidation(False, None, None, 0, f'Failed to parse data CSV: {e}')

    if not dates:
        return ContentValidation(False, None, None, row_count, 'No date values found in data CSV')

    dates.sort()
    latest_data_date = dates[-1]

    try:
        datetime.strptime(latest_data_date, '%Y-%m-%d')
    except ValueError:
        return ContentValidation(False, None, None, row_count, f'Invalid date format in data CSV: {latest_data_date!r}')

    summary_end_date = None
    try:
        summary_reader = csv.DictReader(io.StringIO(summary_text))
        for srow in summary_reader:
            metric = srow.get('Metric', '').strip()
            if metric == SUMMARY_END_DATE_KEY:
                summary_end_date = srow.get('Value', '').strip()
                break
    except Exception:
        return ContentValidation(False, latest_data_date, None, row_count, 'Failed to parse summary CSV')

    if not summary_end_date:
        return ContentValidation(
            False, latest_data_date, None, row_count, f'Missing {SUMMARY_END_DATE_KEY!r} in summary CSV'
        )

    if summary_end_date != latest_data_date:
        return ContentValidation(
            False,
            latest_data_date,
            summary_end_date,
            row_count,
            f'Data/summary date mismatch: data ends {latest_data_date}, summary says {summary_end_date}',
        )

    return ContentValidation(True, latest_data_date, summary_end_date, row_count, None)


def determine_freshness(csv_text, *, now=None):
    """Determine feed freshness by comparing CSV data dates to expected market sessions.

    Uses ``market_dates.resolve_market_date()`` with CSV rows as observations.
    """
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        observations = []
        for row in reader:
            date_val = row.get(DATE_COLUMN, '').strip()
            close_val = row.get('Close') or row.get('close') or row.get('Breadth_Index_200')
            if date_val and close_val:
                try:
                    observations.append({'date': date_val, 'close': float(close_val)})
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        return FreshnessResult(False, None, None, f'Failed to parse CSV for freshness: {e}')

    if not observations:
        return FreshnessResult(False, None, None, 'No valid observations in CSV')

    observations.sort(key=lambda r: r['date'])
    latest_data_date = observations[-1]['date']

    def fetch_observations(start, end):
        return [o for o in observations if start <= o['date'] <= end]

    try:
        resolution = resolve_market_date(fetch_observations, now=now, source='published_csv', grace_minutes=30)
    except Exception as e:
        return FreshnessResult(False, latest_data_date, None, f'Market date resolution failed: {e}')

    if resolution.confidence == 'weekday_only':
        is_fresh = latest_data_date >= resolution.selected_date
        return FreshnessResult(
            is_fresh=is_fresh,
            latest_data_date=latest_data_date,
            expected_date=resolution.selected_date,
            reason=f'weekday_estimate (data={latest_data_date}, expected={resolution.selected_date})',
            resolution=resolution,
        )

    return FreshnessResult(
        is_fresh=resolution.publication_eligible,
        latest_data_date=latest_data_date,
        expected_date=resolution.expected_date,
        reason=resolution.reason,
        resolution=resolution,
    )


def fetch_csv():
    """Download the market breadth CSV from GitHub Pages.

    Returns:
        dict with keys: csv_text, summary_text, last_modified
    """
    data_resp = requests.get(DATA_CSV_URL, timeout=30)
    data_resp.raise_for_status()

    summary_resp = requests.get(SUMMARY_CSV_URL, timeout=30)
    summary_resp.raise_for_status()

    lm = _parse_last_modified(data_resp)

    return {
        'csv_text': data_resp.text,
        'summary_text': summary_resp.text,
        'last_modified': lm,
    }


def trigger_workflow():
    """Trigger the daily-market-breadth workflow via GitHub API.

    Returns:
        dict with status info
    """
    token = _get_github_token()
    resp = requests.post(
        GITHUB_API_URL,
        headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        },
        json={'ref': 'main'},
        timeout=15,
    )
    resp.raise_for_status()
    return {
        'status': 'triggered',
        'message': 'Workflow triggered. Data will be ready in ~5 minutes.',
        'runs_url': (
            'https://github.com/tradermonty/market-breadth-analysis/actions/workflows/daily-market-breadth.yml'
        ),
    }


def fetch_market_breadth(max_age_hours=12, *, now=None):
    """Fetch market breadth data. Trigger workflow if content is stale.

    Primary freshness check is content-based: the latest market date in the
    CSV is compared to the expected completed session via exchange_calendars.

    ``max_age_hours`` is a secondary Last-Modified ceiling kept for backward
    compatibility: if Last-Modified age exceeds it, a warning is emitted even
    when content appears fresh.

    Returns:
        dict with keys:
          - status: "fresh" | "triggered" | "error"
          - data_url / summary_url: CSV URLs
          - last_modified: Last-Modified datetime (if available)
          - csv_text: CSV content (only when status=="fresh")
          - summary_text: Summary CSV content (only when status=="fresh")
          - message: Human-readable status description
          - freshness: FreshnessResult (when content was checked)
          - validation: ContentValidation (when content was checked)
    """
    base = {
        'data_url': DATA_CSV_URL,
        'summary_url': SUMMARY_CSV_URL,
    }

    try:
        result = fetch_csv()
    except requests.RequestException as e:
        return {
            **base,
            'status': 'error',
            'last_modified': None,
            'message': f'Failed to download CSV data: {redact(str(e))}',
        }

    csv_text = result['csv_text']
    summary_text = result['summary_text']
    last_modified = result['last_modified']

    validation = validate_csv_content(csv_text, summary_text)
    if not validation.is_valid:
        return {
            **base,
            'status': 'error',
            'last_modified': last_modified,
            'validation': validation,
            'message': f'Invalid CSV content: {validation.error}',
        }

    freshness = determine_freshness(csv_text, now=now)

    if freshness.is_fresh:
        age_warning = ''
        if last_modified and max_age_hours is not None:
            check_now = now or datetime.now(timezone.utc)
            age_hours = (check_now - last_modified).total_seconds() / 3600
            if age_hours > max_age_hours:
                age_warning = f' (warning: Last-Modified age {age_hours:.1f}h exceeds {max_age_hours}h ceiling)'

        return {
            **base,
            'status': 'fresh',
            'last_modified': last_modified,
            'csv_text': csv_text,
            'summary_text': summary_text,
            'validation': validation,
            'freshness': freshness,
            'message': f'Data is fresh (latest: {freshness.latest_data_date}).{age_warning}',
        }

    # Content is stale — trigger workflow
    try:
        trigger_result = trigger_workflow()
    except OSError as e:
        return {
            **base,
            'status': 'error',
            'last_modified': last_modified,
            'freshness': freshness,
            'validation': validation,
            'message': redact(str(e)),
        }
    except requests.RequestException as e:
        return {
            **base,
            'status': 'error',
            'last_modified': last_modified,
            'freshness': freshness,
            'validation': validation,
            'message': f'Failed to trigger workflow: {redact(str(e))}',
        }

    return {
        **base,
        **trigger_result,
        'last_modified': last_modified,
        'freshness': freshness,
        'validation': validation,
        'message': (
            f'Data is stale (latest: {freshness.latest_data_date}, '
            f'expected: {freshness.expected_date}). {trigger_result["message"]}'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description='Fetch market breadth data or trigger workflow')
    parser.add_argument(
        '--trigger-only', action='store_true', help='Force trigger the workflow without checking freshness'
    )
    parser.add_argument('--fetch-only', action='store_true', help='Fetch CSV data only (no workflow trigger)')
    parser.add_argument(
        '--max-age',
        type=float,
        default=12,
        help='Last-Modified age ceiling in hours (secondary check; default: 12). '
        'Primary freshness is determined from CSV content market dates.',
    )
    args = parser.parse_args()

    if args.trigger_only and args.fetch_only:
        print('Error: --trigger-only and --fetch-only are mutually exclusive.')
        sys.exit(1)

    if args.trigger_only:
        try:
            result = trigger_workflow()
        except (OSError, requests.RequestException) as e:
            print(f'Error: {redact(str(e))}', file=sys.stderr)
            sys.exit(1)
        print(result['message'])
        print(f'Monitor: {result["runs_url"]}')
        return

    if args.fetch_only:
        try:
            result = fetch_csv()
        except requests.RequestException as e:
            print(f'Error: {redact(str(e))}', file=sys.stderr)
            sys.exit(1)
        lm = result['last_modified']
        print(f'Last-Modified: {lm.isoformat() if lm else "unknown"}')

        validation = validate_csv_content(result['csv_text'], result['summary_text'])
        if not validation.is_valid:
            print(f'Validation error: {validation.error}', file=sys.stderr)
            sys.exit(1)
        print(f'Latest data date: {validation.latest_data_date}')
        print(f'Data rows: {validation.row_count}')
        print(result['csv_text'][:500])
        return

    # Default: auto mode
    result = fetch_market_breadth(max_age_hours=args.max_age)
    print(f'Status: {result["status"]}')
    print(f'Message: {result["message"]}')
    if result.get('last_modified'):
        print(f'Last-Modified: {result["last_modified"].isoformat()}')
    if result['status'] == 'fresh':
        lines = result['csv_text'].strip().split('\n')
        print(f'Data rows: {len(lines) - 1}')
        print(f'Preview:\n{chr(10).join(lines[:3])}')


if __name__ == '__main__':
    main()
