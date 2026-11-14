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
    python trigger_market_breadth.py --max-age 6      # Warn if Last-Modified is older (hours)
"""

import argparse
import csv
import io
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from feed_contract import (  # noqa: F401  re-exported for existing callers
    BASE_URL,
    BREADTH_COLUMNS,
    DATA_CSV_URL,
    DATE_COLUMN,
    PRICE_COLUMN,
    REQUIRED_COLUMNS,
    SUMMARY_CSV_URL,
    SUMMARY_END_DATE_KEY,
    ContentValidation,
    _is_html,
    _latest_row_error,
    _parse_iso_date,
    validate_csv_content,
)
from market_dates import _utc_now, resolve_market_date
from secret_redaction import redact, register_secret

load_dotenv()


GITHUB_API_URL = (
    'https://api.github.com/repos/tradermonty/market-breadth-analysis'
    '/actions/workflows/daily-market-breadth.yml/dispatches'
)
GITHUB_RUNS_URL = (
    'https://api.github.com/repos/tradermonty/market-breadth-analysis/actions/workflows/daily-market-breadth.yml/runs'
)
# A run started this recently is given time to publish before another dispatch.
DISPATCH_COOLDOWN = timedelta(minutes=60)


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


def _github_headers(token):
    return {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }


def determine_freshness(csv_text, *, now=None):
    """Determine feed freshness by comparing CSV data dates to expected market sessions.

    Uses ``market_dates.resolve_market_date()`` with CSV rows as observations.
    """
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        observations = []
        for row in reader:
            date_val = (row.get(DATE_COLUMN) or '').strip()
            if _parse_iso_date(date_val) is not None:
                observations.append({'date': date_val, 'close': 1.0})
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
        # Without exchange_calendars, the estimate ignores holidays: be
        # conservative and allow data to lag the estimate by one business day.
        expected = date.fromisoformat(resolution.selected_date)
        latest = date.fromisoformat(latest_data_date)
        prev_business = expected - timedelta(days=1)
        while prev_business.weekday() >= 5:
            prev_business -= timedelta(days=1)
        is_fresh = latest >= prev_business
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


def find_blocking_run(now=None):
    """Return a workflow run that should suppress a new dispatch, or None.

    A run blocks while it is unfinished, and for DISPATCH_COOLDOWN after it was
    created so a failing or provider-delayed publish cannot cause a dispatch loop.
    """
    token = _get_github_token()
    now = _utc_now(now)
    resp = requests.get(GITHUB_RUNS_URL, headers=_github_headers(token), params={'per_page': 10}, timeout=15)
    resp.raise_for_status()
    for run in resp.json()['workflow_runs']:
        created_at = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
        if run['status'] != 'completed' or now - created_at < DISPATCH_COOLDOWN:
            return run
    return None


def trigger_workflow():
    """Trigger the daily-market-breadth workflow via GitHub API.

    Returns:
        dict with status info
    """
    token = _get_github_token()
    resp = requests.post(
        GITHUB_API_URL,
        headers=_github_headers(token),
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

    ``max_age_hours`` no longer decides freshness or triggers a workflow: if
    Last-Modified age exceeds it, the message carries a warning.

    Stale content dispatches the workflow only when no run is unfinished or was
    created within ``DISPATCH_COOLDOWN``. If that check fails, nothing is dispatched.

    Returns:
        dict with keys:
          - status: "fresh" | "triggered" | "pending" | "error"
          - data_url / summary_url: CSV URLs
          - last_modified: Last-Modified datetime (if available)
          - csv_text: CSV content (only when status=="fresh")
          - summary_text: Summary CSV content (only when status=="fresh")
          - message: Human-readable status description
          - freshness: FreshnessResult (when content was checked)
          - validation: ContentValidation (when content was checked)
    """
    if max_age_hours is not None and (not math.isfinite(max_age_hours) or max_age_hours < 0):
        raise ValueError(f'max_age_hours must be finite and nonnegative: {max_age_hours}')

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

    validation = validate_csv_content(csv_text, summary_text, now=now)
    if not validation.is_valid:
        return {
            **base,
            'status': 'error',
            'last_modified': last_modified,
            'validation': validation,
            'message': f'Invalid CSV content: {validation.error}',
        }

    freshness = determine_freshness(csv_text, now=now)
    if freshness.resolution is None:
        return {
            **base,
            'status': 'error',
            'last_modified': last_modified,
            'freshness': freshness,
            'validation': validation,
            'message': f'Could not determine freshness: {freshness.reason}',
        }
    estimate_warning = (
        ' (warning: exchange_calendars unavailable, so holidays are not considered)'
        if freshness.resolution.confidence == 'weekday_only'
        else ''
    )

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
            'message': f'Data is fresh (latest: {freshness.latest_data_date}).{estimate_warning}{age_warning}',
        }

    stale_message = (
        f'Data is stale (latest: {freshness.latest_data_date}, expected: {freshness.expected_date}).{estimate_warning}'
    )

    # Content is stale — trigger workflow unless a run is already covering it
    try:
        blocking_run = find_blocking_run(now=now)
        if blocking_run is not None:
            return {
                **base,
                'status': 'pending',
                'last_modified': last_modified,
                'freshness': freshness,
                'validation': validation,
                'message': (
                    f'{stale_message} Not dispatching because a recent or unfinished run exists: '
                    f'{blocking_run.get("html_url", "unknown run")}'
                ),
            }
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
    except (KeyError, TypeError, ValueError) as e:
        return {
            **base,
            'status': 'error',
            'last_modified': last_modified,
            'freshness': freshness,
            'validation': validation,
            'message': f'Unexpected workflow run response: {redact(repr(e))}',
        }

    return {
        **base,
        **trigger_result,
        'last_modified': last_modified,
        'freshness': freshness,
        'validation': validation,
        'message': f'{stale_message} {trigger_result["message"]}',
    }


def _max_age_hours(value):
    hours = float(value)
    if not math.isfinite(hours) or hours < 0:
        raise argparse.ArgumentTypeError(f'must be finite and nonnegative: {value}')
    return hours


def main():
    parser = argparse.ArgumentParser(description='Fetch market breadth data or trigger workflow')
    parser.add_argument(
        '--trigger-only', action='store_true', help='Force trigger the workflow without checking freshness'
    )
    parser.add_argument('--fetch-only', action='store_true', help='Fetch CSV data only (no workflow trigger)')
    parser.add_argument(
        '--max-age',
        type=_max_age_hours,
        default=12,
        help='Warn when Last-Modified is older than this many hours (default: 12). '
        'Freshness and workflow triggering are decided from the market dates in the CSV.',
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
    if result['status'] == 'error':
        sys.exit(1)


if __name__ == '__main__':
    main()
