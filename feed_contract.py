"""Public market breadth feed URLs and the CSV content contract.

Shared by trigger_market_breadth.py and scripts/monitor_feed.py. This module
must not load .env or read credentials, so importing it has no side effects.
"""

import csv
import io
import math
from dataclasses import dataclass
from datetime import date

from market_dates import NY, _utc_now

BASE_URL = 'https://tradermonty.github.io/market-breadth-analysis'
DATA_CSV_URL = f'{BASE_URL}/market_breadth_data.csv'
SUMMARY_CSV_URL = f'{BASE_URL}/market_breadth_summary.csv'

DATE_COLUMN = 'Date'
SUMMARY_END_DATE_KEY = 'Analysis Period End'
PRICE_COLUMN = 'S&P500_Price'
BREADTH_COLUMNS = ('Breadth_Index_Raw', 'Breadth_Index_200MA')
REQUIRED_COLUMNS = (PRICE_COLUMN, *BREADTH_COLUMNS)


@dataclass(frozen=True)
class ContentValidation:
    is_valid: bool
    latest_data_date: str | None
    summary_end_date: str | None
    row_count: int
    error: str | None


def _is_html(text):
    return text.strip().lower().startswith(('<!doctype', '<html'))


def _parse_iso_date(value):
    """Return a date for a strict YYYY-MM-DD string, or None."""
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.isoformat() == value else None


def _latest_row_error(row):
    """Describe why the latest row cannot be published, or return None."""
    for column in REQUIRED_COLUMNS:
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            return f'Invalid {column} on latest row: {row[column]!r}'
        if not math.isfinite(value):
            return f'Invalid {column} on latest row: {row[column]!r}'
        if column == PRICE_COLUMN and value <= 0:
            return f'Invalid {column} on latest row: {row[column]!r}'
        if column in BREADTH_COLUMNS and not 0 <= value <= 1:
            return f'Invalid {column} on latest row: {row[column]!r}'
    return None


def validate_csv_content(csv_text, summary_text, *, now=None):
    """Validate CSV content and extract the latest data date.

    Rejects HTML error responses, empty content, missing columns, malformed,
    unsorted, duplicate or future dates, unusable latest values, and any
    disagreement between the data CSV and the summary CSV.
    """
    market_today = _utc_now(now).astimezone(NY).date()

    if not csv_text or not csv_text.strip():
        return ContentValidation(False, None, None, 0, 'Empty CSV content')

    if _is_html(csv_text):
        return ContentValidation(False, None, None, 0, 'Response is HTML, not CSV')

    if not summary_text or not summary_text.strip():
        return ContentValidation(False, None, None, 0, 'Empty summary CSV content')

    if _is_html(summary_text):
        return ContentValidation(False, None, None, 0, 'Summary response is HTML, not CSV')

    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        if DATE_COLUMN not in (reader.fieldnames or []):
            return ContentValidation(False, None, None, 0, f'Missing {DATE_COLUMN!r} column in data CSV')

        fieldnames = reader.fieldnames
        rows = list(reader)
    except Exception as e:
        return ContentValidation(False, None, None, 0, f'Failed to parse data CSV: {e}')

    row_count = len(rows)
    if not rows:
        return ContentValidation(False, None, None, 0, 'No date values found in data CSV')

    dates = []
    for row in rows:
        raw_date = (row[DATE_COLUMN] or '').strip()
        parsed = _parse_iso_date(raw_date)
        if parsed is None:
            return ContentValidation(False, None, None, row_count, f'Invalid date format in data CSV: {raw_date!r}')
        dates.append(parsed)

    if dates != sorted(set(dates)):
        return ContentValidation(False, None, None, row_count, 'Dates in data CSV must be sorted and unique')

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        return ContentValidation(False, None, None, row_count, f'Missing columns in data CSV: {missing_columns}')

    latest_row_error = _latest_row_error(rows[-1])
    if latest_row_error:
        return ContentValidation(False, None, None, row_count, latest_row_error)

    latest_data_date = dates[-1].isoformat()
    if dates[-1] > market_today:
        return ContentValidation(
            False, latest_data_date, None, row_count, f'Data CSV has future date {latest_data_date}'
        )

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
