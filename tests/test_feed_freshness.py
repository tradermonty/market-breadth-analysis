"""Tests for Issue #13: Feed freshness from market dates and CSV content consistency.

All tests mock network calls. No real HTTP requests or workflow dispatches.
"""

import os
import sys
from datetime import datetime, timezone
from unittest import mock

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trigger_market_breadth import (
    FreshnessResult,
    determine_freshness,
    fetch_market_breadth,
    validate_csv_content,
)

# --- Fixtures ---

VALID_DATA_CSV = """\
Date,Breadth_Index_200,SP500_Close
2026-10-20,0.65,5800.0
2026-10-21,0.64,5810.0
2026-10-22,0.63,5820.0
2026-10-23,0.62,5830.0
"""

VALID_SUMMARY_CSV = """\
Metric,Value
Average Peaks (200MA),0.628
Analysis Period Start,2026-01-02
Analysis Period End,2026-10-23
Total Trading Days,200
"""

OLD_DATA_CSV = """\
Date,Breadth_Index_200,SP500_Close
2020-01-02,0.65,3200.0
2020-01-03,0.64,3210.0
"""

OLD_SUMMARY_CSV = """\
Metric,Value
Analysis Period End,2020-01-03
"""

MISMATCHED_SUMMARY_CSV = """\
Metric,Value
Analysis Period End,2026-09-15
"""

HTML_RESPONSE = '<!DOCTYPE html><html><body>404 Not Found</body></html>'

HEADER_ONLY_CSV = 'Date,Breadth_Index_200,SP500_Close\n'


def _make_friday_csv(friday_date):
    return f"""\
Date,Breadth_Index_200,SP500_Close
{friday_date},0.65,5800.0
"""


def _make_summary(end_date):
    return f"""\
Metric,Value
Analysis Period End,{end_date}
"""


# --- validate_csv_content tests ---


class TestValidateCsvContent:
    def test_valid_csv_and_summary(self):
        result = validate_csv_content(VALID_DATA_CSV, VALID_SUMMARY_CSV)
        assert result.is_valid
        assert result.latest_data_date == '2026-10-23'
        assert result.summary_end_date == '2026-10-23'
        assert result.row_count == 4
        assert result.error is None

    def test_html_response_rejected(self):
        result = validate_csv_content(HTML_RESPONSE, VALID_SUMMARY_CSV)
        assert not result.is_valid
        assert 'HTML' in result.error

    def test_empty_csv_rejected(self):
        result = validate_csv_content('', VALID_SUMMARY_CSV)
        assert not result.is_valid
        assert 'Empty' in result.error

    def test_header_only_csv_rejected(self):
        result = validate_csv_content(HEADER_ONLY_CSV, VALID_SUMMARY_CSV)
        assert not result.is_valid
        assert 'No date values' in result.error

    def test_missing_date_column_rejected(self):
        no_date = 'Value,Close\n0.5,100\n'
        result = validate_csv_content(no_date, VALID_SUMMARY_CSV)
        assert not result.is_valid
        assert 'Date' in result.error

    def test_data_summary_date_mismatch_rejected(self):
        result = validate_csv_content(VALID_DATA_CSV, MISMATCHED_SUMMARY_CSV)
        assert not result.is_valid
        assert 'mismatch' in result.error
        assert '2026-10-23' in result.error
        assert '2026-09-15' in result.error

    def test_html_summary_rejected(self):
        result = validate_csv_content(VALID_DATA_CSV, HTML_RESPONSE)
        assert not result.is_valid
        assert 'Summary' in result.error and 'HTML' in result.error

    def test_empty_summary_rejected(self):
        result = validate_csv_content(VALID_DATA_CSV, '')
        assert not result.is_valid
        assert 'Empty summary' in result.error

    def test_unparseable_summary_rejected(self):
        result = validate_csv_content(VALID_DATA_CSV, '\x00\x01\x02')
        assert not result.is_valid

    def test_invalid_date_format_rejected(self):
        bad_date_csv = 'Date,Close\nnot-a-date,100\n'
        result = validate_csv_content(bad_date_csv, _make_summary('not-a-date'))
        assert not result.is_valid
        assert 'Invalid date' in result.error


# --- determine_freshness tests ---


class TestDetermineFreshness:
    def test_fresh_data_current_weekday(self):
        # Thursday 2026-10-22 at 22:00 UTC (after market close)
        # Data has 2026-10-22 → should be fresh
        now = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
        csv_text = 'Date,Breadth_Index_200\n2026-10-20,0.65\n2026-10-21,0.64\n2026-10-22,0.63\n'
        result = determine_freshness(csv_text, now=now)
        assert result.is_fresh
        assert result.latest_data_date == '2026-10-22'

    def test_stale_data_old_content(self):
        # Current time is 2026-10-22 but data ends 2020-01-03
        now = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
        result = determine_freshness(OLD_DATA_CSV, now=now)
        assert not result.is_fresh

    def test_friday_data_fresh_on_weekend(self):
        # Friday 2026-10-23 data, checked on Saturday 2026-10-24
        now = datetime(2026, 10, 24, 15, 0, tzinfo=timezone.utc)
        csv_text = _make_friday_csv('2026-10-23')
        result = determine_freshness(csv_text, now=now)
        assert result.is_fresh
        assert result.latest_data_date == '2026-10-23'

    def test_friday_data_fresh_on_sunday(self):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        csv_text = _make_friday_csv('2026-10-23')
        result = determine_freshness(csv_text, now=now)
        assert result.is_fresh

    def test_holiday_wednesday_fresh_on_thanksgiving(self):
        # Thanksgiving 2026 is Nov 26 (Thursday). Market closed.
        # Wednesday Nov 25 data should be fresh on Thursday Nov 26.
        now = datetime(2026, 11, 26, 20, 0, tzinfo=timezone.utc)
        csv_text = 'Date,Breadth_Index_200\n2026-11-25,0.65\n2026-11-24,0.64\n'
        result = determine_freshness(csv_text, now=now)
        assert result.is_fresh

    def test_no_observations_not_fresh(self):
        csv_text = 'Date,Close\n'
        result = determine_freshness(csv_text)
        assert not result.is_fresh
        assert 'No valid observations' in result.reason

    def test_exchange_calendars_unavailable_fallback(self):
        now = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
        csv_text = 'Date,Breadth_Index_200\n2026-10-22,0.63\n'
        with mock.patch('market_dates._get_calendar', side_effect=ImportError('no module')):
            result = determine_freshness(csv_text, now=now)
        # Should still produce a result (weekday fallback)
        assert isinstance(result, FreshnessResult)
        assert result.latest_data_date == '2026-10-22'


# --- fetch_market_breadth integration tests ---


def _mock_response(text, status=200, last_modified='Thu, 22 Oct 2026 20:30:00 GMT'):
    resp = mock.Mock()
    resp.text = text
    resp.status_code = status
    resp.headers = {'Last-Modified': last_modified} if last_modified else {}
    resp.raise_for_status = mock.Mock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f'{status}')
    return resp


class TestFetchMarketBreadth:
    @mock.patch('trigger_market_breadth.requests.get')
    def test_fresh_content_returns_fresh(self, mock_get):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [
            _mock_response(VALID_DATA_CSV),
            _mock_response(VALID_SUMMARY_CSV),
        ]
        result = fetch_market_breadth(max_age_hours=12, now=now)
        assert result['status'] == 'fresh'
        assert 'csv_text' in result
        assert result['validation'].is_valid
        assert result['freshness'].is_fresh

    @mock.patch('trigger_market_breadth.trigger_workflow')
    @mock.patch('trigger_market_breadth.requests.get')
    def test_stale_content_triggers_workflow(self, mock_get, mock_trigger):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [
            _mock_response(OLD_DATA_CSV),
            _mock_response(OLD_SUMMARY_CSV),
        ]
        mock_trigger.return_value = {
            'status': 'triggered',
            'message': 'Workflow triggered.',
            'runs_url': 'https://example.com',
        }
        result = fetch_market_breadth(max_age_hours=12, now=now)
        assert result['status'] == 'triggered'
        mock_trigger.assert_called_once()

    @mock.patch('trigger_market_breadth.requests.get')
    def test_invalid_csv_returns_error(self, mock_get):
        mock_get.side_effect = [
            _mock_response(HTML_RESPONSE),
            _mock_response(VALID_SUMMARY_CSV),
        ]
        result = fetch_market_breadth()
        assert result['status'] == 'error'
        assert 'HTML' in result['message']

    @mock.patch('trigger_market_breadth.requests.get')
    def test_http_error_returns_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError('network down')
        result = fetch_market_breadth()
        assert result['status'] == 'error'
        assert 'download' in result['message'].lower() or 'failed' in result['message'].lower()

    @mock.patch('trigger_market_breadth.requests.get')
    def test_github_token_missing_returns_error(self, mock_get):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [
            _mock_response(OLD_DATA_CSV),
            _mock_response(OLD_SUMMARY_CSV),
        ]
        with mock.patch.dict(os.environ, {}, clear=True):
            result = fetch_market_breadth(max_age_hours=12, now=now)
        assert result['status'] == 'error'
        assert 'GITHUB_TOKEN' in result['message']

    @mock.patch('trigger_market_breadth.requests.get')
    def test_last_modified_age_warning(self, mock_get):
        # Content is fresh but Last-Modified is old
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [
            _mock_response(VALID_DATA_CSV, last_modified='Mon, 20 Oct 2026 10:00:00 GMT'),
            _mock_response(VALID_SUMMARY_CSV),
        ]
        result = fetch_market_breadth(max_age_hours=12, now=now)
        assert result['status'] == 'fresh'
        assert 'warning' in result['message'].lower()

    @mock.patch('trigger_market_breadth.requests.get')
    def test_data_summary_mismatch_error(self, mock_get):
        mock_get.side_effect = [
            _mock_response(VALID_DATA_CSV),
            _mock_response(MISMATCHED_SUMMARY_CSV),
        ]
        result = fetch_market_breadth()
        assert result['status'] == 'error'
        assert 'mismatch' in result['message'].lower()


# --- CLI argument tests ---


class TestCLIArguments:
    def test_trigger_only_and_fetch_only_mutual_exclusion(self):
        with mock.patch('sys.argv', ['trigger_market_breadth.py', '--trigger-only', '--fetch-only']):
            with pytest.raises(SystemExit) as exc_info:
                from trigger_market_breadth import main

                main()
            assert exc_info.value.code == 1
