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
    find_blocking_run,
    validate_csv_content,
)

# --- Fixtures ---

VALID_DATA_CSV = """\
Date,S&P500_Price,Breadth_Index_Raw,Breadth_Index_200MA
2026-10-20,5800.0,0.65,0.60
2026-10-21,5810.0,0.64,0.61
2026-10-22,5820.0,0.63,0.62
2026-10-23,5830.0,0.62,0.63
"""

VALID_SUMMARY_CSV = """\
Metric,Value
Average Peaks (200MA),0.628
Analysis Period Start,2026-01-02
Analysis Period End,2026-10-23
Total Trading Days,200
"""

OLD_DATA_CSV = """\
Date,S&P500_Price,Breadth_Index_Raw,Breadth_Index_200MA
2020-01-02,3200.0,0.65,0.60
2020-01-03,3210.0,0.64,0.61
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

HEADER = 'Date,S&P500_Price,Breadth_Index_Raw,Breadth_Index_200MA\n'
HEADER_ONLY_CSV = HEADER


def _make_friday_csv(friday_date):
    return f"""\
Date,S&P500_Price,Breadth_Index_Raw,Breadth_Index_200MA
{friday_date},5800.0,0.65,0.60
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

    @pytest.mark.parametrize('bad_date', ['2026-10-2', '1999-99-99', '20261020', ''])
    def test_invalid_date_in_non_latest_row_rejected(self, bad_date):
        csv_text = f'{HEADER}{bad_date},5800,0.6,0.6\n2026-10-23,5830,0.62,0.63\n'
        result = validate_csv_content(csv_text, _make_summary('2026-10-23'))
        assert not result.is_valid
        assert 'Invalid date' in result.error

    @pytest.mark.parametrize(
        'rows',
        ['2026-10-23,1,0.6,0.6\n2026-10-22,1,0.6,0.6\n', '2026-10-23,1,0.6,0.6\n2026-10-23,1,0.6,0.6\n'],
    )
    def test_unsorted_or_duplicate_dates_rejected(self, rows):
        result = validate_csv_content(HEADER + rows, _make_summary('2026-10-23'))
        assert not result.is_valid
        assert 'sorted and unique' in result.error

    def test_missing_required_columns_rejected(self):
        result = validate_csv_content('Date,Garbage\n2026-10-23,x\n', _make_summary('2026-10-23'))
        assert not result.is_valid
        assert 'S&P500_Price' in result.error

    @pytest.mark.parametrize(
        'latest',
        ['2026-10-23,,,', '2026-10-23,nan,0.6,0.6', '2026-10-23,-1,0.6,0.6', '2026-10-23,5830,1.5,0.6'],
    )
    def test_invalid_latest_row_values_rejected(self, latest):
        csv_text = f'{HEADER}2026-10-22,5820,0.6,0.6\n{latest}\n'
        result = validate_csv_content(csv_text, _make_summary('2026-10-23'))
        assert not result.is_valid
        assert 'latest row' in result.error

    def test_future_date_rejected(self):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        csv_text = f'{HEADER}2026-10-23,5830,0.6,0.6\n2099-01-01,5840,0.6,0.6\n'
        result = validate_csv_content(csv_text, _make_summary('2099-01-01'), now=now)
        assert not result.is_valid
        assert 'future' in result.error

    @pytest.mark.parametrize('html', ['<!doctype html><html></html>', '<HTML></HTML>'])
    def test_html_detection_is_case_insensitive(self, html):
        assert 'HTML' in validate_csv_content(html, VALID_SUMMARY_CSV).error
        assert 'HTML' in validate_csv_content(VALID_DATA_CSV, html).error


# --- determine_freshness tests ---


class TestDetermineFreshness:
    def test_fresh_data_current_weekday(self):
        # Thursday 2026-10-22 at 22:00 UTC (after market close)
        # Data has 2026-10-22 → should be fresh
        now = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
        csv_text = 'Date,S&P500_Price\n2026-10-20,5800\n2026-10-21,5810\n2026-10-22,5820\n'
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
        # At 20:00 UTC (15:00 ET, before close), weekday estimate = Nov 25
        now = datetime(2026, 11, 26, 20, 0, tzinfo=timezone.utc)
        csv_text = 'Date,S&P500_Price\n2026-11-25,5800\n2026-11-24,5790\n'
        result = determine_freshness(csv_text, now=now)
        assert result.is_fresh

    def test_holiday_wednesday_fresh_on_thanksgiving_after_close(self):
        # At 22:00 UTC (17:00 ET, after close), weekday estimate = Nov 26 (Thu).
        # Without exchange_calendars, fallback thinks Thu is a trading day.
        # The 1-day grace allows Nov 25 data to still be fresh.
        now = datetime(2026, 11, 26, 22, 0, tzinfo=timezone.utc)
        csv_text = 'Date,S&P500_Price\n2026-11-25,5800\n2026-11-24,5790\n'
        result = determine_freshness(csv_text, now=now)
        assert result.is_fresh

    def test_no_observations_not_fresh(self):
        csv_text = 'Date,Close\n'
        result = determine_freshness(csv_text)
        assert not result.is_fresh
        assert 'No valid observations' in result.reason

    def test_exchange_calendars_unavailable_fallback(self):
        now = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
        csv_text = 'Date,S&P500_Price\n2026-10-22,5820\n'
        with mock.patch('market_dates._get_calendar', side_effect=ImportError('no module')):
            result = determine_freshness(csv_text, now=now)
        assert isinstance(result, FreshnessResult)
        assert result.is_fresh
        assert result.latest_data_date == '2026-10-22'
        assert result.resolution.confidence == 'weekday_only'

    def test_fallback_allows_one_business_day_lag_for_unknown_holiday(self):
        # Thanksgiving after the close: the weekday estimate wrongly expects Thursday.
        now = datetime(2026, 11, 26, 22, 0, tzinfo=timezone.utc)
        csv_text = 'Date,S&P500_Price\n2026-11-25,5800\n'
        with mock.patch('market_dates._get_calendar', side_effect=ImportError('no module')):
            assert determine_freshness(csv_text, now=now).is_fresh

    def test_fallback_rejects_two_business_day_lag(self):
        now = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
        csv_text = 'Date,S&P500_Price\n2026-10-20,5800\n'
        with mock.patch('market_dates._get_calendar', side_effect=ImportError('no module')):
            assert not determine_freshness(csv_text, now=now).is_fresh

    def test_non_iso_row_does_not_invalidate_the_whole_window(self):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        csv_text = 'Date,S&P500_Price\n2026-10-2,1\n2026-10-23,1\n'
        assert determine_freshness(csv_text, now=now).is_fresh


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

    @mock.patch('trigger_market_breadth.find_blocking_run', return_value=None)
    @mock.patch('trigger_market_breadth.trigger_workflow')
    @mock.patch('trigger_market_breadth.requests.get')
    def test_stale_content_triggers_workflow(self, mock_get, mock_trigger, _mock_blocking):
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

    @mock.patch('trigger_market_breadth.trigger_workflow')
    @mock.patch('trigger_market_breadth.find_blocking_run')
    @mock.patch('trigger_market_breadth.requests.get')
    def test_blocking_run_skips_dispatch(self, mock_get, mock_blocking, mock_trigger):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [_mock_response(OLD_DATA_CSV), _mock_response(OLD_SUMMARY_CSV)]
        mock_blocking.return_value = {
            'status': 'in_progress',
            'created_at': '2026-10-23T21:58:00Z',
            'html_url': 'https://example.com/run/1',
        }
        result = fetch_market_breadth(now=now)
        assert result['status'] == 'pending'
        assert 'https://example.com/run/1' in result['message']
        mock_trigger.assert_not_called()

    @mock.patch('trigger_market_breadth.trigger_workflow')
    @mock.patch('trigger_market_breadth.find_blocking_run')
    @mock.patch('trigger_market_breadth.requests.get')
    def test_run_check_failure_does_not_dispatch(self, mock_get, mock_blocking, mock_trigger):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [_mock_response(OLD_DATA_CSV), _mock_response(OLD_SUMMARY_CSV)]
        mock_blocking.side_effect = requests.exceptions.ConnectionError('network down')
        result = fetch_market_breadth(now=now)
        assert result['status'] == 'error'
        assert 'network down' in result['message']
        mock_trigger.assert_not_called()

    @mock.patch('trigger_market_breadth.trigger_workflow')
    @mock.patch('trigger_market_breadth.find_blocking_run', return_value=None)
    @mock.patch('trigger_market_breadth.requests.get')
    def test_malformed_row_returns_error_without_dispatch(self, mock_get, _mock_blocking, mock_trigger):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [
            _mock_response(f'{HEADER}2026-10-2,1,0.6,0.6\n2026-10-23,1,0.6,0.6\n'),
            _mock_response(_make_summary('2026-10-23')),
        ]
        result = fetch_market_breadth(now=now)
        assert result['status'] == 'error'
        mock_trigger.assert_not_called()

    @mock.patch('trigger_market_breadth.requests.get')
    def test_invalid_schema_is_not_fresh(self, mock_get):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [
            _mock_response('Date,Garbage\n2026-10-23,x\n'),
            _mock_response(_make_summary('2026-10-23')),
        ]
        assert fetch_market_breadth(now=now)['status'] == 'error'

    @mock.patch('trigger_market_breadth.requests.get')
    def test_republished_old_content_is_not_fresh_despite_new_last_modified(self, mock_get):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [
            _mock_response(OLD_DATA_CSV, last_modified='Fri, 23 Oct 2026 21:59:00 GMT'),
            _mock_response(OLD_SUMMARY_CSV),
        ]
        with mock.patch('trigger_market_breadth.find_blocking_run', return_value={'html_url': 'u'}):
            assert fetch_market_breadth(now=now)['status'] == 'pending'

    def test_naive_now_raises_before_any_dispatch(self):
        with (
            mock.patch('trigger_market_breadth.requests.get') as mock_get,
            mock.patch('trigger_market_breadth.trigger_workflow') as mock_trigger,
        ):
            mock_get.side_effect = [_mock_response(VALID_DATA_CSV), _mock_response(VALID_SUMMARY_CSV)]
            with pytest.raises(ValueError, match='timezone-aware'):
                fetch_market_breadth(now=datetime(2026, 10, 23, 22, 0))
            mock_trigger.assert_not_called()

    @mock.patch('trigger_market_breadth.requests.get')
    def test_weekday_estimate_is_flagged_in_message(self, mock_get):
        now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
        mock_get.side_effect = [_mock_response(VALID_DATA_CSV), _mock_response(VALID_SUMMARY_CSV)]
        with mock.patch('market_dates._get_calendar', side_effect=ImportError('no module')):
            result = fetch_market_breadth(now=now)
        assert result['status'] == 'fresh'
        assert 'exchange_calendars' in result['message']

    @pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
    def test_invalid_max_age_raises(self, value):
        with mock.patch('trigger_market_breadth.requests.get') as mock_get:
            with pytest.raises(ValueError, match='max_age_hours'):
                fetch_market_breadth(max_age_hours=value)
            mock_get.assert_not_called()


# --- find_blocking_run tests ---


class TestFindBlockingRun:
    NOW = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)

    def _find(self, runs):
        resp = mock.Mock()
        resp.json.return_value = {'workflow_runs': runs}
        resp.raise_for_status = mock.Mock()
        with (
            mock.patch('trigger_market_breadth.requests.get', return_value=resp) as mock_get,
            mock.patch.dict(os.environ, {'GITHUB_TOKEN': 'dummy-token'}),
        ):
            result = find_blocking_run(now=self.NOW)
        assert mock_get.call_args.args[0].endswith('/actions/workflows/daily-market-breadth.yml/runs')
        return result

    @pytest.mark.parametrize('status', ['queued', 'in_progress', 'waiting', 'requested', 'pending'])
    def test_unfinished_run_blocks(self, status):
        run = {'status': status, 'created_at': '2026-10-23T10:00:00Z'}
        assert self._find([run]) == run

    def test_recently_created_completed_run_blocks(self):
        run = {'status': 'completed', 'created_at': '2026-10-23T21:30:00Z'}
        assert self._find([run]) == run

    def test_old_completed_run_does_not_block(self):
        assert self._find([{'status': 'completed', 'created_at': '2026-10-23T20:59:00Z'}]) is None

    def test_no_runs_does_not_block(self):
        assert self._find([]) is None

    def test_missing_token_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch('trigger_market_breadth.requests.get') as mock_get:
            with pytest.raises(OSError, match='GITHUB_TOKEN'):
                find_blocking_run(now=self.NOW)
            mock_get.assert_not_called()


# --- CLI argument tests ---


class TestCLIArguments:
    @pytest.mark.parametrize('value', ['-1', 'nan', 'inf', 'abc'])
    def test_invalid_max_age_is_rejected_by_cli(self, value):
        from trigger_market_breadth import main

        with (
            mock.patch('sys.argv', ['trigger_market_breadth.py', '--max-age', value]),
            mock.patch('trigger_market_breadth.fetch_market_breadth') as mock_fetch,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 2
        mock_fetch.assert_not_called()

    def test_zero_max_age_is_accepted(self):
        from trigger_market_breadth import main

        with (
            mock.patch('sys.argv', ['trigger_market_breadth.py', '--max-age', '0']),
            mock.patch(
                'trigger_market_breadth.fetch_market_breadth', return_value={'status': 'pending', 'message': 'm'}
            ) as mock_fetch,
        ):
            main()
        assert mock_fetch.call_args.kwargs['max_age_hours'] == 0

    def test_trigger_only_and_fetch_only_mutual_exclusion(self):
        with mock.patch('sys.argv', ['trigger_market_breadth.py', '--trigger-only', '--fetch-only']):
            with pytest.raises(SystemExit) as exc_info:
                from trigger_market_breadth import main

                main()
            assert exc_info.value.code == 1

    @mock.patch('trigger_market_breadth.requests.get')
    def test_auto_mode_error_exits_nonzero(self, mock_get):
        mock_get.side_effect = [
            _mock_response(HTML_RESPONSE),
            _mock_response(VALID_SUMMARY_CSV),
        ]
        from trigger_market_breadth import main

        with mock.patch('sys.argv', ['trigger_market_breadth.py']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
