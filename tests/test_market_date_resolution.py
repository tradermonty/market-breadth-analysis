"""Issue #8: completed session, observed availability and freshness are distinct."""

import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import market_breadth as mb
from market_dates import resolve_market_date


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Date regressions must not access the network')

    monkeypatch.setattr(socket.socket, 'connect', blocked)


def resolve(now='2026-05-01T22:00:00+00:00', rows=None, **kwargs):
    if rows is None:
        rows = [{'date': '2026-04-30', 'close': 100}, {'date': '2026-05-01', 'close': 101}]
    return resolve_market_date(lambda start, end: rows, now=datetime.fromisoformat(now), **kwargs)


@pytest.mark.parametrize('order', [[0, 1, 2], [2, 1, 0], [1, 0, 2], [0, 2, 1, 2]])
def test_01_latest_observed_date_is_order_independent(order):
    rows = [{'date': date, 'close': 100} for date in ['2026-04-29', '2026-04-30', '2026-05-01']]
    result = resolve(rows=[rows[i] for i in order])
    assert result.selected_date == '2026-05-01'
    assert result.confidence == 'observed'
    assert result.publication_eligible


@pytest.mark.parametrize(
    'now,selected',
    [
        ('2026-05-01T12:00:00+00:00', '2026-04-30'),
        ('2026-05-01T19:59:59+00:00', '2026-04-30'),
        ('2026-05-01T20:00:00+00:00', '2026-05-01'),
        ('2026-05-01T22:00:00+00:00', '2026-05-01'),
        ('2026-05-02T12:00:00+00:00', '2026-05-01'),
        ('2026-05-03T12:00:00+00:00', '2026-05-01'),
    ],
)
def test_02_pre_close_exact_close_post_close_and_weekend(now, selected):
    result = resolve(now)
    assert result.selected_date == result.latest_completed_date == selected


@pytest.mark.parametrize('now', ['2026-05-01T22:00:00+00:00', '2026-05-01T15:00:00-07:00', '2026-05-01T18:00:00-04:00'])
def test_03_equivalent_clock_timezones(now):
    assert resolve(now).selected_date == '2026-05-01'


@pytest.mark.parametrize(
    'now,expected',
    [
        ('2026-07-03T22:00:00+00:00', '2026-07-02'),  # Independence Day observed
        ('2026-04-03T22:00:00+00:00', '2026-04-02'),  # Good Friday
        ('2026-11-27T17:59:59+00:00', '2026-11-25'),  # Thanksgiving short Friday, before close
        ('2026-11-27T18:00:00+00:00', '2026-11-27'),
        ('2026-03-06T20:59:59+00:00', '2026-03-05'),  # standard time
        ('2026-03-06T21:00:00+00:00', '2026-03-06'),
        ('2026-03-09T19:59:59+00:00', '2026-03-06'),  # daylight time
        ('2026-03-09T20:00:00+00:00', '2026-03-09'),
    ],
)
def test_04_holidays_early_closes_and_dst(now, expected):
    rows = [
        {'date': date, 'close': 100}
        for date in [
            '2026-07-02',
            '2026-07-03',
            '2026-04-02',
            '2026-04-03',
            '2026-11-25',
            '2026-11-27',
            '2026-03-05',
            '2026-03-06',
            '2026-03-09',
        ]
    ]
    result = resolve(now, rows)
    assert result.selected_date == expected
    assert result.confidence == 'observed'


def test_05_delivery_grace_changes_expectation_not_actual_availability():
    old = [{'date': '2026-04-30', 'close': 100}]
    inside = resolve('2026-05-01T20:10:00+00:00', old)
    assert inside.selected_date == inside.expected_date == '2026-04-30'
    assert inside.latest_completed_date == '2026-05-01'
    assert inside.publication_eligible
    late = resolve('2026-05-01T20:30:00+00:00', old)
    assert late.selected_date == '2026-04-30'
    assert late.expected_date == '2026-05-01'
    assert not late.publication_eligible
    available = resolve('2026-05-01T20:10:00+00:00')
    assert available.selected_date == '2026-05-01'
    assert available.publication_eligible


def test_06_non_session_and_future_rows_are_not_converted_to_fake_observations():
    result = resolve(rows=[{'date': d, 'close': 100} for d in ['2026-04-30', '2026-05-02', '2026-05-04']])
    assert result.selected_date == '2026-04-30'
    assert result.confidence == 'observed'
    assert not result.publication_eligible


@pytest.mark.parametrize(
    'rows',
    [
        None,
        [],
        {},
        [{'close': 100}],
        [{'date': 'invalid', 'close': 100}],
        [{'date': '2026-05-01', 'close': 'NaN'}],
        [{'date': '2026-05-01', 'close': float('inf')}],
        [{'date': '2026-05-01', 'close': 0}],
        [{'date': '2026-05-01', 'close': -1}],
        [{'date': '2026-05-01', 'close': True}],
        [{'date': '2026-05-02', 'close': 100}],
    ],
)
def test_07_invalid_or_unavailable_observations_are_low_confidence(rows):
    result = resolve_market_date(lambda *args: rows, now=datetime.fromisoformat('2026-05-01T22:00:00+00:00'))
    assert result.selected_date == '2026-05-01'
    assert result.confidence == 'calendar_only'
    assert not result.publication_eligible


def test_08_api_error_metadata_contains_no_secret():
    fetch = Mock(side_effect=RuntimeError('https://example.invalid?apikey=DUMMY_DATE_VALUE_123'))
    result = resolve_market_date(fetch, now=datetime.fromisoformat('2026-05-01T22:00:00+00:00'))
    assert result.reason == 'lookup_failed'
    assert 'DUMMY_DATE_VALUE_123' not in str(result)
    assert not result.publication_eligible


def test_09_calendar_failure_is_explicit_weekday_estimate(monkeypatch):
    monkeypatch.setattr('market_dates._get_calendar', Mock(side_effect=ValueError('calendar unavailable')))
    result = resolve('2026-07-03T22:00:00+00:00')
    assert result.selected_date == '2026-07-03'  # May be a holiday: intentionally unverified.
    assert result.confidence == 'weekday_only'
    assert result.expected_date is None
    assert not result.publication_eligible


def test_10_naive_clocks_and_invalid_grace_are_rejected():
    with pytest.raises(ValueError, match='aware'):
        resolve_market_date(now=datetime(2026, 5, 1))
    for grace in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='grace'):
            resolve(grace_minutes=grace)


@pytest.mark.parametrize('cache', [True, False])
def test_11_saved_date_resolution_never_calls_fmp(monkeypatch, cache):
    frame = pd.DataFrame({'SPY': [100, 101]}, index=pd.to_datetime(['2026-04-30', '2026-05-01']))
    monkeypatch.setattr(mb, 'load_stock_data', lambda name: frame if cache else None)
    api = Mock(side_effect=AssertionError('FMP called during saved-date lookup'))
    monkeypatch.setattr(mb, '_get_fmp_fetcher', api)
    result = mb.resolve_market_end_date(True, now=datetime.fromisoformat('2026-05-01T22:00:00+00:00'))
    assert result.selected_date == '2026-05-01'
    assert (result.confidence == 'observed') == cache
    api.assert_not_called()


def setup_main(monkeypatch, tmp_path):
    monkeypatch.setattr(mb, 'reports_dir', tmp_path)
    dates = pd.bdate_range('2025-01-02', '2026-05-01')
    ticks = Mock(return_value=['A'])
    index = Mock(return_value=pd.Series(100.0, index=dates))
    stocks = Mock(return_value=pd.DataFrame({'A': 100.0}, index=dates))
    plot = Mock(return_value=(None, {'breadth_index_200': pd.Series(0.0, index=dates)}))
    monkeypatch.setattr(mb, 'get_sp500_tickers_from_fmp', ticks)
    monkeypatch.setattr(mb, 'get_sp500_price_data', index)
    monkeypatch.setattr(mb, 'get_multiple_stock_data', stocks)
    monkeypatch.setattr(mb, 'plot_breadth_and_sp500_with_peaks', plot)
    monkeypatch.setattr(mb, 'export_chart_data_to_csv', Mock())
    return ticks, index, stocks, plot


def test_12_main_post_close_date_and_provenance_reach_acquisition(monkeypatch, tmp_path):
    ticks, index, stocks, plot = setup_main(monkeypatch, tmp_path)
    result = resolve()
    monkeypatch.setattr(mb, 'resolve_market_end_date', lambda *args, **kwargs: result)
    monkeypatch.setattr('sys.argv', ['market_breadth.py', '--start_date', '2025-01-02'])
    mb.main()
    assert index.call_args.args[1] == stocks.call_args.args[2] == '2026-05-01'
    assert plot.called
    metadata = json.loads((tmp_path / 'market_date_resolution.json').read_text())
    assert metadata['selected_date'] == '2026-05-01'
    assert metadata['publication_eligible']
    assert metadata['calendar_name'] == 'XNYS'


def test_13_unverified_default_stops_before_analysis_but_records_reason(monkeypatch, tmp_path):
    ticks, _, _, plot = setup_main(monkeypatch, tmp_path)
    result = resolve_market_date(lambda *args: [], now=datetime.fromisoformat('2026-05-01T22:00:00+00:00'))
    monkeypatch.setattr(mb, 'resolve_market_end_date', lambda *args, **kwargs: result)
    monkeypatch.setattr('sys.argv', ['market_breadth.py', '--start_date', '2025-01-02'])
    with pytest.raises(ValueError, match='Unverified'):
        mb.main()
    ticks.assert_not_called()
    plot.assert_not_called()
    assert not json.loads((tmp_path / 'market_date_resolution.json').read_text())['publication_eligible']


@pytest.mark.parametrize('requested', ['2026-04-30', '2026-05-03'])
def test_14_explicit_end_date_is_authoritative_without_default_lookup(monkeypatch, tmp_path, requested):
    _, index, stocks, _ = setup_main(monkeypatch, tmp_path)
    default = Mock(side_effect=AssertionError('default lookup must not run'))
    monkeypatch.setattr(mb, 'resolve_market_end_date', default)
    monkeypatch.setattr('sys.argv', ['market_breadth.py', '--start_date', '2025-01-02', '--end_date', requested])
    mb.main()
    assert index.call_args.args[1] == stocks.call_args.args[2] == requested
    default.assert_not_called()
    assert json.loads((tmp_path / 'market_date_resolution.json').read_text())['source'] == 'explicit'


@pytest.mark.parametrize(
    'args',
    [
        ['--start_date', '2026-05-01', '--end_date', '2026-04-30'],
        ['--end_date', 'not-a-date'],
        ['--data-arrival-grace-minutes', '-1'],
        ['--data-arrival-grace-minutes', 'nan'],
    ],
)
def test_15_bad_cli_dates_or_grace_fail(monkeypatch, tmp_path, args):
    ticks, _, _, plot = setup_main(monkeypatch, tmp_path)
    monkeypatch.setattr('sys.argv', ['market_breadth.py', *args])
    with pytest.raises((ValueError, SystemExit)):
        mb.main()
    ticks.assert_not_called()
    plot.assert_not_called()


@pytest.mark.parametrize('bad_cache', ['multiple-columns', 'duplicate-dates', 'timezone', 'bad-price', 'bad-index'])
def test_16_corrupt_saved_date_cache_is_unverified(monkeypatch, bad_cache):
    saved = pd.DataFrame({'SPY': [100.0, 101.0]}, index=pd.to_datetime(['2026-04-30', '2026-05-01']))
    if bad_cache == 'multiple-columns':
        saved['extra'] = 100
    elif bad_cache == 'duplicate-dates':
        saved.index = pd.to_datetime(['2026-05-01', '2026-05-01'])
    elif bad_cache == 'timezone':
        saved.index = saved.index.tz_localize('UTC')
    elif bad_cache == 'bad-price':
        saved.iloc[1, 0] = float('nan')
    else:
        saved.index = ['broken', 'index']
    monkeypatch.setattr(mb, 'load_stock_data', lambda name: saved)
    result = mb.resolve_market_end_date(True, now=datetime.fromisoformat('2026-05-01T22:00:00+00:00'))
    assert result.confidence == 'calendar_only'
    assert not result.publication_eligible


def test_17_outside_calendar_horizon_does_not_claim_exchange_verification():
    lookup = Mock(side_effect=AssertionError('lookup should not run without calendar'))
    result = resolve_market_date(lookup, now=datetime.fromisoformat('1900-01-02T22:00:00+00:00'))
    assert result.confidence == 'weekday_only'
    assert result.expected_date is None
    lookup.assert_not_called()


def test_18_compatibility_helpers_use_actual_fmp_wrapper_and_bounded_dates(monkeypatch):
    fetcher = Mock()
    fetcher.get_historical_price_data.return_value = [
        {'date': '2026-04-30', 'close': 100},
        {'date': '2026-05-01', 'close': 101},
    ]
    monkeypatch.setattr(mb, '_get_fmp_fetcher', lambda: fetcher)
    now = datetime.fromisoformat('2026-05-01T22:00:00+00:00')
    assert mb.get_latest_market_date(now=now) == '2026-05-01'
    assert mb.resolve_default_end_date(False, now=now) == '2026-05-01'
    assert fetcher.get_historical_price_data.call_args.args == ('SPY', '2026-04-17', '2026-05-01')


def test_19_verified_but_stale_metadata_is_not_publication_eligible(monkeypatch, tmp_path):
    ticks, _, _, plot = setup_main(monkeypatch, tmp_path)
    result = resolve(rows=[{'date': '2026-04-30', 'close': 100}])
    monkeypatch.setattr(mb, 'resolve_market_end_date', lambda *args, **kwargs: result)
    monkeypatch.setattr('sys.argv', ['market_breadth.py', '--start_date', '2025-01-02'])
    mb.main()
    assert ticks.called and plot.called
    metadata = json.loads((tmp_path / 'market_date_resolution.json').read_text())
    assert metadata['confidence'] == 'observed'
    assert not metadata['publication_eligible']


def test_20_real_cli_returns_nonzero_on_unverified_default(tmp_path):
    root = str(Path(mb.__file__).parent)
    code = (
        'import sys; from datetime import datetime; import market_breadth as m; '
        'from market_dates import resolve_market_date; '
        'm.resolve_market_end_date=lambda *a,**k: resolve_market_date(lambda *a:[], '
        'now=datetime.fromisoformat("2026-05-01T22:00:00+00:00")); '
        'sys.argv=["market_breadth.py","--start_date","2025-01-02"]; m.main()'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=root, DOTENV_DISABLED='1', FMP_API_KEY=''),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert 'Unverified default market date' in result.stderr
    assert not json.loads((tmp_path / 'reports/market_date_resolution.json').read_text())['publication_eligible']


def test_21_real_single_column_cache_loader_and_saved_main(monkeypatch, tmp_path):
    ticks, _, _, plot = setup_main(monkeypatch, tmp_path)
    monkeypatch.setattr(mb, 'data_dir', tmp_path)
    pd.DataFrame({'adjusted_close': [100.0, 101.0]}, index=pd.to_datetime(['2026-04-30', '2026-05-01'])).to_csv(
        tmp_path / 'sp500_price_data.csv'
    )
    assert isinstance(mb.load_stock_data('sp500_price_data.csv'), pd.Series)
    api = Mock(side_effect=AssertionError('saved date lookup cannot call FMP'))
    monkeypatch.setattr(mb, '_get_fmp_fetcher', api)
    actual_resolver = mb.resolve_market_end_date
    monkeypatch.setattr(
        mb,
        'resolve_market_end_date',
        lambda saved, **kwargs: actual_resolver(
            saved, now=datetime.fromisoformat('2026-05-01T22:00:00+00:00'), **kwargs
        ),
    )
    monkeypatch.setattr('sys.argv', ['market_breadth.py', '--start_date', '2025-01-02', '--use_saved_data'])
    mb.main()
    assert ticks.called and plot.called
    metadata = json.loads((tmp_path / 'market_date_resolution.json').read_text())
    assert metadata['source'] == 'saved_spy_daily'
    assert metadata['confidence'] == 'observed'
    assert metadata['publication_eligible']
    api.assert_not_called()
