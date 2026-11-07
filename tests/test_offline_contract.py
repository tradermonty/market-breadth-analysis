"""Tests for Issue #14: --use_saved_data strict offline contract.

Verifies that when use_saved_data=True, no FMP API calls are made and
missing/insufficient cache produces clear errors instead of silent fallback.
"""

import os
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import market_breadth as mb


@pytest.fixture()
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect mb.data_dir to a temp directory."""
    monkeypatch.setattr(mb, 'data_dir', tmp_path)
    return tmp_path


@pytest.fixture()
def block_api(monkeypatch):
    """Ensure no FMP API call is made; raises if attempted."""

    def _explode(*args, **kwargs):
        raise AssertionError('FMP API call attempted in offline mode')

    monkeypatch.setattr(mb, '_get_fmp_fetcher', _explode)


def _make_stock_cache(path, tickers, start='2020-01-01', end='2024-12-31'):
    """Write a minimal sp500_all_stocks.csv."""
    dates = pd.bdate_range(start, end)
    rng = np.random.RandomState(42)
    data = pd.DataFrame(
        rng.uniform(100, 200, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    data.to_csv(path)
    return data


def _make_price_cache(path, start='2019-01-01', end='2024-12-31'):
    """Write a minimal sp500_price_data.csv (SPY-like series)."""
    dates = pd.bdate_range(start, end)
    rng = np.random.RandomState(42)
    series = pd.Series(rng.uniform(300, 500, size=len(dates)), index=dates, name='adjusted_close')
    series.to_csv(path, header=True)
    return series


# ---- get_sp500_tickers ----


class TestGetSp500Tickers:
    def test_offline_derives_tickers_from_cache(self, tmp_data_dir, block_api):
        _make_stock_cache(tmp_data_dir / 'sp500_all_stocks.csv', ['AAPL', 'MSFT', 'GOOGL'])
        tickers = mb.get_sp500_tickers_from_fmp(use_saved_data=True)
        assert set(tickers) == {'AAPL', 'MSFT', 'GOOGL'}

    def test_offline_missing_cache_raises(self, tmp_data_dir, block_api):
        with pytest.raises(RuntimeError, match='not found or empty'):
            mb.get_sp500_tickers_from_fmp(use_saved_data=True)

    def test_offline_empty_cache_raises(self, tmp_data_dir, block_api):
        (tmp_data_dir / 'sp500_all_stocks.csv').write_text('')
        with pytest.raises(RuntimeError, match='not found or empty'):
            mb.get_sp500_tickers_from_fmp(use_saved_data=True)

    def test_offline_corrupt_cache_raises(self, tmp_data_dir, block_api):
        (tmp_data_dir / 'sp500_all_stocks.csv').write_bytes(b'\x00\x01\x02')
        with pytest.raises(RuntimeError, match='(failed to read|no ticker columns)'):
            mb.get_sp500_tickers_from_fmp(use_saved_data=True)

    def test_online_calls_api(self, tmp_data_dir):
        with mock.patch.object(mb, '_get_fmp_fetcher') as mock_fmp:
            mock_fmp.return_value.get_sp500_constituents.return_value = ['AAPL']
            result = mb.get_sp500_tickers_from_fmp(use_saved_data=False)
            assert result == ['AAPL']
            mock_fmp.return_value.get_sp500_constituents.assert_called_once()


# ---- get_sp500_price_data ----


class TestGetSp500PriceData:
    def test_offline_with_valid_cache(self, tmp_data_dir, block_api):
        _make_price_cache(tmp_data_dir / 'sp500_price_data.csv')
        result = mb.get_sp500_price_data('2021-01-01', '2024-06-01', use_saved_data=True)
        assert not result.empty

    def test_offline_insufficient_range_raises(self, tmp_data_dir, block_api):
        _make_price_cache(tmp_data_dir / 'sp500_price_data.csv', start='2023-01-01', end='2023-12-31')
        with pytest.raises(RuntimeError, match='does not cover'):
            mb.get_sp500_price_data('2021-01-01', '2024-06-01', use_saved_data=True)

    def test_offline_missing_cache_raises(self, tmp_data_dir, block_api):
        with pytest.raises(RuntimeError, match='does not cover.*missing'):
            mb.get_sp500_price_data('2021-01-01', '2024-06-01', use_saved_data=True)


# ---- get_multiple_stock_data ----


class TestGetMultipleStockData:
    def test_offline_with_valid_cache(self, tmp_data_dir, block_api):
        _make_stock_cache(tmp_data_dir / 'sp500_all_stocks.csv', ['AAPL', 'MSFT'], start='2019-01-01')
        result = mb.get_multiple_stock_data(['AAPL', 'MSFT'], '2020-01-01', '2024-06-01', use_saved_data=True)
        assert not result.empty

    def test_offline_insufficient_range_raises(self, tmp_data_dir, block_api):
        _make_stock_cache(tmp_data_dir / 'sp500_all_stocks.csv', ['AAPL'], start='2023-01-01', end='2023-12-31')
        with pytest.raises(RuntimeError, match='does not cover'):
            mb.get_multiple_stock_data(['AAPL'], '2020-01-01', '2024-06-01', use_saved_data=True)

    def test_offline_missing_cache_raises(self, tmp_data_dir, block_api):
        with pytest.raises(RuntimeError, match='does not cover.*missing'):
            mb.get_multiple_stock_data(['AAPL'], '2020-01-01', '2024-06-01', use_saved_data=True)


# ---- get_stock_price_data ----


class TestGetStockPriceData:
    def _make_symbol_cache(self, path, symbol, start='2018-01-01', end='2024-12-31'):
        dates = pd.bdate_range(start, end)
        rng = np.random.RandomState(42)
        series = pd.Series(rng.uniform(100, 200, size=len(dates)), index=dates, name='adjusted_close')
        series.to_csv(path / f'{symbol}_price_data.csv', header=True)

    def test_offline_with_valid_cache(self, tmp_data_dir, block_api):
        self._make_symbol_cache(tmp_data_dir, 'SPY')
        result = mb.get_stock_price_data('SPY', '2021-01-01', '2024-06-01', use_saved_data=True)
        assert not result.empty

    def test_offline_insufficient_range_raises(self, tmp_data_dir, block_api):
        self._make_symbol_cache(tmp_data_dir, 'SPY', start='2023-01-01', end='2023-12-31')
        with pytest.raises(RuntimeError, match='does not cover'):
            mb.get_stock_price_data('SPY', '2021-01-01', '2024-06-01', use_saved_data=True)

    def test_offline_missing_cache_raises(self, tmp_data_dir, block_api):
        with pytest.raises(RuntimeError, match='does not cover.*missing'):
            mb.get_stock_price_data('SPY', '2021-01-01', '2024-06-01', use_saved_data=True)


# ---- get_stock_price_ohlc ----


class TestGetStockPriceOhlc:
    def _make_ohlc_cache(self, path, symbol, start='2018-01-01', end='2024-12-31'):
        dates = pd.bdate_range(start, end)
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                'open': rng.uniform(100, 200, len(dates)),
                'high': rng.uniform(100, 200, len(dates)),
                'low': rng.uniform(100, 200, len(dates)),
                'adjusted_close': rng.uniform(100, 200, len(dates)),
                'volume': rng.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )
        df.to_csv(path / f'{symbol}_ohlc_data.csv')

    def test_offline_with_valid_cache(self, tmp_data_dir, block_api):
        self._make_ohlc_cache(tmp_data_dir, 'SPY')
        result = mb.get_stock_price_ohlc('SPY', '2021-01-01', '2024-06-01', use_saved_data=True)
        assert not result.empty

    def test_offline_insufficient_range_raises(self, tmp_data_dir, block_api):
        self._make_ohlc_cache(tmp_data_dir, 'SPY', start='2023-01-01', end='2023-12-31')
        with pytest.raises(RuntimeError, match='does not cover'):
            mb.get_stock_price_ohlc('SPY', '2021-01-01', '2024-06-01', use_saved_data=True)

    def test_offline_missing_cache_raises(self, tmp_data_dir, block_api):
        with pytest.raises(RuntimeError, match='does not cover.*missing'):
            mb.get_stock_price_ohlc('SPY', '2021-01-01', '2024-06-01', use_saved_data=True)


# ---- Integration: no API calls in offline mode ----


class TestOfflineIntegration:
    def test_no_fmp_fetcher_initialized_in_offline_mode(self, tmp_data_dir, block_api):
        """Full offline path: tickers + price data + stock data, no API."""
        _make_stock_cache(tmp_data_dir / 'sp500_all_stocks.csv', ['AAPL', 'MSFT'], start='2019-01-01')
        _make_price_cache(tmp_data_dir / 'sp500_price_data.csv')

        tickers = mb.get_sp500_tickers_from_fmp(use_saved_data=True)
        sp500 = mb.get_sp500_price_data('2021-01-01', '2024-06-01', use_saved_data=True)
        stocks = mb.get_multiple_stock_data(tickers, '2020-01-01', '2024-06-01', use_saved_data=True)

        assert len(tickers) == 2
        assert not sp500.empty
        assert not stocks.empty


# ---- Corrupt caches ----


class TestCorruptCaches:
    @pytest.mark.parametrize(
        'content',
        [
            b'\x00\x01\x02',
            b'\xff\xfe\xfa,\xff\n\x80,\x81\n',
            b'not,a,valid,csv\nfoo\n',
            b',AAPL,bad ticker!\n2024-01-02,1,2\n',
        ],
    )
    def test_corrupt_ticker_header_raises_offline_error(self, tmp_data_dir, block_api, content):
        (tmp_data_dir / 'sp500_all_stocks.csv').write_bytes(content)
        with pytest.raises(mb.OfflineCacheError):
            mb.get_sp500_tickers_from_fmp(use_saved_data=True)

    def test_first_column_is_never_a_ticker(self, tmp_data_dir, block_api):
        (tmp_data_dir / 'sp500_all_stocks.csv').write_text('index,BRK-B,BF.B\n2024-01-02,1,2\n')
        assert mb.get_sp500_tickers_from_fmp(use_saved_data=True) == ['BRK-B', 'BF.B']

    @pytest.mark.parametrize('content', [b'\xff\xfe\xfa,\xff\n\x80,\x81\n', b'date,adjusted_close\nfoo,1\nbar,2\n'])
    @pytest.mark.parametrize(
        ('filename', 'call'),
        [
            ('sp500_price_data.csv', lambda: mb.get_sp500_price_data('2024-01-02', '2024-06-03', True)),
            ('sp500_all_stocks.csv', lambda: mb.get_multiple_stock_data(['A'], '2024-01-02', '2024-06-03', True)),
            ('SPY_price_data.csv', lambda: mb.get_stock_price_data('SPY', '2024-01-02', '2024-06-03', True)),
            ('SPY_ohlc_data.csv', lambda: mb.get_stock_price_ohlc('SPY', '2024-01-02', '2024-06-03', True)),
        ],
    )
    def test_unreadable_data_cache_raises_clear_error(self, tmp_data_dir, block_api, filename, call, content):
        (tmp_data_dir / filename).write_bytes(content)
        with pytest.raises(mb.OfflineCacheError, match=filename):
            call()

    def test_corrupt_cache_is_not_reported_as_a_miss(self, tmp_data_dir, block_api):
        (tmp_data_dir / 'SPY_ohlc_data.csv').write_bytes(b'\xff\xfe\xfa,\xff\n\x80,\x81\n')
        with pytest.raises(mb.OfflineCacheError) as exc_info:
            mb.get_stock_price_ohlc('SPY', '2024-01-02', '2024-06-03', True)
        assert not isinstance(exc_info.value, mb.OfflineCacheMissError)

    def test_main_exits_nonzero_on_corrupt_price_cache(self, tmp_data_dir, block_api, monkeypatch):
        _make_stock_cache(tmp_data_dir / 'sp500_all_stocks.csv', ['AAPL'], start='2023-01-02')
        (tmp_data_dir / 'sp500_price_data.csv').write_bytes(b'\xff\xfe\xfa,\xff\n\x80,\x81\n')
        monkeypatch.setattr(
            sys,
            'argv',
            ['market_breadth.py', '--use_saved_data', '--start_date', '2024-01-02', '--end_date', '2024-06-03'],
        )
        with pytest.raises(mb.OfflineCacheError):
            mb.main()


# ---- Coverage is judged in market sessions ----


def _write_ohlc(path, symbol, start, end):
    dates = pd.bdate_range(start, end)
    frame = pd.DataFrame({'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'adjusted_close': 1.0}, index=dates)
    frame.to_csv(path / f'{symbol}_ohlc_data.csv')


class TestSessionAwareCoverage:
    def test_end_date_on_weekend_accepts_cache_ending_friday(self, tmp_data_dir, block_api):
        # 2024-06-02 is a Sunday, and 2024-05-31 is the preceding Friday.
        _write_ohlc(tmp_data_dir, 'SPY', '2018-01-02', '2024-05-31')
        result = mb.get_stock_price_ohlc('SPY', '2021-01-04', '2024-06-02', use_saved_data=True)
        assert result.index.max() == pd.Timestamp('2024-05-31')

    def test_start_date_on_weekend_accepts_cache_starting_monday(self, tmp_data_dir, block_api):
        # 2020-02-29 is a Saturday, and 2020-03-02 is the following Monday.
        _make_price_cache(tmp_data_dir / 'sp500_price_data.csv', start='2020-03-02', end='2024-06-03')
        assert not mb.get_sp500_price_data('2020-02-29', '2024-06-03', use_saved_data=True).empty

    def test_start_on_market_holiday_accepts_next_session(self, tmp_data_dir, block_api):
        # 2024-01-01 is a market holiday, so 2024-01-02 is the first session.
        pytest.importorskip('exchange_calendars')
        _write_ohlc(tmp_data_dir, 'SPY', '2024-01-02', '2024-06-03')
        assert not mb.get_stock_price_ohlc('SPY', '2024-01-01', '2024-06-03', use_saved_data=True).empty

    def test_cache_missing_a_real_session_at_the_end_raises(self, tmp_data_dir, block_api):
        # Friday 2024-05-31 is a session the cache does not contain.
        _write_ohlc(tmp_data_dir, 'SPY', '2018-01-02', '2024-05-30')
        with pytest.raises(mb.OfflineCacheMissError, match='does not cover'):
            mb.get_stock_price_ohlc('SPY', '2021-01-04', '2024-06-02', use_saved_data=True)

    @pytest.mark.parametrize(
        ('start', 'end'),
        [('2008-01-01', '2024-12-31'), ('2020-03-02', '2024-06-02'), ('2020-03-02', '2024-06-03')],
    )
    def test_refresh_then_offline_round_trip(self, tmp_data_dir, start, end):
        def ohlc(_symbol, from_date, to_date):
            days = pd.bdate_range(from_date, to_date)
            return pd.DataFrame({'close': 1.0, 'adjusted_close': 1.0}, index=days)

        def series(_symbol, from_date, to_date):
            days = pd.bdate_range(from_date, to_date)
            return pd.Series(1.0, index=days, name='adjusted_close')

        with mock.patch.object(mb, 'fetch_price_ohlc_fmp', ohlc), mock.patch.object(mb, 'fetch_price_data_fmp', series):
            for call in (mb.get_stock_price_ohlc, mb.get_stock_price_data):
                call('ZZZ', start, end, use_saved_data=False)
                assert not call('ZZZ', start, end, use_saved_data=True).empty
            mb.get_sp500_price_data(start, end, use_saved_data=False)
            assert not mb.get_sp500_price_data(start, end, use_saved_data=True).empty


# ---- Lookback before a symbol existed ----


class TestInception:
    def test_lookback_before_cache_start_warns_but_succeeds(self, tmp_data_dir, block_api):
        _write_ohlc(tmp_data_dir, 'MTUM', '2013-04-18', '2026-02-26')
        with pytest.warns(UserWarning, match='warm-up'):
            result = mb.get_stock_price_ohlc('MTUM', '2014-06-02', '2025-12-31', use_saved_data=True)
        assert result.index.min() == pd.Timestamp('2013-04-18')

    def test_requested_start_before_cache_without_inception_record_raises(self, tmp_data_dir, block_api):
        _write_ohlc(tmp_data_dir, 'MTUM', '2013-04-18', '2026-02-26')
        with pytest.raises(mb.OfflineCacheMissError, match='does not cover'):
            mb.get_stock_price_ohlc('MTUM', '2008-01-02', '2025-12-31', use_saved_data=True)

    def test_refresh_records_inception_for_later_offline_runs(self, tmp_data_dir):
        def ohlc(_symbol, _from, to_date):
            return pd.DataFrame({'close': 1.0, 'adjusted_close': 1.0}, index=pd.bdate_range('2013-04-18', to_date))

        with mock.patch.object(mb, 'fetch_price_ohlc_fmp', ohlc):
            mb.get_stock_price_ohlc('MTUM', '2008-01-02', '2025-12-31', use_saved_data=False)

        fetcher = mock.Mock(side_effect=AssertionError('API call attempted'))
        with mock.patch.object(mb, 'fetch_price_ohlc_fmp', fetcher), pytest.warns(UserWarning, match='first available'):
            result = mb.get_stock_price_ohlc('MTUM', '2008-01-02', '2025-12-31', use_saved_data=True)
        assert result.index.min() == pd.Timestamp('2013-04-18')
        fetcher.assert_not_called()

    def test_price_and_ohlc_share_the_inception_record(self, tmp_data_dir):
        def series(_symbol, _from, to_date):
            return pd.Series(1.0, index=pd.bdate_range('2013-04-18', to_date), name='adjusted_close')

        with mock.patch.object(mb, 'fetch_price_data_fmp', series):
            mb.get_stock_price_data('MTUM', '2008-01-02', '2025-12-31', use_saved_data=False)
        _write_ohlc(tmp_data_dir, 'MTUM', '2013-04-18', '2025-12-31')
        with pytest.warns(UserWarning, match='first available'):
            assert not mb.get_stock_price_ohlc('MTUM', '2008-01-02', '2025-12-31', use_saved_data=True).empty

    def test_full_refresh_clears_an_old_inception_record(self, tmp_data_dir):
        def late(_symbol, _from, to_date):
            return pd.DataFrame({'close': 1.0, 'adjusted_close': 1.0}, index=pd.bdate_range('2013-04-18', to_date))

        def full(_symbol, from_date, to_date):
            return pd.DataFrame({'close': 1.0, 'adjusted_close': 1.0}, index=pd.bdate_range(from_date, to_date))

        with mock.patch.object(mb, 'fetch_price_ohlc_fmp', late):
            mb.get_stock_price_ohlc('MTUM', '2008-01-02', '2025-12-31', use_saved_data=False)
        with mock.patch.object(mb, 'fetch_price_ohlc_fmp', full):
            mb.get_stock_price_ohlc('MTUM', '2008-01-02', '2025-12-31', use_saved_data=False)
        assert not list(tmp_data_dir.glob('MTUM_*.json'))


# ---- End to end with every socket blocked ----


@pytest.fixture()
def block_sockets(monkeypatch):
    def _deny(*args, **kwargs):
        raise AssertionError('Network call attempted in offline mode')

    monkeypatch.setattr('socket.socket.connect', _deny)
    monkeypatch.setattr('socket.create_connection', _deny)


def _make_trending_universe(path, tickers, start, end):
    dates = pd.bdate_range(start, end)
    rng = np.random.RandomState(42)
    frame = pd.DataFrame({t: 100 + rng.normal(0, 1, len(dates)).cumsum() + 100 for t in tickers}, index=dates)
    frame.to_csv(path / 'sp500_all_stocks.csv')
    return frame


class TestOfflineEndToEnd:
    @pytest.fixture()
    def sandbox(self, tmp_path, monkeypatch, block_sockets):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(mb, 'data_dir', tmp_path / 'data')
        monkeypatch.setattr(mb, 'reports_dir', tmp_path / 'reports')
        (tmp_path / 'data').mkdir()
        (tmp_path / 'reports').mkdir()
        monkeypatch.delenv('FMP_API_KEY', raising=False)
        return tmp_path / 'data'

    def test_market_breadth_main_runs_from_cache_alone(self, sandbox, monkeypatch, tmp_path):
        monkeypatch.setattr(mb.go.Figure, 'write_image', lambda *_args, **_kwargs: None)
        _make_trending_universe(sandbox, ['AAPL', 'MSFT', 'GOOGL'], '2022-01-03', '2024-06-03')
        _make_price_cache(sandbox / 'sp500_price_data.csv', start='2022-01-03', end='2024-06-03')
        monkeypatch.setattr(
            sys,
            'argv',
            ['market_breadth.py', '--use_saved_data', '--start_date', '2023-02-01', '--end_date', '2024-06-03'],
        )
        mb.main()
        assert (tmp_path / 'reports' / 'market_breadth.html').exists()

    def test_backtest_runs_from_cache_alone(self, sandbox):
        from backtest.backtest import Backtest

        _make_trending_universe(sandbox, ['AAPL', 'MSFT', 'GOOGL'], '2018-01-02', '2023-12-29')
        _write_ohlc(sandbox, 'SPY', '2018-01-02', '2023-12-29')
        bt = Backtest(
            start_date='2020-01-02', end_date='2023-12-29', symbol='SPY', use_saved_data=True, no_show_plot=True
        )
        bt.run()
        assert not bt.price_data.empty

    def test_backtest_falls_back_to_price_cache_without_ohlc(self, sandbox):
        from backtest.backtest import Backtest

        frame = _make_trending_universe(sandbox, ['AAPL', 'MSFT'], '2018-01-02', '2023-12-29')
        frame['AAPL'].rename('adjusted_close').to_csv(sandbox / 'ZZZ_price_data.csv', header=True)
        bt = Backtest(
            start_date='2020-01-02', end_date='2023-12-29', symbol='ZZZ', use_saved_data=True, no_show_plot=True
        )
        bt.run()
        assert not bt.price_data.empty

    def test_backtest_falls_back_to_universe_column_without_ohlc(self, sandbox):
        from backtest.backtest import Backtest

        _make_trending_universe(sandbox, ['AAPL', 'MSFT'], '2018-01-02', '2023-12-29')
        bt = Backtest(
            start_date='2020-01-02', end_date='2023-12-29', symbol='AAPL', use_saved_data=True, no_show_plot=True
        )
        bt.run()
        assert not bt.price_data.empty

    def test_backtest_does_not_hide_a_corrupt_ohlc_cache(self, sandbox):
        from backtest.backtest import Backtest

        _make_trending_universe(sandbox, ['AAPL', 'MSFT'], '2018-01-02', '2023-12-29')
        (sandbox / 'AAPL_ohlc_data.csv').write_bytes(b'\xff\xfe\xfa,\xff\n\x80,\x81\n')
        bt = Backtest(
            start_date='2020-01-02', end_date='2023-12-29', symbol='AAPL', use_saved_data=True, no_show_plot=True
        )
        with pytest.raises(mb.OfflineCacheError, match='AAPL_ohlc_data.csv'):
            bt.run()

    def test_backtest_raises_when_no_local_source_exists(self, sandbox):
        from backtest.backtest import Backtest

        _make_trending_universe(sandbox, ['AAPL', 'MSFT'], '2018-01-02', '2023-12-29')
        bt = Backtest(
            start_date='2020-01-02', end_date='2023-12-29', symbol='ZZZ', use_saved_data=True, no_show_plot=True
        )
        with pytest.raises(mb.OfflineCacheMissError):
            bt.run()


# ---- Multi-ETF outcome ----


class TestMultiEtfOutcome:
    def test_all_symbols_failing_raises_without_overwriting_summary(self, tmp_path, monkeypatch, block_sockets):
        from backtest import run_multi_etf_backtest as runner

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(mb, 'data_dir', tmp_path / 'data')
        (tmp_path / 'data').mkdir()
        (tmp_path / 'reports').mkdir()
        summary = tmp_path / 'reports' / 'backtest_results_summary.md'
        summary.write_text('previous results')
        with pytest.raises(RuntimeError, match='All backtests failed'):
            runner.run_multi_etf_backtest(
                etfs=['SPY', 'QQQ'],
                start_date='2020-01-02',
                end_date='2023-12-29',
                use_saved_data=True,
                no_show_plot=True,
            )
        assert summary.read_text() == 'previous results'

    def test_exit_status_reports_failed_symbols(self):
        from backtest import run_multi_etf_backtest as runner

        ok = pd.DataFrame({'Symbol': ['SPY', 'QQQ'], 'Total Return': [0.1, 0.2]})
        partial = pd.DataFrame({'Symbol': ['SPY', 'QQQ'], 'Total Return': [0.1, np.nan]})
        assert runner.failed_symbols(ok) == []
        assert runner.failed_symbols(partial) == ['QQQ']
