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

    def test_alias_preserved(self):
        assert mb.get_sp500_tickers is mb.get_sp500_tickers_from_fmp

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
