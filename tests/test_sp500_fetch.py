#!/usr/bin/env python3
"""Test S&P500 ticker fetching from FMP API.

M-11 fix: Converted from return-based to assert-based test so pytest
can detect failures. Skipped by default (requires live API key).
"""

import os

import pytest
from dotenv import load_dotenv

from fmp_data_fetcher import FMPDataFetcher

load_dotenv()


@pytest.mark.skipif(not os.getenv('FMP_API_KEY'), reason='FMP_API_KEY not set')
def test_sp500_fetch():
    """Fetch S&P500 tickers from FMP API and verify basic expectations."""
    api_key = os.getenv('FMP_API_KEY')
    fmp_fetcher = FMPDataFetcher(api_key=api_key)

    tickers = fmp_fetcher.get_sp500_constituents()

    assert tickers, 'No tickers returned from FMP API'
    assert len(tickers) > 400, f'Expected 400+ tickers, got {len(tickers)}'

    expected_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    for ticker in expected_tickers:
        assert ticker in tickers, f'Expected ticker {ticker} not found'
