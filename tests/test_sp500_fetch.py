#!/usr/bin/env python3
"""Test script to verify S&P500 ticker fetching from FMP API"""

import os
from dotenv import load_dotenv
from fmp_data_fetcher import FMPDataFetcher

# Load environment variables
load_dotenv()

def test_sp500_fetch():
    """Test S&P500 ticker fetching"""
    print("=" * 60)
    print("Testing S&P500 Ticker Fetching from FMP API")
    print("=" * 60)

    # Check API key
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        print("ERROR: FMP_API_KEY not found in .env file")
        return False

    print(f"✓ API Key loaded (length: {len(api_key)})")

    # Initialize FMP fetcher
    try:
        fmp_fetcher = FMPDataFetcher(api_key=api_key)
        print("✓ FMP Data Fetcher initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize FMP fetcher: {e}")
        return False

    # Fetch S&P500 tickers
    try:
        print("\nFetching S&P500 constituents...")
        tickers = fmp_fetcher.get_sp500_constituents()

        if not tickers:
            print("ERROR: No tickers returned from FMP API")
            return False

        print(f"✓ Successfully fetched {len(tickers)} S&P500 tickers")
        print(f"\nFirst 10 tickers: {tickers[:10]}")
        print(f"Last 10 tickers: {tickers[-10:]}")

        # Verify expected tickers
        expected_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        found_tickers = [t for t in expected_tickers if t in tickers]
        print(f"\nExpected tickers found: {found_tickers}")

        print("\n" + "=" * 60)
        print("✓ Test PASSED: S&P500 tickers fetched successfully")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nERROR: Failed to fetch S&P500 tickers: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_sp500_fetch()
    exit(0 if success else 1)
