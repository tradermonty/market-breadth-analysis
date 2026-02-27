#!/usr/bin/env bash
# Regenerate SSO baseline trade log for regression testing.
# Run from project root: bash scripts/regenerate_baseline.sh
#
# Prerequisites:
#   data/sp500_all_stocks.csv  — S&P500 breadth data (2006-01-01 or earlier)
#   data/SSO_ohlc_data.csv     — SSO OHLC + adjusted_close
# If missing, run without --use_saved_data first (requires FMP_API_KEY).
set -euo pipefail

REQUIRED_FILES=("data/sp500_all_stocks.csv" "data/SSO_ohlc_data.csv")
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required data file not found: $f" >&2
        echo "Run backtest without --use_saved_data first to fetch data." >&2
        exit 1
    fi
done

# Validate that SSO OHLC data covers the required lookback period.
# get_stock_price_ohlc() needs start_date - 2 years in the cache.
# SSO inception is 2006-06-21, so 2008-01-01 - 2y = 2006-01-01 exceeds it.
# This check prevents a silent fallthrough to API fetch that fails offline.
./venv311/bin/python3 -c "
import pandas as pd, sys
ohlc = pd.read_csv('data/SSO_ohlc_data.csv', index_col=0, parse_dates=True)
start = '2008-01-01'
lookback = pd.Timestamp(start) - pd.DateOffset(years=2)
if ohlc.index.min() > lookback:
    print(f'ERROR: SSO OHLC starts {ohlc.index.min().date()}, '
          f'but backtest needs data from {lookback.date()} '
          f'(start_date {start} minus 2-year lookback).', file=sys.stderr)
    print(f'The cache range check in get_stock_price_ohlc() will reject this '
          f'and fall through to API fetch, which fails offline.', file=sys.stderr)
    print(f'Either refresh the OHLC cache with API access, or adjust start_date '
          f'to >= {(ohlc.index.min() + pd.DateOffset(years=2)).strftime(\"%Y-%m-%d\")}.', file=sys.stderr)
    sys.exit(1)
print(f'SSO OHLC coverage OK: {ohlc.index.min().date()} to {ohlc.index.max().date()}')
"

./venv311/bin/python backtest/backtest.py \
  --symbol SSO \
  --start_date 2008-01-01 \
  --end_date 2026-02-26 \
  --use_saved_data \
  --no_show_plot \
  --tv_mode \
  --no-pyramiding

cp reports/trade_log_SSO_2008-01-01_2026-02-26.csv \
   reports/trade_log_SSO_2008-01-01_2026-02-26_baseline.csv

echo "Baseline updated: reports/trade_log_SSO_2008-01-01_2026-02-26_baseline.csv"
