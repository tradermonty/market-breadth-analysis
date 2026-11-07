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

# --use_saved_data never calls the API: a cache that cannot cover 2008-01-01 to
# 2026-02-26 stops the run with an error. SSO's inception (2006-06-21) only shortens
# the two-year warm-up, which produces a warning.
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
