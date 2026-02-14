# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Market Breadth Analysis Tool - A quantitative trading system that analyzes S&P500 market breadth to generate trading signals and backtest strategies. The system calculates what percentage of S&P500 stocks are trading above their 200-day moving average as a market health indicator.

## Core Architecture

### Data Flow
1. **Data Fetching** (`fmp_data_fetcher.py`) → Retrieves stock prices from Financial Modeling Prep API
2. **Market Breadth Calculation** (`market_breadth.py`) → Computes breadth indicators and visualizations
3. **Backtesting Engine** (`backtest/backtest.py`) → Simulates trading strategies with detailed trade logging
4. **Multi-ETF Analysis** (`backtest/run_multi_etf_backtest.py`) → Runs parallel backtests across multiple symbols

### Key Components

**FMPDataFetcher** (`fmp_data_fetcher.py`)
- Centralized API client with rate limiting (750 calls/min for Premium tier)
- Dynamic rate control that backs off on 429 errors
- Caches S&P500 constituents and historical data
- Never call FMP API directly - always use this wrapper

**Backtest Class** (`backtest/backtest.py`)
- Supports FIFO position management for multiple entries
- Trade logging system records 15 data points per completed trade
- Uses `_process_exit_fifo()` to match exits with entries chronologically
- `save_trade_log()` exports CSV files: `reports/trade_log_{SYMBOL}_{START}_{END}.csv`

**Market Breadth Calculation** (`market_breadth.py`)
- `calculate_above_ma()` computes percentage of stocks above 200-day MA
- Peak detection with `find_peaks()` identifies trend reversals
- `calculate_trend_with_hysteresis()` prevents whipsaw signals
- Background color coding: Pink = downtrend (200MA declining + short MA < long MA)

## Testing

### Run All Tests
```bash
python tests/test_trade_logging.py      # Trade logging unit tests (10 cases)
python tests/test_sp500_fetch.py        # Data fetching tests
python tests/test_market_breadth_utils.py
```

### Run Single Test
```bash
python -m unittest tests.test_trade_logging.TestTradeLogging.test_06_pnl_calculation
```

### Test Strategy
- Follow TDD approach: write tests before implementation
- All tests must pass before committing
- Test files go in `tests/` directory
- Use descriptive test names: `test_{number}_{feature_description}`

## Common Commands

### Market Breadth Analysis
```bash
# Using saved data (no API key required)
python market_breadth.py --use_saved_data

# Fetch fresh data (requires FMP_API_KEY in .env)
python market_breadth.py --start_date 2020-01-01

# Custom MA periods
python market_breadth.py --short_ma 20 --use_saved_data
```

### Data Fetch / Workflow Trigger
```bash
# Auto mode: fetch if fresh, trigger workflow if stale
python trigger_market_breadth.py

# Fetch CSV only (no trigger)
python trigger_market_breadth.py --fetch-only

# Force trigger GitHub Actions workflow
python trigger_market_breadth.py --trigger-only

# Custom staleness threshold (6 hours)
python trigger_market_breadth.py --max-age 6
```

### Backtesting
```bash
# Single symbol backtest (run from project root)
python backtest/backtest.py --symbol SPY --start_date 2020-01-01 --end_date 2023-12-31 --use_saved_data

# Multi-ETF backtest (ALWAYS run from project root to save to reports/)
python backtest/run_multi_etf_backtest.py

# Debug mode with verbose output
python backtest/backtest.py --debug --symbol SSO --use_saved_data

# IMPORTANT: Always run backtest scripts from project root directory
# This ensures outputs go to reports/ not backtest/reports/
```

### Key Arguments
- `--use_saved_data`: Use cached CSV files instead of API calls (faster, no API key needed)
- `--debug`: Enable verbose logging including trade pair matching details
- `--no_show_plot`: Save charts without displaying (useful for batch processing)
- `--stop_loss_pct`: Fixed stop loss percentage (default: 0.08)
- `--use_trailing_stop`: Enable trailing stop instead of fixed stop
- `--ma_type`: 'ema' or 'sma' for moving average calculation

## Data Management

### File Structure
```
data/
  sp500_all_stocks.csv         # All S&P500 constituents (automatically saved)
  {SYMBOL}_price_data.csv      # Individual stock cache files

reports/
  market_breadth.png           # Latest breadth chart
  backtest_results_{SYMBOL}.png
  trade_log_{SYMBOL}_{START}_{END}.csv  # Detailed trade logs
  backtest_results_summary.md
```

### Data Workflow
1. **First run**: Fetches from FMP API, saves to `data/` automatically
2. **Subsequent runs**: Use `--use_saved_data` flag to skip API calls
3. **Update data**: Run without `--use_saved_data` to refresh

### CSV Cache Pattern
The system saves and loads data using these helper functions:
```python
save_stock_data(dataframe, 'filename.csv')  # Auto-saves to data/
load_stock_data('filename.csv')              # Auto-loads from data/
```

## Trade Logging System

### Implementation (Phase 1 Complete)
The backtest system now logs every trade with FIFO matching:

**Data Structures:**
- `self.trade_log[]` - Completed trades with full details
- `self.open_positions[]` - Currently open positions awaiting exit
- `self.next_trade_id` - Auto-incrementing trade counter

**Key Methods:**
- `_execute_entry()` - Adds position to `open_positions`
- `_process_exit_fifo()` - Matches exits with entries using FIFO
- `_record_completed_trade()` - Calculates P&L and logs to `trade_log`
- `save_trade_log()` - Exports CSV with 15 columns per trade

**Trade Log Columns:**
trade_id, entry_date, entry_price, entry_shares, entry_cost, entry_reason, exit_date, exit_price, exit_shares, exit_proceeds, exit_reason, holding_days, pnl_dollar, pnl_percent, cumulative_pnl

### Design Reference
See `docs/trade_logging_design.md` for complete specification including:
- FIFO logic for multiple entries/exits
- Partial exit handling
- Cumulative P&L calculation
- Phase 2 roadmap (console display, statistics)

## Important Patterns

### API Key Security
```python
# ALWAYS use environment variables
load_dotenv()
api_key = os.getenv('FMP_API_KEY')
github_token = os.getenv('GITHUB_TOKEN')

# NEVER hardcode keys
# NEVER commit .env files
```

### Import Market Breadth Functions
Backtest and other modules import from `market_breadth.py`:
```python
from market_breadth import (
    get_sp500_tickers_from_fmp,  # Not from Wikipedia anymore
    calculate_above_ma,
    get_multiple_stock_data,
    # ... other functions
)
```

### Matplotlib Backend Handling
The system auto-detects OS and sets appropriate backend:
```python
setup_matplotlib_backend()  # Call before any plt imports
```
- macOS/Windows: Tries TkAgg, falls back to Agg
- Linux: Uses Agg (non-interactive)

## Git Workflow

### Before Committing
1. Run all tests: `python tests/test_trade_logging.py`
2. Ensure venv and CSV files are not staged (check .gitignore)
3. Place test files in `tests/` directory
4. Design docs go in `docs/` directory

### Commit Message Format
```
Add {feature} to {component}

- Implementation detail 1
- Implementation detail 2
- Test coverage info

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Development Notes

### When Modifying Backtest Logic
1. Update `tests/test_trade_logging.py` first (TDD)
2. Ensure FIFO matching remains consistent in `_process_exit_fifo()`
3. Verify cumulative_pnl calculations after changes
4. Check that `open_positions` is properly cleared on full exits

### When Adding New Entry/Exit Signals
1. Add reason string to `_execute_entry()` or `_execute_exit()` calls
2. Entry reasons: "short_ma_bottom", "long_ma_bottom", "background_color_change"
3. Exit reasons: "peak exit", "stop loss", "background color change"
4. These appear in trade log CSV for signal analysis

### FMP API Compliance
- Use `fmp_fetcher.get_sp500_constituents()` for ticker list (not Wikipedia)
- Historical data endpoint: `/api/v3/historical-price-full/{symbol}`
- Rate limits: 750 calls/min (Premium), auto-throttles on 429
- Batch requests when possible: `get_multiple_stock_data()`

### Matplotlib Color Coding
Pink background indicates market weakness:
```python
if long_ma_trend[i] == -1 and short_ma[i] < long_ma[i]:
    ax.axvspan(dates[i], dates[i+1], color=(1.0, 0.9, 0.96), alpha=0.3)
```

## File Organization
- Core logic: Root directory (`market_breadth.py`, `fmp_data_fetcher.py`)
- Backtesting: `backtest/` directory
- Tests: `tests/` directory
- Design docs: `docs/` directory
- Archives: `archive/` (old versions, not active)
- Trading automation: `trade/` (separate from backtesting)
