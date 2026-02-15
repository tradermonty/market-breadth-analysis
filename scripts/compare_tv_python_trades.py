"""Compare Python backtest trade logs with TradingView trade logs.

Reads both CSV files, normalizes exit reasons, and reports mismatches
in entry/exit dates and reasons using actual trading-day bar distances.
"""

import argparse
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Reason normalization
# ---------------------------------------------------------------------------

REASON_MAP = {
    'stop-loss': 'stop_loss',
    'stop loss': 'stop_loss',
    'exit on breadth peak': 'peak_exit',
    'peak exit': 'peak_exit',
    'peak exit (stage 1)': 'peak_exit_stage1',
    'trend break exit (stage 2)': 'trend_break_exit',
    'exit on breadth trough': 'trough_exit',
}


def normalize_reason(reason: str) -> str:
    """Normalize an exit reason string via REASON_MAP (case-insensitive)."""
    if pd.isna(reason):
        return ''
    key = str(reason).strip().lower()
    return REASON_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Bar distance using actual trading days
# ---------------------------------------------------------------------------


def bar_distance(date_a, date_b, trading_days):
    """Exact bar distance. Raises ValueError if date not in index."""
    try:
        loc_a = trading_days.get_loc(date_a)
        loc_b = trading_days.get_loc(date_b)
    except KeyError as e:
        raise ValueError(f'Date not found in trading days index: {e}') from e
    return abs(loc_a - loc_b)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_trading_days(price_csv: str) -> pd.DatetimeIndex:
    """Load price CSV and return its DatetimeIndex of trading days."""
    df = pd.read_csv(price_csv, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.index.sort_values()


def load_python_trades(csv_path: str) -> pd.DataFrame:
    """Load Python backtest trade log CSV."""
    df = pd.read_csv(csv_path, parse_dates=['entry_date', 'exit_date'])
    df['entry_date'] = pd.to_datetime(df['entry_date']).dt.normalize()
    df['exit_date'] = pd.to_datetime(df['exit_date']).dt.normalize()
    df['exit_reason_norm'] = df['exit_reason'].apply(normalize_reason)
    return df.sort_values('entry_date').reset_index(drop=True)


def load_tv_trades(csv_path: str) -> pd.DataFrame:
    """Load TradingView trade log CSV."""
    df = pd.read_csv(csv_path, parse_dates=['entry_date', 'exit_date'])
    df['entry_date'] = pd.to_datetime(df['entry_date']).dt.normalize()
    df['exit_date'] = pd.to_datetime(df['exit_date']).dt.normalize()
    df['exit_reason_norm'] = df['reason'].apply(normalize_reason)
    return df.sort_values('entry_date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_trades(py_df: pd.DataFrame, tv_df: pd.DataFrame, trading_days, tolerance_bars: int):
    """Compare Python and TV trade logs. Return a results dict."""
    results = {
        'python_count': len(py_df),
        'tv_count': len(tv_df),
        'first_mismatch': None,
        'first_mismatch_bar_diff': 0,
        'reason_matches': 0,
        'reason_total': 0,
        'calendar_fallbacks': 0,
        'pass': True,
    }

    n = min(len(py_df), len(tv_df))
    if n == 0:
        results['pass'] = False
        return results

    first_mismatch_found = False

    for i in range(n):
        py_row = py_df.iloc[i]
        tv_row = tv_df.iloc[i]

        # --- date distances ---
        try:
            entry_diff = bar_distance(py_row['entry_date'], tv_row['entry_date'], trading_days)
        except ValueError as e:
            # Fall back to calendar-day diff if date not in index
            entry_diff = abs((py_row['entry_date'] - tv_row['entry_date']).days)
            results['calendar_fallbacks'] += 1
            print(f'  WARNING: Trade {i + 1} entry date fell back to calendar days: {e}')

        try:
            exit_diff = bar_distance(py_row['exit_date'], tv_row['exit_date'], trading_days)
        except ValueError as e:
            exit_diff = abs((py_row['exit_date'] - tv_row['exit_date']).days)
            results['calendar_fallbacks'] += 1
            print(f'  WARNING: Trade {i + 1} exit date fell back to calendar days: {e}')

        max_diff = max(entry_diff, exit_diff)

        # --- reason comparison ---
        reason_match = py_row['exit_reason_norm'] == tv_row['exit_reason_norm']
        results['reason_total'] += 1
        if reason_match:
            results['reason_matches'] += 1

        # --- first mismatch ---
        if not first_mismatch_found and (max_diff > 0 or not reason_match):
            first_mismatch_found = True
            results['first_mismatch'] = {
                'trade_index': i,
                'py_entry': str(py_row['entry_date'].date()),
                'tv_entry': str(tv_row['entry_date'].date()),
                'entry_bar_diff': entry_diff,
                'py_exit': str(py_row['exit_date'].date()),
                'tv_exit': str(tv_row['exit_date'].date()),
                'exit_bar_diff': exit_diff,
                'py_reason': py_row['exit_reason_norm'],
                'tv_reason': tv_row['exit_reason_norm'],
                'reason_match': reason_match,
            }
            results['first_mismatch_bar_diff'] = max_diff

    # --- acceptance criteria ---
    reason_rate = results['reason_matches'] / results['reason_total'] if results['reason_total'] > 0 else 0.0
    results['reason_match_rate'] = reason_rate

    if results['first_mismatch_bar_diff'] > tolerance_bars:
        results['pass'] = False
    if reason_rate < 0.85:
        results['pass'] = False
    if results['python_count'] != results['tv_count']:
        results['pass'] = False

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results, tolerance_bars: int):
    """Print human-readable comparison report."""
    print('=' * 60)
    print('Trade Log Comparison: Python vs TradingView')
    print('=' * 60)

    print(f'\nTrade counts:  Python={results["python_count"]}  TV={results["tv_count"]}', end='')
    if results['python_count'] != results['tv_count']:
        print('  ** MISMATCH **')
    else:
        print('  OK')

    fm = results.get('first_mismatch')
    if fm:
        print(f'\nFirst mismatch (trade #{fm["trade_index"]}):')
        print(f'  Entry: Python={fm["py_entry"]}  TV={fm["tv_entry"]}  bar_diff={fm["entry_bar_diff"]}')
        print(f'  Exit:  Python={fm["py_exit"]}  TV={fm["tv_exit"]}  bar_diff={fm["exit_bar_diff"]}')
        print(f'  Reason: Python={fm["py_reason"]}  TV={fm["tv_reason"]}  match={fm["reason_match"]}')
    else:
        print('\nNo mismatches found -- all trades match exactly.')

    rate = results.get('reason_match_rate', 0.0)
    print(f'\nReason match rate: {results["reason_matches"]}/{results["reason_total"]} = {rate:.1%}')

    print(f'\nAcceptance criteria (tolerance_bars={tolerance_bars}):')
    print(f'  First mismatch bar diff <= {tolerance_bars}: ', end='')
    bar_ok = results['first_mismatch_bar_diff'] <= tolerance_bars
    print('PASS' if bar_ok else f'FAIL (got {results["first_mismatch_bar_diff"]})')

    print('  Reason match rate >= 85%: ', end='')
    reason_ok = rate >= 0.85
    print('PASS' if reason_ok else f'FAIL (got {rate:.1%})')

    count_ok = results['python_count'] == results['tv_count']
    print('  Trade count match: ', end='')
    print('PASS' if count_ok else 'FAIL')

    fallbacks = results.get('calendar_fallbacks', 0)
    if fallbacks > 0:
        print(f'\n  {fallbacks} date comparison(s) used calendar days instead of trading days')

    verdict = 'PASS' if results['pass'] else 'FAIL'
    print(f'\nVerdict: {verdict}')
    print('=' * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Compare Python backtest trade logs with TradingView trade logs.')
    parser.add_argument('--python_csv', required=True, help='Path to Python trade log CSV')
    parser.add_argument('--tv_csv', required=True, help='Path to TradingView trades CSV')
    parser.add_argument('--price_csv', required=True, help='Path to price CSV with date index (for trading days)')
    parser.add_argument('--tolerance_bars', type=int, default=1, help='Allowed bar difference (default: 1)')
    args = parser.parse_args()

    # Load data
    trading_days = load_trading_days(args.price_csv)
    py_df = load_python_trades(args.python_csv)
    tv_df = load_tv_trades(args.tv_csv)

    # Compare
    results = compare_trades(py_df, tv_df, trading_days, args.tolerance_bars)

    # Report
    print_report(results, args.tolerance_bars)

    sys.exit(0 if results['pass'] else 1)


if __name__ == '__main__':
    main()
