"""Compare Python backtest trade logs with TradingView trade logs.

Reads both CSV files, normalizes exit reasons, and reports mismatches
in entry/exit dates and reasons using actual trading-day bar distances.
"""

import argparse
import sys
from collections import Counter
from numbers import Integral, Number

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
    # Normalize common unicode dash variants seen in TV exports.
    key = (
        key.replace('\u2010', '-')
        .replace('\u2011', '-')
        .replace('\u2012', '-')
        .replace('\u2013', '-')
        .replace('\u2014', '-')
    )
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


def _dates(values, label, normalize=True):
    """Parse dates without dropping invalid rows or silently changing time zones."""
    parsed = []
    for value in values:
        if isinstance(value, Number):
            raise ValueError(f'{label}: numeric epoch dates are unsupported; use explicit dates')
        try:
            date = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label}: invalid date') from exc
        if pd.isna(date) or date.tzinfo is not None:
            raise ValueError(f'{label}: dates must be present and timezone-naive')
        parsed.append(date.normalize() if normalize else date)
    return pd.DatetimeIndex(parsed)


def _calendar(values):
    days = _dates(values, 'price calendar', normalize=False)
    if days.empty or days.has_duplicates or not days.equals(days.normalize()):
        raise ValueError('price calendar must contain nonempty, unique daily dates')
    return days.sort_values()


def _validated_trades(df):
    required = {'entry_date', 'exit_date', 'exit_reason_norm'}
    if not required.issubset(df.columns):
        raise ValueError(f'trade CSV missing required columns: {sorted(required - set(df.columns))}')
    out = df.copy()
    out['entry_date'] = _dates(out['entry_date'], 'entry_date')
    out['exit_date'] = _dates(out['exit_date'], 'exit_date')
    out['exit_reason_norm'] = out['exit_reason_norm'].map(normalize_reason)
    if out['exit_reason_norm'].eq('').any():
        raise ValueError('closed trades require a nonempty exit reason')
    if (out['entry_date'] > out['exit_date']).any():
        raise ValueError('trade entry_date must not be after exit_date')
    return out.sort_values(['entry_date', 'exit_date', 'exit_reason_norm'], kind='stable').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_trading_days(price_csv: str) -> pd.DatetimeIndex:
    """Load price CSV and return its DatetimeIndex of trading days."""
    df = pd.read_csv(price_csv, parse_dates=[0], index_col=0)
    return _calendar(df.index)


def load_python_trades(csv_path: str) -> pd.DataFrame:
    """Load Python backtest trade log CSV."""
    df = pd.read_csv(csv_path)
    if 'exit_reason' not in df:
        raise ValueError('Python trade CSV missing exit_reason')
    df['exit_reason_norm'] = df['exit_reason'].map(normalize_reason)
    return _validated_trades(df)


def load_tv_trades(csv_path: str) -> pd.DataFrame:
    """Load TradingView trade log CSV.

    Supports two schemas:
    1) Normalized format: entry_date/exit_date/reason
    2) Raw TradingView export: Trade # / Type / Date and time / Signal
       (Entry/Exit rows per trade; open trades are excluded)
    """
    df = pd.read_csv(csv_path)
    col_map = {c.lower(): c for c in df.columns}

    # Schema A: normalized trades CSV (one row per closed trade)
    if {'entry_date', 'exit_date'}.issubset(col_map):
        entry_col = col_map['entry_date']
        exit_col = col_map['exit_date']
        reason_col = col_map.get('reason', col_map.get('exit_reason'))
        if reason_col is None:
            raise ValueError(f'TV CSV missing reason column: {csv_path}')

        out = pd.DataFrame()
        out['entry_date'] = df[entry_col]
        out['exit_date'] = df[exit_col]
        out['reason'] = df[reason_col]
        out['exit_reason_norm'] = out['reason'].apply(normalize_reason)
        return _validated_trades(out)

    # Schema B: raw TradingView export (entry/exit rows for each trade)
    required = {'trade #', 'type', 'date and time'}
    if required.issubset(col_map):
        trade_col = col_map['trade #']
        type_col = col_map['type']
        date_col = col_map['date and time']
        signal_col = col_map.get('signal')
        if signal_col is None or df[trade_col].isna().any():
            raise ValueError('raw TV trades require Signal and a Trade # on every row')

        records = []
        for trade_no, grp in df.groupby(trade_col):
            g = grp.copy()
            g[date_col] = _dates(g[date_col], f'TV trade {trade_no}', normalize=False)

            type_series = g[type_col].astype(str).str.strip().str.lower()
            if not type_series.isin(['entry long', 'exit long']).all():
                raise ValueError(f'TV trade {trade_no}: only long entry/exit rows are supported')
            entry_rows = g[type_series.eq('entry long')]
            exit_rows = g[type_series.eq('exit long')]
            if len(entry_rows) != 1 or len(exit_rows) > 1:
                raise ValueError(f'TV trade {trade_no}: ambiguous entry/exit multiplicity')
            if exit_rows.empty:
                # A single valid entry without an exit is an intentionally open trade.
                continue

            entry = entry_rows.iloc[0]
            exit_row = exit_rows.iloc[-1]
            if entry[date_col] > exit_row[date_col]:
                raise ValueError(f'TV trade {trade_no}: exit precedes entry')

            signal = ''
            if signal_col and signal_col in exit_row.index:
                signal = exit_row[signal_col]

            # Exclude open trades (TV marks them with signal "Open").
            if normalize_reason(signal) == 'open':
                continue

            records.append(
                {
                    'trade_no': trade_no,
                    'entry_date': pd.to_datetime(entry[date_col]).normalize(),
                    'exit_date': pd.to_datetime(exit_row[date_col]).normalize(),
                    'reason': signal,
                }
            )

        out = pd.DataFrame(records)
        if out.empty:
            return pd.DataFrame(columns=['entry_date', 'exit_date', 'reason', 'exit_reason_norm'])

        out['exit_reason_norm'] = out['reason'].apply(normalize_reason)
        return _validated_trades(out)

    raise ValueError(f'Unsupported TV CSV schema for {csv_path}. Columns were: {list(df.columns)}')


def _to_trade_key(df: pd.DataFrame):
    keys = Counter()
    for _, row in df.iterrows():
        keys[
            (
                str(pd.to_datetime(row['entry_date']).date()),
                str(pd.to_datetime(row['exit_date']).date()),
                str(row['exit_reason_norm']),
            )
        ] += 1
    return keys


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_trades(py_df: pd.DataFrame, tv_df: pd.DataFrame, trading_days, tolerance_bars: int):
    """Compare Python and TV trade logs. Return a results dict."""
    if not isinstance(tolerance_bars, Integral) or isinstance(tolerance_bars, bool) or tolerance_bars < 0:
        raise ValueError('tolerance_bars must be a nonnegative integer')
    trading_days = _calendar(trading_days)
    py_df = _validated_trades(py_df)
    tv_df = _validated_trades(tv_df)
    for label, df in [('Python', py_df), ('TV', tv_df)]:
        for field in ['entry_date', 'exit_date']:
            if not df[field].isin(trading_days).all():
                raise ValueError(f'{label} {field}: date not found in trading days index; comparison invalid')
    results = {
        'python_count': len(py_df),
        'tv_count': len(tv_df),
        'first_mismatch': None,
        'first_mismatch_bar_diff': 0,
        'reason_matches': 0,
        'reason_total': 0,
        'calendar_fallbacks': 0,
        'max_entry_bar_diff': 0,
        'max_exit_bar_diff': 0,
        'max_bar_diff': 0,
        'failing_trade_indexes': [],
        'exact_matches': 0,
        'tv_only': [],
        'python_only': [],
        'pass': True,  # nosec B105 - results flag, not a password
    }

    n = min(len(py_df), len(tv_df))
    first_mismatch_found = False

    for i in range(n):
        py_row = py_df.iloc[i]
        tv_row = tv_df.iloc[i]

        # --- date distances ---
        entry_diff = bar_distance(py_row['entry_date'], tv_row['entry_date'], trading_days)
        exit_diff = bar_distance(py_row['exit_date'], tv_row['exit_date'], trading_days)

        max_diff = max(entry_diff, exit_diff)
        results['max_entry_bar_diff'] = max(results['max_entry_bar_diff'], entry_diff)
        results['max_exit_bar_diff'] = max(results['max_exit_bar_diff'], exit_diff)
        results['max_bar_diff'] = max(results['max_bar_diff'], max_diff)

        # --- reason comparison ---
        reason_match = py_row['exit_reason_norm'] == tv_row['exit_reason_norm']
        results['reason_total'] += 1
        if reason_match:
            results['reason_matches'] += 1
        if max_diff > tolerance_bars or not reason_match:
            results['failing_trade_indexes'].append(i)

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

    py_keys = _to_trade_key(py_df)
    tv_keys = _to_trade_key(tv_df)
    exact = sorted((py_keys & tv_keys).elements())
    tv_only = sorted((tv_keys - py_keys).elements())
    py_only = sorted((py_keys - tv_keys).elements())
    results['exact_matches'] = len(exact)
    results['tv_only'] = tv_only
    results['python_only'] = py_only

    results['pass'] = bool(n and len(py_df) == len(tv_df) and not results['failing_trade_indexes'])

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
        print('\nNo paired date/reason mismatches found.')

    rate = results.get('reason_match_rate', 0.0)
    print(f'\nReason match rate: {results["reason_matches"]}/{results["reason_total"]} = {rate:.1%}')
    print(f'Exact trade tuple matches: {results["exact_matches"]}')

    if results.get('tv_only'):
        print('\nTV-only trades (not found in Python):')
        for entry_date, exit_date, reason in results['tv_only']:
            print(f'  {entry_date} -> {exit_date}  ({reason})')

    if results.get('python_only'):
        print('\nPython-only trades (not found in TV):')
        for entry_date, exit_date, reason in results['python_only']:
            print(f'  {entry_date} -> {exit_date}  ({reason})')

    print(f'\nAcceptance criteria (tolerance_bars={tolerance_bars}):')
    print(f'  Maximum entry/exit bar diff across all trades <= {tolerance_bars}: ', end='')
    bar_ok = results['max_bar_diff'] <= tolerance_bars
    print('PASS' if bar_ok else f'FAIL (got {results["max_bar_diff"]})')

    print('  Reason match rate = 100%: ', end='')
    reason_ok = rate == 1.0
    print('PASS' if reason_ok else f'FAIL (got {rate:.1%})')

    count_ok = results['python_count'] == results['tv_count']
    print('  Trade count match: ', end='')
    print('PASS' if count_ok else 'FAIL')
    print('  Nonempty closed trades: ' + ('PASS' if results['reason_total'] else 'FAIL'))

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

    try:
        trading_days = load_trading_days(args.price_csv)
        py_df = load_python_trades(args.python_csv)
        tv_df = load_tv_trades(args.tv_csv)
        results = compare_trades(py_df, tv_df, trading_days, args.tolerance_bars)
    except (OSError, ValueError, KeyError) as exc:
        print(f'Invalid comparison input: {exc}', file=sys.stderr)
        print('Verdict: FAIL')
        return 2

    # Report
    print_report(results, args.tolerance_bars)

    return 0 if results['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
