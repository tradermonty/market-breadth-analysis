"""Parameter sensitivity sweep with optional OOS split.

Grid: rebalance_threshold x (overheat_entry, overheat_exit) x ratio_ma x transaction_cost_bps
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rsp_breadth_strategy.backtest import run_strategy_backtest
from rsp_breadth_strategy.metrics import compute_metrics
from rsp_breadth_strategy.signals import build_signal_frame

# Parameter grid
REBALANCE_THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.10]
OVERHEAT_PAIRS = [(0.70, 0.65), (0.75, 0.70), (0.80, 0.75)]
RATIO_MAS = [5, 10, 20]
COST_BPS = [3, 5, 10, 15]

# OOS split dates
IS_END = '2023-12-31'
OOS_START = '2024-01-01'


def _run_single(
    merged: pd.DataFrame,
    rebalance_threshold: float,
    overheat_entry: float,
    overheat_exit: float,
    ratio_ma: int,
    transaction_cost_bps: float,
) -> dict[str, float]:
    """Run a single backtest and return metrics dict with parameters."""
    signals = build_signal_frame(
        merged,
        ratio_ma=ratio_ma,
        regime_mode='series',
        overheat_entry=overheat_entry,
        overheat_exit=overheat_exit,
    )
    bt = run_strategy_backtest(
        signals,
        rebalance_threshold=rebalance_threshold,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_freq='month_end',
    )
    m = compute_metrics(bt)
    return {
        'rebalance_threshold': rebalance_threshold,
        'overheat_entry': overheat_entry,
        'overheat_exit': overheat_exit,
        'ratio_ma': ratio_ma,
        'transaction_cost_bps': transaction_cost_bps,
        **m,
    }


def _build_grid() -> list[dict]:
    """Generate the full parameter grid."""
    combos = list(itertools.product(
        REBALANCE_THRESHOLDS,
        OVERHEAT_PAIRS,
        RATIO_MAS,
        COST_BPS,
    ))
    grid = []
    for thresh, (oh_entry, oh_exit), rma, cost in combos:
        grid.append({
            'rebalance_threshold': thresh,
            'overheat_entry': oh_entry,
            'overheat_exit': oh_exit,
            'ratio_ma': rma,
            'transaction_cost_bps': cost,
        })
    return grid


def _run_sweep(merged: pd.DataFrame, grid: list[dict]) -> pd.DataFrame:
    """Run sweep over parameter grid."""
    rows = []
    total = len(grid)
    for i, params in enumerate(grid, 1):
        if i % 20 == 0 or i == total:
            print(f'  [{i}/{total}]')
        row = _run_single(merged, **params)
        rows.append(row)
    return pd.DataFrame(rows)


def _print_top(df: pd.DataFrame, sort_col: str, n: int = 10, label: str = '') -> None:
    """Print top-N rows sorted by given column."""
    top = df.nlargest(n, sort_col)
    cols = ['rebalance_threshold', 'overheat_entry', 'ratio_ma',
            'transaction_cost_bps', 'cagr', 'sharpe', 'max_drawdown', 'calmar']
    print(f'\n--- Top-{n} by {label or sort_col} ---')
    print(top[cols].to_string(index=False, float_format='%.4f'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Parameter sensitivity sweep')
    parser.add_argument('--breadth-csv', required=True, help='Path to market breadth CSV')
    parser.add_argument('--prices-csv', required=True, help='Path to prices CSV')
    parser.add_argument('--output-dir', default='outputs_sensitivity', help='Output directory')
    parser.add_argument('--oos', action='store_true', help='Run IS/OOS split analysis')
    args = parser.parse_args()

    breadth = pd.read_csv(args.breadth_csv)
    prices = pd.read_csv(args.prices_csv)

    breadth['Date'] = pd.to_datetime(breadth['Date'])
    prices['Date'] = pd.to_datetime(prices['Date'])

    merged = breadth.merge(
        prices[['Date', 'SPY', 'RSP', 'XLE', 'SGOV']],
        on='Date',
        how='inner',
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = _build_grid()
    print(f'Parameter grid: {len(grid)} combinations')

    if not args.oos:
        # Full period sweep
        print('\nRunning full-period sweep...')
        results = _run_sweep(merged, grid)
        csv_path = out_dir / 'sensitivity_full.csv'
        results.to_csv(csv_path, index=False)
        print(f'\nSaved: {csv_path}')
        _print_top(results, 'sharpe', label='Sharpe')
        _print_top(results, 'calmar', label='Calmar (Return/DD)')
    else:
        # IS/OOS split
        is_data = merged[merged['Date'] <= IS_END].copy().reset_index(drop=True)
        oos_data = merged[merged['Date'] >= OOS_START].copy().reset_index(drop=True)

        print(f'\nIS period: {is_data["Date"].min().date()} ~ {is_data["Date"].max().date()} ({len(is_data)} rows)')
        print(f'OOS period: {oos_data["Date"].min().date()} ~ {oos_data["Date"].max().date()} ({len(oos_data)} rows)')

        print('\nRunning IS sweep...')
        is_results = _run_sweep(is_data, grid)
        is_path = out_dir / 'sensitivity_IS.csv'
        is_results.to_csv(is_path, index=False)
        print(f'Saved: {is_path}')

        print('\nRunning OOS sweep...')
        oos_results = _run_sweep(oos_data, grid)
        oos_path = out_dir / 'sensitivity_OOS.csv'
        oos_results.to_csv(oos_path, index=False)
        print(f'Saved: {oos_path}')

        _print_top(is_results, 'sharpe', label='IS Sharpe')
        _print_top(oos_results, 'sharpe', label='OOS Sharpe')

        # Spearman rank correlation (IS Sharpe vs OOS Sharpe)
        is_sharpe = is_results['sharpe'].values
        oos_sharpe = oos_results['sharpe'].values

        # Filter out NaN/inf
        mask = np.isfinite(is_sharpe) & np.isfinite(oos_sharpe)
        if mask.sum() > 2:
            rho, p_value = stats.spearmanr(is_sharpe[mask], oos_sharpe[mask])
            print(f'\n--- IS vs OOS Rank Correlation ---')
            print(f'  Spearman rho = {rho:.4f}  (p = {p_value:.4e})')
            if rho > 0.5:
                print('  => Strong positive rank correlation: parameters generalize well')
            elif rho > 0.2:
                print('  => Moderate correlation: some overfitting risk')
            else:
                print('  => Weak/no correlation: likely overfitting to IS period')


if __name__ == '__main__':
    main()
