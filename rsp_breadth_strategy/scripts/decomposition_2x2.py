"""2x2 decomposition: signal effect vs rebalance frequency effect.

Combinations:
  (OLD=pointwise, daily)   — baseline
  (OLD=pointwise, month_end) — rebalance effect only
  (NEW=series,    daily)   — signal effect only
  (NEW=series,    month_end) — both effects
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rsp_breadth_strategy.backtest import run_strategy_backtest
from rsp_breadth_strategy.metrics import compute_metrics
from rsp_breadth_strategy.signals import build_signal_frame


def _run_combo(
    merged: pd.DataFrame,
    regime_mode: str,
    rebalance_freq: str,
    ratio_ma: int = 10,
    rebalance_threshold: float = 0.05,
    transaction_cost_bps: float = 5.0,
) -> dict[str, float | str]:
    signals = build_signal_frame(merged, ratio_ma=ratio_ma, regime_mode=regime_mode)
    bt = run_strategy_backtest(
        signals,
        rebalance_threshold=rebalance_threshold,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_freq=rebalance_freq,
    )
    m = compute_metrics(bt)
    return {'regime_mode': regime_mode, 'rebalance_freq': rebalance_freq, **m}


def main() -> None:
    parser = argparse.ArgumentParser(description='2x2 decomposition analysis')
    parser.add_argument('--breadth-csv', required=True, help='Path to market breadth CSV')
    parser.add_argument('--prices-csv', required=True, help='Path to prices CSV')
    parser.add_argument('--output-dir', default='outputs_2x2', help='Output directory')
    parser.add_argument('--ratio-ma', type=int, default=10)
    parser.add_argument('--rebalance-threshold', type=float, default=0.05)
    parser.add_argument('--transaction-cost-bps', type=float, default=5.0)
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

    regime_modes = ['pointwise', 'series']
    rebalance_freqs = ['daily', 'month_end']

    rows = []
    for mode, freq in itertools.product(regime_modes, rebalance_freqs):
        result = _run_combo(
            merged,
            regime_mode=mode,
            rebalance_freq=freq,
            ratio_ma=args.ratio_ma,
            rebalance_threshold=args.rebalance_threshold,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        rows.append(result)
        label = f'{mode:>10s} + {freq:>10s}'
        print(f'{label}: CAGR={result["cagr"]:.4f}  Sharpe={result["sharpe"]:.4f}  '
              f'MaxDD={result["max_drawdown"]:.4f}  Equity={result["final_equity"]:.4f}')

    df = pd.DataFrame(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'decomposition_2x2.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # Marginal contribution analysis
    print('\n--- Marginal Contribution ---')

    # Signal effect: avg(NEW - OLD) across rebalance modes
    for freq in rebalance_freqs:
        old_row = df[(df['regime_mode'] == 'pointwise') & (df['rebalance_freq'] == freq)]
        new_row = df[(df['regime_mode'] == 'series') & (df['rebalance_freq'] == freq)]
        if len(old_row) and len(new_row):
            delta_cagr = new_row['cagr'].values[0] - old_row['cagr'].values[0]
            delta_sharpe = new_row['sharpe'].values[0] - old_row['sharpe'].values[0]
            print(f'  Signal effect ({freq:>10s}): dCAGR={delta_cagr:+.4f}  dSharpe={delta_sharpe:+.4f}')

    # Rebalance effect: avg(month_end - daily) across signal modes
    for mode in regime_modes:
        daily_row = df[(df['regime_mode'] == mode) & (df['rebalance_freq'] == 'daily')]
        me_row = df[(df['regime_mode'] == mode) & (df['rebalance_freq'] == 'month_end')]
        if len(daily_row) and len(me_row):
            delta_cagr = me_row['cagr'].values[0] - daily_row['cagr'].values[0]
            delta_sharpe = me_row['sharpe'].values[0] - daily_row['sharpe'].values[0]
            print(f'  Rebalance effect ({mode:>10s}): dCAGR={delta_cagr:+.4f}  dSharpe={delta_sharpe:+.4f}')


if __name__ == '__main__':
    main()
