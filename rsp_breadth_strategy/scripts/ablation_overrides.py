"""Ablation study: individual ON/OFF of Bearish_Signal and Breadth_200MA_Trend.

Tests 4 combinations to isolate which override is responsible for
the CAGR drag observed in the 2x2 decomposition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rsp_breadth_strategy.backtest import run_strategy_backtest
from rsp_breadth_strategy.metrics import compute_metrics
from rsp_breadth_strategy.signals import build_signal_frame

COMBOS = [
    ('both_ON', True, True),
    ('bearish_only', True, False),
    ('trend_only', False, True),
    ('neither', False, False),
]


def _run_ablation(
    merged: pd.DataFrame,
    use_bearish: bool,
    use_trend: bool,
    ratio_ma: int = 10,
    rebalance_threshold: float = 0.05,
    transaction_cost_bps: float = 5.0,
) -> dict[str, float]:
    df = merged.copy()
    if not use_bearish and 'Bearish_Signal' in df.columns:
        df = df.drop(columns=['Bearish_Signal'])
    if not use_trend and 'Breadth_200MA_Trend' in df.columns:
        df = df.drop(columns=['Breadth_200MA_Trend'])

    signals = build_signal_frame(df, ratio_ma=ratio_ma, regime_mode='series')
    bt = run_strategy_backtest(
        signals,
        rebalance_threshold=rebalance_threshold,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_freq='month_end',
    )
    return compute_metrics(bt)


def main() -> None:
    parser = argparse.ArgumentParser(description='Ablation study for override signals')
    parser.add_argument('--breadth-csv', required=True, help='Path to market breadth CSV')
    parser.add_argument('--prices-csv', required=True, help='Path to prices CSV')
    parser.add_argument('--output-dir', default='outputs_ablation', help='Output directory')
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

    rows = []
    for label, use_bearish, use_trend in COMBOS:
        m = _run_ablation(
            merged,
            use_bearish=use_bearish,
            use_trend=use_trend,
            ratio_ma=args.ratio_ma,
            rebalance_threshold=args.rebalance_threshold,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        row = {
            'label': label,
            'bearish_signal': use_bearish,
            'breadth_trend': use_trend,
            **m,
        }
        rows.append(row)
        print(f'{label:>15s} (B={use_bearish}, T={use_trend}): '
              f'CAGR={m["cagr"]:.4f}  Sharpe={m["sharpe"]:.4f}  '
              f'MaxDD={m["max_drawdown"]:.4f}  Equity={m["final_equity"]:.4f}')

    df = pd.DataFrame(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'ablation_overrides.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # Marginal effects
    print('\n--- Marginal Effects ---')
    both = df[df['label'] == 'both_ON'].iloc[0]
    neither = df[df['label'] == 'neither'].iloc[0]
    bearish_only = df[df['label'] == 'bearish_only'].iloc[0]
    trend_only = df[df['label'] == 'trend_only'].iloc[0]

    bearish_effect_cagr = 0.5 * ((both['cagr'] - trend_only['cagr']) +
                                  (bearish_only['cagr'] - neither['cagr']))
    bearish_effect_sharpe = 0.5 * ((both['sharpe'] - trend_only['sharpe']) +
                                    (bearish_only['sharpe'] - neither['sharpe']))
    trend_effect_cagr = 0.5 * ((both['cagr'] - bearish_only['cagr']) +
                                (trend_only['cagr'] - neither['cagr']))
    trend_effect_sharpe = 0.5 * ((both['sharpe'] - bearish_only['sharpe']) +
                                  (trend_only['sharpe'] - neither['sharpe']))

    print(f'  Bearish_Signal effect:     dCAGR={bearish_effect_cagr:+.4f}  dSharpe={bearish_effect_sharpe:+.4f}')
    print(f'  Breadth_200MA_Trend effect: dCAGR={trend_effect_cagr:+.4f}  dSharpe={trend_effect_sharpe:+.4f}')


if __name__ == '__main__':
    main()
