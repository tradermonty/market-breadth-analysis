from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import run_strategy_backtest
from .metrics import compute_metrics
from .signals import build_signal_frame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run RSP breadth strategy backtest')
    parser.add_argument('--breadth-csv', required=True, help='Path to market breadth CSV')
    parser.add_argument('--prices-csv', required=True, help='Path to prices CSV (Date, SPY, RSP, XLE, SGOV)')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--ratio-ma', type=int, default=10, help='Moving average window for relative trend')
    parser.add_argument('--rebalance-threshold', type=float, default=0.05, help='Rebalance threshold')
    parser.add_argument('--transaction-cost-bps', type=float, default=5.0, help='Transaction cost in bps')
    parser.add_argument(
        '--rebalance-freq',
        choices=['month_end', 'daily'],
        default='month_end',
        help='Rebalance frequency (default: month_end)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    breadth = _read_csv(Path(args.breadth_csv))
    prices = _read_csv(Path(args.prices_csv))

    breadth['Date'] = pd.to_datetime(breadth['Date'])
    prices['Date'] = pd.to_datetime(prices['Date'])

    merged = breadth.merge(
        prices[['Date', 'SPY', 'RSP', 'XLE', 'SGOV']],
        on='Date',
        how='inner',
    )
    signals = build_signal_frame(merged, ratio_ma=args.ratio_ma)
    backtest = run_strategy_backtest(
        signals,
        rebalance_threshold=args.rebalance_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
        rebalance_freq=args.rebalance_freq,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_path = out_dir / 'signals.csv'
    backtest_path = out_dir / 'backtest.csv'
    summary_path = out_dir / 'summary.json'

    signals.to_csv(signals_path, index=False)
    backtest.to_csv(backtest_path, index=False)

    metrics = compute_metrics(backtest)
    summary = {
        'rows': int(len(backtest)),
        'start_date': str(backtest['Date'].iloc[0]),
        'end_date': str(backtest['Date'].iloc[-1]),
        **metrics,
        'rebalance_freq': args.rebalance_freq,
        'rebalance_threshold': args.rebalance_threshold,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'Saved: {signals_path}')
    print(f'Saved: {backtest_path}')
    print(f'Saved: {summary_path}')


if __name__ == '__main__':
    main()
