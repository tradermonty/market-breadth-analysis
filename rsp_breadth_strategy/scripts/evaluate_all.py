"""Consolidated evaluation script.

Runs all analysis pipelines in sequence:
  1. Baseline CLI backtest
  2. 2x2 decomposition
  3. Ablation study
  4. Sensitivity sweep (full + OOS)

Usage:
  PYTHONPATH=src python3 scripts/evaluate_all.py \
    --breadth-csv ../reports/market_breadth_data_20260214_ma8.csv \
    --prices-csv ../data/prices_combined.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(label: str, cmd: list[str]) -> None:
    print(f'\n{"=" * 60}', flush=True)
    print(f'  {label}', flush=True)
    print(f'{"=" * 60}', flush=True)
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    if result.returncode != 0:
        print(f'  [WARN] {label} exited with code {result.returncode}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run all evaluation pipelines')
    parser.add_argument('--breadth-csv', required=True, help='Path to market breadth CSV')
    parser.add_argument('--prices-csv', required=True, help='Path to prices CSV')
    parser.add_argument('--output-dir', default='outputs_eval', help='Root output directory')
    parser.add_argument('--skip-sensitivity', action='store_true', help='Skip sensitivity sweep (slow)')
    args = parser.parse_args()

    py = sys.executable
    scripts_dir = Path(__file__).resolve().parent
    out = Path(args.output_dir)

    # 1. Baseline CLI
    _run('1. Baseline backtest', [
        py, '-m', 'rsp_breadth_strategy.cli',
        '--breadth-csv', args.breadth_csv,
        '--prices-csv', args.prices_csv,
        '--output-dir', str(out / 'baseline'),
    ])

    # 2. 2x2 Decomposition
    _run('2. 2x2 Decomposition', [
        py, str(scripts_dir / 'decomposition_2x2.py'),
        '--breadth-csv', args.breadth_csv,
        '--prices-csv', args.prices_csv,
        '--output-dir', str(out / '2x2'),
    ])

    # 3. Ablation
    _run('3. Ablation study', [
        py, str(scripts_dir / 'ablation_overrides.py'),
        '--breadth-csv', args.breadth_csv,
        '--prices-csv', args.prices_csv,
        '--output-dir', str(out / 'ablation'),
    ])

    if not args.skip_sensitivity:
        # 4a. Sensitivity (full)
        _run('4a. Sensitivity sweep (full period)', [
            py, str(scripts_dir / 'sensitivity_sweep.py'),
            '--breadth-csv', args.breadth_csv,
            '--prices-csv', args.prices_csv,
            '--output-dir', str(out / 'sensitivity'),
        ])

        # 4b. Sensitivity (OOS)
        _run('4b. Sensitivity sweep (IS/OOS split)', [
            py, str(scripts_dir / 'sensitivity_sweep.py'),
            '--breadth-csv', args.breadth_csv,
            '--prices-csv', args.prices_csv,
            '--output-dir', str(out / 'sensitivity'),
            '--oos',
        ])

    print(f'\n{"=" * 60}')
    print(f'  All evaluations complete. Output: {out}/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
