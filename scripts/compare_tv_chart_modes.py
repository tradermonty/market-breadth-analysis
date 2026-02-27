"""Compare TV mode vs Chart mode backtest results for all ETFs."""

import os
import sys
import time
from datetime import datetime

import matplotlib
import pandas as pd

matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.backtest import Backtest

ETFS = [
    'SPY',
    'VOO',
    'VTI',
    'QQQ',
    'VUG',
    'VTV',
    'VB',
    'VEA',
    'VWO',
    'XLF',
    'XLE',
    'XLK',
    'XLV',
    'XLI',
    'VGT',
    'SSO',
    'TQQQ',
    'QLD',
    'SPXL',
    'SOXL',
    'TNA',
    'IWR',
    'SCHG',
    'IWF',
    'MTUM',
    'VYM',
    'SCHD',
    'NOBL',
]

# TV mode ETF-specific overrides (same as run_multi_etf_backtest.py)
TV_OVERRIDES = {
    'TQQQ': {'peak_level': 0.80, 'prom_thresh_long': 0.01, 'vol_atr_multiplier': 3.0},
    'SOXL': {'peak_level': 0.80, 'prom_thresh_long': 0.01, 'vol_atr_multiplier': 3.0},
    'QLD': {'peak_level': 0.80, 'prom_thresh_long': 0.008},
    'SPXL': {'peak_level': 0.75, 'prom_thresh_long': 0.008},
    'TNA': {'peak_level': 0.75, 'prom_thresh_long': 0.008},
}

COMMON_PARAMS = {
    'start_date': '2008-01-01',
    'end_date': '2026-02-26',
    'short_ma': 5,
    'long_ma': 200,
    'initial_capital': 50000,
    'slippage': 0.0,
    'commission': 0.0001,
    'use_saved_data': True,
    'debug': False,
    'threshold': 0.5,
    'ma_type': 'ema',
    'stop_loss_pct': 0.08,
    'no_show_plot': True,
}

METRICS = [
    'Total Return',
    'CAGR',
    'Sharpe',
    'Max DD',
    'Win Rate',
    'P/L Ratio',
    'Profit Factor',
    'Calmar',
]


def run_single(symbol, mode):
    """Run a single backtest and return metrics dict."""
    params = dict(COMMON_PARAMS, symbol=symbol)

    if mode == 'tv':
        overrides = TV_OVERRIDES.get(symbol, {})
        params.update(
            tv_mode=True,
            no_pyramiding=True,
            pivot_len_long=overrides.get('pivot_len_long', 20),
            pivot_len_short=overrides.get('pivot_len_short', 10),
            prom_thresh_long=overrides.get('prom_thresh_long', 0.005),
            prom_thresh_short=overrides.get('prom_thresh_short', 0.03),
            peak_level=overrides.get('peak_level', 0.70),
            trough_level_long=overrides.get('trough_level_long', 0.40),
            trough_level_short=overrides.get('trough_level_short', 0.20),
        )
    elif mode == 'chart':
        params['chart_mode'] = True

    bt = Backtest(**params)
    bt.run()
    bt.visualize_results(show_plot=False)

    return {
        'Total Return': bt.total_return,
        'CAGR': bt.cagr,
        'Sharpe': bt.sharpe_ratio,
        'Max DD': bt.max_drawdown,
        'Win Rate': bt.win_rate,
        'P/L Ratio': bt.profit_loss_ratio,
        'Profit Factor': bt.profit_factor,
        'Calmar': bt.calmar_ratio,
        'B&H Return': bt.bh_total_return,
        'B&H CAGR': bt.bh_cagr,
        'B&H Sharpe': bt.bh_sharpe,
        'B&H Max DD': bt.bh_max_drawdown,
    }


def fmt_pct(v):
    if pd.isna(v):
        return 'N/A'
    return f'{v * 100:.1f}%'


def fmt_f2(v):
    if pd.isna(v):
        return 'N/A'
    return f'{v:.2f}'


def main():
    tv_results = []
    chart_results = []

    for symbol in ETFS:
        print(f'\n{"=" * 60}')
        print(f'  {symbol}')
        print(f'{"=" * 60}')

        # TV mode
        try:
            print('  [TV mode] running...')
            tv = run_single(symbol, 'tv')
            tv['Symbol'] = symbol
            tv_results.append(tv)
            print(f'  [TV mode] Return={tv["Total Return"] * 100:.1f}%, Sharpe={tv["Sharpe"]:.2f}')
        except Exception as e:
            print(f'  [TV mode] ERROR: {e}')
            tv_results.append({'Symbol': symbol})

        # Chart mode
        try:
            print('  [Chart mode] running...')
            ch = run_single(symbol, 'chart')
            ch['Symbol'] = symbol
            chart_results.append(ch)
            print(f'  [Chart mode] Return={ch["Total Return"] * 100:.1f}%, Sharpe={ch["Sharpe"]:.2f}')
        except Exception as e:
            print(f'  [Chart mode] ERROR: {e}')
            chart_results.append({'Symbol': symbol})

        time.sleep(1)

    # Build comparison report
    tv_df = pd.DataFrame(tv_results)
    ch_df = pd.DataFrame(chart_results)

    lines = []
    lines.append('# TV Mode vs Chart Mode Backtest Comparison')
    lines.append('')
    lines.append(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Period: {COMMON_PARAMS["start_date"]} to {COMMON_PARAMS["end_date"]}')
    lines.append('')
    lines.append('Common Parameters:')
    lines.append(f'- Short MA: {COMMON_PARAMS["short_ma"]}, Long MA: {COMMON_PARAMS["long_ma"]}')
    lines.append(f'- Stop Loss: {COMMON_PARAMS["stop_loss_pct"]:.0%}')
    lines.append(f'- MA Type: {COMMON_PARAMS["ma_type"].upper()}')
    lines.append(f'- Slippage: {COMMON_PARAMS["slippage"]}, Commission: {COMMON_PARAMS["commission"]}')
    lines.append(f'- Initial Capital: ${COMMON_PARAMS["initial_capital"]:,.0f}')
    lines.append('')
    lines.append('TV Mode: tv_mode=True, no_pyramiding=True, pivot-based detection')
    lines.append('Chart Mode: chart_mode=True, chart-style find_peaks (distance=50, no level filters)')
    lines.append('')

    # Summary table
    lines.append('## Summary Comparison')
    lines.append('')
    header = '| Symbol | TV Return | Chart Return | TV CAGR | Chart CAGR | TV Sharpe | Chart Sharpe | TV MaxDD | Chart MaxDD | B&H Return | B&H MaxDD |'
    sep = '|---|---|---|---|---|---|---|---|---|---|---|'
    lines.append(header)
    lines.append(sep)

    for i, symbol in enumerate(ETFS):
        tv = tv_results[i] if i < len(tv_results) else {}
        ch = chart_results[i] if i < len(chart_results) else {}

        row = f'| {symbol}'
        row += f' | {fmt_pct(tv.get("Total Return"))}'
        row += f' | {fmt_pct(ch.get("Total Return"))}'
        row += f' | {fmt_pct(tv.get("CAGR"))}'
        row += f' | {fmt_pct(ch.get("CAGR"))}'
        row += f' | {fmt_f2(tv.get("Sharpe"))}'
        row += f' | {fmt_f2(ch.get("Sharpe"))}'
        row += f' | {fmt_pct(tv.get("Max DD"))}'
        row += f' | {fmt_pct(ch.get("Max DD"))}'
        row += f' | {fmt_pct(tv.get("B&H Return"))}'
        row += f' | {fmt_pct(tv.get("B&H Max DD"))}'
        row += ' |'
        lines.append(row)

    lines.append('')

    # Detailed table: Win Rate, P/L, Profit Factor, Calmar
    lines.append('## Detailed Metrics')
    lines.append('')
    header2 = (
        '| Symbol | TV WinRate | Chart WinRate | TV P/L | Chart P/L | TV PF | Chart PF | TV Calmar | Chart Calmar |'
    )
    sep2 = '|---|---|---|---|---|---|---|---|---|'
    lines.append(header2)
    lines.append(sep2)

    for i, symbol in enumerate(ETFS):
        tv = tv_results[i] if i < len(tv_results) else {}
        ch = chart_results[i] if i < len(chart_results) else {}

        row = f'| {symbol}'
        row += f' | {fmt_pct(tv.get("Win Rate"))}'
        row += f' | {fmt_pct(ch.get("Win Rate"))}'
        row += f' | {fmt_f2(tv.get("P/L Ratio"))}'
        row += f' | {fmt_f2(ch.get("P/L Ratio"))}'
        row += f' | {fmt_f2(tv.get("Profit Factor"))}'
        row += f' | {fmt_f2(ch.get("Profit Factor"))}'
        row += f' | {fmt_f2(tv.get("Calmar"))}'
        row += f' | {fmt_f2(ch.get("Calmar"))}'
        row += ' |'
        lines.append(row)

    lines.append('')

    # Winner count
    lines.append('## Mode Comparison Summary')
    lines.append('')
    tv_wins_return = 0
    ch_wins_return = 0
    tv_wins_sharpe = 0
    ch_wins_sharpe = 0
    tv_wins_dd = 0
    ch_wins_dd = 0

    for i in range(len(ETFS)):
        tv = tv_results[i] if i < len(tv_results) else {}
        ch = chart_results[i] if i < len(chart_results) else {}

        tv_ret = tv.get('Total Return')
        ch_ret = ch.get('Total Return')
        if tv_ret is not None and ch_ret is not None and not (pd.isna(tv_ret) or pd.isna(ch_ret)):
            if tv_ret > ch_ret:
                tv_wins_return += 1
            elif ch_ret > tv_ret:
                ch_wins_return += 1

        tv_sh = tv.get('Sharpe')
        ch_sh = ch.get('Sharpe')
        if tv_sh is not None and ch_sh is not None and not (pd.isna(tv_sh) or pd.isna(ch_sh)):
            if tv_sh > ch_sh:
                tv_wins_sharpe += 1
            elif ch_sh > tv_sh:
                ch_wins_sharpe += 1

        tv_dd = tv.get('Max DD')
        ch_dd = ch.get('Max DD')
        if tv_dd is not None and ch_dd is not None and not (pd.isna(tv_dd) or pd.isna(ch_dd)):
            # Less negative = better
            if tv_dd > ch_dd:
                tv_wins_dd += 1
            elif ch_dd > tv_dd:
                ch_wins_dd += 1

    lines.append('| Metric | TV Mode Wins | Chart Mode Wins |')
    lines.append('|---|---|---|')
    lines.append(f'| Total Return | {tv_wins_return} | {ch_wins_return} |')
    lines.append(f'| Sharpe Ratio | {tv_wins_sharpe} | {ch_wins_sharpe} |')
    lines.append(f'| Max Drawdown (smaller) | {tv_wins_dd} | {ch_wins_dd} |')
    lines.append('')

    report = '\n'.join(lines)

    # Save report
    report_path = 'reports/tv_vs_chart_mode_comparison.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'\nReport saved to {report_path}')

    # Also save raw data as CSV
    tv_df.to_csv('reports/tv_mode_results.csv', index=False)
    ch_df.to_csv('reports/chart_mode_results.csv', index=False)
    print('Raw data saved to reports/tv_mode_results.csv and reports/chart_mode_results.csv')

    print('\n' + report)


if __name__ == '__main__':
    main()
