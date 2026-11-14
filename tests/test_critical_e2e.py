"""Actual analysis/backtest/trigger entry points over fixed synthetic cached data."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cached(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(21)
    dates = pd.bdate_range('2020-01-02', '2024-06-03')
    wave = 200 + np.sin(np.arange(len(dates)) / 35) * 25 + np.arange(len(dates)) * 0.03
    pd.DataFrame({name: wave + rng.normal(0, 1, len(dates)) for name in ('AAPL', 'MSFT', 'GOOGL')}, index=dates).to_csv(
        data / 'sp500_all_stocks.csv'
    )
    for name in ('sp500', 'SPY', 'QQQ'):
        price = pd.Series(
            300 + np.arange(len(dates)) * 0.1 + np.sin(np.arange(len(dates)) / 25) * 8,
            index=dates,
            name='adjusted_close',
        )
        price.to_csv(data / f'{name}_price_data.csv')
        pd.DataFrame(
            {'adjusted_close': price, 'close': price, 'open': price, 'high': price + 1, 'low': price - 1}, index=dates
        ).to_csv(data / f'{name}_ohlc_data.csv')
    return tmp_path


def child(cwd, args):
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
        FMP_API_KEY='',
        DOTENV_DISABLED='1',
        MPLBACKEND='Agg',
    )
    result = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    return result


def test_01_cached_analysis_emits_required_outputs(cached):
    result = child(
        cached,
        [
            str(ROOT / 'market_breadth.py'),
            '--use_saved_data',
            '--start_date',
            '2022-01-03',
            '--end_date',
            '2024-06-03',
            '--include_50ma',
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = cached / 'reports'
    assert 'Plotly.newPlot' in (report / 'market_breadth.html').read_text()
    data = pd.read_csv(report / 'market_breadth_data.csv')
    assert data['Date'].iloc[-1] == '2024-06-03'
    assert data['Coverage_200'].iloc[-1] == 1
    assert data['Coverage_50'].iloc[-1] == 1
    assert 'Analysis Period End' in (report / 'market_breadth_summary.csv').read_text()


def test_02_single_etf_and_multi_etf_actual_runs(cached):
    single = child(
        cached,
        [
            str(ROOT / 'backtest/backtest.py'),
            '--symbol',
            'SPY',
            '--use_saved_data',
            '--no_show_plot',
            '--start_date',
            '2022-01-03',
            '--end_date',
            '2024-06-03',
        ],
    )
    assert single.returncode == 0, single.stdout + single.stderr
    assert list((cached / 'reports').glob('backtest_results_SPY*.png'))
    code = (
        'from backtest.run_multi_etf_backtest import run_multi_etf_backtest\n'
        'import numpy as np\n'
        'result = run_multi_etf_backtest(["SPY", "QQQ"], start_date="2022-01-03", end_date="2024-06-03", use_saved_data=True, no_show_plot=True)\n'
        'assert list(result["Symbol"]) == ["SPY", "QQQ"]\n'
        'assert np.isfinite(result["Total Return"]).all()\n'
    )
    multi = child(cached, ['-c', code])
    assert multi.returncode == 0, multi.stdout + multi.stderr
    summary = pd.read_csv(cached / 'reports/backtest_results_summary.csv')
    assert set(summary['Symbol']) == {'SPY', 'QQQ'}
    assert np.isfinite(summary['Total Return']).all()


def test_03_trigger_cli_failed_download_is_nonzero(tmp_path):
    code = (
        'from unittest.mock import patch\nimport requests, sys\nimport trigger_market_breadth as trigger\n'
        'sys.argv = ["trigger_market_breadth.py", "--fetch-only"]\n'
        'with patch.object(requests, "get", side_effect=requests.ConnectionError("synthetic failure")):\n'
        '    trigger.main()\n'
    )
    result = child(tmp_path, ['-c', code])
    assert result.returncode == 1
    assert 'Error:' in result.stderr
