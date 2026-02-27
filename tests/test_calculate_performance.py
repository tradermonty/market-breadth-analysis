"""Tests for calculate_performance() edge cases.

Verifies CAGR, Sharpe, B&H guards added by the code review fix.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.backtest import Backtest


def _make_backtest(**kwargs):
    """Create a minimal Backtest instance for performance calculation tests."""
    defaults = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'use_saved_data': True,
        'no_show_plot': True,
        'initial_capital': 50000,
        'symbol': 'SPY',
    }
    defaults.update(kwargs)
    return Backtest(**defaults)


def _inject_equity_curve(bt, dates, values):
    """Inject a hand-crafted equity curve into a Backtest instance
    and set up the minimal price_data needed by calculate_performance().
    """
    bt.equity_curve = [{'date': d, 'equity': v} for d, v in zip(dates, values)]

    # Minimal price_data for B&H calculation
    bt.price_data = pd.DataFrame(
        {'adjusted_close': [v / bt.initial_capital * 100 for v in values]},
        index=dates,
    )
    bt.trades = []  # No trades by default


class TestCalculatePerformance(unittest.TestCase):
    """Test cases for calculate_performance() edge cases and bug fixes."""

    def test_01_cagr_positive_return(self):
        """Positive return produces correct CAGR."""
        bt = _make_backtest()
        dates = pd.date_range('2024-01-01', periods=366, freq='D')
        # 10% total return over ~1 year
        values = np.linspace(50000, 55000, len(dates))
        _inject_equity_curve(bt, dates, values)

        bt.calculate_performance()

        self.assertGreater(bt.cagr, 0)
        self.assertAlmostEqual(bt.total_return, 0.10, places=2)
        # CAGR should be close to 10% for 1 year
        self.assertAlmostEqual(bt.cagr, 0.10, delta=0.01)

    def test_02_cagr_negative_return(self):
        """Negative return produces negative CAGR (sign bug fix verification)."""
        bt = _make_backtest()
        dates = pd.date_range('2024-01-01', periods=366, freq='D')
        # -10% total return over ~1 year
        values = np.linspace(50000, 45000, len(dates))
        _inject_equity_curve(bt, dates, values)

        bt.calculate_performance()

        self.assertLess(bt.cagr, 0)
        self.assertAlmostEqual(bt.total_return, -0.10, places=2)
        # CAGR should be close to -10% for 1 year
        self.assertAlmostEqual(bt.cagr, -0.10, delta=0.01)

    def test_03_cagr_zero_days(self):
        """days=0 (single data point) should not raise, CAGR=0."""
        bt = _make_backtest()
        single_date = pd.Timestamp('2024-01-01')
        bt.equity_curve = [{'date': single_date, 'equity': 50000}]
        bt.price_data = pd.DataFrame(
            {'adjusted_close': [100.0]},
            index=[single_date],
        )
        bt.trades = []

        bt.calculate_performance()

        self.assertEqual(bt.cagr, 0.0)
        self.assertEqual(bt.annual_return, 0.0)

    def test_04_cagr_total_loss(self):
        """total_return=-1 (total loss) should produce CAGR=-1."""
        bt = _make_backtest()
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Drop to zero (total loss)
        values = np.linspace(50000, 0, len(dates))
        values[-1] = 0  # Ensure exact zero
        _inject_equity_curve(bt, dates, values)

        bt.calculate_performance()

        self.assertEqual(bt.total_return, -1.0)
        self.assertEqual(bt.cagr, -1.0)
        self.assertEqual(bt.annual_return, -1.0)

    def test_05_sharpe_zero_std(self):
        """Flat equity curve (zero std) should produce Sharpe=0, not inf/NaN."""
        bt = _make_backtest()
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Completely flat equity
        values = [50000.0] * len(dates)
        _inject_equity_curve(bt, dates, values)

        bt.calculate_performance()

        self.assertEqual(bt.sharpe_ratio, 0.0)
        self.assertFalse(np.isnan(bt.sharpe_ratio))
        self.assertFalse(np.isinf(bt.sharpe_ratio))

    def test_06_bh_zero_shares(self):
        """Very high initial price causing buy_hold_shares=0 should not crash."""
        bt = _make_backtest(initial_capital=10)  # Very small capital
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = np.linspace(10, 11, len(dates))
        bt.equity_curve = [{'date': d, 'equity': v} for d, v in zip(dates, values)]

        # Price much higher than capital → 0 shares
        bt.price_data = pd.DataFrame(
            {'adjusted_close': [10000.0] * len(dates)},
            index=dates,
        )
        bt.trades = []

        bt.calculate_performance()

        self.assertEqual(bt.bh_total_return, 0.0)
        self.assertEqual(bt.bh_cagr, 0.0)
        self.assertEqual(bt.bh_sharpe, 0.0)
        self.assertEqual(bt.bh_max_drawdown, 0.0)

    def test_07_empty_equity_curve(self):
        """Empty equity_curve should set bh_* attributes without error."""
        bt = _make_backtest()
        bt.equity_curve = []
        bt.trades = []

        bt.calculate_performance()

        self.assertTrue(hasattr(bt, 'bh_total_return'))
        self.assertTrue(hasattr(bt, 'bh_cagr'))
        self.assertTrue(hasattr(bt, 'bh_sharpe'))
        self.assertTrue(hasattr(bt, 'bh_max_drawdown'))
        self.assertEqual(bt.bh_total_return, 0)
        self.assertEqual(bt.bh_cagr, 0)

    def test_08_bh_calculation_correctness(self):
        """B&H return matches manual formula calculation."""
        bt = _make_backtest(slippage=0.001, commission=0.001)
        dates = pd.date_range('2024-01-01', periods=252, freq='B')
        initial_price = 100.0
        final_price = 110.0
        prices = np.linspace(initial_price, final_price, len(dates))

        bt.equity_curve = [{'date': d, 'equity': 50000 * (p / initial_price)} for d, p in zip(dates, prices)]
        bt.price_data = pd.DataFrame({'adjusted_close': prices}, index=dates)
        bt.trades = []

        bt.calculate_performance()

        # Manual B&H calculation
        buy_hold_shares = int(bt.initial_capital / (initial_price * (1 + bt.slippage)))
        buy_hold_cost = buy_hold_shares * initial_price * (1 + bt.slippage) * (1 + bt.commission)
        buy_hold_value = buy_hold_shares * final_price * (1 - bt.slippage) * (1 - bt.commission)
        expected_bh_return = (buy_hold_value / buy_hold_cost) - 1

        self.assertAlmostEqual(bt.bh_total_return, expected_bh_return, places=6)
        self.assertGreater(bt.bh_total_return, 0)

    def test_09_run_end_to_end_with_saved_data(self):
        """run() completes without error and sets all expected attributes."""
        import pathlib

        data_file = pathlib.Path('data/sp500_all_stocks.csv')
        if not data_file.exists():
            self.skipTest('Saved data not available')

        # Also check that the CSV covers the required date range to avoid
        # falling through to the API fetch path.
        check_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if (
            check_df.empty
            or check_df.index.min() > pd.Timestamp('2021-01-01')
            or check_df.index.max() < pd.Timestamp('2023-06-30')
        ):
            self.skipTest('Saved data does not cover the required date range')

        bt = _make_backtest(
            start_date='2023-01-01',
            end_date='2023-06-30',
            use_saved_data=True,
            no_show_plot=True,
            symbol='SPY',
        )
        bt.run()

        # Verify all expected attributes exist and are not NaN
        for attr in (
            'total_return',
            'cagr',
            'sharpe_ratio',
            'max_drawdown',
            'bh_total_return',
            'bh_cagr',
            'bh_sharpe',
            'bh_max_drawdown',
        ):
            self.assertTrue(hasattr(bt, attr), f'Missing attribute: {attr}')
            val = getattr(bt, attr)
            self.assertFalse(np.isnan(val), f'{attr} is NaN')
            self.assertFalse(np.isinf(val), f'{attr} is inf')

        # Equity curve should not be empty
        self.assertGreater(len(bt.equity_curve), 0)


if __name__ == '__main__':
    unittest.main()
