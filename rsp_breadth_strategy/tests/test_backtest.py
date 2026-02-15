import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rsp_breadth_strategy.backtest import run_strategy_backtest


class TestBacktest(unittest.TestCase):
    def _base_frame(self):
        dates = pd.date_range('2024-01-31', periods=4, freq='ME')
        return pd.DataFrame(
            {
                'Date': dates,
                'SPY': [100.0, 100.0, 100.0, 100.0],
                'RSP': [100.0, 110.0, 110.0, 110.0],
                'XLE': [100.0, 100.0, 100.0, 100.0],
                'SGOV': [100.0, 100.0, 100.0, 100.0],
                'regime': ['RiskOn', 'RiskOn', 'RiskOn', 'RiskOn'],
                'r_trend': [1, 1, 1, 1],
                'x_trend': [1, 1, 1, 1],
            }
        )

    def test_first_month_return_uses_previous_weights(self):
        frame = self._base_frame()
        result = run_strategy_backtest(
            frame, rebalance_threshold=0.0, transaction_cost_bps=0.0, rebalance_freq='daily',
        )

        # month2 (index=1) return should be 40% weight * +10% RSP = +4%
        self.assertAlmostEqual(result.loc[1, 'portfolio_return'], 0.04, places=6)
        self.assertAlmostEqual(result.loc[1, 'equity'], 1.04, places=6)

    def test_rebalance_cost_is_applied(self):
        frame = self._base_frame()
        frame.loc[1, 'r_trend'] = 0  # target from month2 changes to SPY55/RSP25/XLE10/SGOV10

        result = run_strategy_backtest(
            frame, rebalance_threshold=0.0, transaction_cost_bps=100.0, rebalance_freq='daily',
        )

        # Turnover = (|+10|+|-15|+0|+5)/2 = 15% => 0.15 * 1% = 0.15% cost
        self.assertAlmostEqual(result.loc[1, 'turnover'], 0.15, places=6)
        self.assertAlmostEqual(result.loc[1, 'transaction_cost'], 0.0015, places=6)

    def test_threshold_can_skip_rebalance(self):
        frame = self._base_frame()
        frame.loc[1, 'r_trend'] = 0

        result = run_strategy_backtest(
            frame, rebalance_threshold=0.20, transaction_cost_bps=100.0, rebalance_freq='daily',
        )
        self.assertEqual(int(result.loc[1, 'rebalanced']), 0)
        self.assertAlmostEqual(result.loc[1, 'turnover'], 0.0, places=6)


class TestMonthEndRebalance(unittest.TestCase):
    """Tests for month_end rebalance frequency."""

    def _daily_frame_two_months(self):
        """Create a daily frame spanning 2 months with a regime change mid-month."""
        dates = pd.bdate_range('2024-01-02', '2024-02-29')
        n = len(dates)
        frame = pd.DataFrame(
            {
                'Date': dates,
                'SPY': [100.0] * n,
                'RSP': [100.0] * n,
                'XLE': [100.0] * n,
                'SGOV': [100.0] * n,
                'regime': ['RiskOn'] * n,
                'r_trend': [1] * n,
                'x_trend': [1] * n,
            }
        )
        return frame

    def test_month_end_skips_mid_month_regime_change(self):
        """With month_end freq, mid-month regime changes don't trigger rebalance."""
        frame = self._daily_frame_two_months()
        # Change regime to Deterioration on Jan 15 (mid-month)
        mid_idx = frame[frame['Date'] == '2024-01-16'].index
        if len(mid_idx) == 0:
            mid_idx = frame.index[10:11]
        for idx in mid_idx:
            frame.loc[idx, 'regime'] = 'Deterioration'

        result = run_strategy_backtest(
            frame, rebalance_threshold=0.0, transaction_cost_bps=0.0, rebalance_freq='month_end',
        )

        # Mid-month rows should NOT have rebalanced
        mid_rows = result[(result['Date'].dt.month == 1) & (result['Date'].dt.day < 28)]
        self.assertEqual(int(mid_rows['rebalanced'].sum()), 0)

    def test_daily_freq_backward_compat(self):
        """rebalance_freq='daily' behaves like the original implementation."""
        frame = self._daily_frame_two_months()
        # Force a large regime change on day 5
        frame.loc[4, 'regime'] = 'Deterioration'

        result = run_strategy_backtest(
            frame, rebalance_threshold=0.0, transaction_cost_bps=0.0, rebalance_freq='daily',
        )

        # Day 5 should have triggered a rebalance (regime changed)
        self.assertEqual(int(result.loc[4, 'rebalanced']), 1)


if __name__ == '__main__':
    unittest.main()
