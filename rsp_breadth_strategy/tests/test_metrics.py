import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rsp_breadth_strategy.backtest import run_strategy_backtest
from rsp_breadth_strategy.metrics import compute_metrics


class TestPostCostReturn(unittest.TestCase):
    """Tests for post_cost_return column added to backtest output."""

    def _base_frame(self, n=4):
        dates = pd.date_range('2024-01-31', periods=n, freq='ME')
        return pd.DataFrame(
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

    def test_post_cost_return_column_exists(self):
        """Backtest output must contain post_cost_return column."""
        frame = self._base_frame()
        result = run_strategy_backtest(
            frame, rebalance_threshold=0.0, transaction_cost_bps=0.0, rebalance_freq='daily',
        )
        self.assertIn('post_cost_return', result.columns)

    def test_post_cost_return_equals_equity_pct_change(self):
        """post_cost_return should match equity.pct_change().fillna(0)."""
        frame = self._base_frame()
        frame['RSP'] = [100.0, 110.0, 105.0, 115.0]
        result = run_strategy_backtest(
            frame, rebalance_threshold=0.0, transaction_cost_bps=5.0, rebalance_freq='daily',
        )
        expected = result['equity'].pct_change().fillna(0.0)
        pd.testing.assert_series_equal(
            result['post_cost_return'], expected, check_names=False,
        )

    def test_post_cost_return_lower_on_rebalance_day(self):
        """On a rebalance day with cost, post_cost_return < portfolio_return."""
        frame = self._base_frame()
        frame['RSP'] = [100.0, 110.0, 110.0, 110.0]
        frame.loc[1, 'r_trend'] = 0  # Force regime change -> rebalance

        result = run_strategy_backtest(
            frame, rebalance_threshold=0.0, transaction_cost_bps=100.0, rebalance_freq='daily',
        )
        rebal_rows = result[result['rebalanced'] == 1]
        for _, row in rebal_rows.iterrows():
            if row['transaction_cost'] > 0:
                self.assertLess(row['post_cost_return'], row['portfolio_return'])


class TestComputeMetrics(unittest.TestCase):
    """Tests for compute_metrics()."""

    def test_flat_equity_returns_zero_metrics(self):
        """Flat equity curve -> CAGR=0, vol~0, sharpe=0."""
        dates = pd.date_range('2024-01-02', periods=252, freq='B')
        df = pd.DataFrame({
            'Date': dates,
            'equity': [1.0] * 252,
            'post_cost_return': [0.0] * 252,
            'drawdown': [0.0] * 252,
            'rebalanced': [0] * 252,
            'transaction_cost': [0.0] * 252,
        })
        m = compute_metrics(df)
        self.assertAlmostEqual(m['cagr'], 0.0, places=6)
        self.assertAlmostEqual(m['annualized_vol'], 0.0, places=6)
        self.assertAlmostEqual(m['sharpe'], 0.0, places=6)

    def test_calmar_equals_cagr_over_max_dd(self):
        """calmar = cagr / abs(max_drawdown)."""
        dates = pd.date_range('2024-01-02', periods=504, freq='B')
        equity_vals = np.linspace(1.0, 1.2, 504)
        equity_vals[250:300] = np.linspace(1.1, 0.95, 50)  # drawdown
        equity_vals[300:] = np.linspace(0.95, 1.2, 204)

        equity = pd.Series(equity_vals)
        post_ret = equity.pct_change().fillna(0.0)

        peak = equity.cummax()
        dd = equity / peak - 1.0

        df = pd.DataFrame({
            'Date': dates,
            'equity': equity,
            'post_cost_return': post_ret,
            'drawdown': dd,
            'rebalanced': [0] * 504,
            'transaction_cost': [0.0] * 504,
        })
        m = compute_metrics(df)
        expected_calmar = m['cagr'] / abs(m['max_drawdown'])
        self.assertAlmostEqual(m['calmar'], expected_calmar, places=6)


    def test_sharpe_definitions_are_distinct(self):
        """Geometric sharpe (CAGR/vol) and arithmetic sharpe (mean*252/vol) are both present."""
        dates = pd.date_range('2024-01-02', periods=504, freq='B')
        np.random.seed(42)
        daily_ret = np.random.normal(0.0003, 0.01, 504)
        equity = np.cumprod(1.0 + daily_ret)

        post_ret = pd.Series(daily_ret)
        peak = pd.Series(equity).cummax()
        dd = pd.Series(equity) / peak - 1.0

        df = pd.DataFrame({
            'Date': dates,
            'equity': equity,
            'post_cost_return': post_ret,
            'drawdown': dd,
            'rebalanced': [0] * 504,
            'transaction_cost': [0.0] * 504,
        })
        m = compute_metrics(df)

        # Both keys must exist
        self.assertIn('sharpe', m)
        self.assertIn('sharpe_arithmetic', m)

        # Geometric sharpe = CAGR / annualized_vol
        n_years = (dates[-1] - dates[0]).days / 365.25
        expected_cagr = equity[-1] ** (1.0 / n_years) - 1.0
        expected_vol = post_ret.std() * np.sqrt(252)
        self.assertAlmostEqual(m['sharpe'], expected_cagr / expected_vol, places=6)

        # Arithmetic sharpe = mean(ret)*252 / annualized_vol
        expected_arith = (post_ret.mean() * 252.0) / expected_vol
        self.assertAlmostEqual(m['sharpe_arithmetic'], expected_arith, places=6)

        # They should differ (geometric < arithmetic for positive returns with vol)
        self.assertNotAlmostEqual(m['sharpe'], m['sharpe_arithmetic'], places=4)


if __name__ == '__main__':
    unittest.main()
