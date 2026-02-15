import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from decomposition_2x2 import _run_combo
from sensitivity_sweep import _run_single, _build_grid, _run_sweep


class TestRunSingle(unittest.TestCase):
    """Tests for sensitivity_sweep._run_single()."""

    def _make_merged(self, n=60):
        dates = pd.bdate_range('2024-01-02', periods=n)
        return pd.DataFrame({
            'Date': dates,
            'SPY': [100.0] * n,
            'RSP': [100.0 + 0.1 * i for i in range(n)],
            'XLE': [100.0] * n,
            'SGOV': [100.0] * n,
            'Breadth_Index_Raw': [0.55] * n,
            'Breadth_Index_8MA': [0.56] * n,
            'Breadth_Index_200MA': [0.50] * n,
        })

    def test_run_single_returns_metrics_dict(self):
        """_run_single returns a dict with all expected metric keys."""
        merged = self._make_merged()
        result = _run_single(
            merged,
            rebalance_threshold=0.05,
            overheat_entry=0.75,
            overheat_exit=0.70,
            ratio_ma=10,
            transaction_cost_bps=5.0,
        )
        self.assertIsInstance(result, dict)
        for key in ('cagr', 'sharpe', 'max_drawdown', 'calmar', 'final_equity',
                     'rebalance_threshold', 'overheat_entry', 'ratio_ma'):
            self.assertIn(key, result, f'Missing key: {key}')

    def test_grid_two_combos_produces_two_rows(self):
        """Running sweep with 2 parameter combos produces DataFrame with 2 rows."""
        merged = self._make_merged()
        grid = [
            {'rebalance_threshold': 0.05, 'overheat_entry': 0.75,
             'overheat_exit': 0.70, 'ratio_ma': 10, 'transaction_cost_bps': 5.0},
            {'rebalance_threshold': 0.03, 'overheat_entry': 0.80,
             'overheat_exit': 0.75, 'ratio_ma': 5, 'transaction_cost_bps': 10.0},
        ]
        df = _run_sweep(merged, grid)
        self.assertEqual(len(df), 2)
        self.assertIn('sharpe', df.columns)


class TestGridSize(unittest.TestCase):
    def test_full_grid_size(self):
        """Full grid should have 5 * 3 * 3 * 4 = 180 combinations."""
        grid = _build_grid()
        self.assertEqual(len(grid), 180)


if __name__ == '__main__':
    unittest.main()
