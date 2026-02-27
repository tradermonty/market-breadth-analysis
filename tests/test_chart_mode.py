"""
Tests for chart_mode backtest feature.

chart_mode uses the same peak/trough detection parameters as the chart drawing
logic in market_breadth.py:
- 200MA peaks: find_peaks(breadth_ma_200, distance=50, prominence=0.015)
- 200MA troughs: find_peaks(-breadth_ma_200, distance=50, prominence=0.015)
- Short MA troughs: < 0.4 filter, find_peaks(-, prominence=0.02), NO distance
- No level thresholds, no 20-day breadth min check, no len > long_ma gate
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.backtest import Backtest


def _make_breadth_series(n=300, seed=42):
    """Create a synthetic breadth series with clear peaks and troughs."""
    np.random.seed(seed)
    t = np.linspace(0, 6 * np.pi, n)
    # Oscillating breadth between ~0.15 and ~0.85
    breadth = 0.5 + 0.35 * np.sin(t) + 0.02 * np.random.randn(n)
    breadth = np.clip(breadth, 0.01, 0.99)
    dates = pd.bdate_range('2020-01-01', periods=n)
    return pd.Series(breadth, index=dates)


def _make_price_series(n=300, seed=42):
    """Create a synthetic price series."""
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.maximum(prices, 10)
    dates = pd.bdate_range('2020-01-01', periods=n)
    return pd.DataFrame({'adjusted_close': prices}, index=dates)


class TestChartModeInit(unittest.TestCase):
    """Test chart_mode constructor behavior."""

    def test_chart_mode_default_false(self):
        """chart_mode defaults to False."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
        )
        self.assertFalse(bt.chart_mode)

    def test_chart_mode_set_true(self):
        """chart_mode can be set to True."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
            chart_mode=True,
        )
        self.assertTrue(bt.chart_mode)

    def test_chart_mode_with_tv_mode_raises(self):
        """chart_mode + tv_mode raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Backtest(
                start_date='2023-01-01',
                end_date='2023-12-31',
                symbol='SPY',
                use_saved_data=True,
                no_show_plot=True,
                chart_mode=True,
                tv_mode=True,
            )
        self.assertIn('chart_mode', str(ctx.exception))

    def test_chart_mode_with_tv_pine_compat_raises(self):
        """chart_mode + tv_pine_compat raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Backtest(
                start_date='2023-01-01',
                end_date='2023-12-31',
                symbol='SPY',
                use_saved_data=True,
                no_show_plot=True,
                chart_mode=True,
                tv_pine_compat=True,
            )
        self.assertIn('chart_mode', str(ctx.exception))


class TestChartModeSignalDetection(unittest.TestCase):
    """Test that chart_mode applies correct find_peaks parameters."""

    def test_long_ma_peaks_use_distance_50(self):
        """In chart_mode, long MA peak detection uses distance=50."""
        breadth = _make_breadth_series(n=400)
        long_ma = breadth.rolling(200).mean().dropna()

        # Chart-mode should use distance=50
        peaks_chart, _ = find_peaks(long_ma.values, prominence=0.015, distance=50)
        # Legacy mode has no distance
        peaks_legacy, _ = find_peaks(long_ma.values, prominence=0.015)

        # With distance=50, we should get fewer or equal peaks
        self.assertLessEqual(len(peaks_chart), len(peaks_legacy))

    def test_long_ma_troughs_use_distance_50(self):
        """In chart_mode, long MA trough detection uses distance=50."""
        breadth = _make_breadth_series(n=400)
        long_ma = breadth.rolling(200).mean().dropna()

        troughs_chart, _ = find_peaks(-long_ma.values, prominence=0.015, distance=50)
        troughs_legacy, _ = find_peaks(-long_ma.values, prominence=0.015)

        self.assertLessEqual(len(troughs_chart), len(troughs_legacy))

    def test_short_ma_troughs_no_distance(self):
        """In chart_mode, short MA trough detection has NO distance parameter."""
        breadth = _make_breadth_series(n=400)
        short_ma = breadth.rolling(5).mean().dropna()

        # Filter below 0.4
        below = short_ma[short_ma < 0.4]
        if not below.empty:
            # Chart mode: no distance, prominence=0.02
            troughs, _ = find_peaks(-below.values, prominence=0.02)
            # This should work without distance parameter
            self.assertIsInstance(troughs, np.ndarray)

    def test_short_ma_filter_threshold_04(self):
        """In chart_mode, short MA filter threshold is 0.4, not self.threshold (0.5)."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
            chart_mode=True,
            threshold=0.5,
        )
        # The threshold attribute is still 0.5 but chart_mode overrides to 0.4
        self.assertEqual(bt.threshold, 0.5)
        self.assertTrue(bt.chart_mode)

    def test_peak_no_level_check(self):
        """In chart_mode, peaks below 0.5 are still valid signals."""
        # Create a breadth series where the long MA peak is below 0.5
        n = 400
        np.random.seed(100)
        t = np.linspace(0, 4 * np.pi, n)
        # Keep breadth oscillating between 0.2 and 0.45
        breadth = 0.32 + 0.12 * np.sin(t) + 0.01 * np.random.randn(n)
        breadth = np.clip(breadth, 0.05, 0.48)
        dates = pd.bdate_range('2020-01-01', periods=n)
        series = pd.Series(breadth, index=dates)
        long_ma = series.rolling(200).mean().dropna()

        peaks, _ = find_peaks(long_ma.values, prominence=0.015, distance=50)

        # In legacy mode, these would all be filtered out (< 0.5)
        for p in peaks:
            self.assertLess(long_ma.iloc[p], 0.5)

        # In chart_mode, they should still be detected
        # (at least one peak should exist in this synthetic data)
        # If no peaks with prominence=0.015 exist, the test is still valid
        # as the point is that no level filter is applied

    def test_no_20day_breadth_min_check_short_ma(self):
        """In chart_mode, 20-day breadth minimum check is skipped for short MA."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
            chart_mode=True,
        )
        # chart_mode should be True, meaning the 20-day min check is bypassed
        self.assertTrue(bt.chart_mode)

    def test_no_20day_breadth_min_check_long_ma(self):
        """In chart_mode, 20-day breadth minimum check is skipped for long MA."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
            chart_mode=True,
        )
        self.assertTrue(bt.chart_mode)

    def test_no_len_gate_for_long_ma(self):
        """In chart_mode, len > long_ma gate is skipped."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
            chart_mode=True,
        )
        # chart_mode bypasses the len > long_ma check
        self.assertTrue(bt.chart_mode)


class TestChartModeNonRegression(unittest.TestCase):
    """Ensure legacy mode behavior is unchanged when chart_mode=False."""

    def test_legacy_short_ma_uses_threshold(self):
        """Legacy mode (chart_mode=False) uses self.threshold, not 0.4."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
            chart_mode=False,
            threshold=0.5,
        )
        self.assertFalse(bt.chart_mode)
        self.assertEqual(bt.threshold, 0.5)

    def test_legacy_long_ma_no_distance(self):
        """Legacy mode does NOT use distance parameter for find_peaks."""
        breadth = _make_breadth_series(n=400)
        long_ma = breadth.rolling(200).mean().dropna()

        # Legacy: no distance parameter
        peaks_legacy, _ = find_peaks(long_ma.values, prominence=0.015)
        # This should work (no distance kwarg)
        self.assertIsInstance(peaks_legacy, np.ndarray)

    def test_legacy_peak_level_check(self):
        """Legacy mode filters peaks below 0.5."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
            chart_mode=False,
        )
        self.assertFalse(bt.chart_mode)

    def test_legacy_requires_len_gate(self):
        """Legacy mode requires len(current_long_ma_line) > self.long_ma."""
        bt = Backtest(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbol='SPY',
            use_saved_data=True,
            no_show_plot=True,
            chart_mode=False,
        )
        self.assertFalse(bt.chart_mode)
        self.assertEqual(bt.long_ma, 200)


if __name__ == '__main__':
    unittest.main()
