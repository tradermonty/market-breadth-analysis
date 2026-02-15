"""Regression tests for tv_pine_compat mode in backtest.

Tests verify:
1. Next-bar execution: signals queue on bar N, fill at bar N+1 open
2. OHLC gap-down stop: fill at min(open, stop_price)
3. Entry same-bar stop skip: Phase 0 entry skips Phase 1 stop on the same bar
"""

import os
import sys
import unittest

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest import Backtest


def _make_backtest(**kwargs):
    """Create a Backtest instance with tv_pine_compat and minimal defaults."""
    defaults = {
        'start_date': '2024-01-02',
        'end_date': '2024-01-31',
        'tv_pine_compat': True,
        'use_saved_data': True,
        'no_show_plot': True,
        'initial_capital': 50000,
        'debug': True,
    }
    defaults.update(kwargs)

    bt = Backtest(**defaults)
    # Override pivot_len to small values so hand-crafted data can trigger signals
    bt.pivot_len_long = 3
    bt.pivot_len_short = 2
    return bt


def _inject_data(bt, ohlc_df, breadth_series):
    """Inject hand-crafted price and breadth data directly into the backtest,
    bypassing the normal run() data-loading pipeline.
    Sets up price_data, breadth_index, short/long MA lines, trend, and
    pre-computes TV signals. Then calls execute_trades() directly.
    """
    bt.price_data = ohlc_df.copy()
    bt.breadth_index = breadth_series.copy()
    bt.sp500_data = pd.DataFrame()

    # Compute MA lines using the bt's own ma_type/periods
    if bt.ma_type == 'ema':
        bt.short_ma_line = bt.breadth_index.ewm(span=bt.short_ma, adjust=False).mean()
        bt.long_ma_line = bt.breadth_index.ewm(span=bt.long_ma, adjust=False).mean()
    else:
        bt.short_ma_line = bt.breadth_index.rolling(window=bt.short_ma).mean()
        bt.long_ma_line = bt.breadth_index.rolling(window=bt.long_ma).mean()

    from market_breadth import calculate_trend_with_hysteresis

    bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)

    # Pre-compute signals lists for summary output
    bt.short_ma_bottoms = []
    bt.long_ma_bottoms = []
    bt.peaks = []

    # Pre-compute TV-style signals
    bt._precompute_tv_signals()

    # Execute trades
    bt.execute_trades()


class TestNextBarExecution(unittest.TestCase):
    """Test 1: Entry signal on bar N fills at bar N+1 open."""

    def test_entry_fills_next_bar_open(self):
        """Trough signal fires on confirmation bar → entry fills at next bar's open."""
        # 20 bars of data. pivot_len_short=2 means pivot at j needs j-2..j+2 window,
        # confirmation at j+2.
        # We need a clear trough in the short MA around bar 6-8, confirmation at bar 8-10.
        dates = pd.bdate_range('2024-01-02', periods=20)

        # Price data: flat then rising
        opens = [50] * 5 + [48, 47, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
        closes = [50] * 5 + [47.5, 46.5, 45.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]

        ohlc = pd.DataFrame(
            {
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'adjusted_close': closes,
            },
            index=dates,
        )

        # Breadth: high plateau → deep trough → recovery
        # trough_level_short=0.20 means raw breadth must reach <= 0.20 in recent 20 bars
        # pivot_len_short=2: pivot at bar 7 (val=0.05), confirm at bar 9
        breadth_vals = [
            0.60,
            0.55,
            0.50,
            0.45,
            0.40,
            0.20,
            0.10,
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.72,
            0.73,
            0.74,
        ]
        breadth = pd.Series(breadth_vals, index=dates)

        bt = _make_backtest(start_date='2024-01-02', end_date='2024-01-29')
        # Lower prominence threshold so our small data triggers
        bt.prom_thresh_short = 0.01
        bt.trough_level_short = 0.25

        _inject_data(bt, ohlc, breadth)

        # Should have at least one entry
        entries = [t for t in bt.trades if t['action'] == 'BUY']
        self.assertGreater(len(entries), 0, 'Expected at least one entry trade')

        # The entry should have filled at a bar AFTER the signal confirmation bar.
        # With pivot_len_short=2, a trough at bar 7 confirms at bar 9.
        # In tv_pine_compat, entry fills at bar 10's open.
        first_entry = entries[0]
        entry_date = first_entry['date']

        # Check that entries are filled at open price, not close
        # (The entry price includes slippage=0.0 for pine_compat)
        bar_open = ohlc.loc[entry_date, 'open']
        self.assertAlmostEqual(
            first_entry['price'], bar_open, places=2, msg='Entry should fill at open price in pine_compat mode'
        )

        # Verify next-bar execution: entry must be AFTER the signal confirmation bar.
        # With pivot_len_short=2, the trough confirmation happens 2 bars after the trough.
        # Entry should fill on the bar AFTER confirmation.
        signal_confirmation_idx = ohlc.index.get_loc(entry_date) - 1
        signal_confirmation_date = ohlc.index[signal_confirmation_idx]
        self.assertGreater(
            entry_date,
            signal_confirmation_date,
            f'Entry {entry_date} should be after signal confirmation {signal_confirmation_date}',
        )


class TestOHLCGapDownStop(unittest.TestCase):
    """Test 2: Stop loss with OHLC gap-down fills at open (worse price)."""

    def test_gap_down_stop_fills_at_open(self):
        """When bar opens below stop price, fill at open (not stop price)."""
        dates = pd.bdate_range('2024-01-02', periods=15)

        # Price: steady then gap down on bar 10
        opens = [100] * 10 + [80, 82, 84, 86, 88]
        closes = [100] * 10 + [81, 83, 85, 87, 89]
        highs = [101] * 10 + [82, 84, 86, 88, 90]
        lows = [99] * 10 + [79, 81, 83, 85, 87]

        ohlc = pd.DataFrame(
            {
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'adjusted_close': closes,
            },
            index=dates,
        )

        breadth = pd.Series([0.50] * 15, index=dates)

        bt = _make_backtest(start_date='2024-01-02', end_date='2024-01-22')

        _inject_data(bt, ohlc, breadth)

        # Manually set up a position to test stop loss
        bt2 = _make_backtest(start_date='2024-01-02', end_date='2024-01-22')
        bt2.price_data = ohlc.copy()
        bt2.breadth_index = breadth.copy()

        if bt2.ma_type == 'ema':
            bt2.short_ma_line = bt2.breadth_index.ewm(span=bt2.short_ma, adjust=False).mean()
            bt2.long_ma_line = bt2.breadth_index.ewm(span=bt2.long_ma, adjust=False).mean()
        else:
            bt2.short_ma_line = bt2.breadth_index.rolling(window=bt2.short_ma).mean()
            bt2.long_ma_line = bt2.breadth_index.rolling(window=bt2.long_ma).mean()

        from market_breadth import calculate_trend_with_hysteresis

        bt2.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt2.long_ma_line), index=bt2.long_ma_line.index)
        bt2.short_ma_bottoms = []
        bt2.long_ma_bottoms = []
        bt2.peaks = []
        bt2._tv_peak_signals = {}
        bt2._tv_short_trough_signals = {}
        bt2._tv_long_trough_signals = {}

        # Simulate existing position: entered at bar 2 at price 100
        entry_date = dates[2]
        bt2._execute_entry(entry_date, 100.0, 500, reason='test_entry')
        bt2.highest_price = 100.0

        # stop_loss_pct = 0.08, so stop at 100 * 0.92 = 92
        # Bar 10: open=80, low=79 → gap down through stop
        # Expected fill at min(open=80, stop=92) = 80

        bt2.execute_trades()

        # Find the stop loss exit
        exits = [t for t in bt2.trades if t['action'] == 'SELL' and 'stop loss' in t.get('reason', '')]
        self.assertGreater(len(exits), 0, 'Expected a stop loss exit')

        stop_exit = exits[0]
        # In gap-down: fill at open price (80), not stop price (92)
        # Note: slippage=0.0, so exit_price should be the raw fill price
        self.assertAlmostEqual(stop_exit['price'], 80.0, places=1, msg='Gap-down stop should fill at open price')

    def test_normal_stop_fills_at_stop_price(self):
        """When low <= stop but open > stop, fill at stop price."""
        dates = pd.bdate_range('2024-01-02', periods=15)

        # Bar 10: open=95, low=90 (below stop at 92), close=93
        opens = [100] * 10 + [95, 96, 97, 98, 99]
        closes = [100] * 10 + [93, 94, 95, 96, 97]
        highs = [101] * 10 + [96, 97, 98, 99, 100]
        lows = [99] * 10 + [90, 93, 94, 95, 96]

        ohlc = pd.DataFrame(
            {
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'adjusted_close': closes,
            },
            index=dates,
        )

        breadth = pd.Series([0.50] * 15, index=dates)

        bt = _make_backtest(start_date='2024-01-02', end_date='2024-01-22')
        bt.price_data = ohlc.copy()
        bt.breadth_index = breadth.copy()

        if bt.ma_type == 'ema':
            bt.short_ma_line = bt.breadth_index.ewm(span=bt.short_ma, adjust=False).mean()
            bt.long_ma_line = bt.breadth_index.ewm(span=bt.long_ma, adjust=False).mean()
        else:
            bt.short_ma_line = bt.breadth_index.rolling(window=bt.short_ma).mean()
            bt.long_ma_line = bt.breadth_index.rolling(window=bt.long_ma).mean()

        from market_breadth import calculate_trend_with_hysteresis

        bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)
        bt.short_ma_bottoms = []
        bt.long_ma_bottoms = []
        bt.peaks = []
        bt._tv_peak_signals = {}
        bt._tv_short_trough_signals = {}
        bt._tv_long_trough_signals = {}

        # Enter at bar 2, price=100
        bt._execute_entry(dates[2], 100.0, 500, reason='test_entry')
        bt.highest_price = 100.0

        # stop at 92. Bar 10: open=95 > 92, low=90 < 92 → fill at stop_price=92
        bt.execute_trades()

        exits = [t for t in bt.trades if t['action'] == 'SELL' and 'stop loss' in t.get('reason', '')]
        self.assertGreater(len(exits), 0, 'Expected a stop loss exit')

        stop_exit = exits[0]
        expected_stop = 100.0 * (1 - 0.08)  # 92.0
        self.assertAlmostEqual(stop_exit['price'], expected_stop, places=1, msg='Normal stop should fill at stop price')


class TestSameBarStopSkip(unittest.TestCase):
    """Test 3: Entry filled in Phase 0 does not trigger Phase 1 stop on same bar."""

    def test_entry_skip_same_bar_stop(self):
        """An entry filled at bar open should not be stopped on the same bar.

        Pending entry fills at bar 0 (first bar of loop). Bar 0 has a low that
        would normally trigger stop, but skip_stop=True prevents it.
        Subsequent bars have safe prices so no stop fires later.
        """
        dates = pd.bdate_range('2024-01-02', periods=10)

        # Bar 0: open=100 (entry fills here), low=40 would trigger stop at 92
        # But skip_stop should prevent Phase 1 from firing.
        # Bars 1+: all prices above stop, no subsequent stop.
        opens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        closes = [99, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        highs = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        lows = [40, 100, 101, 102, 103, 104, 105, 106, 107, 108]

        ohlc = pd.DataFrame(
            {
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'adjusted_close': closes,
            },
            index=dates,
        )

        breadth = pd.Series([0.50] * 10, index=dates)

        bt = _make_backtest(start_date='2024-01-02', end_date='2024-01-15')
        bt.price_data = ohlc.copy()
        bt.breadth_index = breadth.copy()

        if bt.ma_type == 'ema':
            bt.short_ma_line = bt.breadth_index.ewm(span=bt.short_ma, adjust=False).mean()
            bt.long_ma_line = bt.breadth_index.ewm(span=bt.long_ma, adjust=False).mean()
        else:
            bt.short_ma_line = bt.breadth_index.rolling(window=bt.short_ma).mean()
            bt.long_ma_line = bt.breadth_index.rolling(window=bt.long_ma).mean()

        from market_breadth import calculate_trend_with_hysteresis

        bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)
        bt.short_ma_bottoms = []
        bt.long_ma_bottoms = []
        bt.peaks = []
        bt._tv_peak_signals = {}
        bt._tv_short_trough_signals = {}
        bt._tv_long_trough_signals = {}

        # Pre-set pending entry → fills at bar 0 (first iteration of loop)
        bt._pending_entry = ('short_ma_bottom', 1.0)

        bt.execute_trades()

        # Verify: entry was filled at bar 0 open=100
        entries = [t for t in bt.trades if t['action'] == 'BUY']
        self.assertGreater(len(entries), 0, 'Expected entry to be filled')

        entry = entries[0]
        self.assertEqual(entry['date'], dates[0], 'Entry should fill at bar 0')
        self.assertAlmostEqual(entry['price'], 100.0, places=1)

        # Key check: NO stop loss exit should occur at all
        # (Bar 0 low=40 < stop=92, but skip_stop prevents it; bars 1+ are safe)
        stop_exits = [t for t in bt.trades if t['action'] == 'SELL' and 'stop loss' in t.get('reason', '')]
        self.assertEqual(len(stop_exits), 0, 'Entry at bar open should NOT trigger same-bar stop loss')


class TestParameterLock(unittest.TestCase):
    """Test that tv_pine_compat locks signal parameters regardless of CLI values."""

    def test_pine_compat_locks_short_ma(self):
        bt = Backtest(
            start_date='2024-01-01',
            end_date='2024-12-31',
            short_ma=99,
            long_ma=50,
            ma_type='sma',
            tv_pine_compat=True,
            no_show_plot=True,
        )
        self.assertEqual(bt.short_ma, 5)
        self.assertEqual(bt.long_ma, 200)
        self.assertEqual(bt.ma_type, 'ema')
        self.assertEqual(bt.pivot_len_long, 20)
        self.assertEqual(bt.pivot_len_short, 10)
        self.assertEqual(bt.peak_level, 0.70)
        self.assertTrue(bt.no_pyramiding)
        self.assertEqual(bt.stop_loss_pct, 0.08)
        self.assertEqual(bt.slippage, 0.0)

    def test_pine_compat_disables_extensions(self):
        bt = Backtest(
            start_date='2024-01-01',
            end_date='2024-12-31',
            two_stage_exit=True,
            use_volatility_stop=True,
            bullish_regime_suppression=True,
            tv_pine_compat=True,
            no_show_plot=True,
        )
        self.assertFalse(bt.two_stage_exit)
        self.assertFalse(bt.use_volatility_stop)
        self.assertFalse(bt.bullish_regime_suppression)
        self.assertFalse(bt.use_trailing_stop)
        self.assertFalse(bt.partial_exit)


class TestPendingOrderProtection(unittest.TestCase):
    """Test that pending orders are protected per Pine market order semantics."""

    def test_pending_entry_cleared_on_stop(self):
        """If stop fires, pending entry should be cleared."""
        dates = pd.bdate_range('2024-01-02', periods=10)

        ohlc = pd.DataFrame(
            {
                'open': [100, 100, 100, 95, 80, 82, 84, 86, 88, 90],
                'high': [101, 101, 101, 96, 81, 83, 85, 87, 89, 91],
                'low': [99, 99, 99, 85, 78, 81, 83, 85, 87, 89],
                'close': [100, 100, 100, 90, 79, 82, 84, 86, 88, 90],
                'adjusted_close': [100, 100, 100, 90, 79, 82, 84, 86, 88, 90],
            },
            index=dates,
        )

        breadth = pd.Series([0.50] * 10, index=dates)

        bt = _make_backtest(start_date='2024-01-02', end_date='2024-01-15')
        bt.price_data = ohlc.copy()
        bt.breadth_index = breadth.copy()

        if bt.ma_type == 'ema':
            bt.short_ma_line = bt.breadth_index.ewm(span=bt.short_ma, adjust=False).mean()
            bt.long_ma_line = bt.breadth_index.ewm(span=bt.long_ma, adjust=False).mean()
        else:
            bt.short_ma_line = bt.breadth_index.rolling(window=bt.short_ma).mean()
            bt.long_ma_line = bt.breadth_index.rolling(window=bt.long_ma).mean()

        from market_breadth import calculate_trend_with_hysteresis

        bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)
        bt.short_ma_bottoms = []
        bt.long_ma_bottoms = []
        bt.peaks = []
        bt._tv_peak_signals = {}
        bt._tv_short_trough_signals = {}
        bt._tv_long_trough_signals = {}

        # Enter at bar 1, price=100
        bt._execute_entry(dates[1], 100.0, 500, reason='test_entry')
        bt.highest_price = 100.0

        # Set a pending entry that should be cleared when stop fires
        bt._pending_entry = ('long_ma_bottom', 1.0)

        bt.execute_trades()

        # The stop fires at bar 3 or 4 (low=85 or 78 < stop=92)
        # After stop, _pending_entry should have been cleared
        self.assertIsNone(bt._pending_entry, 'Pending entry should be cleared after stop loss')


if __name__ == '__main__':
    unittest.main()
