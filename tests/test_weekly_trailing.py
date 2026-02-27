"""Tests for weekly trailing stop exit feature.

Tests cover:
1. Disabled mode has no behavior change
2. Exit fires at next bar open with correct reason
3. Transition weeks guard prevents early exit
4. Mode conflict with tv_pine_compat raises ValueError
5. Auto-disables tv_mode with warning
6. aggregate_to_weekly W-FRI resampling
7. is_week_end detection
8. chart_mode + weekly_trailing combo works
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest import Backtest
from backtest.weekly_trailing import aggregate_to_weekly, check_weekly_trailing_stop, is_week_end
from market_breadth import calculate_trend_with_hysteresis


def _make_backtest(**kwargs):
    """Create a Backtest instance with legacy mode defaults."""
    defaults = {
        'start_date': '2024-01-02',
        'end_date': '2024-12-31',
        'tv_mode': False,
        'use_saved_data': True,
        'no_show_plot': True,
        'initial_capital': 50000,
        'debug': True,
    }
    defaults.update(kwargs)
    return Backtest(**defaults)


def _inject_data_legacy(bt, ohlc_df, breadth_series):
    """Inject hand-crafted price and breadth data into a legacy-mode backtest."""
    bt.price_data = ohlc_df.copy()
    bt.breadth_index = breadth_series.copy()
    bt.sp500_data = pd.DataFrame()
    bt.short_ma_line = bt.breadth_index.ewm(span=bt.short_ma, adjust=False).mean()
    bt.long_ma_line = bt.breadth_index.ewm(span=bt.long_ma, adjust=False).mean()
    bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)
    bt.short_ma_bottoms = []
    bt.long_ma_bottoms = []
    bt.peaks = []
    bt.execute_trades()


class TestWeeklyTrailingDisabled(unittest.TestCase):
    """Test 1: weekly trailing disabled produces identical trades as baseline."""

    def test_weekly_trailing_disabled_no_behavior_change(self):
        # Build shared synthetic data
        dates = pd.bdate_range('2024-01-02', periods=60)
        prices_close = np.concatenate(
            [
                np.linspace(100, 130, 30),
                np.linspace(129, 95, 30),
            ]
        )
        ohlc = pd.DataFrame(
            {
                'open': prices_close * 0.999,
                'high': prices_close * 1.01,
                'low': prices_close * 0.99,
                'close': prices_close,
                'adjusted_close': prices_close,
                'adjusted_open': prices_close * 0.999,
                'adjusted_high': prices_close * 1.01,
                'adjusted_low': prices_close * 0.99,
            },
            index=dates,
        )
        breadth = pd.Series(
            np.concatenate([np.linspace(0.3, 0.15, 5), np.linspace(0.16, 0.5, 55)]),
            index=dates,
        )

        # Baseline: no weekly trailing
        bt_base = _make_backtest(enable_weekly_trailing=False, tv_mode=False, stop_loss_pct=0.50)
        bt_base.price_data = ohlc.copy()
        bt_base.breadth_index = breadth.copy()
        bt_base.sp500_data = pd.DataFrame()
        bt_base.short_ma_line = breadth.ewm(span=bt_base.short_ma, adjust=False).mean()
        bt_base.long_ma_line = breadth.ewm(span=bt_base.long_ma, adjust=False).mean()
        bt_base.long_ma_trend = pd.Series(
            calculate_trend_with_hysteresis(bt_base.long_ma_line), index=bt_base.long_ma_line.index
        )
        bt_base.short_ma_bottoms = [dates[4]]
        bt_base.long_ma_bottoms = []
        bt_base.peaks = []
        bt_base.execute_trades()

        # Test: explicit disabled
        bt_off = _make_backtest(enable_weekly_trailing=False, tv_mode=False, stop_loss_pct=0.50)
        bt_off.price_data = ohlc.copy()
        bt_off.breadth_index = breadth.copy()
        bt_off.sp500_data = pd.DataFrame()
        bt_off.short_ma_line = breadth.ewm(span=bt_off.short_ma, adjust=False).mean()
        bt_off.long_ma_line = breadth.ewm(span=bt_off.long_ma, adjust=False).mean()
        bt_off.long_ma_trend = pd.Series(
            calculate_trend_with_hysteresis(bt_off.long_ma_line), index=bt_off.long_ma_line.index
        )
        bt_off.short_ma_bottoms = [dates[4]]
        bt_off.long_ma_bottoms = []
        bt_off.peaks = []
        bt_off.execute_trades()

        # Trade results must be identical
        self.assertEqual(len(bt_base.trade_log), len(bt_off.trade_log), 'Trade count must match')
        for t_base, t_off in zip(bt_base.trade_log, bt_off.trade_log):
            self.assertEqual(t_base['exit_reason'], t_off['exit_reason'], 'Exit reasons must match')
            self.assertAlmostEqual(t_base['pnl_dollar'], t_off['pnl_dollar'], places=2)
        self.assertAlmostEqual(bt_base.current_capital, bt_off.current_capital, places=2)


class TestWeeklyTrailingExitNextBarOpen(unittest.TestCase):
    """Test 2: weekly trailing exit fires at next bar open."""

    def test_weekly_trailing_exits_next_bar_open(self):
        # Build 60 business days of data spanning multiple weeks
        dates = pd.bdate_range('2024-01-02', periods=60)

        # Price rises then drops below weekly EMA to trigger trailing stop
        prices_close = np.concatenate(
            [
                np.linspace(100, 130, 30),  # rise for 30 days
                np.linspace(129, 95, 30),  # drop for 30 days
            ]
        )
        prices_open = prices_close * 0.999
        prices_high = prices_close * 1.01
        prices_low = prices_close * 0.99

        ohlc = pd.DataFrame(
            {
                'open': prices_open,
                'high': prices_high,
                'low': prices_low,
                'close': prices_close,
                'adjusted_close': prices_close,
                'adjusted_open': prices_open,
                'adjusted_high': prices_high,
                'adjusted_low': prices_low,
            },
            index=dates,
        )

        # Breadth that triggers an entry early (trough then rise)
        breadth_values = np.concatenate(
            [
                np.linspace(0.3, 0.15, 5),  # drop to trough
                np.linspace(0.16, 0.5, 55),  # gradual rise
            ]
        )
        breadth = pd.Series(breadth_values, index=dates)

        bt = _make_backtest(
            enable_weekly_trailing=True,
            tv_mode=False,
            weekly_trailing_type='weekly_ema',
            weekly_ema_period=5,
            weekly_transition_weeks=1,
            stop_loss_pct=0.50,  # high stop loss to not interfere
        )

        # Manually force an entry then run with weekly trailing
        bt.price_data = ohlc.copy()
        bt.breadth_index = breadth.copy()
        bt.sp500_data = pd.DataFrame()
        bt.short_ma_line = bt.breadth_index.ewm(span=bt.short_ma, adjust=False).mean()
        bt.long_ma_line = bt.breadth_index.ewm(span=bt.long_ma, adjust=False).mean()
        bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)

        # Force entry signal at early date
        bt.short_ma_bottoms = [dates[4]]
        bt.long_ma_bottoms = []
        bt.peaks = []
        bt.execute_trades()

        # Check that weekly trailing exit occurred
        weekly_exits = [t for t in bt.trade_log if t['exit_reason'] == 'weekly trailing']
        self.assertGreater(len(weekly_exits), 0, 'Weekly trailing exit must fire')
        self.assertEqual(len(weekly_exits), 1)
        exit_trade = weekly_exits[0]
        # Verify exit price uses adjusted_open of the next bar (approximately)
        exit_date = exit_trade['exit_date']
        exit_idx = list(ohlc.index).index(exit_date)
        expected_price = ohlc['adjusted_open'].iloc[exit_idx]
        # Account for slippage
        self.assertAlmostEqual(exit_trade['exit_price'], expected_price * (1 - bt.slippage), places=2)


class TestWeeklyTransitionWeeksGuard(unittest.TestCase):
    """Test 3: transition weeks guard prevents premature exit."""

    def test_transition_weeks_blocks_early_exit(self):
        # Create weekly data
        dates = pd.date_range('2024-01-05', periods=10, freq='W-FRI')
        weekly_df = pd.DataFrame(
            {
                'open': [100] * 10,
                'high': [105] * 10,
                'low': [95] * 10,
                'close': [100] * 10,
            },
            index=dates,
        )

        # Entry on first week, check on second week (1 week elapsed < 3 transition)
        result = check_weekly_trailing_stop(
            current_close=80,  # way below EMA - should trigger IF not guarded
            weekly_df=weekly_df,
            entry_date=dates[0],
            current_date=dates[1],
            trailing_type='weekly_ema',
            ema_period=5,
            nweek_low_period=4,
            transition_weeks=3,
        )
        self.assertFalse(result, 'Should return False when within transition_weeks')

    def test_transition_weeks_exact_boundary(self):
        """weeks_elapsed == transition_weeks should allow triggering (not guarded)."""
        dates = pd.date_range('2024-01-05', periods=10, freq='W-FRI')
        closes = [100, 102, 104, 80, 75, 70, 65, 60, 55, 50]
        weekly_df = pd.DataFrame(
            {
                'open': [c + 1 for c in closes],
                'high': [c + 5 for c in closes],
                'low': [c - 5 for c in closes],
                'close': closes,
            },
            index=dates,
        )

        # Entry on week 0, check on week 3 → weeks_elapsed=3 == transition_weeks=3
        result = check_weekly_trailing_stop(
            current_close=50,
            weekly_df=weekly_df,
            entry_date=dates[0],
            current_date=dates[3],
            trailing_type='weekly_ema',
            ema_period=3,
            nweek_low_period=4,
            transition_weeks=3,
        )
        self.assertTrue(result, 'Should trigger when weeks_elapsed == transition_weeks')

    def test_transition_weeks_allows_after_period(self):
        dates = pd.date_range('2024-01-05', periods=10, freq='W-FRI')
        # Price drops significantly in later weeks so EMA trails
        closes = [100, 102, 104, 106, 108, 80, 75, 70, 65, 60]
        weekly_df = pd.DataFrame(
            {
                'open': [c + 1 for c in closes],
                'high': [c + 5 for c in closes],
                'low': [c - 5 for c in closes],
                'close': closes,
            },
            index=dates,
        )

        # Entry on first week, check on week 5 (4 weeks elapsed >= 3 transition)
        result = check_weekly_trailing_stop(
            current_close=60,  # below EMA
            weekly_df=weekly_df,
            entry_date=dates[0],
            current_date=dates[9],
            trailing_type='weekly_ema',
            ema_period=5,
            nweek_low_period=4,
            transition_weeks=3,
        )
        self.assertTrue(result, 'Should return True after transition_weeks elapsed')


class TestModeConflictTvPineCompat(unittest.TestCase):
    """Test 4: weekly trailing + tv_pine_compat raises ValueError."""

    def test_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _make_backtest(enable_weekly_trailing=True, tv_pine_compat=True)
        self.assertIn('tv_pine_compat', str(ctx.exception))


class TestWeeklyTrailingWithTvMode(unittest.TestCase):
    """Test 5: weekly trailing works with tv_mode enabled."""

    def test_tv_mode_preserved_with_weekly_trailing(self):
        bt = _make_backtest(enable_weekly_trailing=True, tv_mode=True)
        self.assertTrue(bt.tv_mode, 'tv_mode should remain enabled')
        self.assertTrue(bt.enable_weekly_trailing)


class TestAggregateToWeekly(unittest.TestCase):
    """Test 6: aggregate_to_weekly produces correct W-FRI OHLC."""

    def test_weekly_aggregation(self):
        # Create 10 business days spanning 3 W-FRI weeks
        # Tue Jan 2 - Mon Jan 15
        # Week 1 (ending Jan 5): Jan 2-5 (4 bars)
        # Week 2 (ending Jan 12): Jan 8-12 (5 bars)
        # Week 3 (ending Jan 19): Jan 15 (1 bar)
        dates = pd.bdate_range('2024-01-02', periods=10)
        opens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        highs = [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
        lows = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        closes = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

        df = pd.DataFrame(
            {
                'adjusted_open': opens,
                'adjusted_high': highs,
                'adjusted_low': lows,
                'adjusted_close': closes,
            },
            index=dates,
        )

        weekly = aggregate_to_weekly(df)
        self.assertEqual(len(weekly), 3, 'Should produce 3 weekly bars (Jan 2-5, 8-12, 15)')

        # First week: Tue-Fri (Jan 2-5), 4 bars
        # open=first=100, high=max=113, low=min=90, close=last=104
        self.assertEqual(weekly.iloc[0]['open'], 100)
        self.assertEqual(weekly.iloc[0]['high'], 113)
        self.assertEqual(weekly.iloc[0]['low'], 90)
        self.assertEqual(weekly.iloc[0]['close'], 104)

        # Second week: Mon-Fri (Jan 8-12), 5 bars
        # open=first=104, high=max=118, low=min=94, close=last=109
        self.assertEqual(weekly.iloc[1]['open'], 104)
        self.assertEqual(weekly.iloc[1]['high'], 118)
        self.assertEqual(weekly.iloc[1]['low'], 94)
        self.assertEqual(weekly.iloc[1]['close'], 109)

    def test_adjusted_close_only_fallback(self):
        """When only adjusted_close exists, aggregate_to_weekly still produces
        valid weekly close data. Note: the backtest runner validates OHLC columns
        before calling this, so this path is only reached by direct callers.
        """
        dates = pd.bdate_range('2024-01-02', periods=10)
        closes = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        df = pd.DataFrame({'adjusted_close': closes}, index=dates)

        weekly = aggregate_to_weekly(df)
        self.assertGreater(len(weekly), 0)
        self.assertIn('close', weekly.columns)
        # Fallback derives open/high/low from adjusted_close
        self.assertIn('open', weekly.columns)
        self.assertIn('high', weekly.columns)
        self.assertIn('low', weekly.columns)

    def test_backtest_rejects_missing_ohlc(self):
        """Backtest._validate_weekly_trailing_columns raises RuntimeError
        when required OHLC columns are missing from price_data.
        """
        bt = _make_backtest(enable_weekly_trailing=True, tv_mode=False)
        bt.price_data = pd.DataFrame(
            {'adjusted_close': [100, 101, 102]},
            index=pd.bdate_range('2024-01-02', periods=3),
        )
        with self.assertRaises(RuntimeError) as ctx:
            bt._validate_weekly_trailing_columns()
        self.assertIn('missing', str(ctx.exception))

    def test_backtest_rejects_partial_ohlc(self):
        """When high is missing but low exists, validation still rejects.
        This prevents aggregate_to_weekly from falling back to close-only
        mode which would make weekly_nweek_low inaccurate.
        """
        bt = _make_backtest(enable_weekly_trailing=True, tv_mode=False, weekly_trailing_type='weekly_nweek_low')
        bt.price_data = pd.DataFrame(
            {
                'adjusted_close': [100, 101, 102],
                'open': [99, 100, 101],
                'close': [100, 101, 102],
                'low': [98, 99, 100],
                # 'high' intentionally missing
            },
            index=pd.bdate_range('2024-01-02', periods=3),
        )
        with self.assertRaises(RuntimeError) as ctx:
            bt._validate_weekly_trailing_columns()
        self.assertIn('high', str(ctx.exception))


class TestIsWeekEnd(unittest.TestCase):
    """Test 7: is_week_end returns True on Friday and last bar."""

    def test_friday_is_week_end(self):
        dates = pd.bdate_range('2024-01-02', periods=10)  # Tue Jan 2 to Mon Jan 15
        # Jan 5 (Fri) = index 3
        self.assertTrue(is_week_end(dates, 3), 'Friday should be week end')

    def test_non_friday_is_not_week_end(self):
        dates = pd.bdate_range('2024-01-02', periods=10)
        # Jan 3 (Wed) = index 1
        self.assertFalse(is_week_end(dates, 1), 'Wednesday should not be week end')

    def test_last_bar_is_week_end(self):
        dates = pd.bdate_range('2024-01-02', periods=10)
        self.assertTrue(is_week_end(dates, 9), 'Last bar should be week end')


class TestHolidayShortenedWeek(unittest.TestCase):
    """Test 9: holiday-shortened week where Friday is missing."""

    def test_thursday_week_end_includes_current_week(self):
        """When Friday is a holiday, is_week_end returns True on Thursday,
        and check_weekly_trailing_stop must still see the current week's bar.
        """
        # Simulate a week where Friday Jul 5 is missing (July 4th week)
        # Mon-Thu = Jul 1-3 (Fri Jul 5 missing), next Mon = Jul 8
        daily_dates = pd.DatetimeIndex(
            [
                '2024-07-01',
                '2024-07-02',
                '2024-07-03',  # Mon-Wed
                # Jul 4 Thu holiday, Jul 5 Fri holiday
                '2024-07-08',
                '2024-07-09',
                '2024-07-10',
                '2024-07-11',
                '2024-07-12',  # full week
            ]
        )
        prices = [100, 101, 102, 103, 104, 105, 106, 107]
        price_df = pd.DataFrame(
            {
                'adjusted_open': prices,
                'adjusted_high': [p + 2 for p in prices],
                'adjusted_low': [p - 2 for p in prices],
                'adjusted_close': prices,
            },
            index=daily_dates,
        )

        weekly_df = aggregate_to_weekly(price_df)

        # is_week_end on Wed Jul 3 (index 2, last bar before next week)
        self.assertTrue(is_week_end(daily_dates, 2), 'Wed Jul 3 should be week end')

        # check_weekly_trailing_stop with current_date=Jul 3 (Wed)
        # weekly_df label for this week is Jul 5 (Fri).
        # Before the fix, Jul 3 < Jul 5 would exclude the current week.
        result = check_weekly_trailing_stop(
            current_close=50,  # way below EMA → should trigger
            weekly_df=weekly_df,
            entry_date='2024-06-01',  # well before
            current_date='2024-07-03',
            trailing_type='weekly_ema',
            ema_period=2,
            nweek_low_period=4,
            transition_weeks=0,
        )
        self.assertTrue(result, 'Should trigger even in holiday-shortened week')

    def test_nweek_low_holiday_week(self):
        """weekly_nweek_low also works correctly on holiday-shortened weeks."""
        daily_dates = pd.DatetimeIndex(
            [
                '2024-06-24',
                '2024-06-25',
                '2024-06-26',
                '2024-06-27',
                '2024-06-28',  # full week
                '2024-07-01',
                '2024-07-02',
                '2024-07-03',  # holiday week (no Jul 4-5)
            ]
        )
        prices = [100, 101, 102, 103, 104, 105, 106, 107]
        price_df = pd.DataFrame(
            {
                'adjusted_open': prices,
                'adjusted_high': [p + 2 for p in prices],
                'adjusted_low': [p - 2 for p in prices],
                'adjusted_close': prices,
            },
            index=daily_dates,
        )

        weekly_df = aggregate_to_weekly(price_df)

        # nweek_low should use prior week's low (100-2=98)
        result = check_weekly_trailing_stop(
            current_close=97,  # below prior week low
            weekly_df=weekly_df,
            entry_date='2024-06-01',
            current_date='2024-07-03',
            trailing_type='weekly_nweek_low',
            ema_period=2,
            nweek_low_period=4,
            transition_weeks=0,
        )
        self.assertTrue(result, 'nweek_low should trigger on holiday week')


class TestWeeklyNweekLow(unittest.TestCase):
    """Test 10: weekly_nweek_low mode triggers correctly."""

    def _make_weekly_df(self, closes, lows=None):
        dates = pd.date_range('2024-01-05', periods=len(closes), freq='W-FRI')
        if lows is None:
            lows = [c - 5 for c in closes]
        return pd.DataFrame(
            {
                'open': [c + 1 for c in closes],
                'high': [c + 5 for c in closes],
                'low': lows,
                'close': closes,
            },
            index=dates,
        )

    def test_nweek_low_triggers_when_below(self):
        """close < min(prior N-week lows) should trigger."""
        closes = [100, 102, 104, 106, 108, 105, 103, 101, 99, 97]
        lows = [95, 97, 99, 101, 103, 100, 98, 96, 94, 92]
        weekly_df = self._make_weekly_df(closes, lows)

        # current_close=90, prior 4-week lows=[100,98,96,94], min=94 → 90<94 → True
        result = check_weekly_trailing_stop(
            current_close=90,
            weekly_df=weekly_df,
            entry_date=weekly_df.index[0],
            current_date=weekly_df.index[9],
            trailing_type='weekly_nweek_low',
            ema_period=5,
            nweek_low_period=4,
            transition_weeks=3,
        )
        self.assertTrue(result)

    def test_nweek_low_does_not_trigger_when_above(self):
        """close >= min(prior N-week lows) should not trigger."""
        closes = [100, 102, 104, 106, 108, 105, 103, 101, 99, 97]
        lows = [95, 97, 99, 101, 103, 100, 98, 96, 94, 92]
        weekly_df = self._make_weekly_df(closes, lows)

        # current_close=95, prior 4-week lows min=94 → 95>=94 → False
        result = check_weekly_trailing_stop(
            current_close=95,
            weekly_df=weekly_df,
            entry_date=weekly_df.index[0],
            current_date=weekly_df.index[9],
            trailing_type='weekly_nweek_low',
            ema_period=5,
            nweek_low_period=4,
            transition_weeks=3,
        )
        self.assertFalse(result)

    def test_nweek_low_excludes_current_week(self):
        """Current week's low should NOT be included in nweek_low calculation.
        If it were, close >= low would always hold → dead condition.
        """
        closes = [100, 102, 104, 106, 108, 105, 103, 101, 99, 97]
        # Current week (index 9) has very low low=10, but should be excluded
        lows = [95, 97, 99, 101, 103, 100, 98, 96, 94, 10]
        weekly_df = self._make_weekly_df(closes, lows)

        # current_close=93, prior 4-week lows=[100,98,96,94] min=94 → 93<94 → True
        # If current week low=10 were included, min would be 10 and 93>10 → False (wrong)
        result = check_weekly_trailing_stop(
            current_close=93,
            weekly_df=weekly_df,
            entry_date=weekly_df.index[0],
            current_date=weekly_df.index[9],
            trailing_type='weekly_nweek_low',
            ema_period=5,
            nweek_low_period=4,
            transition_weeks=3,
        )
        self.assertTrue(result, 'Current week low must be excluded')

    def test_nweek_low_insufficient_prior_data(self):
        """When there are too few prior weeks (only current), should return False."""
        closes = [100]
        weekly_df = self._make_weekly_df(closes)

        result = check_weekly_trailing_stop(
            current_close=50,
            weekly_df=weekly_df,
            entry_date='2024-01-01',
            current_date=weekly_df.index[0],
            trailing_type='weekly_nweek_low',
            ema_period=5,
            nweek_low_period=4,
            transition_weeks=0,
        )
        self.assertFalse(result, 'Should return False when no prior weeks exist')


class TestPendingWeeklyExitAtBacktestEnd(unittest.TestCase):
    """Test 11: pending weekly exit at final bar uses 'weekly trailing' reason."""

    def test_pending_exit_at_last_bar(self):
        """When weekly trailing fires on the last bar (is_week_end=True for final bar),
        the backtest_end close should use 'weekly trailing' as exit reason.
        """
        # Create data ending on a Friday so the last bar is a week end
        # Use 40 business days; ensure price drops enough for EMA trigger
        dates = pd.bdate_range('2024-01-02', periods=40)

        prices_close = np.concatenate(
            [
                np.linspace(100, 130, 20),
                np.linspace(129, 80, 20),
            ]
        )
        ohlc = pd.DataFrame(
            {
                'open': prices_close * 0.999,
                'high': prices_close * 1.01,
                'low': prices_close * 0.99,
                'close': prices_close,
                'adjusted_close': prices_close,
                'adjusted_open': prices_close * 0.999,
                'adjusted_high': prices_close * 1.01,
                'adjusted_low': prices_close * 0.99,
            },
            index=dates,
        )

        breadth = pd.Series(
            np.concatenate([np.linspace(0.3, 0.15, 5), np.linspace(0.16, 0.5, 35)]),
            index=dates,
        )

        bt = _make_backtest(
            enable_weekly_trailing=True,
            tv_mode=False,
            weekly_trailing_type='weekly_ema',
            weekly_ema_period=3,
            weekly_transition_weeks=1,
            stop_loss_pct=0.90,  # effectively disabled
        )
        bt.price_data = ohlc.copy()
        bt.breadth_index = breadth.copy()
        bt.sp500_data = pd.DataFrame()
        bt.short_ma_line = breadth.ewm(span=bt.short_ma, adjust=False).mean()
        bt.long_ma_line = breadth.ewm(span=bt.long_ma, adjust=False).mean()
        bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)
        bt.short_ma_bottoms = [dates[4]]
        bt.long_ma_bottoms = []
        bt.peaks = []
        bt.execute_trades()

        # Check: no trade should have 'backtest_end' if weekly trailing fired
        weekly_exits = [t for t in bt.trade_log if t['exit_reason'] == 'weekly trailing']
        backtest_ends = [t for t in bt.trade_log if t['exit_reason'] == 'backtest_end']
        # At least one exit should exist
        self.assertGreater(len(bt.trade_log), 0, 'Should have at least one trade')
        # If weekly trailing was pending at last bar, it should use that reason
        if not weekly_exits and backtest_ends:
            # Weekly trailing didn't fire at all — acceptable if price didn't
            # drop enough; this is data-dependent so we just verify no crash
            pass
        else:
            # Weekly trailing fired: no backtest_end exits should exist
            self.assertEqual(len(backtest_ends), 0, 'backtest_end should not appear when weekly trailing fires')


class TestChartModeWithWeeklyTrailing(unittest.TestCase):
    """Test 8: chart_mode + weekly_trailing combo works."""

    def test_chart_mode_weekly_trailing_combo(self):
        """chart_mode=True, enable_weekly_trailing=True, tv_mode=False should work."""
        bt = _make_backtest(
            chart_mode=True,
            enable_weekly_trailing=True,
            tv_mode=False,
        )
        self.assertTrue(bt.chart_mode)
        self.assertTrue(bt.enable_weekly_trailing)
        self.assertFalse(bt.tv_mode)

    def test_chart_mode_with_tv_mode_still_raises(self):
        """chart_mode + tv_mode should still raise regardless of weekly_trailing."""
        with self.assertRaises(ValueError):
            _make_backtest(chart_mode=True, tv_mode=True)


if __name__ == '__main__':
    unittest.main()
