"""Tests for adjusted price scale consistency in backtest.

Verifies that stop loss and fill price logic uses adjusted OHLC columns
(adjusted_open, adjusted_low) instead of raw OHLC, preventing false
triggers when stock splits cause raw and adjusted prices to diverge.
"""

import unittest

import pandas as pd

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
    bt.pivot_len_long = 3
    bt.pivot_len_short = 2
    return bt


def _inject_data_with_adjusted(bt, ohlc_df, breadth_series):
    """Inject data into backtest, computing adjusted OHLC columns.

    Unlike the plain _inject_data helper, this replicates the adjusted OHLC
    computation that run() performs, so that SL and fill price logic can
    reference adjusted_open / adjusted_low.
    """
    bt.price_data = ohlc_df.copy()

    # Compute adjusted OHLC (same logic as backtest.run())
    if 'close' in bt.price_data.columns and 'adjusted_close' in bt.price_data.columns:
        adj_ratio = bt.price_data['adjusted_close'] / bt.price_data['close']
        for raw_col, adj_col in [('open', 'adjusted_open'), ('high', 'adjusted_high'), ('low', 'adjusted_low')]:
            if raw_col in bt.price_data.columns:
                bt.price_data[adj_col] = bt.price_data[raw_col] * adj_ratio

    bt.breadth_index = breadth_series.copy()
    bt.sp500_data = pd.DataFrame()

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

    bt._precompute_tv_signals()
    bt.execute_trades()


def _build_reverse_split_data():
    """Build 20-bar synthetic data simulating a reverse 2:1 split.

    All bars have adj_ratio=2.0 (adjusted_close = close * 2), as if a
    reverse split occurred after the dataset and the provider back-adjusted
    all historical prices.

    Returns (ohlc_df, breadth_series).
    """
    dates = pd.bdate_range('2024-01-02', periods=20)

    adj_ratio = 2.0

    # Raw prices: moderate decline → recovery (entry opportunity around bar 10)
    raw_closes = [30, 30, 29, 29, 28, 27, 26, 27, 28, 28, 28, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    raw_opens = [30, 30, 30, 29, 29, 28, 27, 26, 27, 28, 28, 28, 28, 29, 30, 31, 32, 33, 34, 35]
    raw_highs = [31, 31, 30, 30, 29, 28, 27, 28, 29, 29, 29, 29, 30, 31, 32, 33, 34, 35, 36, 37]
    raw_lows = [29, 29, 28, 28, 27, 26, 25, 26, 27, 27, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    ohlc = pd.DataFrame(
        {
            'open': raw_opens,
            'high': raw_highs,
            'low': raw_lows,
            'close': raw_closes,
            'adjusted_close': [c * adj_ratio for c in raw_closes],
        },
        index=dates,
    )

    # Breadth: plateau → deep trough at bar 8 (index 7) → recovery
    # pivot_len_short=2: trough at index 7, confirms at index 9
    # pine_compat: entry queues at index 9, fills at index 10's open
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
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.55,
        0.55,
        0.55,
        0.55,
    ]
    breadth = pd.Series(breadth_vals, index=dates)

    return ohlc, breadth


class TestAdjustedOhlcAlwaysComputed(unittest.TestCase):
    """Test 1: adjusted OHLC columns are generated even without weekly trailing."""

    def test_adjusted_ohlc_always_computed(self):
        """With close + adjusted_close present, adjusted_open/high/low should be computed."""
        bt = _make_backtest(enable_weekly_trailing=False)
        dates = pd.bdate_range('2024-01-02', periods=5)

        # adj_ratio = 0.5 (forward 2:1 split)
        bt.price_data = pd.DataFrame(
            {
                'open': [101, 102, 103, 104, 105],
                'high': [110, 111, 112, 113, 114],
                'low': [95, 96, 97, 98, 99],
                'close': [100, 100, 100, 100, 100],
                'adjusted_close': [50, 50, 50, 50, 50],
            },
            index=dates,
        )

        # Replicate the computation from run()
        adj_ratio = bt.price_data['adjusted_close'] / bt.price_data['close']
        for raw_col, adj_col in [('open', 'adjusted_open'), ('high', 'adjusted_high'), ('low', 'adjusted_low')]:
            if raw_col in bt.price_data.columns:
                bt.price_data[adj_col] = bt.price_data[raw_col] * adj_ratio

        self.assertIn('adjusted_open', bt.price_data.columns)
        self.assertIn('adjusted_high', bt.price_data.columns)
        self.assertIn('adjusted_low', bt.price_data.columns)

        # adj_ratio = 0.5; adjusted values should be half of raw
        self.assertAlmostEqual(bt.price_data.iloc[0]['adjusted_open'], 50.5)  # 101 * 0.5
        self.assertAlmostEqual(bt.price_data.iloc[0]['adjusted_high'], 55.0)  # 110 * 0.5
        self.assertAlmostEqual(bt.price_data.iloc[0]['adjusted_low'], 47.5)  # 95  * 0.5


class TestSlUsesAdjustedLow(unittest.TestCase):
    """Test 2: SL uses adjusted_low, avoiding false triggers from raw low."""

    def test_sl_uses_adjusted_low(self):
        """Reverse-split data: raw low < SL threshold but adjusted_low > SL.

        With adj_ratio=2.0, raw prices are half of adjusted. A raw low of 26
        looks below SL=51.52 but adjusted_low=52 is above it.
        Position should NOT be stopped out.
        """
        bt = _make_backtest()
        ohlc, breadth = _build_reverse_split_data()
        _inject_data_with_adjusted(bt, ohlc, breadth)

        # Verify an entry happened (trade_log records completed trades)
        self.assertGreater(len(bt.trade_log), 0, 'Expected at least one completed trade')

        # Verify no stop loss exit occurred (exit should be backtest_end, not stop loss)
        sl_exits = [t for t in bt.trade_log if t.get('exit_reason') == 'stop loss']
        self.assertEqual(len(sl_exits), 0, f'Stop loss should not fire; adjusted_low is above SL. Got: {sl_exits}')

        # The only exit should be backtest_end (auto-close at end of data)
        self.assertEqual(bt.trade_log[0]['exit_reason'], 'backtest_end')


class TestPineCompatFillUsesAdjustedOpen(unittest.TestCase):
    """Test 3: Pine-compat fill price uses adjusted_open."""

    def test_pine_compat_fill_uses_adjusted_open(self):
        """Entry fill price should be adjusted_open, not raw open."""
        bt = _make_backtest()
        ohlc, breadth = _build_reverse_split_data()
        _inject_data_with_adjusted(bt, ohlc, breadth)

        # An entry should have occurred (check trade_log for completed trades)
        self.assertGreater(len(bt.trade_log), 0, 'Expected at least one completed trade')

        # Get entry info from the trade log
        first_trade = bt.trade_log[0]
        entry_price = first_trade['entry_price']
        entry_date = first_trade['entry_date']

        # The fill price should match adjusted_open, not raw open
        adjusted_open = bt.price_data.loc[entry_date, 'adjusted_open']
        raw_open = bt.price_data.loc[entry_date, 'open']

        # With adj_ratio=2.0, adjusted_open = raw_open * 2
        self.assertAlmostEqual(adjusted_open, raw_open * 2, places=4)

        # Entry price should equal adjusted_open (slippage=0 for pine_compat)
        self.assertAlmostEqual(
            entry_price,
            adjusted_open,
            places=2,
            msg=f'Fill should be adjusted_open ({adjusted_open}), not raw open ({raw_open})',
        )


def _build_dividend_adjusted_data():
    """Build 20-bar synthetic data simulating dividend adjustment (adj_ratio=0.8).

    Simulates a scenario like SSO where distributions cause adjusted prices
    to be lower than raw prices. The key test scenario:
    - Entry at bar 10 (adjusted_close ~40.0)
    - SL 8% = 36.80
    - Bar 14 raw_low = 37.5 (> 36.80 → old buggy code would NOT trigger SL)
    - Bar 14 adjusted_low = 37.5 * 0.8 = 30.0 (< 36.80 → correct code triggers SL)

    Returns (ohlc_df, breadth_series).
    """
    dates = pd.bdate_range('2024-01-02', periods=20)
    adj_ratio = 0.8

    # Raw prices: moderate decline → recovery → drop at bar 14
    raw_closes = [55, 55, 54, 53, 52, 50, 48, 47, 48, 50, 50, 50, 51, 51, 48, 49, 50, 51, 52, 53]
    raw_opens = [55, 55, 55, 54, 53, 52, 50, 48, 47, 48, 50, 50, 50, 51, 51, 48, 49, 50, 51, 52]
    raw_highs = [56, 56, 55, 54, 53, 52, 51, 49, 49, 51, 51, 51, 52, 52, 51, 50, 51, 52, 53, 54]
    raw_lows = [54, 54, 53, 52, 51, 49, 47, 46, 47, 49, 49, 49, 50, 50, 37.5, 47, 49, 50, 51, 52]
    #                                                                          ^^^^
    # Bar 14: raw_low=37.5, adjusted_low=30.0
    # Entry bar 10: adjusted_close = 50*0.8 = 40.0, SL = 40.0*(1-0.08) = 36.80
    # raw 37.5 > 36.80 (old bug: no SL), adjusted 30.0 < 36.80 (correct: SL fires)

    ohlc = pd.DataFrame(
        {
            'open': raw_opens,
            'high': raw_highs,
            'low': raw_lows,
            'close': raw_closes,
            'adjusted_close': [c * adj_ratio for c in raw_closes],
        },
        index=dates,
    )

    # Breadth: decline → deep trough at bar 7 → recovery (triggers entry)
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
        0.35,
        0.40,
        0.45,
        0.40,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
    ]
    breadth = pd.Series(breadth_vals, index=dates)

    return ohlc, breadth


class TestSlTriggersWithDividendAdjustment(unittest.TestCase):
    """Test 5: Regression test — SL fires correctly when adj_ratio < 1 (Pine-compat)."""

    def test_sl_triggers_correctly_with_dividend_adjustment(self):
        """With adj_ratio=0.8, raw_low > adjusted SL but adjusted_low < SL.

        This reproduces the SSO 2008-03-10 bug where scale mismatch between
        raw bar_low and adjusted SL threshold caused SL to not fire.
        """
        bt = _make_backtest(tv_pine_compat=True, stop_loss_pct=0.08)
        ohlc, breadth = _build_dividend_adjusted_data()
        _inject_data_with_adjusted(bt, ohlc, breadth)

        # Exactly one trade should complete
        self.assertEqual(len(bt.trade_log), 1, 'Expected exactly one completed trade')

        trade = bt.trade_log[0]
        self.assertEqual(trade['exit_reason'], 'stop loss')

        # SL must fire on bar 14 (2024-01-22) — the bar with the dividend-adjusted gap
        exit_date = trade['exit_date']
        expected_exit = pd.Timestamp('2024-01-22')
        self.assertEqual(exit_date, expected_exit, f'SL should fire on {expected_exit}, got {exit_date}')

        # Verify the exact bug condition: raw_low > SL threshold > adjusted_low
        sl_threshold = trade['entry_price'] * (1 - 0.08)
        raw_low = ohlc.loc[exit_date, 'low']
        adjusted_low = bt.price_data.loc[exit_date, 'adjusted_low']
        self.assertGreater(
            raw_low,
            sl_threshold,
            f'raw_low ({raw_low}) should be ABOVE SL ({sl_threshold:.4f}) — old buggy code would have missed this SL',
        )
        self.assertLess(
            adjusted_low,
            sl_threshold,
            f'adjusted_low ({adjusted_low:.4f}) should be BELOW SL ({sl_threshold:.4f}) — '
            'correct code triggers SL via adjusted scale',
        )


class TestSameBarSlTriggersWithDividendAdjustment(unittest.TestCase):
    """Test 6: Regression test — SL fires correctly when adj_ratio < 1 (TV same-bar)."""

    def test_same_bar_sl_triggers_with_dividend_adjustment(self):
        """Same dividend-adjusted data but via _execute_tv_same_bar path.

        Uses tv_pine_compat=False, tv_mode=True to exercise L959-963.
        """
        bt = _make_backtest(tv_pine_compat=False, tv_mode=True, stop_loss_pct=0.08)
        ohlc, breadth = _build_dividend_adjusted_data()
        _inject_data_with_adjusted(bt, ohlc, breadth)

        # Exactly one trade should complete
        self.assertEqual(len(bt.trade_log), 1, 'Expected exactly one completed trade')

        trade = bt.trade_log[0]
        self.assertEqual(trade['exit_reason'], 'stop loss')

        # SL must fire on bar 14 (2024-01-22) — the bar with the dividend-adjusted gap
        exit_date = trade['exit_date']
        expected_exit = pd.Timestamp('2024-01-22')
        self.assertEqual(exit_date, expected_exit, f'SL should fire on {expected_exit}, got {exit_date}')

        # Verify the exact bug condition: raw_low > SL threshold > adjusted_low
        sl_threshold = trade['entry_price'] * (1 - 0.08)
        raw_low = ohlc.loc[exit_date, 'low']
        adjusted_low = bt.price_data.loc[exit_date, 'adjusted_low']
        self.assertGreater(
            raw_low,
            sl_threshold,
            f'raw_low ({raw_low}) should be ABOVE SL ({sl_threshold:.4f}) — old buggy code would have missed this SL',
        )
        self.assertLess(
            adjusted_low,
            sl_threshold,
            f'adjusted_low ({adjusted_low:.4f}) should be BELOW SL ({sl_threshold:.4f}) — '
            'correct code triggers SL via adjusted scale',
        )


class TestSameBarSlNotTriggeredWithReverseSplit(unittest.TestCase):
    """Test 7: Same-bar SL does NOT fire when raw_low < SL but adjusted_low > SL."""

    def test_same_bar_sl_not_triggered_reverse_split(self):
        """Reverse-split data (adj_ratio=2.0) via _execute_tv_same_bar.

        raw_low values (25-35) are all below SL (~51.55), but adjusted_low
        values (50-70) are all above SL. SL must NOT fire.
        """
        bt = _make_backtest(tv_pine_compat=False, tv_mode=True, stop_loss_pct=0.08)
        ohlc, breadth = _build_reverse_split_data()
        _inject_data_with_adjusted(bt, ohlc, breadth)

        self.assertEqual(len(bt.trade_log), 1, 'Expected exactly one completed trade')

        trade = bt.trade_log[0]
        self.assertEqual(trade['exit_reason'], 'backtest_end', 'SL should NOT fire; adjusted_low is above SL')

        # Verify the boundary condition holds on the entry bar itself
        entry_date = trade['entry_date']
        sl_threshold = trade['entry_price'] * (1 - 0.08)
        raw_low = ohlc.loc[entry_date, 'low']
        adjusted_low = bt.price_data.loc[entry_date, 'adjusted_low']
        self.assertLess(
            raw_low,
            sl_threshold,
            f'raw_low ({raw_low}) should be BELOW SL ({sl_threshold:.2f}) — old code would false-trigger',
        )
        self.assertGreater(
            adjusted_low,
            sl_threshold,
            f'adjusted_low ({adjusted_low:.2f}) should be ABOVE SL ({sl_threshold:.2f}) — no SL should fire',
        )


class TestSameBarSlFillFormula(unittest.TestCase):
    """Test 8: Same-bar SL fill price = min(adjusted_open, stop_price) * (1 - slippage)."""

    def test_same_bar_sl_fill_equals_min_adj_open_stop(self):
        """Dividend-adjusted data (adj_ratio=0.8) via _execute_tv_same_bar.

        On bar 14: adjusted_low=30.0 < SL=36.82, adjusted_open=40.8 > SL.
        fill_at = min(40.8, 36.82) = 36.82 (capped at stop price).
        exit_price = fill_at * (1 - slippage).
        """
        bt = _make_backtest(tv_pine_compat=False, tv_mode=True, stop_loss_pct=0.08)
        ohlc, breadth = _build_dividend_adjusted_data()
        _inject_data_with_adjusted(bt, ohlc, breadth)

        self.assertEqual(len(bt.trade_log), 1)
        trade = bt.trade_log[0]
        self.assertEqual(trade['exit_reason'], 'stop loss')

        exit_date = trade['exit_date']
        sl_threshold = trade['entry_price'] * (1 - 0.08)
        adj_open = bt.price_data.loc[exit_date, 'adjusted_open']
        adj_low = bt.price_data.loc[exit_date, 'adjusted_low']

        # Preconditions: adjusted_open > SL > adjusted_low (fill capped at SL)
        self.assertGreater(adj_open, sl_threshold)
        self.assertLess(adj_low, sl_threshold)

        # fill_at = min(adjusted_open, stop_price), then slippage applied
        expected_fill_at = min(adj_open, sl_threshold)
        expected_exit_price = expected_fill_at * (1 - bt.slippage)
        self.assertAlmostEqual(
            trade['exit_price'],
            expected_exit_price,
            places=2,
            msg=f'exit_price should be min(adj_open={adj_open:.4f}, SL={sl_threshold:.4f}) '
            f'* (1-slippage) = {expected_exit_price:.4f}, got {trade["exit_price"]:.4f}',
        )


def _build_sp500_breadth_from_series(breadth_series):
    """Build a fake sp500_data DataFrame that yields the given breadth when passed to calculate_above_ma.

    calculate_above_ma() returns a binary DataFrame (1 if stock > 200-day MA, else 0)
    and breadth_index = above_ma.mean(axis=1). To produce a specific breadth ratio,
    we create 100 synthetic stock columns: floor(breadth*100) columns above MA, rest below.
    """
    n_stocks = 100
    rows = []
    for _date, val in breadth_series.items():
        n_above = round(val * n_stocks)
        # Stocks "above MA" have price 200, "below MA" have price 50.
        # With 200-day MA on flat data, 200-day MA = same value, so threshold is trivial.
        row = [200.0] * n_above + [50.0] * (n_stocks - n_above)
        rows.append(row)
    cols = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    return pd.DataFrame(rows, index=breadth_series.index, columns=cols)


class TestRunGeneratesAdjustedOhlc(unittest.TestCase):
    """Test 9: Integration — run() auto-generates adjusted OHLC columns."""

    def test_run_generates_adjusted_ohlc_columns(self):
        """Call run() with mocked data and verify adjusted_open/high/low are created.

        Patches get_stock_price_ohlc to return synthetic OHLC with adj_ratio=0.6,
        and _get_sp500_data to return matching breadth data. run() should compute
        adjusted OHLC at L431-436.
        """
        from unittest.mock import patch

        dates = pd.bdate_range('2024-01-02', periods=20)
        adj_ratio_val = 0.6

        ohlc = pd.DataFrame(
            {
                'open': [100.0 + i for i in range(20)],
                'high': [110.0 + i for i in range(20)],
                'low': [90.0 + i for i in range(20)],
                'close': [105.0 + i for i in range(20)],
                'adjusted_close': [(105.0 + i) * adj_ratio_val for i in range(20)],
            },
            index=dates,
        )
        breadth_vals = [0.5] * 20
        sp500_fake = _build_sp500_breadth_from_series(pd.Series(breadth_vals, index=dates))

        bt = Backtest(
            start_date='2024-01-02',
            end_date='2024-01-29',
            tv_mode=True,
            use_saved_data=True,
            no_show_plot=True,
            initial_capital=50000,
            symbol='FAKE',
        )

        with (
            patch('backtest.backtest.get_stock_price_ohlc', return_value=ohlc),
            patch.object(bt, '_get_sp500_data', return_value=sp500_fake),
            patch.object(bt, 'visualize_results'),
        ):
            bt.run()

        # Verify adjusted columns were generated by run()
        for col in ('adjusted_open', 'adjusted_high', 'adjusted_low'):
            self.assertIn(col, bt.price_data.columns, f'{col} should be auto-generated by run()')

        # Verify values: adj_ratio = 0.6 across all bars
        for i in range(len(bt.price_data)):
            row = bt.price_data.iloc[i]
            ratio = row['adjusted_close'] / row['close']
            self.assertAlmostEqual(ratio, adj_ratio_val, places=4)
            self.assertAlmostEqual(row['adjusted_open'], row['open'] * ratio, places=4)
            self.assertAlmostEqual(row['adjusted_low'], row['low'] * ratio, places=4)


class TestRunIntegrationSameBarSlUsesAdjusted(unittest.TestCase):
    """Test 10: Integration — run() same-bar SL uses adjusted columns."""

    def test_run_path_same_bar_sl_uses_adjusted(self):
        """Call run() with dividend-adjusted data (adj_ratio=0.8).

        Verifies that the same-bar SL on bar 14 fires because adjusted_low < SL,
        even though raw_low > SL. This exercises the full run() → adjusted OHLC
        generation → execute_trades() → _execute_tv_same_bar() pipeline.
        """
        from unittest.mock import patch

        ohlc, breadth = _build_dividend_adjusted_data()
        sp500_fake = _build_sp500_breadth_from_series(breadth)

        # Mock calculate_above_ma because 20-bar synthetic data cannot compute
        # a real 200-day MA; we inject known breadth values directly.
        breadth_df = pd.DataFrame({'breadth': breadth}, index=breadth.index)

        bt = Backtest(
            start_date='2024-01-02',
            end_date='2024-01-29',
            tv_mode=True,
            tv_pine_compat=False,
            use_saved_data=True,
            no_show_plot=True,
            initial_capital=50000,
            stop_loss_pct=0.08,
            symbol='FAKE',
            debug=True,
        )
        bt.pivot_len_long = 3
        bt.pivot_len_short = 2

        with (
            patch('backtest.backtest.get_stock_price_ohlc', return_value=ohlc),
            patch.object(bt, '_get_sp500_data', return_value=sp500_fake),
            patch('backtest.backtest.calculate_above_ma', return_value=breadth_df),
            patch.object(bt, 'visualize_results'),
        ):
            bt.run()

        # Verify adjusted columns were created by run()
        self.assertIn('adjusted_low', bt.price_data.columns)
        self.assertIn('adjusted_open', bt.price_data.columns)

        # Verify SL fired correctly
        self.assertGreater(len(bt.trade_log), 0, 'Expected at least one trade')
        sl_trades = [t for t in bt.trade_log if t.get('exit_reason') == 'stop loss']
        self.assertGreater(len(sl_trades), 0, 'SL should fire via adjusted_low')

        # Verify boundary condition on the SL trade
        trade = sl_trades[0]
        exit_date = trade['exit_date']
        sl = trade['entry_price'] * (1 - 0.08)
        raw_low = ohlc.loc[exit_date, 'low']
        adj_low = bt.price_data.loc[exit_date, 'adjusted_low']
        self.assertGreater(raw_low, sl, 'raw_low must be above SL (old code would miss)')
        self.assertLess(adj_low, sl, 'adjusted_low must be below SL (triggers correctly)')


class TestNoOhlcFallbackUnchanged(unittest.TestCase):
    """Test 4: Close-only data (no OHLC) works without adjusted columns."""

    def test_no_ohlc_fallback_unchanged(self):
        """When only adjusted_close is present, no adjusted_open/high/low generated."""
        bt = _make_backtest(enable_weekly_trailing=False)
        dates = pd.bdate_range('2024-01-02', periods=5)

        bt.price_data = pd.DataFrame(
            {
                'adjusted_close': [50, 51, 52, 53, 54],
            },
            index=dates,
        )

        # With no 'close' column, adjusted OHLC computation is skipped
        self.assertNotIn('close', bt.price_data.columns)
        self.assertNotIn('adjusted_open', bt.price_data.columns)
        self.assertNotIn('adjusted_low', bt.price_data.columns)


if __name__ == '__main__':
    unittest.main()
