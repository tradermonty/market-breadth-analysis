"""
Test code for trade logging functionality (Phase 1)
TDD approach - Tests written before implementation
"""

import os
import unittest

import pandas as pd

from backtest.backtest import Backtest


class TestTradeLogging(unittest.TestCase):
    """Test cases for trade logging functionality"""

    def setUp(self):
        """Setup test environment"""
        # Use a short test period
        self.start_date = '2023-01-01'
        self.end_date = '2023-12-31'
        self.symbol = 'SPY'

    def test_01_trade_log_initialization(self):
        """Test: trade_log and open_positions are initialized as empty lists"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Verify that trade_log and open_positions exist and are empty
        self.assertTrue(hasattr(backtest, 'trade_log'))
        self.assertTrue(hasattr(backtest, 'open_positions'))
        self.assertEqual(len(backtest.trade_log), 0)
        self.assertEqual(len(backtest.open_positions), 0)

    def test_02_open_positions_on_entry(self):
        """Test: Entry adds to open_positions"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Simulate an entry
        test_date = pd.Timestamp('2023-06-01')
        test_price = 100.0
        test_shares = 50
        test_reason = 'long_ma_bottom'

        # Execute entry (this will be implemented)
        backtest._execute_entry(test_date, test_price, test_shares, test_reason)

        # Verify open_positions has one entry
        self.assertEqual(len(backtest.open_positions), 1)

        # Verify the content of open_positions
        position = backtest.open_positions[0]
        self.assertEqual(position['entry_date'], test_date)
        self.assertEqual(position['entry_price'], test_price * (1 + backtest.slippage))
        self.assertEqual(position['entry_shares'], test_shares)
        self.assertEqual(position['entry_reason'], test_reason)
        self.assertIn('entry_cost', position)

    def test_03_multiple_entries(self):
        """Test: Multiple entries are recorded in open_positions"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # First entry
        backtest._execute_entry(pd.Timestamp('2023-06-01'), 100.0, 50, 'long_ma_bottom')

        # Second entry (buy more)
        backtest._execute_entry(pd.Timestamp('2023-06-15'), 105.0, 30, 'short_ma_bottom')

        # Verify both entries are in open_positions
        self.assertEqual(len(backtest.open_positions), 2)
        self.assertEqual(backtest.open_positions[0]['entry_shares'], 50)
        self.assertEqual(backtest.open_positions[1]['entry_shares'], 30)

    def test_04_trade_log_on_complete_exit(self):
        """Test: Complete exit creates trade_log entry and clears open_positions"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Entry
        entry_date = pd.Timestamp('2023-06-01')
        backtest._execute_entry(entry_date, 100.0, 50, 'long_ma_bottom')

        # Exit
        exit_date = pd.Timestamp('2023-07-01')
        backtest._execute_exit(exit_date, 110.0, reason='peak exit')

        # Verify trade_log has one entry
        self.assertEqual(len(backtest.trade_log), 1)

        # Verify open_positions is empty
        self.assertEqual(len(backtest.open_positions), 0)

        # Verify trade_log content
        trade = backtest.trade_log[0]
        self.assertEqual(trade['trade_id'], 1)
        self.assertEqual(trade['entry_date'], entry_date)
        self.assertEqual(trade['exit_date'], exit_date)
        self.assertEqual(trade['entry_shares'], 50)
        self.assertEqual(trade['exit_shares'], 50)
        self.assertEqual(trade['entry_reason'], 'long_ma_bottom')
        self.assertEqual(trade['exit_reason'], 'peak exit')
        self.assertIn('holding_days', trade)
        self.assertIn('pnl_dollar', trade)
        self.assertIn('pnl_percent', trade)
        self.assertIn('cumulative_pnl', trade)

    def test_05_holding_days_calculation(self):
        """Test: Holding days are calculated correctly"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Entry and exit with known dates
        entry_date = pd.Timestamp('2023-06-01')
        exit_date = pd.Timestamp('2023-06-15')

        backtest._execute_entry(entry_date, 100.0, 50, 'long_ma_bottom')
        backtest._execute_exit(exit_date, 110.0, reason='peak exit')

        # Verify holding days
        trade = backtest.trade_log[0]
        expected_days = (exit_date - entry_date).days
        self.assertEqual(trade['holding_days'], expected_days)

    def test_06_pnl_calculation(self):
        """Test: P&L is calculated correctly"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            slippage=0.001,
            commission=0.001,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Entry
        entry_price = 100.0
        shares = 50
        backtest._execute_entry(pd.Timestamp('2023-06-01'), entry_price, shares, 'long_ma_bottom')

        # Exit
        exit_price = 110.0
        backtest._execute_exit(pd.Timestamp('2023-07-01'), exit_price, reason='peak exit')

        # Calculate expected P&L
        entry_price_with_slippage = entry_price * (1 + backtest.slippage)
        entry_commission = entry_price_with_slippage * shares * backtest.commission
        entry_cost = entry_price_with_slippage * shares + entry_commission

        exit_price_with_slippage = exit_price * (1 - backtest.slippage)
        exit_commission = exit_price_with_slippage * shares * backtest.commission
        exit_proceeds = exit_price_with_slippage * shares - exit_commission

        expected_pnl = exit_proceeds - entry_cost
        expected_pnl_pct = (expected_pnl / entry_cost) * 100

        # Verify P&L
        trade = backtest.trade_log[0]
        self.assertAlmostEqual(trade['pnl_dollar'], expected_pnl, places=2)
        self.assertAlmostEqual(trade['pnl_percent'], expected_pnl_pct, places=2)

    def test_07_fifo_logic_on_exit(self):
        """Test: FIFO logic works correctly with multiple entries"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # First entry: 50 shares
        backtest._execute_entry(pd.Timestamp('2023-06-01'), 100.0, 50, 'long_ma_bottom')

        # Second entry: 30 shares
        backtest._execute_entry(pd.Timestamp('2023-06-15'), 105.0, 30, 'short_ma_bottom')

        # Exit all 80 shares
        backtest._execute_exit(pd.Timestamp('2023-07-01'), 110.0, reason='peak exit')

        # Verify two trade_log entries (FIFO)
        self.assertEqual(len(backtest.trade_log), 2)

        # First trade should match first entry
        self.assertEqual(backtest.trade_log[0]['entry_shares'], 50)
        self.assertEqual(backtest.trade_log[0]['entry_reason'], 'long_ma_bottom')

        # Second trade should match second entry
        self.assertEqual(backtest.trade_log[1]['entry_shares'], 30)
        self.assertEqual(backtest.trade_log[1]['entry_reason'], 'short_ma_bottom')

        # Verify open_positions is empty
        self.assertEqual(len(backtest.open_positions), 0)

    def test_08_cumulative_pnl(self):
        """Test: Cumulative P&L is calculated correctly across multiple trades"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Trade 1: Entry and exit
        backtest._execute_entry(pd.Timestamp('2023-06-01'), 100.0, 50, 'long_ma_bottom')
        backtest._execute_exit(pd.Timestamp('2023-07-01'), 110.0, reason='peak exit')

        # Trade 2: Entry and exit
        backtest._execute_entry(pd.Timestamp('2023-08-01'), 105.0, 50, 'long_ma_bottom')
        backtest._execute_exit(pd.Timestamp('2023-09-01'), 115.0, reason='peak exit')

        # Verify cumulative P&L
        self.assertEqual(len(backtest.trade_log), 2)

        # First trade cumulative P&L should equal its own P&L
        self.assertAlmostEqual(backtest.trade_log[0]['cumulative_pnl'], backtest.trade_log[0]['pnl_dollar'], places=2)

        # Second trade cumulative P&L should be sum of both
        expected_cumulative = backtest.trade_log[0]['pnl_dollar'] + backtest.trade_log[1]['pnl_dollar']
        self.assertAlmostEqual(backtest.trade_log[1]['cumulative_pnl'], expected_cumulative, places=2)

    def test_09_save_trade_log_csv(self):
        """Test: save_trade_log() creates CSV file with correct structure"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Create some trades
        backtest._execute_entry(pd.Timestamp('2023-06-01'), 100.0, 50, 'long_ma_bottom')
        backtest._execute_exit(pd.Timestamp('2023-07-01'), 110.0, reason='peak exit')

        # Save trade log
        filename = backtest.save_trade_log()

        # Verify file exists
        self.assertTrue(os.path.exists(filename))

        # Read CSV and verify structure
        df = pd.read_csv(filename)

        expected_columns = [
            'trade_id',
            'entry_date',
            'entry_price',
            'entry_shares',
            'entry_cost',
            'entry_reason',
            'exit_date',
            'exit_price',
            'exit_shares',
            'exit_proceeds',
            'exit_reason',
            'holding_days',
            'pnl_dollar',
            'pnl_percent',
            'cumulative_pnl',
        ]

        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Verify data
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['trade_id'], 1)

        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)

    def test_10_save_trade_log_with_custom_filename(self):
        """Test: save_trade_log() accepts custom filename"""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Create a trade
        backtest._execute_entry(pd.Timestamp('2023-06-01'), 100.0, 50, 'long_ma_bottom')
        backtest._execute_exit(pd.Timestamp('2023-07-01'), 110.0, reason='peak exit')

        # Save with custom filename
        custom_filename = 'reports/test_custom_trade_log.csv'
        result_filename = backtest.save_trade_log(filename=custom_filename)

        # Verify correct filename is used
        self.assertEqual(result_filename, custom_filename)
        self.assertTrue(os.path.exists(custom_filename))

        # Cleanup
        if os.path.exists(custom_filename):
            os.remove(custom_filename)

    def test_11_buy_cost_allocation_with_split_exits(self):
        """Test: Split exits allocate buy_cost proportionally to original BUY shares."""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # 1 BUY split into 2 SELLs (equivalent to partial-exit style accounting).
        backtest.trades = [
            {
                'date': pd.Timestamp('2023-01-01'),
                'action': 'BUY',
                'price': 10.0,
                'shares': 100,
                'total_cost': 1000.0,
            },
            {
                'date': pd.Timestamp('2023-01-10'),
                'action': 'SELL',
                'price': 12.0,
                'shares': 50,
                'total_proceeds': 600.0,
            },
            {
                'date': pd.Timestamp('2023-01-20'),
                'action': 'SELL',
                'price': 9.0,
                'shares': 50,
                'total_proceeds': 450.0,
            },
        ]

        trade_pairs = backtest._get_trade_pairs()
        self.assertEqual(len(trade_pairs), 2)
        self.assertAlmostEqual(trade_pairs[0]['buy_cost'], 500.0, places=6)
        self.assertAlmostEqual(trade_pairs[1]['buy_cost'], 500.0, places=6)
        self.assertAlmostEqual(sum(pair['buy_cost'] for pair in trade_pairs), 1000.0, places=6)

        # Win/loss metrics should reflect the two split exits correctly.
        self.assertAlmostEqual(backtest._calculate_win_rate(), 0.5, places=6)
        self.assertAlmostEqual(backtest._calculate_profit_factor(), 2.0, places=6)

    def test_12_open_positions_closed_at_backtest_end(self):
        """Test: Open positions are force-closed with reason 'backtest_end' at end of backtest."""
        backtest = Backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            symbol=self.symbol,
            use_saved_data=True,
            no_show_plot=True,
        )

        # Simulate an entry without exit
        test_date = pd.Timestamp('2023-06-01')
        test_price = 100.0
        test_shares = 50
        backtest._execute_entry(test_date, test_price, test_shares, 'long_ma_bottom')

        # Verify position is open
        self.assertEqual(backtest.current_position, test_shares)
        self.assertEqual(len(backtest.open_positions), 1)

        # Set up minimal price_data for execute_trades to use

        dates = pd.date_range('2023-06-01', '2023-06-10', freq='D')
        n = len(dates)
        price_values = [100.0 + i * 0.5 for i in range(n)]

        backtest.price_data = pd.DataFrame(
            {
                'adjusted_close': price_values,
                'open': price_values,
                'high': [p + 1 for p in price_values],
                'low': [p - 1 for p in price_values],
                'close': price_values,
            },
            index=dates,
        )

        # Simulate backtest end force-close
        final_date = backtest.price_data.index[-1]
        final_price = backtest.price_data['adjusted_close'].iloc[-1]

        if backtest.current_position > 0:
            backtest._execute_exit(final_date, final_price, reason='backtest_end')

        # Verify position is closed
        self.assertEqual(backtest.current_position, 0)
        self.assertEqual(len(backtest.open_positions), 0)

        # Verify trade_log entry has 'backtest_end' reason
        self.assertEqual(len(backtest.trade_log), 1)
        self.assertEqual(backtest.trade_log[0]['exit_reason'], 'backtest_end')

    def test_13_backtest_end_equity_reflects_close_costs(self):
        """Integration test: execute_trades() force-close updates equity curve with slippage/commission."""
        import numpy as np

        from market_breadth import calculate_trend_with_hysteresis

        bt = Backtest(
            start_date='2024-01-01',
            end_date='2024-03-31',
            tv_mode=True,
            no_pyramiding=True,
            use_saved_data=True,
            no_show_plot=True,
            initial_capital=50000,
            slippage=0.001,
            commission=0.001,
            debug=False,
        )
        bt.pivot_len_long = 3
        bt.pivot_len_short = 2

        # Build 60 bars of price/breadth data with a trough near the start
        n = 60
        dates = pd.date_range('2024-01-02', periods=n, freq='B')

        # Breadth: dip then recover — triggers a long MA trough entry
        breadth_vals = np.concatenate(
            [
                np.linspace(0.55, 0.30, 10),  # decline
                np.linspace(0.30, 0.35, 5),  # trough area
                np.linspace(0.35, 0.55, 45),  # recovery (no peak exit)
            ]
        )
        breadth = pd.Series(breadth_vals, index=dates)

        prices = np.linspace(100, 120, n)
        ohlc = pd.DataFrame(
            {
                'adjusted_close': prices,
                'open': prices,
                'high': prices + 1,
                'low': prices - 1,
                'close': prices,
            },
            index=dates,
        )

        # Inject data and run execute_trades()
        bt.price_data = ohlc.copy()
        bt.breadth_index = breadth.copy()
        bt.sp500_data = pd.DataFrame()
        bt.short_ma_line = breadth.ewm(span=bt.short_ma, adjust=False).mean()
        bt.long_ma_line = breadth.ewm(span=bt.long_ma, adjust=False).mean()
        bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)
        bt.short_ma_bottoms = []
        bt.long_ma_bottoms = []
        bt.peaks = []
        bt._precompute_tv_signals()

        bt.execute_trades()

        # If a position was opened and force-closed at backtest_end
        backtest_end_trades = [t for t in bt.trade_log if t['exit_reason'] == 'backtest_end']
        if backtest_end_trades:
            # Final equity in curve should match current_capital (no open position)
            final_equity = bt.equity_curve[-1]['equity']
            self.assertEqual(bt.current_position, 0)
            self.assertAlmostEqual(final_equity, bt.current_capital, places=2)
        else:
            # No position opened — equity should still be initial_capital
            final_equity = bt.equity_curve[-1]['equity']
            self.assertAlmostEqual(final_equity, bt.current_capital, delta=1.0)

    def test_14_equity_curve_matches_capital_plus_position(self):
        """Every equity_curve entry equals current_capital + position * price."""
        import numpy as np

        from market_breadth import calculate_trend_with_hysteresis

        bt = Backtest(
            start_date='2024-01-01',
            end_date='2024-03-31',
            tv_mode=True,
            no_pyramiding=True,
            use_saved_data=True,
            no_show_plot=True,
            initial_capital=50000,
            slippage=0.001,
            commission=0.001,
            debug=True,  # Enable equity invariant check
        )
        bt.pivot_len_long = 3
        bt.pivot_len_short = 2

        # Build 60 bars with a trough then peak to trigger entry and exit
        n = 60
        dates = pd.date_range('2024-01-02', periods=n, freq='B')

        breadth_vals = np.concatenate(
            [
                np.linspace(0.55, 0.30, 10),
                np.linspace(0.30, 0.35, 5),
                np.linspace(0.35, 0.75, 25),
                np.linspace(0.75, 0.65, 10),
                np.linspace(0.65, 0.60, 10),
            ]
        )
        breadth = pd.Series(breadth_vals, index=dates)

        prices = np.linspace(100, 120, n)
        ohlc = pd.DataFrame(
            {
                'adjusted_close': prices,
                'open': prices,
                'high': prices + 1,
                'low': prices - 1,
                'close': prices,
            },
            index=dates,
        )

        bt.price_data = ohlc.copy()
        bt.breadth_index = breadth.copy()
        bt.sp500_data = pd.DataFrame()
        bt.short_ma_line = breadth.ewm(span=bt.short_ma, adjust=False).mean()
        bt.long_ma_line = breadth.ewm(span=bt.long_ma, adjust=False).mean()
        bt.long_ma_trend = pd.Series(calculate_trend_with_hysteresis(bt.long_ma_line), index=bt.long_ma_line.index)
        bt.short_ma_bottoms = []
        bt.long_ma_bottoms = []
        bt.peaks = []
        bt._precompute_tv_signals()

        # execute_trades() with debug=True will call _assert_equity_invariant
        # on every bar — if there's a mismatch, AssertionError is raised
        bt.execute_trades()

        # Additionally verify all equity_curve entries manually
        for entry in bt.equity_curve:
            eq_val = entry['equity']
            self.assertFalse(np.isnan(eq_val), f'NaN equity at {entry["date"]}')
            self.assertGreater(eq_val, 0, f'Non-positive equity at {entry["date"]}')

        # Final entry should match cash + position
        final_equity = bt.equity_curve[-1]['equity']
        final_price = bt.price_data['adjusted_close'].iloc[-1]
        expected_final = bt.current_capital + (bt.current_position * final_price)
        self.assertAlmostEqual(final_equity, expected_final, places=2)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
