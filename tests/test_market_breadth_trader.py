import logging
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Add parent directory to path to import market_breadth_trade
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from trade.run_market_breadth_trade import MarketBreadthTrader

# Test logging configuration
logging.basicConfig(level=logging.INFO)


class TestMarketBreadthTrader(unittest.TestCase):
    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def setUp(self, mock_initialize_alpaca):
        """Test setup"""
        self.mock_api = Mock()
        mock_initialize_alpaca.return_value = self.mock_api
        self.trader = MarketBreadthTrader(
            short_ma=8, long_ma=200, initial_capital=50000, symbol='SSO', use_saved_data=True
        )

        # Mock data setup
        self.mock_bar = Mock()
        self.mock_bar.c = 50.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        self.mock_position = Mock()
        self.mock_position.qty = 100
        self.mock_api.get_position.return_value = self.mock_position

        self.mock_clock = Mock()
        self.mock_clock.is_open = True
        self.mock_api.get_clock.return_value = self.mock_clock

        # Calendar mock
        calendar_day = Mock()
        calendar_day.close = '16:00'
        self.mock_api.get_calendar.return_value = [calendar_day]

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.trader.symbol, 'SSO')
        self.assertEqual(self.trader.short_ma, 8)
        self.assertEqual(self.trader.long_ma, 200)
        self.assertEqual(self.trader.initial_capital, 50000)
        self.assertEqual(self.trader.current_position, 0)

    def test_is_market_open(self):
        """Test market open status"""
        # When market is open
        self.assertTrue(self.trader.is_market_open())

        # When market is closed
        self.mock_clock.is_open = False
        self.assertFalse(self.trader.is_market_open())

    def test_get_current_position(self):
        """Test current position retrieval"""
        position = self.trader.get_current_position()
        self.assertEqual(position, 100)

        # When position doesn't exist
        self.mock_api.get_position.side_effect = Exception('Position not found')
        position = self.trader.get_current_position()
        self.assertEqual(position, 0)

    def test_get_current_price(self):
        """Test current price retrieval"""
        price = self.trader.get_current_price()
        self.assertEqual(price, 50.0)

        # When price cannot be retrieved
        self.mock_api.get_latest_bar.return_value = None
        price = self.trader.get_current_price()
        self.assertIsNone(price)

    def test_execute_buy(self):
        """Test buy order execution"""
        # Success case
        mock_order = Mock()
        self.mock_api.submit_order.return_value = mock_order
        order = self.trader.execute_buy(100, 'test buy')
        self.assertIsNotNone(order)

        # Failure case
        self.mock_api.submit_order.side_effect = Exception('Order failed')
        order = self.trader.execute_buy(100, 'test buy')
        self.assertIsNone(order)

    def test_execute_sell(self):
        """Test sell order execution"""
        # Success case
        mock_order = Mock()
        self.mock_api.submit_order.return_value = mock_order
        order = self.trader.execute_sell(100, 'test sell')
        self.assertIsNotNone(order)

        # Failure case
        self.mock_api.submit_order.side_effect = Exception('Order failed')
        order = self.trader.execute_sell(100, 'test sell')
        self.assertIsNone(order)

    def test_is_closing_time_range(self):
        """Test if current time is within specified minutes before market close"""
        # Mock setup
        mock_api = MagicMock()
        mock_calendar = Mock()
        mock_calendar.close = '16:00'
        mock_api.get_calendar.return_value = [mock_calendar]

        # Create MarketBreadthTrader instance with mocked API initialization
        with patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca', return_value=mock_api):
            trader = MarketBreadthTrader()

        # Case: 1 hour before market close
        with patch('trade.run_market_breadth_trade.datetime') as mock_datetime:
            # Explicitly set timezone
            mock_datetime.now.return_value = datetime(2024, 1, 1, 15, 0, tzinfo=ZoneInfo('US/Eastern'))
            # Mock datetime.combine method
            mock_datetime.combine = lambda date, time, tzinfo=None: datetime(
                2024, 1, 1, 16, 0, tzinfo=ZoneInfo('US/Eastern')
            )
            # Mock datetime.strptime method
            mock_datetime.strptime = lambda date_str, format_str: datetime.strptime(date_str, format_str)
            self.assertTrue(trader.is_closing_time_range(60))

        # Case: Not 1 hour before market close
        with patch('trade.run_market_breadth_trade.datetime') as mock_datetime:
            # Explicitly set timezone
            mock_datetime.now.return_value = datetime(2024, 1, 1, 14, 0, tzinfo=ZoneInfo('US/Eastern'))
            # Mock datetime.combine method
            mock_datetime.combine = lambda date, time, tzinfo=None: datetime(
                2024, 1, 1, 16, 0, tzinfo=ZoneInfo('US/Eastern')
            )
            # Mock datetime.strptime method
            mock_datetime.strptime = lambda date_str, format_str: datetime.strptime(date_str, format_str)
            self.assertFalse(trader.is_closing_time_range(60))

    @patch('trade.run_market_breadth_trade.get_sp500_tickers_from_fmp')
    @patch('trade.run_market_breadth_trade.get_multiple_stock_data')
    def test_analyze_market(self, mock_get_data, mock_get_tickers):
        """Test market analysis"""
        # Prepare mock data
        mock_get_tickers.return_value = ['AAPL', 'MSFT', 'GOOGL']

        # Create test stock price data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        data = pd.DataFrame(np.random.randn(len(dates), 3), index=dates, columns=['AAPL', 'MSFT', 'GOOGL'])
        mock_get_data.return_value = data

        # Execute analysis
        self.trader.analyze_market()

        # Verify results
        self.assertIsNotNone(self.trader.breadth_index)
        self.assertIsNotNone(self.trader.short_ma_line)
        self.assertIsNotNone(self.trader.long_ma_line)

    def test_calculate_shares(self):
        """Test share calculation includes both slippage and commission"""
        # Normal case
        shares = self.trader._calculate_shares(10000, 50.0)
        expected_shares = int(10000 / (50.0 * (1 + self.trader.slippage + self.trader.commission)))
        self.assertEqual(shares, expected_shares)

        # Case with fractional shares
        shares = self.trader._calculate_shares(10000, 33.33)
        self.assertEqual(shares, int(10000 / (33.33 * (1 + self.trader.slippage + self.trader.commission))))

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_detect_signals_populates_peaks_and_troughs(self, mock_init_alpaca):
        """Test that _detect_signals() populates peaks and troughs from synthetic data"""
        mock_init_alpaca.return_value = Mock()

        trader = MarketBreadthTrader(short_ma=5, long_ma=20, symbol='SSO')

        # Generate synthetic breadth data: sin wave with 200 business days
        np.random.seed(42)
        n = 200
        dates = pd.bdate_range(start='2024-01-01', periods=n)
        t = np.linspace(0, 4 * np.pi, n)
        breadth = 0.5 + 0.35 * np.sin(t)
        breadth = np.clip(breadth, 0.01, 0.99)

        trader.breadth_index = pd.Series(breadth, index=dates)
        trader.short_ma_line = trader.breadth_index.ewm(span=5, adjust=False).mean()
        trader.long_ma_line = trader.breadth_index.ewm(span=20, adjust=False).mean()

        trader._detect_signals()

        # Should detect at least one peak and one trough
        self.assertGreater(len(trader.peaks), 0, 'Should detect at least one peak')
        has_troughs = len(trader.long_ma_bottoms) > 0 or len(trader.short_ma_bottoms) > 0
        self.assertTrue(has_troughs, 'Should detect at least one trough (long or short)')

        # Signal dictionaries should be populated
        self.assertIsInstance(trader._tv_peak_signals, dict)
        self.assertIsInstance(trader._tv_long_trough_signals, dict)
        self.assertIsInstance(trader._tv_short_trough_signals, dict)

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_detect_signals_matches_backtest_precompute(self, mock_init_alpaca):
        """Test that live _detect_signals() matches backtest _precompute_tv_signals()"""
        from backtest.backtest import detect_pivot_high, detect_pivot_low

        mock_init_alpaca.return_value = Mock()

        # Generate same synthetic data for both
        np.random.seed(42)
        n = 200
        dates = pd.bdate_range(start='2024-01-01', periods=n)
        t = np.linspace(0, 4 * np.pi, n)
        breadth = 0.5 + 0.35 * np.sin(t)
        breadth = np.clip(breadth, 0.01, 0.99)

        breadth_series = pd.Series(breadth, index=dates)
        short_ma_line = breadth_series.ewm(span=5, adjust=False).mean()
        long_ma_line = breadth_series.ewm(span=20, adjust=False).mean()

        # --- Live trader side ---
        trader = MarketBreadthTrader(short_ma=5, long_ma=20, symbol='SSO')
        trader.breadth_index = breadth_series
        trader.short_ma_line = short_ma_line
        trader.long_ma_line = long_ma_line
        trader._detect_signals()

        # --- Backtest side (replicate _precompute_tv_signals logic) ---
        bt_peak_signals = {}
        for confirm_date, pivot_date, val in detect_pivot_high(
            long_ma_line, trader.pivot_len_long, trader.prom_thresh_long, trader.peak_level
        ):
            if confirm_date not in bt_peak_signals:
                bt_peak_signals[confirm_date] = (pivot_date, val)

        bt_long_trough_signals = {}
        for confirm_date, pivot_date, val in detect_pivot_low(
            long_ma_line, trader.pivot_len_long, trader.prom_thresh_long
        ):
            if val < trader.trough_level_long:
                if confirm_date not in bt_long_trough_signals:
                    bt_long_trough_signals[confirm_date] = (pivot_date, val)

        bt_short_trough_signals = {}
        for confirm_date, pivot_date, val in detect_pivot_low(
            short_ma_line, trader.pivot_len_short, trader.prom_thresh_short
        ):
            confirm_loc = breadth_series.index.get_loc(confirm_date)
            start_loc = max(0, confirm_loc - 19)
            recent_min = breadth_series.iloc[start_loc : confirm_loc + 1].min()
            if recent_min <= trader.trough_level_short:
                if confirm_date not in bt_short_trough_signals:
                    bt_short_trough_signals[confirm_date] = (pivot_date, val)

        # Compare signal dictionaries
        self.assertEqual(trader._tv_peak_signals, bt_peak_signals, 'Peak signals mismatch')
        self.assertEqual(trader._tv_long_trough_signals, bt_long_trough_signals, 'Long trough signals mismatch')
        self.assertEqual(trader._tv_short_trough_signals, bt_short_trough_signals, 'Short trough signals mismatch')

    def test_stop_loss_triggers(self):
        """Test that stop loss triggers sell when price drops below threshold"""
        # Set up position with entry price
        self.trader.current_position = 100
        self.trader.entry_prices = [50.0]

        # Set up mock: price below stop loss (50 * (1 - 0.08) = 46.0)
        self.mock_bar.c = 45.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Mock _sync_position_from_broker to preserve our test state
        self.trader._sync_position_from_broker = Mock()
        self.trader._sync_position_from_broker.side_effect = lambda: None

        # Set up mock for sell order
        mock_order = Mock()
        mock_order.id = 'order-stop'
        self.mock_api.submit_order.return_value = mock_order

        # Mock get_order for _wait_for_fill
        filled_order = Mock()
        filled_order.status = 'filled'
        filled_order.filled_qty = 100
        filled_order.filled_avg_price = '45.0'
        self.mock_api.get_order.return_value = filled_order

        self.trader.check_signals_and_trade()

        # Verify sell was called with stop loss reason
        self.mock_api.submit_order.assert_called_once_with(
            symbol='SSO', qty=100, side='sell', type='market', time_in_force='day'
        )
        self.assertEqual(self.trader.entry_prices, [])
        self.assertEqual(self.trader.current_position, 0)

    def test_stop_loss_recovery_from_broker(self):
        """Test that stop loss works after process restart (entry_prices recovered from broker)"""
        # Simulate process restart: entry_prices is empty
        self.trader.entry_prices = []
        self.trader.current_position = 0

        # Set up broker position with avg_entry_price
        self.mock_position.qty = 100
        self.mock_position.avg_entry_price = '50.0'
        self.mock_api.get_position.return_value = self.mock_position

        # Price below stop loss (50 * 0.92 = 46.0)
        self.mock_bar.c = 45.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Set up mock for sell order
        mock_order = Mock()
        mock_order.id = 'order-stop-recovery'
        self.mock_api.submit_order.return_value = mock_order

        # Mock get_order for _wait_for_fill
        filled_order = Mock()
        filled_order.status = 'filled'
        filled_order.filled_qty = 100
        filled_order.filled_avg_price = '45.0'
        self.mock_api.get_order.return_value = filled_order

        self.trader.check_signals_and_trade()

        # Verify entry_prices was recovered from broker
        # and stop loss triggered sell
        self.mock_api.submit_order.assert_called_once_with(
            symbol='SSO', qty=100, side='sell', type='market', time_in_force='day'
        )
        self.assertEqual(self.trader.current_position, 0)
        self.assertEqual(self.trader.entry_prices, [])

    def test_no_pyramiding_blocks_second_entry(self):
        """Test that no_pyramiding=True blocks entry when position already exists"""
        # Set up existing position
        self.trader.current_position = 100
        self.trader.entry_prices = [50.0]
        self.trader.no_pyramiding = True

        # Mock _sync_position_from_broker to preserve our test state
        self.trader._sync_position_from_broker = Mock()
        self.trader._sync_position_from_broker.side_effect = lambda: None

        # Set up price above stop loss so stop loss doesn't trigger
        self.mock_bar.c = 55.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Set up a long_ma_bottom signal for today
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        self.trader.long_ma_bottoms = [today]
        self.trader.short_ma_bottoms = []
        self.trader.peaks = []

        self.trader.check_signals_and_trade()

        # Verify no buy order was submitted (pyramiding blocked)
        buy_calls = [call for call in self.mock_api.submit_order.call_args_list if call[1].get('side') == 'buy']
        self.assertEqual(len(buy_calls), 0, 'Should not place buy order when pyramiding is disabled')

    def test_allow_pyramiding_permits_second_entry(self):
        """Test that no_pyramiding=False allows additional entry with existing position"""
        # Set up existing position
        self.trader.current_position = 100
        self.trader.entry_prices = [50.0]
        self.trader.no_pyramiding = False

        # Mock _sync_position_from_broker to preserve our test state
        self.trader._sync_position_from_broker = Mock()
        self.trader._sync_position_from_broker.side_effect = lambda: None

        # Set up price above stop loss so stop loss doesn't trigger
        self.mock_bar.c = 55.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Set up a long_ma_bottom signal for today
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        self.trader.long_ma_bottoms = [today]
        self.trader.short_ma_bottoms = []
        self.trader.peaks = []

        # Mock account for buying power
        mock_account = Mock()
        mock_account.cash = '10000.0'
        self.mock_api.get_account.return_value = mock_account

        # Set up mock for buy order
        mock_order = Mock()
        mock_order.id = 'order-buy'
        self.mock_api.submit_order.return_value = mock_order

        # Mock get_order for _wait_for_fill
        filled_order = Mock()
        filled_order.status = 'filled'
        filled_order.filled_qty = 181
        filled_order.filled_avg_price = '55.0'
        self.mock_api.get_order.return_value = filled_order

        self.trader.check_signals_and_trade()

        # Verify buy order was submitted (pyramiding allowed)
        buy_calls = [call for call in self.mock_api.submit_order.call_args_list if call[1].get('side') == 'buy']
        self.assertEqual(len(buy_calls), 1, 'Should place buy order when pyramiding is allowed')
        self.assertEqual(len(self.trader.entry_prices), 2, 'Should append new entry price')

    def test_sync_clears_entry_prices_when_no_position(self):
        """Test that _sync_position_from_broker clears entry_prices when broker has no position"""
        # Simulate stale entry_prices from a previous session
        self.trader.entry_prices = [50.0, 52.0]

        # Broker returns 404 (no position)
        self.mock_api.get_position.side_effect = Exception('position does not exist')

        self.trader._sync_position_from_broker()

        self.assertEqual(self.trader.current_position, 0)
        self.assertEqual(self.trader.entry_prices, [], 'entry_prices should be cleared when no position')

    def test_sync_raises_on_api_error(self):
        """Test that _sync_position_from_broker raises on transient API errors"""
        # Simulate a network/API error (not a 404)
        self.mock_api.get_position.side_effect = Exception('connection timeout')

        with self.assertRaises(Exception) as ctx:
            self.trader._sync_position_from_broker()

        self.assertIn('connection timeout', str(ctx.exception))

    def test_missing_api_key_raises(self):
        """C-7: Missing API key raises EnvironmentError at startup"""
        with (
            patch('trade.run_market_breadth_trade.ALPACA_API_KEY', None),
            patch('trade.run_market_breadth_trade.ALPACA_SECRET_KEY', 'some_secret'),
            patch('trade.run_market_breadth_trade.ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            patch.dict('sys.modules', {'alpaca_trade_api': MagicMock()}),
        ):
            with self.assertRaises(OSError) as ctx:
                MarketBreadthTrader()
            self.assertIn('ALPACA_API_KEY', str(ctx.exception))

    def test_live_url_requires_confirmation(self):
        """C-6: Live trading URL requires ALPACA_LIVE_CONFIRMED=true"""
        with (
            patch('trade.run_market_breadth_trade.ALPACA_API_KEY', 'test_key'),
            patch('trade.run_market_breadth_trade.ALPACA_SECRET_KEY', 'test_secret'),
            patch('trade.run_market_breadth_trade.ALPACA_BASE_URL', 'https://api.alpaca.markets'),
            patch.dict('sys.modules', {'alpaca_trade_api': MagicMock()}),
            patch.dict(os.environ, {'ALPACA_LIVE_CONFIRMED': ''}, clear=False),
        ):
            with self.assertRaises(OSError) as ctx:
                MarketBreadthTrader()
            self.assertIn('Live trading URL detected', str(ctx.exception))

    def test_wait_for_fill_success(self):
        """C-1: _wait_for_fill returns filled order on success"""
        mock_order = Mock()
        mock_order.id = 'order-123'

        filled = Mock()
        filled.status = 'filled'
        filled.filled_qty = 100
        filled.filled_avg_price = '55.0'
        self.mock_api.get_order.return_value = filled

        result = self.trader._wait_for_fill(mock_order)
        self.assertEqual(result, filled)
        self.mock_api.get_order.assert_called_with('order-123')

    def test_wait_for_fill_canceled(self):
        """C-1: _wait_for_fill returns None on canceled order"""
        mock_order = Mock()
        mock_order.id = 'order-456'

        canceled = Mock()
        canceled.status = 'canceled'
        self.mock_api.get_order.return_value = canceled

        result = self.trader._wait_for_fill(mock_order)
        self.assertIsNone(result)

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_wait_for_fill_testmode(self, mock_init):
        """C-1: _wait_for_fill returns True in testmode without polling"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-01')
        result = trader._wait_for_fill(Mock())
        self.assertTrue(result)

    def test_signal_lookback_catches_missed_day(self):
        """M-1: _find_recent_signal catches signal from 1-3 days ago"""
        signal_dates = [pd.Timestamp('2024-01-10'), pd.Timestamp('2024-01-15')]

        # Exact match
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-10'))
        self.assertEqual(result, pd.Timestamp('2024-01-10'))

        # 2 days later (within lookback)
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-12'))
        self.assertEqual(result, pd.Timestamp('2024-01-10'))

        # 4 days later (outside default 3-day lookback)
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-14'))
        self.assertIsNone(result)

        # Returns most recent signal when multiple in range
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-16'))
        self.assertEqual(result, pd.Timestamp('2024-01-15'))

    def test_exception_with_position_logs_critical(self):
        """C-5: Exception with open position logs CRITICAL alert"""
        self.trader.testmode = True
        self.trader.test_date = '2024-01-02'
        self.trader.current_position = 100

        # Make is_closing_time_range return True, then analyze_market raise
        self.trader.is_closing_time_range = Mock(return_value=True)
        self.trader.analyze_market = Mock(side_effect=RuntimeError('API down'))

        with self.assertLogs('market_breadth_trade', level='CRITICAL') as cm:
            self.trader.run()

        self.assertTrue(any('ALERT' in msg and 'open position' in msg for msg in cm.output))


if __name__ == '__main__':
    unittest.main()
