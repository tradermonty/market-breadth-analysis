import json
import logging
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trade.run_market_breadth_trade import MarketBreadthTrader, _now_et

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

        # MJ-003: Reset acted signals to prevent cross-test contamination
        self.trader._acted_signals = set()

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
        self.trader.entry_lots = [{'price': 50.0, 'shares': self.trader.current_position}]

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
        self.trader.entry_lots = [{'price': 50.0, 'shares': self.trader.current_position}]
        self.trader.no_pyramiding = True

        # Mock _sync_position_from_broker to preserve our test state
        self.trader._sync_position_from_broker = Mock()
        self.trader._sync_position_from_broker.side_effect = lambda: None

        # Set up price above stop loss so stop loss doesn't trigger
        self.mock_bar.c = 55.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Set up a long_ma_bottom signal for today
        today = pd.to_datetime(_now_et().strftime('%Y-%m-%d'))
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
        self.trader.entry_lots = [{'price': 50.0, 'shares': self.trader.current_position}]
        self.trader.no_pyramiding = False

        # Mock _sync_position_from_broker to preserve our test state
        self.trader._sync_position_from_broker = Mock()
        self.trader._sync_position_from_broker.side_effect = lambda: None

        # Set up price above stop loss so stop loss doesn't trigger
        self.mock_bar.c = 55.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Set up a long_ma_bottom signal for today
        today = pd.to_datetime(_now_et().strftime('%Y-%m-%d'))
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
        self.trader.entry_lots = [{'price': 50.0, 'shares': 50}, {'price': 52.0, 'shares': 50}]

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
        """C-1: _wait_for_fill returns SimpleNamespace in testmode without polling"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-01')
        result = trader._wait_for_fill(Mock())
        self.assertIsNotNone(result)
        self.assertIsNone(result.filled_avg_price)
        self.assertIsNone(result.filled_qty)
        self.assertEqual(result.status, 'filled')

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_testmode_buy_uses_current_price(self, mock_init):
        """Testmode buy appends current_price when filled_avg_price is None"""
        mock_api = Mock()
        mock_init.return_value = mock_api

        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-01', symbol='SSO')
        trader.current_position = 0
        trader.entry_prices = []
        trader.no_pyramiding = True

        # Mock _sync_position_from_broker and price
        trader._sync_position_from_broker = Mock()
        mock_bar = Mock()
        mock_bar.c = 55.0
        mock_api.get_latest_bar.return_value = mock_bar

        # Set up signal for today
        today = pd.to_datetime('2024-01-01')
        trader.long_ma_bottoms = [today]
        trader.short_ma_bottoms = []
        trader.peaks = []

        # Mock account
        mock_account = Mock()
        mock_account.cash = '10000.0'
        mock_api.get_account.return_value = mock_account

        trader.check_signals_and_trade()

        # In testmode, filled_avg_price is None, so current_price (55.0) should be used
        self.assertEqual(trader.entry_prices, [55.0])

    def test_signal_lookback_catches_missed_day(self):
        """M-1: _find_recent_signal catches signal from 1-3 days ago"""
        signal_dates = [pd.Timestamp('2024-01-10'), pd.Timestamp('2024-01-15')]

        # Exact match
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-10'))
        self.assertEqual(result, pd.Timestamp('2024-01-10'))

        # 2 days later (within lookback)
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-12'))
        self.assertEqual(result, pd.Timestamp('2024-01-10'))

        # 4 days later (within default 5-day lookback for weekend coverage)
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-14'))
        self.assertEqual(result, pd.Timestamp('2024-01-10'))

        # 1 day after second signal (within lookback, returns most recent)
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-16'))
        self.assertEqual(result, pd.Timestamp('2024-01-15'))

        # 6 days after second signal (outside default 5-day lookback)
        result = self.trader._find_recent_signal(signal_dates, pd.Timestamp('2024-01-21'))
        self.assertIsNone(result)

        # Thursday signal → Monday check (3 calendar days, within 5-day lookback)
        thursday_signal = [pd.Timestamp('2024-01-11')]  # Thursday
        monday = pd.Timestamp('2024-01-15')  # Monday
        result = self.trader._find_recent_signal(thursday_signal, monday)
        self.assertEqual(result, pd.Timestamp('2024-01-11'))

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

    @patch('trade.run_market_breadth_trade.time')
    def test_wait_for_fill_timeout_cancels_order(self, mock_time):
        """_wait_for_fill cancels order on timeout and returns None"""
        mock_order = Mock()
        mock_order.id = 'order-timeout'

        # Simulate immediate timeout: time.time() returns past-deadline values
        mock_time.time.side_effect = [0, 100, 200]
        mock_time.sleep = Mock()

        # Mock cancel_order and final get_order (canceled after cancel)
        canceled = Mock()
        canceled.status = 'canceled'
        self.mock_api.get_order.return_value = canceled
        self.mock_api.cancel_order = Mock()

        result = self.trader._wait_for_fill(mock_order, timeout_seconds=60)

        self.assertIsNone(result)
        self.mock_api.cancel_order.assert_called_once_with('order-timeout')

    def test_buy_order_not_filled_skips_entry_price(self):
        """Buy order where _wait_for_fill returns None does not append entry_price"""
        self.trader.current_position = 0
        self.trader.entry_prices = []
        self.trader.no_pyramiding = True

        # Mock _sync_position_from_broker to preserve test state
        self.trader._sync_position_from_broker = Mock()

        # Price above stop loss
        self.mock_bar.c = 55.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Set up a long_ma_bottom signal for today
        today = pd.to_datetime(_now_et().strftime('%Y-%m-%d'))
        self.trader.long_ma_bottoms = [today]
        self.trader.short_ma_bottoms = []
        self.trader.peaks = []

        # Mock account
        mock_account = Mock()
        mock_account.cash = '10000.0'
        self.mock_api.get_account.return_value = mock_account

        # submit_order succeeds
        mock_order = Mock()
        mock_order.id = 'order-unfilled'
        self.mock_api.submit_order.return_value = mock_order

        # _wait_for_fill returns None (order not filled)
        self.trader._wait_for_fill = Mock(return_value=None)

        self.trader.check_signals_and_trade()

        # entry_prices should remain empty since fill failed
        self.assertEqual(self.trader.entry_prices, [], 'entry_prices should not be updated when fill fails')

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_shutdown_flag_breaks_loop(self, mock_init):
        """Shutdown flag causes run() loop to exit immediately"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-02')
        trader._shutdown_requested = True

        # run() should exit immediately without calling analyze_market
        trader.analyze_market = Mock()
        trader.run()
        trader.analyze_market.assert_not_called()

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_shutdown_with_position_logs_warning(self, mock_init):
        """Shutdown with open position logs warning about manual check"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-02')
        trader._shutdown_requested = True
        trader.current_position = 100

        with self.assertLogs('market_breadth_trade', level='WARNING') as cm:
            trader.run()

        self.assertTrue(any('Open position' in msg and '100 shares' in msg for msg in cm.output))

    @patch('trade.run_market_breadth_trade.time')
    def test_wait_for_fill_partial_then_filled(self, mock_time):
        """Partial fill followed by full fill returns the filled order"""
        mock_order = Mock()
        mock_order.id = 'order-partial'

        # time.time returns values within deadline
        mock_time.time.side_effect = [0, 10, 20, 30]
        mock_time.sleep = Mock()

        partial = Mock()
        partial.status = 'partially_filled'
        partial.filled_qty = 50
        partial.qty = 100

        filled = Mock()
        filled.status = 'filled'
        filled.filled_qty = 100
        filled.filled_avg_price = '55.0'

        self.mock_api.get_order.side_effect = [partial, filled]

        result = self.trader._wait_for_fill(mock_order, timeout_seconds=60)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, 'filled')

    @patch('trade.run_market_breadth_trade.time')
    def test_wait_for_fill_partial_on_timeout(self, mock_time):
        """Partial fill on timeout returns the partial order after cancel"""
        mock_order = Mock()
        mock_order.id = 'order-partial-timeout'

        # Immediate timeout
        mock_time.time.side_effect = [0, 100]
        mock_time.sleep = Mock()

        # After cancel, final check shows partial fill
        final = Mock()
        final.status = 'partially_filled'
        final.filled_qty = '30'
        final.filled_avg_price = '55.0'
        self.mock_api.get_order.return_value = final
        self.mock_api.cancel_order = Mock()

        result = self.trader._wait_for_fill(mock_order, timeout_seconds=60)
        self.assertIsNotNone(result)
        self.assertEqual(result.filled_qty, '30')
        self.mock_api.cancel_order.assert_called_once_with('order-partial-timeout')

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_entry_prices_persisted_to_disk(self, mock_init):
        """Entry prices are saved to JSON file after buy"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-01', symbol='TEST')

        # Use temp directory to avoid polluting project
        tmp_dir = tempfile.mkdtemp()
        trader._entry_prices_path = lambda: os.path.join(tmp_dir, 'entry_prices_TEST.json')

        trader.entry_lots = [{'price': 55.0, 'shares': 100}, {'price': 60.0, 'shares': 50}]
        trader.entry_prices = [55.0, 60.0]
        trader._save_entry_prices()

        # Verify file exists and contents are correct (MJ-002: new format)
        path = trader._entry_prices_path()
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            saved = json.load(f)
        self.assertEqual(saved, {'lots': [{'price': 55.0, 'shares': 100}, {'price': 60.0, 'shares': 50}]})

        # Cleanup
        os.remove(path)
        os.rmdir(tmp_dir)

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_entry_prices_recovered_from_disk(self, mock_init):
        """Entry lots are recovered from disk when format matches broker qty (MJ-002)"""
        mock_api = Mock()
        mock_init.return_value = mock_api

        trader = MarketBreadthTrader(symbol='TEST')

        # Write entry lots to disk in new format
        tmp_dir = tempfile.mkdtemp()
        entry_path = os.path.join(tmp_dir, 'entry_prices_TEST.json')
        trader._entry_prices_path = lambda: entry_path
        new_format = {'lots': [{'price': 55.0, 'shares': 60}, {'price': 60.0, 'shares': 40}]}
        with open(entry_path, 'w') as f:
            json.dump(new_format, f)

        # Simulate broker position without local entry_lots
        mock_position = Mock()
        mock_position.qty = 100  # 60 + 40 = 100 matches
        mock_position.avg_entry_price = '57.5'
        mock_api.get_position.return_value = mock_position

        trader.entry_prices = []
        trader.entry_lots = []
        trader._sync_position_from_broker()

        # Should recover lots from disk (2 lots totaling 100 shares)
        self.assertEqual(len(trader.entry_lots), 2)
        self.assertEqual(trader.entry_lots[0]['price'], 55.0)
        self.assertEqual(sum(lot['shares'] for lot in trader.entry_lots), 100)
        self.assertEqual(trader.entry_prices, [55.0, 60.0])
        self.assertEqual(trader.current_position, 100)

        # Cleanup
        os.remove(entry_path)
        os.rmdir(tmp_dir)

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_entry_prices_cleared_on_full_exit(self, mock_init):
        """Entry prices file is deleted on full position exit"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-01', symbol='TEST')

        # Create a temp entry prices file
        tmp_dir = tempfile.mkdtemp()
        entry_path = os.path.join(tmp_dir, 'entry_prices_TEST.json')
        trader._entry_prices_path = lambda: entry_path
        with open(entry_path, 'w') as f:
            json.dump([55.0], f)

        self.assertTrue(os.path.exists(entry_path))

        trader._clear_entry_prices_file()

        self.assertFalse(os.path.exists(entry_path))

        # Cleanup
        os.rmdir(tmp_dir)

    def test_stop_loss_partial_fill(self):
        """Partial fill on stop loss correctly updates remaining position"""
        self.trader.current_position = 100
        self.trader.entry_prices = [50.0]
        self.trader.entry_lots = [{'price': 50.0, 'shares': self.trader.current_position}]

        # Price below stop loss
        self.mock_bar.c = 45.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Mock _sync_position_from_broker to preserve test state
        self.trader._sync_position_from_broker = Mock()

        # Set up mock for sell order
        mock_order = Mock()
        mock_order.id = 'order-partial-stop'
        self.mock_api.submit_order.return_value = mock_order

        # _wait_for_fill returns partial fill (60 of 100)
        partial_fill = Mock()
        partial_fill.filled_qty = 60
        partial_fill.filled_avg_price = '45.0'
        self.trader._wait_for_fill = Mock(return_value=partial_fill)

        # Mock _save_entry_prices and _clear_entry_prices_file
        self.trader._save_entry_prices = Mock()
        self.trader._clear_entry_prices_file = Mock()

        self.trader.check_signals_and_trade()

        # Position should be reduced, not zeroed
        self.assertEqual(self.trader.current_position, 40)
        self.trader._save_entry_prices.assert_called_once()

    def test_buy_partial_fill_updates_position(self):
        """Partial buy fill correctly updates current_position with filled_qty"""
        self.trader.current_position = 0
        self.trader.entry_prices = []
        self.trader.no_pyramiding = True

        # Mock _sync_position_from_broker to preserve test state
        self.trader._sync_position_from_broker = Mock()

        # Price above stop loss
        self.mock_bar.c = 55.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Set up a long_ma_bottom signal for today
        today = pd.to_datetime(_now_et().strftime('%Y-%m-%d'))
        self.trader.long_ma_bottoms = [today]
        self.trader.short_ma_bottoms = []
        self.trader.peaks = []

        # Mock account
        mock_account = Mock()
        mock_account.cash = '10000.0'
        self.mock_api.get_account.return_value = mock_account

        # submit_order succeeds
        mock_order = Mock()
        mock_order.id = 'order-partial-buy'
        self.mock_api.submit_order.return_value = mock_order

        # _wait_for_fill returns partial fill (100 of 181 requested)
        partial_fill = Mock()
        partial_fill.filled_qty = 100
        partial_fill.filled_avg_price = '55.0'
        self.trader._wait_for_fill = Mock(return_value=partial_fill)

        # Mock persistence
        self.trader._save_entry_prices = Mock()

        self.trader.check_signals_and_trade()

        # current_position should reflect actual filled qty, not requested shares
        self.assertEqual(self.trader.current_position, 100)
        self.assertEqual(self.trader.entry_prices, [55.0])
        self.trader._save_entry_prices.assert_called_once()

    def test_buy_full_fill_updates_position(self):
        """Full buy fill correctly updates current_position"""
        self.trader.current_position = 0
        self.trader.entry_prices = []
        self.trader.no_pyramiding = True

        # Mock _sync_position_from_broker to preserve test state
        self.trader._sync_position_from_broker = Mock()

        # Price above stop loss
        self.mock_bar.c = 55.0
        self.mock_api.get_latest_bar.return_value = self.mock_bar

        # Set up a long_ma_bottom signal for today
        today = pd.to_datetime(_now_et().strftime('%Y-%m-%d'))
        self.trader.long_ma_bottoms = [today]
        self.trader.short_ma_bottoms = []
        self.trader.peaks = []

        # Mock account
        mock_account = Mock()
        mock_account.cash = '10000.0'
        self.mock_api.get_account.return_value = mock_account

        # submit_order succeeds
        mock_order = Mock()
        mock_order.id = 'order-full-buy'
        self.mock_api.submit_order.return_value = mock_order

        # _wait_for_fill returns full fill
        full_fill = Mock()
        full_fill.filled_qty = 181
        full_fill.filled_avg_price = '55.0'
        self.trader._wait_for_fill = Mock(return_value=full_fill)

        # Mock persistence
        self.trader._save_entry_prices = Mock()

        self.trader.check_signals_and_trade()

        # current_position should reflect filled qty
        self.assertEqual(self.trader.current_position, 181)
        self.assertEqual(len(self.trader.entry_lots), 1)
        self.assertEqual(self.trader.entry_lots[0]['price'], 55.0)
        self.assertEqual(self.trader.entry_lots[0]['shares'], 181)

    # --- Step 1 test: shutdown path in _wait_for_fill ---

    @patch('trade.run_market_breadth_trade.time')
    def test_wait_for_fill_shutdown_while_polling(self, mock_time):
        """Shutdown during polling returns None without calling cancel_order"""
        mock_order = Mock()
        mock_order.id = 'order-shutdown'

        # time.time returns values within deadline
        mock_time.time.side_effect = [0, 10, 20]
        mock_time.sleep = Mock()

        # First poll returns pending, then shutdown triggers
        pending = Mock()
        pending.status = 'new'

        def set_shutdown_and_return(*args):
            self.trader._shutdown_requested = True
            return pending

        self.mock_api.get_order.side_effect = set_shutdown_and_return
        self.mock_api.cancel_order = Mock()

        result = self.trader._wait_for_fill(mock_order, timeout_seconds=60)

        self.assertIsNone(result)
        self.mock_api.cancel_order.assert_not_called()

    # --- Step 2 test: stale entry_prices file cleared on no-position ---

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_sync_clears_stale_file_when_no_position(self, mock_init):
        """Disk entry_prices file is removed when broker returns 404"""
        mock_api = Mock()
        mock_init.return_value = mock_api

        trader = MarketBreadthTrader(symbol='STALE')

        # Write a stale file to disk
        tmp_dir = tempfile.mkdtemp()
        entry_path = os.path.join(tmp_dir, 'entry_prices_STALE.json')
        trader._entry_prices_path = lambda: entry_path
        with open(entry_path, 'w') as f:
            json.dump([50.0, 55.0], f)

        self.assertTrue(os.path.exists(entry_path))

        # Broker returns 404
        mock_api.get_position.side_effect = Exception('position does not exist')

        trader._sync_position_from_broker()

        self.assertEqual(trader.current_position, 0)
        self.assertEqual(trader.entry_prices, [])
        self.assertFalse(os.path.exists(entry_path), 'Stale entry prices file should be removed')

        # Cleanup
        os.rmdir(tmp_dir)

    # --- Step 4 tests: execute_buy/sell testmode returns SimpleNamespace ---

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_execute_buy_testmode_returns_namespace(self, mock_init):
        """Testmode execute_buy returns SimpleNamespace with id and qty attributes"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-01')

        result = trader.execute_buy(100, reason='test')

        self.assertTrue(hasattr(result, 'id'))
        self.assertTrue(hasattr(result, 'qty'))
        self.assertEqual(result.id, 'TEST')
        self.assertEqual(result.qty, 100)
        self.assertEqual(result.status, 'accepted')

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_execute_sell_testmode_returns_namespace(self, mock_init):
        """Testmode execute_sell returns SimpleNamespace with id and qty attributes"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-01')

        result = trader.execute_sell(50, reason='test')

        self.assertTrue(hasattr(result, 'id'))
        self.assertTrue(hasattr(result, 'qty'))
        self.assertEqual(result.id, 'TEST')
        self.assertEqual(result.qty, 50)
        self.assertEqual(result.status, 'accepted')

    # --- Step 6 tests: coverage expansion ---

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_sync_uses_broker_avg_when_no_disk_file(self, mock_init):
        """_sync_position_from_broker falls back to broker avg_entry_price when no disk file"""
        mock_api = Mock()
        mock_init.return_value = mock_api

        trader = MarketBreadthTrader(symbol='NODISK')

        # No disk file exists
        tmp_dir = tempfile.mkdtemp()
        trader._entry_prices_path = lambda: os.path.join(tmp_dir, 'entry_prices_NODISK.json')

        # Broker returns position with avg_entry_price
        mock_position = Mock()
        mock_position.qty = 50
        mock_position.avg_entry_price = '42.50'
        mock_api.get_position.return_value = mock_position

        trader.entry_prices = []
        trader._sync_position_from_broker()

        self.assertEqual(trader.current_position, 50)
        self.assertEqual(trader.entry_prices, [42.50])

        # Cleanup
        os.rmdir(tmp_dir)

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_run_full_cycle_testmode(self, mock_init):
        """Testmode run() calls analyze_market then check_signals_and_trade when in closing time"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(testmode=True, test_date='2024-01-02')

        # is_closing_time_range returns True on first check
        trader.is_closing_time_range = Mock(return_value=True)
        trader.analyze_market = Mock()
        trader.check_signals_and_trade = Mock()

        trader.run()

        trader.analyze_market.assert_called_once()
        trader.check_signals_and_trade.assert_called_once()

    @patch('trade.run_market_breadth_trade.time')
    def test_wait_for_fill_filled_during_cancel(self, mock_time):
        """Order that fills during cancel attempt is returned successfully"""
        mock_order = Mock()
        mock_order.id = 'order-fill-during-cancel'

        # Immediate timeout
        mock_time.time.side_effect = [0, 100]
        mock_time.sleep = Mock()

        # After cancel, final get_order shows filled
        filled = Mock()
        filled.status = 'filled'
        filled.filled_qty = 100
        filled.filled_avg_price = '55.0'
        self.mock_api.get_order.return_value = filled
        self.mock_api.cancel_order = Mock()

        result = self.trader._wait_for_fill(mock_order, timeout_seconds=60)

        self.assertIsNotNone(result)
        self.assertEqual(result.status, 'filled')
        self.mock_api.cancel_order.assert_called_once_with('order-fill-during-cancel')

    def test_check_signals_skips_when_price_unavailable(self):
        """check_signals_and_trade returns early when get_current_price is None"""
        # Mock _sync_position_from_broker
        self.trader._sync_position_from_broker = Mock()

        # Price returns None
        self.mock_api.get_latest_bar.return_value = None

        # Set up signals that should NOT be reached
        self.trader.peaks = []
        self.trader.long_ma_bottoms = []
        self.trader.short_ma_bottoms = []

        # Should not raise, should return early
        self.trader.check_signals_and_trade()

        # submit_order should never be called
        self.mock_api.submit_order.assert_not_called()

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_load_entry_prices_corrupt_file_returns_none(self, mock_init):
        """Corrupt or non-list JSON in entry_prices file returns None"""
        mock_init.return_value = Mock()
        trader = MarketBreadthTrader(symbol='CORRUPT')

        tmp_dir = tempfile.mkdtemp()
        entry_path = os.path.join(tmp_dir, 'entry_prices_CORRUPT.json')
        trader._entry_prices_path = lambda: entry_path

        # Test with non-list JSON (dict)
        with open(entry_path, 'w') as f:
            json.dump({'price': 50.0}, f)
        result = trader._load_entry_prices()
        self.assertIsNone(result, 'Non-list JSON should return None')

        # Test with invalid JSON
        with open(entry_path, 'w') as f:
            f.write('not valid json{{{')
        result = trader._load_entry_prices()
        self.assertIsNone(result, 'Invalid JSON should return None')

        # Test with list of non-numeric values
        with open(entry_path, 'w') as f:
            json.dump(['not', 'numbers'], f)
        result = trader._load_entry_prices()
        self.assertIsNone(result, 'List of non-numeric values should return None')

        # Cleanup
        os.remove(entry_path)
        os.rmdir(tmp_dir)


class TestTimezoneCorrectness(unittest.TestCase):
    """CR-005 regression: verify ET date derivation under UTC-host conditions.

    When deployed on a UTC server after 21:00 UTC (= next calendar day in UTC but
    still the same trading day in ET), all date-dependent logic must use the ET date.
    """

    TZ_NY = ZoneInfo('US/Eastern')

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def setUp(self, mock_init):
        self.mock_api = Mock()
        mock_init.return_value = self.mock_api
        self.trader = MarketBreadthTrader(symbol='SSO')
        self.trader._acted_signals = set()

    @patch('trade.run_market_breadth_trade._now_et')
    def test_analyze_market_uses_et_date_not_utc(self, mock_now):
        """analyze_market() must use ET date, not UTC date, for FMP data fetch."""
        # Simulate 2026-04-11 00:30 UTC = 2026-04-10 20:30 ET
        # UTC calendar date is Apr 11, but ET calendar date is still Apr 10
        et_time = datetime(2026, 4, 10, 20, 30, tzinfo=self.TZ_NY)
        mock_now.return_value = et_time

        # Stub out the heavy analyze_market internals — we only care about the date calc
        self.trader._detect_signals = Mock()

        # Patch get_sp500_tickers_from_fmp and data fetching to avoid real API calls
        with (
            patch('trade.run_market_breadth_trade.get_sp500_tickers_from_fmp', return_value=['AAPL']),
            patch('trade.run_market_breadth_trade.get_multiple_stock_data') as mock_stock_data,
            patch.object(self.trader, '_get_latest_prices_from_alpaca') as mock_alpaca,
        ):
            # Return minimal data so analyze_market can compute breadth
            dates = pd.date_range('2025-01-01', '2026-04-10', freq='B')
            mock_stock_data.return_value = pd.DataFrame(
                {'AAPL': np.random.default_rng(42).random(len(dates)) * 100 + 100},
                index=dates,
            )
            mock_alpaca.return_value = pd.DataFrame(
                {'AAPL': [155.0]},
                index=[pd.Timestamp('2026-04-10')],
            )

            self.trader.analyze_market()

        # Verify: analyze_market computes yesterday = today - 1 day (ET-based).
        # _now_et() returns Apr 10 ET, so yesterday = Apr 9, NOT Apr 10 (which UTC would give).
        call_args = mock_stock_data.call_args
        end_date_arg = call_args[1].get('end_date') or call_args[0][2]
        self.assertIn(
            '2026-04-09',
            str(end_date_arg),
            'yesterday should be based on ET date (Apr 10 - 1 = Apr 9), not UTC (Apr 11 - 1 = Apr 10)',
        )

    @patch('trade.run_market_breadth_trade._now_et')
    def test_check_signals_uses_et_date(self, mock_now):
        """check_signals_and_trade() must derive current_date from ET, not UTC."""
        # 2026-04-11 00:30 UTC = 2026-04-10 20:30 ET
        et_time = datetime(2026, 4, 10, 20, 30, tzinfo=self.TZ_NY)
        mock_now.return_value = et_time

        self.trader.current_position = 0
        self.trader.entry_prices = []
        self.trader.entry_lots = []
        self.trader._sync_position_from_broker = Mock()

        # Set signal for ET date (Apr 10)
        et_date = pd.Timestamp('2026-04-10')
        self.trader.long_ma_bottoms = [et_date]
        self.trader.short_ma_bottoms = []
        self.trader.peaks = []

        self.mock_api.get_latest_bar.return_value = Mock(c=100.0)
        mock_account = Mock()
        mock_account.cash = '50000.0'
        self.mock_api.get_account.return_value = mock_account

        mock_order = Mock()
        mock_order.id = 'test-order'
        self.mock_api.submit_order.return_value = mock_order

        filled = Mock()
        filled.filled_qty = 499
        filled.filled_avg_price = '100.0'
        self.trader._wait_for_fill = Mock(return_value=filled)
        self.trader._save_entry_prices = Mock()
        self.trader._save_acted_signals = Mock()

        self.trader.check_signals_and_trade()

        # Signal on Apr 10 should match when _now_et() returns Apr 10 ET
        self.assertEqual(
            self.trader.current_position, 499, 'Signal on ET date should fire when _now_et returns that ET date'
        )

    @patch('trade.run_market_breadth_trade._now_et')
    def test_check_signals_misses_utc_date_signal(self, mock_now):
        """A signal set for UTC date (Apr 11) should NOT fire when ET date is Apr 10."""
        et_time = datetime(2026, 4, 10, 20, 30, tzinfo=self.TZ_NY)
        mock_now.return_value = et_time

        self.trader.current_position = 0
        self.trader.entry_prices = []
        self.trader.entry_lots = []
        self.trader._sync_position_from_broker = Mock()

        # Signal for UTC date (Apr 11) — wrong date from ET perspective
        utc_date = pd.Timestamp('2026-04-11')
        self.trader.long_ma_bottoms = [utc_date]
        self.trader.short_ma_bottoms = []
        self.trader.peaks = []

        self.mock_api.get_latest_bar.return_value = Mock(c=100.0)

        self.trader.check_signals_and_trade()

        # Should NOT fire because Apr 11 is tomorrow in ET
        self.assertEqual(self.trader.current_position, 0, 'Signal on UTC date (tomorrow in ET) should not fire')

    @patch('trade.run_market_breadth_trade._now_et')
    def test_get_latest_prices_uses_et_date_index(self, mock_now):
        """_get_latest_prices_from_alpaca() index must use ET date."""
        et_time = datetime(2026, 4, 10, 20, 30, tzinfo=self.TZ_NY)
        mock_now.return_value = et_time

        mock_bar = Mock()
        mock_bar.c = 155.0
        self.mock_api.get_latest_bar.return_value = mock_bar

        result = self.trader._get_latest_prices_from_alpaca(['AAPL'])

        # Index should be ET date (Apr 10), not UTC date (Apr 11)
        self.assertEqual(result.index[0], pd.Timestamp('2026-04-10'), 'Alpaca price index should use ET date')


class TestUpdateLotsAfterSell(unittest.TestCase):
    """M-02: _update_lots_after_sell oversell guard and FIFO correctness."""

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def setUp(self, mock_init):
        mock_init.return_value = Mock()
        self.trader = MarketBreadthTrader(symbol='TEST')
        self.trader._acted_signals = set()
        self.trader._save_entry_prices = Mock()
        self.trader._clear_entry_prices_file = Mock()

    def test_oversell_beyond_position_clears_to_zero(self):
        """Selling more than lots total should not crash; position goes to 0."""
        self.trader.entry_lots = [{'price': 50.0, 'shares': 80}]
        self.trader.current_position = 80

        self.trader._update_lots_after_sell(100)  # oversell by 20

        self.assertEqual(self.trader.current_position, 0)
        self.assertEqual(self.trader.entry_lots, [])
        self.assertEqual(self.trader.entry_prices, [])

    def test_oversell_with_stale_lots_forces_zero(self):
        """filled_qty exceeds lots total but not current_position — forces zero."""
        # Stale state: position=100 but lots only account for 80 shares
        self.trader.entry_lots = [{'price': 50.0, 'shares': 80}]
        self.trader.current_position = 100

        self.trader._update_lots_after_sell(90)  # 90 > lots(80), but < position(100)

        # Must force to zero, not leave position=10 with empty lots
        self.assertEqual(self.trader.current_position, 0)
        self.assertEqual(self.trader.entry_lots, [])
        self.assertEqual(self.trader.entry_prices, [])

    def test_fifo_partial_lot_reduction(self):
        """Selling part of a lot reduces that lot's shares correctly."""
        self.trader.entry_lots = [
            {'price': 50.0, 'shares': 100},
            {'price': 60.0, 'shares': 50},
        ]
        self.trader.current_position = 150

        self.trader._update_lots_after_sell(120)  # consumes lot1 (100) + 20 from lot2

        self.assertEqual(self.trader.current_position, 30)
        self.assertEqual(len(self.trader.entry_lots), 1)
        self.assertEqual(self.trader.entry_lots[0]['price'], 60.0)
        self.assertEqual(self.trader.entry_lots[0]['shares'], 30)


class TestSyncPositionLotsMismatch(unittest.TestCase):
    """M-03: _sync_position_from_broker falls back to broker when lots mismatch."""

    @patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca')
    def test_mismatch_falls_back_to_broker_avg(self, mock_init):
        mock_api = Mock()
        mock_init.return_value = mock_api

        trader = MarketBreadthTrader(symbol='TEST')
        trader._acted_signals = set()

        # Write lots file with total 80 shares
        tmp_dir = tempfile.mkdtemp()
        entry_path = os.path.join(tmp_dir, 'entry_prices_TEST.json')
        trader._entry_prices_path = lambda: entry_path
        with open(entry_path, 'w') as f:
            json.dump({'lots': [{'price': 55.0, 'shares': 50}, {'price': 60.0, 'shares': 30}]}, f)

        # Broker says 100 shares (mismatch: 80 != 100)
        mock_position = Mock()
        mock_position.qty = 100
        mock_position.avg_entry_price = '57.0'
        mock_api.get_position.return_value = mock_position

        trader.entry_lots = []
        trader.entry_prices = []
        trader._sync_position_from_broker()

        # Should fall back to single lot at broker avg
        self.assertEqual(len(trader.entry_lots), 1)
        self.assertEqual(trader.entry_lots[0]['price'], 57.0)
        self.assertEqual(trader.entry_lots[0]['shares'], 100)
        self.assertEqual(trader.current_position, 100)

        os.remove(entry_path)
        os.rmdir(tmp_dir)


if __name__ == '__main__':
    unittest.main()
