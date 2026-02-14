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
    @patch('alpaca_trade_api.REST')
    def setUp(self, mock_rest):
        """Test setup"""
        self.mock_api = mock_rest.return_value
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

        # Create MarketBreadthTrader instance
        trader = MarketBreadthTrader()
        # Replace api attribute with mock
        trader.api = mock_api

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

    @patch('trade.run_market_breadth_trade.get_sp500_tickers_from_wikipedia')
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
        """Test share calculation"""
        # Normal case
        shares = self.trader._calculate_shares(10000, 50.0)
        expected_shares = int(10000 / (50.0 * (1 + self.trader.slippage)))
        self.assertEqual(shares, expected_shares)

        # Case with fractional shares
        shares = self.trader._calculate_shares(10000, 33.33)
        self.assertEqual(shares, int(10000 / (33.33 * (1 + self.trader.slippage))))


if __name__ == '__main__':
    unittest.main()
