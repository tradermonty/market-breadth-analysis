import argparse
import json
import logging
import logging.handlers
import os
import pathlib
import signal
import sys
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path to import market_breadth
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_breadth import (
    calculate_above_ma,
    calculate_trend_with_hysteresis,
    get_multiple_stock_data,
    get_sp500_tickers_from_fmp,
)

# Log settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'trade/market_breadth_trade.log', maxBytes=10 * 1024 * 1024, backupCount=5
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('market_breadth_trade')

# Timezone settings
TZ_NY = ZoneInfo('US/Eastern')
TZ_UTC = ZoneInfo('UTC')


def _now_et():
    """Return current datetime in US/Eastern. Patchable for tests."""
    return datetime.now(tz=TZ_NY)


# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Create necessary directories
reports_dir = pathlib.Path('reports')
reports_dir.mkdir(exist_ok=True)


class MarketBreadthTrader:
    def __init__(
        self,
        short_ma=5,
        long_ma=200,
        initial_capital=50000,
        slippage=0.0005,
        commission=0.0001,
        use_saved_data=False,
        debug=False,
        ma_type='ema',
        symbol='SSO',
        stop_loss_pct=0.08,
        disable_short_ma_entry=False,
        closing_time_minutes=20,
        testmode=False,
        test_date=None,
        # TV mode pivot detection parameters (aligned with backtest)
        pivot_len_long=20,
        pivot_len_short=10,
        prom_thresh_long=0.005,
        prom_thresh_short=0.03,
        peak_level=0.70,
        trough_level_long=0.40,
        trough_level_short=0.20,
        no_pyramiding=True,
    ):
        self.symbol = symbol
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.commission = commission
        self.use_saved_data = use_saved_data
        self.debug = debug
        self.ma_type = ma_type.lower()  # 'ema' or 'sma'
        self.stop_loss_pct = stop_loss_pct
        self.disable_short_ma_entry = disable_short_ma_entry
        self.closing_time_minutes = closing_time_minutes
        self.testmode = testmode
        self.test_date = test_date
        self.test_dt = None  # Variable to hold current time in test mode

        # TV mode pivot detection parameters
        self.pivot_len_long = pivot_len_long
        self.pivot_len_short = pivot_len_short
        self.prom_thresh_long = prom_thresh_long
        self.prom_thresh_short = prom_thresh_short
        self.peak_level = peak_level
        self.trough_level_long = trough_level_long
        self.trough_level_short = trough_level_short
        self.no_pyramiding = no_pyramiding

        # Initialize Alpaca API
        self.api = self._initialize_alpaca()

        # Graceful shutdown flag
        self._shutdown_requested = False

        # Initialize variables
        self.current_position = 0
        self.entry_prices = []
        self.entry_lots = []  # MJ-002: [{'price': float, 'shares': int}, ...]

        # MJ-003: Track acted-on signals to prevent duplicate trades
        self._acted_signals = self._load_acted_signals()

        # Initialize signal-related variables
        self.short_ma_bottoms = []
        self.long_ma_bottoms = []
        self.peaks = []

        # TV mode signal dictionaries (populated by _detect_signals)
        self._tv_peak_signals = {}
        self._tv_long_trough_signals = {}
        self._tv_short_trough_signals = {}

        logger.info(f'MarketBreadthTrader initialized with symbol: {self.symbol}')
        if self.testmode:
            logger.info(f'Test mode enabled for date: {self.test_date}')

    def _initialize_alpaca(self):
        """Initialize Alpaca API client only when needed."""
        try:
            import alpaca_trade_api as tradeapi
        except ImportError as exc:
            raise ImportError(
                'alpaca-trade-api is required for live trading. Install it with: pip install alpaca-trade-api'
            ) from exc

        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise OSError('ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment')

        if 'paper' not in ALPACA_BASE_URL:
            if os.getenv('ALPACA_LIVE_CONFIRMED', '').lower() != 'true':
                raise OSError(
                    f'Live trading URL detected: {ALPACA_BASE_URL}. Set ALPACA_LIVE_CONFIRMED=true to confirm.'
                )
            logger.warning('LIVE TRADING MODE — connecting to real money account')

        return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

    def is_closing_time_range(self, range_minutes=20):
        """Check if current time is within the specified minutes before market close"""
        if self.testmode:
            # Use time advanced in run function for test mode
            current_dt = self.test_dt
            logger.debug(f'Test mode time: {current_dt}')
        else:
            current_dt = _now_et()

        cal = self.api.get_calendar(start=str(current_dt.date()), end=str(current_dt.date()))

        if len(cal) > 0:
            close_time = cal[0].close
            if isinstance(close_time, str):
                close_dt = datetime.combine(
                    current_dt.date(), datetime.strptime(close_time, '%H:%M').time(), tzinfo=TZ_NY
                )
            else:
                close_dt = datetime.combine(current_dt.date(), close_time, tzinfo=TZ_NY)

            logger.debug(f'Market close time: {close_dt}')
            logger.debug(f'Time difference: {close_dt - current_dt}')

            if close_dt - timedelta(minutes=range_minutes) <= current_dt < close_dt:
                logger.info('In closing time range')
                return True
            else:
                logger.debug(f"{current_dt}, it's not in closing time range")
                return False
        else:
            logger.info('Market will not open on the date.')
            return False

    def is_market_open(self):
        """Check if the market is open"""
        clock = self.api.get_clock()
        return clock.is_open

    def get_current_position(self):
        """Get current position"""
        try:
            # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
            position = self.api.get_position(self.symbol)
            return int(position.qty)
        except Exception as e:
            logger.info(f'No position found for {self.symbol}: {e}')
            return 0

    def _sync_position_from_broker(self):
        """Sync position and entry lots from broker.

        MJ-002: Recovers entry_lots from disk or broker's avg_entry_price.
        Validates lots total against broker qty for consistency.

        Raises on transient API/network errors to prevent accidental trades.
        """
        try:
            position = self.api.get_position(self.symbol)
            broker_qty = int(position.qty)
            self.current_position = broker_qty
            if self.current_position > 0 and not self.entry_lots:
                saved = self._load_entry_prices()
                use_broker_fallback = True

                if saved and isinstance(saved, dict) and 'lots' in saved:
                    lots = saved['lots']
                    lots_total = sum(lot['shares'] for lot in lots)
                    if lots_total == broker_qty:
                        self.entry_lots = lots
                        use_broker_fallback = False
                        logger.info(f'Recovered {len(lots)} entry lots from disk')
                    else:
                        logger.warning(
                            f'Lots total ({lots_total}) != broker qty ({broker_qty}); '
                            f'falling back to broker avg_entry_price'
                        )

                if use_broker_fallback:
                    avg_price = float(position.avg_entry_price)
                    self.entry_lots = [{'price': avg_price, 'shares': broker_qty}]
                    logger.info(f'Using broker avg_entry_price: ${avg_price:.2f} x {broker_qty} shares')

                self.entry_prices = [lot['price'] for lot in self.entry_lots]
        except Exception as e:
            # Alpaca returns 404 with "position does not exist" when no position
            err_str = str(e)
            if 'position does not exist' in err_str.lower() or '404' in err_str:
                logger.info(f'No position found for {self.symbol}')
                self.current_position = 0
                self.entry_prices = []
                self.entry_lots = []
                self._clear_entry_prices_file()
            else:
                logger.error(f'API error syncing position for {self.symbol}: {e}')
                raise

    def get_current_price(self):
        """Get current price"""
        try:
            logger.info(f'Starting: Getting current price for {self.symbol}')

            # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
            # Get latest stock price
            bars = self.api.get_latest_bar(self.symbol)
            if bars and hasattr(bars, 'c'):
                logger.info(f'Success: Got current price for {self.symbol}: ${bars.c:.2f}')
                return float(bars.c)
            else:
                logger.error(f'Failed: Could not get current price for {self.symbol} (no valid bar data)')
                return None
        except Exception as e:
            logger.error(f'Error: Error occurred while getting current price for {self.symbol}: {e}', exc_info=True)
            return None

    def execute_buy(self, shares, reason=''):
        """Execute buy order"""
        if self.testmode:
            logger.info(f'[TEST MODE] Would execute buy order: {shares} shares of {self.symbol}, reason: {reason}')
            return SimpleNamespace(id='TEST', qty=shares, status='accepted')

        try:
            order = self.api.submit_order(
                symbol=self.symbol, qty=shares, side='buy', type='market', time_in_force='day'
            )
            logger.info(f'Buy order executed: {shares} shares of {self.symbol}, reason: {reason}')
            return order
        except Exception as e:
            logger.error(f'Error executing buy order: {e}', exc_info=True)
            return None

    def execute_sell(self, shares, reason=''):
        """Execute sell order"""
        if self.testmode:
            logger.info(f'[TEST MODE] Would execute sell order: {shares} shares of {self.symbol}, reason: {reason}')
            return SimpleNamespace(id='TEST', qty=shares, status='accepted')

        try:
            order = self.api.submit_order(
                symbol=self.symbol, qty=shares, side='sell', type='market', time_in_force='day'
            )
            logger.info(f'Sell order executed: {shares} shares of {self.symbol}, reason: {reason}')
            return order
        except Exception as e:
            logger.error(f'Error executing sell order: {e}', exc_info=True)
            return None

    def _wait_for_fill(self, order, timeout_seconds=60):
        """Poll order until filled, canceled, or timeout."""
        if self.testmode:
            return SimpleNamespace(filled_avg_price=None, filled_qty=None, status='filled')
        deadline = time.time() + timeout_seconds
        shutdown_break = False
        while time.time() < deadline:
            if self._shutdown_requested:
                logger.warning(f'Shutdown requested — aborting fill wait for order {order.id}')
                shutdown_break = True
                break
            updated = self.api.get_order(order.id)
            if updated.status == 'filled':
                logger.info(f'Order {order.id} filled: {updated.filled_qty} @ ${float(updated.filled_avg_price):.2f}')
                return updated
            if updated.status == 'partially_filled':
                logger.info(f'Order {order.id} partially filled: {updated.filled_qty} of {updated.qty}')
            if updated.status in ('canceled', 'expired', 'rejected', 'suspended'):
                logger.warning(f'Order {order.id} ended: {updated.status}')
                return None
            time.sleep(2)
        if shutdown_break:
            return None
        logger.error(f'Order {order.id} not filled within {timeout_seconds}s — canceling')
        try:
            self.api.cancel_order(order.id)
            # Check if it filled during cancellation
            final = self.api.get_order(order.id)
            if final.status == 'filled':
                logger.info(
                    f'Order {order.id} filled during cancel: {final.filled_qty} @ ${float(final.filled_avg_price):.2f}'
                )
                return final
            if int(final.filled_qty or 0) > 0:
                logger.warning(f'Order {order.id} partial fill after cancel: {final.filled_qty} shares')
                return final
        except Exception as e:
            logger.error(f'Failed to cancel order {order.id}: {e}', exc_info=True)
        return None

    def run(self):
        """Execute trading"""
        logger.info('Starting market breadth trading...')

        if not self.testmode:

            def _handle_shutdown(signum, frame):
                sig_name = signal.Signals(signum).name
                logger.warning(f'Received {sig_name} — initiating graceful shutdown')
                self._shutdown_requested = True

            signal.signal(signal.SIGTERM, _handle_shutdown)
            signal.signal(signal.SIGINT, _handle_shutdown)

        if self.testmode:
            # Set initial time for test mode (EST 15:30)
            self.test_dt = datetime.strptime(self.test_date, '%Y-%m-%d')
            self.test_dt = self.test_dt.replace(hour=15, minute=30, tzinfo=TZ_NY)
            logger.info(f'Test mode started at {self.test_dt} (EST)')

        while True:
            if self._shutdown_requested:
                logger.info('Shutdown requested — exiting trading loop')
                if self.current_position > 0:
                    logger.warning(f'Open position: {self.current_position} shares. Manual check recommended.')
                break

            if self.testmode:
                # Use specified time in test mode
                current_dt = self.test_dt
                logger.debug(f'Current test time: {current_dt}')
            else:
                # Use current time in normal mode
                current_dt = _now_et()

            # Check if market is open (skip in test mode)
            if not self.testmode and not self.is_market_open():
                logger.info('Market is closed today. Exiting trading.')
                break

            # Check if within closing time range
            if self.is_closing_time_range(self.closing_time_minutes):
                try:
                    # Analyze market data
                    logger.info('Analyzing market data...')
                    self.analyze_market()

                    # Check signals and execute trades
                    logger.info('Checking signals and executing trades...')
                    self.check_signals_and_trade()

                    if self.testmode:
                        logger.info('Test mode trading completed for the day.')
                    else:
                        logger.info('Trading completed for today.')
                    break
                except Exception as e:
                    logger.error(f'Error during trading: {e!s}', exc_info=True)
                    if self.current_position > 0:
                        logger.critical(
                            f'ALERT: Exception with open position ({self.current_position} shares). '
                            'Manual intervention may be required.'
                        )
                    break

            if self.testmode:
                # Advance time by 1 minute in test mode
                self.test_dt += timedelta(minutes=1)
                logger.debug(f'Test time advanced to: {self.test_dt}')
                # No actual waiting in test mode
                continue
            else:
                # Wait 1 minute in normal mode
                if self._shutdown_requested:
                    continue  # Loop back to top where break will happen
                logger.debug('Waiting for closing time range...')
                for _ in range(60):
                    if self._shutdown_requested:
                        break
                    time.sleep(1)

        logger.info('Trading session ended.')

    def analyze_market(self):
        """Analyze market data"""
        try:
            logger.info('Starting market data analysis')

            # Get past data (using yesterday's date for FMP)
            if self.testmode:
                today = pd.Timestamp(self.test_date)
                logger.info(f'Test mode: Using test date {today.strftime("%Y-%m-%d")} as current date')
            else:
                today = _now_et()

            yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')

            logger.info(f'Data retrieval period: {start_date} to {yesterday}')

            # Get S&P500 ticker list
            logger.info('Getting S&P500 ticker list...')
            sp500_tickers = get_sp500_tickers_from_fmp()
            logger.info(f'Got S&P500 ticker list: {len(sp500_tickers)} tickers')

            # Get historical data from FMP - using data up to yesterday
            logger.info('Getting historical data from FMP...')
            historical_data = get_multiple_stock_data(
                sp500_tickers, start_date, yesterday, use_saved_data=self.use_saved_data
            )
            logger.info(f'Got historical data: {len(historical_data.columns)} tickers, {len(historical_data)} days')

            # Get today's latest data from Alpaca
            logger.info("Getting today's latest data from Alpaca...")
            today_data = self._get_latest_prices_from_alpaca(sp500_tickers)
            logger.info(f"Got today's latest data: {len(today_data.columns)} tickers")

            # Combine historical and today's data
            logger.info('Combining data...')
            all_data = pd.concat([historical_data, today_data])
            # Remove fully empty rows
            all_data = all_data.dropna(how='all')
            # Sort by index (date)
            all_data = all_data.sort_index()
            # Remove duplicated indices
            all_data = all_data[~all_data.index.duplicated(keep='first')]
            logger.info(f'Combined data: {len(all_data.columns)} tickers, {len(all_data)} days')

            # Calculate Market Breadth Index
            logger.info('Calculating Market Breadth Index...')
            self.above_ma = calculate_above_ma(all_data)
            logger.info('Market Breadth Index calculation completed')

            # Calculate moving averages
            logger.info('Calculating moving averages...')
            self.breadth_index = self.above_ma.mean(axis=1)

            if self.ma_type == 'ema':
                self.short_ma_line = self.breadth_index.ewm(span=self.short_ma, adjust=False).mean()
                self.long_ma_line = self.breadth_index.ewm(span=self.long_ma, adjust=False).mean()
            else:
                self.short_ma_line = self.breadth_index.rolling(window=self.short_ma).mean()
                self.long_ma_line = self.breadth_index.rolling(window=self.long_ma).mean()

            logger.info(
                f'Moving average calculation completed: {self.ma_type.upper()} {self.short_ma} days, {self.ma_type.upper()} {self.long_ma} days'
            )

            # Calculate trend
            logger.info('Calculating trend...')
            self.long_ma_trend = pd.Series(
                calculate_trend_with_hysteresis(self.long_ma_line), index=self.long_ma_line.index
            )
            logger.info('Trend calculation completed')

            # Extract price data for specified symbol
            logger.info(f'Extracting price data for {self.symbol}...')
            if self.symbol in all_data.columns:
                self.price_data = pd.DataFrame(all_data[self.symbol], columns=['adjusted_close'])
                logger.info(f'Extracted price data for {self.symbol}: {len(self.price_data)} days')
            else:
                # Get data individually if symbol not included
                logger.info(f'{self.symbol} not included in data, getting data individually...')
                self.price_data = self._get_latest_price_from_alpaca(self.symbol)
                if isinstance(self.price_data, pd.Series):
                    self.price_data = pd.DataFrame(self.price_data, columns=['adjusted_close'])
                    logger.info(f'Got price data for {self.symbol}: {len(self.price_data)} days')

            # Detect signals
            logger.info('Detecting signals...')
            self._detect_signals()
            logger.info('Signal detection completed')

            logger.info('Market data analysis completed')

        except Exception as e:
            logger.error(f'Error during market data analysis: {e!s}', exc_info=True)
            raise

    def _get_latest_prices_from_alpaca(self, tickers):
        """Get latest stock prices using Alpaca API"""
        try:
            # Get current date (Timestamp type)
            if self.testmode:
                today = pd.Timestamp(self.test_date)
                logger.info(f'Test mode: Using test date {today.strftime("%Y-%m-%d")} as current date')
            else:
                today = pd.Timestamp(_now_et().strftime('%Y-%m-%d'))

            # Temporarily store price data in dictionary
            price_dict = {}

            # Variables for progress display
            total_tickers = len(tickers)
            processed_tickers = 0
            success_count = 0
            failure_count = 0

            logger.info(f'Starting: Getting latest prices for {total_tickers} tickers')

            # Show progress using progress bar
            for ticker in tqdm(tickers, desc='Getting prices', unit='ticker'):
                try:
                    # Show progress
                    processed_tickers += 1

                    # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
                    # Get latest stock price
                    bars = self.api.get_latest_bar(ticker)
                    if bars and hasattr(bars, 'c'):
                        price_dict[ticker] = bars.c
                        success_count += 1
                        logger.debug(f'Success: Got latest price for {ticker}: ${bars.c:.2f}')
                    else:
                        logger.warning(f'Failed to get latest price for {ticker} (no valid bar data)')
                        failure_count += 1
                except Exception as e:
                    logger.warning(f'Error getting latest price for {ticker}: {e!s}')
                    failure_count += 1

            # Convert all price data to DataFrame at once
            latest_prices = pd.DataFrame(price_dict, index=[today])

            logger.info(
                f'Completed: Processed {total_tickers} tickers (Success: {success_count}, Failure: {failure_count})'
            )
            logger.info(f'Retrieved price data: {len(price_dict)} tickers')

            return latest_prices

        except Exception as e:
            logger.error(f'Error getting latest prices from Alpaca: {e!s}', exc_info=True)
            raise

    def _get_latest_price_from_alpaca(self, ticker):
        """Get latest stock price for specific ticker using Alpaca API"""
        try:
            logger.info(f'Starting: Getting latest price for {ticker}')

            # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
            # Get latest stock price
            bars = self.api.get_latest_bar(ticker)
            if bars and hasattr(bars, 'c'):
                logger.info(f'Success: Got latest price for {ticker}: ${bars.c:.2f}')
                return pd.Series([bars.c], index=[pd.Timestamp(_now_et().strftime('%Y-%m-%d'))])
            else:
                logger.warning(f'Failed: Could not get latest price for {ticker} (no valid bar data)')
                return pd.Series()

        except Exception as e:
            logger.error(f'Error: Error occurred while getting latest price for {ticker}: {e!s}', exc_info=True)
            return pd.Series()

    def _detect_signals(self):
        """Detect signals using TV mode pivot detection (aligned with backtest)."""
        from backtest.backtest import detect_pivot_high, detect_pivot_low

        logger.info('Starting TV mode signal detection')

        # Initialize signal lists
        self.short_ma_bottoms = []
        self.long_ma_bottoms = []
        self.peaks = []

        # Peak signals on long MA (exit signals)
        raw_peaks = detect_pivot_high(self.long_ma_line, self.pivot_len_long, self.prom_thresh_long, self.peak_level)
        self._tv_peak_signals = {}
        for confirm_date, pivot_date, val in raw_peaks:
            if confirm_date not in self._tv_peak_signals:
                self._tv_peak_signals[confirm_date] = (pivot_date, val)

        # Long MA trough signals (entry signals)
        raw_long_troughs = detect_pivot_low(self.long_ma_line, self.pivot_len_long, self.prom_thresh_long)
        self._tv_long_trough_signals = {}
        for confirm_date, pivot_date, val in raw_long_troughs:
            if val < self.trough_level_long:
                if confirm_date not in self._tv_long_trough_signals:
                    self._tv_long_trough_signals[confirm_date] = (pivot_date, val)

        # Short MA trough signals (entry signals)
        if not self.disable_short_ma_entry:
            raw_short_troughs = detect_pivot_low(self.short_ma_line, self.pivot_len_short, self.prom_thresh_short)
            self._tv_short_trough_signals = {}
            for confirm_date, pivot_date, val in raw_short_troughs:
                confirm_loc = self.breadth_index.index.get_loc(confirm_date)
                start_loc = max(0, confirm_loc - 19)
                recent_min = self.breadth_index.iloc[start_loc : confirm_loc + 1].min()
                if recent_min <= self.trough_level_short:
                    if confirm_date not in self._tv_short_trough_signals:
                        self._tv_short_trough_signals[confirm_date] = (pivot_date, val)
        else:
            self._tv_short_trough_signals = {}

        # Populate legacy lists for check_signals_and_trade() compatibility
        self.short_ma_bottoms = list(self._tv_short_trough_signals.keys())
        self.long_ma_bottoms = list(self._tv_long_trough_signals.keys())
        self.peaks = list(self._tv_peak_signals.keys())

        logger.info('TV signal detection completed:')
        logger.info(f'  Peak signals (exit): {len(self._tv_peak_signals)}')
        logger.info(f'  Long MA trough signals (entry): {len(self._tv_long_trough_signals)}')
        logger.info(f'  Short MA trough signals (entry): {len(self._tv_short_trough_signals)}')

    def _find_recent_signal(self, signal_dates, current_date, signal_type='', lookback_days=5):
        """Find the most recent signal within lookback_days of current_date.

        MJ-003: Skips signals already in _acted_signals to prevent duplicates.
        """
        for d in sorted(signal_dates, reverse=True):
            delta = (current_date - d).days
            if 0 <= delta <= lookback_days:
                if signal_type and (signal_type, d.strftime('%Y-%m-%d')) in self._acted_signals:
                    continue
                return d
        return None

    def check_signals_and_trade(self):
        """Check signals and execute trades"""
        logger.info('Checking signals and executing trades')

        # Sync position and entry prices from broker
        self._sync_position_from_broker()
        logger.info(f'Current position: {self.current_position} shares')

        # Get current price
        current_price = self.get_current_price()
        if current_price is None:
            logger.error('Failed to get current price. Exiting.')
            return

        logger.info(f'Current price: ${current_price:.2f}')

        # --- Stop loss check (before signal-based trading) ---
        # MJ-002: Use cost-weighted average for stop loss calculation
        if self.current_position > 0 and self.entry_lots:
            total_cost = sum(lot['price'] * lot['shares'] for lot in self.entry_lots)
            total_shares = sum(lot['shares'] for lot in self.entry_lots)
            avg_entry = total_cost / total_shares if total_shares > 0 else 0
            stop_loss_price = avg_entry * (1 - self.stop_loss_pct)

            logger.info(f'Stop loss check: avg_entry=${avg_entry:.2f}, stop=${stop_loss_price:.2f}')

            if current_price <= stop_loss_price:
                logger.info(f'Stop loss triggered: price ${current_price:.2f} <= stop ${stop_loss_price:.2f}')
                order = self.execute_sell(self.current_position, reason='stop loss')
                if order:
                    filled = self._wait_for_fill(order)
                    if filled:
                        filled_qty = int(filled.filled_qty) if filled.filled_qty is not None else self.current_position
                        self._update_lots_after_sell(filled_qty)
                        logger.info(f'Stop loss exit: {filled_qty} shares at ${current_price:.2f}')
                    else:
                        logger.error('Stop loss order not filled — position remains open')
                else:
                    logger.error('Failed to execute stop loss exit')
                return  # Stop loss takes priority, skip other signals

        # Current date
        if self.testmode:
            current_date = pd.Timestamp(self.test_date)
            logger.info(f'Test mode: Using test date {current_date.strftime("%Y-%m-%d")} as current date')
        else:
            current_date = pd.to_datetime(_now_et().strftime('%Y-%m-%d'))

        # Check for signals (with lookback for missed days)
        # MJ-003: Keep signal dates (not bools) for acted_signals tracking
        peak_signal_date = self._find_recent_signal(self.peaks, current_date, 'peak')
        long_trough_date = self._find_recent_signal(self.long_ma_bottoms, current_date, 'long_trough')
        short_trough_date = (
            self._find_recent_signal(self.short_ma_bottoms, current_date, 'short_trough')
            if not self.disable_short_ma_entry
            else None
        )

        # Log signal detection status
        logger.info('Signal detection status:')
        logger.info(f'  Current date: {current_date.strftime("%Y-%m-%d")}')
        logger.info(f'  Short MA bottom signal: {"Detected" if short_trough_date else "Not detected"}')
        logger.info(f'  Long MA bottom signal: {"Detected" if long_trough_date else "Not detected"}')
        logger.info(f'  Peak signal: {"Detected" if peak_signal_date else "Not detected"}')

        # Check if current date matches signal date
        if peak_signal_date is not None and self.current_position > 0:
            # Exit at 200MA peak (check exit before entry)
            logger.info(f'Long MA peak signal detected for {peak_signal_date.strftime("%Y-%m-%d")}')

            order = self.execute_sell(self.current_position, reason='peak exit')
            if order:
                filled = self._wait_for_fill(order)
                if filled:
                    filled_qty = int(filled.filled_qty) if filled.filled_qty is not None else self.current_position
                    logger.info(f'Exit executed at long MA peak: {filled_qty} shares at ${current_price:.2f}')
                    self._update_lots_after_sell(filled_qty)
                    # MJ-003: Record acted signal only after confirmed fill
                    self._acted_signals.add(('peak', peak_signal_date.strftime('%Y-%m-%d')))
                    self._save_acted_signals()
                else:
                    logger.error('Peak exit order not filled — position remains open')
            else:
                logger.error('Failed to execute exit at long MA peak')

        elif long_trough_date is not None:
            # Entry at 200MA bottom
            logger.info(f'Long MA bottom signal detected for {long_trough_date.strftime("%Y-%m-%d")}')

            # Skip if already have position (no_pyramiding)
            if self.no_pyramiding and self.current_position > 0:
                logger.info(f'Already have position ({self.current_position} shares), no_pyramiding=True, skipping')
            else:
                # Use all available capital (100%)
                account = self.api.get_account()
                available_capital = float(account.cash)

                shares = self._calculate_shares(available_capital, current_price)

                if shares > 0:
                    order = self.execute_buy(shares, reason='long_ma_bottom')
                    if order:
                        filled = self._wait_for_fill(order)
                        if filled:
                            filled_qty = int(filled.filled_qty) if filled.filled_qty is not None else shares
                            fill_price = (
                                float(filled.filled_avg_price) if filled.filled_avg_price is not None else current_price
                            )
                            self.current_position += filled_qty
                            logger.info(f'Entry executed at long MA bottom: {filled_qty} shares at ${fill_price:.2f}')
                            self.entry_lots.append({'price': fill_price, 'shares': filled_qty})
                            self.entry_prices = [lot['price'] for lot in self.entry_lots]
                            self._save_entry_prices()
                            # MJ-003: Record acted signal only after confirmed fill
                            self._acted_signals.add(('long_trough', long_trough_date.strftime('%Y-%m-%d')))
                            self._save_acted_signals()
                            if filled_qty < shares:
                                logger.warning(f'Partial buy fill: {filled_qty} of {shares} requested')
                        else:
                            logger.error('Long MA bottom buy order not filled')
                    else:
                        logger.error('Failed to execute entry at long MA bottom')
                else:
                    logger.info('No shares to buy due to insufficient capital')

        elif short_trough_date is not None:
            # Entry at short MA bottom
            logger.info(f'Short MA bottom signal detected for {short_trough_date.strftime("%Y-%m-%d")}')

            # Skip if already have position (no_pyramiding)
            if self.no_pyramiding and self.current_position > 0:
                logger.info(f'Already have position ({self.current_position} shares), no_pyramiding=True, skipping')
            else:
                # Use all available capital (100%, aligned with backtest)
                account = self.api.get_account()
                available_capital = float(account.cash)

                shares = self._calculate_shares(available_capital, current_price)

                if shares > 0:
                    order = self.execute_buy(shares, reason='short_ma_bottom')
                    if order:
                        filled = self._wait_for_fill(order)
                        if filled:
                            filled_qty = int(filled.filled_qty) if filled.filled_qty is not None else shares
                            fill_price = (
                                float(filled.filled_avg_price) if filled.filled_avg_price is not None else current_price
                            )
                            self.current_position += filled_qty
                            logger.info(f'Entry executed at short MA bottom: {filled_qty} shares at ${fill_price:.2f}')
                            self.entry_lots.append({'price': fill_price, 'shares': filled_qty})
                            self.entry_prices = [lot['price'] for lot in self.entry_lots]
                            self._save_entry_prices()
                            # MJ-003: Record acted signal only after confirmed fill
                            self._acted_signals.add(('short_trough', short_trough_date.strftime('%Y-%m-%d')))
                            self._save_acted_signals()
                            if filled_qty < shares:
                                logger.warning(f'Partial buy fill: {filled_qty} of {shares} requested')
                        else:
                            logger.error('Short MA bottom buy order not filled')
                    else:
                        logger.error('Failed to execute entry at short MA bottom')
                else:
                    logger.info('No shares to buy due to insufficient capital')
        else:
            logger.info('No trading signals detected for today')

        logger.info('Signal check and trade execution completed')

    def _calculate_shares(self, amount, price):
        """Calculate number of shares to buy, accounting for slippage and commission."""
        return int(amount / (price * (1 + self.slippage + self.commission)))

    def _entry_prices_path(self):
        """Return the file path for persisted entry prices."""
        return f'trade/entry_prices_{self.symbol}.json'

    def _save_entry_prices(self):
        """Persist entry_lots to disk as JSON (MJ-002: new format with lots)."""
        try:
            data = {'lots': self.entry_lots}
            with open(self._entry_prices_path(), 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f'Failed to save entry prices: {e}', exc_info=True)

    def _load_entry_prices(self):
        """Load entry data from disk. Returns dict with 'lots' key, or None.

        MJ-002: Supports new format {"lots": [...]}. Old flat-list format
        returns None so the caller falls back to broker avg_entry_price.
        """
        try:
            path = self._entry_prices_path()
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'lots' in data:
                    lots = data['lots']
                    # Validate lot schema: each lot must have numeric 'price' and int 'shares'
                    if isinstance(lots, list) and all(
                        isinstance(lot, dict)
                        and isinstance(lot.get('price'), int | float)
                        and isinstance(lot.get('shares'), int)
                        and lot['shares'] > 0
                        for lot in lots
                    ):
                        return data  # New format, validated
                    logger.warning('Corrupted lots data in entry_prices file; falling back to broker data')
                    return None
                # Old format (flat list) — cannot infer per-lot quantities
                logger.info('Old entry_prices format detected; falling back to broker data')
                return None
        except Exception as e:
            logger.error(f'Failed to load entry prices: {e}', exc_info=True)
        return None

    def _clear_entry_prices_file(self):
        """Remove persisted entry prices file."""
        path = self._entry_prices_path()
        if os.path.exists(path):
            os.remove(path)

    def _update_lots_after_sell(self, filled_qty):
        """Update entry_lots using FIFO after a sell, and sync position/prices (MJ-002)."""
        remaining_to_sell = filled_qty
        while remaining_to_sell > 0 and self.entry_lots:
            lot = self.entry_lots[0]
            if lot['shares'] <= remaining_to_sell:
                remaining_to_sell -= lot['shares']
                self.entry_lots.pop(0)
            else:
                lot['shares'] -= remaining_to_sell
                remaining_to_sell = 0

        self.current_position -= filled_qty
        if self.current_position <= 0:
            self.current_position = 0
            self.entry_lots = []
            self.entry_prices = []
            self._clear_entry_prices_file()
        else:
            self.entry_prices = [lot['price'] for lot in self.entry_lots]
            self._save_entry_prices()

    # --- MJ-003: Acted signals persistence ---

    def _acted_signals_path(self):
        """Return the file path for persisted acted signals."""
        return f'trade/acted_signals_{self.symbol}.json'

    def _save_acted_signals(self):
        """Persist acted signals to disk."""
        try:
            data = [list(item) for item in self._acted_signals]
            with open(self._acted_signals_path(), 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f'Failed to save acted signals: {e}', exc_info=True)

    def _load_acted_signals(self):
        """Load acted signals from disk. Returns set of (type, date_str) tuples."""
        try:
            path = self._acted_signals_path()
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return {tuple(item) for item in data if isinstance(item, list) and len(item) == 2}
        except Exception as e:
            logger.error(f'Failed to load acted signals: {e}', exc_info=True)
        return set()


def main():
    parser = argparse.ArgumentParser(description='Market Breadth Trading')
    parser.add_argument('--short_ma', type=int, default=5, help='Short-term moving average period (default: 5)')
    parser.add_argument('--long_ma', type=int, default=200, help='Long-term moving average period (default: 200)')
    parser.add_argument(
        '--initial_capital', type=float, default=50000, help='Initial investment amount (default: 50000 dollars)'
    )
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage (default: 0.05%)')
    parser.add_argument('--commission', type=float, default=0.0001, help='Transaction fee (default: 0.01%)')
    parser.add_argument('--use_saved_data', action='store_true', help='Whether to use saved data')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--ma_type', type=str, default='ema', help='Moving average type (default: ema)')
    parser.add_argument('--symbol', type=str, default='SSO', help='Stock symbol (default: SSO)')
    parser.add_argument('--stop_loss_pct', type=float, default=0.08, help='Stop loss percentage (default: 8%)')
    parser.add_argument('--disable_short_ma_entry', action='store_true', help='Disable short-term moving average entry')
    parser.add_argument(
        '--closing_time_minutes',
        type=int,
        default=20,
        help='Minutes before market close to execute trades (default: 20)',
    )
    parser.add_argument('--testmode', action='store_true', help='Enable test mode (no actual trading)')
    parser.add_argument('--test_date', type=str, help='Test date in YYYY-MM-DD format (required for test mode)')
    # TV mode pivot detection parameters
    parser.add_argument('--pivot_len_long', type=int, default=20, help='Pivot window for long MA (default: 20)')
    parser.add_argument('--pivot_len_short', type=int, default=10, help='Pivot window for short MA (default: 10)')
    parser.add_argument(
        '--prom_thresh_long', type=float, default=0.005, help='Prominence threshold for long MA (default: 0.005)'
    )
    parser.add_argument(
        '--prom_thresh_short', type=float, default=0.03, help='Prominence threshold for short MA (default: 0.03)'
    )
    parser.add_argument('--peak_level', type=float, default=0.70, help='Peak exit level threshold (default: 0.70)')
    parser.add_argument(
        '--trough_level_long', type=float, default=0.40, help='Long MA trough entry level (default: 0.40)'
    )
    parser.add_argument(
        '--trough_level_short', type=float, default=0.20, help='Short MA trough entry level (default: 0.20)'
    )
    parser.add_argument(
        '--allow_pyramiding', action='store_true', default=False, help='Allow pyramiding (default: disabled)'
    )

    args = parser.parse_args()

    if args.testmode and not args.test_date:
        parser.error('--test_date is required when --testmode is enabled')

    trader = MarketBreadthTrader(
        short_ma=args.short_ma,
        long_ma=args.long_ma,
        initial_capital=args.initial_capital,
        slippage=args.slippage,
        commission=args.commission,
        use_saved_data=args.use_saved_data,
        debug=args.debug,
        ma_type=args.ma_type,
        symbol=args.symbol,
        stop_loss_pct=args.stop_loss_pct,
        disable_short_ma_entry=args.disable_short_ma_entry,
        closing_time_minutes=args.closing_time_minutes,
        testmode=args.testmode,
        test_date=args.test_date,
        pivot_len_long=args.pivot_len_long,
        pivot_len_short=args.pivot_len_short,
        prom_thresh_long=args.prom_thresh_long,
        prom_thresh_short=args.prom_thresh_short,
        peak_level=args.peak_level,
        trough_level_long=args.trough_level_long,
        trough_level_short=args.trough_level_short,
        no_pyramiding=not args.allow_pyramiding,
    )

    trader.run()


if __name__ == '__main__':
    main()
