import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from scipy.signal import find_peaks
import pathlib
import sys
import os
import time
import logging
from zoneinfo import ZoneInfo
import alpaca_trade_api as tradeapi
from alpaca_trade_api.common import URL
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path to import market_breadth
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_breadth import (
    get_sp500_price_data,
    calculate_above_ma,
    setup_matplotlib_backend,
    get_sp500_tickers_from_wikipedia,
    load_stock_data,
    save_stock_data,
    convert_ticker_symbol,
    plot_breadth_and_sp500_with_peaks,
    calculate_trend_with_hysteresis,
    get_multiple_stock_data,
    get_stock_price_data
)

# Log settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trade/market_breadth_trade.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("market_breadth_trade")

# Timezone settings
TZ_NY = ZoneInfo("US/Eastern")
TZ_UTC = ZoneInfo('UTC')

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Create necessary directories
reports_dir = pathlib.Path('reports')
reports_dir.mkdir(exist_ok=True)

class MarketBreadthTrader:
    def __init__(self, short_ma=8, long_ma=200, initial_capital=50000, slippage=0.001,
                 commission=0.001, use_saved_data=False, debug=False,
                 threshold=0.5, ma_type='ema', symbol='SSO', stop_loss_pct=0.10,
                 disable_short_ma_entry=False, use_trailing_stop=False, trailing_stop_pct=0.2,
                 background_exit_threshold=0.5, use_background_color_signals=False,
                 partial_exit=False, closing_time_minutes=20):
        self.symbol = symbol
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.commission = commission
        self.use_saved_data = use_saved_data
        self.debug = debug
        self.threshold = threshold
        self.ma_type = ma_type.lower()  # 'ema' or 'sma'
        self.stop_loss_pct = stop_loss_pct
        self.disable_short_ma_entry = disable_short_ma_entry
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.background_exit_threshold = background_exit_threshold
        self.use_background_color_signals = use_background_color_signals
        self.partial_exit = partial_exit
        self.closing_time_minutes = closing_time_minutes
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
        
        # Initialize variables
        self.current_position = 0
        self.entry_prices = []
        self.stop_loss_prices = []
        self.highest_price = None
        
        # Initialize signal-related variables
        self.short_ma_bottoms = []
        self.long_ma_bottoms = []
        self.peaks = []
        
        logger.info(f"MarketBreadthTrader initialized with symbol: {self.symbol}")
    
    def is_closing_time_range(self, range_minutes=20):
        """Check if current time is within the specified minutes before market close"""
        current_dt = datetime.now().astimezone(TZ_NY)
        
        cal = self.api.get_calendar(start=str(current_dt.date()), end=str(current_dt.date()))
        
        if len(cal) > 0:
            close_time = cal[0].close
            if isinstance(close_time, str):
                close_hour, close_minute = map(int, close_time.split(':'))
                close_dt = datetime.combine(
                    current_dt.date(),
                    datetime.strptime(close_time, '%H:%M').time(),
                    tzinfo=TZ_NY
                )
            else:
                close_dt = datetime.combine(current_dt.date(), close_time, tzinfo=TZ_NY)
        else:
            logger.info("Market will not open on the date.")
            return False
        
        if close_dt - timedelta(minutes=range_minutes) <= current_dt < close_dt:
            logger.info("In closing time range")
            return True
        else:
            logger.info(f"{current_dt}, it's not in closing time range")
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
            logger.info(f"No position found for {self.symbol}: {e}")
            return 0
    
    def get_current_price(self):
        """Get current price"""
        try:
            logger.info(f"Starting: Getting current price for {self.symbol}")
            
            # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
            # Get latest stock price
            bars = self.api.get_latest_bar(self.symbol)
            if bars and hasattr(bars, 'c'):
                logger.info(f"Success: Got current price for {self.symbol}: ${bars.c:.2f}")
                return float(bars.c)
            else:
                logger.error(f"Failed: Could not get current price for {self.symbol} (no valid bar data)")
                return None
        except Exception as e:
            logger.error(f"Error: Error occurred while getting current price for {self.symbol}: {e}")
            return None
    
    def execute_buy(self, shares, reason=""):
        """Execute buy order"""
        try:
            # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day'
            )
            logger.info(f"Buy order executed: {shares} shares of {self.symbol}, reason: {reason}")
            return order
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
            return None
    
    def execute_sell(self, shares, reason=""):
        """Execute sell order"""
        try:
            # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=shares,
                side='sell',
                type='market',
                time_in_force='day'
            )
            logger.info(f"Sell order executed: {shares} shares of {self.symbol}, reason: {reason}")
            return order
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            return None
    
    def run(self):
        """Execute trading"""
        logger.info("Starting market breadth trading...")
        
        while True:
            # Check if market is open
            if not self.is_market_open():
                logger.info("Market is closed today. Exiting trading.")
                break
            
            # Check if within closing time range
            if self.is_closing_time_range(self.closing_time_minutes):
                try:
                    # Analyze market data
                    logger.info("Analyzing market data...")
                    self.analyze_market()
                    
                    # Check signals and execute trades
                    logger.info("Checking signals and executing trades...")
                    self.check_signals_and_trade()
                    
                    logger.info("Trading completed for today.")
                    break
                except Exception as e:
                    logger.error(f"Error during trading: {str(e)}")
                    break
            
            # Wait for 1 minute
            logger.info("Waiting for closing time range...")
            time.sleep(60)
        
        logger.info("Trading session ended.")
    
    def analyze_market(self):
        """Analyze market data"""
        try:
            logger.info("Starting market data analysis")
            
            # Get past data (using yesterday's date for EODHD)
            today = datetime.now()
            yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logger.info(f"Data retrieval period: {start_date} to {yesterday}")
            
            # Get S&P500 ticker list
            logger.info("Getting S&P500 ticker list...")
            sp500_tickers = get_sp500_tickers_from_wikipedia()
            logger.info(f"Got S&P500 ticker list: {len(sp500_tickers)} tickers")
            
            # Get historical data from EODHD - using data up to yesterday
            logger.info("Getting historical data from EODHD...")
            historical_data = get_multiple_stock_data(sp500_tickers, start_date, yesterday, use_saved_data=True)
            logger.info(f"Got historical data: {len(historical_data.columns)} tickers, {len(historical_data)} days")
            
            # Get today's latest data from Alpaca
            logger.info("Getting today's latest data from Alpaca...")
            today_data = self._get_latest_prices_from_alpaca(sp500_tickers)
            logger.info(f"Got today's latest data: {len(today_data.columns)} tickers")
            
            # Combine historical and today's data
            logger.info("Combining data...")
            all_data = pd.concat([historical_data, today_data])
            logger.info(f"Combined data: {len(all_data.columns)} tickers, {len(all_data)} days")
            
            # Calculate Market Breadth Index
            logger.info("Calculating Market Breadth Index...")
            self.above_ma = calculate_above_ma(all_data)
            logger.info("Market Breadth Index calculation completed")
            
            # Calculate moving averages
            logger.info("Calculating moving averages...")
            self.breadth_index = self.above_ma.mean(axis=1)
            
            if self.ma_type == 'ema':
                self.short_ma_line = self.breadth_index.ewm(span=self.short_ma, adjust=False).mean()
                self.long_ma_line = self.breadth_index.ewm(span=self.long_ma, adjust=False).mean()
            else:
                self.short_ma_line = self.breadth_index.rolling(window=self.short_ma).mean()
                self.long_ma_line = self.breadth_index.rolling(window=self.long_ma).mean()
            
            logger.info(f"Moving average calculation completed: {self.ma_type.upper()} {self.short_ma} days, {self.ma_type.upper()} {self.long_ma} days")
            
            # Calculate trend
            logger.info("Calculating trend...")
            self.long_ma_trend = pd.Series(
                calculate_trend_with_hysteresis(self.long_ma_line),
                index=self.long_ma_line.index
            )
            logger.info("Trend calculation completed")
            
            # Extract price data for specified symbol
            logger.info(f"Extracting price data for {self.symbol}...")
            if self.symbol in all_data.columns:
                self.price_data = pd.DataFrame(all_data[self.symbol], columns=['adjusted_close'])
                logger.info(f"Extracted price data for {self.symbol}: {len(self.price_data)} days")
            else:
                # Get data individually if symbol not included
                logger.info(f"{self.symbol} not included in data, getting data individually...")
                self.price_data = self._get_latest_price_from_alpaca(self.symbol)
                if isinstance(self.price_data, pd.Series):
                    self.price_data = pd.DataFrame(self.price_data, columns=['adjusted_close'])
                    logger.info(f"Got price data for {self.symbol}: {len(self.price_data)} days")
            
            # Detect signals
            logger.info("Detecting signals...")
            self._detect_signals()
            logger.info("Signal detection completed")
            
            logger.info("Market data analysis completed")
            
        except Exception as e:
            logger.error(f"Error during market data analysis: {str(e)}")
            raise
    
    def _get_latest_prices_from_alpaca(self, tickers):
        """Get latest stock prices using Alpaca API"""
        try:
            # Get current date (Timestamp type)
            today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
            
            # Temporarily store price data in dictionary
            price_dict = {}
            
            # Variables for progress display
            total_tickers = len(tickers)
            processed_tickers = 0
            success_count = 0
            failure_count = 0
            
            logger.info(f"Starting: Getting latest prices for {total_tickers} tickers")
            
            # Show progress using progress bar
            for ticker in tqdm(tickers, desc="Getting prices", unit="ticker"):
                try:
                    # Show progress
                    processed_tickers += 1
                    
                    # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
                    # Get latest stock price
                    bars = self.api.get_latest_bar(ticker)
                    if bars and hasattr(bars, 'c'):
                        price_dict[ticker] = bars.c
                        success_count += 1
                    else:
                        logger.warning(f"Failed to get latest price for {ticker} (no valid bar data)")
                        failure_count += 1
                except Exception as e:
                    logger.warning(f"Error getting latest price for {ticker}: {str(e)}")
                    failure_count += 1
            
            # Convert all price data to DataFrame at once
            latest_prices = pd.DataFrame(price_dict, index=[today])
            
            logger.info(f"Completed: Processed {total_tickers} tickers (Success: {success_count}, Failure: {failure_count})")
            logger.info(f"Retrieved price data: {len(price_dict)} tickers")
            
            return latest_prices
            
        except Exception as e:
            logger.error(f"Error getting latest prices from Alpaca: {str(e)}")
            raise
    
    def _get_latest_price_from_alpaca(self, ticker):
        """Get latest stock price for specific ticker using Alpaca API"""
        try:
            logger.info(f"Starting: Getting latest price for {ticker}")
            
            # No need to convert symbol for Alpaca (Alpaca uses original symbol format)
            # Get latest stock price
            bars = self.api.get_latest_bar(ticker)
            if bars and hasattr(bars, 'c'):
                logger.info(f"Success: Got latest price for {ticker}: ${bars.c:.2f}")
                return pd.Series([bars.c], index=[pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))])
            else:
                logger.warning(f"Failed: Could not get latest price for {ticker} (no valid bar data)")
                return pd.Series()
                
        except Exception as e:
            logger.error(f"Error: Error occurred while getting latest price for {ticker}: {str(e)}")
            return pd.Series()
    
    def _get_sp500_data(self):
        """Get data for all S&P500 stocks"""
        filename = 'sp500_all_stocks.csv'
        
        # Set calculation start date (2 years before current date)
        today = datetime.now()
        yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        calculation_start_date = (pd.to_datetime(today.strftime('%Y-%m-%d')) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        
        # Load saved data
        if self.use_saved_data:
            saved_data = load_stock_data(filename)
            if saved_data is not None and not saved_data.empty:
                # Check date range and data quality
                if (pd.to_datetime(calculation_start_date) >= saved_data.index.min() and 
                    pd.to_datetime(yesterday) <= saved_data.index.max()):
                    # Check for missing or invalid data
                    if saved_data.isnull().sum().sum() == 0:
                        # Extract data for calculation period
                        mask = (saved_data.index >= pd.to_datetime(calculation_start_date)) & \
                              (saved_data.index <= pd.to_datetime(yesterday))
                        return saved_data.loc[mask]
                    else:
                        logger.warning("Saved data contains missing values, fetching fresh data")
        
        # Get S&P500 ticker list
        tickers = get_sp500_tickers_from_wikipedia()
        logger.info(f"Number of tickers retrieved: {len(tickers)}")
        
        # Get data for all stocks (from calculation start date to yesterday)
        all_data = get_multiple_stock_data(
            tickers,
            calculation_start_date,
            yesterday,
            use_saved_data=self.use_saved_data
        )
        
        if not all_data.empty:
            # Check data quality
            if all_data.isnull().sum().sum() > 0:
                logger.warning("Fetched data contains missing values, attempting to fill")
                # Forward fill missing values
                all_data = all_data.fillna(method='ffill')
                # Backward fill any remaining missing values
                all_data = all_data.fillna(method='bfill')
            
            # Save data
            save_stock_data(all_data, filename)
            # Extract data for calculation period
            mask = (all_data.index >= pd.to_datetime(calculation_start_date)) & \
                  (all_data.index <= pd.to_datetime(yesterday))
            return all_data.loc[mask]
        
        return pd.DataFrame()
    
    def _detect_signals(self):
        """Detect signals"""
        try:
            # Initialize signal detection variables
            self.short_ma_bottoms = []
            self.long_ma_bottoms = []
            self.peaks = []
            
            # Variables to record detected signals
            detected_short_ma_bottoms = set()
            detected_long_ma_bottoms = set()
            detected_peaks = set()
            
            # Get today's date
            today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
            
            # Calculate start date for signal detection (2 years before today)
            start_date = today - pd.DateOffset(years=2)
            
            # Filter data for signal detection period
            mask = (self.short_ma_line.index >= start_date) & (self.short_ma_line.index <= today)
            filtered_short_ma = self.short_ma_line.loc[mask]
            filtered_long_ma = self.long_ma_line.loc[mask]
            
            # Check data quality
            if filtered_short_ma.isnull().sum() > 0 or filtered_long_ma.isnull().sum() > 0:
                logger.error("Moving average data contains missing values")
                raise ValueError("Moving average data contains missing values")
            
            # Debug: Print breadth_index data details
            logger.info("Debug: breadth_index data details")
            logger.info(f"Total data points: {len(self.breadth_index)}")
            logger.info(f"First 5 dates: {[d.strftime('%Y-%m-%d') if not pd.isna(d) else 'NaT' for d in self.breadth_index.index[:5]]}")
            logger.info(f"Last 5 dates: {[d.strftime('%Y-%m-%d') if not pd.isna(d) else 'NaT' for d in self.breadth_index.index[-5:]]}")
            logger.info(f"Date range: {self.breadth_index.index[0].strftime('%Y-%m-%d') if not pd.isna(self.breadth_index.index[0]) else 'NaT'} to {self.breadth_index.index[-1].strftime('%Y-%m-%d') if not pd.isna(self.breadth_index.index[-1]) else 'NaT'}")
            
            # Process data sequentially from past to present
            for i in range(len(filtered_short_ma)):
                try:
                    current_date = filtered_short_ma.index[i]
                    if pd.isna(current_date):
                        logger.warning(f"Found NaT date at index {i}, skipping...")
                        continue
                        
                    current_data = filtered_short_ma.iloc[:i+1]
                    
                    # Detect 8MA bottoms (only if disable_short_ma_entry is False)
                    if not self.disable_short_ma_entry and len(current_data) > self.short_ma:
                        # Extract data points below threshold
                        below_threshold_short = current_data[current_data < self.threshold]
                        
                        if not below_threshold_short.empty:
                            # Keep original indices
                            original_indices = np.where(current_data < self.threshold)[0]
                            bottoms_short, _ = find_peaks(
                                -below_threshold_short.values,
                                prominence=0.02
                            )
                            
                            # Process detected bottoms
                            for bottom_idx in bottoms_short:
                                try:
                                    # Get index position in original data
                                    original_idx = original_indices[bottom_idx]
                                    bottom_date = current_data.index[original_idx]
                                    
                                    if pd.isna(bottom_date):
                                        logger.warning(f"Found NaT bottom date at index {original_idx}, skipping...")
                                        continue
                                    
                                    # Calculate minimum Market Breadth for past 20 days
                                    # Get the index of bottom_date in breadth_index
                                    try:
                                        bottom_idx_in_breadth = self.breadth_index.index.get_loc(bottom_date)
                                        
                                        # Calculate start and end indices for past 20 days
                                        start_idx = max(0, bottom_idx_in_breadth - 20)
                                        end_idx = bottom_idx_in_breadth + 1
                                        
                                        # Get past 20 days data
                                        past_20days_data = self.breadth_index.iloc[start_idx:end_idx]
                                        past_20days_min = past_20days_data.min()
                                        
                                        # Debug: Print data for past 20 days calculation
                                        logger.info(f"Debug: Past 20 days data for {bottom_date.strftime('%Y-%m-%d')}")
                                        logger.info(f"  Data range: {past_20days_data.index[0].strftime('%Y-%m-%d') if not pd.isna(past_20days_data.index[0]) else 'NaT'} to {past_20days_data.index[-1].strftime('%Y-%m-%d') if not pd.isna(past_20days_data.index[-1]) else 'NaT'}")
                                        logger.info(f"  Data points: {len(past_20days_data)}")
                                        logger.info(f"  Minimum value: {past_20days_min:.4f}")
                                        logger.info(f"  Data values: {past_20days_data.values}")
                                        
                                        if past_20days_min <= 0.3:  # Check actual value condition
                                            # Add only if bottom not already detected
                                            if bottom_date not in detected_short_ma_bottoms:
                                                detected_short_ma_bottoms.add(bottom_date)
                                                # Use the date when the bottom was first detected as signal date
                                                signal_date = current_date
                                                self.short_ma_bottoms.append(signal_date)
                                                logger.info(f"New {self.short_ma}{self.ma_type.upper()} bottom detected at: {bottom_date.strftime('%Y-%m-%d')}")
                                                logger.info(f"  Signal date: {signal_date.strftime('%Y-%m-%d')}")
                                                logger.info(f"  Bottom value: {current_data.iloc[original_idx]:.4f}")
                                                logger.info(f"  Past 20 days minimum: {past_20days_min:.4f}")
                                    except KeyError:
                                        logger.warning(f"Date {bottom_date.strftime('%Y-%m-%d')} not found in breadth_index")
                                        continue
                                except Exception as e:
                                    logger.error(f"Error processing bottom at index {bottom_idx}: {str(e)}")
                                    continue
                    
                    # Detect 200MA bottoms
                    current_long_data = filtered_long_ma.iloc[:i+1]
                    if len(current_long_data) > self.long_ma:
                        bottoms_long, _ = find_peaks(
                            -current_long_data.values,
                            prominence=0.015
                        )
                        
                        # Process detected bottoms
                        for bottom_idx in bottoms_long:
                            try:
                                bottom_date = current_long_data.index[bottom_idx]
                                
                                if pd.isna(bottom_date):
                                    logger.warning(f"Found NaT bottom date at index {bottom_idx}, skipping...")
                                    continue
                                
                                # Get index position in original data
                                original_idx = bottom_idx
                                
                                # Calculate minimum Market Breadth for past 20 days
                                if original_idx >= 20:  # Only check if we have 20 days of past data
                                    past_20days_min = self.breadth_index.iloc[original_idx-20:original_idx+1].min()
                                    if past_20days_min <= 0.5:  # Check actual value condition
                                        # Add only if bottom not already detected
                                        if bottom_date not in detected_long_ma_bottoms:
                                            detected_long_ma_bottoms.add(bottom_date)
                                            # Use the date when the bottom was first detected as signal date
                                            signal_date = current_date
                                            self.long_ma_bottoms.append(signal_date)
                                            logger.info(f"New {self.long_ma}{self.ma_type.upper()} bottom detected at: {bottom_date.strftime('%Y-%m-%d')}")
                                            logger.info(f"  Signal date: {signal_date.strftime('%Y-%m-%d')}")
                                            logger.info(f"  Bottom value: {current_long_data.iloc[original_idx]:.4f}")
                                            logger.info(f"  Past 20 days minimum: {past_20days_min:.4f}")
                            except Exception as e:
                                logger.error(f"Error processing long MA bottom at index {bottom_idx}: {str(e)}")
                                continue
                    
                    # Detect 200MA peaks
                    if len(current_long_data) > self.long_ma:
                        peaks, _ = find_peaks(
                            current_long_data.values,
                            prominence=0.015
                        )
                        
                        # Process detected peaks
                        for peak_idx in peaks:
                            try:
                                peak_date = current_long_data.index[peak_idx]
                                
                                if pd.isna(peak_date):
                                    logger.warning(f"Found NaT peak date at index {peak_idx}, skipping...")
                                    continue
                                
                                # Verify 200MA value is above 0.5
                                if current_long_data.iloc[peak_idx] >= 0.5:
                                    # Add only if peak not already detected
                                    if peak_date not in detected_peaks:
                                        detected_peaks.add(peak_date)
                                        # Use the date when the peak was first detected as signal date
                                        signal_date = current_date
                                        self.peaks.append(signal_date)
                                        logger.info(f"New {self.long_ma}{self.ma_type.upper()} peak detected at: {peak_date.strftime('%Y-%m-%d')}")
                                        logger.info(f"  Signal date: {signal_date.strftime('%Y-%m-%d')}")
                                        logger.info(f"  Peak value: {current_long_data.iloc[peak_idx]:.4f}")
                            except Exception as e:
                                logger.error(f"Error processing peak at index {peak_idx}: {str(e)}")
                                continue
                except Exception as e:
                    logger.error(f"Error processing data at index {i}: {str(e)}")
                    continue
            
            logger.info("Signal detection completed successfully")
            
        except Exception as e:
            logger.error(f"Error detecting signals: {str(e)}")
            raise
    
    def check_signals_and_trade(self):
        """Check signals and execute trades"""
        logger.info("Checking signals and executing trades")
        
        # Get current position
        self.current_position = self.get_current_position()
        logger.info(f"Current position: {self.current_position} shares")
        
        # Get current price
        current_price = self.get_current_price()
        if current_price is None:
            logger.error("Failed to get current price. Exiting.")
            return
        
        logger.info(f"Current price: ${current_price:.2f}")
        
        # Current date
        current_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        
        # Check for signals
        has_short_ma_bottom = current_date in self.short_ma_bottoms and not self.disable_short_ma_entry
        has_long_ma_bottom = current_date in self.long_ma_bottoms
        has_peak = current_date in self.peaks
        
        # Log signal detection status
        logger.info("Signal detection status:")
        logger.info(f"  Short MA bottom signal: {'Detected' if has_short_ma_bottom else 'Not detected'}")
        logger.info(f"  Long MA bottom signal: {'Detected' if has_long_ma_bottom else 'Not detected'}")
        logger.info(f"  Peak signal: {'Detected' if has_peak else 'Not detected'}")
        
        # Check if current date matches signal date
        if has_short_ma_bottom:
            # Entry at 8MA bottom
            logger.info(f"Short MA bottom signal detected for {current_date.strftime('%Y-%m-%d')}")
            
            # Enter if no position
            if self.current_position == 0:
                # Use half of available capital
                account = self.api.get_account()
                available_capital = float(account.buying_power)
                entry_amount = available_capital / 2
                
                # Calculate number of shares to buy
                shares = self._calculate_shares(entry_amount, current_price)
                
                if shares > 0:
                    # Execute entry
                    order = self.execute_buy(shares, reason="short_ma_bottom")
                    if order:
                        logger.info(f"Entry executed at short MA bottom: {shares} shares at ${current_price:.2f}")
                        # Initialize highest price
                        self.highest_price = current_price
                    else:
                        logger.error("Failed to execute entry at short MA bottom")
                else:
                    logger.info("No shares to buy due to insufficient capital")
            else:
                logger.info(f"Already have position ({self.current_position} shares), not entering at short MA bottom")
        
        elif has_long_ma_bottom:
            # Entry at 200MA bottom
            logger.info(f"Long MA bottom signal detected for {current_date.strftime('%Y-%m-%d')}")
            
            # Enter if no position
            if self.current_position == 0:
                # Use all available capital
                account = self.api.get_account()
                available_capital = float(account.buying_power)
                
                # Calculate number of shares to buy
                shares = self._calculate_shares(available_capital, current_price)
                
                if shares > 0:
                    # Execute entry
                    order = self.execute_buy(shares, reason="long_ma_bottom")
                    if order:
                        logger.info(f"Entry executed at long MA bottom: {shares} shares at ${current_price:.2f}")
                        # Initialize highest price
                        self.highest_price = current_price
                    else:
                        logger.error("Failed to execute entry at long MA bottom")
                else:
                    logger.info("No shares to buy due to insufficient capital")
            else:
                logger.info(f"Already have position ({self.current_position} shares), not entering at long MA bottom")
        
        elif has_peak and self.current_position > 0:
            # Exit at 200MA peak
            logger.info(f"Long MA peak signal detected for {current_date.strftime('%Y-%m-%d')}")
            
            # Execute exit
            order = self.execute_sell(self.current_position, reason="peak exit")
            if order:
                logger.info(f"Exit executed at long MA peak: {self.current_position} shares at ${current_price:.2f}")
                # Reset position
                self.current_position = 0
                self.entry_prices = []
                self.stop_loss_prices = []
                self.highest_price = None
            else:
                logger.error("Failed to execute exit at long MA peak")
        else:
            logger.info("No trading signals detected for today")
        
        logger.info("Signal check and trade execution completed")
    
    def _calculate_shares(self, amount, price):
        """Calculate number of shares to buy"""
        return int(amount / (price * (1 + self.slippage)))

def main():
    parser = argparse.ArgumentParser(description='Market Breadth Trading')
    parser.add_argument('--short_ma', type=int, default=8,
                      help='Short-term moving average period (default: 8)')
    parser.add_argument('--long_ma', type=int, default=200,
                      help='Long-term moving average period (default: 200)')
    parser.add_argument('--initial_capital', type=float, default=50000,
                      help='Initial investment amount (default: 50000 dollars)')
    parser.add_argument('--slippage', type=float, default=0.001,
                      help='Slippage (default: 0.1%)')
    parser.add_argument('--commission', type=float, default=0.001,
                      help='Transaction fee (default: 0.1%)')
    parser.add_argument('--use_saved_data', action='store_true',
                      help='Whether to use saved data')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for bottom detection (default: 0.5)')
    parser.add_argument('--ma_type', type=str, default='ema',
                      help='Moving average type (default: ema)')
    parser.add_argument('--symbol', type=str, default='SSO',
                      help='Stock symbol (default: SSO)')
    parser.add_argument('--stop_loss_pct', type=float, default=0.08,
                      help='Stop loss percentage (default: 8%)')
    parser.add_argument('--disable_short_ma_entry', action='store_true',
                      help='Disable short-term moving average entry')
    parser.add_argument('--use_trailing_stop', action='store_true',
                      help='Use trailing stop instead of fixed stop loss')
    parser.add_argument('--trailing_stop_pct', type=float, default=0.2,
                      help='Trailing stop percentage (default: 20%)')
    parser.add_argument('--background_exit_threshold', type=float, default=0.5,
                      help='Background exit threshold (default: 0.5)')
    parser.add_argument('--use_background_color_signals', action='store_true',
                      help='Use background color change signals for entry and exit')
    parser.add_argument('--partial_exit', action='store_true',
                      help='Exit with half of the position when exit signal is triggered')
    parser.add_argument('--closing_time_minutes', type=int, default=20,
                      help='Minutes before market close to execute trades (default: 20)')
    
    args = parser.parse_args()
    
    trader = MarketBreadthTrader(
        short_ma=args.short_ma,
        long_ma=args.long_ma,
        initial_capital=args.initial_capital,
        slippage=args.slippage,
        commission=args.commission,
        use_saved_data=args.use_saved_data,
        debug=args.debug,
        threshold=args.threshold,
        ma_type=args.ma_type,
        symbol=args.symbol,
        stop_loss_pct=args.stop_loss_pct,
        disable_short_ma_entry=args.disable_short_ma_entry,
        use_trailing_stop=args.use_trailing_stop,
        trailing_stop_pct=args.trailing_stop_pct,
        background_exit_threshold=args.background_exit_threshold,
        use_background_color_signals=args.use_background_color_signals,
        partial_exit=args.partial_exit,
        closing_time_minutes=args.closing_time_minutes
    )
    
    trader.run()

if __name__ == '__main__':
    main() 