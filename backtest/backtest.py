import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
from datetime import datetime
import argparse
from scipy.signal import find_peaks
import pathlib
import sys
import os

# Add parent directory to path to import market_breadth
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_breadth import (
    get_sp500_price_data,
    calculate_above_ma,
    get_sp500_tickers_from_fmp,
    load_stock_data,
    save_stock_data,
    convert_ticker_symbol,
    plot_breadth_and_sp500_with_peaks,
    calculate_trend_with_hysteresis,
    get_multiple_stock_data,
    get_stock_price_data
)


def _setup_matplotlib_backend():
    """Set up matplotlib backend based on the operating system."""
    system = platform.system().lower()
    if system in ('darwin', 'windows'):
        try:
            matplotlib.use('TkAgg')
        except (ImportError, ModuleNotFoundError):
            matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')

# Create necessary directories
reports_dir = pathlib.Path('reports')
reports_dir.mkdir(exist_ok=True)

class Backtest:
    def __init__(self, start_date=None, end_date=None,
                 short_ma=8, long_ma=200, initial_capital=50000, slippage=0.001,
                 commission=0.001, use_saved_data=False, debug=False,
                 threshold=0.5, ma_type='ema', symbol='SSO', stop_loss_pct=0.10,
                 disable_short_ma_entry=False, use_trailing_stop=False, trailing_stop_pct=0.2,
                 background_exit_threshold=0.5, use_background_color_signals=False,
                 partial_exit=False, no_show_plot=False):
        self.symbol = symbol  # Changed to allow symbol specification
        self.start_date = start_date
        self.end_date = end_date
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.commission = commission
        self.use_saved_data = use_saved_data
        self.debug = debug
        self.threshold = threshold
        self.ma_type = ma_type.lower()  # 'ema' or 'sma'
        self.stop_loss_pct = stop_loss_pct  # Percentage for stop loss
        self.disable_short_ma_entry = disable_short_ma_entry  # Option to disable entry by short_ma
        self.use_trailing_stop = use_trailing_stop  # Whether to use trailing stop
        self.trailing_stop_pct = trailing_stop_pct  # Percentage for trailing stop
        self.background_exit_threshold = background_exit_threshold  # Exit threshold when background color changes
        self.use_background_color_signals = use_background_color_signals  # Whether to use signals based on background color changes
        self.partial_exit = partial_exit  # Whether to sell only half of the position on exit
        self.no_show_plot = no_show_plot  # Whether to not show the plot
        
        # Variables to store backtest results
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_capital = initial_capital
        self.current_position = 0
        self.entry_prices = []  # List to record entry prices
        self.stop_loss_prices = []  # List to record stop loss prices
        self.highest_price = None  # Highest price during position holding

        # Trade logging variables (Phase 1)
        self.trade_log = []  # Detailed trade log for each complete trade
        self.open_positions = []  # Currently open positions
        self.next_trade_id = 1  # Counter for trade IDs
        
    def run(self):
        """Execute the backtest"""
        # Get data for all S&P500 stocks (including past data for calculation)
        self.sp500_data = self._get_sp500_data()
        
        if self.sp500_data.empty:
            print("Failed to retrieve data.")
            return
            
        print(f"\nData period: {self.sp500_data.index.min()} to {self.sp500_data.index.max()}")
        
        # Extract price data for the specified symbol
        if self.symbol in self.sp500_data.columns:
            self.price_data = pd.DataFrame(self.sp500_data[self.symbol], columns=['adjusted_close'])
        else:
            # If the symbol is not included, retrieve it separately
            calculation_start_date = (pd.to_datetime(self.start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
            self.price_data = get_stock_price_data(
                self.symbol,
                calculation_start_date,  # Use calculation start date
                self.end_date,
                use_saved_data=self.use_saved_data
            )
            if isinstance(self.price_data, pd.Series):
                self.price_data = pd.DataFrame(self.price_data, columns=['adjusted_close'])
        
        # Calculate Breadth Index
        self.above_ma = calculate_above_ma(self.sp500_data)
        
        # Ensure data period consistency
        common_dates = self.price_data.index.intersection(self.above_ma.index)
        self.price_data = self.price_data.loc[common_dates]
        self.above_ma = self.above_ma.loc[common_dates]
        
        # Calculate moving averages
        self.breadth_index = self.above_ma.mean(axis=1)
        
        # Calculate based on the type of moving average
        if self.ma_type == 'ema':
            # Exponential Moving Average (EMA)
            self.short_ma_line = self.breadth_index.ewm(span=self.short_ma, adjust=False).mean()
            self.long_ma_line = self.breadth_index.ewm(span=self.long_ma, adjust=False).mean()
        else:
            # Simple Moving Average (SMA)
            self.short_ma_line = self.breadth_index.rolling(window=self.short_ma).mean()
            self.long_ma_line = self.breadth_index.rolling(window=self.long_ma).mean()
        
        # Calculate trend
        self.long_ma_trend = pd.Series(
            calculate_trend_with_hysteresis(self.long_ma_line),
            index=self.long_ma_line.index
        )
        
        # Extract data for the specified period only (for backtest)
        mask = (self.price_data.index >= pd.to_datetime(self.start_date)) & \
               (self.price_data.index <= pd.to_datetime(self.end_date))
        self.price_data = self.price_data.loc[mask]
        self.breadth_index = self.breadth_index.loc[mask]
        self.short_ma_line = self.short_ma_line.loc[mask]
        self.long_ma_line = self.long_ma_line.loc[mask]
        self.long_ma_trend = self.long_ma_trend.loc[mask]

        # Sort all data by date in ascending order (oldest to newest)
        self.price_data = self.price_data.sort_index()
        self.breadth_index = self.breadth_index.sort_index()
        self.short_ma_line = self.short_ma_line.sort_index()
        self.long_ma_line = self.long_ma_line.sort_index()
        self.long_ma_trend = self.long_ma_trend.sort_index()
        
        # Initialize variables for signal detection
        self.short_ma_bottoms = []
        self.long_ma_bottoms = []
        self.peaks = []
                
        # Execute trades
        self.execute_trades()
        
        # Calculate performance
        self.calculate_performance()
        
        # Visualize results
        self.visualize_results(show_plot=not self.no_show_plot)
    
    def _get_sp500_data(self):
        """Retrieve data for all S&P500 stocks"""
        filename = 'sp500_all_stocks.csv'
        
        # Set calculation start date (2 years before the specified start date)
        calculation_start_date = (pd.to_datetime(self.start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        
        # Try to load saved data
        if self.use_saved_data:
            saved_data = load_stock_data(filename)
            if saved_data is not None and not saved_data.empty:
                # Check date range (from calculation start date to end date)
                if (pd.to_datetime(calculation_start_date) >= saved_data.index.min() and 
                    pd.to_datetime(self.end_date) <= saved_data.index.max()):
                    # Extract data for the calculation period
                    mask = (saved_data.index >= pd.to_datetime(calculation_start_date)) & \
                          (saved_data.index <= pd.to_datetime(self.end_date))
                    return saved_data.loc[mask]
        
        # Get S&P500 ticker list
        tickers = get_sp500_tickers_from_fmp()
        print(f"Number of tickers retrieved: {len(tickers)}")
        
        # Get data for all stocks (from calculation start date)
        all_data = get_multiple_stock_data(
            tickers,
            calculation_start_date,  # Use calculation start date
            self.end_date,
            use_saved_data=self.use_saved_data
        )
        
        if not all_data.empty:
            # Save data
            save_stock_data(all_data, filename)
            # Extract data for the calculation period
            mask = (all_data.index >= pd.to_datetime(calculation_start_date)) & \
                  (all_data.index <= pd.to_datetime(self.end_date))
            return all_data.loc[mask]
        
        return pd.DataFrame()
    
    def execute_trades(self):
        """Execute trades"""
        available_capital = self.initial_capital
                
        # 検出済みのシグナルを記録する変数
        detected_short_ma_bottoms = set()
        detected_long_ma_bottoms = set()
        detected_peaks = set()
        
        # Execute trades
        for i, date in enumerate(self.price_data.index):
            price = self.price_data.loc[date, 'adjusted_close']
            
            # Detect signals using data up to the current date
            current_data = self.price_data.iloc[:i+1]
            current_breadth_index = self.breadth_index.iloc[:i+1]
            current_short_ma_line = self.short_ma_line.iloc[:i+1]
            current_long_ma_line = self.long_ma_line.iloc[:i+1]
            
            # Get start and end dates of the data period
            data_start_date = current_short_ma_line.index[0].strftime('%Y-%m-%d')
            data_end_date = current_short_ma_line.index[-1].strftime('%Y-%m-%d')
            
            # Detect 20MA bottoms (only if disable_short_ma_entry is False)
            if not self.disable_short_ma_entry and len(current_short_ma_line) > self.short_ma:
                below_threshold_short = current_short_ma_line[current_short_ma_line < self.threshold]
                if not below_threshold_short.empty:
                    # Preserve original indices
                    original_indices = np.where(current_short_ma_line < self.threshold)[0]
                    bottoms_short, _ = find_peaks(
                        -below_threshold_short.values,
                        prominence=0.02
                    )
                    
                    # Select only those bottoms detected by find_peaks that meet the Market Breadth actual value conditions
                    for bottom_idx in bottoms_short:
                        # Get index position in the original data
                        original_idx = original_indices[bottom_idx]
                        bottom_date = current_short_ma_line.index[original_idx]
                        
                        # Calculate minimum Market Breadth value over the past 20 days
                        if original_idx >= 20:  # Only check if there is data for the past 20 days
                            past_20days_min = current_breadth_index.iloc[original_idx-20:original_idx+1].min()
                            if past_20days_min <= 0.3:  # Check actual value conditions
                                # Only add if the bottom has not been detected yet
                                if bottom_date not in detected_short_ma_bottoms:
                                    detected_short_ma_bottoms.add(bottom_date)
                                    # Use current date as signal date (when we can confirm the bottom)
                                    signal_date = date
                                    self.short_ma_bottoms.append(signal_date)
                                    print(f"New {self.short_ma}{self.ma_type.upper()} bottom detected at: {bottom_date.strftime('%Y-%m-%d')}")
                                    print(f"  Data period: {data_start_date} to {data_end_date}")
                                    print(f"  Signal date (trade execution): {signal_date.strftime('%Y-%m-%d')}")
            
            # Detect 200MA bottoms
            if len(current_long_ma_line) > self.long_ma:
                bottoms_long, _ = find_peaks(
                    -current_long_ma_line.values,
                    prominence=0.015
                )
                for bottom_idx in bottoms_long:
                    bottom_date = current_long_ma_line.index[bottom_idx]
                    
                    # Get index position in the original data
                    original_idx = bottom_idx
                    
                    # Calculate minimum Market Breadth value over the past 20 days
                    if original_idx >= 20:  # Only check if there is data for the past 20 days
                        past_20days_min = current_breadth_index.iloc[original_idx-20:original_idx+1].min()
                        if past_20days_min <= 0.5:  # Check actual value conditions
                            if bottom_date not in detected_long_ma_bottoms:
                                detected_long_ma_bottoms.add(bottom_date)
                                # Use current date as signal date (when we can confirm the bottom)
                                signal_date = date
                                self.long_ma_bottoms.append(signal_date)
                                print(f"New {self.long_ma}{self.ma_type.upper()} bottom detected at: {bottom_date.strftime('%Y-%m-%d')}")
                                print(f"  Data period: {data_start_date} to {data_end_date}")
                                print(f"  Signal date (trade execution): {signal_date.strftime('%Y-%m-%d')}")
            
            # Detect 200MA peaks
            if len(current_long_ma_line) > self.long_ma:
                peaks, _ = find_peaks(
                    current_long_ma_line.values,
                    prominence=0.015
                )
                for peak_idx in peaks:
                    peak_date = current_long_ma_line.index[peak_idx]
                    
                    # Verify that the 200MA value is 0.6 or higher
                    if current_long_ma_line.iloc[peak_idx] >= 0.5:
                        if peak_date not in detected_peaks:
                            detected_peaks.add(peak_date)
                            # Use current date as signal date (when we can confirm the peak)
                            signal_date = date
                            self.peaks.append(signal_date)
                            print(f"New {self.long_ma}{self.ma_type.upper()} peak detected at: {peak_date.strftime('%Y-%m-%d')}")
                            print(f"  Data period: {data_start_date} to {data_end_date}")
                            print(f"  Signal date (trade execution): {signal_date.strftime('%Y-%m-%d')}")
                            print(f"  {self.long_ma}{self.ma_type.upper()} value: {current_long_ma_line.iloc[peak_idx]:.4f}")
            
            # Entry at 20MA bottom (only if disable_short_ma_entry is False)
            if not self.disable_short_ma_entry and date in self.short_ma_bottoms:
                if available_capital > 0:  # Increase position if capital is available
                    entry_amount = available_capital / 2
                    shares = self._calculate_shares(entry_amount, price)
                    if shares > 0:  # Only enter if shares can be purchased
                        self._execute_entry(date, price, shares, reason="short_ma_bottom")
                        available_capital -= entry_amount
                        # Initialize highest price
                        self.highest_price = price
                        print(f"\nEntry at {self.short_ma}{self.ma_type.upper()} bottom (buy more):")
                        print(f"Date: {date.strftime('%Y-%m-%d')}")
                        print(f"Price: ${price:.2f}")
                        print(f"Shares: {shares}")
                        print(f"Investment amount: ${entry_amount:.2f}")
                        print(f"Remaining available capital: ${available_capital:.2f}")
                        print(f"Total position: {self.current_position} shares")
                else:
                    print(f"\n{self.short_ma}{self.ma_type.upper()} bottom detected but no available capital:")
                    print(f"Date: {date.strftime('%Y-%m-%d')}")
                    print(f"Current position: {self.current_position} shares")
                    print(f"Current valuation: ${self.current_position * price:.2f}")
            
            # Entry at 200MA bottom
            elif date in self.long_ma_bottoms:
                if available_capital > 0:  # Increase position if capital is available
                    # For 200MA, use all remaining available capital
                    shares = self._calculate_shares(available_capital, price)
                    if shares > 0:
                        self._execute_entry(date, price, shares, reason="long_ma_bottom")
                        # Initialize highest price
                        self.highest_price = price
                        print(f"\nEntry at {self.long_ma}{self.ma_type.upper()} bottom (buy more):")
                        print(f"Date: {date.strftime('%Y-%m-%d')}")
                        print(f"Price: ${price:.2f}")
                        print(f"Shares: {shares}")
                        print(f"Investment amount: ${available_capital:.2f}")
                        print(f"Total position: {self.current_position} shares")
                        available_capital = 0
                        print(f"Remaining available capital: ${available_capital:.2f}")
                else:
                    print(f"\n{self.long_ma}{self.ma_type.upper()} bottom detected but no available capital:")
                    print(f"Date: {date.strftime('%Y-%m-%d')}")
                    print(f"Current position: {self.current_position} shares")
                    print(f"Current valuation: ${self.current_position * price:.2f}")
            
            # Entry when background changes to white (new condition)
            elif self.use_background_color_signals and self.current_position == 0 and i > 0:
                # Check if background color has changed by comparing with previous day's data
                prev_trend = self.long_ma_trend.iloc[i-1]
                prev_short_ma = self.short_ma_line.iloc[i-1]
                prev_long_ma = self.long_ma_line.iloc[i-1]
                
                # Previous day had pink background (conditions were met)
                prev_condition = (prev_trend == -1 and prev_short_ma < prev_long_ma)
                
                # Today the background is white (conditions are not met)
                current_trend = self.long_ma_trend.iloc[i]
                current_short_ma = self.short_ma_line.iloc[i]
                current_long_ma = self.long_ma_line.iloc[i]
                current_condition = not (current_trend == -1 and current_short_ma < current_long_ma)
                
                # If background color changes from pink to white (previous day met conditions, today does not)
                # And if the 200MA value is above the specified threshold
                if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                    if available_capital > 0:  # Increase position if capital is available
                        # Use all remaining available capital
                        shares = self._calculate_shares(available_capital, price)
                        if shares > 0:
                            self._execute_entry(date, price, shares, reason="background_color_change")
                            # Initialize highest price
                            self.highest_price = price
                            print(f"\nEntry at background color change (pink to white):")
                            print(f"Date: {date.strftime('%Y-%m-%d')}")
                            print(f"Price: ${price:.2f}")
                            print(f"Shares: {shares}")
                            print(f"Investment amount: ${available_capital:.2f}")
                            print(f"Total position: {self.current_position} shares")
                            print(f"Trend: {current_trend}, Short MA: {current_short_ma:.4f}, Long MA: {current_long_ma:.4f}")
                            print(f"Long MA threshold: {self.background_exit_threshold:.2f}")
                            available_capital = 0
                            print(f"Remaining available capital: ${available_capital:.2f}")
                    else:
                        print(f"\nBackground color change (pink to white) detected but no available capital:")
                        print(f"Date: {date.strftime('%Y-%m-%d')}")
                        print(f"Current position: {self.current_position} shares")
                        print(f"Current valuation: ${self.current_position * price:.2f}")
            
            # Exit at peak
            elif date in self.peaks and self.current_position > 0:
                print(f"\nExit at {self.long_ma}{self.ma_type.upper()} peak:")
                print(f"Date: {date.strftime('%Y-%m-%d')}")
                print(f"Price: ${price:.2f}")
                print(f"Shares: {self.current_position}")
                proceeds = self.current_position * price * (1 - self.slippage) * (1 - self.commission)
                print(f"Proceeds (after fees and slippage): ${proceeds:.2f}")
                
                self._execute_exit(date, price, reason="peak exit")
                available_capital = self.current_capital
                print(f"Available capital: ${available_capital:.2f}")
            
            # Exit at the moment background changes to pink (new condition)
            elif self.use_background_color_signals and self.current_position > 0 and i > 0:
                # Check if background color has changed by comparing with previous day's data
                prev_trend = self.long_ma_trend.iloc[i-1]
                prev_short_ma = self.short_ma_line.iloc[i-1]
                prev_long_ma = self.long_ma_line.iloc[i-1]
                
                # Previous day did not have pink background (or did not meet conditions)
                prev_condition = not (prev_trend == -1 and prev_short_ma < prev_long_ma)
                
                # Today the background is pink (conditions are met)
                current_trend = self.long_ma_trend.iloc[i]
                current_short_ma = self.short_ma_line.iloc[i]
                current_long_ma = self.long_ma_line.iloc[i]
                current_condition = (current_trend == -1 and current_short_ma < current_long_ma)
                
                # If background color has changed (previous day did not meet conditions, today does)
                # And if the 200MA value is above the specified threshold
                if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                    print(f"\nExit at background color change (trend change):")
                    print(f"Date: {date.strftime('%Y-%m-%d')}")
                    print(f"Price: ${price:.2f}")
                    print(f"Shares: {self.current_position}")
                    print(f"Trend: {current_trend}, Short MA: {current_short_ma:.4f}, Long MA: {current_long_ma:.4f}")
                    print(f"Long MA threshold: {self.background_exit_threshold:.2f}")
                    proceeds = self.current_position * price * (1 - self.slippage) * (1 - self.commission)
                    print(f"Proceeds (after fees and slippage): ${proceeds:.2f}")
                    
                    self._execute_exit(date, price, reason="background color change")
                    available_capital = self.current_capital
                    print(f"Available capital: ${available_capital:.2f}")
            
            # Stop loss logic
            elif self.current_position > 0 and self.entry_prices:
                # Get latest entry price
                latest_entry_price = self.entry_prices[-1]
                
                # Update highest price
                if self.highest_price is None or price > self.highest_price:
                    self.highest_price = price
                
                # Calculate stop loss price
                if self.use_trailing_stop and self.highest_price is not None:
                    # When using trailing stop
                    stop_loss_price = self.highest_price * (1 - self.trailing_stop_pct)
                else:
                    # When using regular stop loss
                    stop_loss_price = latest_entry_price * (1 - self.stop_loss_pct)
                
                # If current price falls below stop loss price
                if price <= stop_loss_price:
                    print(f"\nStop loss triggered:")
                    print(f"Date: {date.strftime('%Y-%m-%d')}")
                    print(f"Entry price: ${latest_entry_price:.2f}")
                    print(f"Current price: ${price:.2f}")
                    print(f"Stop loss price: ${stop_loss_price:.2f}")
                    if self.use_trailing_stop:
                        print(f"Highest price: ${self.highest_price:.2f}")
                        print(f"Trailing stop percentage: {self.trailing_stop_pct:.1%}")
                    print(f"Shares: {self.current_position}")
                    proceeds = self.current_position * price * (1 - self.slippage) * (1 - self.commission)
                    print(f"Proceeds (after fees and slippage): ${proceeds:.2f}")
                    
                    self._execute_exit(date, price, reason="stop loss")
                    available_capital = self.current_capital
                    print(f"Available capital: ${available_capital:.2f}")
            
            # Update equity curve
            self.equity_curve.append({
                'date': date,
                'equity': self.current_capital + 
                         (self.current_position * price)
            })
        
        print("\nTrade execution results:")
        print("-------------------")
        print(f"{self.short_ma}{self.ma_type.upper()} bottoms detected: {len(self.short_ma_bottoms)}")
        print(f"{self.long_ma}{self.ma_type.upper()} bottoms detected: {len(self.long_ma_bottoms)}")
        print(f"{self.long_ma}{self.ma_type.upper()} peaks detected: {len(self.peaks)}")
        
        # Calculate number of exits due to background color changes
        background_change_exits = 0
        for trade in self.trades:
            if trade['action'] == 'SELL':
                # Check if the previous trade was an exit due to background color change
                trade_idx = self.trades.index(trade)
                if trade_idx > 0:
                    prev_trade = self.trades[trade_idx - 1]
                    if prev_trade['action'] == 'BUY':
                        # Check if the previous trade was an exit due to background color change
                        if "background color change" in trade.get('reason', ''):
                            background_change_exits += 1
        
        print(f"Background color change exits: {background_change_exits}")
        
        # Calculate number of entries due to background color changes
        background_change_entries = 0
        for trade in self.trades:
            if trade['action'] == 'BUY':
                # Check if entry was due to background color change
                if "background color change" in trade.get('reason', ''):
                    background_change_entries += 1
        
        print(f"Background color change entries: {background_change_entries}")
    
    def _calculate_shares(self, amount, price):
        """Calculate the number of shares that can be purchased"""
        return int(amount / (price * (1 + self.slippage)))

    def _process_exit_fifo(self, exit_date, exit_price, total_shares_to_sell, total_proceeds, exit_reason):
        """Process exit using FIFO logic and record completed trades (Phase 1)"""
        remaining_shares = total_shares_to_sell

        while remaining_shares > 0 and self.open_positions:
            # Get the oldest open position (FIFO)
            entry_info = self.open_positions[0]

            # Determine how many shares to match with this entry
            shares_to_match = min(remaining_shares, entry_info['entry_shares'])

            # Calculate proportional proceeds for this portion
            proceeds_for_this_trade = (total_proceeds / total_shares_to_sell) * shares_to_match

            # Record the completed trade
            self._record_completed_trade(
                entry_info, exit_date, exit_price,
                shares_to_match, proceeds_for_this_trade, exit_reason
            )

            # Update remaining shares
            remaining_shares -= shares_to_match

            # Update or remove the open position
            if shares_to_match >= entry_info['entry_shares']:
                # Fully matched, remove this position
                self.open_positions.pop(0)
            else:
                # Partially matched, update the position
                entry_info['entry_shares'] -= shares_to_match
                entry_info['entry_cost'] = (entry_info['entry_cost'] /
                                           (entry_info['entry_shares'] + shares_to_match)) * entry_info['entry_shares']

    def _record_completed_trade(self, entry_info, exit_date, exit_price,
                                exit_shares, exit_proceeds, exit_reason):
        """Record a completed trade to trade_log (Phase 1)"""
        # Calculate holding days
        holding_days = (exit_date - entry_info['entry_date']).days

        # Calculate P&L
        entry_cost_per_share = entry_info['entry_cost'] / entry_info['entry_shares']
        entry_cost_for_sold_shares = entry_cost_per_share * exit_shares
        exit_proceeds_for_sold_shares = (exit_proceeds / exit_shares) * exit_shares if exit_shares > 0 else 0

        pnl_dollar = exit_proceeds_for_sold_shares - entry_cost_for_sold_shares
        pnl_percent = (pnl_dollar / entry_cost_for_sold_shares) * 100 if entry_cost_for_sold_shares > 0 else 0

        # Calculate cumulative P&L
        cumulative_pnl = sum(trade['pnl_dollar'] for trade in self.trade_log) + pnl_dollar

        # Create trade record
        trade_record = {
            'trade_id': self.next_trade_id,
            'entry_date': entry_info['entry_date'],
            'entry_price': entry_info['entry_price'],
            'entry_shares': exit_shares,
            'entry_cost': entry_cost_for_sold_shares,
            'entry_reason': entry_info['entry_reason'],
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_shares': exit_shares,
            'exit_proceeds': exit_proceeds_for_sold_shares,
            'exit_reason': exit_reason,
            'holding_days': holding_days,
            'pnl_dollar': pnl_dollar,
            'pnl_percent': pnl_percent,
            'cumulative_pnl': cumulative_pnl
        }

        self.trade_log.append(trade_record)
        self.next_trade_id += 1
    
    def _execute_entry(self, date, price, shares, reason=''):
        """Execute entry"""
        entry_price = price * (1 + self.slippage)
        commission = entry_price * shares * self.commission
        total_cost = entry_price * shares + commission

        self.current_position += shares  # Changed to += to support buying more
        self.current_capital -= total_cost

        # Record entry price
        self.entry_prices.append(entry_price)

        self.trades.append({
            'date': date,
            'action': 'BUY',
            'price': entry_price,
            'shares': shares,
            'commission': commission,
            'total_cost': total_cost,
            'reason': reason
        })

        # Add to open_positions for trade logging (Phase 1)
        self.open_positions.append({
            'entry_date': date,
            'entry_price': entry_price,
            'entry_shares': shares,
            'entry_cost': total_cost,
            'entry_reason': reason
        })
        
    def _execute_exit(self, date, price, reason=''):
        """Execute exit"""
        exit_price = price * (1 - self.slippage)
        commission = exit_price * self.current_position * self.commission
        total_proceeds = exit_price * self.current_position - commission

        # For partial exit, sell only half of the position
        if self.partial_exit and self.current_position > 1:
            shares_to_sell = self.current_position // 2
            self.current_position -= shares_to_sell
            commission = exit_price * shares_to_sell * self.commission
            total_proceeds = exit_price * shares_to_sell - commission
            self.current_capital += total_proceeds

            self.trades.append({
                'date': date,
                'action': 'SELL',
                'price': exit_price,
                'shares': shares_to_sell,
                'commission': commission,
                'total_proceeds': total_proceeds,
                'reason': reason + ' (partial)'
            })

            # Record completed trades using FIFO logic (Phase 1)
            self._process_exit_fifo(date, exit_price, shares_to_sell, total_proceeds, reason + ' (partial)')

            print(f"Partial exit: Sold {shares_to_sell} shares, remaining {self.current_position} shares")
        else:
            # Full exit
            self.current_capital += total_proceeds

            self.trades.append({
                'date': date,
                'action': 'SELL',
                'price': exit_price,
                'shares': self.current_position,
                'commission': commission,
                'total_proceeds': total_proceeds,
                'reason': reason
            })

            # Record completed trades using FIFO logic (Phase 1)
            self._process_exit_fifo(date, exit_price, self.current_position, total_proceeds, reason)

            self.current_position = 0
            self.entry_prices = []  # Clear entry price list
            self.stop_loss_prices = []  # Clear stop loss price list
            self.highest_price = None  # Reset highest price
    
    def calculate_performance(self):
        """Calculate performance"""
        self.equity_df = pd.DataFrame(self.equity_curve)
        self.equity_df.set_index('date', inplace=True)
        
        # Calculate final asset value (cash + position value)
        final_equity = self.equity_df['equity'].iloc[-1]
        
        # Total return (revised)
        self.total_return = (final_equity / self.initial_capital) - 1
        
        # Annual return and CAGR calculation
        days = (self.equity_df.index[-1] - self.equity_df.index[0]).days
        years = days / 365
        
        # CAGR calculation (handles negative returns)
        if self.total_return <= -1:
            self.cagr = -1
        else:
            self.cagr = np.sign(self.total_return) * (abs(1 + self.total_return) ** (1 / years) - 1)
        
        # Annual Return calculation (handles negative returns)
        if self.total_return <= -1:
            self.annual_return = -1
        else:
            self.annual_return = np.sign(self.total_return) * (abs(1 + self.total_return) ** (365 / days) - 1)
        
        # Add debug information
        if self.debug:
            print("\nReturn calculation details:")
            print(f"Initial capital: ${self.initial_capital:.2f}")
            print(f"Final equity: ${final_equity:.2f}")
            print(f"Current cash: ${self.current_capital:.2f}")
            print(f"Current position: {self.current_position} shares")
            if self.current_position > 0:
                current_price = self.price_data['adjusted_close'].iloc[-1]
                position_value = self.current_position * current_price
                print(f"Position value: ${position_value:.2f}")
        
        # Sharpe ratio
        daily_returns = self.equity_df['equity'].pct_change()
        self.sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # Maximum drawdown
        self.max_drawdown = self._calculate_max_drawdown()
        
        # Win rate
        self.win_rate = self._calculate_win_rate()
        
        # Profit-loss ratio
        self.profit_loss_ratio = self._calculate_profit_loss_ratio()
        
        # Calculate new metrics
        self.profit_factor = self._calculate_profit_factor()
        self.calmar_ratio = self._calculate_calmar_ratio()
        self.expected_value = self._calculate_expected_value()
        self.avg_pnl_per_trade = self._calculate_avg_pnl_per_trade()
        self.pareto_ratio = self._calculate_pareto_ratio()
        
        # Buy & Hold performance calculation
        initial_price = self.price_data['adjusted_close'].iloc[0]
        final_price = self.price_data['adjusted_close'].iloc[-1]
        buy_hold_shares = int(self.initial_capital / (initial_price * (1 + self.slippage)))
        buy_hold_cost = buy_hold_shares * initial_price * (1 + self.slippage) * (1 + self.commission)
        buy_hold_value = buy_hold_shares * final_price * (1 - self.slippage) * (1 - self.commission)
        buy_hold_return = (buy_hold_value / buy_hold_cost) - 1
        buy_hold_cagr = (buy_hold_value / buy_hold_cost) ** (1 / years) - 1
        
        # Buy & Hold daily returns
        buy_hold_daily_returns = self.price_data['adjusted_close'].pct_change()
        buy_hold_sharpe = np.sqrt(252) * buy_hold_daily_returns.mean() / buy_hold_daily_returns.std()
        
        # Buy & Hold maximum drawdown
        buy_hold_cummax = self.price_data['adjusted_close'].expanding().max()
        buy_hold_drawdown = self.price_data['adjusted_close'] / buy_hold_cummax - 1
        buy_hold_max_drawdown = buy_hold_drawdown.min()
        
        # Display performance metrics
        print("\nBacktest results:")
        print("-------------------")
        print("Strategy performance:")
        print(f"Total return: {self.total_return:.2%}")
        print(f"Annual return (CAGR): {self.cagr:.2%}")
        print(f"Sharpe ratio: {self.sharpe_ratio:.2f}")
        print(f"Maximum drawdown: {self.max_drawdown:.2%}")
        print(f"Win rate: {self.win_rate:.2%}")
        print(f"Profit-loss ratio: {self.profit_loss_ratio:.2f}")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print(f"Calmar Ratio: {self.calmar_ratio:.2f}")
        print(f"Expected Value: ${self.expected_value:.2f}")
        print(f"Avg. PnL per trade: ${self.avg_pnl_per_trade:.2f}")
        print(f"Pareto Ratio: {self.pareto_ratio:.2f}")
        
        print("\nBuy & Hold performance:")
        print(f"Total return: {buy_hold_return:.2%}")
        print(f"Annual return (CAGR): {buy_hold_cagr:.2%}")
        print(f"Sharpe ratio: {buy_hold_sharpe:.2f}")
        print(f"Maximum drawdown: {buy_hold_max_drawdown:.2%}")
        
        print(f"\nInvestment period: {days:.1f} days ({years:.1f} years)")
        
        # Calculate relative performance
        relative_return = self.total_return - buy_hold_return
        relative_cagr = self.cagr - buy_hold_cagr
        print("\nRelative performance (Strategy vs Buy & Hold):")
        print(f"Return difference: {relative_return:.2%}")
        print(f"CAGR difference: {relative_cagr:.2%}")
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        equity = self.equity_df['equity']
        rolling_max = equity.expanding().max()
        drawdowns = equity / rolling_max - 1
        return drawdowns.min()
    
    def _calculate_win_rate(self):
        """Calculate win rate for each individual trade"""
        if not self.trades:
            return 0
        
        # Get trade pairs
        trade_pairs = self._get_trade_pairs()
        
        # Calculate profit/loss for each trade pair
        profitable_trades = 0
        total_trades = len(trade_pairs)
        
        for pair in trade_pairs:
            profit = pair['sell_proceeds'] - pair['buy_cost']
            if profit > 0:
                profitable_trades += 1
        
        # Calculate win rate
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        if self.debug:
            print(f"\nWin rate details:")
            print(f"Total trades: {total_trades}")
            print(f"Winning trades: {profitable_trades}")
            print(f"Losing trades: {total_trades - profitable_trades}")
            
            # Display detailed trade information
            print("\nTrade details:")
            for i, pair in enumerate(trade_pairs):
                profit = pair['sell_proceeds'] - pair['buy_cost']
                profit_pct = (profit / pair['buy_cost']) * 100
                print(f"Trade {i+1}:")
                print(f"  Buy date: {pair['buy_date'].strftime('%Y-%m-%d')}")
                print(f"  Sell date: {pair['sell_date'].strftime('%Y-%m-%d')}")
                print(f"  Shares: {pair['shares']}")
                print(f"  Buy price: ${pair['buy_price']:.2f}")
                print(f"  Sell price: ${pair['sell_price']:.2f}")
                print(f"  Profit: ${profit:.2f} ({profit_pct:.2f}%)")
        
        return win_rate
    
    def _calculate_profit_loss_ratio(self):
        """Calculate profit-loss ratio"""
        if not self.trades:
            return 0
            
        # Get trade pairs
        trade_pairs = self._get_trade_pairs()
        
        # Calculate profit/loss for each trade pair
        profits = []
        losses = []
        
        for pair in trade_pairs:
            profit = pair['sell_proceeds'] - pair['buy_cost']
            if profit > 0:
                profits.append(profit)
            else:
                losses.append(abs(profit))
        
        # Calculate profit-loss ratio
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if self.debug:
            print(f"\nProfit-Loss ratio details:")
            print(f"Total trades: {len(trade_pairs)}")
            print(f"Profitable trades: {len(profits)}")
            print(f"Losing trades: {len(losses)}")
            print(f"Average profit: ${avg_profit:.2f}")
            print(f"Average loss: ${avg_loss:.2f}")
            
            # Display detailed trade information
            print("\nTrade details:")
            for i, pair in enumerate(trade_pairs):
                profit = pair['sell_proceeds'] - pair['buy_cost']
                profit_pct = (profit / pair['buy_cost']) * 100
                print(f"Trade {i+1}:")
                print(f"  Buy date: {pair['buy_date'].strftime('%Y-%m-%d')}")
                print(f"  Sell date: {pair['sell_date'].strftime('%Y-%m-%d')}")
                print(f"  Shares: {pair['shares']}")
                print(f"  Buy price: ${pair['buy_price']:.2f}")
                print(f"  Sell price: ${pair['sell_price']:.2f}")
                print(f"  Profit: ${profit:.2f} ({profit_pct:.2f}%)")
        
        return avg_profit / avg_loss if avg_loss > 0 else float('inf')
    
    def _calculate_profit_factor(self):
        """Calculate profit factor"""
        if not self.trades:
            return 0
            
        # Get trade pairs
        trade_pairs = self._get_trade_pairs()
        
        # Calculate profit/loss for each trade pair
        total_profit = 0
        total_loss = 0
        
        for pair in trade_pairs:
            profit = pair['sell_proceeds'] - pair['buy_cost']
            if profit > 0:
                total_profit += profit
            else:
                total_loss += abs(profit)
        
        # Calculate profit factor
        return total_profit / total_loss if total_loss > 0 else float('inf')
    
    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio"""
        if not self.trades:
            return 0
            
        # Calculate annual return
        days = (self.equity_df.index[-1] - self.equity_df.index[0]).days
        years = days / 365
        
        # Annual Return calculation (handles negative returns)
        if self.total_return <= -1:
            annual_return = -1
        else:
            annual_return = np.sign(self.total_return) * (abs(1 + self.total_return) ** (1 / years) - 1)
        
        # Get maximum drawdown
        max_drawdown = abs(self.max_drawdown)
        
        # Calculate Calmar ratio
        if max_drawdown == 0:
            return 0  # Return 0 if there is no drawdown
        else:
            return annual_return / max_drawdown
    
    def _calculate_expected_value(self):
        """Calculate expected value per trade"""
        if not self.trades:
            return 0
            
        # Get trade pairs
        trade_pairs = self._get_trade_pairs()
        
        # Calculate profit/loss for each trade pair
        total_profit = 0
        total_trades = len(trade_pairs)
        
        for pair in trade_pairs:
            profit = pair['sell_proceeds'] - pair['buy_cost']
            total_profit += profit
        
        # Calculate expected value
        return total_profit / total_trades if total_trades > 0 else 0
    
    def _calculate_avg_pnl_per_trade(self):
        """Calculate average PnL per trade"""
        if not self.trades:
            return 0
            
        # Get trade pairs
        trade_pairs = self._get_trade_pairs()
        
        # Calculate profit/loss for each trade pair
        total_pnl = 0
        total_trades = len(trade_pairs)
        
        for pair in trade_pairs:
            pnl = pair['sell_proceeds'] - pair['buy_cost']
            total_pnl += pnl
        
        # Calculate average PnL
        return total_pnl / total_trades if total_trades > 0 else 0
    
    def _calculate_pareto_ratio(self):
        """Calculate Pareto ratio (80/20 rule)"""
        if not self.trades:
            return 0
            
        # Get trade pairs
        trade_pairs = self._get_trade_pairs()
        
        # Calculate profit/loss for each trade pair
        trade_pnls = []
        
        for pair in trade_pairs:
            pnl = pair['sell_proceeds'] - pair['buy_cost']
            trade_pnls.append(pnl)
        
        # Sort profits/losses in descending order
        trade_pnls.sort(reverse=True)
        
        # Calculate total profit/loss
        total_pnl = sum(trade_pnls)
        
        if total_pnl <= 0:
            return 0
        
        # Calculate total profit/loss of top 20% trades
        top_20_percent_count = max(1, int(len(trade_pnls) * 0.2))
        top_20_percent_pnl = sum(trade_pnls[:top_20_percent_count])
        
        # Calculate Pareto ratio
        return top_20_percent_pnl / total_pnl
    
    def _get_trade_pairs(self):
        """Helper method to get trade pairs"""
        trade_pairs = []
        current_buy_trades = []
        
        # Copy trade history to operate on (don't modify original data)
        trades_copy = []
        for trade in self.trades:
            trade_copy = trade.copy()
            # Copy number of shares
            if 'shares' in trade_copy:
                trade_copy['shares'] = trade_copy['shares']
            trades_copy.append(trade_copy)
        
        # Debug information
        if self.debug:
            print("\nTrade pairs calculation:")
            print(f"Total trades: {len(trades_copy)}")
            print(f"Buy trades: {sum(1 for t in trades_copy if t['action'] == 'BUY')}")
            print(f"Sell trades: {sum(1 for t in trades_copy if t['action'] == 'SELL')}")
        
        for trade in trades_copy:
            if trade['action'] == 'BUY':
                if trade['shares'] > 0:
                    current_buy_trades.append(trade)
                    if self.debug:
                        print(f"Added buy trade: {trade['date'].strftime('%Y-%m-%d')}, Shares: {trade['shares']}")
            elif trade['action'] == 'SELL':
                remaining_shares = trade['shares']
                
                if self.debug:
                    print(f"Processing sell trade: {trade['date'].strftime('%Y-%m-%d')}, Shares: {remaining_shares}")
                    print(f"Current buy trades: {len(current_buy_trades)}")
                
                while remaining_shares > 0 and current_buy_trades:
                    buy_trade = current_buy_trades[0]
                    if buy_trade['shares'] <= 0:
                        current_buy_trades.pop(0)
                        if self.debug:
                            print(f"Removed empty buy trade")
                        continue
                        
                    matched_shares = min(remaining_shares, buy_trade['shares'])
                    
                    trade_pairs.append({
                        'buy_date': buy_trade['date'],
                        'sell_date': trade['date'],
                        'shares': matched_shares,
                        'buy_price': buy_trade['price'],
                        'sell_price': trade['price'],
                        'buy_cost': buy_trade['total_cost'] * (matched_shares / buy_trade['shares']),
                        'sell_proceeds': trade['total_proceeds'] * (matched_shares / trade['shares'])
                    })
                    
                    if self.debug:
                        print(f"Created trade pair: Buy: {buy_trade['date'].strftime('%Y-%m-%d')}, "
                              f"Sell: {trade['date'].strftime('%Y-%m-%d')}, Shares: {matched_shares}")
                    
                    remaining_shares -= matched_shares
                    buy_trade['shares'] -= matched_shares
                    
                    if buy_trade['shares'] == 0:
                        current_buy_trades.pop(0)
                        if self.debug:
                            print(f"Removed fully matched buy trade")
        
        if self.debug:
            print(f"Total trade pairs created: {len(trade_pairs)}")
        
        return trade_pairs
    
    def visualize_results(self, show_plot=True):
        """Visualize results"""
        _setup_matplotlib_backend()
        
        # Create subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))
        
        # Price chart and trade points
        ax1.plot(self.price_data.index, self.price_data['adjusted_close'], label=f'{self.symbol} Price')
        
        # Display trade points
        for trade in self.trades:
            if trade['action'] == 'BUY':
                ax1.scatter(trade['date'], trade['price'], 
                          color='green', marker='^', s=100, label='Buy')
            elif trade['action'] == 'SELL':
                # Check if this was a stop loss by checking the previous entry price
                trade_idx = self.trades.index(trade)
                if trade_idx > 0:
                    prev_trade = self.trades[trade_idx - 1]
                    if prev_trade['action'] == 'BUY':
                        entry_price = prev_trade['price']
                        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                        
                        # Determine if this was a stop loss
                        if trade['price'] <= stop_loss_price:
                            # Display stop loss trades with special markers
                            if self.use_trailing_stop:
                                # Display in blue for trailing stop
                                ax1.scatter(trade['date'], trade['price'], 
                                          color='blue', marker='x', s=150, label='Trailing Stop')
                            else:
                                # Display in purple for regular stop loss
                                ax1.scatter(trade['date'], trade['price'], 
                                          color='purple', marker='x', s=150, label='Stop Loss')
                        else:
                            # Regular sell
                            ax1.scatter(trade['date'], trade['price'], 
                                      color='red', marker='v', s=100, label='Sell')
        
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(0.02, 0.5))
        
        ax1.set_title(f'{self.symbol} Price Chart with Trade Points')
        
        # Breadth Index and moving averages
        ax2.plot(self.breadth_index.index, self.breadth_index, label='Breadth Index')
        ax2.plot(self.short_ma_line.index, self.short_ma_line, 
                label=f'{self.short_ma}{self.ma_type.upper()}')
        ax2.plot(self.long_ma_line.index, self.long_ma_line, 
                label=f'{self.long_ma}{self.ma_type.upper()}')
        
        # Set background color (based on trend)
        for i in range(len(self.long_ma_trend) - 1):
            if (self.long_ma_trend.iloc[i] == -1 and 
                self.short_ma_line.iloc[i] < self.long_ma_line.iloc[i]):
                ax2.axvspan(
                    self.long_ma_line.index[i],
                    self.long_ma_line.index[i + 1],
                    color=(1.0, 0.9, 0.96),
                    alpha=0.3
                )
        
        # Detect and display background color change points
        background_changes = []
        white_to_pink_changes = []  # White to pink change (exit)
        pink_to_white_changes = []  # Pink to white change (entry)
        
        # Only detect background color changes if use_background_color_signals is enabled
        if self.use_background_color_signals:
            for i in range(1, len(self.long_ma_trend)):
                prev_trend = self.long_ma_trend.iloc[i-1]
                prev_short_ma = self.short_ma_line.iloc[i-1]
                prev_long_ma = self.long_ma_line.iloc[i-1]
                
                # Today's data
                current_trend = self.long_ma_trend.iloc[i]
                current_short_ma = self.short_ma_line.iloc[i]
                current_long_ma = self.long_ma_line.iloc[i]
                
                # White to pink change (exit)
                prev_condition = not (prev_trend == -1 and prev_short_ma < prev_long_ma)
                current_condition = (current_trend == -1 and current_short_ma < current_long_ma)
                
                if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                    white_to_pink_changes.append(self.long_ma_line.index[i])
                
                # Pink to white change (entry)
                prev_condition = (prev_trend == -1 and prev_short_ma < prev_long_ma)
                current_condition = not (current_trend == -1 and current_short_ma < current_long_ma)
                
                if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                    pink_to_white_changes.append(self.long_ma_line.index[i])
        
        # Display white to pink change points (exit)
        if white_to_pink_changes and self.use_background_color_signals:
            ax2.scatter(white_to_pink_changes, 
                       self.long_ma_line[white_to_pink_changes],
                       color='orange', marker='x', s=150,
                       label=f'White to Pink (Exit, MA≥{self.background_exit_threshold:.2f})')
        
        # Display pink to white change points (entry)
        if pink_to_white_changes and self.use_background_color_signals:
            ax2.scatter(pink_to_white_changes, 
                       self.long_ma_line[pink_to_white_changes],
                       color='green', marker='^', s=150,
                       label=f'Pink to White (Entry, MA≥{self.background_exit_threshold:.2f})')
        
        # Display signal points
        ax2.scatter(self.short_ma_bottoms, 
                   self.short_ma_line[self.short_ma_bottoms],
                   color='green', marker='^', s=100,
                   label=f'{self.short_ma}{self.ma_type.upper()} Bottom')
        ax2.scatter(self.long_ma_bottoms, 
                   self.long_ma_line[self.long_ma_bottoms],
                   color='blue', marker='^', s=100,
                   label=f'{self.long_ma}{self.ma_type.upper()} Bottom')
        ax2.scatter(self.peaks, 
                   self.long_ma_line[self.peaks],
                   color='red', marker='v', s=100,
                   label=f'{self.long_ma}{self.ma_type.upper()} Peak')
        
        ax2.set_title('Breadth Index and Moving Averages')
        ax2.legend(loc='center left', bbox_to_anchor=(0.02, 0.5))
        
        # Equity curve comparison
        initial_price = self.price_data['adjusted_close'].iloc[0]
        buy_hold_shares = int(self.initial_capital / (initial_price * (1 + self.slippage)))
        buy_hold_equity = self.price_data['adjusted_close'] * buy_hold_shares
        
        ax3.plot(self.equity_df.index, self.equity_df['equity'], label='Strategy')
        ax3.plot(buy_hold_equity.index, buy_hold_equity, label='Buy & Hold')
        ax3.set_title('Equity Curve Comparison')
        ax3.legend(loc='center left', bbox_to_anchor=(0.02, 0.5))
        
        # Drawdown chart
        equity = self.equity_df['equity']
        rolling_max = equity.expanding().max()
        drawdown = equity / rolling_max - 1
        
        # Buy & Hold's drawdown calculation
        buy_hold_rolling_max = buy_hold_equity.expanding().max()
        buy_hold_drawdown = buy_hold_equity / buy_hold_rolling_max - 1
        
        # Plot both drawdowns
        ax4.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Strategy')
        ax4.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax4.plot(buy_hold_drawdown.index, buy_hold_drawdown, color='blue', linewidth=1, label='Buy & Hold')
        ax4.fill_between(buy_hold_drawdown.index, buy_hold_drawdown, 0, color='blue', alpha=0.3)
        
        ax4.set_title('Drawdown Comparison')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True)
        ax4.legend(loc='center left', bbox_to_anchor=(0.02, 0.5))
        
        # Add horizontal line at -10% for reference
        ax4.axhline(y=-0.1, color='darkred', linestyle='--', alpha=0.7)
        ax4.text(drawdown.index[-1], -0.1, ' -10%', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'reports/backtest_results_{self.symbol}.png')
        if show_plot:
            plt.show()  # Display chart

    def save_trade_log(self, filename=None):
        """Save trade log to CSV file (Phase 1)"""
        if not self.trade_log:
            print("No trades to save.")
            return None

        # Generate default filename if not provided
        if filename is None:
            filename = f'reports/trade_log_{self.symbol}_{self.start_date}_{self.end_date}.csv'

        # Convert trade_log to DataFrame
        trade_df = pd.DataFrame(self.trade_log)

        # Format datetime columns
        trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date']).dt.strftime('%Y-%m-%d')
        trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date']).dt.strftime('%Y-%m-%d')

        # Save to CSV
        trade_df.to_csv(filename, index=False)
        print(f"\nTrade log saved to: {filename}")

        return filename

def main():
    parser = argparse.ArgumentParser(description='Backtest using Market Breadth indicator')
    parser.add_argument('--start_date', type=str,
                      help='Backtest start date (YYYY-MM-DD format). If not specified, 10 years before end date')
    parser.add_argument('--end_date', type=str,
                      help='Backtest end date (YYYY-MM-DD format). If not specified, current date')
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
    parser.add_argument('--stop_loss_pct', type=float, default=0.1,
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
    parser.add_argument('--no_show_plot', action='store_true',
                      help='Do not show plot after saving (default: show plot)')
    
    args = parser.parse_args()
    
    # Set default values if dates are not specified
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Using current date as end date: {args.end_date}")
    
    if args.start_date is None:
        start_date = pd.to_datetime(args.end_date) - pd.DateOffset(years=10)
        args.start_date = start_date.strftime('%Y-%m-%d')
        print(f"Using date 10 years before end date as start date: {args.start_date}")
    
    print(f"args.start_date: {args.start_date}")
    
    backtest = Backtest(
        start_date=args.start_date,
        end_date=args.end_date,
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
        no_show_plot=args.no_show_plot
    )
    
    backtest.run()

    # Save trade log if trades were executed
    if backtest.trade_log:
        backtest.save_trade_log()

if __name__ == '__main__':
    main() 