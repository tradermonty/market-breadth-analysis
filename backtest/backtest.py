import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# 必要なディレクトリの作成
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
        self.symbol = symbol  # シンボルを指定可能に変更
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
        self.stop_loss_pct = stop_loss_pct  # 損切りパーセンテージ
        self.disable_short_ma_entry = disable_short_ma_entry  # short_maによるエントリーを無効にするオプション
        self.use_trailing_stop = use_trailing_stop  # Trailing stopを使用するかどうか
        self.trailing_stop_pct = trailing_stop_pct  # Trailing stopのパーセンテージ
        self.background_exit_threshold = background_exit_threshold  # 背景色変化時のイグジットしきい値
        self.use_background_color_signals = use_background_color_signals  # 背景色の変化によるシグナルを使用するかどうか
        self.partial_exit = partial_exit  # イグジット時に保有資産の半分だけ売却するかどうか
        self.no_show_plot = no_show_plot  # プロットを表示しないかどうか
        
        # Variables to store backtest results
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_capital = initial_capital
        self.current_position = 0
        self.entry_prices = []  # エントリー価格を記録するリスト
        self.stop_loss_prices = []  # 損切り価格を記録するリスト
        self.highest_price = None  # ポジション保有中の最高価格
        
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
        
        # 移動平均線の種類に基づいて計算
        if self.ma_type == 'ema':
            # 指数移動平均（EMA）
            self.short_ma_line = self.breadth_index.ewm(span=self.short_ma, adjust=False).mean()
            self.long_ma_line = self.breadth_index.ewm(span=self.long_ma, adjust=False).mean()
        else:
            # 単純移動平均（SMA）
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
        
        # Initialize lists to store signals
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
        tickers = get_sp500_tickers_from_wikipedia()
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
        
        # シグナル検出用の変数を初期化
        self.short_ma_bottoms = []
        self.long_ma_bottoms = []
        self.peaks = []
        
        # 検出済みのシグナルを記録する変数
        detected_short_ma_bottoms = set()
        detected_long_ma_bottoms = set()
        detected_peaks = set()
        
        # 取引実行
        for i, date in enumerate(self.price_data.index):
            price = self.price_data.loc[date, 'adjusted_close']
            
            # 現在の日付までのデータを使用してシグナルを検出
            current_data = self.price_data.iloc[:i+1]
            current_breadth_index = self.breadth_index.iloc[:i+1]
            current_short_ma_line = self.short_ma_line.iloc[:i+1]
            current_long_ma_line = self.long_ma_line.iloc[:i+1]
            
            # データ期間の開始日と終了日を取得
            data_start_date = current_short_ma_line.index[0].strftime('%Y-%m-%d')
            data_end_date = current_short_ma_line.index[-1].strftime('%Y-%m-%d')
            
            # 20MAのボトム検出（disable_short_ma_entryがFalseの場合のみ）
            if not self.disable_short_ma_entry and len(current_short_ma_line) > self.short_ma:
                below_threshold_short = current_short_ma_line[current_short_ma_line < self.threshold]
                if not below_threshold_short.empty:
                    # 元のインデックスを保持
                    original_indices = np.where(current_short_ma_line < self.threshold)[0]
                    bottoms_short, _ = find_peaks(
                        -below_threshold_short.values,
                        prominence=0.02
                    )
                    
                    # find_peaksで検出された底から、Market Breadthの実測値が条件を満たすものだけを選択
                    for bottom_idx in bottoms_short:
                        # 元のデータでのインデックス位置を取得
                        original_idx = original_indices[bottom_idx]
                        bottom_date = current_short_ma_line.index[original_idx]
                        
                        # 過去20日間のMarket Breadthの最小値を計算
                        if original_idx >= 20:  # 過去20日分のデータがある場合のみチェック
                            past_20days_min = current_breadth_index.iloc[original_idx-20:original_idx+1].min()
                            if past_20days_min <= 0.3:  # 実測値の条件をチェック
                                # まだ検出されていないボトムの場合のみ追加
                                if bottom_date not in detected_short_ma_bottoms:
                                    detected_short_ma_bottoms.add(bottom_date)
                                    # シグナル発生日としてData periodの終わりの日付を使用
                                    signal_date = pd.to_datetime(data_end_date)
                                    self.short_ma_bottoms.append(signal_date)
                                    print(f"New {self.short_ma}{self.ma_type.upper()} bottom detected at: {bottom_date.strftime('%Y-%m-%d')}")
                                    print(f"  Data period: {data_start_date} to {data_end_date}")
                                    print(f"  Signal date: {signal_date.strftime('%Y-%m-%d')}")
            
            # 200MAのボトム検出
            if len(current_long_ma_line) > self.long_ma:
                bottoms_long, _ = find_peaks(
                    -current_long_ma_line.values,
                    prominence=0.015
                )
                for bottom_idx in bottoms_long:
                    bottom_date = current_long_ma_line.index[bottom_idx]
                    
                    # 元のデータでのインデックス位置を取得
                    original_idx = bottom_idx
                    
                    # 過去20日間のMarket Breadthの最小値を計算
                    if original_idx >= 20:  # 過去20日分のデータがある場合のみチェック
                        past_20days_min = current_breadth_index.iloc[original_idx-20:original_idx+1].min()
                        if past_20days_min <= 0.5:  # 実測値の条件をチェック
                            if bottom_date not in detected_long_ma_bottoms:
                                detected_long_ma_bottoms.add(bottom_date)
                                # シグナル発生日としてData periodの終わりの日付を使用
                                signal_date = pd.to_datetime(data_end_date)
                                self.long_ma_bottoms.append(signal_date)
                                print(f"New {self.long_ma}{self.ma_type.upper()} bottom detected at: {bottom_date.strftime('%Y-%m-%d')}")
                                print(f"  Data period: {data_start_date} to {data_end_date}")
                                print(f"  Signal date: {signal_date.strftime('%Y-%m-%d')}")
            
            # 200MAのピーク検出
            if len(current_long_ma_line) > self.long_ma:
                peaks, _ = find_peaks(
                    current_long_ma_line.values,
                    prominence=0.015
                )
                for peak_idx in peaks:
                    peak_date = current_long_ma_line.index[peak_idx]
                    
                    # 200MAの値が0.6以上であることを確認
                    if current_long_ma_line.iloc[peak_idx] >= 0.5:
                        if peak_date not in detected_peaks:
                            detected_peaks.add(peak_date)
                            # シグナル発生日としてData periodの終わりの日付を使用
                            signal_date = pd.to_datetime(data_end_date)
                            self.peaks.append(signal_date)
                            print(f"New {self.long_ma}{self.ma_type.upper()} peak detected at: {peak_date.strftime('%Y-%m-%d')}")
                            print(f"  Data period: {data_start_date} to {data_end_date}")
                            print(f"  Signal date: {signal_date.strftime('%Y-%m-%d')}")
                            print(f"  {self.long_ma}{self.ma_type.upper()} value: {current_long_ma_line.iloc[peak_idx]:.4f}")
            
            # 20MAのボトムでのエントリー（disable_short_ma_entryがFalseの場合のみ）
            if not self.disable_short_ma_entry and date in self.short_ma_bottoms:
                if available_capital > 0:  # 資金がある場合はポジションを増やす
                    entry_amount = available_capital / 2
                    shares = self._calculate_shares(entry_amount, price)
                    if shares > 0:  # 購入可能な株式がある場合のみエントリー
                        self._execute_entry(date, price, shares, reason="short_ma_bottom")
                        available_capital -= entry_amount
                        # 最高価格を初期化
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
                
            # 200MAのボトムでのエントリー
            elif date in self.long_ma_bottoms:
                if available_capital > 0:  # 資金がある場合はポジションを増やす
                    # 200MAの場合は残りの利用可能資金をすべて使用
                    shares = self._calculate_shares(available_capital, price)
                    if shares > 0:
                        self._execute_entry(date, price, shares, reason="long_ma_bottom")
                        # 最高価格を初期化
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
            
            # 背景が白に変わったときのエントリー（新しい条件）
            elif self.use_background_color_signals and self.current_position == 0 and i > 0:
                # 前日のデータと比較して背景色が変わったかどうかを確認
                prev_trend = self.long_ma_trend.iloc[i-1]
                prev_short_ma = self.short_ma_line.iloc[i-1]
                prev_long_ma = self.long_ma_line.iloc[i-1]
                
                # 前日は背景がピンクだった（条件を満たしていた）
                prev_condition = (prev_trend == -1 and prev_short_ma < prev_long_ma)
                
                # 今日は背景が白になった（条件を満たしていない）
                current_trend = self.long_ma_trend.iloc[i]
                current_short_ma = self.short_ma_line.iloc[i]
                current_long_ma = self.long_ma_line.iloc[i]
                current_condition = not (current_trend == -1 and current_short_ma < current_long_ma)
                
                # 背景色がピンクから白に変わった場合（前日は条件を満たしていて、今日は満たしていない）
                # かつ、200MAの値が指定されたしきい値以上である場合
                if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                    if available_capital > 0:  # 資金がある場合はポジションを増やす
                        # 残りの利用可能資金をすべて使用
                        shares = self._calculate_shares(available_capital, price)
                        if shares > 0:
                            self._execute_entry(date, price, shares, reason="background_color_change")
                            # 最高価格を初期化
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
            
            # ピークでのエグジット
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
            
            # 背景がピンクに変わる瞬間でのエグジット（新しい条件）
            elif self.use_background_color_signals and self.current_position > 0 and i > 0:
                # 前日のデータと比較して背景色が変わったかどうかを確認
                prev_trend = self.long_ma_trend.iloc[i-1]
                prev_short_ma = self.short_ma_line.iloc[i-1]
                prev_long_ma = self.long_ma_line.iloc[i-1]
                
                # 前日は背景がピンクでなかった（または条件を満たしていなかった）
                prev_condition = not (prev_trend == -1 and prev_short_ma < prev_long_ma)
                
                # 今日は背景がピンクになった（条件を満たした）
                current_trend = self.long_ma_trend.iloc[i]
                current_short_ma = self.short_ma_line.iloc[i]
                current_long_ma = self.long_ma_line.iloc[i]
                current_condition = (current_trend == -1 and current_short_ma < current_long_ma)
                
                # 背景色が変わった場合（前日は条件を満たしていなくて、今日は満たした）
                # かつ、200MAの値が指定されたしきい値以上である場合
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
            
            # 損切りロジック
            elif self.current_position > 0 and self.entry_prices:
                # 最新のエントリー価格を取得
                latest_entry_price = self.entry_prices[-1]
                
                # 最高価格を更新
                if self.highest_price is None or price > self.highest_price:
                    self.highest_price = price
                
                # 損切り価格を計算
                if self.use_trailing_stop and self.highest_price is not None:
                    # トレーリングストップを使用する場合
                    stop_loss_price = self.highest_price * (1 - self.trailing_stop_pct)
                else:
                    # 通常の損切りを使用する場合
                    stop_loss_price = latest_entry_price * (1 - self.stop_loss_pct)
                
                # 現在の価格が損切り価格を下回った場合
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
            
            # エクイティカーブの更新
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
        
        # 背景色の変化によるイグジットの回数を計算
        background_change_exits = 0
        for trade in self.trades:
            if trade['action'] == 'SELL':
                # 直前の取引が背景色の変化によるイグジットかどうかを確認
                trade_idx = self.trades.index(trade)
                if trade_idx > 0:
                    prev_trade = self.trades[trade_idx - 1]
                    if prev_trade['action'] == 'BUY':
                        # 直前の取引が背景色の変化によるイグジットかどうかを確認
                        if "background color change" in trade.get('reason', ''):
                            background_change_exits += 1
        
        print(f"Background color change exits: {background_change_exits}")
        
        # 背景色の変化によるエントリーの回数を計算
        background_change_entries = 0
        for trade in self.trades:
            if trade['action'] == 'BUY':
                # 背景色の変化によるエントリーかどうかを確認
                if "background color change" in trade.get('reason', ''):
                    background_change_entries += 1
        
        print(f"Background color change entries: {background_change_entries}")
    
    def _calculate_shares(self, amount, price):
        """Calculate the number of shares that can be purchased"""
        return int(amount / (price * (1 + self.slippage)))
    
    def _execute_entry(self, date, price, shares, reason=''):
        """Execute entry"""
        entry_price = price * (1 + self.slippage)
        commission = entry_price * shares * self.commission
        total_cost = entry_price * shares + commission
        
        self.current_position += shares  # Changed to += to support buying more
        self.current_capital -= total_cost
        
        # エントリー価格を記録
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
        
    def _execute_exit(self, date, price, reason=''):
        """Execute exit"""
        exit_price = price * (1 - self.slippage)
        commission = exit_price * self.current_position * self.commission
        total_proceeds = exit_price * self.current_position - commission
        
        # 部分イグジットの場合、保有資産の半分だけ売却
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
            
            print(f"Partial exit: Sold {shares_to_sell} shares, remaining {self.current_position} shares")
        else:
            # 全額イグジット
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
            
            self.current_position = 0
            self.entry_prices = []  # エントリー価格リストをクリア
            self.stop_loss_prices = []  # 損切り価格リストをクリア
            self.highest_price = None  # 最高価格をリセット
    
    def calculate_performance(self):
        """Calculate performance"""
        self.equity_df = pd.DataFrame(self.equity_curve)
        self.equity_df.set_index('date', inplace=True)
        
        # 最終的な資産価値を計算（現金 + ポジション価値）
        final_equity = self.equity_df['equity'].iloc[-1]
        
        # Total return（修正版）
        self.total_return = (final_equity / self.initial_capital) - 1
        
        # Annual return and CAGR calculation
        days = (self.equity_df.index[-1] - self.equity_df.index[0]).days
        years = days / 365
        
        # CAGRの計算（負のリターンに対応）
        if self.total_return <= -1:
            self.cagr = -1
        else:
            self.cagr = np.sign(self.total_return) * (abs(1 + self.total_return) ** (1 / years) - 1)
        
        # Annual Returnの計算（負のリターンに対応）
        if self.total_return <= -1:
            self.annual_return = -1
        else:
            self.annual_return = np.sign(self.total_return) * (abs(1 + self.total_return) ** (365 / days) - 1)
        
        # デバッグ情報を追加
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
        
        # 新しい指標を計算
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
        
        # 取引ペアを取得
        trade_pairs = self._get_trade_pairs()
        
        # 各取引ペアの損益を計算
        profitable_trades = 0
        total_trades = len(trade_pairs)
        
        for pair in trade_pairs:
            profit = pair['sell_proceeds'] - pair['buy_cost']
            if profit > 0:
                profitable_trades += 1
        
        # 勝率を計算
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        if self.debug:
            print(f"\nWin rate details:")
            print(f"Total trades: {total_trades}")
            print(f"Winning trades: {profitable_trades}")
            print(f"Losing trades: {total_trades - profitable_trades}")
            
            # 詳細な取引情報を表示
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
            
        # 取引ペアを取得
        trade_pairs = self._get_trade_pairs()
        
        # 各取引ペアの損益を計算
        profits = []
        losses = []
        
        for pair in trade_pairs:
            profit = pair['sell_proceeds'] - pair['buy_cost']
            if profit > 0:
                profits.append(profit)
            else:
                losses.append(abs(profit))
        
        # プロフィット・ロス比を計算
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if self.debug:
            print(f"\nProfit-Loss ratio details:")
            print(f"Total trades: {len(trade_pairs)}")
            print(f"Profitable trades: {len(profits)}")
            print(f"Losing trades: {len(losses)}")
            print(f"Average profit: ${avg_profit:.2f}")
            print(f"Average loss: ${avg_loss:.2f}")
            
            # 詳細な取引情報を表示
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
            
        # 取引ペアを取得
        trade_pairs = self._get_trade_pairs()
        
        # 各取引ペアの損益を計算
        total_profit = 0
        total_loss = 0
        
        for pair in trade_pairs:
            profit = pair['sell_proceeds'] - pair['buy_cost']
            if profit > 0:
                total_profit += profit
            else:
                total_loss += abs(profit)
        
        # プロフィットファクターを計算
        return total_profit / total_loss if total_loss > 0 else float('inf')
    
    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio"""
        if not self.trades:
            return 0
            
        # 年率リターンを計算
        days = (self.equity_df.index[-1] - self.equity_df.index[0]).days
        years = days / 365
        
        # Annual Returnの計算（負のリターンに対応）
        if self.total_return <= -1:
            annual_return = -1
        else:
            annual_return = np.sign(self.total_return) * (abs(1 + self.total_return) ** (1 / years) - 1)
        
        # 最大ドローダウンを取得
        max_drawdown = abs(self.max_drawdown)
        
        # Calmar比率を計算
        if max_drawdown == 0:
            return 0  # ドローダウンがない場合は0を返す
        else:
            return annual_return / max_drawdown
    
    def _calculate_expected_value(self):
        """Calculate expected value per trade"""
        if not self.trades:
            return 0
            
        # 取引ペアを取得
        trade_pairs = self._get_trade_pairs()
        
        # 各取引ペアの損益を計算
        total_profit = 0
        total_trades = len(trade_pairs)
        
        for pair in trade_pairs:
            profit = pair['sell_proceeds'] - pair['buy_cost']
            total_profit += profit
        
        # 期待値を計算
        return total_profit / total_trades if total_trades > 0 else 0
    
    def _calculate_avg_pnl_per_trade(self):
        """Calculate average PnL per trade"""
        if not self.trades:
            return 0
            
        # 取引ペアを取得
        trade_pairs = self._get_trade_pairs()
        
        # 各取引ペアの損益を計算
        total_pnl = 0
        total_trades = len(trade_pairs)
        
        for pair in trade_pairs:
            pnl = pair['sell_proceeds'] - pair['buy_cost']
            total_pnl += pnl
        
        # 平均PnLを計算
        return total_pnl / total_trades if total_trades > 0 else 0
    
    def _calculate_pareto_ratio(self):
        """Calculate Pareto ratio (80/20 rule)"""
        if not self.trades:
            return 0
            
        # 取引ペアを取得
        trade_pairs = self._get_trade_pairs()
        
        # 各取引ペアの損益を計算
        trade_pnls = []
        
        for pair in trade_pairs:
            pnl = pair['sell_proceeds'] - pair['buy_cost']
            trade_pnls.append(pnl)
        
        # 損益を降順にソート
        trade_pnls.sort(reverse=True)
        
        # 合計損益を計算
        total_pnl = sum(trade_pnls)
        
        if total_pnl <= 0:
            return 0
        
        # 上位20%の取引の損益合計を計算
        top_20_percent_count = max(1, int(len(trade_pnls) * 0.2))
        top_20_percent_pnl = sum(trade_pnls[:top_20_percent_count])
        
        # Pareto比率を計算
        return top_20_percent_pnl / total_pnl
    
    def _get_trade_pairs(self):
        """Helper method to get trade pairs"""
        trade_pairs = []
        current_buy_trades = []
        
        # 取引履歴をコピーして操作（元のデータを変更しない）
        trades_copy = []
        for trade in self.trades:
            trade_copy = trade.copy()
            # 株式数をコピー
            if 'shares' in trade_copy:
                trade_copy['shares'] = trade_copy['shares']
            trades_copy.append(trade_copy)
        
        # デバッグ情報
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
        setup_matplotlib_backend()
        
        # Create subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))
        
        # Price chart and trade points
        ax1.plot(self.price_data.index, self.price_data['adjusted_close'], label=f'{self.symbol} Price')
        
        # 取引ポイントを表示
        for trade in self.trades:
            if trade['action'] == 'BUY':
                ax1.scatter(trade['date'], trade['price'], 
                          color='green', marker='^', s=100, label='Buy')
            elif trade['action'] == 'SELL':
                # 損切りかどうかを判断するために、直前のエントリー価格を確認
                trade_idx = self.trades.index(trade)
                if trade_idx > 0:
                    prev_trade = self.trades[trade_idx - 1]
                    if prev_trade['action'] == 'BUY':
                        entry_price = prev_trade['price']
                        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                        
                        # 損切りかどうかを判断
                        if trade['price'] <= stop_loss_price:
                            # 損切りされた取引は特別なマーカーで表示
                            if self.use_trailing_stop:
                                # トレーリングストップの場合は青色で表示
                                ax1.scatter(trade['date'], trade['price'], 
                                          color='blue', marker='x', s=150, label='Trailing Stop')
                            else:
                                # 通常の損切りは紫色で表示
                                ax1.scatter(trade['date'], trade['price'], 
                                          color='purple', marker='x', s=150, label='Stop Loss')
                        else:
                            # 通常の売却
                            ax1.scatter(trade['date'], trade['price'], 
                                      color='red', marker='v', s=100, label='Sell')
        
        # 重複するラベルを削除
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        
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
        
        # 背景色の変化ポイントを検出して表示
        background_changes = []
        white_to_pink_changes = []  # 白からピンクへの変化（エグジット）
        pink_to_white_changes = []  # ピンクから白への変化（エントリー）
        
        # use_background_color_signalsが有効な場合のみ背景色の変化を検出
        if self.use_background_color_signals:
            for i in range(1, len(self.long_ma_trend)):
                prev_trend = self.long_ma_trend.iloc[i-1]
                prev_short_ma = self.short_ma_line.iloc[i-1]
                prev_long_ma = self.long_ma_line.iloc[i-1]
                
                # 今日のデータ
                current_trend = self.long_ma_trend.iloc[i]
                current_short_ma = self.short_ma_line.iloc[i]
                current_long_ma = self.long_ma_line.iloc[i]
                
                # 白からピンクへの変化（エグジット）
                prev_condition = not (prev_trend == -1 and prev_short_ma < prev_long_ma)
                current_condition = (current_trend == -1 and current_short_ma < current_long_ma)
                
                if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                    white_to_pink_changes.append(self.long_ma_line.index[i])
                
                # ピンクから白への変化（エントリー）
                prev_condition = (prev_trend == -1 and prev_short_ma < prev_long_ma)
                current_condition = not (current_trend == -1 and current_short_ma < current_long_ma)
                
                if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                    pink_to_white_changes.append(self.long_ma_line.index[i])
        
        # 白からピンクへの変化ポイントを表示（エグジット）
        if white_to_pink_changes and self.use_background_color_signals:
            ax2.scatter(white_to_pink_changes, 
                       self.long_ma_line[white_to_pink_changes],
                       color='orange', marker='x', s=150,
                       label=f'White to Pink (Exit, MA≥{self.background_exit_threshold:.2f})')
        
        # ピンクから白への変化ポイントを表示（エントリー）
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
        ax2.legend()
        
        # Equity curve comparison
        initial_price = self.price_data['adjusted_close'].iloc[0]
        buy_hold_shares = int(self.initial_capital / (initial_price * (1 + self.slippage)))
        buy_hold_equity = self.price_data['adjusted_close'] * buy_hold_shares
        
        ax3.plot(self.equity_df.index, self.equity_df['equity'], label='Strategy')
        ax3.plot(buy_hold_equity.index, buy_hold_equity, label='Buy & Hold')
        ax3.set_title('Equity Curve Comparison')
        ax3.legend()
        
        # Drawdown chart
        equity = self.equity_df['equity']
        rolling_max = equity.expanding().max()
        drawdown = equity / rolling_max - 1
        
        # Buy & Holdのdrawdownを計算
        buy_hold_rolling_max = buy_hold_equity.expanding().max()
        buy_hold_drawdown = buy_hold_equity / buy_hold_rolling_max - 1
        
        # 両方のdrawdownをプロット
        ax4.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Strategy')
        ax4.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax4.plot(buy_hold_drawdown.index, buy_hold_drawdown, color='blue', linewidth=1, label='Buy & Hold')
        ax4.fill_between(buy_hold_drawdown.index, buy_hold_drawdown, 0, color='blue', alpha=0.3)
        
        ax4.set_title('Drawdown Comparison')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True)
        ax4.legend()
        
        # Add horizontal line at -10% for reference
        ax4.axhline(y=-0.1, color='darkred', linestyle='--', alpha=0.7)
        ax4.text(drawdown.index[-1], -0.1, ' -10%', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'reports/backtest_results_{self.symbol}.png')
        if show_plot:
            plt.show()  # チャートを表示

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

if __name__ == '__main__':
    main() 