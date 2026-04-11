import os
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Add parent directory to path to import market_breadth
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_breadth import (
    calculate_above_ma,
    calculate_trend_with_hysteresis,
    get_multiple_stock_data,
    get_sp500_tickers_from_fmp,
    get_stock_price_data,
    get_stock_price_ohlc,
    load_breadth_series_from_csv,
    load_stock_data,
    save_stock_data,
)

# Create necessary directories
reports_dir = pathlib.Path('reports')
reports_dir.mkdir(exist_ok=True)


# Module-level functions for pivot detection (shared with trade/)
try:
    from backtest.pivot_detection import detect_pivot_high, detect_pivot_low
except ModuleNotFoundError:
    from pivot_detection import detect_pivot_high, detect_pivot_low

# Performance metric functions
try:
    from backtest.performance_metrics import (
        calculate_avg_pnl_per_trade,
        calculate_calmar_ratio,
        calculate_expected_value,
        calculate_max_drawdown,
        calculate_pareto_ratio,
        calculate_profit_factor,
        calculate_profit_loss_ratio,
        calculate_win_rate,
        get_trade_pairs,
    )
except ModuleNotFoundError:
    from performance_metrics import (
        calculate_avg_pnl_per_trade,
        calculate_calmar_ratio,
        calculate_expected_value,
        calculate_max_drawdown,
        calculate_pareto_ratio,
        calculate_profit_factor,
        calculate_profit_loss_ratio,
        calculate_win_rate,
        get_trade_pairs,
    )

# Weekly trailing stop functions
try:
    from backtest.weekly_trailing import aggregate_to_weekly, check_weekly_trailing_stop, is_week_end
except (ModuleNotFoundError, ImportError):
    from weekly_trailing import aggregate_to_weekly, check_weekly_trailing_stop, is_week_end


class Backtest:
    def __init__(
        self,
        start_date=None,
        end_date=None,
        short_ma=5,
        long_ma=200,
        initial_capital=50000,
        slippage=0.0005,
        commission=0.0001,
        use_saved_data=False,
        debug=False,
        threshold=0.5,
        ma_type='ema',
        symbol='SSO',
        stop_loss_pct=0.08,
        disable_short_ma_entry=False,
        use_trailing_stop=False,
        trailing_stop_pct=0.2,
        background_exit_threshold=0.5,
        use_background_color_signals=False,
        partial_exit=False,
        no_show_plot=False,
        # TradingView alignment parameters
        tv_mode=True,
        tv_pine_compat=False,
        tv_breadth_csv=None,
        tv_price_csv=None,
        pivot_len_long=20,
        pivot_len_short=10,
        prom_thresh_long=0.005,
        prom_thresh_short=0.03,
        peak_level=0.70,
        trough_level_long=0.40,
        trough_level_short=0.20,
        no_pyramiding=True,
        # Two-stage exit parameters
        two_stage_exit=False,
        stage2_exit_mode='trend_break',
        # Volatility stop parameters
        use_volatility_stop=False,
        vol_atr_period=14,
        vol_atr_multiplier=2.5,
        vol_trailing_mode=True,
        # Bullish regime suppression
        bullish_regime_suppression=False,
        bullish_breadth_threshold=0.55,
        # Chart-mode peak/trough detection
        chart_mode=False,
        # Weekly trailing stop parameters
        enable_weekly_trailing=False,
        weekly_trailing_type='weekly_ema',
        weekly_ema_period=10,
        weekly_nweek_low_period=4,
        weekly_transition_weeks=3,
    ):
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
        self.use_background_color_signals = (
            use_background_color_signals  # Whether to use signals based on background color changes
        )
        self.partial_exit = partial_exit  # Whether to sell only half of the position on exit
        self.no_show_plot = no_show_plot  # Whether to not show the plot

        # TradingView alignment parameters
        self.tv_mode = tv_mode
        self.tv_pine_compat = tv_pine_compat
        self.tv_breadth_csv = tv_breadth_csv
        self.tv_price_csv = tv_price_csv
        self.pivot_len_long = pivot_len_long
        self.pivot_len_short = pivot_len_short
        self.prom_thresh_long = prom_thresh_long
        self.prom_thresh_short = prom_thresh_short
        self.peak_level = peak_level
        self.trough_level_long = trough_level_long
        self.trough_level_short = trough_level_short
        self.no_pyramiding = no_pyramiding

        # Two-stage exit parameters (TV mode only)
        self.two_stage_exit = two_stage_exit
        self.stage2_exit_mode = stage2_exit_mode

        # Volatility stop parameters (TV mode only)
        self.use_volatility_stop = use_volatility_stop
        self.vol_atr_period = vol_atr_period
        self.vol_atr_multiplier = vol_atr_multiplier
        self.vol_trailing_mode = vol_trailing_mode

        # Bullish regime suppression (TV mode only)
        self.bullish_regime_suppression = bullish_regime_suppression
        self.bullish_breadth_threshold = bullish_breadth_threshold

        # Weekly trailing stop
        self.enable_weekly_trailing = enable_weekly_trailing
        self.weekly_trailing_type = weekly_trailing_type
        self.weekly_ema_period = weekly_ema_period
        self.weekly_nweek_low_period = weekly_nweek_low_period
        self.weekly_transition_weeks = weekly_transition_weeks

        # Weekly trailing mode check
        if self.enable_weekly_trailing and self.tv_pine_compat:
            raise ValueError('--enable_weekly_trailing cannot be used with --tv_pine_compat')

        # Chart-mode peak/trough detection
        self.chart_mode = chart_mode
        if self.chart_mode and (self.tv_mode or self.tv_pine_compat):
            raise ValueError('--chart_mode cannot be used with --tv_mode or --tv_pine_compat')

        # Variables to store backtest results
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_capital = initial_capital
        self.current_position = 0
        self.entry_prices = []  # List to record entry prices
        self.highest_price = None  # Highest price during position holding

        # Trade logging variables (Phase 1)
        self.trade_log = []  # Detailed trade log for each complete trade
        self.open_positions = []  # Currently open positions
        self.next_trade_id = 1  # Counter for trade IDs
        self._cached_trade_pairs = None  # Cache for _get_trade_pairs()

        # Two-stage exit state
        self._half_exited = False
        self._stage1_exit_date = None
        self._pending_trend_break = False

        # Next-bar execution queue (tv_pine_compat only)
        self._pending_entry = None  # (reason, capital_frac) or None
        self._pending_exit = None  # (reason,) or None

        # Force Pine-compatible defaults if requested.
        if self.tv_pine_compat:
            self._apply_tv_pine_compat_defaults(stop_loss_pct=stop_loss_pct)

    def _apply_tv_pine_compat_defaults(self, stop_loss_pct=0.10):
        """Apply TradingView Pine-script-compatible defaults.

        This mode intentionally disables non-Pine extensions so behavior can be
        compared against the original Pine strategy more directly.
        """
        self.tv_mode = True
        self.no_pyramiding = True

        # Disable extended exit features that do not exist in the reference Pine.
        self.two_stage_exit = False
        self.use_volatility_stop = False
        self.bullish_regime_suppression = False
        self.use_trailing_stop = False
        self.use_background_color_signals = False
        self.partial_exit = False

        # Align trading costs / stop model to Pine defaults.
        # Do NOT override stop_loss_pct — honor caller's value.
        self.slippage = 0.0
        self.commission = 0.0002

        # Lock signal detection parameters to match reference Pine script.
        self.short_ma = 5
        self.long_ma = 200
        self.ma_type = 'ema'
        self.pivot_len_long = 20
        self.pivot_len_short = 10
        self.prom_thresh_long = 0.005
        self.prom_thresh_short = 0.03
        self.peak_level = 0.70
        self.trough_level_long = 0.40
        self.trough_level_short = 0.20
        self.disable_short_ma_entry = False

        if not self.tv_breadth_csv:
            import warnings

            warnings.warn('tv_pine_compat is most accurate with --tv_breadth_csv', UserWarning, stacklevel=2)

    def _validate_weekly_trailing_columns(self):
        """Validate that price_data has all OHLC columns needed for weekly trailing.

        Weekly trailing always requires full OHLC so that aggregate_to_weekly
        uses the adjusted OHLC path (not the close-only fallback).
        """
        required_cols = {'adjusted_close', 'open', 'close', 'high', 'low'}
        missing = required_cols - set(self.price_data.columns)
        if missing:
            raise RuntimeError(
                f'Weekly trailing requires columns {required_cols} but missing: {missing}. '
                'Ensure OHLC data is available for the symbol.'
            )

    def _load_tv_price_data(self):
        """Load OHLC price data from a TV-exported CSV."""
        import pathlib as _pathlib

        path = _pathlib.Path(self.tv_price_csv)
        if not path.exists():
            raise FileNotFoundError(f'TV price CSV not found: {path}')

        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f'TV price CSV is empty: {path}')

        # Case-insensitive column lookup
        col_map = {c.lower(): c for c in df.columns}

        # Resolve date column
        date_col = None
        for candidate in ('date', 'time'):
            if candidate in col_map:
                date_col = col_map[candidate]
                break
        if date_col is None:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None).dt.normalize()
        df.set_index(date_col, inplace=True)

        # Map standard columns
        ohlc_map = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}
        result = pd.DataFrame(index=df.index)
        for std_name, _ in ohlc_map.items():
            src = col_map.get(std_name)
            if src and src in df.columns:
                result[std_name] = pd.to_numeric(df[src], errors='coerce')

        if 'close' in result.columns:
            result['adjusted_close'] = result['close']

        # Validation
        if self.tv_pine_compat:
            required = {'open', 'high', 'low', 'close', 'adjusted_close'}
            missing = required - set(result.columns)
            if missing:
                raise ValueError(f'TV price CSV missing required columns for pine_compat: {missing}')
        else:
            if 'adjusted_close' not in result.columns:
                raise ValueError('TV price CSV must contain at least a close column')

        result = result.sort_index()
        if self.debug:
            print(f'\nLoaded TV price data from {path}')
            print(f'  Shape: {result.shape}, Columns: {list(result.columns)}')
            print(f'  Range: {result.index.min()} to {result.index.max()}')

        return result

    def _load_tv_breadth_index(self):
        """Load breadth index from user-provided CSV (typically TV export)."""
        if not self.tv_breadth_csv:
            raise ValueError('tv_breadth_csv is not configured')

        breadth_series = load_breadth_series_from_csv(self.tv_breadth_csv)
        if breadth_series.empty:
            raise ValueError(f'No breadth data found in CSV: {self.tv_breadth_csv}')

        breadth_series = breadth_series.astype(float).sort_index()
        if breadth_series.max() > 1.5:
            breadth_series = breadth_series / 100.0

        breadth_series.name = 'breadth_index'
        if self.debug:
            print('\nLoaded TV breadth series:')
            print(f'  Source: {self.tv_breadth_csv}')
            print(f'  Range: {breadth_series.index.min()} to {breadth_series.index.max()}')
            print(f'  Min/Max: {breadth_series.min():.4f} / {breadth_series.max():.4f}')

        return breadth_series

    def run(self):
        """Execute the backtest"""
        # SP500 data is only needed when breadth or price must be derived from it.
        needs_sp500 = not self.tv_breadth_csv or not (getattr(self, 'tv_price_csv', None) or self.tv_pine_compat)
        if needs_sp500:
            self.sp500_data = self._get_sp500_data()
            if self.sp500_data.empty:
                print('Failed to retrieve data.')
                return
            print(f'\nData period: {self.sp500_data.index.min()} to {self.sp500_data.index.max()}')
        else:
            self.sp500_data = pd.DataFrame()

        # Extract price data for the specified symbol.
        # Priority: tv_price_csv > tv_pine_compat OHLC > sp500_data column > individual fetch.
        if getattr(self, 'tv_price_csv', None):
            self.price_data = self._load_tv_price_data()
        elif self.tv_pine_compat or self.tv_mode or self.enable_weekly_trailing:
            ohlc = get_stock_price_ohlc(
                self.symbol,
                self.start_date,
                self.end_date,
                use_saved_data=self.use_saved_data,
            )
            if not ohlc.empty and 'adjusted_close' in ohlc.columns:
                self.price_data = ohlc
            elif self.symbol in self.sp500_data.columns:
                self.price_data = pd.DataFrame(self.sp500_data[self.symbol], columns=['adjusted_close'])
            else:
                self.price_data = get_stock_price_data(
                    self.symbol,
                    self.start_date,
                    self.end_date,
                    use_saved_data=self.use_saved_data,
                )
                if isinstance(self.price_data, pd.Series):
                    self.price_data = pd.DataFrame(self.price_data, columns=['adjusted_close'])
        elif self.symbol in self.sp500_data.columns:
            self.price_data = pd.DataFrame(self.sp500_data[self.symbol], columns=['adjusted_close'])
        else:
            self.price_data = get_stock_price_data(
                self.symbol,
                self.start_date,
                self.end_date,
                use_saved_data=self.use_saved_data,
            )
            if isinstance(self.price_data, pd.Series):
                self.price_data = pd.DataFrame(self.price_data, columns=['adjusted_close'])

        # Compute adjusted OHLC from raw OHLC + adjustment ratio (for price-scale consistency)
        if 'close' in self.price_data.columns and 'adjusted_close' in self.price_data.columns:
            adj_ratio = self.price_data['adjusted_close'] / self.price_data['close']
            for raw_col, adj_col in [('open', 'adjusted_open'), ('high', 'adjusted_high'), ('low', 'adjusted_low')]:
                if raw_col in self.price_data.columns:
                    self.price_data[adj_col] = self.price_data[raw_col] * adj_ratio

        # Validate columns required for weekly trailing (stricter check)
        if self.enable_weekly_trailing:
            self._validate_weekly_trailing_columns()

        # Build breadth source (S&P500-derived breadth or external TV-compatible breadth CSV).
        if self.tv_breadth_csv:
            self.breadth_index = self._load_tv_breadth_index()
            # Keep a DataFrame so existing chart/report code paths remain compatible.
            self.above_ma = pd.DataFrame({'TV_BREADTH': self.breadth_index}, index=self.breadth_index.index)
        else:
            self.above_ma = calculate_above_ma(self.sp500_data)
            self.breadth_index = self.above_ma.mean(axis=1)

        # Strip timezone info to prevent tz-aware vs tz-naive mismatch in intersection
        if self.price_data.index.tz is not None:
            self.price_data.index = self.price_data.index.tz_localize(None)
        self.price_data.index = self.price_data.index.normalize()
        if self.breadth_index.index.tz is not None:
            self.breadth_index.index = self.breadth_index.index.tz_localize(None)
        self.breadth_index.index = self.breadth_index.index.normalize()

        # Ensure data period consistency
        common_dates = self.price_data.index.intersection(self.breadth_index.index)
        self.price_data = self.price_data.loc[common_dates]
        self.breadth_index = self.breadth_index.loc[common_dates]
        self.above_ma = self.above_ma.loc[common_dates]

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
            calculate_trend_with_hysteresis(self.long_ma_line), index=self.long_ma_line.index
        )

        # Ensure DatetimeIndex for reliable comparisons
        for attr in ('price_data', 'breadth_index', 'short_ma_line', 'long_ma_line', 'long_ma_trend'):
            obj = getattr(self, attr)
            if not isinstance(obj.index, pd.DatetimeIndex):
                obj.index = pd.to_datetime(obj.index)

        # Extract data for the specified period only (for backtest)
        mask = (self.price_data.index >= pd.to_datetime(self.start_date)) & (
            self.price_data.index <= pd.to_datetime(self.end_date)
        )
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

        # Pre-compute TV-style signals if tv_mode is enabled
        if self.tv_mode:
            self._precompute_tv_signals()

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
                if (
                    pd.to_datetime(calculation_start_date) >= saved_data.index.min()
                    and pd.to_datetime(self.end_date) <= saved_data.index.max()
                ):
                    # Extract data for the calculation period
                    mask = (saved_data.index >= pd.to_datetime(calculation_start_date)) & (
                        saved_data.index <= pd.to_datetime(self.end_date)
                    )
                    return saved_data.loc[mask]

        # Get S&P500 ticker list
        tickers = get_sp500_tickers_from_fmp()
        print(f'Number of tickers retrieved: {len(tickers)}')

        # Get data for all stocks (from calculation start date)
        all_data = get_multiple_stock_data(
            tickers,
            calculation_start_date,  # Use calculation start date
            self.end_date,
            use_saved_data=self.use_saved_data,
        )

        if not all_data.empty:
            # Save data
            save_stock_data(all_data, filename)
            # Extract data for the calculation period
            mask = (all_data.index >= pd.to_datetime(calculation_start_date)) & (
                all_data.index <= pd.to_datetime(self.end_date)
            )
            return all_data.loc[mask]

        return pd.DataFrame()

    def _process_pending_weekly_exit(self, i, date, price, pending):
        """Process a pending weekly trailing stop exit at bar open. Returns True if fired."""
        if pending and self.current_position > 0:
            exit_price = (
                self.price_data['adjusted_open'].iloc[i] if 'adjusted_open' in self.price_data.columns else price
            )
            self._execute_exit(date, exit_price, reason='weekly trailing', force_full_exit=True)
            return True
        return False

    def _check_weekly_trailing_at_week_end(self, i, date, price, weekly_df):
        """Check weekly trailing stop at week-end. Returns True if should set pending exit."""
        if self.enable_weekly_trailing and self.current_position > 0 and self.open_positions:
            if is_week_end(self.price_data.index, i):
                earliest_entry = min(pos['entry_date'] for pos in self.open_positions)
                return check_weekly_trailing_stop(
                    current_close=price,
                    weekly_df=weekly_df,
                    entry_date=earliest_entry,
                    current_date=date,
                    trailing_type=self.weekly_trailing_type,
                    ema_period=self.weekly_ema_period,
                    nweek_low_period=self.weekly_nweek_low_period,
                    transition_weeks=self.weekly_transition_weeks,
                )
        return False

    def execute_trades(self):
        """Execute trades"""
        # 検出済みのシグナルを記録する変数（_detect_legacy_signals で使用）
        self._detected_short_ma_bottoms = set()
        self._detected_long_ma_bottoms = set()
        self._detected_peaks = set()

        # Weekly trailing stop preparation
        if self.enable_weekly_trailing:
            weekly_df = aggregate_to_weekly(self.price_data)
            _pending_weekly_exit = False
        else:
            weekly_df = None
            _pending_weekly_exit = False

        # Execute trades
        for i, date in enumerate(self.price_data.index):
            price = self.price_data.loc[date, 'adjusted_close']

            # Process pending weekly trailing exit at bar open
            weekly_exit_fired = self._process_pending_weekly_exit(i, date, price, _pending_weekly_exit)
            if weekly_exit_fired:
                _pending_weekly_exit = False

            if self.tv_mode:
                if not weekly_exit_fired:
                    self._execute_tv_mode_trades(i, date, price)
            else:
                self._detect_legacy_signals(i, date)
                if not weekly_exit_fired:
                    self._execute_legacy_trades(i, date, price)

            # Check weekly trailing stop at week-end
            if not _pending_weekly_exit and self._check_weekly_trailing_at_week_end(i, date, price, weekly_df):
                _pending_weekly_exit = True

            # Update equity curve
            self.equity_curve.append({'date': date, 'equity': self.current_capital + (self.current_position * price)})
            if self.debug:
                self._assert_equity_invariant(date, price)

        # Close any open positions at backtest end
        if self.current_position > 0:
            final_date = self.price_data.index[-1]
            final_price = self.price_data['adjusted_close'].iloc[-1]
            exit_reason = 'weekly trailing' if _pending_weekly_exit else 'backtest_end'
            self._execute_exit(final_date, final_price, reason=exit_reason, force_full_exit=_pending_weekly_exit)
            # Update final equity to reflect slippage/commission from forced close
            self.equity_curve[-1] = {
                'date': final_date,
                'equity': self.current_capital + (self.current_position * final_price),
            }
            if self.debug:
                self._assert_equity_invariant(final_date, final_price)

        print('\nTrade execution results:')
        print('-------------------')
        print(f'{self.short_ma}{self.ma_type.upper()} bottoms detected: {len(self.short_ma_bottoms)}')
        print(f'{self.long_ma}{self.ma_type.upper()} bottoms detected: {len(self.long_ma_bottoms)}')
        print(f'{self.long_ma}{self.ma_type.upper()} peaks detected: {len(self.peaks)}')

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
                        if 'background color change' in trade.get('reason', ''):
                            background_change_exits += 1

        print(f'Background color change exits: {background_change_exits}')

        # Calculate number of entries due to background color changes
        background_change_entries = 0
        for trade in self.trades:
            if trade['action'] == 'BUY':
                # Check if entry was due to background color change
                if 'background color change' in trade.get('reason', ''):
                    background_change_entries += 1

        print(f'Background color change entries: {background_change_entries}')

    def _detect_legacy_signals(self, i, date):
        """Legacy mode signal detection (non-TV mode).

        Detects short MA bottoms, long MA bottoms, and long MA peaks
        using data up to the current bar index. Appends to self.short_ma_bottoms,
        self.long_ma_bottoms, self.peaks. Uses self._detected_* sets for dedup.
        """
        current_breadth_index = self.breadth_index.iloc[: i + 1]
        current_short_ma_line = self.short_ma_line.iloc[: i + 1]
        current_long_ma_line = self.long_ma_line.iloc[: i + 1]

        data_start_date = current_short_ma_line.index[0].strftime('%Y-%m-%d')
        data_end_date = current_short_ma_line.index[-1].strftime('%Y-%m-%d')

        # Detect short MA bottoms
        short_ma_len_ok = self.chart_mode or len(current_short_ma_line) > self.short_ma
        if not self.disable_short_ma_entry and short_ma_len_ok:
            filter_threshold = 0.4 if self.chart_mode else self.threshold
            below_threshold_short = current_short_ma_line[current_short_ma_line < filter_threshold]
            if not below_threshold_short.empty:
                original_indices = np.where(current_short_ma_line < filter_threshold)[0]
                bottoms_short, _ = find_peaks(-below_threshold_short.values, prominence=0.02)
                for bottom_idx in bottoms_short:
                    original_idx = original_indices[bottom_idx]
                    bottom_date = current_short_ma_line.index[original_idx]
                    if self.chart_mode:
                        should_signal = True
                    elif original_idx >= 20:
                        past_20days_min = current_breadth_index.iloc[original_idx - 20 : original_idx + 1].min()
                        should_signal = past_20days_min <= 0.3
                    else:
                        should_signal = False
                    if should_signal and bottom_date not in self._detected_short_ma_bottoms:
                        self._detected_short_ma_bottoms.add(bottom_date)
                        signal_date = date
                        self.short_ma_bottoms.append(signal_date)
                        print(
                            f'New {self.short_ma}{self.ma_type.upper()} bottom detected at: '
                            f'{bottom_date.strftime("%Y-%m-%d")}'
                        )
                        print(f'  Data period: {data_start_date} to {data_end_date}')
                        print(f'  Signal date (trade execution): {signal_date.strftime("%Y-%m-%d")}')

        # Detect long MA bottoms
        if self.chart_mode or len(current_long_ma_line) > self.long_ma:
            fp_kwargs = {'prominence': 0.015}
            if self.chart_mode:
                fp_kwargs['distance'] = 50
            bottoms_long, _ = find_peaks(-current_long_ma_line.values, **fp_kwargs)
            for bottom_idx in bottoms_long:
                bottom_date = current_long_ma_line.index[bottom_idx]
                original_idx = bottom_idx
                if self.chart_mode:
                    should_signal = True
                elif original_idx >= 20:
                    past_20days_min = current_breadth_index.iloc[original_idx - 20 : original_idx + 1].min()
                    should_signal = past_20days_min <= 0.5
                else:
                    should_signal = False
                if should_signal and bottom_date not in self._detected_long_ma_bottoms:
                    self._detected_long_ma_bottoms.add(bottom_date)
                    signal_date = date
                    self.long_ma_bottoms.append(signal_date)
                    print(
                        f'New {self.long_ma}{self.ma_type.upper()} bottom detected at: '
                        f'{bottom_date.strftime("%Y-%m-%d")}'
                    )
                    print(f'  Data period: {data_start_date} to {data_end_date}')
                    print(f'  Signal date (trade execution): {signal_date.strftime("%Y-%m-%d")}')

        # Detect long MA peaks
        if self.chart_mode or len(current_long_ma_line) > self.long_ma:
            fp_kwargs = {'prominence': 0.015}
            if self.chart_mode:
                fp_kwargs['distance'] = 50
            peaks, _ = find_peaks(current_long_ma_line.values, **fp_kwargs)
            for peak_idx in peaks:
                peak_date = current_long_ma_line.index[peak_idx]
                if self.chart_mode:
                    should_signal = True
                else:
                    should_signal = current_long_ma_line.iloc[peak_idx] >= 0.5
                if should_signal and peak_date not in self._detected_peaks:
                    self._detected_peaks.add(peak_date)
                    signal_date = date
                    self.peaks.append(signal_date)
                    print(
                        f'New {self.long_ma}{self.ma_type.upper()} peak detected at: {peak_date.strftime("%Y-%m-%d")}'
                    )
                    print(f'  Data period: {data_start_date} to {data_end_date}')
                    print(f'  Signal date (trade execution): {signal_date.strftime("%Y-%m-%d")}')
                    print(f'  {self.long_ma}{self.ma_type.upper()} value: {current_long_ma_line.iloc[peak_idx]:.4f}')

    def _execute_tv_mode_trades(self, i, date, price):
        """TV mode trade execution logic (pine compat and same-bar execution).

        Also populates signal lists for summary output.
        """
        # Populate signal lists for summary output
        if date in getattr(self, '_tv_short_trough_signals', {}):
            self.short_ma_bottoms.append(date)
        if date in getattr(self, '_tv_long_trough_signals', {}):
            self.long_ma_bottoms.append(date)
        if date in getattr(self, '_tv_peak_signals', {}):
            self.peaks.append(date)

        if self.tv_pine_compat:
            self._execute_tv_pine_compat_bar(i, date, price)
        else:
            self._execute_tv_same_bar(i, date, price)

    def _execute_tv_pine_compat_bar(self, i, date, price):
        """TV Pine-compatible next-bar execution model."""
        skip_stop = False

        # Phase 0: Fill pending orders from previous bar at this bar's open
        if self._pending_exit is not None or self._pending_entry is not None:
            if 'adjusted_open' in self.price_data.columns:
                fill_price = self.price_data.loc[date, 'adjusted_open']
            elif 'open' in self.price_data.columns:
                fill_price = self.price_data.loc[date, 'open']
            else:
                fill_price = price

            if self._pending_exit is not None and self.current_position > 0:
                pend_reason = self._pending_exit[0]
                if self.debug:
                    print(
                        f'\n[COMPAT] Filling pending exit at {date.strftime("%Y-%m-%d")} '
                        f'open=${fill_price:.2f}, reason={pend_reason}'
                    )
                self._execute_exit(date, fill_price, reason=pend_reason)
                self._pending_exit = None

            if self._pending_entry is not None:
                pend_reason, pend_frac = self._pending_entry
                entry_amount = self.current_capital * pend_frac
                if entry_amount > 0:
                    shares = self._calculate_shares(entry_amount, fill_price)
                    if shares > 0:
                        if self.debug:
                            print(
                                f'\n[COMPAT] Filling pending entry at {date.strftime("%Y-%m-%d")} '
                                f'open=${fill_price:.2f}, reason={pend_reason}, shares={shares}'
                            )
                        if self._execute_entry(date, fill_price, shares, reason=pend_reason):
                            self.highest_price = fill_price
                            skip_stop = True
                self._pending_entry = None

        # Phase 1: Stop loss
        stop_loss_fired = False
        if not skip_stop and self.current_position > 0 and self.entry_prices:
            avg_entry = self._calculate_avg_entry_price()
            if self.highest_price is None or price > self.highest_price:
                self.highest_price = price
            stop_loss_price = avg_entry * (1 - self.stop_loss_pct)

            triggered = False
            fill_at = price
            if 'adjusted_low' in self.price_data.columns:
                bar_low = self.price_data.loc[date, 'adjusted_low']
                bar_open = self.price_data.loc[date, 'adjusted_open']
                if pd.notna(bar_low) and pd.notna(bar_open):
                    if bar_low <= stop_loss_price:
                        fill_at = min(bar_open, stop_loss_price)
                        triggered = True
                elif price <= stop_loss_price:
                    triggered = True
            elif price <= stop_loss_price:
                triggered = True

            if triggered:
                if self.debug:
                    print(
                        f'\n[COMPAT] Stop loss at {date.strftime("%Y-%m-%d")}, '
                        f'fill=${fill_at:.2f}, stop=${stop_loss_price:.2f}'
                    )
                self._execute_exit(date, fill_at, reason='stop loss')
                stop_loss_fired = True
                self._pending_entry = None
                self._pending_exit = None

        # Phase 2: Exit signal -> queue for next bar
        if not stop_loss_fired and self.current_position > 0:
            if self._pending_exit is None and date in self._tv_peak_signals:
                pivot_date, pivot_val = self._tv_peak_signals[date]
                if self.debug:
                    print(
                        f'\n[COMPAT] Queueing exit at peak '
                        f'(pivot {pivot_date.strftime("%Y-%m-%d")}, val={pivot_val:.4f}), '
                        f'bar={date.strftime("%Y-%m-%d")}'
                    )
                self._pending_exit = ('peak exit',)

        # Phase 3: Entry signal -> queue for next bar
        if not stop_loss_fired:
            can_enter = (self.current_position == 0) or (self._pending_exit is not None)
            if can_enter and self._pending_entry is None:
                entered = False
                if not self.disable_short_ma_entry and date in self._tv_short_trough_signals:
                    pivot_date, pivot_val = self._tv_short_trough_signals[date]
                    frac = 1.0 if self.no_pyramiding else 0.5
                    if self.debug:
                        print(
                            f'\n[COMPAT] Queueing entry at short trough '
                            f'(pivot {pivot_date.strftime("%Y-%m-%d")}, val={pivot_val:.4f}), '
                            f'bar={date.strftime("%Y-%m-%d")}'
                        )
                    self._pending_entry = ('short_ma_bottom', frac)
                    entered = True
                if not entered and date in self._tv_long_trough_signals:
                    pivot_date, pivot_val = self._tv_long_trough_signals[date]
                    if self.debug:
                        print(
                            f'\n[COMPAT] Queueing entry at long trough '
                            f'(pivot {pivot_date.strftime("%Y-%m-%d")}, val={pivot_val:.4f}), '
                            f'bar={date.strftime("%Y-%m-%d")}'
                        )
                    self._pending_entry = ('long_ma_bottom', 1.0)

        if self.current_position > 0 and not stop_loss_fired:
            if self.highest_price is None or price > self.highest_price:
                self.highest_price = price

    def _execute_tv_same_bar(self, i, date, price):
        """TV mode same-bar execution (non-pine-compat)."""
        stop_loss_fired = False
        exit_fired = False

        # Phase 1: Stop loss
        if self.current_position > 0 and self.entry_prices:
            avg_entry = self._calculate_avg_entry_price()
            if self.highest_price is None or price > self.highest_price:
                self.highest_price = price
            if self.use_volatility_stop:
                reference = self.highest_price if self.vol_trailing_mode else avg_entry
                stop_loss_price = self._compute_volatility_stop(i, reference)
            elif self.use_trailing_stop and self.highest_price is not None:
                stop_loss_price = self.highest_price * (1 - self.trailing_stop_pct)
            else:
                stop_loss_price = avg_entry * (1 - self.stop_loss_pct)

            triggered = False
            fill_at = price
            if 'adjusted_low' in self.price_data.columns:
                bar_low = self.price_data.loc[date, 'adjusted_low']
                bar_open = self.price_data.loc[date, 'adjusted_open']
                if pd.notna(bar_low) and pd.notna(bar_open):
                    if bar_low <= stop_loss_price:
                        fill_at = min(bar_open, stop_loss_price)
                        triggered = True
                elif price <= stop_loss_price:
                    triggered = True
            elif price <= stop_loss_price:
                triggered = True

            if triggered:
                print('\n[TV] Stop loss triggered:')
                print(f'Date: {date.strftime("%Y-%m-%d")}')
                print(f'Avg entry price: ${avg_entry:.2f}')
                print(f'Current price: ${price:.2f}')
                print(f'Stop loss price: ${stop_loss_price:.2f}')
                print(f'Fill price: ${fill_at:.2f}')
                self._execute_exit(date, fill_at, reason='stop loss')
                stop_loss_fired = True

        # Phase 2: Exit signal (peak / two-stage)
        if not stop_loss_fired and self.current_position > 0:
            if self.two_stage_exit:
                if not self._half_exited and date in self._tv_peak_signals:
                    if self._is_bullish_regime(i):
                        print(f'\n[TV] Peak suppressed by bullish regime at {date.strftime("%Y-%m-%d")}')
                    else:
                        pivot_date, pivot_val = self._tv_peak_signals[date]
                        print(
                            f'\n[TV] Stage 1 exit at peak '
                            f'(pivot {pivot_date.strftime("%Y-%m-%d")}, val={pivot_val:.4f}):'
                        )
                        print(f'Date: {date.strftime("%Y-%m-%d")}, Price: ${price:.2f}')
                        self._execute_stage1_exit(date, price)
                        exit_fired = True
                elif self._half_exited and self._check_trend_break(i):
                    print('\n[TV] Stage 2 trend break exit:')
                    print(f'Date: {date.strftime("%Y-%m-%d")}, Price: ${price:.2f}')
                    self._execute_exit(date, price, reason='trend break exit (stage 2)')
                    exit_fired = True
            else:
                if date in self._tv_peak_signals:
                    if self._is_bullish_regime(i):
                        print(f'\n[TV] Peak suppressed by bullish regime at {date.strftime("%Y-%m-%d")}')
                    else:
                        pivot_date, pivot_val = self._tv_peak_signals[date]
                        print(f'\n[TV] Exit at peak (pivot {pivot_date.strftime("%Y-%m-%d")}, val={pivot_val:.4f}):')
                        print(f'Date: {date.strftime("%Y-%m-%d")}, Price: ${price:.2f}')
                        self._execute_exit(date, price, reason='peak exit')
                        exit_fired = True

        # Phase 3: Entry signals
        if not stop_loss_fired and not exit_fired:
            if self.no_pyramiding and self.current_position > 0:
                pass
            else:
                entered = False
                if not self.disable_short_ma_entry and date in self._tv_short_trough_signals:
                    pivot_date, pivot_val = self._tv_short_trough_signals[date]
                    entry_amount = self.current_capital if self.no_pyramiding else self.current_capital / 2
                    if entry_amount > 0:
                        shares = self._calculate_shares(entry_amount, price)
                        if shares > 0:
                            if self._execute_entry(date, price, shares, reason='short_ma_bottom'):
                                self.highest_price = price
                                entered = True
                                print(
                                    f'\n[TV] Entry at short MA trough '
                                    f'(pivot {pivot_date.strftime("%Y-%m-%d")}, val={pivot_val:.4f}):'
                                )
                                print(f'Date: {date.strftime("%Y-%m-%d")}, Price: ${price:.2f}, Shares: {shares}')

                if not entered and date in self._tv_long_trough_signals:
                    pivot_date, pivot_val = self._tv_long_trough_signals[date]
                    entry_amount = self.current_capital
                    if entry_amount > 0:
                        shares = self._calculate_shares(entry_amount, price)
                        if shares > 0:
                            if self._execute_entry(date, price, shares, reason='long_ma_bottom'):
                                self.highest_price = price
                                print(
                                    f'\n[TV] Entry at long MA trough '
                                    f'(pivot {pivot_date.strftime("%Y-%m-%d")}, val={pivot_val:.4f}):'
                                )
                                print(f'Date: {date.strftime("%Y-%m-%d")}, Price: ${price:.2f}, Shares: {shares}')

        if self.current_position > 0 and not stop_loss_fired:
            if self.highest_price is None or price > self.highest_price:
                self.highest_price = price

    def _execute_legacy_trades(self, i, date, price):
        """Legacy mode trade execution logic.

        Signal priority mirrors TV mode: Phase 1 (stop loss) -> Phase 2 (exit) -> Phase 3 (entry).
        """
        stop_loss_fired = False
        exit_fired = False

        # Phase 1: Stop loss (checked FIRST — CR-001 fix)
        if self.current_position > 0 and self.entry_prices:
            latest_entry_price = self.entry_prices[-1]
            if self.highest_price is None or price > self.highest_price:
                self.highest_price = price
            if self.use_trailing_stop and self.highest_price is not None:
                stop_loss_price = self.highest_price * (1 - self.trailing_stop_pct)
            else:
                stop_loss_price = latest_entry_price * (1 - self.stop_loss_pct)
            if price <= stop_loss_price:
                print('\nStop loss triggered:')
                print(f'Date: {date.strftime("%Y-%m-%d")}')
                print(f'Entry price: ${latest_entry_price:.2f}')
                print(f'Current price: ${price:.2f}')
                print(f'Stop loss price: ${stop_loss_price:.2f}')
                if self.use_trailing_stop:
                    print(f'Highest price: ${self.highest_price:.2f}')
                    print(f'Trailing stop percentage: {self.trailing_stop_pct:.1%}')
                print(f'Shares: {self.current_position}')
                proceeds = self.current_position * price * (1 - self.slippage) * (1 - self.commission)
                print(f'Proceeds (after fees and slippage): ${proceeds:.2f}')
                self._execute_exit(date, price, reason='stop loss')
                print(f'Available capital: ${self.current_capital:.2f}')
                stop_loss_fired = True

        # Phase 2: Exit signals (peak exit, background color exit)
        if not stop_loss_fired and date in self.peaks and self.current_position > 0:
            print(f'\nExit at {self.long_ma}{self.ma_type.upper()} peak:')
            print(f'Date: {date.strftime("%Y-%m-%d")}')
            print(f'Price: ${price:.2f}')
            print(f'Shares: {self.current_position}')
            proceeds = self.current_position * price * (1 - self.slippage) * (1 - self.commission)
            print(f'Proceeds (after fees and slippage): ${proceeds:.2f}')
            self._execute_exit(date, price, reason='peak exit')
            print(f'Available capital: ${self.current_capital:.2f}')
            exit_fired = True

        if (
            not stop_loss_fired
            and not exit_fired
            and self.use_background_color_signals
            and self.current_position > 0
            and i > 0
        ):
            prev_trend = self.long_ma_trend.iloc[i - 1]
            prev_short_ma = self.short_ma_line.iloc[i - 1]
            prev_long_ma = self.long_ma_line.iloc[i - 1]
            prev_condition = not (prev_trend == -1 and prev_short_ma < prev_long_ma)
            current_trend = self.long_ma_trend.iloc[i]
            current_short_ma = self.short_ma_line.iloc[i]
            current_long_ma = self.long_ma_line.iloc[i]
            current_condition = current_trend == -1 and current_short_ma < current_long_ma
            if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                print('\nExit at background color change (trend change):')
                print(f'Date: {date.strftime("%Y-%m-%d")}')
                print(f'Price: ${price:.2f}')
                print(f'Shares: {self.current_position}')
                proceeds = self.current_position * price * (1 - self.slippage) * (1 - self.commission)
                print(f'Proceeds (after fees and slippage): ${proceeds:.2f}')
                self._execute_exit(date, price, reason='background color change')
                print(f'Available capital: ${self.current_capital:.2f}')
                exit_fired = True

        # Phase 3: Entry signals (MJ-005: respect no_pyramiding)
        if not stop_loss_fired and not exit_fired:
            if self.no_pyramiding and self.current_position > 0:
                pass  # Block entry when pyramiding is disabled
            elif not self.disable_short_ma_entry and date in self.short_ma_bottoms:
                if self.current_capital > 0:
                    entry_amount = self.current_capital / 2
                    shares = self._calculate_shares(entry_amount, price)
                    if shares > 0:
                        if self._execute_entry(date, price, shares, reason='short_ma_bottom'):
                            self.highest_price = price
                            print(f'\nEntry at {self.short_ma}{self.ma_type.upper()} bottom (buy more):')
                            print(f'Date: {date.strftime("%Y-%m-%d")}')
                            print(f'Price: ${price:.2f}')
                            print(f'Shares: {shares}')
                            print(f'Investment amount: ${entry_amount:.2f}')
                            print(f'Remaining available capital: ${self.current_capital:.2f}')
                            print(f'Total position: {self.current_position} shares')
                else:
                    print(f'\n{self.short_ma}{self.ma_type.upper()} bottom detected but no available capital:')
                    print(f'Date: {date.strftime("%Y-%m-%d")}')
                    print(f'Current position: {self.current_position} shares')

            elif date in self.long_ma_bottoms:
                if self.current_capital > 0:
                    entry_amount = self.current_capital
                    shares = self._calculate_shares(entry_amount, price)
                    if shares > 0:
                        if self._execute_entry(date, price, shares, reason='long_ma_bottom'):
                            self.highest_price = price
                            print(f'\nEntry at {self.long_ma}{self.ma_type.upper()} bottom (buy more):')
                            print(f'Date: {date.strftime("%Y-%m-%d")}')
                            print(f'Price: ${price:.2f}')
                            print(f'Shares: {shares}')
                            print(f'Investment amount: ${entry_amount:.2f}')
                            print(f'Total position: {self.current_position} shares')
                            print(f'Remaining available capital: ${self.current_capital:.2f}')
                else:
                    print(f'\n{self.long_ma}{self.ma_type.upper()} bottom detected but no available capital:')
                    print(f'Date: {date.strftime("%Y-%m-%d")}')
                    print(f'Current position: {self.current_position} shares')

            elif self.use_background_color_signals and self.current_position == 0 and i > 0:
                prev_trend = self.long_ma_trend.iloc[i - 1]
                prev_short_ma = self.short_ma_line.iloc[i - 1]
                prev_long_ma = self.long_ma_line.iloc[i - 1]
                prev_condition = prev_trend == -1 and prev_short_ma < prev_long_ma
                current_trend = self.long_ma_trend.iloc[i]
                current_short_ma = self.short_ma_line.iloc[i]
                current_long_ma = self.long_ma_line.iloc[i]
                current_condition = not (current_trend == -1 and current_short_ma < current_long_ma)
                if prev_condition and current_condition and current_long_ma >= self.background_exit_threshold:
                    if self.current_capital > 0:
                        entry_amount = self.current_capital
                        shares = self._calculate_shares(entry_amount, price)
                        if shares > 0:
                            if self._execute_entry(date, price, shares, reason='background_color_change'):
                                self.highest_price = price
                                print('\nEntry at background color change (pink to white):')
                                print(f'Date: {date.strftime("%Y-%m-%d")}')
                                print(f'Price: ${price:.2f}')
                                print(f'Shares: {shares}')
                                print(f'Investment amount: ${entry_amount:.2f}')
                                print(f'Total position: {self.current_position} shares')

        # Update highest price tracking
        if self.current_position > 0 and not stop_loss_fired:
            if self.highest_price is None or price > self.highest_price:
                self.highest_price = price

    def _calculate_shares(self, amount, price):
        """Calculate the number of shares that can be purchased (including commission)"""
        return int(amount / (price * (1 + self.slippage + self.commission)))

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
                entry_info, exit_date, exit_price, shares_to_match, proceeds_for_this_trade, exit_reason
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
                entry_info['entry_cost'] = (
                    entry_info['entry_cost'] / (entry_info['entry_shares'] + shares_to_match)
                ) * entry_info['entry_shares']

    def _record_completed_trade(self, entry_info, exit_date, exit_price, exit_shares, exit_proceeds, exit_reason):
        """Record a completed trade to trade_log (Phase 1)"""
        # Calculate holding days
        holding_days = (exit_date - entry_info['entry_date']).days

        # Calculate P&L
        entry_cost_per_share = entry_info['entry_cost'] / entry_info['entry_shares']
        entry_cost_for_sold_shares = entry_cost_per_share * exit_shares
        exit_proceeds_for_sold_shares = exit_proceeds if exit_shares > 0 else 0

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
            'cumulative_pnl': cumulative_pnl,
        }

        self.trade_log.append(trade_record)
        self.next_trade_id += 1

    def _execute_entry(self, date, price, shares, reason=''):
        """Execute entry. Returns True if successful, False if insufficient capital."""
        entry_price = price * (1 + self.slippage)
        commission = entry_price * shares * self.commission
        total_cost = entry_price * shares + commission

        if total_cost > self.current_capital:
            if self.debug:
                print(f'Insufficient capital: need {total_cost:.2f}, have {self.current_capital:.2f}')
            return False

        self.current_position += shares  # Changed to += to support buying more
        self.current_capital -= total_cost

        # Record entry price
        self.entry_prices.append(entry_price)

        self.trades.append(
            {
                'date': date,
                'action': 'BUY',
                'price': entry_price,
                'shares': shares,
                'commission': commission,
                'total_cost': total_cost,
                'reason': reason,
            }
        )

        # Add to open_positions for trade logging (Phase 1)
        self.open_positions.append(
            {
                'entry_date': date,
                'entry_price': entry_price,
                'entry_shares': shares,
                'entry_cost': total_cost,
                'entry_reason': reason,
            }
        )

        return True

    def _execute_exit(self, date, price, reason='', force_full_exit=False):
        """Execute exit"""
        exit_price = price * (1 - self.slippage)

        # Determine shares to sell FIRST (CR-002 fix)
        if self.partial_exit and not force_full_exit and self.current_position > 1:
            shares_to_sell = self.current_position // 2
            is_partial = True
        else:
            shares_to_sell = self.current_position
            is_partial = False

        # Calculate ONCE for the actual shares being sold
        commission = exit_price * shares_to_sell * self.commission
        total_proceeds = exit_price * shares_to_sell - commission
        self.current_capital += total_proceeds

        if is_partial:
            self.current_position -= shares_to_sell

            self.trades.append(
                {
                    'date': date,
                    'action': 'SELL',
                    'price': exit_price,
                    'shares': shares_to_sell,
                    'commission': commission,
                    'total_proceeds': total_proceeds,
                    'reason': reason + ' (partial)',
                }
            )

            # Record completed trades using FIFO logic (Phase 1)
            self._process_exit_fifo(date, exit_price, shares_to_sell, total_proceeds, reason + ' (partial)')

            print(f'Partial exit: Sold {shares_to_sell} shares, remaining {self.current_position} shares')
        else:
            self.trades.append(
                {
                    'date': date,
                    'action': 'SELL',
                    'price': exit_price,
                    'shares': shares_to_sell,
                    'commission': commission,
                    'total_proceeds': total_proceeds,
                    'reason': reason,
                }
            )

            # Record completed trades using FIFO logic (Phase 1)
            self._process_exit_fifo(date, exit_price, self.current_position, total_proceeds, reason)

            self.current_position = 0
            self.entry_prices = []  # Clear entry price list
            self.highest_price = None  # Reset highest price
            self._reset_exit_state()  # Reset two-stage exit state

    def _assert_equity_invariant(self, date, price):
        """Verify equity = cash + position_value (debug mode only)."""
        expected = self.current_capital + (self.current_position * price)
        actual = self.equity_curve[-1]['equity']
        if abs(expected - actual) > 0.01:
            raise AssertionError(f'Equity mismatch at {date}: expected={expected:.2f}, actual={actual:.2f}')

    def calculate_performance(self):
        """Calculate performance"""
        if not self.equity_curve:
            print('No equity data to calculate performance (empty price_data?).')
            self.equity_df = pd.DataFrame(columns=['equity'])
            self.total_return = 0
            self.cagr = 0
            self.annual_return = 0
            self.sharpe_ratio = 0
            self.max_drawdown = 0
            self.win_rate = 0
            self.profit_loss_ratio = 0
            self.profit_factor = 0
            self.calmar_ratio = 0
            self.expected_value = 0
            self.avg_pnl_per_trade = 0
            self.pareto_ratio = 0
            self.bh_total_return = 0
            self.bh_cagr = 0
            self.bh_sharpe = 0
            self.bh_max_drawdown = 0
            return
        self.equity_df = pd.DataFrame(self.equity_curve)
        self.equity_df.set_index('date', inplace=True)

        # Calculate final asset value (cash + position value)
        final_equity = self.equity_df['equity'].iloc[-1]

        # Total return (revised)
        self.total_return = (final_equity / self.initial_capital) - 1

        # Annual return and CAGR calculation
        days = (self.equity_df.index[-1] - self.equity_df.index[0]).days
        if days <= 0:
            self.cagr = 0.0
            self.annual_return = 0.0
        else:
            years = days / 365

            # CAGR calculation (handles negative returns)
            if self.total_return <= -1:
                self.cagr = -1.0
            else:
                self.cagr = (1 + self.total_return) ** (1 / years) - 1

            # Annual Return calculation (handles negative returns)
            if self.total_return <= -1:
                self.annual_return = -1.0
            else:
                self.annual_return = (1 + self.total_return) ** (365 / days) - 1

        # Add debug information
        if self.debug:
            print('\nReturn calculation details:')
            print(f'Initial capital: ${self.initial_capital:.2f}')
            print(f'Final equity: ${final_equity:.2f}')
            print(f'Current cash: ${self.current_capital:.2f}')
            print(f'Current position: {self.current_position} shares')
            if self.current_position > 0:
                current_price = self.price_data['adjusted_close'].iloc[-1]
                position_value = self.current_position * current_price
                print(f'Position value: ${position_value:.2f}')

        # Sharpe ratio
        daily_returns = self.equity_df['equity'].pct_change()
        std = daily_returns.std()
        self.sharpe_ratio = np.sqrt(252) * daily_returns.mean() / std if std > 0 else 0.0

        # Maximum drawdown
        self.max_drawdown = self._calculate_max_drawdown()

        # Cache trade pairs for reuse across metric calculations
        self._cached_trade_pairs = self._get_trade_pairs()

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

        if buy_hold_shares == 0 or days <= 0:
            self.bh_total_return = 0.0
            self.bh_cagr = 0.0
            self.bh_sharpe = 0.0
            self.bh_max_drawdown = 0.0
        else:
            buy_hold_cost = buy_hold_shares * initial_price * (1 + self.slippage) * (1 + self.commission)
            buy_hold_value = buy_hold_shares * final_price * (1 - self.slippage) * (1 - self.commission)
            buy_hold_return = (buy_hold_value / buy_hold_cost) - 1
            buy_hold_cagr = (buy_hold_value / buy_hold_cost) ** (1 / years) - 1

            # Buy & Hold daily returns
            buy_hold_daily_returns = self.price_data['adjusted_close'].pct_change()
            bh_std = buy_hold_daily_returns.std()
            buy_hold_sharpe = np.sqrt(252) * buy_hold_daily_returns.mean() / bh_std if bh_std > 0 else 0.0

            # Buy & Hold maximum drawdown
            buy_hold_cummax = self.price_data['adjusted_close'].expanding().max()
            buy_hold_drawdown = self.price_data['adjusted_close'] / buy_hold_cummax - 1
            buy_hold_max_drawdown = buy_hold_drawdown.min()

            # Store Buy & Hold metrics as instance attributes
            self.bh_total_return = buy_hold_return
            self.bh_cagr = buy_hold_cagr
            self.bh_sharpe = buy_hold_sharpe
            self.bh_max_drawdown = buy_hold_max_drawdown

        # Display performance metrics
        print('\nBacktest results:')
        print('-------------------')
        print('Strategy performance:')
        print(f'Total return: {self.total_return:.2%}')
        print(f'Annual return (CAGR): {self.cagr:.2%}')
        print(f'Sharpe ratio: {self.sharpe_ratio:.2f}')
        print(f'Maximum drawdown: {self.max_drawdown:.2%}')
        print(f'Win rate: {self.win_rate:.2%}')
        print(f'Profit-loss ratio: {self.profit_loss_ratio:.2f}')
        print(f'Profit Factor: {self.profit_factor:.2f}')
        print(f'Calmar Ratio: {self.calmar_ratio:.2f}')
        print(f'Expected Value: ${self.expected_value:.2f}')
        print(f'Avg. PnL per trade: ${self.avg_pnl_per_trade:.2f}')
        print(f'Pareto Ratio: {self.pareto_ratio:.2f}')

        print('\nBuy & Hold performance:')
        print(f'Total return: {self.bh_total_return:.2%}')
        print(f'Annual return (CAGR): {self.bh_cagr:.2%}')
        print(f'Sharpe ratio: {self.bh_sharpe:.2f}')
        print(f'Maximum drawdown: {self.bh_max_drawdown:.2%}')

        print(f'\nInvestment period: {days:.1f} days ({days / 365:.1f} years)')

        # Calculate relative performance
        relative_return = self.total_return - self.bh_total_return
        relative_cagr = self.cagr - self.bh_cagr
        print('\nRelative performance (Strategy vs Buy & Hold):')
        print(f'Return difference: {relative_return:.2%}')
        print(f'CAGR difference: {relative_cagr:.2%}')

    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown — delegates to performance_metrics module."""
        return calculate_max_drawdown(self.equity_df)

    def _calculate_win_rate(self):
        """Calculate win rate — delegates to performance_metrics module."""
        if not self.trades:
            return 0
        trade_pairs = self._cached_trade_pairs if self._cached_trade_pairs is not None else self._get_trade_pairs()
        return calculate_win_rate(trade_pairs, self.debug)

    def _calculate_profit_loss_ratio(self):
        """Calculate profit-loss ratio — delegates to performance_metrics module."""
        if not self.trades:
            return 0
        trade_pairs = self._cached_trade_pairs if self._cached_trade_pairs is not None else self._get_trade_pairs()
        return calculate_profit_loss_ratio(trade_pairs, self.debug)

    def _calculate_profit_factor(self):
        """Calculate profit factor — delegates to performance_metrics module."""
        if not self.trades:
            return 0
        trade_pairs = self._cached_trade_pairs if self._cached_trade_pairs is not None else self._get_trade_pairs()
        return calculate_profit_factor(trade_pairs)

    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio — delegates to performance_metrics module."""
        if not self.trades:
            return 0
        return calculate_calmar_ratio(self.total_return, self.max_drawdown, self.equity_df)

    def _calculate_expected_value(self):
        """Calculate expected value per trade — delegates to performance_metrics module."""
        if not self.trades:
            return 0
        trade_pairs = self._cached_trade_pairs if self._cached_trade_pairs is not None else self._get_trade_pairs()
        return calculate_expected_value(trade_pairs)

    def _calculate_avg_pnl_per_trade(self):
        """Calculate average PnL per trade — delegates to performance_metrics module."""
        if not self.trades:
            return 0
        trade_pairs = self._cached_trade_pairs if self._cached_trade_pairs is not None else self._get_trade_pairs()
        return calculate_avg_pnl_per_trade(trade_pairs)

    def _calculate_pareto_ratio(self):
        """Calculate Pareto ratio — delegates to performance_metrics module."""
        if not self.trades:
            return 0
        trade_pairs = self._cached_trade_pairs if self._cached_trade_pairs is not None else self._get_trade_pairs()
        return calculate_pareto_ratio(trade_pairs)

    def _get_trade_pairs(self):
        """Get trade pairs — delegates to performance_metrics module."""
        return get_trade_pairs(self.trades, self.debug)

    def visualize_results(self, show_plot=True):
        """Visualize results — delegates to visualization module."""
        try:
            from backtest.visualization import visualize_backtest_results
        except ModuleNotFoundError:
            from visualization import visualize_backtest_results
        visualize_backtest_results(self, show_plot)

    def save_trade_log(self, filename=None):
        """Save trade log to CSV file — delegates to visualization module."""
        try:
            from backtest.visualization import save_trade_log_csv
        except ModuleNotFoundError:
            from visualization import save_trade_log_csv
        return save_trade_log_csv(self.trade_log, self.symbol, self.start_date, self.end_date, filename)

    # --- TradingView alignment methods ---

    def _detect_pivot_high(self, series, pivot_len, prom_thresh, level_thresh):
        """Delegate to module-level detect_pivot_high()."""
        return detect_pivot_high(series, pivot_len, prom_thresh, level_thresh)

    def _detect_pivot_low(self, series, pivot_len, prom_thresh):
        """Delegate to module-level detect_pivot_low()."""
        return detect_pivot_low(series, pivot_len, prom_thresh)

    def _precompute_tv_signals(self):
        """Pre-compute TradingView-style pivot signals before trade execution."""
        # Peak signals on long MA (exit signals)
        raw_peaks = self._detect_pivot_high(
            self.long_ma_line, self.pivot_len_long, self.prom_thresh_long, self.peak_level
        )
        self._tv_peak_signals = {}
        for confirm_date, pivot_date, val in raw_peaks:
            if confirm_date not in self._tv_peak_signals:
                self._tv_peak_signals[confirm_date] = (pivot_date, val)

        # Long MA trough signals (entry signals)
        raw_long_troughs = self._detect_pivot_low(self.long_ma_line, self.pivot_len_long, self.prom_thresh_long)
        self._tv_long_trough_signals = {}
        for confirm_date, pivot_date, val in raw_long_troughs:
            if val < self.trough_level_long:
                if confirm_date not in self._tv_long_trough_signals:
                    self._tv_long_trough_signals[confirm_date] = (pivot_date, val)

        # Short MA trough signals (entry signals)
        raw_short_troughs = self._detect_pivot_low(self.short_ma_line, self.pivot_len_short, self.prom_thresh_short)
        self._tv_short_trough_signals = {}
        for confirm_date, pivot_date, val in raw_short_troughs:
            # Check recent 20-bar minimum of raw breadth <= trough_level_short
            confirm_loc = self.breadth_index.index.get_loc(confirm_date)
            start_loc = max(0, confirm_loc - 19)
            recent_min = self.breadth_index.iloc[start_loc : confirm_loc + 1].min()
            if recent_min <= self.trough_level_short:
                if confirm_date not in self._tv_short_trough_signals:
                    self._tv_short_trough_signals[confirm_date] = (pivot_date, val)

        if self.debug:
            print('\nTV signal pre-computation:')
            print(f'  Peak signals: {len(self._tv_peak_signals)}')
            print(f'  Long MA trough signals: {len(self._tv_long_trough_signals)}')
            print(f'  Short MA trough signals: {len(self._tv_short_trough_signals)}')

    def _calculate_avg_entry_price(self):
        """Calculate weighted average entry price across open positions.

        Equivalent to TradingView strategy.position_avg_price.
        """
        if not self.open_positions:
            return 0.0
        total_cost = sum(p['entry_cost'] for p in self.open_positions)
        total_shares = sum(p['entry_shares'] for p in self.open_positions)
        return total_cost / total_shares if total_shares > 0 else 0.0

    def _reset_exit_state(self):
        """Reset two-stage exit state after full exit."""
        self._half_exited = False
        self._stage1_exit_date = None
        self._pending_trend_break = False

    def _execute_stage1_exit(self, date, price):
        """Execute stage 1 exit: sell half of current position."""
        shares_to_sell = self.current_position // 2
        if shares_to_sell <= 0:
            return
        exit_price = price * (1 - self.slippage)
        commission = exit_price * shares_to_sell * self.commission
        proceeds = exit_price * shares_to_sell - commission

        self.current_position -= shares_to_sell
        self.current_capital += proceeds

        self.trades.append(
            {
                'date': date,
                'action': 'SELL',
                'price': exit_price,
                'shares': shares_to_sell,
                'commission': commission,
                'total_proceeds': proceeds,
                'reason': 'peak exit (stage 1)',
            }
        )

        # Record completed trades using FIFO logic
        self._process_exit_fifo(date, exit_price, shares_to_sell, proceeds, 'peak exit (stage 1)')

        self._half_exited = True
        self._stage1_exit_date = date

        print(f'  Stage 1: Sold {shares_to_sell} shares, remaining {self.current_position} shares')

    def _check_trend_break(self, i):
        """Check if trend break condition is met for stage 2 exit."""
        if self.stage2_exit_mode == 'trend_break':
            return self.long_ma_trend.iloc[i] == -1
        elif self.stage2_exit_mode == 'ma_cross':
            return self.short_ma_line.iloc[i] < self.long_ma_line.iloc[i]
        return False

    def _is_bullish_regime(self, i):
        """Check if current bar is in a bullish regime (suppress peak exits)."""
        if not self.bullish_regime_suppression:
            return False
        trend_up = self.long_ma_trend.iloc[i] == 1
        ma_above = self.short_ma_line.iloc[i] > self.long_ma_line.iloc[i]
        breadth_high = self.breadth_index.iloc[i] > self.bullish_breadth_threshold
        return trend_up and ma_above and breadth_high

    def _compute_volatility_stop(self, i, reference_price):
        """Compute dynamic stop price based on close-to-close volatility."""
        if i < self.vol_atr_period:
            return reference_price * (1 - self.stop_loss_pct)
        prices = self.price_data['adjusted_close'].iloc[max(0, i - self.vol_atr_period) : i + 1]
        daily_returns = prices.pct_change().dropna()
        volatility = daily_returns.std()
        stop_distance = volatility * self.vol_atr_multiplier * reference_price
        return reference_price - stop_distance


def main():
    try:
        from backtest.cli import build_argument_parser
    except ModuleNotFoundError:
        from cli import build_argument_parser

    parser = build_argument_parser()
    args = parser.parse_args()

    # Set default values if dates are not specified
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
        print(f'Using current date as end date: {args.end_date}')

    if args.start_date is None:
        start_date = pd.to_datetime(args.end_date) - pd.DateOffset(years=10)
        args.start_date = start_date.strftime('%Y-%m-%d')
        print(f'Using date 10 years before end date as start date: {args.start_date}')

    print(f'args.start_date: {args.start_date}')

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
        no_show_plot=args.no_show_plot,
        tv_mode=args.tv_mode,
        tv_pine_compat=args.tv_pine_compat,
        tv_breadth_csv=args.tv_breadth_csv,
        tv_price_csv=args.tv_price_csv,
        pivot_len_long=args.pivot_len_long,
        pivot_len_short=args.pivot_len_short,
        prom_thresh_long=args.prom_thresh_long,
        prom_thresh_short=args.prom_thresh_short,
        peak_level=args.peak_level,
        trough_level_long=args.trough_level_long,
        trough_level_short=args.trough_level_short,
        no_pyramiding=not args.pyramiding,
        two_stage_exit=args.two_stage_exit,
        stage2_exit_mode=args.stage2_exit_mode,
        use_volatility_stop=args.use_volatility_stop,
        vol_atr_period=args.vol_atr_period,
        vol_atr_multiplier=args.vol_atr_multiplier,
        vol_trailing_mode=args.vol_trailing_mode,
        bullish_regime_suppression=args.bullish_regime_suppression,
        bullish_breadth_threshold=args.bullish_breadth_threshold,
        chart_mode=args.chart_mode,
        enable_weekly_trailing=args.enable_weekly_trailing,
        weekly_trailing_type=args.weekly_trailing_type,
        weekly_ema_period=args.weekly_ema_period,
        weekly_nweek_low_period=args.weekly_nweek_low_period,
        weekly_transition_weeks=args.weekly_transition_weeks,
    )

    backtest.run()

    # Save trade log if trades were executed
    if backtest.trade_log:
        backtest.save_trade_log()


if __name__ == '__main__':
    main()
