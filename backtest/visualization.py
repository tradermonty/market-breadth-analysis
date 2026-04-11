"""Visualization functions for backtest results.

Extracted from backtest.py — matplotlib charts and trade log CSV export.
"""

import os
import platform
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to import market_breadth
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_breadth import plot_breadth_and_sp500_with_peaks


def setup_matplotlib_backend():
    """Set up matplotlib backend based on the operating system."""
    system = platform.system().lower()
    if system in ('darwin', 'windows'):
        try:
            matplotlib.use('TkAgg')
        except (ImportError, ModuleNotFoundError):
            matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')


def visualize_backtest_results(bt, show_plot=True):
    """Visualize backtest results using matplotlib.

    Takes a Backtest instance (bt) and renders 4-panel chart:
    1. Price chart with trade points
    2. Breadth index and moving averages
    3. Equity curve comparison (Strategy vs Buy & Hold)
    4. Drawdown comparison
    """
    if bt.equity_df.empty:
        print('No data to visualize.')
        return

    setup_matplotlib_backend()

    # Create subplots
    _fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))

    # Price chart and trade points
    ax1.plot(bt.price_data.index, bt.price_data['adjusted_close'], label=f'{bt.symbol} Price')

    # Display trade points
    for trade_idx, trade in enumerate(bt.trades):
        if trade['action'] == 'BUY':
            ax1.scatter(trade['date'], trade['price'], color='green', marker='^', s=100, label='Buy')
        elif trade['action'] == 'SELL':
            # Check if this was a stop loss by checking the previous entry price
            if trade_idx > 0:
                prev_trade = bt.trades[trade_idx - 1]
                if prev_trade['action'] == 'BUY':
                    entry_price = prev_trade['price']
                    stop_loss_price = entry_price * (1 - bt.stop_loss_pct)

                    # Determine if this was a stop loss
                    if trade['price'] <= stop_loss_price:
                        # Display stop loss trades with special markers
                        if bt.use_trailing_stop:
                            # Display in blue for trailing stop
                            ax1.scatter(
                                trade['date'],
                                trade['price'],
                                color='blue',
                                marker='x',
                                s=150,
                                label='Trailing Stop',
                            )
                        else:
                            # Display in purple for regular stop loss
                            ax1.scatter(
                                trade['date'], trade['price'], color='purple', marker='x', s=150, label='Stop Loss'
                            )
                    else:
                        # Regular sell
                        ax1.scatter(trade['date'], trade['price'], color='red', marker='v', s=100, label='Sell')

    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(0.02, 0.5))

    ax1.set_title(f'{bt.symbol} Price Chart with Trade Points')

    # Breadth Index and moving averages
    ax2.plot(bt.breadth_index.index, bt.breadth_index, label='Breadth Index')
    ax2.plot(bt.short_ma_line.index, bt.short_ma_line, label=f'{bt.short_ma}{bt.ma_type.upper()}')
    ax2.plot(bt.long_ma_line.index, bt.long_ma_line, label=f'{bt.long_ma}{bt.ma_type.upper()}')

    # Set background color (based on trend)
    for i in range(len(bt.long_ma_trend) - 1):
        if bt.long_ma_trend.iloc[i] == -1 and bt.short_ma_line.iloc[i] < bt.long_ma_line.iloc[i]:
            ax2.axvspan(bt.long_ma_line.index[i], bt.long_ma_line.index[i + 1], color=(1.0, 0.9, 0.96), alpha=0.3)

    # Detect and display background color change points
    white_to_pink_changes = []  # White to pink change (exit)
    pink_to_white_changes = []  # Pink to white change (entry)

    # Only detect background color changes if use_background_color_signals is enabled
    if bt.use_background_color_signals:
        for i in range(1, len(bt.long_ma_trend)):
            prev_trend = bt.long_ma_trend.iloc[i - 1]
            prev_short_ma = bt.short_ma_line.iloc[i - 1]
            prev_long_ma = bt.long_ma_line.iloc[i - 1]

            # Today's data
            current_trend = bt.long_ma_trend.iloc[i]
            current_short_ma = bt.short_ma_line.iloc[i]
            current_long_ma = bt.long_ma_line.iloc[i]

            # White to pink change (exit)
            prev_condition = not (prev_trend == -1 and prev_short_ma < prev_long_ma)
            current_condition = current_trend == -1 and current_short_ma < current_long_ma

            if prev_condition and current_condition and current_long_ma >= bt.background_exit_threshold:
                white_to_pink_changes.append(bt.long_ma_line.index[i])

            # Pink to white change (entry)
            prev_condition = prev_trend == -1 and prev_short_ma < prev_long_ma
            current_condition = not (current_trend == -1 and current_short_ma < current_long_ma)

            if prev_condition and current_condition and current_long_ma >= bt.background_exit_threshold:
                pink_to_white_changes.append(bt.long_ma_line.index[i])

    # Display white to pink change points (exit)
    if white_to_pink_changes and bt.use_background_color_signals:
        ax2.scatter(
            white_to_pink_changes,
            bt.long_ma_line[white_to_pink_changes],
            color='orange',
            marker='x',
            s=150,
            label=f'White to Pink (Exit, MA≥{bt.background_exit_threshold:.2f})',
        )

    # Display pink to white change points (entry)
    if pink_to_white_changes and bt.use_background_color_signals:
        ax2.scatter(
            pink_to_white_changes,
            bt.long_ma_line[pink_to_white_changes],
            color='green',
            marker='^',
            s=150,
            label=f'Pink to White (Entry, MA≥{bt.background_exit_threshold:.2f})',
        )

    # Display signal points
    ax2.scatter(
        bt.short_ma_bottoms,
        bt.short_ma_line[bt.short_ma_bottoms],
        color='green',
        marker='^',
        s=100,
        label=f'{bt.short_ma}{bt.ma_type.upper()} Bottom',
    )
    ax2.scatter(
        bt.long_ma_bottoms,
        bt.long_ma_line[bt.long_ma_bottoms],
        color='blue',
        marker='^',
        s=100,
        label=f'{bt.long_ma}{bt.ma_type.upper()} Bottom',
    )
    ax2.scatter(
        bt.peaks,
        bt.long_ma_line[bt.peaks],
        color='red',
        marker='v',
        s=100,
        label=f'{bt.long_ma}{bt.ma_type.upper()} Peak',
    )

    ax2.set_title('Breadth Index and Moving Averages')
    ax2.legend(loc='center left', bbox_to_anchor=(0.02, 0.5))

    # Equity curve comparison
    initial_price = bt.price_data['adjusted_close'].iloc[0]
    buy_hold_shares = int(bt.initial_capital / (initial_price * (1 + bt.slippage)))
    buy_hold_equity = bt.price_data['adjusted_close'] * buy_hold_shares

    ax3.plot(bt.equity_df.index, bt.equity_df['equity'], label='Strategy')
    ax3.plot(buy_hold_equity.index, buy_hold_equity, label='Buy & Hold')
    ax3.set_title('Equity Curve Comparison')
    ax3.legend(loc='center left', bbox_to_anchor=(0.02, 0.5))

    # Drawdown chart
    equity = bt.equity_df['equity']
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
    plt.savefig(f'reports/backtest_results_{bt.symbol}.png')
    if show_plot:
        plt.show()  # Display chart
    plt.close(_fig)

    # Generate Plotly breadth chart with TV signal markers (if TV mode)
    if bt.tv_mode and hasattr(bt, '_tv_peak_signals'):
        # Merge long + short trough signals into one dict for the chart
        tv_trough_merged = {}
        for sig_dict in (
            getattr(bt, '_tv_long_trough_signals', {}),
            getattr(bt, '_tv_short_trough_signals', {}),
        ):
            for k, v in sig_dict.items():
                if k not in tv_trough_merged:
                    tv_trough_merged[k] = v

        # Extract S&P500 price Series for the chart (expects 1-D, not multi-column DF)
        if 'SPY' in bt.sp500_data.columns:
            sp500_price_series = bt.sp500_data['SPY']
        else:
            # Fallback: use the backtest symbol's price data
            sp500_price_series = bt.price_data['adjusted_close']
            sp500_price_series.name = bt.symbol

        try:
            plot_breadth_and_sp500_with_peaks(
                bt.above_ma,
                sp500_price_series,
                short_ma_period=bt.short_ma,
                start_date=bt.start_date,
                end_date=bt.end_date,
                output_dir='reports',
                tv_peak_signals=bt._tv_peak_signals,
                tv_trough_signals=tv_trough_merged,
            )
        except Exception as e:
            print(f'Plotly chart generation skipped: {e}')


def save_trade_log_csv(trade_log, symbol, start_date, end_date, filename=None):
    """Save trade log to CSV file.

    Args:
        trade_log: List of trade log dictionaries.
        symbol: Stock symbol.
        start_date: Backtest start date string.
        end_date: Backtest end date string.
        filename: Optional output filename. Auto-generated if None.

    Returns:
        The filename where the log was saved, or None if no trades.
    """
    if not trade_log:
        print('No trades to save.')
        return None

    # Generate default filename if not provided
    if filename is None:
        filename = f'reports/trade_log_{symbol}_{start_date}_{end_date}.csv'

    # Convert trade_log to DataFrame
    trade_df = pd.DataFrame(trade_log)

    # Format datetime columns
    trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date']).dt.strftime('%Y-%m-%d')
    trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date']).dt.strftime('%Y-%m-%d')

    # Save to CSV
    trade_df.to_csv(filename, index=False)
    print(f'\nTrade log saved to: {filename}')

    return filename
