"""CLI argument definitions for the backtest module.

Extracted from backtest.py main() — keeps argparse definitions in one place.
"""

import argparse


def build_argument_parser():
    """Build and return the argument parser for backtest CLI."""
    parser = argparse.ArgumentParser(description='Backtest using Market Breadth indicator')
    parser.add_argument(
        '--start_date',
        type=str,
        help='Backtest start date (YYYY-MM-DD format). If not specified, 10 years before end date',
    )
    parser.add_argument(
        '--end_date', type=str, help='Backtest end date (YYYY-MM-DD format). If not specified, current date'
    )
    parser.add_argument('--short_ma', type=int, default=5, help='Short-term moving average period (default: 5)')
    parser.add_argument('--long_ma', type=int, default=200, help='Long-term moving average period (default: 200)')
    parser.add_argument(
        '--initial_capital', type=float, default=50000, help='Initial investment amount (default: 50000 dollars)'
    )
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage (default: 0.05%%)')
    parser.add_argument('--commission', type=float, default=0.0001, help='Transaction fee (default: 0.01%%)')
    parser.add_argument('--use_saved_data', action='store_true', help='Whether to use saved data')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for bottom detection (default: 0.5)')
    parser.add_argument('--ma_type', type=str, default='ema', help='Moving average type (default: ema)')
    parser.add_argument('--symbol', type=str, default='SSO', help='Stock symbol (default: SSO)')
    parser.add_argument('--stop_loss_pct', type=float, default=0.08, help='Stop loss percentage (default: 8%%)')
    parser.add_argument('--disable_short_ma_entry', action='store_true', help='Disable short-term moving average entry')
    parser.add_argument('--use_trailing_stop', action='store_true', help='Use trailing stop instead of fixed stop loss')
    parser.add_argument('--trailing_stop_pct', type=float, default=0.2, help='Trailing stop percentage (default: 20%%)')
    parser.add_argument(
        '--background_exit_threshold', type=float, default=0.5, help='Background exit threshold (default: 0.5)'
    )
    parser.add_argument(
        '--use_background_color_signals',
        action='store_true',
        help='Use background color change signals for entry and exit',
    )
    parser.add_argument(
        '--partial_exit', action='store_true', help='Exit with half of the position when exit signal is triggered'
    )
    parser.add_argument(
        '--no_show_plot', action='store_true', help='Do not show plot after saving (default: show plot)'
    )

    # TradingView alignment options
    parser.add_argument(
        '--tv_mode',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='TradingView-aligned signal detection (default: on)',
    )
    parser.add_argument('--tv_pine_compat', action='store_true', help='Enable Pine-compatible TV backtest mode')
    parser.add_argument(
        '--tv_breadth_csv',
        type=str,
        default=None,
        help='Path to breadth CSV (e.g., S5TH export with date/close columns)',
    )
    parser.add_argument(
        '--tv_price_csv',
        type=str,
        default=None,
        help='Path to TV-exported price CSV (date,open,high,low,close)',
    )
    parser.add_argument(
        '--pivot_len_long', type=int, default=20, help='Pivot confirmation bars for long MA (default: 20)'
    )
    parser.add_argument(
        '--pivot_len_short', type=int, default=10, help='Pivot confirmation bars for short MA (default: 10)'
    )
    parser.add_argument(
        '--prom_thresh_long', type=float, default=0.005, help='Prominence threshold for long MA pivots (default: 0.005)'
    )
    parser.add_argument(
        '--prom_thresh_short', type=float, default=0.03, help='Prominence threshold for short MA pivots (default: 0.03)'
    )
    parser.add_argument('--peak_level', type=float, default=0.70, help='Peak exit level threshold (default: 0.70)')
    parser.add_argument(
        '--trough_level_long', type=float, default=0.40, help='Long MA trough entry level (default: 0.40)'
    )
    parser.add_argument('--trough_level_short', type=float, default=0.20, help='Short MA trough level (default: 0.20)')
    parser.add_argument(
        '--pyramiding',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Allow multiple entries (pyramiding). Default: off (single position, 100%% equity)',
    )

    # Enhanced TV mode options
    parser.add_argument(
        '--two_stage_exit', action='store_true', help='Enable two-stage exit (50%% profit + trend break)'
    )
    parser.add_argument(
        '--stage2_exit_mode',
        type=str,
        default='trend_break',
        help='Stage 2 exit mode: trend_break or ma_cross (default: trend_break)',
    )
    parser.add_argument('--use_volatility_stop', action='store_true', help='Use volatility-based stop instead of fixed')
    parser.add_argument('--vol_atr_period', type=int, default=14, help='Volatility calculation period (default: 14)')
    parser.add_argument(
        '--vol_atr_multiplier', type=float, default=2.5, help='Volatility stop multiplier (default: 2.5)'
    )
    parser.add_argument(
        '--vol_trailing_mode',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Volatility stop trails highest price (use --no-vol_trailing_mode to disable)',
    )
    parser.add_argument(
        '--bullish_regime_suppression', action='store_true', help='Suppress peak exits in bullish regime'
    )
    parser.add_argument(
        '--bullish_breadth_threshold',
        type=float,
        default=0.55,
        help='Breadth threshold for bullish regime (default: 0.55)',
    )

    # Chart-mode option
    parser.add_argument(
        '--chart_mode',
        action='store_true',
        help='Use chart-style peak/trough detection (find_peaks with distance=50 for long MA, '
        'no level filters). Walk-forward: signal dates may differ from chart peak/trough positions',
    )

    # Weekly trailing stop options
    parser.add_argument(
        '--enable_weekly_trailing',
        action='store_true',
        help='Enable weekly trailing stop exit (auto-disables tv_mode)',
    )
    parser.add_argument(
        '--weekly_trailing_type',
        type=str,
        default='weekly_ema',
        choices=['weekly_ema', 'weekly_nweek_low'],
        help='Weekly trailing type (default: weekly_ema)',
    )
    parser.add_argument('--weekly_ema_period', type=int, default=10, help='Weekly EMA period (default: 10)')
    parser.add_argument('--weekly_nweek_low_period', type=int, default=4, help='N-week low period (default: 4)')
    parser.add_argument(
        '--weekly_transition_weeks',
        type=int,
        default=3,
        help='Transition weeks before weekly trailing activates (default: 3)',
    )

    return parser
