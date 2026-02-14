import os
import sys
import time
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # Set backend to not display plots

# Add parent directory to path to import backtest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backtest import Backtest


def run_multi_etf_backtest(
    etfs,
    start_date=None,
    end_date=None,
    short_ma=20,
    long_ma=200,
    initial_capital=50000,
    slippage=0.001,
    commission=0.001,
    use_saved_data=True,
    debug=False,
    threshold=0.5,
    ma_type='ema',
    stop_loss_pct=0.10,
    no_show_plot=True,
):
    """
    Run backtest for multiple ETFs and output a summary table in MD format

    Parameters:
    -----------
    etfs : list
        List of ETF symbols to backtest
    start_date : str, optional
        Backtest start date (YYYY-MM-DD format)
    end_date : str, optional
        Backtest end date (YYYY-MM-DD format)
    short_ma : int, optional
        Short-term moving average period
    long_ma : int, optional
        Long-term moving average period
    initial_capital : float, optional
        Initial investment amount
    slippage : float, optional
        Slippage rate
    commission : float, optional
        Commission rate
    use_saved_data : bool, optional
        Whether to use saved data
    debug : bool, optional
        Whether to enable debug mode
    threshold : float, optional
        Threshold for bottom detection
    ma_type : str, optional
        Type of moving average ('ema' or 'sma')
    stop_loss_pct : float, optional
        Stop loss percentage
    no_show_plot : bool, optional
        Whether to not show plots (default: True)
    """
    # If dates are not specified, default to 10 years ago to present
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')

    # List to store results
    results = []

    # Run backtest for each ETF
    for symbol in etfs:
        print(f'\n{"=" * 50}')
        print(f'Running backtest for {symbol}...')
        print(f'{"=" * 50}')

        try:
            # Execute backtest
            backtest = Backtest(
                start_date=start_date,
                end_date=end_date,
                short_ma=short_ma,
                long_ma=long_ma,
                initial_capital=initial_capital,
                slippage=slippage,
                commission=commission,
                use_saved_data=use_saved_data,
                debug=debug,
                threshold=threshold,
                ma_type=ma_type,
                symbol=symbol,
                stop_loss_pct=stop_loss_pct,
                no_show_plot=no_show_plot,
            )

            # Run backtest
            backtest.run()

            # Visualize results (pass no_show_plot parameter)
            backtest.visualize_results(show_plot=not no_show_plot)

            # Save trade log (Phase 1)
            if backtest.trade_log:
                backtest.save_trade_log()
                print(f'Trade log saved for {symbol}: {len(backtest.trade_log)} trades')

            # Get results
            result = {
                'Symbol': symbol,
                'Total Return': backtest.total_return,
                'Annual Return (CAGR)': backtest.cagr,
                'Sharpe Ratio': backtest.sharpe_ratio,
                'Maximum Drawdown': backtest.max_drawdown,
                'Win Rate': backtest.win_rate,
                'Profit-Loss Ratio': backtest.profit_loss_ratio,
                'Profit Factor': backtest.profit_factor,
                'Calmar Ratio': backtest.calmar_ratio,
                'Expected Value': backtest.expected_value,
                'Avg. PnL per Trade': backtest.avg_pnl_per_trade,
                'Pareto Ratio': backtest.pareto_ratio,
            }

            results.append(result)

            # Wait briefly to limit API requests
            time.sleep(2)

        except Exception as e:
            print(f'Error running backtest for {symbol}: {e}')
            # Add empty result and continue even if error occurs
            results.append(
                {
                    'Symbol': symbol,
                    'Total Return': np.nan,
                    'Annual Return (CAGR)': np.nan,
                    'Sharpe Ratio': np.nan,
                    'Maximum Drawdown': np.nan,
                    'Win Rate': np.nan,
                    'Profit-Loss Ratio': np.nan,
                    'Profit Factor': np.nan,
                    'Calmar Ratio': np.nan,
                    'Expected Value': np.nan,
                    'Avg. PnL per Trade': np.nan,
                    'Pareto Ratio': np.nan,
                }
            )
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    print('\n' + '=' * 100)
    print('BACKTEST RESULTS SUMMARY')
    print('=' * 100)

    # Format and display results
    formatted_results = results_df.copy()

    # Convert to percentage format
    formatted_results['Total Return'] = formatted_results['Total Return'].map('{:.2%}'.format)
    formatted_results['Annual Return (CAGR)'] = formatted_results['Annual Return (CAGR)'].map('{:.2%}'.format)
    formatted_results['Maximum Drawdown'] = formatted_results['Maximum Drawdown'].map('{:.2%}'.format)
    formatted_results['Win Rate'] = formatted_results['Win Rate'].map('{:.2%}'.format)

    # Round to 2 decimal places
    formatted_results['Sharpe Ratio'] = formatted_results['Sharpe Ratio'].map('{:.2f}'.format)
    formatted_results['Profit-Loss Ratio'] = formatted_results['Profit-Loss Ratio'].map('{:.2f}'.format)
    formatted_results['Profit Factor'] = formatted_results['Profit Factor'].map('{:.2f}'.format)
    formatted_results['Calmar Ratio'] = formatted_results['Calmar Ratio'].map('{:.2f}'.format)
    formatted_results['Expected Value'] = formatted_results['Expected Value'].map('${:.2f}'.format)
    formatted_results['Avg. PnL per Trade'] = formatted_results['Avg. PnL per Trade'].map('${:.2f}'.format)
    formatted_results['Pareto Ratio'] = formatted_results['Pareto Ratio'].map('{:.2f}'.format)

    # Generate MD format table
    md_table = formatted_results.to_markdown(index=False)

    # Display results
    print(md_table)

    # Save results to file
    with open('reports/backtest_results_summary.md', 'w') as f:
        f.write('# ETF Backtest Results Summary\n\n')
        f.write(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'Period: {start_date} to {end_date}\n\n')
        f.write('Parameters:\n')
        f.write(f'- Short MA: {short_ma}\n')
        f.write(f'- Long MA: {long_ma}\n')
        f.write(f'- Initial Capital: ${initial_capital:,.2f}\n')
        f.write(f'- Slippage: {slippage:.3f}\n')
        f.write(f'- Commission: {commission:.3f}\n')
        f.write(f'- MA Type: {ma_type.upper()}\n')
        f.write(f'- Stop Loss: {stop_loss_pct:.1%}\n\n')
        f.write(md_table)

    print('\nResults saved to reports/backtest_results_summary.md')

    # Also save results to CSV file
    results_df.to_csv('reports/backtest_results_summary.csv', index=False)
    print('Results also saved to reports/backtest_results_summary.csv')

    return results_df


if __name__ == '__main__':
    # List of ETFs to backtest (organized by category)
    etfs = [
        # ETFs for US stock market diversification
        'SPY',  # SPDR S&P 500 ETF
        'VOO',  # Vanguard S&P 500 ETF (S&P500 index)
        'VTI',  # Vanguard Total Stock Market ETF (Total US stock market)
        'QQQ',  # Invesco QQQ Trust
        'VUG',  # Vanguard Growth ETF (Growth stocks)
        'VTV',  # Vanguard Value ETF (Value stocks)
        'VB',  # Vanguard Small-Cap ETF (Small-cap stocks)
        'VEA',  # Vanguard Developed Markets ETF (Developed markets)
        'VWO',  # Vanguard Emerging Markets ETF (Emerging markets)
        # Sector-specific ETFs
        'XLF',  # Financial Select Sector SPDR Fund (Financial sector)
        'XLE',  # Energy Select Sector SPDR Fund (Energy sector)
        'XLK',  # Technology Select Sector SPDR Fund (Technology sector)
        'XLV',  # Health Care Select Sector SPDR Fund (Healthcare sector)
        'XLI',  # Industrial Select Sector SPDR Fund (Industrial sector)
        'VGT',  # Vanguard Information Technology ETF
        # Leveraged/Bull ETFs
        'SSO',  # ProShares Ultra S&P 500
        'TQQQ',  # ProShares UltraPro QQQ
        'QLD',  # ProShares Ultra QQQ
        'SPXL',  # Direxion Daily S&P 500 Bull 3X Shares
        'SOXL',  # Direxion Daily Semiconductor Bull 3X Shares
        # Small/Mid-cap ETFs
        'TNA',  # Direxion Daily Small Cap Bull 3X Shares
        'IWR',  # iShares Russell Mid-Cap ETF
        # Growth ETFs
        'SCHG',  # Schwab U.S. Large-Cap Growth ETF
        'IWF',  # iShares Russell 1000 Growth ETF
        # Factor ETFs
        'MTUM',  # iShares MSCI USA Momentum Factor ETF
        # Dividend-focused ETFs
        'VYM',  # Vanguard High Dividend Yield ETF (High dividend stocks)
        'SCHD',  # Schwab U.S. Dividend Equity ETF (Dividend stocks)
        'NOBL',  # ProShares S&P 500 Dividend Aristocrats ETF (Dividend aristocrats)
    ]

    # Run backtest
    run_multi_etf_backtest(
        etfs=etfs,
        start_date='2001-01-01',
        end_date='2025-09-30',
        short_ma=8,
        long_ma=200,
        initial_capital=50000,
        slippage=0.001,
        commission=0.001,
        use_saved_data=True,
        debug=False,
        threshold=0.5,
        ma_type='ema',
        stop_loss_pct=0.08,
        no_show_plot=True,
    )
