"""Performance metric calculations for backtest results.

Extracted from backtest.py — all functions are pure (no class dependency).
Each function takes explicit parameters instead of accessing self.*.
"""

import numpy as np


def calculate_max_drawdown(equity_df):
    """Calculate maximum drawdown from equity DataFrame."""
    equity = equity_df['equity']
    rolling_max = equity.expanding().max()
    drawdowns = equity / rolling_max - 1
    return drawdowns.min()


def calculate_win_rate(trade_pairs, debug=False):
    """Calculate win rate for each individual trade."""
    # Calculate profit/loss for each trade pair
    profitable_trades = 0
    total_trades = len(trade_pairs)

    for pair in trade_pairs:
        profit = pair['sell_proceeds'] - pair['buy_cost']
        if profit > 0:
            profitable_trades += 1

    # Calculate win rate
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    if debug:
        print('\nWin rate details:')
        print(f'Total trades: {total_trades}')
        print(f'Winning trades: {profitable_trades}')
        print(f'Losing trades: {total_trades - profitable_trades}')

        # Display detailed trade information
        print('\nTrade details:')
        for i, pair in enumerate(trade_pairs):
            profit = pair['sell_proceeds'] - pair['buy_cost']
            profit_pct = (profit / pair['buy_cost']) * 100
            print(f'Trade {i + 1}:')
            print(f'  Buy date: {pair["buy_date"].strftime("%Y-%m-%d")}')
            print(f'  Sell date: {pair["sell_date"].strftime("%Y-%m-%d")}')
            print(f'  Shares: {pair["shares"]}')
            print(f'  Buy price: ${pair["buy_price"]:.2f}')
            print(f'  Sell price: ${pair["sell_price"]:.2f}')
            print(f'  Profit: ${profit:.2f} ({profit_pct:.2f}%)')

    return win_rate


def calculate_profit_loss_ratio(trade_pairs, debug=False):
    """Calculate profit-loss ratio."""
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

    if debug:
        print('\nProfit-Loss ratio details:')
        print(f'Total trades: {len(trade_pairs)}')
        print(f'Profitable trades: {len(profits)}')
        print(f'Losing trades: {len(losses)}')
        print(f'Average profit: ${avg_profit:.2f}')
        print(f'Average loss: ${avg_loss:.2f}')

        # Display detailed trade information
        print('\nTrade details:')
        for i, pair in enumerate(trade_pairs):
            profit = pair['sell_proceeds'] - pair['buy_cost']
            profit_pct = (profit / pair['buy_cost']) * 100
            print(f'Trade {i + 1}:')
            print(f'  Buy date: {pair["buy_date"].strftime("%Y-%m-%d")}')
            print(f'  Sell date: {pair["sell_date"].strftime("%Y-%m-%d")}')
            print(f'  Shares: {pair["shares"]}')
            print(f'  Buy price: ${pair["buy_price"]:.2f}')
            print(f'  Sell price: ${pair["sell_price"]:.2f}')
            print(f'  Profit: ${profit:.2f} ({profit_pct:.2f}%)')

    return avg_profit / avg_loss if avg_loss > 0 else float('inf')


def calculate_profit_factor(trade_pairs):
    """Calculate profit factor."""
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


def calculate_calmar_ratio(total_return, max_drawdown, equity_df):
    """Calculate Calmar ratio."""
    # Calculate annual return
    days = (equity_df.index[-1] - equity_df.index[0]).days
    if days <= 0:
        return 0
    years = days / 365

    # Annual Return calculation (handles negative returns)
    if total_return <= -1:
        annual_return = -1.0
    else:
        annual_return = (1 + total_return) ** (1 / years) - 1

    # Get maximum drawdown
    max_dd = abs(max_drawdown)

    # Calculate Calmar ratio
    if max_dd == 0:
        return 0  # Return 0 if there is no drawdown
    else:
        return annual_return / max_dd


def calculate_expected_value(trade_pairs):
    """Calculate expected value per trade."""
    # Calculate profit/loss for each trade pair
    total_profit = 0
    total_trades = len(trade_pairs)

    for pair in trade_pairs:
        profit = pair['sell_proceeds'] - pair['buy_cost']
        total_profit += profit

    # Calculate expected value
    return total_profit / total_trades if total_trades > 0 else 0


def calculate_avg_pnl_per_trade(trade_pairs):
    """Calculate average PnL per trade."""
    # Calculate profit/loss for each trade pair
    total_pnl = 0
    total_trades = len(trade_pairs)

    for pair in trade_pairs:
        pnl = pair['sell_proceeds'] - pair['buy_cost']
        total_pnl += pnl

    # Calculate average PnL
    return total_pnl / total_trades if total_trades > 0 else 0


def calculate_pareto_ratio(trade_pairs):
    """Calculate Pareto ratio (80/20 rule)."""
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


def get_trade_pairs(trades, debug=False):
    """Get trade pairs from trade history using FIFO matching."""
    trade_pairs = []
    current_buy_trades = []

    # Copy trade history to operate on (don't modify original data)
    trades_copy = []
    for trade in trades:
        trade_copy = trade.copy()
        # Keep mutable remaining shares and immutable original shares for cost allocation.
        if 'shares' in trade_copy:
            trade_copy['shares'] = int(trade_copy['shares'])
        if trade_copy.get('action') == 'BUY' and 'shares' in trade_copy:
            trade_copy['original_shares'] = trade_copy['shares']
        trades_copy.append(trade_copy)

    # Debug information
    if debug:
        print('\nTrade pairs calculation:')
        print(f'Total trades: {len(trades_copy)}')
        print(f'Buy trades: {sum(1 for t in trades_copy if t["action"] == "BUY")}')
        print(f'Sell trades: {sum(1 for t in trades_copy if t["action"] == "SELL")}')

    for trade in trades_copy:
        if trade['action'] == 'BUY':
            if trade['shares'] > 0:
                current_buy_trades.append(trade)
                if debug:
                    print(f'Added buy trade: {trade["date"].strftime("%Y-%m-%d")}, Shares: {trade["shares"]}')
        elif trade['action'] == 'SELL':
            remaining_shares = trade['shares']

            if debug:
                print(f'Processing sell trade: {trade["date"].strftime("%Y-%m-%d")}, Shares: {remaining_shares}')
                print(f'Current buy trades: {len(current_buy_trades)}')

            while remaining_shares > 0 and current_buy_trades:
                buy_trade = current_buy_trades[0]
                if buy_trade['shares'] <= 0:
                    current_buy_trades.pop(0)
                    if debug:
                        print('Removed empty buy trade')
                    continue

                matched_shares = min(remaining_shares, buy_trade['shares'])
                original_shares = buy_trade.get('original_shares', buy_trade['shares'])

                trade_pairs.append(
                    {
                        'buy_date': buy_trade['date'],
                        'sell_date': trade['date'],
                        'shares': matched_shares,
                        'buy_price': buy_trade['price'],
                        'sell_price': trade['price'],
                        'buy_cost': buy_trade['total_cost'] * (matched_shares / original_shares),
                        'sell_proceeds': trade['total_proceeds'] * (matched_shares / trade['shares']),
                    }
                )

                if debug:
                    print(
                        f'Created trade pair: Buy: {buy_trade["date"].strftime("%Y-%m-%d")}, '
                        f'Sell: {trade["date"].strftime("%Y-%m-%d")}, Shares: {matched_shares}'
                    )

                remaining_shares -= matched_shares
                buy_trade['shares'] -= matched_shares

                if buy_trade['shares'] == 0:
                    current_buy_trades.pop(0)
                    if debug:
                        print('Removed fully matched buy trade')

    if debug:
        print(f'Total trade pairs created: {len(trade_pairs)}')

    return trade_pairs
