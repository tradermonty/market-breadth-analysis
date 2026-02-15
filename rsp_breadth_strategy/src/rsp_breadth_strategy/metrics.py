from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(backtest_df: pd.DataFrame) -> dict[str, float]:
    """Compute strategy performance metrics from post-cost equity curve.

    All metrics are derived from ``equity`` (which is already cost-adjusted)
    and ``post_cost_return``.
    """
    equity = backtest_df['equity']
    post_ret = backtest_df['post_cost_return']

    final_equity = float(equity.iloc[-1])

    # Trading days
    if 'Date' in backtest_df.columns:
        dates = pd.to_datetime(backtest_df['Date'])
        n_years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    else:
        n_years = len(backtest_df) / 252.0

    if n_years <= 0:
        n_years = len(backtest_df) / 252.0

    # CAGR
    if final_equity > 0 and n_years > 0:
        cagr = final_equity ** (1.0 / n_years) - 1.0
    else:
        cagr = 0.0

    # Annualized volatility
    annualized_vol = float(post_ret.std() * np.sqrt(252))

    # Sharpe — geometric definition: CAGR / annualized_vol (rf=0)
    sharpe = cagr / annualized_vol if annualized_vol > 0 else 0.0

    # Sharpe — arithmetic definition: mean(daily_ret)*252 / annualized_vol
    ann_mean = float(post_ret.mean()) * 252.0
    sharpe_arithmetic = ann_mean / annualized_vol if annualized_vol > 0 else 0.0

    # Max drawdown
    max_drawdown = float(backtest_df['drawdown'].min())

    # Calmar
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Rebalance count and total cost
    rebalance_count = int(backtest_df['rebalanced'].sum())
    total_cost = float(backtest_df['transaction_cost'].sum())

    return {
        'cagr': cagr,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'sharpe_arithmetic': sharpe_arithmetic,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'final_equity': final_equity,
        'rebalance_count': rebalance_count,
        'total_cost': total_cost,
    }
