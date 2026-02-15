from __future__ import annotations

from typing import Iterable

import pandas as pd

from .weights import ASSETS, compute_target_weights

REQUIRED_COLUMNS = ('regime', 'r_trend', 'x_trend', *ASSETS)


def _validate_columns(columns: Iterable[str]) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        joined = ', '.join(missing)
        raise ValueError(f'Missing required columns: {joined}')


def _turnover(current: dict[str, float], target: dict[str, float]) -> float:
    return 0.5 * sum(abs(target[a] - current[a]) for a in ASSETS)


def _month_end_mask(dates: pd.Series) -> pd.Series:
    """Return a boolean mask that is True on the last trading day of each month."""
    dates = pd.to_datetime(dates)
    month_key = dates.dt.to_period('M')
    last_idx = dates.groupby(month_key).transform('idxmax')
    return pd.Series(dates.index == last_idx, index=dates.index)


def run_strategy_backtest(
    signal_frame: pd.DataFrame,
    rebalance_threshold: float = 0.05,
    transaction_cost_bps: float = 5.0,
    rebalance_freq: str = 'month_end',
) -> pd.DataFrame:
    if rebalance_threshold < 0:
        raise ValueError('rebalance_threshold must be >= 0')
    if transaction_cost_bps < 0:
        raise ValueError('transaction_cost_bps must be >= 0')
    if rebalance_freq not in ('month_end', 'daily'):
        raise ValueError(f"rebalance_freq must be 'month_end' or 'daily', got '{rebalance_freq}'")

    _validate_columns(signal_frame.columns)

    frame = signal_frame.copy()
    if 'Date' in frame.columns:
        frame['Date'] = pd.to_datetime(frame['Date'])
        frame = frame.sort_values('Date').reset_index(drop=True)

    returns = frame[list(ASSETS)].pct_change().fillna(0.0)

    if rebalance_freq == 'month_end' and 'Date' in frame.columns:
        is_rebalance_date = _month_end_mask(frame['Date'])
    else:
        is_rebalance_date = pd.Series(True, index=frame.index)

    current_weights = compute_target_weights(
        regime=str(frame.loc[0, 'regime']),
        r_trend=int(frame.loc[0, 'r_trend']),
        x_trend=int(frame.loc[0, 'x_trend']),
    )

    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for i in range(len(frame)):
        date_value = frame.loc[i, 'Date'] if 'Date' in frame.columns else i
        portfolio_return = 0.0
        turnover = 0.0
        cost = 0.0
        rebalanced = 0

        if i > 0:
            portfolio_return = float(sum(current_weights[a] * returns.loc[i, a] for a in ASSETS))
            equity *= 1.0 + portfolio_return
            peak = max(peak, equity)
            drawdown = equity / peak - 1.0

            if is_rebalance_date.iloc[i]:
                target_weights = compute_target_weights(
                    regime=str(frame.loc[i, 'regime']),
                    r_trend=int(frame.loc[i, 'r_trend']),
                    x_trend=int(frame.loc[i, 'x_trend']),
                    portfolio_drawdown=drawdown,
                    xle_monthly_return=float(returns.loc[i, 'XLE']),
                )

                max_abs_diff = max(abs(target_weights[a] - current_weights[a]) for a in ASSETS)
                if max_abs_diff >= rebalance_threshold:
                    turnover = _turnover(current_weights, target_weights)
                    cost = turnover * (transaction_cost_bps / 10000.0)
                    equity *= 1.0 - cost
                    current_weights = target_weights
                    rebalanced = 1
                    peak = max(peak, equity)
                    drawdown = equity / peak - 1.0

        rows.append(
            {
                'Date': date_value,
                'portfolio_return': portfolio_return,
                'turnover': turnover,
                'transaction_cost': cost,
                'rebalanced': rebalanced,
                'equity': equity,
                'drawdown': drawdown,
                'weight_SPY': current_weights['SPY'],
                'weight_RSP': current_weights['RSP'],
                'weight_XLE': current_weights['XLE'],
                'weight_SGOV': current_weights['SGOV'],
            }
        )

    result = pd.DataFrame(rows)
    result['post_cost_return'] = result['equity'].pct_change().fillna(0.0)
    return result
