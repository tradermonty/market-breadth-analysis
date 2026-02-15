from __future__ import annotations

from typing import Dict

ASSETS = ('SPY', 'RSP', 'XLE', 'SGOV')

BASE_WEIGHTS: dict[str, dict[str, float]] = {
    'Recovery': {'SPY': 0.25, 'RSP': 0.45, 'XLE': 0.10, 'SGOV': 0.20},
    'RiskOn': {'SPY': 0.45, 'RSP': 0.40, 'XLE': 0.10, 'SGOV': 0.05},
    'Overheat': {'SPY': 0.40, 'RSP': 0.30, 'XLE': 0.00, 'SGOV': 0.30},
    'Deterioration': {'SPY': 0.35, 'RSP': 0.35, 'XLE': 0.05, 'SGOV': 0.25},
}

RSP_MAX = 0.55
XLE_MAX = 0.15
DEFAULT_SGOV_MIN = 0.05
DEFENSIVE_SGOV_MIN = 0.20


def _transfer(weights: Dict[str, float], source: str, target: str, amount: float) -> None:
    if amount <= 0:
        return
    moved = min(amount, weights[source])
    weights[source] -= moved
    weights[target] += moved


def _apply_caps_and_floors(weights: Dict[str, float], regime: str) -> Dict[str, float]:
    if weights['RSP'] > RSP_MAX:
        excess = weights['RSP'] - RSP_MAX
        weights['RSP'] = RSP_MAX
        weights['SGOV'] += excess

    if weights['XLE'] > XLE_MAX:
        excess = weights['XLE'] - XLE_MAX
        weights['XLE'] = XLE_MAX
        weights['SGOV'] += excess

    sgov_min = DEFENSIVE_SGOV_MIN if regime in {'Overheat', 'Deterioration'} else DEFAULT_SGOV_MIN
    if weights['SGOV'] < sgov_min:
        shortfall = sgov_min - weights['SGOV']
        for source in ('SPY', 'RSP', 'XLE'):
            if shortfall <= 0:
                break
            before = weights['SGOV']
            _transfer(weights, source, 'SGOV', shortfall)
            shortfall -= weights['SGOV'] - before

    total = sum(weights.values())
    if total <= 0:
        raise ValueError('Total weight must be positive')

    return {asset: weights[asset] / total for asset in ASSETS}


def compute_target_weights(
    regime: str,
    r_trend: int,
    x_trend: int,
    portfolio_drawdown: float | None = None,
    xle_monthly_return: float | None = None,
) -> Dict[str, float]:
    if regime not in BASE_WEIGHTS:
        raise ValueError(f'Unknown regime: {regime}')

    weights = BASE_WEIGHTS[regime].copy()

    if not r_trend:
        _transfer(weights, 'RSP', 'SPY', 0.10)
        _transfer(weights, 'RSP', 'SGOV', 0.05)

    if not x_trend:
        _transfer(weights, 'XLE', 'SGOV', weights['XLE'])

    if regime == 'Recovery' and r_trend and x_trend:
        _transfer(weights, 'SPY', 'RSP', 0.03)
        _transfer(weights, 'SPY', 'XLE', 0.02)

    if portfolio_drawdown is not None and portfolio_drawdown <= -0.12:
        _transfer(weights, 'RSP', 'SGOV', 0.10)
        _transfer(weights, 'XLE', 'SGOV', 0.05)

    if xle_monthly_return is not None and xle_monthly_return <= -0.12:
        _transfer(weights, 'XLE', 'SGOV', weights['XLE'])

    return _apply_caps_and_floors(weights, regime)
