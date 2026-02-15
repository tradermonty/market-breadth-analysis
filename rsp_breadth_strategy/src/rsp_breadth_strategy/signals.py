from __future__ import annotations

from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = (
    'SPY',
    'RSP',
    'XLE',
    'Breadth_Index_Raw',
    'Breadth_Index_8MA',
    'Breadth_Index_200MA',
)


def classify_regime(raw: float, ma8: float, ma200: float) -> str:
    if pd.isna(raw) or pd.isna(ma8) or pd.isna(ma200):
        return 'Deterioration'

    if raw < ma200:
        return 'Recovery'
    if raw >= ma200 and ma8 >= ma200 and raw <= 0.75:
        return 'RiskOn'
    if raw > 0.75 and raw < ma8:
        return 'Overheat'
    return 'Deterioration'


def classify_regime_series(
    raw: pd.Series,
    ma8: pd.Series,
    ma200: pd.Series,
    bearish_signal: pd.Series | None = None,
    breadth_trend: pd.Series | None = None,
    overheat_entry: float = 0.75,
    overheat_exit: float = 0.70,
) -> pd.Series:
    """Vectorized regime classification with hysteresis and overrides.

    Priority:
    1. Bearish_Signal override: True -> Deterioration
    2. Overheat hysteresis: entry at overheat_entry, exit at overheat_exit
    3. Breadth_200MA_Trend: trend=-1 forces Recovery -> Deterioration
    4. Normal classification (Recovery / RiskOn / Deterioration)
    """
    n = len(raw)
    result = ['Deterioration'] * n
    in_overheat = False

    for i in range(n):
        r, m8, m200 = raw.iloc[i], ma8.iloc[i], ma200.iloc[i]

        # Base classification (same as classify_regime)
        if pd.isna(r) or pd.isna(m8) or pd.isna(m200):
            regime = 'Deterioration'
        elif r < m200:
            regime = 'Recovery'
        elif r >= m200 and m8 >= m200 and r <= overheat_entry:
            regime = 'RiskOn'
        elif r > overheat_entry and r < m8:
            regime = 'Overheat'
        else:
            regime = 'Deterioration'

        # Overheat hysteresis
        if not in_overheat and r > overheat_entry and r < m8:
            in_overheat = True
            regime = 'Overheat'
        elif in_overheat:
            if r <= overheat_exit:
                in_overheat = False
                # Re-classify without Overheat
                if pd.isna(r) or pd.isna(m8) or pd.isna(m200):
                    regime = 'Deterioration'
                elif r < m200:
                    regime = 'Recovery'
                elif r >= m200 and m8 >= m200:
                    regime = 'RiskOn'
                else:
                    regime = 'Deterioration'
            else:
                regime = 'Overheat'

        # Breadth_200MA_Trend override: trend=-1 forces Recovery -> Deterioration
        if breadth_trend is not None and regime == 'Recovery':
            if breadth_trend.iloc[i] == -1:
                regime = 'Deterioration'

        # Bearish_Signal override (highest priority)
        if bearish_signal is not None and bearish_signal.iloc[i]:
            regime = 'Deterioration'

        result[i] = regime

    return pd.Series(result, index=raw.index)


def _validate_columns(columns: Iterable[str]) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        joined = ', '.join(missing)
        raise ValueError(f'Missing required columns: {joined}')


def build_signal_frame(
    frame: pd.DataFrame,
    ratio_ma: int = 10,
    regime_mode: str = 'series',
    overheat_entry: float = 0.75,
    overheat_exit: float = 0.70,
) -> pd.DataFrame:
    """Build signal frame with regime classification.

    Parameters
    ----------
    regime_mode : str
        ``'series'`` uses ``classify_regime_series()`` with hysteresis (default).
        ``'pointwise'`` uses ``classify_regime()`` row-by-row (legacy behaviour).
    overheat_entry, overheat_exit : float
        Passed through to ``classify_regime_series()`` when *regime_mode='series'*.
    """
    if ratio_ma <= 1:
        raise ValueError('ratio_ma must be > 1')
    if regime_mode not in ('series', 'pointwise'):
        raise ValueError(f"regime_mode must be 'series' or 'pointwise', got '{regime_mode}'")

    _validate_columns(frame.columns)

    out = frame.copy()
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date'])
        out = out.sort_values('Date').reset_index(drop=True)

    if (out['SPY'] == 0).any():
        raise ValueError('SPY price contains zero; ratio cannot be computed')

    out['r_ratio'] = out['RSP'] / out['SPY']
    out['x_ratio'] = out['XLE'] / out['SPY']

    out['r_ratio_sma'] = out['r_ratio'].rolling(window=ratio_ma, min_periods=ratio_ma).mean()
    out['x_ratio_sma'] = out['x_ratio'].rolling(window=ratio_ma, min_periods=ratio_ma).mean()

    out['r_trend'] = (out['r_ratio'] > out['r_ratio_sma']).astype(int)
    out['x_trend'] = (out['x_ratio'] > out['x_ratio_sma']).astype(int)

    if regime_mode == 'pointwise':
        out['regime'] = out.apply(
            lambda row: classify_regime(
                row['Breadth_Index_Raw'],
                row['Breadth_Index_8MA'],
                row['Breadth_Index_200MA'],
            ),
            axis=1,
        )
    else:
        bearish = out.get('Bearish_Signal')
        breadth_tr = out.get('Breadth_200MA_Trend')

        out['regime'] = classify_regime_series(
            raw=out['Breadth_Index_Raw'],
            ma8=out['Breadth_Index_8MA'],
            ma200=out['Breadth_Index_200MA'],
            bearish_signal=bearish,
            breadth_trend=breadth_tr,
            overheat_entry=overheat_entry,
            overheat_exit=overheat_exit,
        )

    return out
