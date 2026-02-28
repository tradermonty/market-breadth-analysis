"""Pivot detection functions for TradingView-compatible signal detection.

Extracted from backtest.py — pure functions with no class dependency.
"""

import numpy as np


def detect_pivot_high(series, pivot_len, prom_thresh, level_thresh):
    """Detect pivot highs equivalent to TradingView ta.pivothigh(source, left, right).

    A bar j is a pivot high if it is the maximum in [j-pivot_len, j+pivot_len].
    Confirmation date = j + pivot_len (the bar where the pivot can first be observed).

    Returns list of (confirm_date, pivot_date, pivot_value).
    """
    values = series.values
    dates = series.index
    n = len(values)
    results = []

    for j in range(pivot_len, n - pivot_len):
        window = values[j - pivot_len : j + pivot_len + 1]
        if values[j] == np.max(window):
            # Prominence check: peak - window min
            prominence = values[j] - np.min(window)
            if prominence >= prom_thresh and values[j] >= level_thresh:
                confirm_idx = j + pivot_len
                results.append((dates[confirm_idx], dates[j], values[j]))

    return results


def detect_pivot_low(series, pivot_len, prom_thresh):
    """Detect pivot lows equivalent to TradingView ta.pivotlow(source, left, right).

    A bar j is a pivot low if it is the minimum in [j-pivot_len, j+pivot_len].
    Confirmation date = j + pivot_len.
    Level check is done by the caller (differs for 200-EMA vs short EMA).

    Returns list of (confirm_date, pivot_date, pivot_value).
    """
    values = series.values
    dates = series.index
    n = len(values)
    results = []

    for j in range(pivot_len, n - pivot_len):
        window = values[j - pivot_len : j + pivot_len + 1]
        if values[j] == np.min(window):
            # Prominence check: window max - trough
            prominence = np.max(window) - values[j]
            if prominence >= prom_thresh:
                confirm_idx = j + pivot_len
                results.append((dates[confirm_idx], dates[j], values[j]))

    return results
