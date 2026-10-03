"""Build explicit survivor-cohort and reconstructed-membership research datasets."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def trend_with_hysteresis(series):
    trend = 0
    result = []
    for slope in series.diff():
        if trend <= 0 and slope > 0.001:
            trend = 1
        elif trend >= 0 and slope < -0.001:
            trend = -1
        result.append(trend)
    return result


def prepare(directory):
    manifest = json.loads((directory / 'acquisition_manifest.json').read_text())
    constituents = json.loads((directory / 'current_constituents.json').read_text())
    changes = json.loads((directory / 'historical_constituents.json').read_text())
    current = {x['symbol'] for x in constituents}
    spy_payload = json.loads((directory / 'spy_legacy.json').read_text())['historical']
    spy = pd.DataFrame(spy_payload).assign(date=lambda x: pd.to_datetime(x['date'])).set_index('date').sort_index()
    frames = {}
    for item in manifest:
        if not item.get('usable'):
            continue
        path = directory / 'prices' / (quote(item['symbol'], safe='') + '.csv.gz')
        frame = pd.read_csv(path, parse_dates=['date']).set_index('date')['adjClose'].sort_index()
        frames[item['symbol']] = frame
    calendar = pd.DatetimeIndex(
        sorted(set(spy.index) | {date for frame in frames.values() for date in frame.index if date < spy.index.min()})
    )
    prices = pd.DataFrame(frames, index=calendar)
    # Every constituent remains in the denominator, even when its history is missing.
    prices = prices.reindex(columns=sorted(set(json.loads((directory / 'universe.json').read_text()))))
    moving = prices.rolling(200, min_periods=200).mean()
    eligible = prices.notna() & moving.notna()
    above = (prices > moving) & eligible
    groups = collections.defaultdict(list)
    for item in changes:
        if item.get('date') and '1990-01-01' <= item['date'] <= '2026-10-02':
            groups[pd.Timestamp(item['date'])].append(item)
    dates = sorted(groups, reverse=True)
    cursor = 0
    state = set(current)
    symbols = list(prices.columns)
    membership = np.zeros(prices.shape, dtype=bool)
    for i in range(len(calendar) - 1, -1, -1):
        day = calendar[i]
        while cursor < len(dates) and dates[cursor] > day:
            records = groups[dates[cursor]]
            additions = {str(x.get('symbol') or '').strip() for x in records} - {''}
            removals = {str(x.get('removedTicker') or '').strip() for x in records} - {''}
            state.difference_update(additions)
            state.update(removals)
            cursor += 1
        membership[i] = [symbol in state for symbol in symbols]
    member = pd.DataFrame(membership, index=calendar, columns=prices.columns)
    fixed = pd.DataFrame(
        np.broadcast_to([symbol in current for symbol in symbols], prices.shape), index=calendar, columns=prices.columns
    )
    datasets = {}
    for name, active, eligible_denominator in [
        ('current_fixed', fixed, False),
        ('current_eligible', fixed, True),
        ('historical_reconstructed', member, True),
    ]:
        count = active.sum(axis=1)
        valid_count = (active & eligible).sum(axis=1)
        observed_count = (active & prices.notna()).sum(axis=1)
        denominator = valid_count if eligible_denominator else count
        raw = (above & active).sum(axis=1) / denominator.replace(0, np.nan)
        # Do not manufacture zero breadth before any stock has a complete MA window.
        raw = raw.dropna()
        long = raw.ewm(span=200, adjust=False).mean()
        short = raw.ewm(span=8, adjust=False).mean()
        trend = pd.Series(trend_with_hysteresis(long), index=raw.index)
        frame = pd.DataFrame(
            {
                'Date': raw.index,
                'S&P500_Price': spy.adjClose.reindex(raw.index).to_numpy(),
                'Breadth_Index_Raw': raw.to_numpy(),
                'Breadth_Index_200MA': long.to_numpy(),
                'Breadth_Index_8MA': short.to_numpy(),
                'Breadth_200MA_Trend': trend.to_numpy(),
                'Bearish_Signal': ((trend == -1) & (short < long)).to_numpy(),
                'Constituent_Count': count.reindex(raw.index).to_numpy(),
                'Eligible_Count': valid_count.reindex(raw.index).to_numpy(),
                'Observed_Count': observed_count.reindex(raw.index).to_numpy(),
                'Eligibility_Coverage': (valid_count / count).reindex(raw.index).to_numpy(),
            }
        )
        frame = frame.dropna(subset=['S&P500_Price']).reset_index(drop=True)
        frame['Is_Peak'] = False
        frame['Is_Trough'] = False
        frame['Is_Trough_8MA_Below_04'] = False
        peaks, _ = find_peaks(frame.Breadth_Index_200MA, distance=50, prominence=0.015)
        troughs, _ = find_peaks(-frame.Breadth_Index_200MA, distance=50, prominence=0.015)
        short_positions = np.flatnonzero(frame.Breadth_Index_8MA.to_numpy() < 0.4)
        short_pivots, _ = find_peaks(-frame.Breadth_Index_8MA.iloc[short_positions].to_numpy(), prominence=0.02)
        frame.loc[peaks, 'Is_Peak'] = True
        frame.loc[troughs, 'Is_Trough'] = True
        frame.loc[short_positions[short_pivots], 'Is_Trough_8MA_Below_04'] = True
        path = directory / (name + '.csv')
        frame.to_csv(path, index=False)
        by_year = frame.groupby(frame.Date.dt.year).agg(
            median_coverage=('Eligibility_Coverage', 'median'),
            minimum_coverage=('Eligibility_Coverage', 'min'),
            median_eligible=('Eligible_Count', 'median'),
            median_members=('Constituent_Count', 'median'),
        )
        datasets[name] = {
            'path': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'first_date': str(frame.Date.iloc[0].date()),
            'last_date': str(frame.Date.iloc[-1].date()),
            'rows': len(frame),
            'last_pink': bool(frame.Bearish_Signal.iloc[-1]),
            'annual_coverage': {str(y): record for y, record in by_year.to_dict('index').items()},
        }
    (directory / 'dataset_audit.json').write_text(json.dumps(datasets, indent=2))
    print(json.dumps(datasets, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.directory)
