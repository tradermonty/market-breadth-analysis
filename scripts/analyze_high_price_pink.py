"""Offline high-price pink-onset cohort study using frozen research inputs."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from scripts.analyze_clustered_pink import merge_intervals, pink_intervals


def chattering_starts(events, dates, max_gap=5, max_span=40, min_segments=3):
    positions = {str(value)[:10]: i for i, value in enumerate(dates)}
    intervals = [(e['start_index'], positions[e['end']]) for e in events]
    groups = [
        group
        for group in merge_intervals(intervals, max_gap)
        if len(group['members']) >= min_segments and group['end'] - group['start'] <= max_span
    ]
    excluded = {start for group in groups for start, _ in group['members']}
    return excluded, groups


def study(directory, output, exclude_chattering=False):
    result = json.loads((directory / 'downtrend_forecast_statistics.json').read_text())
    frame = pd.read_csv(directory / 'downtrend_forecast_input.csv')
    price = frame['S&P500_Price']
    dd = 100 * (price / price.rolling(63, min_periods=63).max() - 1)
    current_start = len(frame) - result['current']['observed_sessions']
    original_events = [
        {'start_index': start, 'start': str(frame['Date'].iloc[start]), 'end': str(frame['Date'].iloc[end])}
        for start, end in pink_intervals(frame['Bearish_Signal'].astype(str).str.lower().eq('true'))
        if start >= 200 and end < current_start
    ]
    excluded, groups = chattering_starts(original_events, frame['Date']) if exclude_chattering else (set(), [])
    events = []
    for event in result['events']:
        if event['start_index'] in excluded:
            continue
        row = dict(event['episode_outcome'])
        onset, anchor = event['start_index'], event['anchor_index']
        row.update(
            start=event['start'],
            onset_high_gap_pct=float(dd.iloc[onset]),
            same_age_eligible=event['same_age_eligible'],
            anchor_high_gap_pct=float(dd.iloc[anchor]),
        )
        row['forward_60_return_pct'] = (
            float(100 * (price.iloc[anchor + 60] / price.iloc[anchor] - 1))
            if event['same_age_eligible'] and anchor + 60 < current_start
            else None
        )
        events.append(row)

    def summary(rows):
        n = len(rows)
        forward = [e['forward_60_return_pct'] for e in rows if e['forward_60_return_pct'] is not None]
        return {
            'n': n,
            'drop_counts': {str(t): sum(e['bottom_return_pct'] < -t for e in rows) for t in (5, 10, 20)},
            'metrics': {
                key: {
                    'mean': float(pd.Series([e[key] for e in rows]).mean()),
                    'median': float(pd.Series([e[key] for e in rows]).median()),
                }
                for key in (
                    'bottom_return_pct',
                    'bottom_sessions',
                    'bottom_to_end_sessions',
                    'end_sessions',
                    'end_return_pct',
                )
            }
            if rows
            else {},
            'forward_60_n': len(forward),
            'forward_60_negative': sum(v < 0 for v in forward),
        }

    selected = [e for e in events if e['onset_high_gap_pct'] >= -2]
    same_age = [e for e in selected if e['same_age_eligible']]
    still_high = [e for e in same_age if e['anchor_high_gap_pct'] >= -2]
    cohorts = {
        'all_high_onsets': summary(selected),
        'same_age': summary(same_age),
        'same_age_still_high': summary(still_high),
    }
    sensitivity = {
        str(t): summary([e for e in events if e['same_age_eligible'] and e['onset_high_gap_pct'] >= -t])
        for t in (1, 2, 5)
    }
    data = {
        'as_of': result['current']['as_of'],
        'current_start': result['current']['start'],
        'current_onset_high_gap_pct': float(dd.iloc[current_start]),
        'current_high_gap_pct': float(dd.iloc[-1]),
        'cohorts': cohorts,
        'sensitivity': sensitivity,
        'events': selected,
        'chattering_filter': {
            'enabled': exclude_chattering,
            'max_nonpink_gap': 5,
            'max_span_sessions': 40,
            'min_pink_segments': 3,
            'excluded_starts': [e['start'] for e in original_events if e['start_index'] in excluded],
            'excluded_high_onsets': [
                e['start'] for e in original_events if e['start_index'] in excluded and dd.iloc[e['start_index']] >= -2
            ],
            'groups': groups,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(data, indent=2))
    pd.DataFrame(selected).to_csv(output.with_suffix('.csv'), index=False)
    rows = ''.join(
        f'<tr><td>{e["start"]}</td><td>{e["onset_high_gap_pct"]:.2f}%</td>'
        f'<td>{"Yes" if e["same_age_eligible"] else "No"}</td>'
        f'<td>{e["bottom_date"]}</td><td>{e["bottom_return_pct"]:.1f}%</td>'
        f'<td>{e["bottom_sessions"]}</td><td>{e["end_sessions"]}</td>'
        f'<td>{e["end_return_pct"]:+.1f}%</td></tr>'
        for e in selected
    )
    cohort_rows = ''
    for label, key in [
        ('All high-price onsets', 'all_high_onsets'),
        ('Still pink on session 11', 'same_age'),
        ('Still pink AND within 2% of high on session 11', 'same_age_still_high'),
    ]:
        s = cohorts[key]
        counts = ''.join(
            f'<td>{s["drop_counts"][str(t)]}/{s["n"]} ({100 * s["drop_counts"][str(t)] / s["n"]:.1f}%)</td>'
            for t in (5, 10, 20)
        )
        cohort_rows += f'<tr><td>{label}</td><td>{s["n"]}</td>{counts}</tr>'
    labels = {
        'bottom_return_pct': 'Onset to price low: return (%)',
        'bottom_sessions': 'Onset to price low: trading sessions',
        'bottom_to_end_sessions': 'Price low to first non-pink day: trading sessions',
        'end_sessions': 'Onset to first non-pink day: trading sessions',
        'end_return_pct': 'Onset to first non-pink day: return (%)',
    }
    metric_rows = ''.join(
        f'<tr><td>{html.escape(labels[k])}</td><td>{v["mean"]:.2f}</td><td>{v["median"]:.2f}</td></tr>'
        for k, v in cohorts['same_age']['metrics'].items()
    )
    sensitivity_rows = ''.join(
        f'<tr><td>{t}%</td><td>{s["n"]}</td><td>{s["drop_counts"]["10"]}/{s["n"]}</td></tr>'
        for t, s in sensitivity.items()
    )
    filter_note = ''
    if exclude_chattering:
        removed = ', '.join(data['chattering_filter']['excluded_starts']) or 'None'
        removed_high = ', '.join(data['chattering_filter']['excluded_high_onsets']) or 'None'
        filter_note = f"""<h2>Rapid-switch exclusion only</h2>
<p>Exclude all segments in a completed chain of at least three pink episodes, separated by at most five non-pink sessions, whose total first-onset-to-final-exit span is at most 40 trading sessions. Retained episodes keep their original dates, lows and returns. No episodes are merged in this report.</p>
<p>Excluded onset dates: {removed}. Among near-high onsets, excluded: {removed_high}.</p>
<p>For this snapshot the near-high cohort falls from 15 to 14 episodes. The removed near-high episode lasted only one session, so the nine session-11 cases and four session-11 near-high cases are unchanged.</p>
<p class="note">This is a retrospective noise-filter sensitivity study. Whether a full rapid-switch chain forms is known only afterward; the exclusion is not a signal available on the initial onset or necessarily on session 11. It does not establish a more accurate live forecast.</p>
<p>2018-10-05 remains in the onset and session-11 cohorts. It falls outside the narrower session-11 near-high cohort because its price was already 5.48% below its trailing high.</p>"""
    high252_count = sum(
        e['start_index'] not in excluded
        for e in result['events']
        if 100
        * (price.iloc[e['start_index']] / price.iloc[max(0, e['start_index'] - 251) : e['start_index'] + 1].max() - 1)
        >= -2
    )
    s = cohorts['same_age']
    output.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Pink Onsets Near Price Highs</title><style>
body{{background:#10131c;color:#e1e4ec;font:16px/1.65 Arial,sans-serif;max-width:1150px;margin:45px auto;padding:0 24px}}
h1,h2{{color:#f0b3ce}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:12px;border-bottom:1px solid #394254;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{background:#1d2535;padding:18px;border-radius:10px}}a{{color:#b9d8ff}}
</style><h1>Pink Onsets Near Price Highs{' — Rapid Switching Excluded' if exclude_chattering else ''}</h1>
<p><a href="index.html">Market breadth dashboard</a> · <a href="downtrend_forecast.html">Long-history scenarios</a></p>
<p>Frozen data through {data['as_of']}. Current onset: {data['current_start']}; current episode is on session 11.</p>
<p class="note">Exploratory study using the current 504-stock survivor cohort with the project's fixed denominator. Missing historical prices and membership limitations affect early episodes. These are historical frequencies, not calibrated forecasts. The ongoing episode is excluded.</p>
{filter_note}
<h2>Definition and current position</h2><p>Near a high means the onset close is no more than 2% below the highest adjusted SPY close in the trailing 63 trading sessions, including onset. This 63-session lookback defines the prior high; outcomes follow the entire pink episode, without a 63-session cutoff.</p>
<p>The current onset was {data['current_onset_high_gap_pct']:.2f}% below this high. The latest close is {data['current_high_gap_pct']:.2f}% below its trailing high. A trailing 252-session high selects {high252_count} retained completed onset episodes at the 2% threshold in this dataset.</p>
<h2>Full-episode decline frequencies</h2><table><tr><th>Cohort</th><th>n</th><th>More than 5%</th><th>More than 10%</th><th>More than 20%</th></tr>{cohort_rows}</table>
<p>Declines compare the first minimum close during pink with the onset close. Thresholds are strict. Counts overlap. Conditioning on session 11 matches today's observed pink duration; the additional high-price condition is evaluated at that same historical anchor, without using future prices.</p>
<h2>Full-episode outcomes: {s['n']} same-age cases</h2><table><tr><th>Metric (returns: %, times: sessions)</th><th>Mean</th><th>Median</th></tr>{metric_rows}</table>
<p>From the session-11 anchor, 60 sessions later the price was lower in {s['forward_60_negative']}/{s['forward_60_n']} cases ({100 * s['forward_60_negative'] / s['forward_60_n']:.1f}%). This uses the anchor price, whereas the full-episode decline table uses the onset price.</p>
<h2>Individual historical episodes</h2><table><tr><th>Onset</th><th>Gap to high</th><th>Pink at session 11</th><th>Price-low date</th><th>Low return</th><th>Sessions to low</th><th>Sessions to end</th><th>End return</th></tr>{rows}</table>
<p>The four cases still near a high on session 11 began 2005-08-19, 2014-04-25, 2014-09-12 and 2015-05-28. Only four observations remain: absence of a 20% decline here does not establish zero future risk.</p>
<h2>Threshold sensitivity: still-pink cohort</h2><table><tr><th>Onset gap allowed</th><th>n</th><th>More than 10% decline</th></tr>{sensitivity_rows}</table>
<p>Near-high filtering reduces sample size and may leave dependent episodes from the same market period. No entry rule or causal conclusion is validated by this subset.</p>
</html>
""")
    print(json.dumps({k: v for k, v in data.items() if k != 'events'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-chattering', action='store_true')
    args = parser.parse_args()
    study(args.directory, args.output, args.exclude_chattering)
