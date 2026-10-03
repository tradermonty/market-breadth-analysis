"""Cluster short pink interruptions, then condition on onset and current-age prices."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def merge_intervals(intervals, max_gap):
    """Merge [start, end) pink intervals when non-pink observations <= max_gap."""
    if max_gap < 0:
        raise ValueError('max_gap must be nonnegative')
    merged = []
    for start, end in intervals:
        if end <= start or (merged and start < merged[-1]['end']):
            raise ValueError('Intervals must be sorted, disjoint and nonempty')
        if merged and start - merged[-1]['end'] <= max_gap:
            merged[-1]['end'] = end
            merged[-1]['members'].append([start, end])
        else:
            merged.append({'start': start, 'end': end, 'members': [[start, end]]})
    return merged


def pink_intervals(mask):
    intervals = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals


def summarize(rows, keys):
    n = len(rows)
    metrics = {}
    for key in keys:
        values = pd.Series([row[key] for row in rows], dtype=float)
        metrics[key] = {'mean': float(values.mean()), 'median': float(values.median())} if n else None
    return {
        'n': n,
        'metrics': metrics,
        'drop_counts': {str(t): sum(row['low_return_pct'] < -t for row in rows) for t in (5, 10, 20)},
    }


def analyze(frame, gap):
    dates = pd.to_datetime(frame['Date'])
    price = frame['S&P500_Price'].reset_index(drop=True)
    mask = frame['Bearish_Signal'].astype(str).str.lower().eq('true').tolist()
    if not mask[-1]:
        raise ValueError('Latest observation must be pink')
    clusters = merge_intervals(pink_intervals(mask), gap)
    current = clusters[-1]
    age = len(frame) - 1 - current['start']
    high63 = 100 * (price / price.rolling(63, min_periods=63).max() - 1)
    high252 = 100 * (price / price.rolling(252, min_periods=252).max() - 1)
    rows = []
    for cluster in clusters[:-1]:
        start, end = cluster['start'], cluster['end']
        # Drop initialization and require the interruption rule to have resolved
        # before current onset; never use an unresolved cluster's outcome.
        if start < 200 or end + gap >= current['start']:
            continue
        low = int(price.iloc[start:end].idxmin())
        anchor = start + age
        same_age = anchor < end and mask[anchor]
        row = {
            'start': str(dates.iloc[start].date()),
            'end': str(dates.iloc[end].date()),
            'start_index': start,
            'end_index': end,
            'anchor_index': anchor,
            'subepisodes': len(cluster['members']),
            'subepisode_starts': [str(dates.iloc[a].date()) for a, _ in cluster['members']],
            'onset_high_gap_pct': float(high63.iloc[start]),
            'onset_252_high_gap_pct': float(high252.iloc[start]),
            'same_age_pink': same_age,
            'anchor_high_gap_pct': float(high63.iloc[anchor]) if anchor < len(price) else None,
            'low_date': str(dates.iloc[low].date()),
            'low_sessions': low - start,
            'low_calendar_days': int((dates.iloc[low] - dates.iloc[start]).days),
            'low_return_pct': float(100 * (price.iloc[low] / price.iloc[start] - 1)),
            'low_to_end_sessions': end - low,
            'end_sessions': end - start,
            'end_return_pct': float(100 * (price.iloc[end] / price.iloc[start] - 1)),
        }
        if same_age:
            row['forward'] = {}
            for horizon in (21, 60, 126):
                if anchor + horizon >= current['start']:
                    continue
                path = price.iloc[anchor : anchor + horizon + 1]
                row['forward'][str(horizon)] = {
                    'return_pct': float(100 * (path.iloc[-1] / path.iloc[0] - 1)),
                    'worst_return_pct': float(100 * (path.min() / path.iloc[0] - 1)),
                }
        rows.append(row)
    high = [row for row in rows if row['onset_high_gap_pct'] >= -2]
    same_age = [row for row in high if row['same_age_pink']]
    still_high = [row for row in same_age if row['anchor_high_gap_pct'] >= -2]
    keys = ('low_return_pct', 'low_sessions', 'low_to_end_sessions', 'end_sessions', 'end_return_pct')
    cohorts = {}
    for name, selected in [
        ('all', rows),
        ('high_onset', high),
        ('high_onset_same_age', same_age),
        ('high_onset_same_age_still_high', still_high),
    ]:
        cohorts[name] = summarize(selected, keys)
        cohorts[name]['starts'] = [row['start'] for row in selected]
        forward = {}
        for horizon in (21, 60, 126):
            outcomes = [row['forward'][str(horizon)] for row in selected if str(horizon) in row.get('forward', {})]
            n = len(outcomes)
            forward[str(horizon)] = {
                'n': n,
                'negative_count': sum(row['return_pct'] < 0 for row in outcomes),
                'mean': sum(row['return_pct'] for row in outcomes) / n if n else None,
                'median': float(pd.Series([row['return_pct'] for row in outcomes]).median()) if n else None,
                'within_drop_counts': {
                    str(t): sum(row['worst_return_pct'] < -t for row in outcomes) for t in (5, 10, 20)
                },
            }
        cohorts[name]['forward'] = forward
    return {
        'gap': gap,
        'as_of': str(dates.iloc[-1].date()),
        'current_start': str(dates.iloc[current['start']].date()),
        'current_sessions': age + 1,
        'current_onset_gap_pct': float(high63.iloc[current['start']]),
        'current_gap_pct': float(high63.iloc[-1]),
        'cohorts': cohorts,
        'rows': rows,
        'threshold_sensitivity': {
            str(t): summarize([row for row in rows if row['onset_high_gap_pct'] >= -t and row['same_age_pink']], keys)
            for t in (1, 2, 5)
        },
    }


def build(directory, output):
    frame = pd.read_csv(directory / 'downtrend_forecast_input.csv')
    studies = {str(gap): analyze(frame, gap) for gap in (0, 5, 10)}
    reference = studies['5']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(studies, indent=2, allow_nan=False))
    for gap, study in studies.items():
        pd.DataFrame(study['rows']).to_csv(output.with_name(output.stem + f'_gap{gap}.csv'), index=False)

    def fmt(value):
        return f'{value:.1f}' if value is not None else '—'

    comparison = ''
    for gap, study in studies.items():
        for name, title in [
            ('all', '1. All clusters'),
            ('high_onset', '2. Near-high onset'),
            ('high_onset_same_age', 'Pink on session 11'),
            ('high_onset_same_age_still_high', '3. Pink and near high on session 11'),
        ]:
            s = study['cohorts'][name]
            counts = ''.join(f'<td>{s["drop_counts"][str(t)]}/{s["n"]}</td>' for t in (5, 10, 20))
            metric = s['metrics']['low_return_pct']
            pair = f'{metric["mean"]:.1f}% / {metric["median"]:.1f}%' if metric else '—'
            comparison += f'<tr><td>{gap}</td><td>{title}</td><td>{s["n"]}</td>{counts}<td>{pair}</td></tr>'
    details = ''
    chart = go.Figure()
    selected_starts = reference['cohorts']['high_onset_same_age']['starts']
    strict_starts = reference['cohorts']['high_onset_same_age_still_high']['starts']
    for row in reference['rows']:
        if row['start'] not in selected_starts:
            continue
        details += (
            f'<tr><td>{row["start"]}</td><td>{row["subepisodes"]}</td>'
            f'<td>{row["onset_high_gap_pct"]:.2f}%</td><td>{row["anchor_high_gap_pct"]:.2f}%</td>'
            f'<td>{row["low_return_pct"]:.1f}%</td><td>{row["low_sessions"]}</td>'
            f'<td>{row["end_sessions"]}</td><td>{row["end_return_pct"]:+.1f}%</td>'
            f'<td>{"Yes" if row["start"] in strict_starts else "No"}</td></tr>'
        )
        start, end = row['start_index'], row['end_index']
        prices = frame['S&P500_Price'].iloc[start : end + 1]
        chart.add_trace(
            go.Scatter(
                x=list(range(len(prices))),
                y=100 * (prices / prices.iloc[0] - 1),
                name=row['start'],
                mode='lines',
                line={'width': 3 if row['start'] in strict_starts else 1.5},
            )
        )
    start = len(frame) - reference['current_sessions']
    prices = frame['S&P500_Price'].iloc[start:]
    chart.add_trace(
        go.Scatter(
            x=list(range(len(prices))),
            y=100 * (prices / prices.iloc[0] - 1),
            name='Current (unfinished)',
            mode='lines',
            line={'width': 4, 'color': '#ffffff'},
        )
    )
    chart.add_vline(x=reference['current_sessions'] - 1, line_dash='dot')
    chart.update_layout(
        template='plotly_dark',
        height=520,
        title='Price paths: near-high onset and pink on session 11 (5-session merge)',
        xaxis_title='Trading sessions from first onset (day 0)',
        yaxis_title='Return from first onset (%)',
        margin={'t': 70},
    )
    chart_html = chart.to_html(full_html=False, include_plotlyjs='https://cdn.plot.ly/plotly-2.35.0.min.js')
    labels = {
        'low_return_pct': 'First onset → price low (%)',
        'low_sessions': 'First onset → price low (sessions)',
        'low_to_end_sessions': 'Price low → last pink exit (sessions)',
        'end_sessions': 'First onset → last pink exit (sessions)',
        'end_return_pct': 'First onset → last pink exit (%)',
    }
    metrics = ''
    forward = ''
    for name, label in [
        ('high_onset_same_age', 'Near-high onset, pink on session 11'),
        ('high_onset_same_age_still_high', 'Also near high on session 11'),
    ]:
        s = reference['cohorts'][name]
        for key, value in s['metrics'].items():
            metrics += f'<tr><td>{label}</td><td>{html.escape(labels[key])}</td><td>{fmt(value["mean"]) if value else "—"}</td><td>{fmt(value["median"]) if value else "—"}</td></tr>'
        for horizon, value in s['forward'].items():
            n = value['n']
            frequency = f'{value["negative_count"]}/{n} ({100 * value["negative_count"] / n:.1f}%)' if n else '—'
            forward += f'<tr><td>{label}</td><td>{horizon}</td><td>{frequency}</td><td>{fmt(value["mean"])}%</td><td>{fmt(value["median"])}%</td></tr>'
    constituents = ''
    for row in reference['rows']:
        if row['subepisodes'] > 1:
            constituents += (
                f'<tr><td>{row["start"]}</td><td>{row["end"]}</td><td>{", ".join(row["subepisode_starts"])}</td></tr>'
            )
    output.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clustered Pink Episodes Near Price Highs</title><style>
body{{background:#10131c;color:#e1e4ec;font:16px/1.65 Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px}}h1,h2{{color:#f0b3ce}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:10px;border-bottom:1px solid #394254;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{background:#1d2535;padding:18px;border-radius:10px}}.scroll{{overflow-x:auto}}a{{color:#b9d8ff}}
</style><h1>Clustered Pink Episodes Near Price Highs</h1><p>Frozen data through {reference['as_of']}. Current first onset: {reference['current_start']}; session {reference['current_sessions']}.</p>
<p class="note">Exploratory current-constituent study. Historical prices and membership are incomplete, with survivorship bias and a fixed denominator. Clustering reduces repeated signals but does not prove statistical independence. Small conditional samples describe historical scenarios, not calibrated probabilities.</p>
<h2>1. Merge short interruptions</h2><p>Baseline: keep the original episodes. Primary variant: merge episodes separated by at most 5 non-pink trading observations. Sensitivity: allow 10 observations. Merging is transitive; no limit is imposed on the total resulting episode length. First onset stays fixed. The low includes intervening non-pink days; the end is the first non-pink day after the final pink segment. The end is only confirmed once the allowed interruption has elapsed. This is retrospective episode bookkeeping, not a new live trading signal.</p>
<h2>2. Keep near-high first onsets</h2><p>Near a high means the first onset close is within 2% of the highest adjusted SPY close in the trailing 63 sessions, including onset. A later subepisode beginning near a high cannot make an earlier low-price cluster qualify. This avoids choosing a favorable restart after seeing outcomes. The current onset is {reference['current_onset_gap_pct']:.2f}% below this high.</p>
<h2>3. Match observed age and continued price strength</h2><p>First retain clusters whose original pink condition is active on session 11; an anchor in an intervening non-pink gap is excluded. Then require that anchor close also be within 2% of its trailing 63-session high. The current close is {reference['current_gap_pct']:.2f}% below its high. Today's onset and age are unchanged under both merge rules.</p>
<div class="scroll"><table><tr><th>Gap allowed (sessions)</th><th>Stage</th><th>n</th><th>Drop &gt;5%</th><th>Drop &gt;10%</th><th>Drop &gt;20%</th><th>Low return mean / median</th></tr>{comparison}</table></div>
<p>Decline thresholds are strict; counts overlap. Full-episode declines use the first onset close and the minimum close before the last pink exit. Outcomes can include lows before session 11 and are not equivalent to future losses from today's price. Current unfinished cluster and unresolved historical clusters are excluded.</p>
<h2>Historical paths: primary 5-session rule</h2>{chart_html}<p>Thicker historical lines also satisfy the session-11 near-high filter. The vertical line marks today's equivalent age; the white line is the unfinished current episode.</p>
<h2>Individual comparable clusters</h2><div class="scroll"><table><tr><th>First onset</th><th>Segments</th><th>Onset gap to high</th><th>Session-11 gap</th><th>Low return</th><th>Sessions to low</th><th>Sessions to exit</th><th>Exit return</th><th>Still near high</th></tr>{details}</table></div>
<h2>Full-episode descriptive statistics</h2><div class="scroll"><table><tr><th>Cohort</th><th>Metric</th><th>Mean</th><th>Median</th></tr>{metrics}</table></div>
<h2>Forward scenarios from the session-11 price</h2><p>Entire horizon must be observed before the current first onset. Pink may end during the horizon. Return means terminal adjusted close relative to the session-11 close.</p><div class="scroll"><table><tr><th>Cohort</th><th>Sessions ahead</th><th>Lower terminal close</th><th>Mean return</th><th>Median return</th></tr>{forward}</table></div>
<h2>Audit: merged constituent episodes (5-session rule)</h2><div class="scroll"><table><tr><th>First onset</th><th>Last exit</th><th>Original onset dates</th></tr>{constituents}</table></div>
<p>The accompanying JSON includes all episodes, 0/5/10-session sensitivity, 1/2/5% onset filters and 21/60/126-session forward outcomes. Original daily pink definitions and original reports remain unchanged. No entry strategy or causal relationship is established.</p></html>""")
    print(json.dumps({gap: study['cohorts'] for gap, study in studies.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.directory, args.output)
