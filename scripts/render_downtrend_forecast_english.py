"""Render the English publication page from the reproducible event-study artifacts."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def render(statistics, data, output):
    r = json.loads(statistics.read_text())
    d = pd.read_csv(data, parse_dates=['Date'])
    c = r['current']
    ep = r['episode_summary']

    def percent(x):
        return f'{x:+.1f}%'

    def table(headers, rows):
        return (
            '<div class="scroll"><table><thead><tr>'
            + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows)
            + '</tbody></table></div>'
        )

    sections = [
        f'<h1>Scenarios After 200MA Breadth Turns Pink</h1><p class="muted">Data through {c["as_of"]} · Current episode began {c["start"]} · Session {c["observed_sessions"]} · Completed historical episodes: {ep["n"]}</p>'
    ]
    sections.append(
        '<div class="note"><strong>Scope:</strong> The main analysis follows each entire pink episode. A 63-session window represents about three months, not the full warning cycle. Historical frequencies describe a small sample; they are not calibrated probabilities for the current episode.</div>'
    )
    sections.append(
        '<p><a href="index.html">Market breadth dashboard</a> · <a href="downtrend_forecast_high_price.html">Pink Onsets Near Price Highs</a></p>'
    )
    noise = r.get('chattering_filter', {})
    if noise.get('enabled'):
        removed = ', '.join(noise['excluded_starts']) or 'None'
        sections.append(
            '<h2>Rapid switching excluded</h2><p>Exclude all original segments in a completed chain of at least three pink episodes separated by at most five non-pink sessions, with a total first-onset-to-final-exit span of at most 40 sessions. Retained episodes keep their original dates and outcomes; episodes are not merged.</p>'
            f'<p>Original completed episodes: {noise["original_n"]}; retained: {noise["retained_n"]}. Excluded onset dates: {removed}.</p>'
            '<div class="note">This retrospective noise filter uses subsequent regime changes. It cannot necessarily be evaluated at onset or on session 11. Filtered statistics are descriptive sensitivity results, not a more accurate live probability estimate. The same rule is applied separately to each universe sensitivity model.</div>'
        )
    if r.get('long_history_audit'):
        audit = r['long_history_audit']
        sections.append(
            '<h2>Fresh long-history data and universe sensitivity</h2><div class="note"><strong>Exploratory survivor-cohort study:</strong> The reference model uses the current 504 constituents and preserves the project’s fixed denominator: a missing or unavailable 200-day average counts as not above the average. Earlier periods therefore contain both survivorship bias and denominator dilution. This is not a verified point-in-time reconstruction of the historical S&amp;P 500.</div>'
        )
        sections.append(
            f'<p>Fresh retrieval covered {audit["requested"]} current/former symbols; usable adjusted price history was obtained for {audit["usable"]}, including all {audit["current_count"]} current constituents. SPY covers {r["history_start"]} through {c["as_of"]}. Prices use FMP adjClose; the provider describes it as adjusted for splits and dividends. The archived inputs and acquisition manifest record individual coverage and hashes.</p>'
        )
        sections.append(
            f'<p>Historical membership was also reconstructed from the provider’s changes, but {audit["membership_anomalies"]} reverse-history inconsistencies remain. Missing delisted histories prevent complete historical coverage. Its results are a sensitivity model, not the primary forecast evidence.</p>'
        )
        quality_rows = []
        for year in ('1993', '2000', '2008', '2016', '2026'):
            fixed = audit['datasets']['current_fixed']['annual_coverage'][year]
            historical = audit['datasets']['historical_reconstructed']['annual_coverage'][year]
            quality_rows.append(
                [
                    year,
                    f'{fixed["median_coverage"] * 100:.1f}%',
                    f'{historical["median_coverage"] * 100:.1f}%',
                    f'{historical["median_eligible"]:.0f} / {historical["median_members"]:.0f}',
                ]
            )
        sections.append(
            table(
                [
                    'Year',
                    'Current-cohort median MA eligibility',
                    'Reconstructed-history median MA eligibility',
                    'Historical eligible / members (medians)',
                ],
                quality_rows,
            )
        )
        model_rows = []
        model_names = {
            'current_fixed': 'Current cohort, fixed denominator (reference)',
            'current_eligible': 'Current cohort, MA-eligible denominator',
            'historical_reconstructed': 'Historical changes, available MA-eligible members (incomplete)',
        }
        for name, summary in audit['model_summaries'].items():
            model_rows.append(
                [
                    model_names[name],
                    summary['n'],
                    f'{summary["bottom_sessions"]["mean"]:.1f} / {summary["bottom_sessions"]["median"]:.1f}',
                    f'{percent(summary["bottom_return_pct"]["mean"])} / {percent(summary["bottom_return_pct"]["median"])}',
                    f'{summary["end_sessions"]["mean"]:.1f} / {summary["end_sessions"]["median"]:.1f}',
                ]
            )
        sections.append(
            table(
                [
                    'Model',
                    'Completed n',
                    'Onset → low sessions: mean / median',
                    'Low return: mean / median',
                    'Onset → end sessions: mean / median',
                ],
                model_rows,
            )
        )
        sections.append(
            '<p>Different denominator and membership models change episode dates and counts. They should not be pooled into one probability estimate. The reference preserves the visible September 18 onset; the MA-eligible current-cohort variant produces a September 16 onset.</p>'
        )
        age = audit['same_age_summary']
        sections.append(
            f'<h3>Full-episode results conditioned on surviving to the current age</h3><p>Only {age["n"]} reference episodes were still pink on session {c["observed_sessions"]}. For those completed episodes, onset-to-low time averaged {age["bottom_sessions"]["mean"]:.1f} sessions (median {age["bottom_sessions"]["median"]:.1f}); low-to-end time averaged {age["bottom_to_end_sessions"]["mean"]:.1f} (median {age["bottom_to_end_sessions"]["median"]:.1f}); the low return averaged {percent(age["bottom_return_pct"]["mean"])} (median {percent(age["bottom_return_pct"]["median"])}). The all-episode table below includes shorter warnings too.</p>'
        )
    short_50 = (
        f' Short-term smoothing of the 50-day breadth series: {c["short_50_pct"]:.1f}%.' if 'short_50_pct' in c else ''
    )
    sections.append(
        f'<h2>Current conditions</h2><p>Smoothed 200-day breadth: {c["long_pct"]:.1f}% (EMA200); short-term breadth: {c["short_pct"]:.1f}% (EMA8); unsmoothed percentage above the stock-level 200-day moving average: {c["raw_pct"]:.1f}%.{short_50}</p><p>The SPY-derived price series is {c["price"]:.2f}, {percent(c["return_since_start"])} since the pink onset and {percent(c["drawdown_63"])} below its highest close in the latest 63 sessions.</p>'
    )
    sections.append(
        '<h2>Full episode: onset, price low, and warning end</h2><p>The price low is the first minimum close within the pink interval: onset included, first non-pink session excluded. Warning end is the first non-pink session and its close. Onset is day 0. The current unfinished episode is excluded from completed-episode statistics. These are retrospective measurements, not dates knowable in advance.</p>'
    )
    labels = [
        ('bottom_sessions', 'Onset → price low: trading sessions', 'sessions'),
        ('bottom_calendar_days', 'Onset → price low: calendar days', 'days'),
        ('bottom_return_pct', 'Onset → price low: price return', '%'),
        ('bottom_to_end_sessions', 'Price low → first non-pink day: trading sessions', 'sessions'),
        ('bottom_to_end_calendar_days', 'Price low → first non-pink day: calendar days', 'days'),
        ('end_sessions', 'Onset → first non-pink day: trading sessions', 'sessions'),
        ('end_calendar_days', 'Onset → first non-pink day: calendar days', 'days'),
        ('end_return_pct', 'Onset → first non-pink day: price return', '%'),
        ('last_pink_return_pct', 'Onset → last pink day: price return (reference)', '%'),
    ]
    rows = []
    for key, label, unit in labels:
        v = ep[key]
        rows.append(
            [
                label,
                ep['n'],
                f'{v["mean"]:.1f} {unit}',
                f'{v["median"]:.1f} {unit}',
                f'{v["min"]:.1f} to {v["max"]:.1f} {unit}',
            ]
        )
    sections.append(table(['Measure', 'n', 'Mean', 'Median', 'Minimum–maximum'], rows))
    zero = ep['n'] - r['declining_episode_summary']['n']
    sections.append(
        f'<p>{zero} episodes never closed below their onset close during the pink interval. They contribute zero days and zero return to the price-low measurements. Means of the two time segments add to the mean total; medians generally do not.</p>'
    )
    declines = r['declining_episode_summary']
    if declines['n']:
        sections.append(
            f'<p>Among only the {declines["n"]} episodes that did decline below onset, time to the price low averaged {declines["bottom_sessions"]["mean"]:.1f} sessions (median {declines["bottom_sessions"]["median"]:.1f}), and the price-low return averaged {percent(declines["bottom_return_pct"]["mean"])} (median {percent(declines["bottom_return_pct"]["median"])}). This outcome-conditioned subset is a diagnostic, not an unconditional forecast.</p>'
        )
    episode_rows = []
    for e in r['events']:
        v = e['episode_outcome']
        episode_rows.append(
            [
                e['start'],
                v['bottom_date'],
                f'{v["bottom_sessions"]} / {v["bottom_calendar_days"]}',
                percent(v['bottom_return_pct']),
                f'{v["bottom_to_end_sessions"]} / {v["bottom_to_end_calendar_days"]}',
                e['end'],
                percent(v['end_return_pct']),
            ]
        )
    sections.append(
        table(
            [
                'Pink onset',
                'Price-low date',
                'Onset → low: sessions / days',
                'Low return',
                'Low → end: sessions / days',
                'First non-pink date',
                'End return',
            ],
            episode_rows,
        )
    )
    fig = go.Figure()
    for e in r['events']:
        start = e['start_index']
        end = int(d.index[d.Date == pd.Timestamp(e['end'])][0])
        path = (d['S&P500_Price'].iloc[start : end + 1] / d['S&P500_Price'].iloc[start] - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=list(range(len(path))),
                y=path.tolist(),
                name=e['start'],
                hovertemplate='%{x} sessions since onset: %{y:.1f}%<extra>%{fullData.name}</extra>',
            )
        )
    fig.update_layout(
        template='plotly_dark',
        title='Historical price paths over the entire pink episode',
        xaxis_title='Trading sessions since onset (day 0)',
        yaxis_title='Price return from onset (%)',
        height=470,
        legend={'orientation': 'h'},
        margin={'t': 60, 'b': 100},
    )
    if r.get('long_history_audit'):
        buttons = [
            {
                'label': 'All episodes',
                'method': 'update',
                'args': [{'visible': [True] * len(r['events'])}, {'showlegend': False}],
            }
        ]
        for decade in (1990, 2000, 2010, 2020):
            buttons.append(
                {
                    'label': f'{decade}s',
                    'method': 'update',
                    'args': [
                        {'visible': [decade <= int(e['start'][:4]) < decade + 10 for e in r['events']]},
                        {'showlegend': True},
                    ],
                }
            )
        fig.update_layout(
            showlegend=False,
            updatemenus=[{'buttons': buttons, 'x': 1, 'xanchor': 'right', 'y': 1.15}],
        )
    sections.append(
        fig.to_html(
            full_html=False, include_plotlyjs='https://cdn.plot.ly/plotly-2.35.0.min.js', config={'responsive': True}
        )
    )
    sections.append(
        '<p>Each path ends at the first non-pink close. Episodes have different lengths; this chart is retrospective and is not a forecast band.</p>'
    )
    fs = r['fixed_126_bottom_summary']
    sections.append(
        f'<h3>Sensitivity to the definition of a bottom</h3><p>Using the minimum close in a fixed 126-session window from onset instead, n={fs["n"]}: time to the low averages {fs["sessions"]["mean"]:.1f} sessions / {fs["calendar_days"]["mean"]:.1f} calendar days, with medians of {fs["sessions"]["median"]:.1f} / {fs["calendar_days"]["median"]:.1f}. Return averages {percent(fs["return_pct"]["mean"])}; median {percent(fs["return_pct"]["median"])}. A fixed window can miss a later low or end during an ongoing decline.</p>'
    )
    low = c['lowest_close_to_date']
    sections.append(
        f'<p><strong>Current episode:</strong> the lowest close observed so far occurred on {low["date"]}, {low["sessions"]} sessions after onset, with an onset-relative return of {percent(low["return_pct"])}. This is not a confirmed final bottom.</p>'
    )
    sections.append(
        '<h2>Price low versus breadth-bottom detection</h2><p>All price returns use the pink-onset close as the denominator. A plotted bottom date and the date that bottom can first be detected are different. Compare only matching episodes; do not subtract averages from different samples or subtract marginal medians to estimate the median paired difference.</p>'
    )
    for mode, label in [('long', '200MA breadth bottom'), ('short', '8MA breadth bottom below 40%')]:
        cohort = r['breadth_bottom_comparison'][mode]
        s = cohort['summary']
        sections.append(f'<h3>{label}: detected in {s["n"]} episodes; absent in {s["missing"]}</h3>')
        measures = [
            ('price_bottom_return_pct', 'Onset → within-episode price low', '%'),
            ('marker_return_pct', 'Onset → breadth-bottom marker date', '%'),
            ('detection_return_pct', 'Onset → first detectable date', '%'),
            ('marker_gap_pp', 'Marker-date return minus price-low return', 'percentage points'),
            ('detection_gap_pp', 'Detection-date return minus price-low return', 'percentage points'),
            ('price_change_bottom_to_detection_pct', 'Detection price relative to the within-episode minimum', '%'),
            ('marker_to_detection_sessions', 'Marker date → first detection', 'sessions'),
        ]
        rows = []
        for key, name, unit in measures:
            v = s[key]
            rows.append(
                [name, f'{v["mean"]:+.1f} {unit}' if v else 'N/A', f'{v["median"]:+.1f} {unit}' if v else 'N/A']
            )
        sections.append(table(['Paired measure', 'Mean', 'Median'], rows))
        rows = []
        for v in cohort['rows']:
            rows.append(
                [
                    v['start'],
                    percent(v['price_bottom_return_pct']),
                    v.get('marker_date', 'Not detected'),
                    v.get('detection_date', '—'),
                    percent(v['detection_return_pct']) if v['detected'] else '—',
                    f'{v["detection_gap_pp"]:+.1f} pp' if v['detected'] else '—',
                ]
            )
        sections.append(
            table(
                ['Pink onset', 'Price-low return', 'Marker date', 'First detection', 'Detection return', 'Paired gap'],
                rows,
            )
        )
        sections.append(
            f'<p>Detection preceded the eventual within-episode price low in {s["detection_before_price_bottom_count"]}/{s["n"]} matched episodes. A ratio between detection price and the eventual low is not necessarily a realizable forward return.</p>'
        )
    sections.append(
        '<p>To determine detection dates, the chart algorithm was replayed on each daily data prefix. For 200MA breadth: find_peaks on the negative series, distance=50 and prominence=0.015. For short breadth: retain EMA8 observations below 40%, then find_peaks with prominence=0.02. A qualifying pivot must be inside the pink interval; the first detection is sought before the next pink onset. Later-revised or disappearing candidates still count as first detections. These are chart detections, not the separate trading engine’s entry signals.</p>'
    )
    sections.append(
        '<h2>Supplement: fixed-horizon outcomes from today’s equivalent age</h2><p>Historical episodes must still be pink at the same elapsed age as the current episode. Outcomes are measured from that equivalent date, including after pink ends. Only fully observed horizons that end before the current episode begins are used. The 63-session horizon covers roughly three months ahead—about 73 elapsed sessions from onset in this report—and is not selected for demonstrated predictive power.</p>'
    )
    rows = []
    for s in r['primary']:
        q = s['return_quartiles']
        rows.append(
            [
                s['horizon'],
                s['n'],
                percent(q[1]),
                f'{percent(q[0])} to {percent(q[2])}',
                f'{s["positive_count"]}/{s["n"]}',
                percent(s['worst_quartiles'][1]),
            ]
        )
    sections.append(
        table(
            [
                'Sessions ahead',
                'n',
                'Median endpoint return',
                '25th–75th percentiles',
                'Positive endpoint',
                'Median worst close vs anchor',
            ],
            rows,
        )
    )
    sections.append(
        '<p>5 / 21 / 63 / 126 / 252 sessions are approximately one week / one month / three months / six months / twelve months. The quartiles describe observed dispersion, not confidence intervals. Longer windows have fewer fully observed cases.</p>'
    )
    scenario_names = ['Less than 5% decline', '5% to less than 10% decline', '10% or larger decline']
    rows = []
    for s in r['primary']:
        if s['horizon'] not in (63, 126, 252):
            continue
        for name, v in zip(scenario_names, s['scenarios'], strict=True):
            rows.append(
                [
                    s['horizon'],
                    name,
                    f'{v["count"]}/{s["n"]}',
                    f'{v["share_pct"]:.1f}%',
                    f'{v["wilson"][0]:.1f}% to {v["wilson"][1]:.1f}%',
                ]
            )
    sections.append(
        table(
            ['Sessions ahead', 'Worst-close scenario', 'Count', 'Historical share', 'Reference 95% Wilson interval'],
            rows,
        )
    )
    sections.append(
        '<p>These mutually exclusive scenarios use the lowest close relative to the equivalent-age anchor, not peak-to-trough drawdown or intraday lows. An episode can fall substantially and still end the horizon with a positive return.</p>'
    )
    sens = next(s for s in r['sensitivity'] if s['horizon'] == 63)
    shares = ', '.join(
        f'{name}: {v["count"]}/{sens["n"]} ({v["share_pct"]:.1f}%)'
        for name, v in zip(scenario_names, sens['scenarios'], strict=True)
    )
    sections.append(
        f'<h3>Repeated-signal sensitivity</h3><p>Keeping onsets more than 126 sessions apart reduces the same-age sample to {sens["n"]} for the 63-session horizon: {shares}. Selection depends on onset dates, not future price outcomes. The sample still contains related market episodes.</p>'
    )
    sections.append(
        '<h2>What to monitor</h2><ul><li>Recovery: short breadth turns up and its gap to long breadth narrows. Distinguish a price-only rally from broader participation.</li><li>Further weakness: short breadth falls again while price makes new lows. The 40% and 30% short-breadth levels are monitoring references; conditional probabilities at those thresholds were not estimated here.</li><li>Warning end: pink ends when either defining condition fails. That change does not guarantee a final price bottom.</li></ul>'
    )
    sections.append(
        '<h2>Methods and limitations</h2><ul><li>Pink means Breadth_200MA_Trend == -1 and Breadth_Index_8MA &lt; Breadth_Index_200MA. The trend uses the code’s hysteresis rule.</li><li>Stock-level 200-day breadth is smoothed using EMA200; its short line is EMA8. This is different from an index price crossing its own 200-day moving average.</li><li>The study uses a single published CSV snapshot beginning in 2016, excluding signal onsets in the first 200 observations to reduce initialization effects. Rising-price episodes are retained. The old report’s hand-selected 2007–2025 sample is not merged into this study.</li><li>Constituent history is not verified as point-in-time; survivorship and missing-data denominator biases may affect the underlying breadth. Constituents and raw provider prices were not independently reconciled.</li><li>S&amp;P500_Price is the project’s SPY-derived series, not the index level. The CSV alone does not establish adjustment or complete dividend-reinvestment conventions.</li><li>The sample is small and related episodes are not independent. Wilson intervals are reference intervals under a binomial independence assumption, not calibrated forecast intervals. Similarity weighting and extra outcome filters are not applied.</li><li>Price lows and full pink intervals are identified retrospectively. Reported returns exclude costs and do not constitute simulated trades. No trading profitability or optimal entry rule is established.</li><li>Price data use closing observations, not intraday lows. Future news and macroeconomic conditions after the data cutoff are not incorporated.</li></ul>'
    )
    sections.append(
        '<h2>Changes from the previous publication</h2><p>The previous page described the March 12, 2026 episode as of March 26, excluded rising-price divergence cases, and assigned similarity-weighted bottom forecasts. This report studies the September episode with reproducible criteria, retains rising-price cases, treats March as a completed historical episode, and does not carry forward the old probabilities, dates, or price targets.</p>'
    )
    sections.append(
        f'<p>Source: <a href="{html.escape(r["source"])}">published breadth CSV</a>; <a href="https://tradermonty.github.io/market-breadth-analysis/market_breadth.html">daily chart</a>. History begins {r["history_start"]}. Snapshot SHA-256: <code>{r["input_sha256"]}</code>. This report is frozen at {c["as_of"]}; daily chart updates do not automatically refresh its statistics.</p>'
    )
    css = 'body{background:#10141f;color:#e4e9f2;font:16px/1.8 system-ui,sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:28px}h1{font-size:28px}h2{margin-top:36px;color:#ffbad4}a{color:#8ecbff}.note{background:#1c2333;border:1px solid #39465c;border-radius:12px;padding:18px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #39465c;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}.muted{color:#abb6ca}li{margin-bottom:10px}code{overflow-wrap:anywhere}@media(max-width:700px){main{padding:16px}}'
    if r.get('long_history_audit'):
        sections = [
            text.replace(
                'The study uses a single published CSV snapshot beginning in 2016, excluding signal onsets in the first 200 observations to reduce initialization effects.',
                'The study uses freshly retrieved price histories with a 1990 warmup, aligned to SPY from 1993; signal onsets in the first 200 SPY observations are excluded to reduce initialization effects.',
            )
            .replace(
                'The CSV alone does not establish adjustment or complete dividend-reinvestment conventions.',
                'FMP describes adjClose as split- and dividend-adjusted; these price-return calculations are not a transaction-level dividend cash-flow simulation.',
            )
            .replace('Source: <a', 'Data-source documentation: <a')
            .replace('>published breadth CSV</a>', '>FMP historical price API</a>')
            for text in sections
        ]
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Scenarios After 200MA Breadth Turns Pink</title><style>'
        + css
        + '</style></head><body><main>'
        + '\n'.join(sections)
        + '</main></body></html>'
    )
    output.write_text(page + '\n')
    print(f'English publication page written: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', type=Path, default=Path('reports/downtrend_forecast_statistics.json'))
    parser.add_argument('--input', type=Path, default=Path('reports/downtrend_forecast_input.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports/downtrend_forecast_en.html'))
    args = parser.parse_args()
    render(args.statistics, args.input, args.output)
