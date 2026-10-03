"""Reproducible, API-free event study of the published pink breadth regime."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

from scripts.analyze_high_price_pink import chattering_starts

SOURCE = 'https://tradermonty.github.io/market-breadth-analysis/market_breadth_data.csv'
PRICE = 'S&P500_Price'
LONG = 'Breadth_Index_200MA'
SHORT = 'Breadth_Index_8MA'


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.95996398454
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(100 * (mid - half)), float(100 * (mid + half))]


def episode_outcome(d, start, end):
    """Bottom is the first minimum close in [start, end); end is first non-pink day."""
    path = d[PRICE].iloc[start:end]
    bottom = start + int(np.argmin(path.to_numpy()))
    first_price = float(d[PRICE].iloc[start])
    return {
        'bottom_date': str(d.Date.iloc[bottom].date()),
        'bottom_sessions': bottom - start,
        'bottom_calendar_days': int((d.Date.iloc[bottom] - d.Date.iloc[start]).days),
        'bottom_return_pct': float(100 * (d[PRICE].iloc[bottom] / first_price - 1)),
        'bottom_to_end_sessions': end - bottom,
        'bottom_to_end_calendar_days': int((d.Date.iloc[end] - d.Date.iloc[bottom]).days),
        'end_sessions': end - start,
        'end_calendar_days': int((d.Date.iloc[end] - d.Date.iloc[start]).days),
        'end_return_pct': float(100 * (d[PRICE].iloc[end] / first_price - 1)),
        'last_pink_return_pct': float(100 * (d[PRICE].iloc[end - 1] / first_price - 1)),
    }


def summarize_outcomes(outcomes):
    keys = [
        'bottom_sessions',
        'bottom_calendar_days',
        'bottom_return_pct',
        'bottom_to_end_sessions',
        'bottom_to_end_calendar_days',
        'end_sessions',
        'end_calendar_days',
        'end_return_pct',
        'last_pink_return_pct',
    ]
    result = {'n': len(outcomes)}
    for key in keys:
        values = [x[key] for x in outcomes]
        result[key] = (
            {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
            if values
            else None
        )
    return result


def breadth_bottom_comparison(d, events, current_start):
    """Replay the chart algorithm on prefixes; never price a pivot as its detection date."""
    result = {}
    for mode in ('long', 'short'):
        rows = []
        for k, e in enumerate(events):
            start = e['start_index']
            end = int(d.index[d.Date == pd.Timestamp(e['end'])][0])
            limit = e.get('next_start_index', events[k + 1]['start_index'] if k + 1 < len(events) else current_start)
            found = None
            for observed in range(start + 1, limit):
                if mode == 'long':
                    pivots, _ = find_peaks(-d[LONG].iloc[: observed + 1].to_numpy(), distance=50, prominence=0.015)
                else:
                    short = d[SHORT].iloc[: observed + 1]
                    positions = np.flatnonzero(short.to_numpy() < 0.4)
                    peaks, _ = find_peaks(-short.iloc[positions].to_numpy(), prominence=0.02)
                    pivots = positions[peaks]
                candidates = [int(p) for p in pivots if start <= p < end]
                if candidates:
                    found = (observed, candidates[0])
                    break
            base = float(d[PRICE].iloc[start])
            row = {
                'start': e['start'],
                'price_bottom_date': e['episode_outcome']['bottom_date'],
                'price_bottom_return_pct': e['episode_outcome']['bottom_return_pct'],
                'detected': found is not None,
            }
            if found:
                observed, pivot = found
                marker_return = float(100 * (d[PRICE].iloc[pivot] / base - 1))
                detection_return = float(100 * (d[PRICE].iloc[observed] / base - 1))
                bottom_return = row['price_bottom_return_pct']
                row.update(
                    {
                        'marker_date': str(d.Date.iloc[pivot].date()),
                        'detection_date': str(d.Date.iloc[observed].date()),
                        'marker_return_pct': marker_return,
                        'detection_return_pct': detection_return,
                        'marker_gap_pp': marker_return - bottom_return,
                        'detection_gap_pp': detection_return - bottom_return,
                        'price_change_bottom_to_detection_pct': float(
                            100 * ((1 + detection_return / 100) / (1 + bottom_return / 100) - 1)
                        ),
                        'marker_to_detection_sessions': observed - pivot,
                        'detection_before_price_bottom': pd.Timestamp(d.Date.iloc[observed])
                        < pd.Timestamp(row['price_bottom_date']),
                    }
                )
            rows.append(row)
        paired = [r for r in rows if r['detected']]
        stats = {
            'n': len(paired),
            'missing': len(rows) - len(paired),
            'detection_before_price_bottom_count': sum(r['detection_before_price_bottom'] for r in paired),
        }
        for key in (
            'price_bottom_return_pct',
            'marker_return_pct',
            'detection_return_pct',
            'marker_gap_pp',
            'detection_gap_pp',
            'price_change_bottom_to_detection_pct',
            'marker_to_detection_sessions',
        ):
            values = [r[key] for r in paired]
            stats[key] = {'mean': float(np.mean(values)), 'median': float(np.median(values))} if values else None
        result[mode] = {'summary': stats, 'rows': rows}
    return result


def analyze(frame, exclude_chattering=False):
    d = frame.copy().reset_index(drop=True)
    d['Date'] = pd.to_datetime(d['Date'], errors='raise')
    if d['Date'].duplicated().any() or not d['Date'].is_monotonic_increasing:
        raise ValueError('Dates must be sorted and unique')
    required = [PRICE, LONG, SHORT, 'Breadth_200MA_Trend']
    if d[required].isna().any().any() or not np.isfinite(d[required]).all().all():
        raise ValueError('Required values must be finite')
    if (d[PRICE] <= 0).any() or not ((d[[LONG, SHORT]] >= 0) & (d[[LONG, SHORT]] <= 1)).all().all():
        raise ValueError('Invalid price or breadth range')
    mask = (d['Breadth_200MA_Trend'] == -1) & (d[SHORT] < d[LONG])
    exported = d['Bearish_Signal'].astype(str).str.lower().map({'true': True, 'false': False})
    if exported.isna().any() or not exported.eq(mask).all():
        raise ValueError('Exported Bearish_Signal does not match the documented rule')
    starts = np.flatnonzero(mask & ~mask.shift(1, fill_value=False))
    if not mask.iloc[-1]:
        raise ValueError('Latest observation is not in the pink regime')
    current_start = int(starts[-1])
    age = len(d) - 1 - current_start
    # Exclude the initial EMA initialization period. No outcome-based exclusions.
    starts = [int(i) for i in starts if i >= 200 and i < current_start]
    original_events = []
    for i in starts:
        off = np.flatnonzero(~mask.iloc[i:].to_numpy())
        end = i + int(off[0])
        original_events.append(
            {'start_index': i, 'start': str(d.Date.iloc[i].date()), 'end': str(d.Date.iloc[end].date())}
        )
    excluded, groups = chattering_starts(original_events, d.Date) if exclude_chattering else (set(), [])
    next_starts = dict(zip(starts, [*starts[1:], current_start]))
    events = []
    eligible = []
    for i in starts:
        if i in excluded:
            continue
        off = np.flatnonzero(~mask.iloc[i:].to_numpy())
        end = i + int(off[0]) if len(off) else len(d)
        anchor = i + age
        item = {
            'start_index': i,
            'next_start_index': next_starts[i],
            'start': str(d.Date.iloc[i].date()),
            'end': str(d.Date.iloc[end].date()) if end < len(d) else None,
            'pink_sessions': end - i,
            'same_age_eligible': anchor < end,
            'anchor_index': anchor,
            'remaining_pink_sessions': max(0, end - anchor),
            'anchor_long_pct': float(d[LONG].iloc[anchor] * 100) if anchor < len(d) else None,
            'anchor_short_pct': float(d[SHORT].iloc[anchor] * 100) if anchor < len(d) else None,
        }
        if end < len(d):
            item['episode_outcome'] = episode_outcome(d, i, end)
        if i + 126 < current_start:
            bottom_126 = i + int(np.argmin(d[PRICE].iloc[i : i + 127].to_numpy()))
            item['fixed_126_bottom'] = {
                'date': str(d.Date.iloc[bottom_126].date()),
                'sessions': bottom_126 - i,
                'calendar_days': int((d.Date.iloc[bottom_126] - d.Date.iloc[i]).days),
                'return_pct': float(100 * (d[PRICE].iloc[bottom_126] / d[PRICE].iloc[i] - 1)),
            }
        events.append(item)
        if anchor < end:
            eligible.append(item)

    def study(cohort):
        stats = []
        for h in (5, 21, 63, 126, 252):
            rows = []
            for e in cohort:
                a = e['anchor_index']
                # Fully observed horizon must finish before the current episode begins.
                if a + h >= current_start:
                    continue
                path = d[PRICE].iloc[a : a + h + 1].to_numpy()
                ret = 100 * (path[-1] / path[0] - 1)
                worst = round(float(100 * (path.min() / path[0] - 1)), 10)
                mdd = 100 * (path / np.maximum.accumulate(path) - 1).min()
                row = {
                    'start': e['start'],
                    'return_pct': float(ret),
                    'worst_close_pct': float(worst),
                    'max_drawdown_pct': float(mdd),
                }
                rows.append(row)
                if h == 63:
                    e['forward_63'] = row
            n = len(rows)
            if not n:
                continue
            r = np.array([x['return_pct'] for x in rows])
            w = np.array([x['worst_close_pct'] for x in rows])

            def q(x):
                return [float(v) for v in np.percentile(x, [25, 50, 75])]

            categories = [int((w > -5).sum()), int(((w <= -5) & (w > -10)).sum()), int((w <= -10).sum())]
            stats.append(
                {
                    'horizon': h,
                    'n': n,
                    'return_quartiles': q(r),
                    'worst_quartiles': q(w),
                    'positive_count': int((r > 0).sum()),
                    'positive_wilson': wilson(int((r > 0).sum()), n),
                    'scenarios': [{'count': k, 'share_pct': k / n * 100, 'wilson': wilson(k, n)} for k in categories],
                    'rows': rows,
                }
            )
        return stats

    primary = study(eligible)
    # Fixed 126-session gap: membership depends only on signal onset, not future returns.
    clustered = []
    previous = -10000
    for e in events:
        if e['start_index'] - previous <= 126:
            continue
        previous = e['start_index']
        if e['same_age_eligible']:
            clustered.append(e)
    sensitivity = study(clustered)
    cur = d.iloc[-1]
    since = d[PRICE].iloc[current_start:]
    current = {
        'as_of': str(cur.Date.date()),
        'start': str(d.Date.iloc[current_start].date()),
        'age_offset': age,
        'observed_sessions': age + 1,
        'price': float(cur[PRICE]),
        'long_pct': float(cur[LONG] * 100),
        'short_pct': float(cur[SHORT] * 100),
        'raw_pct': float(cur['Breadth_Index_Raw'] * 100),
        'return_since_start': float(100 * (cur[PRICE] / since.iloc[0] - 1)),
        'drawdown_63': float(100 * (cur[PRICE] / d[PRICE].iloc[-63:].max() - 1)),
    }
    if 'Breadth_50_Index_8MA' in d:
        current['short_50_pct'] = float(cur['Breadth_50_Index_8MA'] * 100)
    low_index = current_start + int(np.argmin(since.to_numpy()))
    current['lowest_close_to_date'] = {
        'date': str(d.Date.iloc[low_index].date()),
        'sessions': low_index - current_start,
        'return_pct': float(100 * (d[PRICE].iloc[low_index] / since.iloc[0] - 1)),
    }
    completed = [e['episode_outcome'] for e in events if 'episode_outcome' in e]
    fixed = [e['fixed_126_bottom'] for e in events if 'fixed_126_bottom' in e]
    fixed_summary = {'n': len(fixed)}
    for key in ('sessions', 'calendar_days', 'return_pct'):
        values = [x[key] for x in fixed]
        fixed_summary[key] = {'mean': float(np.mean(values)), 'median': float(np.median(values))} if values else None
    return d, {
        'current': current,
        'primary': primary,
        'sensitivity': sensitivity,
        'events': events,
        'history_start': str(d.Date.iloc[0].date()),
        'eligible_count': len(eligible),
        'sensitivity_count': len(clustered),
        'episode_summary': summarize_outcomes(completed),
        'declining_episode_summary': summarize_outcomes([x for x in completed if x['bottom_return_pct'] < -1e-9]),
        'fixed_126_bottom_summary': fixed_summary,
        'breadth_bottom_comparison': breadth_bottom_comparison(d, events, current_start),
        'chattering_filter': {
            'enabled': exclude_chattering,
            'excluded_starts': [e['start'] for e in original_events if e['start_index'] in excluded],
            'original_n': len(starts),
            'retained_n': len(events),
            'max_nonpink_gap': 5,
            'max_span_sessions': 40,
            'min_pink_segments': 3,
            'groups': groups,
        },
    }


def build(input_path, output_dir, exclude_chattering=False):
    raw = input_path.read_bytes()
    d, result = analyze(pd.read_csv(input_path), exclude_chattering)
    result['source'] = SOURCE
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    c = result['current']
    h63 = next(s for s in result['primary'] if s['horizon'] == 63)
    sens = next(s for s in result['sensitivity'] if s['horizon'] == 63)
    remaining = [e['remaining_pink_sessions'] for e in result['events'] if e['same_age_eligible'] and e['end']]
    duration_text = (
        f'現在相当日から最初の非ピンク日までの営業日数は、完了済み{len(remaining)}件で'
        f'中央値{np.median(remaining):.0f}日、範囲{min(remaining)}〜{max(remaining)}日。'
        '色の終了時期の記述統計であり、価格の底までの日数ではありません。'
    )
    labels = ['A：下落が5%未満に収まる', 'B：5%以上・10%未満の調整', 'C：10%以上の下落']

    def fmt(x):
        return f'{x:+.1f}%'

    def interval(x):
        return f'{x[0]:.1f}–{x[1]:.1f}%'

    ep = result['episode_summary']
    names = [
        ('bottom_sessions', '開始→期間内最安終値：営業日'),
        ('bottom_calendar_days', '開始→期間内最安終値：暦日'),
        ('bottom_return_pct', '開始→期間内最安終値：騰落率'),
        ('bottom_to_end_sessions', '期間内最安終値→最初の非ピンク日：営業日'),
        ('bottom_to_end_calendar_days', '期間内最安終値→最初の非ピンク日：暦日'),
        ('end_sessions', '開始→最初の非ピンク日：営業日'),
        ('end_calendar_days', '開始→最初の非ピンク日：暦日'),
        ('end_return_pct', '開始→最初の非ピンク日：騰落率'),
        ('last_pink_return_pct', '開始→最後のピンク日：騰落率（参考）'),
    ]
    duration_rows = ''
    for key, label in names:
        v = ep[key]
        unit = '%' if key.endswith('_pct') else '日'
        duration_rows += (
            f'<tr><td>{label}</td><td>{ep["n"]}</td><td>{v["mean"]:.1f}{unit}</td>'
            f'<td>{v["median"]:.1f}{unit}</td><td>{v["min"]:.1f}〜{v["max"]:.1f}{unit}</td></tr>'
        )
    duration_events = ''
    for e in result['events']:
        v = e.get('episode_outcome')
        if not v:
            continue
        duration_events += (
            f'<tr><td>{e["start"]}</td><td>{v["bottom_date"]}</td>'
            f'<td>{v["bottom_sessions"]} / {v["bottom_calendar_days"]}</td>'
            f'<td>{fmt(v["bottom_return_pct"])}</td><td>{e["end"]}</td>'
            f'<td>{v["end_sessions"]} / {v["end_calendar_days"]}</td>'
            f'<td>{fmt(v["end_return_pct"])}</td></tr>'
        )
    fs = result['fixed_126_bottom_summary']
    fixed_text = (
        f'開始後126営業日の固定窓内の最安終値で底を定義し直すと n={fs["n"]}。'
        f'底まで平均{fs["sessions"]["mean"]:.1f}・中央値{fs["sessions"]["median"]:.1f}営業日、'
        f'平均{fs["calendar_days"]["mean"]:.1f}・中央値{fs["calendar_days"]["median"]:.1f}暦日。'
        f'騰落率は平均{fmt(fs["return_pct"]["mean"])}・中央値{fmt(fs["return_pct"]["median"])}。'
        '期間外の底や窓の末日にまだ下落中の可能性を含むため、真の景気循環の底を保証しません。'
    )
    no_decline = ep['n'] - result['declining_episode_summary']['n']
    declines = result['declining_episode_summary']
    decline_text = (
        (
            f'参考として、実際に開始終値を下回った{declines["n"]}件だけでは、底まで'
            f'平均{declines["bottom_sessions"]["mean"]:.1f}・中央値{declines["bottom_sessions"]["median"]:.1f}営業日、'
            f'騰落率は平均{fmt(declines["bottom_return_pct"]["mean"])}・中央値{fmt(declines["bottom_return_pct"]["median"])}。'
            'これは下落を事後条件とする集計なので、今回の無条件の予測には使えません。'
        )
        if declines['n']
        else '開始終値を下回った事例はありません。'
    )
    current_low = c['lowest_close_to_date']
    comparison_section = '<h2>株価の底とBreadth底検出時の価格差</h2>'
    comparison_section += (
        f'<p>株価最安終値までの騰落率は完了済み全{ep["n"]}局面で'
        f'平均{fmt(ep["bottom_return_pct"]["mean"])}・中央値{fmt(ep["bottom_return_pct"]["median"])}。'
        '以下の差は、底が検出された同一局面だけで対応比較します。各平均・中央値を異なる標本間で引き算しません。</p>'
    )
    for mode, label in [('long', '200MAブレッドの底'), ('short', '短期8MAブレッド40%未満の底（参考）')]:
        comparison = result['breadth_bottom_comparison'][mode]
        stats = comparison['summary']
        comparison_section += f'<h3>{label}：検出あり{stats["n"]}件・なし{stats["missing"]}件</h3>'
        comparison_section += '<div class="scroll"><table><thead><tr><th>同じ局面で比較した指標</th><th>平均</th><th>中央値</th></tr></thead><tbody>'
        for key, name, unit in [
            ('price_bottom_return_pct', 'ピンク開始→株価最安終値', '%'),
            ('marker_return_pct', 'ピンク開始→Breadth底マーカー日', '%'),
            ('detection_return_pct', 'ピンク開始→初めて底を検出できた日', '%'),
            ('marker_gap_pp', 'マーカー日騰落率 − 株価底騰落率', 'pt'),
            ('detection_gap_pp', '検出日騰落率 − 株価底騰落率', 'pt'),
            ('price_change_bottom_to_detection_pct', '検出日価格の期間内最安値に対する差', '%'),
            ('marker_to_detection_sessions', 'マーカー日→検出日：営業日', '日'),
        ]:
            value = stats[key]
            means = f'{value["mean"]:+.1f}{unit}' if value else '対象なし'
            medians = f'{value["median"]:+.1f}{unit}' if value else '対象なし'
            comparison_section += f'<tr><td>{name}</td><td>{means}</td><td>{medians}</td></tr>'
        comparison_section += '</tbody></table></div><div class="scroll"><table><thead><tr><th>ピンク開始</th><th>株価底騰落率</th><th>Breadth底マーカー</th><th>初検出日</th><th>検出時騰落率</th><th>株価底との差</th></tr></thead><tbody>'
        for row in comparison['rows']:
            fields = (
                f'<td>{row["marker_date"]}</td><td>{row["detection_date"]}</td><td>{fmt(row["detection_return_pct"])}</td><td>{row["detection_gap_pp"]:+.1f}pt</td>'
                if row['detected']
                else '<td colspan="4">対象となる底検出なし</td>'
            )
            comparison_section += (
                f'<tr><td>{row["start"]}</td><td>{fmt(row["price_bottom_return_pct"])}</td>{fields}</tr>'
            )
        comparison_section += '</tbody></table></div>'
        comparison_section += (
            f'<p>期間内の最終的な株価底より先に検出された局面：'
            f'{stats["detection_before_price_bottom_count"]}/{stats["n"]}件。'
            '検出日価格と最安値の差は、底から順方向に実現できた利益を表すものではありません。</p>'
        )
    comparison_section += '<p>グラフと同じ算法を、各日までのデータだけで再実行しました。200MA系はfind_peaks（distance=50、prominence=0.015）、短期系は8MAが40%未満の観測を抽出してprominence=0.02。ピンク期間内を底日とする最初の検出を、次のピンク開始前まで追跡しています。後で消える・移動する候補も初検出として数え、最終履歴の底マーカーと一致を要求しません。現在の未終了局面は除外。これはグラフの底検出であり、閾値40%や追加条件を持つ売買ロジックの確定シグナルとは異なります。</p><p>差の単位ptは騰落率の差（パーセントポイント）。株価底から検出日までの実際の価格変化率も別に掲載。短期の底が後の株価最安値より先に検出される場合があり、「底検出＝株価の最終底打ち」とは限りません。</p>'
    duration_section = f"""<h2>ピンク開始から底・警戒終了まで</h2>
<p>対象は初期200観測を除く完了済み全{ep['n']}局面。「底」はピンク期間内（開始日を含む、最初の非ピンク日は含まない）の最安終値で、同値なら最初の日。将来を見て確定する事後統計です。「終了」は最初の非ピンク日の終値。期間は開始日を0日とした差で、開始日の翌営業日が1日です。</p>
<div class="scroll"><table><thead><tr><th>指標</th><th>n</th><th>平均</th><th>中央値</th><th>最小〜最大</th></tr></thead><tbody>{duration_rows}</tbody></table></div>
<p>開始日より安い終値がなかった局面も{no_decline}件含め、底まで0日・0%として数えます。騰落率は開始終値を基準とし、途中の高値からの最大ドローダウンとは異なります。短いピンク期間も含め、上記の「11営業日目に継続中」という条件はこの集計には適用しません。</p>
<p>{decline_text}</p>
<div class="scroll"><table><thead><tr><th>ピンク開始</th><th>期間内の底</th><th>底まで営業 / 暦日</th><th>底まで騰落率</th><th>最初の非ピンク日</th><th>終了まで営業 / 暦日</th><th>終了時騰落率</th></tr></thead><tbody>{duration_events}</tbody></table></div>
<p><strong>底の定義の感度分析：</strong>{fixed_text}</p>
<p><strong>今回（未終了）：</strong>現時点までの最安終値は{current_low['date']}、開始から{current_low['sessions']}営業日、騰落率{fmt(current_low['return_pct'])}。底打ち確定ではなく、完了済み局面の平均・中央値には含めません。</p>"""
    cards = ''.join(
        f'<article><h3>{label}</h3><b>{s["share_pct"]:.1f}%</b>'
        f'<p>{s["count"]}/{h63["n"]}件 · Wilson 95%区間 {interval(s["wilson"])}</p>'
        '<p>現在相当日から63営業日内の、終値による最大下落率で分類。</p></article>'
        for label, s in zip(labels, h63['scenarios'], strict=True)
    )
    rows = ''.join(
        f'<tr><td>{s["horizon"]}営業日</td><td>{s["n"]}</td>'
        f'<td>{fmt(s["return_quartiles"][1])}</td>'
        f'<td>{fmt(s["return_quartiles"][0])} ～ {fmt(s["return_quartiles"][2])}</td>'
        f'<td>{s["positive_count"]}/{s["n"]} ({100 * s["positive_count"] / s["n"]:.1f}%)</td>'
        f'<td>{fmt(s["worst_quartiles"][1])}</td></tr>'
        for s in result['primary']
    )
    event_rows = ''
    for e in result['events']:
        f = e.get('forward_63')
        metrics = (
            f'<td>{fmt(f["return_pct"])}</td><td>{fmt(f["worst_close_pct"])}</td>'
            if f
            else '<td>未観測</td><td>未観測</td>'
        )
        event_rows += (
            f'<tr><td>{e["start"]}</td><td>{e["end"] or "継続中"}</td>'
            f'<td>{e["pink_sessions"]}</td><td>{"対象" if e["same_age_eligible"] else "終了済み・除外"}</td>'
            f'<td>{e["anchor_long_pct"]:.1f}% / {e["anchor_short_pct"]:.1f}%</td>{metrics}</tr>'
        )
    fig = go.Figure()
    paths = []
    a = len(d) - 1
    for e in result['events']:
        if not e['same_age_eligible'] or e['anchor_index'] + 63 >= len(d) - c['observed_sessions']:
            continue
        j = e['anchor_index']
        p = (d[PRICE].iloc[j : j + 64].to_numpy() / d[PRICE].iloc[j] - 1) * 100
        paths.append(p)
        fig.add_trace(
            go.Scatter(
                x=list(range(64)),
                y=p,
                name=e['start'],
                opacity=0.45,
                hovertemplate='%{x}営業日後: %{y:.1f}%<extra>%{fullData.name}</extra>',
            )
        )
    if paths:
        q = np.percentile(paths, [25, 50, 75], axis=0)
        fig.add_trace(go.Scatter(x=list(range(64)), y=q[0], line={'width': 0}, showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=list(range(64)),
                y=q[2],
                fill='tonexty',
                fillcolor='rgba(255,170,200,.18)',
                line={'width': 0},
                name='25–75%分位帯',
            )
        )
        fig.add_trace(go.Scatter(x=list(range(64)), y=q[1], line={'width': 4, 'color': '#ffb3cd'}, name='各日の中央値'))
    fig.update_layout(
        template='plotly_dark',
        height=480,
        title='現在相当日を0%とした過去事例の終値推移',
        xaxis_title='現在相当日からの営業日数',
        yaxis_title='価格変化 (%)',
        legend={'orientation': 'h'},
        margin={'l': 55, 'r': 20, 't': 60, 'b': 100},
    )
    plot = fig.to_html(full_html=False, include_plotlyjs=True, config={'responsive': True})
    history = go.Figure()
    recent = d.iloc[max(0, a - 125) :]
    for col, name in [(LONG, '200MAブレッド（EMA200）'), (SHORT, '8MAブレッド（EMA8）')]:
        history.add_trace(go.Scatter(x=recent.Date, y=recent[col] * 100, name=name))
    history.add_vrect(x0=c['start'], x1=c['as_of'], fillcolor='pink', opacity=0.15, line_width=0)
    history.update_layout(
        template='plotly_dark',
        height=350,
        title='直近のブレッド：ピンクは現在の警戒期間',
        yaxis_title='構成銘柄比率の平滑値 (%)',
        legend={'orientation': 'h'},
    )
    plot += history.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    limitations = [
        '頻度は過去標本での記述統計であり、今回の確率を校正した予測モデルではありません。価格上昇事例も除外していません。',
        '全履歴を一貫して取得できる公開CSV（2016年開始）だけを使用。既存版の2007年以降の手作業データとは混ぜず、初期200観測のシグナルを除外。',
        '各過去局面も11営業日目にピンクが継続していたものに限定。経過日数が異なる事例との比較や、未来を使うIs_Peak/Is_Troughの選別は行いません。',
        'シグナル終了を超えても価格を追跡し、63営業日の固定窓を用います。下落基準は現在相当日の終値。将来のピークから測る下落や日中安値とは異なります。',
        '未来の底値・反転日を予測しません。終値の最小値を後から集計しており、当日に認識できる売買シグナルとは異なります。',
        '標本が小さく、同じ危機内の複数シグナル・重複する将来窓に依存があります。Wilson区間は独立標本を仮定した参考幅。126営業日以内の再発を除外した感度分析も併記。',
        '過去時点の指数構成銘柄を保証しない元のブレッド計算には生存者バイアスや欠損分母の影響があり得ます。構成銘柄・価格原票の独立照合は今回実施していません。',
        'S&P500_Price列はコード上SPY由来の価格系列。指数そのものの水準ではありません。調整価格の仕様・配当再投資の完全性はCSVだけでは保証できません。',
        '200MAは「株価が200日MAを超える銘柄割合」をEMA200で平滑化したもの。価格の200日MA割れという条件とは異なります。EMA初期化と移動する公開履歴の影響にも注意。',
        'データ締切後のニュースやマクロ要因、取引コスト、売買判断を含まないイベント研究です。',
    ]
    sensitivity_text = ' / '.join(
        f'{label[0]} {s["count"]}/{sens["n"]}={s["share_pct"]:.1f}%'
        for label, s in zip(labels, sens['scenarios'], strict=True)
    )
    css = 'body{background:#10141f;color:#e4e9f2;font:16px/1.8 system-ui,sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:28px}h1{font-size:28px}h2{margin-top:36px;color:#ffbad4}a{color:#8ecbff}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article,.note{background:#1c2333;border:1px solid #39465c;border-radius:12px;padding:18px}article b{font-size:32px;color:#ffbad4}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #39465c;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}.muted{color:#abb6ca}li{margin-bottom:10px}@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:16px}}'
    content = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>200MAブレッド警戒局面：更新分析 {c['as_of']}</title><style>{css}</style></head><body><main>
<h1>200MAブレッドがピンクになった後のシナリオ</h1>
<p class="muted">データ締切 {c['as_of']} · 発生日 {c['start']} · 現在 {c['observed_sessions']}営業日目 · 同経過日数の過去事例 n={h63['n']}</p>
<div class="note"><strong>結論：</strong>ブレッドの悪化は確認できますが、価格が必ず下がるわけではありません。過去の同条件では、今後63営業日の終値下落が5%未満だった事例は {h63['scenarios'][0]['count']}/{h63['n']}件、10%以上の下落は {h63['scenarios'][2]['count']}/{h63['n']}件。点推定より、標本数と区間の広さを重視してください。</div>
<h2>現在地</h2><p>EMA200ブレッド {c['long_pct']:.1f}% ／ EMA8 {c['short_pct']:.1f}% ／ 生の200日MA超え比率 {c['raw_pct']:.1f}%。短期線は長期線を下回り、長期線のヒステリシス判定は下降中です。50日MA系の短期平滑値は {c.get('short_50_pct', float('nan')):.1f}%です。</p>
<p>SPY由来の価格 {c['price']:.2f}、ピンク開始日から {fmt(c['return_since_start'])}、直近63営業日の最高終値から {fmt(c['drawdown_63'])}。これは観測済みの変化であり、将来の価格目標ではありません。</p>
<h2>今後63営業日の3シナリオ</h2><div class="grid">{cards}</div>
<p>3分類は相互排他的で全事例をカバーします。Aも上昇を保証せず、B/Cでも最終的なリターンは正になることがあります。閾値5%・10%は結果に合わせて最適化していない記述上の区分です。</p>
<h2>期間別の結果</h2><div class="scroll"><table><thead><tr><th>今後の期間</th><th>n</th><th>終点リターン中央値</th><th>25–75%分位</th><th>終点プラス</th><th>途中の最悪終値・中央値</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>5/21/63/126/252営業日は概ね1週/1か月/3か月/6か月/12か月。63営業日は比較用の3か月窓で、警戒期間全体の長さを表しません。四分位範囲は観測値の分散で、中央値の信頼区間ではありません。終点の上昇率と途中の下落率を分けて読むことが重要です。</p>
{duration_section}
{comparison_section}
<h2>経路を見る</h2>{plot}<p>帯は各営業日の25–75%分位で、将来の予測区間ではありません。実際の観測経路は凡例をクリックして比較できます。</p>
<h2>感度分析・今回の見方</h2><p>同じ警戒局面の再発を126営業日間隔で間引くと n={sens['n']}：{sensitivity_text}。主分析と結論が変わる場合は、危機内の事例の数え方に依存していると解釈します。</p>
<p>{duration_text}</p><p>現在のSPY由来価格に当てはめた下落幅の目安は、−5%が{c['price'] * 0.95:.2f}、−10%が{c['price'] * 0.9:.2f}。これは分類境界の換算値で、目標価格ではありません。</p>
<ul><li><strong>Aの確認材料：</strong>短期ブレッドが上向き、長期線との差が縮小するか。価格だけの反発と参加銘柄の回復を区別する。</li><li><strong>B/Cの確認材料：</strong>短期ブレッドの再低下と価格安値更新が同時に進むか。8MAが40%/30%を下回ることは監視用の記述であり、今回の研究では条件付き確率を推定していない。</li><li><strong>警戒解除：</strong>元の2条件のどちらかが外れればピンク表示は終了。ただし色の終了が価格の底打ちを保証するわけではない。</li></ul>
<h2>過去局面の全一覧</h2><div class="scroll"><table><thead><tr><th>開始日</th><th>最初の非ピンク日</th><th>ピンク営業日数</th><th>同経過日数</th><th>当時の長期 / 短期ブレッド</th><th>その後63日リターン</th><th>その後63日最悪終値</th></tr></thead><tbody>{event_rows}</tbody></table></div><p>条件と経過日数を揃えた比較であり、ブレッド水準・価格トレンド・ボラティリティまで一致する類似事例の確率ではありません。標本9件に対する追加フィルタや類似度重みは過学習を避けるため採用していません。</p>
<h2>旧レポートからの変更</h2><p>旧版は2026年3月12日開始局面の3月26日時点を扱い、価格上昇事例を除外した12件に類似度の重みを付けていました。今回の9月局面ではそれらの確率・底値目標を引き継ぎません。新たな入力と明示的な対象条件で再集計し、3月局面は観測済みの過去事例として含めています。</p>
<h2>方法と限界</h2><ul>{''.join('<li>' + html.escape(x) + '</li>' for x in limitations)}</ul>
<p>ピンク条件：Breadth_200MA_Trend == -1 かつ Breadth_Index_8MA &lt; Breadth_Index_200MA。価格下落の有無による標本選別はなし。現在と同じ営業日オフセットでピンク継続中の局面を比較し、現在の局面開始前までに完全観測できる将来窓のみを集計。</p>
<p>データ：<a href="{SOURCE}">公開CSV</a> ／ <a href="https://tradermonty.github.io/market-breadth-analysis/market_breadth.html">日次チャート</a>。履歴開始 {result['history_start']}。入力SHA-256：<code>{result['input_sha256']}</code>。</p>
</main></body></html>'''
    (output_dir / 'downtrend_forecast.html').write_text(content)
    (output_dir / 'downtrend_forecast_statistics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    # Snapshot accompanies the report so public history changes cannot silently alter reproduction.
    (output_dir / 'downtrend_forecast_input.csv').write_bytes(raw)
    pd.DataFrame(
        [
            {'pink_start': e['start'], 'first_non_pink_date': e['end'], **e['episode_outcome']}
            for e in result['events']
            if 'episode_outcome' in e
        ]
    ).to_csv(output_dir / 'downtrend_forecast_episodes.csv', index=False)
    pd.DataFrame(
        [
            {'breadth_mode': mode, **row}
            for mode, comparison in result['breadth_bottom_comparison'].items()
            for row in comparison['rows']
        ]
    ).to_csv(output_dir / 'downtrend_forecast_bottom_comparison.csv', index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    args = parser.parse_args()
    build(args.input, args.output_dir)
