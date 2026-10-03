import pandas as pd

from scripts.analyze_high_price_pink import chattering_starts


def make_events(intervals):
    dates = pd.bdate_range('2000-01-03', periods=400).strftime('%Y-%m-%d')
    return [{'start_index': start, 'end': dates[end]} for start, end in intervals], dates


def test_only_short_repeated_chain_is_removed():
    events, dates = make_events([(200, 205), (210, 215), (220, 225), (260, 280)])
    removed, groups = chattering_starts(events, dates)
    assert removed == {200, 210, 220}
    assert len(groups) == 1
    assert events[-1]['start_index'] not in removed


def test_two_segments_and_long_clusters_remain():
    for intervals in [[(200, 205), (210, 215)], [(200, 205), (210, 215), (220, 250)]]:
        events, dates = make_events(intervals)
        assert chattering_starts(events, dates)[0] == set()


def test_gap_and_span_boundaries():
    events, dates = make_events([(200, 205), (210, 215), (220, 240)])
    assert chattering_starts(events, dates)[0] == {200, 210, 220}
    assert chattering_starts(events, dates, max_span=39)[0] == set()
    assert chattering_starts(events, dates, max_gap=4)[0] == set()
