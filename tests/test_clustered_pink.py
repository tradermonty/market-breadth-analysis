import pandas as pd
import pytest

from scripts.analyze_clustered_pink import analyze, merge_intervals


def test_merge_gap_boundary_and_transitive_chain():
    intervals = [(200, 210), (215, 225), (230, 240), (246, 250)]
    merged = merge_intervals(intervals, 5)
    assert [(row['start'], row['end']) for row in merged] == [(200, 240), (246, 250)]
    assert len(merged[0]['members']) == 3
    assert len(merge_intervals(intervals, 4)) == 4
    assert len(merge_intervals([(200, 210), (211, 220)], 0)) == 2


@pytest.mark.parametrize('intervals,gap', [([(10, 20), (15, 30)], 5), ([(10, 10)], 5), ([], -1)])
def test_invalid_intervals(intervals, gap):
    with pytest.raises(ValueError):
        merge_intervals(intervals, gap)


def fixture_frame():
    frame = pd.DataFrame(
        {'Date': pd.bdate_range('2000-01-03', periods=300), 'S&P500_Price': 100.0, 'Bearish_Signal': False}
    )
    frame.loc[200:213, 'Bearish_Signal'] = True
    frame.loc[216:229, 'Bearish_Signal'] = True
    frame.loc[290:299, 'Bearish_Signal'] = True
    return frame


def test_bridged_low_first_onset_fixed_and_original_baseline_preserved():
    frame = fixture_frame()
    frame.loc[215, 'S&P500_Price'] = 80.0
    original = analyze(frame, 0)
    merged = analyze(frame, 2)
    assert original['cohorts']['all']['n'] == 2
    assert merged['cohorts']['all']['n'] == 1
    row = merged['rows'][0]
    assert row['start_index'] == 200
    assert row['subepisodes'] == 2
    assert row['low_sessions'] == 15
    assert row['low_return_pct'] == pytest.approx(-20)
    assert original['rows'][0]['low_return_pct'] == 0
    assert row['end_sessions'] == 30
    assert row['low_sessions'] + row['low_to_end_sessions'] == row['end_sessions']


def test_anchor_in_nonpink_gap_is_not_current_condition_match():
    frame = fixture_frame()
    frame.loc[200:229, 'Bearish_Signal'] = False
    frame.loc[200:207, 'Bearish_Signal'] = True
    frame.loc[211:229, 'Bearish_Signal'] = True
    result = analyze(frame, 3)
    assert result['rows'][0]['anchor_index'] == 209
    assert result['cohorts']['high_onset']['n'] == 1
    assert result['cohorts']['high_onset_same_age']['n'] == 0


def test_open_cluster_changes_current_age_and_has_no_outcome():
    frame = fixture_frame()
    frame.loc[280:284, 'Bearish_Signal'] = True
    result = analyze(frame, 5)
    assert result['current_start'] == str(frame['Date'].iloc[280].date())
    assert result['current_sessions'] == 20
    assert all(row['start_index'] < 280 for row in result['rows'])


def test_high_price_restart_cannot_qualify_low_price_first_onset():
    frame = fixture_frame()
    frame.loc[200, 'S&P500_Price'] = 90.0
    result = analyze(frame, 2)
    assert result['rows'][0]['subepisodes'] == 2
    assert result['cohorts']['high_onset']['n'] == 0
