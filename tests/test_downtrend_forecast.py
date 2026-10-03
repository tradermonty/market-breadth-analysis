import numpy as np
import pandas as pd
import pytest

from scripts.build_downtrend_forecast import LONG, PRICE, SHORT, analyze, episode_outcome, wilson


def fixture():
    n = 700
    mask = np.zeros(n, dtype=bool)
    mask[220:250] = True
    mask[400:430] = True
    mask[680:] = True
    return pd.DataFrame(
        {
            'Date': pd.bdate_range('2000-01-03', periods=n),
            PRICE: np.full(n, 100.0),
            LONG: 0.6,
            SHORT: np.where(mask, 0.5, 0.7),
            'Breadth_200MA_Trend': -1,
            'Bearish_Signal': mask,
            'Breadth_Index_Raw': 0.6,
        }
    )


@pytest.mark.parametrize('low,category', [(96, 0), (95, 1), (90, 2), (80, 2)])
def test_scenario_boundaries_and_recovery_are_distinct(low, category):
    d = fixture()
    d.loc[250, PRICE] = low
    _, result = analyze(d)
    h = next(x for x in result['primary'] if x['horizon'] == 63)
    assert h['n'] == 2
    assert h['scenarios'][category]['count'] >= 1
    assert h['rows'][0]['return_pct'] == 0
    assert h['rows'][0]['worst_close_pct'] == low - 100


def test_episode_eligibility_and_future_cutoff():
    d = fixture()
    # First historical signal ends before the current same-age anchor.
    d.loc[230:249, SHORT] = 0.7
    d.loc[230:249, 'Bearish_Signal'] = False
    # Add a historical episode whose 63-session future overlaps the current one.
    d.loc[630:665, SHORT] = 0.5
    d.loc[630:665, 'Bearish_Signal'] = True
    _, result = analyze(d)
    assert result['events'][0]['same_age_eligible'] is False
    h = next(x for x in result['primary'] if x['horizon'] == 63)
    assert [x['start'] for x in h['rows']] == [str(d.Date.iloc[400].date())]


@pytest.mark.parametrize('fault', ['duplicate', 'mask', 'nan', 'negative'])
def test_invalid_input_is_rejected(fault):
    d = fixture()
    if fault == 'duplicate':
        d.loc[1, 'Date'] = d.loc[0, 'Date']
    elif fault == 'mask':
        d.loc[0, 'Bearish_Signal'] = True
    elif fault == 'nan':
        d.loc[0, PRICE] = np.nan
    else:
        d.loc[0, PRICE] = -1
    with pytest.raises(ValueError):
        analyze(d)


def test_wilson_and_no_price_outcome_selection():
    assert wilson(0, 0) == [None, None]
    assert wilson(0, 9)[1] > 0
    assert wilson(9, 9)[0] < 100
    d = fixture()
    d.loc[240:350, PRICE] = 110
    _, result = analyze(d)
    assert len(result['events']) == 2
    assert result['primary'][2]['n'] == 2
    assert result['primary'][2]['positive_count'] == 1


def test_bottom_uses_first_equal_low_and_excludes_exit_day():
    d = fixture()
    d.loc[225, PRICE] = 90
    d.loc[228, PRICE] = 90
    d.loc[249, PRICE] = 105
    d.loc[250, PRICE] = 80
    v = episode_outcome(d, 220, 250)
    assert v['bottom_date'] == str(d.Date.iloc[225].date())
    assert v['bottom_sessions'] == 5
    assert v['bottom_calendar_days'] == (d.Date.iloc[225] - d.Date.iloc[220]).days
    assert v['bottom_return_pct'] == pytest.approx(-10)
    assert v['end_sessions'] == 30
    assert v['bottom_to_end_sessions'] == 25
    assert v['bottom_to_end_calendar_days'] == (d.Date.iloc[250] - d.Date.iloc[225]).days
    assert v['bottom_sessions'] + v['bottom_to_end_sessions'] == v['end_sessions']
    assert v['end_return_pct'] == pytest.approx(-20)
    assert v['last_pink_return_pct'] == pytest.approx(5)


def test_completed_summary_includes_zero_decline_but_excludes_open_episode():
    d = fixture()
    d.loc[225, PRICE] = 90
    # The open episode's severe loss must not contaminate completed-episode statistics.
    d.loc[690, PRICE] = 1
    _, r = analyze(d)
    s = r['episode_summary']
    assert s['n'] == 2
    assert s['bottom_return_pct']['mean'] == pytest.approx(-5)
    assert s['bottom_return_pct']['median'] == pytest.approx(-5)
    assert s['bottom_sessions']['mean'] == 2.5
    assert r['declining_episode_summary']['n'] == 1
    assert r['current']['lowest_close_to_date']['return_pct'] == pytest.approx(-99)


def test_short_finished_episode_is_in_start_based_summary():
    d = fixture()
    d.loc[230:249, SHORT] = 0.7
    d.loc[230:249, 'Bearish_Signal'] = False
    _, r = analyze(d)
    assert r['events'][0]['same_age_eligible'] is False
    assert r['episode_summary']['n'] == 2
    assert r['episode_summary']['end_sessions']['mean'] == 20


def test_detection_is_priced_on_confirmation_and_survives_future_revision():
    d = fixture()
    d.loc[220:249, SHORT] = 0.1
    d.loc[225, LONG] = 0.3
    d.loc[226, LONG] = 0.32
    d.loc[225, PRICE] = 90
    _, result = analyze(d)
    row = result['breadth_bottom_comparison']['long']['rows'][0]
    assert row['marker_date'] == str(d.Date.iloc[225].date())
    assert row['detection_date'] == str(d.Date.iloc[226].date())
    assert row['marker_return_pct'] == pytest.approx(-10)
    assert row['detection_return_pct'] == 0
    assert row['detection_gap_pp'] == pytest.approx(10)
    assert row['price_change_bottom_to_detection_pct'] == pytest.approx(100 / 9)
    # A later deeper trough must not change the first historical detection date.
    d.loc[240, LONG] = 0.2
    d.loc[241, LONG] = 0.25
    _, later = analyze(d)
    assert later['breadth_bottom_comparison']['long']['rows'][0]['detection_date'] == row['detection_date']


def test_missing_breadth_bottom_is_not_zero_return_detection():
    _, result = analyze(fixture())
    summary = result['breadth_bottom_comparison']['long']['summary']
    assert summary['n'] == 0
    assert summary['missing'] == 2
    assert summary['detection_return_pct'] is None


def test_chattering_filter_recomputes_every_cohort_and_preserves_detection_limits():
    d = fixture()
    for start in (280, 288, 296):
        d.loc[start : start + 2, SHORT] = 0.5
        d.loc[start : start + 2, 'Bearish_Signal'] = True
    _, original = analyze(d)
    _, filtered = analyze(d, exclude_chattering=True)
    excluded = [str(d.Date.iloc[start].date()) for start in (280, 288, 296)]
    assert filtered['chattering_filter']['excluded_starts'] == excluded
    assert original['episode_summary']['n'] == 5
    assert filtered['episode_summary']['n'] == 2
    assert filtered['fixed_126_bottom_summary']['n'] == 2
    assert filtered['events'][0]['next_start_index'] == 280
    assert filtered['events'][0]['episode_outcome'] == original['events'][0]['episode_outcome']
    for mode in ('long', 'short'):
        assert len(filtered['breadth_bottom_comparison'][mode]['rows']) == 2
    for horizon in filtered['primary']:
        assert all(row['start'] not in excluded for row in horizon['rows'])
