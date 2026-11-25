"""Tests for refreshing the long-history research snapshot to a new cutoff date."""

import json

import numpy as np
import pandas as pd

from scripts.build_downtrend_forecast import PRICE, build
from scripts.fetch_long_history import fetch_history
from scripts.prepare_long_history import prepare
from scripts.refresh_long_history_metadata import build_universe, membership_audit
from scripts.render_downtrend_forecast_english import (
    model_onset_sentence,
    previous_publication_paragraph,
    render,
    sessions_text,
)
from tests.test_downtrend_forecast import fixture


def _write_minimal_cache(directory, changes):
    dates = pd.bdate_range('1990-01-01', '1994-01-31')
    (directory / 'prices').mkdir()
    for symbol in ('A', 'NEW'):
        frame = pd.DataFrame({'date': dates, 'adjClose': np.arange(len(dates)) + 100.0})
        frame.to_csv(directory / 'prices' / f'{symbol}.csv.gz', index=False, compression='gzip')
    spy = [{'date': str(date.date()), 'adjClose': 100.0} for date in dates[dates >= '1993-01-04']]
    for filename, payload in [
        ('acquisition_manifest.json', [{'symbol': 'A', 'usable': True}, {'symbol': 'NEW', 'usable': True}]),
        ('current_constituents.json', [{'symbol': 'A'}, {'symbol': 'NEW'}]),
        ('historical_constituents.json', changes),
        ('spy_legacy.json', {'historical': spy}),
        ('universe.json', ['A', 'NEW', 'OLD']),
    ]:
        (directory / filename).write_text(json.dumps(payload))


def test_01_prepare_applies_membership_changes_only_up_to_end_date(tmp_path):
    change = [{'date': '1993-06-01', 'symbol': 'NEW', 'removedTicker': 'OLD'}]
    _write_minimal_cache(tmp_path, change)
    prepare(tmp_path, end='1993-05-31')
    historical = pd.read_csv(tmp_path / 'historical_reconstructed.csv').set_index('Date')
    # The change is after the cutoff, so NEW is treated as a member for the whole history.
    assert historical.loc['1993-01-04', 'Constituent_Count'] == 2
    assert historical.loc['1993-01-04', 'Observed_Count'] == 2


def test_02_prepare_default_end_preserves_original_snapshot(tmp_path):
    change = [{'date': '1993-06-01', 'symbol': 'NEW', 'removedTicker': 'OLD'}]
    _write_minimal_cache(tmp_path, change)
    prepare(tmp_path)
    historical = pd.read_csv(tmp_path / 'historical_reconstructed.csv').set_index('Date')
    assert historical.loc['1993-01-04', 'Observed_Count'] == 1


def test_03_fetch_history_requests_the_given_end_date(tmp_path, monkeypatch):
    (tmp_path / 'universe.json').write_text(json.dumps(['AAA']))
    env = tmp_path / '.env'
    env.write_text('FMP_API_KEY=dummy\n')
    seen = []

    class Response:
        status_code = 200
        content = b'{}'

        def json(self):
            return {'historical': [{'date': '2026-11-24', 'close': 10.0, 'adjClose': 10.0}]}

    def fake_get(url, params, timeout):
        seen.append(params)
        return Response()

    monkeypatch.setattr('scripts.fetch_long_history.requests.get', fake_get)
    fetch_history(tmp_path, env, end='2026-11-24')
    assert seen[0]['to'] == '2026-11-24'
    assert seen[0]['from'] == '1990-01-01'


def test_04_universe_keeps_previous_symbols_and_adds_post_cutoff_changes():
    previous = ['A', 'B', 'OLD']
    current = [{'symbol': 'A'}, {'symbol': 'NEW'}]
    changes = [
        {'date': '2026-11-10', 'symbol': 'NEW', 'removedTicker': 'B'},
        {'date': '2026-10-15', 'symbol': 'MID', 'removedTicker': 'GONE'},
        {'date': '2020-01-01', 'symbol': 'IGNORED_EARLY', 'removedTicker': 'X'},
        {'date': '2026-12-01', 'symbol': 'FUTURE', 'removedTicker': ''},
    ]
    universe = build_universe(previous, current, changes, previous_end='2026-10-02', end='2026-11-24')
    assert universe == sorted({'A', 'B', 'OLD', 'NEW', 'MID', 'GONE'})


def test_05_membership_audit_reports_sizes_and_reverse_history_anomalies():
    current = [{'symbol': 'A'}, {'symbol': 'NEW'}]
    changes = [
        {'date': '2026-11-10', 'symbol': 'NEW', 'removedTicker': 'B'},
        {'date': '2026-11-01', 'symbol': 'GHOST', 'removedTicker': ''},
        {'date': '2026-12-01', 'symbol': 'FUTURE', 'removedTicker': ''},
    ]
    audit = membership_audit(current, changes, end='2026-11-24')
    assert audit['sizes'] == [['2026-11-10', 2], ['2026-11-01', 2]]
    assert audit['anomalies'] == [['2026-11-01', ['GHOST']]]


def _rendered_page(tmp_path, previous=None):
    path = tmp_path / 'input.csv'
    fixture().to_csv(path, index=False)
    build(path, tmp_path / 'out', exclude_chattering=True)
    output = tmp_path / 'page.html'
    render(tmp_path / 'out' / 'downtrend_forecast_statistics.json', path, output, previous=previous)
    return output.read_text()


def test_06_render_narrative_uses_current_episode_age(tmp_path):
    page = _rendered_page(tmp_path)
    # Fixture: current onset at row 680 of 700 -> session 20, age offset 19 -> 19 + 63 = 82.
    assert 'about 82 elapsed sessions' in page
    assert 'on session 20' in page
    assert 'session 11' not in page
    assert 'September' not in page


def test_07_render_describes_previous_publication_from_data(tmp_path):
    previous = {'as_of': '2001-08-31', 'start': str(pd.bdate_range('2000-01-03', periods=700)[400].date())}
    page = _rendered_page(tmp_path, previous=previous)
    end = str(pd.bdate_range('2000-01-03', periods=700)[430].date())
    assert 'data through 2001-08-31' in page
    assert f'ended on {end}' in page


def test_08_model_onset_sentence_lists_each_model_onset():
    sentence = model_onset_sentence(
        {'current_fixed': '2026-11-19', 'current_eligible': '2026-11-18', 'historical_reconstructed': '2026-11-19'}
    )
    assert '2026-11-19' in sentence
    assert 'MA-eligible current-cohort variant: 2026-11-18' in sentence


def test_09_previous_publication_lists_onset_dates_changed_by_cohort(tmp_path):
    dates = pd.bdate_range('2000-01-03', periods=700)
    previous = {
        'as_of': '2001-08-31',
        'start': str(dates[400].date()),
        'current_count': 3,
        'onsets': [str(dates[221].date()), str(dates[400].date())],
    }
    page = _rendered_page(tmp_path, previous=previous)
    assert 'had 3 current constituents' in page
    assert f'only in the previous snapshot: {dates[221].date()}' in page
    assert f'only in this snapshot: {dates[220].date()}' in page


def test_10_previous_snapshot_reads_reference_model_of_old_cache(tmp_path):
    from scripts.publish_long_history_report import previous_snapshot

    fixture().to_csv(tmp_path / 'current_fixed.csv', index=False)
    (tmp_path / 'current_constituents.json').write_text(json.dumps([{'symbol': 'A'}, {'symbol': 'B'}]))
    dates = pd.bdate_range('2000-01-03', periods=700)
    snapshot = previous_snapshot(tmp_path)
    assert snapshot['as_of'] == str(dates[-1].date())
    assert snapshot['start'] == str(dates[680].date())
    assert snapshot['onsets'] == [str(dates[i].date()) for i in (220, 400, 680)]
    assert snapshot['current_count'] == 2


def test_11_previous_paragraph_reports_decline_after_warning_end(tmp_path):
    dates = pd.bdate_range('2000-01-03', periods=700)
    path = tmp_path / 'input.csv'
    frame = fixture()
    frame.loc[500, PRICE] = 80.0  # after the 400-430 episode ended, before the current onset
    frame.to_csv(path, index=False)
    build(path, tmp_path / 'out', exclude_chattering=True)
    output = tmp_path / 'page.html'
    previous = {'as_of': '2001-08-31', 'start': str(dates[400].date())}
    render(tmp_path / 'out' / 'downtrend_forecast_statistics.json', path, output, previous=previous)
    page = output.read_text()
    assert f'After the warning ended, the lowest close was on {dates[500].date()}, -20.0% from that onset' in page


def test_12_ongoing_previous_episode_is_not_listed_as_missing():
    outcome = {'bottom_date': '2001-01-05', 'bottom_return_pct': -1.0}
    events = [{'start': '2001-01-02', 'end': '2001-02-01', 'episode_outcome': outcome}]
    previous = {'as_of': '2001-03-30', 'start': '2001-03-01', 'onsets': ['2001-01-02', '2001-03-01']}
    text = previous_publication_paragraph(previous, events, current_start='2001-03-01')
    assert 'only in the previous snapshot: none' in text


def test_13_prepare_without_end_date_uses_metadata_manifest(tmp_path):
    change = [{'date': '1993-06-01', 'symbol': 'NEW', 'removedTicker': 'OLD'}]
    _write_minimal_cache(tmp_path, change)
    (tmp_path / 'metadata_manifest.json').write_text(json.dumps({'end': '1993-05-31'}))
    prepare(tmp_path, end=None)
    historical = pd.read_csv(tmp_path / 'historical_reconstructed.csv').set_index('Date')
    assert historical.loc['1993-01-04', 'Observed_Count'] == 2


def test_14_session_count_is_singular_for_one():
    assert sessions_text(1) == '1 session'
    assert sessions_text(3) == '3 sessions'


def test_15_metadata_request_errors_do_not_leak_api_key(monkeypatch):
    import pytest
    import requests

    from scripts.refresh_long_history_metadata import _get

    def boom(url, params, timeout):
        raise requests.ConnectionError(f'failed: {url}?apikey={params["apikey"]}')

    monkeypatch.setattr('scripts.refresh_long_history_metadata.requests.get', boom)
    with pytest.raises(RuntimeError) as error:
        _get('sp500_constituent', 'SECRET123')
    assert 'SECRET123' not in str(error.value)
    assert error.value.__cause__ is None and error.value.__suppress_context__
