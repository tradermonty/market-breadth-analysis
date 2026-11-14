"""Tests for Issue #22: independent feed monitoring with deduplicated incident issues.

Every HTTP call goes through a fake client and sockets are blocked, so no test
reaches the network, creates an issue or dispatches a workflow.
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import monitor_feed as mf  # noqa: E402

pytest.importorskip('exchange_calendars')

REPO = 'tradermonty/market-breadth-analysis'
API = f'https://api.github.com/repos/{REPO}'
WORKFLOW_URL = f'{API}/actions/workflows/daily-market-breadth.yml'
RUNS_URL = f'{WORKFLOW_URL}/runs'
ISSUES_URL = f'{API}/issues'
TOKEN = 'ghs_dummyMonitorToken1234567890'  # pragma: allowlist secret
BOT = {'login': 'github-actions[bot]', 'type': 'Bot'}

# 2026-11-13 is a Friday; 2026-11-12 closes 21:00 UTC.
NOW = datetime(2026, 11, 13, 23, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def block_sockets(monkeypatch):
    def _deny(*args, **kwargs):
        raise AssertionError('network access attempted')

    monkeypatch.setattr('socket.socket.connect', _deny)
    monkeypatch.setattr('socket.create_connection', _deny)


# ---------- fixtures ----------


def data_csv(last_date, coverage_200=0.99, coverage_50=0.99, extra_dates=(), include_coverage=True):
    header = 'Date,S&P500_Price,Breadth_Index_Raw,Breadth_Index_200MA'
    if include_coverage:
        header += ',Coverage_200,Coverage_50'
    lines = [header]
    for day in [*extra_dates, last_date]:
        row = f'{day},5800.0,0.65,0.60'
        if include_coverage:
            row += f',{coverage_200},{coverage_50}'
        lines.append(row)
    return '\n'.join(lines) + '\n'


def summary_csv(end_date):
    return f'Metric,Value\nAnalysis Period Start,2026-01-02\nAnalysis Period End,{end_date}\n'


def provenance(selected, confidence='observed', eligible=True):
    return json.dumps(
        {
            'selected_date': selected,
            'latest_completed_date': selected,
            'expected_date': selected,
            'source': 'fmp_spy_daily',
            'confidence': confidence,
            'publication_eligible': eligible,
            'calendar_name': 'XNYS',
        }
    )


def run(conclusion, n=1, branch='main'):
    return {
        'id': n,
        'status': 'completed',
        'conclusion': conclusion,
        'head_branch': branch,
        'created_at': '2026-11-13T02:00:00Z',
        'html_url': f'https://github.com/{REPO}/actions/runs/{n}',
    }


def state_block(fingerprint, last_notified_at):
    payload = json.dumps({'fingerprint': fingerprint, 'last_notified_at': last_notified_at})
    return f'<!-- feed-monitor:state {payload} -->'


def incident_issue(number, fingerprint='feed_stale', notified='2026-11-13T00:00:00+00:00', user=BOT, created=None):
    return {
        'number': number,
        'title': f'Feed monitor incident: {fingerprint}',
        'body': f'Incident\n\n{state_block(fingerprint, notified)}\n',
        'user': user,
        'created_at': created or f'2026-11-0{number % 9 + 1}T00:00:00Z',
        'html_url': f'https://github.com/{REPO}/issues/{number}',
    }


class Resp:
    def __init__(self, status=200, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError('no json')
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error', response=self)


class FakeHttp:
    """Routes requests by method and URL; a route value may be a list consumed per call."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, *, headers=None, params=None, json=None, timeout=None):
        self.calls.append({'method': method, 'url': url, 'headers': headers or {}, 'params': params, 'json': json})
        key = (method, url)
        if key not in self.routes:
            raise AssertionError(f'unexpected request {method} {url}')
        value = self.routes[key]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if callable(value):
            value = value(params=params, json=json)
        if isinstance(value, Exception):
            raise value
        return value

    def writes(self):
        return [c for c in self.calls if c['method'] != 'GET']


def routes(
    *,
    latest='2026-11-12',
    csv=None,
    summary=None,
    prov=None,
    state='active',
    runs=None,
    issues=None,
):
    return {
        ('GET', mf.DATA_CSV_URL): Resp(text=csv if csv is not None else data_csv(latest)),
        ('GET', mf.SUMMARY_CSV_URL): Resp(text=summary if summary is not None else summary_csv(latest)),
        ('GET', mf.PROVENANCE_URL): Resp(text=prov if prov is not None else provenance(latest)),
        ('GET', WORKFLOW_URL): Resp(payload={'state': state}),
        ('GET', RUNS_URL): Resp(payload={'workflow_runs': runs if runs is not None else [run('success')]}),
        ('GET', ISSUES_URL): Resp(payload=issues if issues is not None else []),
        ('POST', ISSUES_URL): Resp(201, payload={'number': 99, 'html_url': f'https://github.com/{REPO}/issues/99'}),
    }


def monitor(http, now=NOW, **kwargs):
    kwargs.setdefault('sleep', lambda seconds: None)
    return mf.run_monitor(http=http, now=now, repo=REPO, **kwargs)


def check(result, code):
    return next(c for c in result.checks if c.code == code)


# ---------- checks ----------


class TestChecks:
    def test_01_healthy_feed_and_producer(self):
        http = FakeHttp(routes())
        result = monitor(http)
        assert result.status == 'HEALTHY'
        assert result.exit_code == 0
        assert http.writes() == []
        assert all(c.state == 'OK' for c in result.checks)

    @pytest.mark.parametrize('state', ['disabled_inactivity', 'disabled_manually'])
    def test_02_disabled_producer_is_incident(self, state):
        result = monitor(FakeHttp(routes(state=state)))
        assert check(result, 'producer_disabled').state == 'INCIDENT'
        assert result.status == 'INCIDENT'
        assert result.exit_code == 1

    def test_03a_two_consecutive_failures_are_incident(self):
        result = monitor(FakeHttp(routes(runs=[run('failure', 2), run('timed_out', 1), run('success', 0)])))
        assert check(result, 'producer_failing').state == 'INCIDENT'

    def test_03b_one_latest_failure_is_warning_only(self):
        result = monitor(FakeHttp(routes(runs=[run('failure', 2), run('success', 1)])))
        assert check(result, 'producer_failing').state == 'WARN'
        assert result.status == 'HEALTHY'
        assert result.exit_code == 0

    def test_03c_ignored_conclusions_are_skipped(self):
        runs = [run('cancelled', 4), run('failure', 3), run('skipped', 2), run('startup_failure', 1)]
        result = monitor(FakeHttp(routes(runs=runs)))
        assert check(result, 'producer_failing').state == 'INCIDENT'

    def test_03d_failure_then_success_is_ok(self):
        result = monitor(FakeHttp(routes(runs=[run('success', 2), run('failure', 1)])))
        assert check(result, 'producer_failing').state == 'OK'

    def test_03e_runs_query_filters_main_and_completed(self):
        http = FakeHttp(routes())
        monitor(http)
        params = next(c['params'] for c in http.calls if c['url'] == RUNS_URL)
        assert params['branch'] == 'main'
        assert params['status'] == 'completed'

    def test_03f_no_decisive_run_is_unknown(self):
        result = monitor(FakeHttp(routes(runs=[run('cancelled', 2), run('skipped', 1)])))
        assert check(result, 'producer_failing').state == 'UNKNOWN'
        assert result.status == 'UNKNOWN'
        assert result.exit_code == 2

    def test_04_stale_feed_is_incident(self):
        result = monitor(FakeHttp(routes(latest='2026-11-11')))
        assert check(result, 'feed_stale').state == 'INCIDENT'
        assert '2026-11-12' in check(result, 'feed_stale').message

    def test_05_within_delivery_grace_is_ok(self):
        now = datetime(2026, 11, 13, 20, 0, tzinfo=timezone.utc)
        result = monitor(FakeHttp(routes(latest='2026-11-11')), now=now)
        assert check(result, 'feed_stale').state == 'OK'

    @pytest.mark.parametrize(
        'latest,now',
        [
            ('2026-11-25', datetime(2026, 11, 27, 23, 30, tzinfo=timezone.utc)),  # Thanksgiving 11-26
            ('2026-11-13', datetime(2026, 11, 15, 23, 30, tzinfo=timezone.utc)),  # weekend
            ('2026-11-25', datetime(2026, 11, 28, 17, 0, tzinfo=timezone.utc)),  # 11-27 closes 18:00 UTC
        ],
    )
    def test_06a_holiday_weekend_and_early_close_do_not_alarm(self, latest, now):
        result = monitor(FakeHttp(routes(latest=latest)), now=now)
        assert check(result, 'feed_stale').state == 'OK'

    def test_06b_after_early_close_grace_the_feed_is_stale(self):
        now = datetime(2026, 11, 28, 23, 30, tzinfo=timezone.utc)
        result = monitor(FakeHttp(routes(latest='2026-11-25')), now=now)
        assert check(result, 'feed_stale').state == 'INCIDENT'

    def test_06c_long_outage_is_incident_not_unknown(self):
        result = monitor(FakeHttp(routes(latest='2026-10-15')))
        assert check(result, 'feed_stale').state == 'INCIDENT'

    def test_07a_low_coverage_is_incident(self):
        csv = data_csv('2026-11-12', coverage_200=0.85)
        result = monitor(FakeHttp(routes(csv=csv)))
        assert check(result, 'coverage_low').state == 'INCIDENT'

    def test_07b_missing_coverage_column_is_incident(self):
        csv = data_csv('2026-11-12', include_coverage=False)
        result = monitor(FakeHttp(routes(csv=csv)))
        assert check(result, 'coverage_low').state == 'INCIDENT'

    def test_07c_min_coverage_is_configurable(self):
        result = monitor(FakeHttp(routes()), min_coverage=1.01)
        assert check(result, 'coverage_low').state == 'INCIDENT'

    def test_08a_html_response_is_invalid_feed(self):
        result = monitor(FakeHttp(routes(csv='<!DOCTYPE html><html></html>')))
        assert check(result, 'feed_invalid').state == 'INCIDENT'
        assert result.status == 'INCIDENT'

    def test_08b_persistent_download_error_is_invalid_feed(self):
        r = routes()
        r[('GET', mf.DATA_CSV_URL)] = requests.ConnectionError('down')
        http = FakeHttp(r)
        result = monitor(http)
        assert check(result, 'feed_invalid').state == 'INCIDENT'
        assert sum(c['url'] == mf.DATA_CSV_URL for c in http.calls) == mf.MAX_ATTEMPTS

    def test_08c_transient_server_error_is_retried(self):
        r = routes()
        r[('GET', mf.DATA_CSV_URL)] = [Resp(503, text='busy'), Resp(text=data_csv('2026-11-12'))]
        result = monitor(FakeHttp(r))
        assert check(result, 'feed_invalid').state == 'OK'

    def test_08d_deploy_skew_resolves_on_refetch(self):
        r = routes()
        r[('GET', mf.SUMMARY_CSV_URL)] = [Resp(text=summary_csv('2026-11-11')), Resp(text=summary_csv('2026-11-12'))]
        sleeps = []
        http = FakeHttp(r)
        result = monitor(http, sleep=sleeps.append)
        assert check(result, 'feed_invalid').state == 'OK'
        assert mf.SKEW_WAIT_SECONDS in sleeps
        for url in (mf.DATA_CSV_URL, mf.SUMMARY_CSV_URL, mf.PROVENANCE_URL):
            assert sum(c['url'] == url for c in http.calls) == 2

    def test_08e_persistent_mismatch_is_incident(self):
        result = monitor(FakeHttp(routes(summary=summary_csv('2026-11-11'))))
        assert check(result, 'feed_invalid').state == 'INCIDENT'

    @pytest.mark.parametrize(
        'ages,expected',
        [([100.0, 300.0, 590.0], 530.0), ([0.0, 0.0, 0.0], 630.0), ([590.0, 595.0, 599.0], 40.0), ([10.0, None], 630)],
    )
    def test_08f_skew_wait_follows_age_headers(self, ages, expected):
        # Pages ignores query strings, so the monitor waits until every cached copy expired.
        assert mf.skew_wait_seconds(ages) == expected

    def test_08g_skew_wait_uses_response_age(self):
        r = routes()
        aged = Resp(text=summary_csv('2026-11-11'))
        aged.headers = {'Age': '500'}
        r[('GET', mf.SUMMARY_CSV_URL)] = [aged, Resp(text=summary_csv('2026-11-12'))]
        for url in (mf.DATA_CSV_URL, mf.PROVENANCE_URL):
            r[('GET', url)].headers = {'Age': '200'}
        sleeps = []
        monitor(FakeHttp(r), sleep=sleeps.append)
        assert sleeps == [430.0]

    def test_08h_missing_workflow_is_incident(self):
        r = routes()
        r[('GET', WORKFLOW_URL)] = Resp(404, payload={'message': 'Not Found'})
        result = monitor(FakeHttp(r))
        assert check(result, 'producer_disabled').state == 'INCIDENT'

    @pytest.mark.parametrize(
        'prov',
        [
            'not json',
            provenance('2026-11-12', confidence='calendar_only'),
            provenance('2026-11-12', eligible=False),
            provenance('2026-11-11'),
        ],
    )
    def test_09_invalid_provenance_is_incident(self, prov):
        result = monitor(FakeHttp(routes(prov=prov)))
        assert check(result, 'provenance_invalid').state == 'INCIDENT'

    def test_10a_github_api_error_is_unknown(self):
        r = routes()
        r[('GET', WORKFLOW_URL)] = Resp(500, text='oops')
        result = monitor(FakeHttp(r))
        assert check(result, 'producer_disabled').state == 'UNKNOWN'
        assert result.status == 'UNKNOWN'
        assert result.exit_code == 2

    def test_10b_calendar_unavailable_is_unknown(self, monkeypatch):
        import market_dates

        def broken():
            raise ImportError('no calendar')

        monkeypatch.setattr(market_dates, '_get_calendar', broken)
        result = monitor(FakeHttp(routes()))
        assert check(result, 'feed_stale').state == 'UNKNOWN'
        assert result.exit_code == 2

    def test_10c_incident_takes_precedence_over_unknown(self):
        r = routes(latest='2026-11-11')
        r[('GET', WORKFLOW_URL)] = Resp(500, text='oops')
        result = monitor(FakeHttp(r))
        assert result.status == 'INCIDENT'


# ---------- notification ----------


def notify(http, **kwargs):
    kwargs.setdefault('token', TOKEN)
    kwargs.setdefault('in_actions', True)
    return monitor(http, notify=True, **kwargs)


def body_state(body):
    match = re.search(r'<!-- feed-monitor:state (\{.*?\}) -->', body)
    assert match, body
    return json.loads(match.group(1))


class TestNotification:
    def test_11a_incident_without_issue_creates_one(self):
        r = routes(latest='2026-11-11')
        r[('POST', ISSUES_URL)] = Resp(201, payload={'number': 99, 'assignees': [{'login': 'tradermonty'}]})
        http = FakeHttp(r)
        result = notify(http, assignee='tradermonty')
        assert not any(a.startswith('WARN') for a in result.actions)
        writes = http.writes()
        assert [w['method'] for w in writes] == ['POST']
        payload = writes[0]['json']
        assert payload['title'] == 'Feed monitor incident: feed_stale'
        assert body_state(payload['body'])['fingerprint'] == 'feed_stale'
        assert payload['assignees'] == ['tradermonty']
        assert result.exit_code == 0

    def test_11b_empty_assignee_is_unset(self):
        http = FakeHttp(routes(latest='2026-11-11'))
        notify(http, assignee='')
        assert 'assignees' not in http.writes()[0]['json']

    def test_11c_rejected_assignee_retries_without_it(self):
        r = routes(latest='2026-11-11')
        r[('POST', ISSUES_URL)] = [Resp(422, payload={'message': 'invalid'}), Resp(201, payload={'number': 99})]
        http = FakeHttp(r)
        result = notify(http, assignee='nobody')
        posts = [w['json'] for w in http.writes()]
        assert len(posts) == 2
        assert 'assignees' not in posts[1]
        assert any(a.startswith('WARN') for a in result.actions)
        assert result.exit_code == 0

    def test_11d_silently_dropped_assignee_warns(self):
        http = FakeHttp(routes(latest='2026-11-11'))
        result = notify(http, assignee='outsider')
        assert any("did not assign 'outsider'" in a for a in result.actions)
        assert result.exit_code == 0

    def test_12e_unknown_previous_code_is_carried_forward(self):
        issue = incident_issue(5, fingerprint='feed_stale,producer_disabled', notified='2026-11-13T23:00:00+00:00')
        r = routes(latest='2026-11-11', issues=[issue])
        r[('GET', WORKFLOW_URL)] = Resp(500, text='oops')
        http = FakeHttp(r)
        result = notify(http)
        assert http.writes() == []
        assert result.fingerprint == 'feed_stale,producer_disabled'

    def test_12a_same_fingerprint_within_interval_writes_nothing(self):
        issue = incident_issue(5, notified='2026-11-12T23:30:00+00:00')
        http = FakeHttp(routes(latest='2026-11-11', issues=[issue]))
        result = notify(http)
        assert http.writes() == []
        assert result.exit_code == 0

    def test_12b_reminder_after_interval(self):
        issue = incident_issue(5, notified='2026-11-10T23:00:00+00:00')
        r = routes(latest='2026-11-11', issues=[issue])
        r[('POST', f'{ISSUES_URL}/5/comments')] = Resp(201, payload={})
        r[('PATCH', f'{ISSUES_URL}/5')] = Resp(200, payload={})
        http = FakeHttp(r)
        notify(http)
        methods = [(w['method'], w['url']) for w in http.writes()]
        assert methods == [('PATCH', f'{ISSUES_URL}/5'), ('POST', f'{ISSUES_URL}/5/comments')]
        assert body_state(http.writes()[0]['json']['body'])['last_notified_at'] == NOW.isoformat()

    def test_12c_changed_fingerprint_comments_immediately(self):
        issue = incident_issue(5, fingerprint='coverage_low', notified='2026-11-13T23:00:00+00:00')
        r = routes(latest='2026-11-11', issues=[issue])
        r[('POST', f'{ISSUES_URL}/5/comments')] = Resp(201, payload={})
        r[('PATCH', f'{ISSUES_URL}/5')] = Resp(200, payload={})
        http = FakeHttp(r)
        notify(http)
        patch = http.writes()[0]['json']
        assert body_state(patch['body'])['fingerprint'] == 'feed_stale'
        assert patch['title'] == 'Feed monitor incident: feed_stale'

    def test_12f_injected_state_block_in_feed_text_is_ignored(self):
        fake = state_block('x', '2099-01-01T00:00:00+00:00')
        checks = [mf.Check('feed_invalid', 'INCIDENT', f'Invalid date format: {fake!r}')]
        body = mf._issue_body(checks, 'feed_invalid', NOW)
        state = mf._state_of({'body': body})
        assert state == {'fingerprint': 'feed_invalid', 'last_notified_at': NOW.isoformat()}
        assert '<!--' not in mf.sanitize(fake)

    def test_12g_naive_last_notified_at_is_rejected(self):
        assert mf._state_of({'body': state_block('feed_stale', '2026-11-13T00:00:00')}) is None

    def test_12h_state_is_saved_before_the_comment(self):
        issue = incident_issue(5, fingerprint='coverage_low', notified='2026-11-13T23:00:00+00:00')
        r = routes(latest='2026-11-11', issues=[issue])
        r[('PATCH', f'{ISSUES_URL}/5')] = Resp(200, payload={})
        r[('POST', f'{ISSUES_URL}/5/comments')] = Resp(500, text='down')
        http = FakeHttp(r)
        result = notify(http)
        # The state PATCH went through first, so the next run will not repeat "changed".
        assert [w['method'] for w in http.writes()] == ['PATCH', 'POST']
        assert result.exit_code == 2

    def test_12i_state_patch_is_retried_on_server_error(self):
        issue = incident_issue(5, fingerprint='coverage_low', notified='2026-11-13T23:00:00+00:00')
        r = routes(latest='2026-11-11', issues=[issue])
        r[('PATCH', f'{ISSUES_URL}/5')] = [Resp(502, text='bad gateway'), Resp(200, payload={})]
        r[('POST', f'{ISSUES_URL}/5/comments')] = Resp(201, payload={})
        http = FakeHttp(r)
        result = notify(http)
        assert [w['method'] for w in http.writes()] == ['PATCH', 'PATCH', 'POST']
        assert result.exit_code == 0

    def test_12j_reminder_tolerates_schedule_drift(self):
        # 68 h since the last notice: a daily run that started earlier than the last one.
        issue = incident_issue(5, notified='2026-11-11T03:30:00+00:00')
        r = routes(latest='2026-11-11', issues=[issue])
        r[('PATCH', f'{ISSUES_URL}/5')] = Resp(200, payload={})
        r[('POST', f'{ISSUES_URL}/5/comments')] = Resp(201, payload={})
        http = FakeHttp(r)
        notify(http)
        assert len(http.writes()) == 2

    def test_12k_unknown_previous_codes_are_dropped(self):
        issue = incident_issue(5, fingerprint='feed_stale,`@evil`', notified='2026-11-13T23:00:00+00:00')
        r = routes(latest='2026-11-11', issues=[issue])
        r[('PATCH', f'{ISSUES_URL}/5')] = Resp(200, payload={})
        r[('POST', f'{ISSUES_URL}/5/comments')] = Resp(201, payload={})
        http = FakeHttp(r)
        result = notify(http)
        # Only the known code survives, so the incident is unchanged and nothing is echoed.
        assert http.writes() == []
        assert result.fingerprint == 'feed_stale'

    def test_15b_close_happens_before_recovery_comment(self):
        r = routes(issues=[incident_issue(3)])
        r[('PATCH', f'{ISSUES_URL}/3')] = Resp(200, payload={})
        r[('POST', f'{ISSUES_URL}/3/comments')] = Resp(500, text='down')
        http = FakeHttp(r)
        result = notify(http)
        assert [w['method'] for w in http.writes()] == ['PATCH', 'POST']
        assert result.exit_code == 2

    def test_11e_only_422_retries_without_assignee(self):
        r = routes(latest='2026-11-11')
        r[('POST', ISSUES_URL)] = [requests.Timeout('slow'), Resp(201, payload={'number': 99})]
        http = FakeHttp(r)
        result = notify(http, assignee='tradermonty')
        assert len(http.writes()) == 1
        assert result.exit_code == 2

    def test_12d_fingerprint_ignores_moving_dates(self):
        first = monitor(FakeHttp(routes(latest='2026-11-10')))
        later = monitor(FakeHttp(routes(latest='2026-11-10')), now=NOW + timedelta(days=3))
        assert first.fingerprint == later.fingerprint == 'feed_stale'

    def test_13a_foreign_issue_with_marker_is_ignored(self):
        foreign = incident_issue(7, user={'login': 'someone', 'type': 'User'})
        http = FakeHttp(routes(latest='2026-11-11', issues=[foreign]))
        notify(http)
        assert [w['method'] for w in http.writes()] == ['POST']

    def test_13b_pull_requests_are_skipped(self):
        pr = {**incident_issue(8), 'pull_request': {'url': 'x'}}
        http = FakeHttp(routes(latest='2026-11-11', issues=[pr]))
        notify(http)
        assert [w['method'] for w in http.writes()] == ['POST']

    def test_13c_issue_listing_is_paginated(self):
        filler = [
            {'number': 1000 + i, 'body': '', 'user': BOT, 'created_at': '2026-11-01T00:00:00Z'} for i in range(100)
        ]
        pages = {1: filler, 2: [incident_issue(5, notified='2026-11-13T23:00:00+00:00')]}
        r = routes(latest='2026-11-11')
        r[('GET', ISSUES_URL)] = lambda params, json: Resp(payload=pages.get(params['page'], []))
        http = FakeHttp(r)
        notify(http)
        assert http.writes() == []

    def test_13d_oldest_matching_issue_is_used(self):
        older = incident_issue(3, notified='2026-11-13T23:00:00+00:00', created='2026-11-01T00:00:00Z')
        newer = incident_issue(4, notified='2026-11-01T00:00:00+00:00', created='2026-11-05T00:00:00Z')
        http = FakeHttp(routes(latest='2026-11-11', issues=[newer, older]))
        result = notify(http)
        assert http.writes() == []
        assert any('#4' in a for a in result.actions)

    def test_14_feed_text_is_neutralized(self):
        text = '@octocat ```\x07' + 'x' * 2000
        safe = mf.sanitize(text)
        assert '@octocat' not in safe
        assert '`' not in safe
        assert '\x07' not in safe
        assert len(safe) <= mf.MAX_MESSAGE_CHARS

    def test_15_recovery_comments_and_closes_all(self):
        issues = [incident_issue(3, created='2026-11-01T00:00:00Z'), incident_issue(4, created='2026-11-02T00:00:00Z')]
        r = routes(issues=issues)
        for n in (3, 4):
            r[('POST', f'{ISSUES_URL}/{n}/comments')] = Resp(201, payload={})
            r[('PATCH', f'{ISSUES_URL}/{n}')] = Resp(200, payload={})
        http = FakeHttp(r)
        result = notify(http)
        closes = [w for w in http.writes() if w['method'] == 'PATCH']
        assert [w['json'] for w in closes] == [{'state': 'closed', 'state_reason': 'completed'}] * 2
        comments = [w for w in http.writes() if w['method'] == 'POST']
        assert [w['url'] for w in comments] == [f'{ISSUES_URL}/3/comments', f'{ISSUES_URL}/4/comments']
        assert all(w['json']['body'].startswith('Recovered') for w in comments)
        assert result.exit_code == 0

    def test_16_unknown_never_writes(self):
        r = routes(issues=[incident_issue(3)])
        r[('GET', WORKFLOW_URL)] = Resp(500, text='oops')
        http = FakeHttp(r)
        result = notify(http)
        assert http.writes() == []
        assert result.exit_code == 2

    def test_17a_notify_without_token_is_dry_run(self):
        http = FakeHttp(routes(latest='2026-11-11'))
        result = monitor(http, notify=True, token=None, in_actions=True)
        assert http.writes() == []
        assert result.exit_code == 2
        assert any('FEED_MONITOR_TOKEN' in a for a in result.actions)

    def test_17b_local_notify_requires_explicit_author(self):
        http = FakeHttp(routes(latest='2026-11-11'))
        result = monitor(http, notify=True, token=TOKEN, in_actions=False)
        assert http.writes() == []
        assert result.exit_code == 2

    def test_17c_failed_write_exits_2(self):
        r = routes(latest='2026-11-11')
        r[('POST', ISSUES_URL)] = Resp(403, text='forbidden')
        result = notify(FakeHttp(r))
        assert result.exit_code == 2

    def test_17d_dry_run_reports_planned_action(self):
        http = FakeHttp(routes(latest='2026-11-11'))
        result = monitor(http)
        assert http.writes() == []
        assert any(a.startswith('DRY RUN') for a in result.actions)


# ---------- CLI and token handling ----------


class TestCli:
    def test_18a_token_never_printed(self, capsys):
        r = routes(latest='2026-11-11')
        r[('POST', ISSUES_URL)] = requests.HTTPError(f'403 for url with token={TOKEN} Authorization: Bearer {TOKEN}')
        notify(FakeHttp(r))
        out = capsys.readouterr()
        assert TOKEN not in out.out + out.err

    def test_18b_github_token_alone_is_not_used(self, monkeypatch):
        monkeypatch.setenv('GITHUB_TOKEN', TOKEN)
        monkeypatch.delenv('FEED_MONITOR_TOKEN', raising=False)
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return mf.MonitorResult('HEALTHY', [], [], 0, '')

        monkeypatch.setattr(mf, 'run_monitor', fake_run)
        assert mf.main(['--notify', 'github-issue']) == 0
        assert captured['token'] is None

    def test_18c_main_reads_process_environment(self, monkeypatch):
        monkeypatch.setenv('FEED_MONITOR_TOKEN', TOKEN)
        monkeypatch.setenv('FEED_MONITOR_ASSIGNEE', '')
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return mf.MonitorResult('HEALTHY', [], [], 0, '')

        monkeypatch.setattr(mf, 'run_monitor', fake_run)
        mf.main(['--min-coverage', '1.01'])
        assert captured['token'] == TOKEN
        assert captured['assignee'] is None
        assert captured['min_coverage'] == 1.01

    def test_18d_import_never_loads_dotenv(self, tmp_path):
        import subprocess

        (tmp_path / '.env').write_text(f'FEED_MONITOR_TOKEN={TOKEN}\n')
        probe = (
            'import os, sys; sys.path.insert(0, sys.argv[1]); import monitor_feed; '
            "print(os.environ.get('FEED_MONITOR_TOKEN'), 'dotenv' in sys.modules)"
        )
        env = {k: v for k, v in os.environ.items() if k != 'FEED_MONITOR_TOKEN'}
        out = subprocess.run(
            [sys.executable, '-c', probe, str(ROOT / 'scripts')],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert out == ['None', 'False']


# ---------- workflow ----------


class TestWorkflow:
    @pytest.fixture
    def workflow(self):
        yaml = pytest.importorskip('yaml')
        text = (ROOT / '.github' / 'workflows' / 'feed-monitor.yml').read_text()
        return text, yaml.safe_load(text)

    def test_19a_triggers_and_inputs(self, workflow):
        _, wf = workflow
        on = wf[True] if True in wf else wf['on']
        assert on['schedule'] == [{'cron': '30 23 * * *'}]
        inputs = on['workflow_dispatch']['inputs']
        assert inputs['notify']['type'] == 'boolean'
        assert inputs['notify']['default'] is False

    def test_19b_permissions_are_minimal(self, workflow):
        _, wf = workflow
        assert wf['permissions'] == {'contents': 'read', 'actions': 'read', 'issues': 'write'}

    def test_19c_only_github_token_secret(self, workflow):
        text, _ = workflow
        assert set(re.findall(r'secrets\.(\w+)', text)) == {'GITHUB_TOKEN'}
        assert 'FMP' not in text

    def test_19d_notify_is_gated_to_default_branch(self, workflow):
        text, _ = workflow
        assert "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)" in text
        assert "github.event_name == 'schedule' || inputs.notify == true" in text
        assert 'persist-credentials: false' in text

    def test_19g_inputs_reach_shell_only_through_env(self, workflow):
        _, wf = workflow
        job = wf['jobs']['monitor']
        assert job['timeout-minutes'] == 30
        for step in job['steps']:
            assert '${{' not in step.get('run', '')

    def test_19e_dependencies_match_requirements(self, workflow):
        text, _ = workflow
        reqs = {}
        for line in (ROOT / 'requirements.txt').read_text().splitlines():
            spec = line.split('#')[0].strip()
            if spec:
                name = re.split(r'[<>=!~]', spec)[0].strip().lower()
                reqs[name] = spec
        installed = re.findall(r"'([A-Za-z0-9_.-]+[<>=!~][^']*)'", text)
        assert installed, 'workflow must quote pinned specifiers'
        for spec in installed:
            name = re.split(r'[<>=!~]', spec)[0].strip().lower()
            assert reqs[name] == spec

    def test_19f_independent_of_producer_jobs(self, workflow):
        text, wf = workflow
        assert 'needs' not in text
        assert 'deploy' not in wf['jobs']


def test_20_module_does_not_dispatch():
    source = (ROOT / 'scripts' / 'monitor_feed.py').read_text()
    assert '/dispatches' not in source
    assert 'trigger_market_breadth' not in source
    assert 'trigger_workflow' not in source
    assert os.path.basename(mf.__file__) == 'monitor_feed.py'
