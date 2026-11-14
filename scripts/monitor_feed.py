"""Independently monitor the public market breadth feed and its producer workflow (Issue #22).

Read-only by default: prints a report and the action it would take. With
``--notify github-issue`` it keeps one open GitHub issue per incident and
closes it on recovery. It never dispatches a workflow.

Exit codes in a dry run: 0 healthy, 1 incident, 2 unknown or configuration error.
With ``--notify``: 0 when the issue state is in sync, 2 on unknown, a
configuration error or a failed write. See docs/feed_monitoring.md.
"""

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

# feed_contract never loads .env, so only main() reads credentials, from the process environment.
from feed_contract import BASE_URL, DATA_CSV_URL, SUMMARY_CSV_URL, validate_csv_content  # noqa: E402
from market_dates import _utc_now, resolve_market_date  # noqa: E402
from secret_redaction import redact, register_secret  # noqa: E402

PROVENANCE_URL = f'{BASE_URL}/market_date_resolution.json'
DEFAULT_REPO = 'tradermonty/market-breadth-analysis'
PRODUCER_WORKFLOW = 'daily-market-breadth.yml'
DEFAULT_ACTIONS_AUTHOR = 'github-actions[bot]'

MAX_ATTEMPTS = 3
PAGES_MAX_AGE = 600  # GitHub Pages: cache-control max-age=600, per file
SKEW_MARGIN_SECONDS = 30
SKEW_WAIT_SECONDS = PAGES_MAX_AGE + SKEW_MARGIN_SECONDS  # upper bound when Age is unknown
MAX_MESSAGE_CHARS = 500
TIMEOUT = 30
ISSUES_PER_PAGE = 100
REMINDER_TOLERANCE = timedelta(hours=12)

FAILED_CONCLUSIONS = {'failure', 'timed_out', 'startup_failure'}
SUCCESS_CONCLUSIONS = {'success'}
COVERAGE_COLUMNS = ('Coverage_200', 'Coverage_50')
CHECK_ORDER = (
    'producer_disabled',
    'producer_failing',
    'feed_invalid',
    'feed_stale',
    'coverage_low',
    'provenance_invalid',
)

STATE_PATTERN = re.compile(r'<!-- feed-monitor:state (\{.*?\}) -->')


@dataclass(frozen=True)
class Check:
    code: str
    state: str  # OK | WARN | INCIDENT | UNKNOWN
    message: str


@dataclass
class MonitorResult:
    status: str  # HEALTHY | INCIDENT | UNKNOWN
    checks: list
    actions: list
    exit_code: int
    fingerprint: str = field(default='')


class WriteError(RuntimeError):
    """A GitHub issue write failed; ``status`` is the HTTP status when one was received."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def sanitize(text):
    """Make feed- or API-derived text safe inside a fenced block of an issue."""
    text = ''.join(ch for ch in str(text) if ch == '\n' or ord(ch) >= 32)
    text = text.replace('`', "'").replace('@', '@​')
    # HTML comment delimiters could forge or hide the hidden state block.
    text = text.replace('<!--', '<! --').replace('-->', '-- >')
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[: MAX_MESSAGE_CHARS - 3] + '...'
    return text


def _github_headers(token):
    headers = {'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def _get(http, url, *, sleep, params=None, headers=None):
    """GET with retries on connection errors, timeouts and 5xx; 4xx raises at once."""
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = http.request('GET', url, headers=headers, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp
            last_error = requests.HTTPError(f'{resp.status_code} server error for {url}')
        if attempt < MAX_ATTEMPTS - 1:
            sleep(2**attempt)
    raise last_error


def _write(http, method, url, token, payload, sleep=None):
    """Send one write. PATCH sets absolute values, so it is retried on 5xx and
    connection errors when ``sleep`` is given; POST is never retried, because a
    repeated POST could duplicate an issue or a comment."""
    attempts = MAX_ATTEMPTS if method == 'PATCH' and sleep is not None else 1
    for attempt in range(attempts):
        try:
            resp = http.request(method, url, headers=_github_headers(token), json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            error = WriteError(f'{method} {url} failed: {redact(str(exc))}', status)
            if (status is not None and status < 500) or attempt == attempts - 1:
                raise error from None
            sleep(2**attempt)
    raise AssertionError('unreachable')


# ---------- producer checks ----------


def _check_producer_state(http, api, token, sleep):
    try:
        payload = _get(
            http, f'{api}/actions/workflows/{PRODUCER_WORKFLOW}', sleep=sleep, headers=_github_headers(token)
        ).json()
        state = payload['state']
    except requests.HTTPError as exc:
        if getattr(exc.response, 'status_code', None) == 404:
            return Check('producer_disabled', 'INCIDENT', f'Producer workflow {PRODUCER_WORKFLOW} not found')
        return Check('producer_disabled', 'UNKNOWN', f'Could not read producer workflow state: {redact(str(exc))}')
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return Check('producer_disabled', 'UNKNOWN', f'Could not read producer workflow state: {redact(str(exc))}')
    if state != 'active':
        return Check('producer_disabled', 'INCIDENT', f'Producer workflow state is {state!r}')
    return Check('producer_disabled', 'OK', 'Producer workflow is active')


def _check_producer_runs(http, api, token, sleep, threshold):
    try:
        payload = _get(
            http,
            f'{api}/actions/workflows/{PRODUCER_WORKFLOW}/runs',
            sleep=sleep,
            headers=_github_headers(token),
            params={'branch': 'main', 'status': 'completed', 'per_page': 20},
        ).json()
        runs = payload['workflow_runs']
        decisive = [r for r in runs if r['conclusion'] in FAILED_CONCLUSIONS | SUCCESS_CONCLUSIONS]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return Check('producer_failing', 'UNKNOWN', f'Could not read producer runs: {redact(str(exc))}')
    if not decisive:
        return Check('producer_failing', 'UNKNOWN', 'No failed or successful producer run among the latest 20')
    failed = 0
    for item in decisive:
        if item['conclusion'] not in FAILED_CONCLUSIONS:
            break
        failed += 1
    latest = decisive[0].get('html_url', 'unknown run')
    if failed >= threshold:
        return Check('producer_failing', 'INCIDENT', f'Latest {failed} producer runs failed; latest: {latest}')
    if failed:
        return Check('producer_failing', 'WARN', f'Latest producer run failed: {latest}')
    return Check('producer_failing', 'OK', 'Latest producer run succeeded')


# ---------- feed checks ----------


def _age(resp):
    try:
        return max(0.0, float(resp.headers.get('Age')))
    except (TypeError, ValueError):
        return None


def _fetch_feed(http, sleep):
    texts = {'ages': []}
    for name, url in (('csv', DATA_CSV_URL), ('summary', SUMMARY_CSV_URL)):
        resp = _get(http, url, sleep=sleep)
        texts[name] = resp.text
        texts['ages'].append(_age(resp))
    try:
        resp = _get(http, PROVENANCE_URL, sleep=sleep)
        texts['provenance'] = resp.text
        texts['ages'].append(_age(resp))
    except requests.RequestException as exc:
        texts['provenance'] = None
        texts['provenance_error'] = redact(str(exc))
    return texts


def skew_wait_seconds(ages):
    """Seconds until every cached copy seen has expired; Pages ignores query strings."""
    if not ages or any(age is None for age in ages):
        return SKEW_WAIT_SECONDS
    return min(SKEW_WAIT_SECONDS, max(0.0, PAGES_MAX_AGE - min(ages)) + SKEW_MARGIN_SECONDS)


def _parse_provenance(text):
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _inconsistent(texts, validation, now):
    if not validation.is_valid:
        return (validation.error or '').startswith('Data/summary date mismatch')
    payload = _parse_provenance(texts.get('provenance'))
    return payload is not None and payload.get('selected_date') != validation.latest_data_date


def _check_stale(latest, now, grace_hours):
    resolution = resolve_market_date(lambda *_: [], now=now, grace_minutes=grace_hours * 60)
    if resolution.confidence == 'weekday_only' or resolution.expected_date is None:
        return Check('feed_stale', 'UNKNOWN', 'Exchange calendar unavailable; staleness not judged')
    if date.fromisoformat(latest) < date.fromisoformat(resolution.expected_date):
        return Check(
            'feed_stale',
            'INCIDENT',
            f'Latest data {latest} is older than expected session {resolution.expected_date} '
            f'({grace_hours:g} h delivery grace)',
        )
    return Check('feed_stale', 'OK', f'Latest data {latest}; expected session {resolution.expected_date}')


def _check_coverage(csv_text, min_coverage):
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    latest = rows[-1]
    problems = []
    for column in COVERAGE_COLUMNS:
        try:
            value = float(latest[column])
        except (KeyError, TypeError, ValueError):
            problems.append(f'{column} missing or invalid')
            continue
        if not math.isfinite(value) or value < min_coverage:
            problems.append(f'{column}={value:.4f} below {min_coverage:g}')
    if problems:
        return Check('coverage_low', 'INCIDENT', '; '.join(problems))
    return Check('coverage_low', 'OK', f'Coverage at or above {min_coverage:g}')


def _check_provenance(texts, latest):
    if texts.get('provenance') is None:
        return Check('provenance_invalid', 'INCIDENT', f'Provenance unavailable: {texts.get("provenance_error")}')
    payload = _parse_provenance(texts['provenance'])
    if payload is None:
        return Check('provenance_invalid', 'INCIDENT', 'Provenance is not a JSON object')
    if payload.get('confidence') != 'observed' or payload.get('publication_eligible') is not True:
        return Check(
            'provenance_invalid',
            'INCIDENT',
            f'Provenance confidence={payload.get("confidence")!r} eligible={payload.get("publication_eligible")!r}',
        )
    if payload.get('selected_date') != latest:
        return Check(
            'provenance_invalid',
            'INCIDENT',
            f'Provenance selected_date {payload.get("selected_date")!r} differs from latest data {latest}',
        )
    return Check('provenance_invalid', 'OK', 'Provenance is observed and matches the data')


def _feed_checks(http, sleep, now, grace_hours, min_coverage):
    unknown = [Check(code, 'UNKNOWN', 'Feed unusable') for code in ('feed_stale', 'coverage_low', 'provenance_invalid')]
    try:
        texts = _fetch_feed(http, sleep)
        validation = validate_csv_content(texts['csv'], texts['summary'], now=now)
        if _inconsistent(texts, validation, now):
            # Pages caches each file separately; a deploy in progress can mix versions.
            sleep(skew_wait_seconds(texts['ages']))
            texts = _fetch_feed(http, sleep)
            validation = validate_csv_content(texts['csv'], texts['summary'], now=now)
    except requests.RequestException as exc:
        return [Check('feed_invalid', 'INCIDENT', f'Feed download failed: {redact(str(exc))}'), *unknown]
    if not validation.is_valid:
        return [Check('feed_invalid', 'INCIDENT', f'Invalid feed content: {validation.error}'), *unknown]
    latest = validation.latest_data_date
    return [
        Check('feed_invalid', 'OK', f'Feed content valid ({validation.row_count} rows)'),
        _check_stale(latest, now, grace_hours),
        _check_coverage(texts['csv'], min_coverage),
        _check_provenance(texts, latest),
    ]


# ---------- incident issues ----------


def _state_of(issue):
    # The monitor writes its block last, after any feed-derived report text.
    matches = STATE_PATTERN.findall(issue.get('body') or '')
    if not matches:
        return None
    try:
        state = json.loads(matches[-1])
        notified = datetime.fromisoformat(state['last_notified_at'])
    except (ValueError, KeyError, TypeError):
        return None
    if notified.tzinfo is None or not isinstance(state.get('fingerprint'), str):
        return None
    return state


def _list_incident_issues(http, api, token, sleep, author):
    matches = []
    page = 1
    while True:
        batch = _get(
            http,
            f'{api}/issues',
            sleep=sleep,
            headers=_github_headers(token),
            params={'state': 'open', 'per_page': ISSUES_PER_PAGE, 'page': page},
        ).json()
        for issue in batch:
            user = issue.get('user') or {}
            if 'pull_request' in issue or user.get('login') != author:
                continue
            if author == DEFAULT_ACTIONS_AUTHOR and user.get('type') != 'Bot':
                continue
            if _state_of(issue) is not None:
                matches.append(issue)
        if len(batch) < ISSUES_PER_PAGE:
            break
        page += 1
    return sorted(matches, key=lambda i: (i.get('created_at', ''), i['number']))


def _report_markdown(checks):
    lines = ['| Check | State |', '|---|---|']
    lines += [f'| `{c.code}` | {c.state} |' for c in checks]
    lines.append('')
    for c in checks:
        if c.state != 'OK':
            lines += [f'**{c.code}**', '```text', sanitize(c.message), '```']
    return '\n'.join(lines)


def _issue_title(fingerprint):
    return f'Feed monitor incident: {fingerprint}'


def _issue_body(checks, fingerprint, now):
    state = json.dumps({'fingerprint': fingerprint, 'last_notified_at': now.isoformat()})
    return (
        'The independent feed monitor detected an incident. '
        'Recovery steps: docs/feed_monitoring.md.\n\n'
        f'{_report_markdown(checks)}\n\n<!-- feed-monitor:state {state} -->\n'
    )


def _sync_issues(http, api, token, sleep, result, now, author, assignee, reminder_hours):
    issues = _list_incident_issues(http, api, token, sleep, author)
    actions = []
    if result.status == 'HEALTHY':
        for issue in issues:
            # Close first: if the comment then fails, the next run does not repeat it.
            _write(
                http,
                'PATCH',
                f'{api}/issues/{issue["number"]}',
                token,
                {'state': 'closed', 'state_reason': 'completed'},
                sleep,
            )
            _write(
                http,
                'POST',
                f'{api}/issues/{issue["number"]}/comments',
                token,
                {'body': 'Recovered: every check passes.\n\n' + _report_markdown(result.checks)},
            )
            actions.append(f'Closed recovered incident issue #{issue["number"]}')
        return actions or ['No open incident issue']

    body = _issue_body(result.checks, result.fingerprint, now)
    if not issues:
        payload = {'title': _issue_title(result.fingerprint), 'body': body}
        if assignee:
            payload['assignees'] = [assignee]
        try:
            resp = _write(http, 'POST', f'{api}/issues', token, payload)
        except WriteError as exc:
            # Only a validation error is safe to retry; after a timeout or 5xx the
            # issue may already exist, and a retry would duplicate it.
            if not assignee or exc.status != 422:
                raise
            payload.pop('assignees')
            resp = _write(http, 'POST', f'{api}/issues', token, payload)
            actions.append(f'WARN: could not assign {assignee!r}; opened the issue unassigned')
        created = resp.json()
        if assignee and 'assignees' in payload:
            logins = {a.get('login') for a in created.get('assignees') or [] if isinstance(a, dict)}
            if assignee not in logins:
                # The API silently drops logins the caller cannot assign.
                actions.append(f'WARN: GitHub did not assign {assignee!r}')
        actions.append(f'Opened incident issue #{created.get("number")}')
        return actions

    primary, extras = issues[0], issues[1:]
    actions += [f'NOTE: duplicate incident issue #{i["number"]} left open' for i in extras]
    state = _state_of(primary)
    # A previous incident code that cannot be judged this run stays in the fingerprint,
    # so a one-run API blip does not post two "changed" comments.
    unknown = {c.code for c in result.checks if c.state == 'UNKNOWN'}
    # The body is editable by collaborators: keep only known check codes.
    previous = {code for code in state['fingerprint'].split(',') if code in CHECK_ORDER}
    previous_fingerprint = ','.join(sorted(previous))
    carried = set(result.fingerprint.split(',')) | (previous & unknown)
    result.fingerprint = ','.join(sorted(code for code in carried if code))
    body = _issue_body(result.checks, result.fingerprint, now)
    last = datetime.fromisoformat(state['last_notified_at'])
    # Scheduled starts drift by hours, so allow REMINDER_TOLERANCE early.
    due = timedelta(hours=reminder_hours) - min(REMINDER_TOLERANCE, timedelta(hours=reminder_hours) / 6)
    if previous_fingerprint != result.fingerprint:
        heading = f'Incident changed: {previous_fingerprint or "(none)"} -> {result.fingerprint}'
    elif now - last >= due:
        heading = f'Incident persists: {result.fingerprint}'
    else:
        actions.append(f'Incident issue #{primary["number"]} unchanged; no notification')
        return actions
    number = primary['number']
    # Save the new state first: if the comment then fails, the next run does not repeat it.
    _write(
        http,
        'PATCH',
        f'{api}/issues/{number}',
        token,
        {'title': _issue_title(result.fingerprint), 'body': body},
        sleep,
    )
    _write(
        http,
        'POST',
        f'{api}/issues/{number}/comments',
        token,
        {'body': f'{heading}\n\n{_report_markdown(result.checks)}'},
    )
    actions.append(f'Commented on incident issue #{number}: {heading}')
    return actions


# ---------- entry points ----------


def _overall(checks):
    states = {c.state for c in checks}
    if 'INCIDENT' in states:
        return 'INCIDENT'
    if 'UNKNOWN' in states:
        return 'UNKNOWN'
    return 'HEALTHY'


def _print_report(result):
    print(f'Feed monitor status: {result.status}')
    for c in result.checks:
        print(redact(f'  [{c.state}] {c.code}: {c.message}'))
    for action in result.actions:
        print(redact(f'  -> {action}'))


def run_monitor(
    *,
    http,
    now=None,
    repo=DEFAULT_REPO,
    token=None,
    notify=False,
    in_actions=False,
    author=None,
    assignee=None,
    min_coverage=0.9,
    grace_hours=24,
    failure_threshold=2,
    reminder_hours=72,
    sleep=time.sleep,
):
    now = _utc_now(now)
    register_secret(token)
    api = f'https://api.github.com/repos/{repo}'
    checks = [
        _check_producer_state(http, api, token, sleep),
        _check_producer_runs(http, api, token, sleep, failure_threshold),
        *_feed_checks(http, sleep, now, grace_hours, min_coverage),
    ]
    checks.sort(key=lambda c: CHECK_ORDER.index(c.code))
    status = _overall(checks)
    fingerprint = ','.join(sorted(c.code for c in checks if c.state == 'INCIDENT'))
    result = MonitorResult(status, checks, [], {'HEALTHY': 0, 'INCIDENT': 1}.get(status, 2), fingerprint)

    if not notify:
        if status == 'INCIDENT':
            result.actions.append(f'DRY RUN: would open or update an incident issue for {fingerprint}')
        elif status == 'HEALTHY':
            result.actions.append('DRY RUN: would close any open incident issue')
        else:
            result.actions.append('DRY RUN: incident state would be left unchanged')
        _print_report(result)
        return result

    if author is None and in_actions:
        author = DEFAULT_ACTIONS_AUTHOR
    error = None
    if not token:
        error = 'ERROR: --notify needs FEED_MONITOR_TOKEN; ran as a dry run'
    elif author is None:
        error = 'ERROR: --notify outside GitHub Actions needs --incident-author; ran as a dry run'
    elif status == 'UNKNOWN':
        error = 'UNKNOWN: incident issues left unchanged'
    if error:
        result.actions.append(error)
        result.exit_code = 2
        _print_report(result)
        return result

    try:
        result.actions += _sync_issues(http, api, token, sleep, result, now, author, assignee, reminder_hours)
        result.exit_code = 0
    except (WriteError, requests.RequestException, ValueError, KeyError, TypeError) as exc:
        result.actions.append(f'ERROR: incident issue sync failed: {redact(str(exc))}')
        result.exit_code = 2
    _print_report(result)
    return result


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f'must be finite and positive: {value}')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only monitor for the public market breadth feed.')
    parser.add_argument('--notify', choices=['github-issue'], help='Open, update and close incident issues')
    parser.add_argument(
        '--incident-author', help='Login that owns incident issues (required for --notify outside Actions)'
    )
    parser.add_argument('--repo', default=os.environ.get('GITHUB_REPOSITORY') or DEFAULT_REPO)
    parser.add_argument('--min-coverage', type=_positive_float, default=0.9)
    parser.add_argument('--delivery-grace-hours', type=_positive_float, default=24)
    parser.add_argument('--reminder-hours', type=_positive_float, default=72)
    parser.add_argument('--failure-threshold', type=int, default=2)
    args = parser.parse_args(argv)
    if args.failure_threshold < 1:
        parser.error('--failure-threshold must be at least 1')
    result = run_monitor(
        http=requests.Session(),
        repo=args.repo,
        token=os.environ.get('FEED_MONITOR_TOKEN') or None,
        notify=args.notify is not None,
        in_actions=os.environ.get('GITHUB_ACTIONS') == 'true',
        author=args.incident_author,
        assignee=os.environ.get('FEED_MONITOR_ASSIGNEE') or None,
        min_coverage=args.min_coverage,
        grace_hours=args.delivery_grace_hours,
        failure_threshold=args.failure_threshold,
        reminder_hours=args.reminder_hours,
    )
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
