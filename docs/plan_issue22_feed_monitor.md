# Plan: Independent feed monitoring and recovery runbook (Issue #22)

## Goal

Detect, independently of the producer workflow's own success, when the public
market breadth feed stops or degrades, raise one deduplicated notification per
incident, clear it on recovery, and document cause-specific recovery.
Keepalive (Issue #2 / PR #3) is out of scope and is not reimplemented.

## Observed baseline (2026-11-13 15:30 UTC, read-only GETs)

- Producer workflow `daily-market-breadth.yml` state `active`; recent runs are
  scheduled and successful.
- Scheduled runs start hours late: in the last 40 runs, 0.3 to 10.3 hours.
- Public `market_date_resolution.json` reports `selected_date` 2026-11-12,
  `confidence` `observed`, `publication_eligible` true.
- Public data CSV ends on 2026-11-12 with `Coverage_200` 0.994 and `Coverage_50` 0.998.

The delivery grace therefore has to cover at least one fully delayed scheduled run.

## Design

### Component: `scripts/monitor_feed.py`

A read-only CLI and library function `run_monitor(...)`. The HTTP client, the clock,
the sleep function, the token and the repository are injected, so tests never touch
the network. It reuses `validate_csv_content` from `feed_contract.py` (moved out of `trigger_market_breadth.py`, see below) and
`resolve_market_date` from `market_dates.py`. It never dispatches a workflow.
`determine_freshness()` is not reused because it hardcodes a 30-minute grace.

Each check returns `OK`, `WARN`, `INCIDENT` or `UNKNOWN` with a stable code and a
redacted message.

| Code | Source | INCIDENT when |
|---|---|---|
| `producer_disabled` | `GET /actions/workflows/daily-market-breadth.yml` | `state` is not `active` |
| `producer_failing` | `GET .../runs?branch=main&status=completed&per_page=20` | the latest N decisive runs (default 2) all failed. `failure`, `timed_out` and `startup_failure` count as failed and `success` as succeeded; every other conclusion (`cancelled`, `skipped`, `neutral`, `action_required`, `stale`) is ignored. One latest failure is `WARN`. No decisive run among the 20 is `UNKNOWN` |
| `feed_invalid` | data and summary CSVs | download still fails after retries, or `validate_csv_content` rejects them |
| `feed_stale` | data CSV dates | the latest date is older than `expected_date` from `resolve_market_date(lambda *_: [], grace_minutes=G)`, the function the publication gate uses, but with the monitor's own grace instead of the gate's 30 minutes. A session counts as expected only once its close plus the delivery grace G (default 24 h) has passed. This does not depend on the resolver's 14-day observation window, so a feed weeks old is still an INCIDENT |
| `coverage_low` | latest CSV row | `Coverage_200` or `Coverage_50` is missing, not finite, or below the minimum (default 0.9, same as the publication gate) |
| `provenance_invalid` | `market_date_resolution.json` | missing, malformed, not `observed` and `publication_eligible`, or `selected_date` differs from the latest CSV date |

`provenance_invalid` overlaps the publication gate on purpose: anything the
producer publishes has passed the gate, so this check catches a deploy made
outside the producer or a partially updated site.

`UNKNOWN` is used when the monitor itself cannot judge: the GitHub API is
unreachable or returns an unexpected payload, or `exchange_calendars` is
unavailable (`confidence == 'weekday_only'`). Without a calendar the monitor does
not guess holidays and reports `feed_stale` as `UNKNOWN`.

Holidays, weekends and early closes come from the XNYS calendar. The expected
session only moves forward as time passes, so the result does not depend on how
late the monitor's own scheduled run starts. Friday sessions get only one
producer attempt (the Friday 22:00 UTC cron), so a failure of that run is a true
positive over the weekend.

Fetching:
- Every GET retries up to 3 times with backoff on connection errors, timeouts
  and 5xx.
- Pages caches each file separately for 600 s and ignores query strings, so
  cache-busting does not work. If the CSV, summary and JSON disagree (a deploy
  in progress), the monitor waits until every cached copy has expired,
  `max(600 - Age) + 30` seconds from the responses' `Age` headers (630 s when a
  header is missing), and fetches all three once more before reporting
  INCIDENT. The job timeout is 30 minutes to fit that wait, two fetch rounds
  with retries and the API reads.

Overall status: `INCIDENT` if any check is `INCIDENT`; otherwise `UNKNOWN` if any
check is `UNKNOWN`; otherwise `HEALTHY`. Warnings are reported but do not open
an incident.

### Notification and incident state

- Default mode is a dry run: print the report and the action it would take,
  write nothing. Exit codes: 0 healthy, 1 incident, 2 unknown or configuration error.
- `--notify github-issue` keeps one open GitHub issue per incident.
  - Identification does not use labels, because the repository has no
    suitable label and issue creation silently drops labels it cannot apply.
    The monitor lists all open issues (every page, pull requests skipped) and
    keeps those whose `user.login` equals the configured author (default
    `github-actions[bot]`), whose `user.type` is `Bot` for that default, and
    whose body contains the state block. The author filter runs in code, not
    through the `creator` query parameter. Issues opened by anyone else are never
    adopted, even if they copy the marker. If several match, the oldest is
    the incident issue and the others are reported; on recovery all of them
    are closed.
  - State lives in the bot-owned issue body, in a hidden block
    `<!-- feed-monitor:state {"fingerprint": "...", "last_notified_at": "..."} -->`,
    updated with PATCH. Comments are never read, so their count does not matter.
  - The fingerprint is the sorted, comma-joined set of INCIDENT check codes
    only. Messages, dates and values are excluded, so a persistent stale feed
    whose expected date moves keeps the same fingerprint.
  - INCIDENT and no open incident issue: create one. The title is fixed text
    plus the codes, never feed text. If `FEED_MONITOR_ASSIGNEE` is non-empty it
    is assigned; an empty value means unset. If creation with the assignee
    returns 422, the monitor retries once without it and reports a WARN.
  - INCIDENT, same fingerprint, notified less than the reminder interval ago
    (default 72 h): no write.
  - INCIDENT with a changed fingerprint: comment immediately. Reminder interval
    elapsed: one reminder comment. Both update the state block.
  - HEALTHY and open incident issues: add a recovery comment to each and close them.
  - UNKNOWN: never create, comment on or close an issue, so a monitor failure
    cannot clear an incident.
  - If a person closes an issue while the incident persists, the next run opens
    a new one. A flapping feed opens and closes issues. Both are documented.
- Feed-derived text goes into fenced code blocks. Backticks are replaced,
  control characters stripped, `@` neutralized and each message capped at 500
  characters, so issue content cannot close the fence, mention users or inject
  markdown.
- In notify mode the exit code reports whether the monitor and the notification
  worked: 0 when the issue state is in sync, 2 on UNKNOWN or a failed write.
  The incident itself lives in the issue, so a persistent incident does not
  fail the workflow every day.
- The write token is read only from `FEED_MONITOR_TOKEN`, never from
  `GITHUB_TOKEN`. `validate_csv_content` and the feed URLs move to a new module
  `feed_contract.py` that does not call `load_dotenv()`; `trigger_market_breadth`
  re-exports them unchanged. The monitor therefore never loads `.env`, and only
  `main()` reads the process environment; `run_monitor()` takes the token and
  assignee as arguments.
- The fingerprint carries forward: a code that was in the previous fingerprint
  and is `UNKNOWN` in this run stays in it, so a one-run API blip does not post
  two "changed" comments.
- A 404 from the producer workflow endpoint (deleted or renamed workflow) is a
  `producer_disabled` INCIDENT; network errors, 5xx and malformed payloads stay
  `UNKNOWN`.
- After creating an issue with an assignee, the monitor compares the response's
  `assignees` with the request, because the API silently drops logins it cannot
  assign, and reports a WARN when they differ.
- `--min-coverage` accepts any finite value above 0, including values above 1
  that force a `coverage_low` incident for the smoke test.
- `--notify` outside GitHub Actions (`GITHUB_ACTIONS != 'true'`) requires
  `--incident-author <login>`, because a personal token creates issues as that
  user. Such issues form a separate set that Actions runs never adopt, so one
  incident can produce two open issues and a recovery seen by one side does not
  close the other's. The docs say this plainly and recommend dry runs locally.
- `--notify` without `FEED_MONITOR_TOKEN` prints a diagnostic, falls back to the
  dry run and exits 2.
- Only issues (title, body, labels, assignees, comments, state) are written. The
  token is registered with `secret_redaction` and never printed. Reads use the
  token when present and work unauthenticated otherwise (public repository).

### Workflow: `.github/workflows/feed-monitor.yml`

- Separate workflow file, so a producer build failure, deploy failure or a
  manual disable of the producer does not stop it.
- `schedule: '30 23 * * *'` daily, plus `workflow_dispatch` with a boolean
  `notify` input that defaults to false and a string `min_coverage` input that
  defaults to `0.9` (used for the post-merge smoke test).
- Notification runs only when `github.ref` is the default branch (the producer's
  expression) and `github.event_name == 'schedule' || inputs.notify == true`,
  using the boolean `inputs` context; any other run is a dry run.
- `actions/checkout` with `persist-credentials: false`; `setup-python` 3.11.
- `permissions: contents: read, actions: read, issues: write`; no FMP or other
  secrets. `FEED_MONITOR_TOKEN: ${{ secrets.GITHUB_TOKEN }}` and
  `FEED_MONITOR_ASSIGNEE: ${{ vars.FEED_MONITOR_ASSIGNEE }}`.
  `concurrency: feed-monitor`, `timeout-minutes: 30`. Dispatch inputs reach
  the shell only through `env:`, never as `${{ inputs.* }}` inside `run:`.
- Installs only `requests`, `python-dotenv`, `pandas` and `exchange-calendars`
  with the same specifiers as `requirements.txt`; a test keeps them in sync.

### Documentation

- `docs/feed_monitoring.md`:
  - what is checked and why, thresholds and grace, the notification lifecycle,
    configuration and permissions;
  - who receives which signal: the incident issue reaches repository watchers
    and the optional assignee; a monitor run that exits 2 fails the workflow,
    and GitHub emails the user who last changed its schedule;
  - running the monitor outside GitHub, as a concrete recipe: a sample cron
    line and a launchd plist that run the dry run and raise a local alert
    (mail or a desktop notification) on a nonzero exit, because a nonzero exit
    alone notifies nobody. The docs state which failure modes only this path
    detects: repository-wide `disabled_inactivity` (both workflows are disabled
    together) and loss of the Actions monitor itself. Local `--notify` needs a
    fine-grained token limited to issues write and `--incident-author`;
  - an entry for unscheduled market closures missing from the installed
    `exchange_calendars` release, which raise a false `feed_stale`;
  - a post-merge smoke test, run only with the owner's approval: dispatch with
    `notify: true` and `min_coverage: 1.01` to force a `coverage_low` incident,
    confirm the issue, then dispatch with `notify: true` and the default to
    confirm the recovery comment and close;
  - a recovery runbook per code, with the publication gate `reason_code` values
    from `publish_market_breadth.py`, ending with enable → refresh → verify:
    `gh workflow enable`, `gh workflow run` or `trigger_market_breadth.py --trigger-only`,
    inspect the `publication-validation` artifact `reason_code`, rerun the monitor
    and confirm exit 0 and the issue closed.
- README.md and README-ja.md: a short section linking to the runbook.

## TDD test list (`tests/test_feed_monitor.py`)

Fake HTTP client, fixed clocks and a no-op sleep; an autouse fixture blocks sockets.
Calendar-dependent tests use `pytest.importorskip('exchange_calendars')`; CI installs it.

1. Healthy feed and producer: `HEALTHY`, exit 0, no writes in dry run.
2. Disabled producer (`disabled_inactivity`, `disabled_manually`): `producer_disabled` INCIDENT.
3. Two consecutive failed runs: `producer_failing` INCIDENT. One failure then success: OK. One latest failure: WARN only. Cancelled and other ignored conclusions are skipped. The request filters `branch=main` and `status=completed`.
4. Stale CSV: latest 2026-11-11 at 2026-11-13 23:30 UTC: `feed_stale` INCIDENT.
5. Delivery grace: latest 2026-11-11 at 2026-11-13 20:00 UTC, before the 11-12 close + 24 h: OK.
6. Thanksgiving: latest 2026-11-25 at 2026-11-27 23:30 UTC: OK. Weekend: latest 2026-11-13 at 2026-11-15 23:30 UTC: OK.
   Early close: latest 2026-11-25 at 2026-11-28 17:00 UTC: OK, because 11-27 closes at 18:00 UTC; at 2026-11-28 23:30 UTC: stale.
   Long outage: latest about 20 sessions old: `feed_stale` INCIDENT, not UNKNOWN.
7. Coverage below 0.9 and missing coverage column: `coverage_low` INCIDENT.
8. HTML or malformed CSV, download error after retries: `feed_invalid` INCIDENT. One transient 5xx then success: OK. A CSV/summary mismatch that is consistent on re-fetch: OK; still inconsistent: INCIDENT. The wait before the re-fetch follows the `Age` headers.
8a. Workflow endpoint 404: `producer_disabled` INCIDENT.
8b. A previous incident code that is UNKNOWN in this run stays in the fingerprint; no comment is posted.
8c. An assignee missing from the create response: WARN.
8d. Importing the monitor in a fresh process with a `.env` holding `FEED_MONITOR_TOKEN` leaves the environment and `dotenv` untouched.
9. Provenance missing, not observed, or date mismatch: `provenance_invalid` INCIDENT.
10. GitHub API error or calendar unavailable: `UNKNOWN`, exit 2.
11. Notify: an incident without an open issue creates one with a fixed title, the state block and an optional assignee. Empty assignee is unset. A 422 with an assignee retries once without it and reports WARN.
12. Notify: same fingerprint within the reminder interval: no write. After it: one comment and a state update. Changed fingerprint: one comment. A persistent stale feed whose expected date moves keeps its fingerprint.
13. Notify: a matching issue from another author is ignored; several bot issues use the oldest; issue listing is paginated; pull requests are skipped.
14. Feed text with `@user`, backticks, control characters and very long values is neutralized and capped.
15. Notify: healthy with open issues: a recovery comment on each, then close them all.
15a. `producer_failing`: no decisive run among 20: UNKNOWN.
15b. Local `--notify` without `--incident-author` outside Actions: diagnostic, no write, exit 2. `FEED_MONITOR_TOKEN` present only in `.env`: ignored.
16. Notify: UNKNOWN with an open issue: no write, exit 2.
17. Notify without `FEED_MONITOR_TOKEN`, even with `GITHUB_TOKEN` set: diagnostic, no write, exit 2.
18. The token never appears in output, including HTTP error messages.
19. Workflow file: schedule and dispatch with `notify` defaulting to false, notification gated to the default branch with the boolean `inputs` context, `persist-credentials: false`, minimal permissions, no secrets other than `GITHUB_TOKEN`, dependency specifiers equal to `requirements.txt`, no reference to the producer's jobs.

## Validation

- Full suite in a CI-like venv, `ruff check`, `ruff format --check`, `codespell`,
  `bandit`, `mypy` as configured.
- One read-only dry run against the live feed and GitHub API. No issue is
  created and no workflow is dispatched during implementation.
- The PR records the tested revision, evidence and remaining limitations.

## Remaining limitations (expected)

- The monitor runs on GitHub's scheduler. If the whole repository becomes
  inactive for 60 days, GitHub disables both workflows, and loss of the monitor
  itself is not detected. The documented path outside GitHub is a local cron or
  launchd job, as a dry run that the operator watches or with `--notify`. A
  second keepalive is not added, because Issue #2 owns keepalive.
- Repository-wide `disabled_inactivity`, the failure most like Issue #2,
  disables the monitor together with the producer. Only the off-GitHub recipe
  detects it. If the producer is disabled manually, its keepalive stops, and
  about 60 days later the monitor is disabled too; the incident is reported
  long before that.
- Issue creation, assignee handling and GitHub notifications are verified
  only against fakes until the post-merge smoke test runs with the owner's
  approval.
- Detection latency is up to about one day plus the delivery grace.
  A persistent incident gets a reminder at most every 72 hours.
- Coverage and provenance checks validate the published contract, not the
  underlying provider data.
- The notification destination is GitHub issues and watchers' notification
  settings; email or chat delivery is not added.
