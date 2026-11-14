# Feed monitoring and recovery runbook (Issue #22)

`scripts/monitor_feed.py` checks the public feed and the producer workflow
(`daily-market-breadth.yml`) independently of the producer's own success.
It is read-only by default and never dispatches a workflow.
Keepalive belongs to Issue #2 and is not part of the monitor.

## What is checked

| Code | INCIDENT when |
|---|---|
| `producer_disabled` | The producer workflow state is not `active`, or the workflow no longer exists (404). |
| `producer_failing` | The two latest decisive runs on `main` failed (`failure`, `timed_out`, `startup_failure`). One failure is only a warning. `cancelled`, `skipped`, `neutral`, `action_required` and `stale` are ignored. |
| `feed_invalid` | The data or summary CSV cannot be downloaded (connection errors, timeouts and 5xx are retried up to 3 attempts; a 4xx such as 404 fails at once), or fails the content contract shared with `trigger_market_breadth.py` (`feed_contract.validate_csv_content`). |
| `feed_stale` | The latest CSV date is older than the expected XNYS session. A session is expected once its close plus the delivery grace (default 24 hours) has passed. Weekends, holidays and early closes come from `exchange_calendars`. |
| `coverage_low` | `Coverage_200` or `Coverage_50` on the latest row is missing or below the minimum (default 0.9, the publication gate's value). |
| `provenance_invalid` | `market_date_resolution.json` is missing or malformed, is not `observed` and `publication_eligible`, or names a different date than the CSV. Anything the producer publishes has passed this gate, so this catches deploys made outside the producer and partially updated sites. |

A check is `UNKNOWN` when the monitor cannot judge it: the GitHub API fails,
returns 5xx or an unexpected payload, or `exchange_calendars` is unavailable.
The monitor does not guess holidays.

Why a 24-hour grace: scheduled producer runs often start hours late (0.3 to
10.3 hours across 40 runs before 2026-11-13). With 24 hours, each weekday
session gets both producer attempts before an alarm. A Friday session gets
only the Friday 22:00 UTC attempt, so a failed Friday run is a real weekend
outage.

Deploys in progress: GitHub Pages caches each file for 600 seconds and ignores
query strings. If the CSV, summary and JSON disagree, the monitor waits until
every cached copy has expired (`600 - Age + 30` seconds, at most 630) and checks
once more before reporting an incident.

## Modes and exit codes

```bash
python scripts/monitor_feed.py                        # dry run: report and planned action
python scripts/monitor_feed.py --notify github-issue  # keep the incident issue in sync
```

| Mode | 0 | 1 | 2 |
|---|---|---|---|
| Dry run | healthy | incident | unknown or configuration error |
| `--notify` | issue state is in sync, including an open incident | (not used) | unknown, configuration error or a failed write |

Options: `--min-coverage` (any value above 0; a value above 1 forces an
incident for testing), `--delivery-grace-hours` (24), `--reminder-hours` (72),
`--failure-threshold` (2), `--repo`, `--incident-author`. `--repo` changes
only the repository used for the GitHub API (producer state, runs and
issues); the feed URLs always point at the tradermonty Pages site.

## Incident issues

- With `--notify github-issue`, one open issue represents one incident. The
  title is `Feed monitor incident: <codes>`, and the body carries a hidden
  state block with the fingerprint and the last notification time.
- Only issues created by the expected author (`github-actions[bot]` in Actions)
  that carry the state block are adopted. Anyone else's issue is ignored, even
  if it copies the marker. Labels are not used.
- The fingerprint is the sorted list of INCIDENT codes, without dates or values.
  A persistent stale feed keeps the same fingerprint as days pass. A code that
  was in the incident and is `UNKNOWN` in this run is kept, for as long as it
  stays `UNKNOWN`, so the title can name a code that is not confirmed in the
  latest run; the check table in each comment shows the current states.
- A new fingerprint posts a comment right away. An unchanged incident posts a
  reminder about every 72 hours; up to 12 hours early is accepted because the
  daily run's start time drifts.
- The state block is saved before the comment, and an issue is closed before
  its recovery comment, so a failed comment is not repeated by the next run.
  State updates are retried on 5xx; new issues and comments are not, so they
  cannot be duplicated.
- If someone removes the state block from an incident issue, the monitor no
  longer recognizes it: it is not closed on recovery, and the next incident
  opens a new issue. Close such an issue by hand.
- When every check passes, each open incident issue gets a recovery comment
  and is closed. `UNKNOWN` never opens, comments on or closes an issue.
- If a person closes the issue while the incident continues, the next run opens
  a new one. A flapping feed opens and closes issues.
- Feed text in issues is put in code blocks, with backticks replaced, control
  characters removed, `@` neutralized and a 500-character cap per message.

## Who receives what

| Signal | Recipient |
|---|---|
| A new incident issue, and its comments | Repository watchers (according to their notification settings), and `FEED_MONITOR_ASSIGNEE` if set |
| A monitor run that exits 2 | GitHub's failed-workflow email to the user who last changed the monitor's schedule |
| A producer run that fails | Same, for the producer workflow |

Set the optional repository variable `FEED_MONITOR_ASSIGNEE` (Settings →
Secrets and variables → Actions → Variables) to a collaborator's login. An
empty value means no assignee. If GitHub rejects the login or silently drops
it, the issue is still opened and the run reports a warning.

## GitHub Actions configuration

`.github/workflows/feed-monitor.yml` runs daily at 23:30 UTC and on manual
dispatch. It is a separate workflow, so a producer build failure, deploy
failure or a manual disable of the producer does not stop it.

- Permissions: `contents: read`, `actions: read`, `issues: write`. The only
  secret is the built-in `GITHUB_TOKEN`, passed as `FEED_MONITOR_TOKEN`.
  The checkout does not keep credentials.
- Issues are written only on the default branch, for the schedule or a
  dispatch with `notify: true`. Every other run is a dry run.
- Dispatch inputs reach the script only through environment variables.

The monitor reads its write token only from `FEED_MONITOR_TOKEN`. It never
uses `GITHUB_TOKEN` and never loads `.env`, where the local dispatch token lives.

## Running outside GitHub

The GitHub-scheduled monitor cannot detect two failures:

- Repository-wide inactivity. After 60 days without activity, GitHub disables
  every scheduled workflow, including this monitor (`disabled_inactivity`, the
  failure behind Issue #2).
- Loss of the monitor workflow itself.

If the producer is disabled manually, its keepalive stops too. The monitor
reports the producer at once, but about 60 days later it is disabled as well.

A local scheduled dry run covers these. A nonzero exit alone notifies nobody,
so wrap it with an alert. Use the absolute path of a virtual environment that
has `requirements.txt` installed: cron and launchd start with a minimal `PATH`,
and a system `python3` without `requests`, `pandas` and `exchange_calendars`
would fail, and alert, every day. cron example, daily at 23:45 UTC on a machine whose
clock is UTC:

```cron
45 23 * * * cd /path/to/market-breadth-analysis && /path/to/venv/bin/python scripts/monitor_feed.py > /tmp/feed-monitor.log 2>&1 || mail -s "Feed monitor: exit $?" you@example.com < /tmp/feed-monitor.log
```

launchd on macOS (`~/Library/LaunchAgents/com.example.feed-monitor.plist`,
then `launchctl load` it):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.example.feed-monitor</string>
  <key>ProgramArguments</key><array>
    <string>/bin/sh</string><string>-c</string>
    <string>cd /path/to/market-breadth-analysis &amp;&amp; /path/to/venv/bin/python scripts/monitor_feed.py || osascript -e 'display notification "Feed monitor needs attention" with title "Market breadth"'</string>
  </array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer></dict>
</dict></plist>
```

Local dry runs are recommended. `--notify` outside Actions needs a
fine-grained token limited to this repository, with Issues read and write
and Actions read (the token is also sent on the workflow and run reads), in
`FEED_MONITOR_TOKEN`, and `--incident-author <your login>`. Issues created
that way belong to you, not to `github-actions[bot]`: Actions runs never adopt
or close them, and local runs never adopt the bot's. One incident can then have
two open issues, and a recovery seen by one side does not close the other's.

## Recovery runbook

Start from the codes in the incident issue or the dry-run report.

### `producer_disabled`

1. Check the state: `gh workflow view daily-market-breadth.yml`.
2. `disabled_inactivity`: run `gh workflow enable daily-market-breadth.yml`,
   then make sure the keepalive job succeeds on the next scheduled run.
3. `disabled_manually`: confirm with the owner why it was disabled before
   enabling it.
4. 404: the workflow file was renamed or deleted. Restore it, or update
   `PRODUCER_WORKFLOW` in `scripts/monitor_feed.py`.

### `producer_failing`

1. Open the latest failed run linked in the issue.
2. Download the `publication-validation` artifact and read `validation.json`.
   Its `reason_code` comes from `scripts/publish_market_breadth.py`:

| `reason_code` | Meaning and first action |
|---|---|
| `PREREQUISITE`, `SOURCE`, `CONFIG`, `STATIC` | Checkout or configuration problem. Check the last change to `main` and the static `reports/` assets. |
| `ANALYSIS`, `ANALYSIS_TIMEOUT` | The analysis failed or exceeded 720 s. Check the run log for FMP errors, rate limits or an expired `FMP_API_KEY`. |
| `CALENDAR` | `exchange_calendars` could not provide sessions. Check the installed version and its calendar horizon. |
| `FRESHNESS`, `PROVENANCE` | Provider data for the expected session was not available yet. Usually fixed by the next scheduled run; otherwise dispatch once later. |
| `COVERAGE` | Too many constituents are missing. Check FMP availability. |
| `ARTIFACT`, `HTML`, `DATE`, `VALUES`, `SUMMARY`, `INVALID_OUTPUT` | Generated outputs failed validation. Treat as a code or data regression and inspect the diff since the last success. |
| `OUTPUT_EXISTS`, `UNKNOWN` | Unexpected runner state. Rerun once; if it repeats, inspect the log. |

3. Fix the cause, then refresh as below.

### `feed_stale`

1. Check whether the producer ran: `gh run list --workflow daily-market-breadth.yml --limit 5`.
2. If no run is in progress, refresh once: `gh workflow run daily-market-breadth.yml --ref main`,
   or `python trigger_market_breadth.py --trigger-only`.
3. If runs fail, follow `producer_failing`.
4. If the alarm fires on a day the exchange was closed without notice, the
   installed `exchange_calendars` release does not know the closure, for
   example one announced after that release was published. Upgrade
   `exchange-calendars` within the pinned range, then close the issue by hand.

### `feed_invalid` and `provenance_invalid`

1. Open the public files directly:
   `https://tradermonty.github.io/market-breadth-analysis/market_breadth_data.csv`,
   `market_breadth_summary.csv` and `market_date_resolution.json`.
2. An HTML page or 404 means the Pages deployment is broken. Check the
   producer's `deploy` job and the repository's Pages settings.
3. Mismatched files that persist beyond 10 minutes mean a partial or manual
   deploy. Refresh through the producer.

### `coverage_low`

Check `Missing_Count_200` and `Missing_Count_50` on the latest rows of the
data CSV. A rise usually means FMP returned no data for many constituents.
Refresh after the provider recovers.

### Verify recovery

1. Enable or fix the producer.
2. Refresh: `gh workflow run daily-market-breadth.yml --ref main`, and wait
   for the run to succeed.
3. Verify the public CSV: `python trigger_market_breadth.py --fetch-only`
   prints the latest data date.
4. Run `python scripts/monitor_feed.py` and confirm exit 0.
5. Close the incident: dispatch the Feed Monitor workflow with `notify: true`,
   or wait for the next scheduled run. Confirm the issue has a recovery
   comment and is closed.

## Post-merge smoke test

Run this only with the repository owner's approval, because it creates and
closes a real issue.

1. Dispatch Feed Monitor on `main` with `notify: true` and `min_coverage: 1.01`.
   Expect a new `coverage_low` incident issue.
2. Dispatch again with `notify: true` and the default `min_coverage`. Expect a
   recovery comment and the issue closed.

Until this runs, issue creation, assignment and notification are verified
only against test fakes.
