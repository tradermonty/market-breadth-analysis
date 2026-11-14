# Issue #21: deterministic critical-path validation

## Selection and isolation

Base main `6af9e4a`. Issue #21 is unassigned with no comments or related open PR.
P2 is the highest remaining implementation priority; #18 depends on the open
#17/PR #38 and is deferred. #4 is a tracker; #36 needs policy answers, and
#19/#20 are P3. Existing worktrees cover #13, #22 and #17. Work is isolated in
`/private/tmp/market-breadth-issue21-20261114`, branch
`feat/issue-21-critical-validation`. Preserve other worktrees and local state.

PR #38 owns dependency locking, existing CI/producer/monitor workflows,
installation docs, the individual API test guard, and rendering-backend fixes.
This task does not modify those files. Introduce a separate validation workflow
and config/test files; use existing dependency installation until #38 merges.
No prerequisite PR remains for #21 itself. Never merge/deploy or contact a live broker/provider.

## Final behavior to build

1. Add a repository-wide API-free test mode controlled by SKIP_API_TESTS=true.
   Explicitly skip the known real-API tests even when a credential is present.
   Block and record network attempts (socket connect/create_connection and DNS)
   throughout collection, fixtures, tests and Python subprocesses. Detect even
   swallowed connection errors through an attempt log checked by a wrapper.
   Intentional negative probes use a nested wrapper and distinct attempt log;
   assert that inner validation fails while the outer run remains clean. This
   is regression protection for Python code, not a security sandbox or a
   claim to block arbitrary non-Python network clients.
   Do not log credentials or payloads. Normal opt-in integration runs remain
   available when API-free mode is disabled. Add subprocess probes proving
   caught requests and child requests cannot yield a successful validation.
2. Measure branch and line coverage for acquisition, calendar/calculation,
   single/multi backtests, comparison/trigger/publication and order recovery.
   Commit an explicit coverage configuration, a reproducible report/checker,
   and documented per-module floors based on measured current behavior.
   Set floors only after baseline measurement; document uncovered lines and
   branch exceptions without claiming all behavior validated or an arbitrary
   100% target. Publish JSON/HTML and coverage-gap reports even on failures.
   Make a missing/invalid report, missing measured file or a lower-than-reviewed
   module line/branch floor fail CI. Also enforce a checked-in list of required
   critical pytest selectors: each must appear and pass without a skip in JUnit.
   The wrapper requires a new output directory and generates a fresh run ID.
   Bind JSON/JUnit digests, branch-enabled metadata, tool version and hashes of
   measured source/tests/config to that run; the gate requires the expected ID
   and compares source hashes to the current checkout. Regression-test the
   coverage gate with stale run IDs/source, missing/skipped obligations and
   deliberate missing/empty/low-coverage inputs.
3. Add deterministic socket-blocked E2E fixtures: real saved-data analysis
   generating CSV/HTML, a single ETF backtest and a multi-ETF run, trigger CLI
   exit status and comparison CLI PASS/FAIL. Use synthetic cached prices with
   fixed seeds/clocks, explicit dates, isolated cwd/output/state paths, and no
   dependence on external market data. Reuse existing publication corruption
   and restart/lost-response tests as explicit critical-branch obligations.
   Preserve raw expected failure results; never accept faulted outputs as PASS.
4. Extend static checks to feed_contract.py and trade/order_state.py in a
   separate mypy config. Add useful boundary annotations without trading
   behavior changes, with documented incremental exemptions for dynamic broker
   payloads/legacy untyped internals. Include a deliberate invalid call probe
   that fails this type gate.
5. Add an independent Ubuntu validation workflow (Python 3.11) that installs
   the existing runtime requirements and a small pinned validation-tool set,
   runs API-free branch measurement + coverage gates and the incremental type
   gate, and uploads diagnostics unconditionally. Do not alter #38's existing
   workflow or dependency inputs. Track subprocess coverage for the new E2Es;
   record source revision and interpreter/tool versions with reports.

## Verification and documentation

Use TDD for the guard/gates and incremental annotations. First run the existing
suite to establish coverage and separate baseline failures from changes.
Demonstrate deliberate network attempts, stale/missing coverage, broken required
artifacts, comparison mismatch and invalid typed input fail their actual gates.
Keep an acceptance checklist mapping each critical branch to an existing/new
test, measured coverage, and explicit gaps. Explain how to reproduce reports,
run API-free versus opt-in integration tests, and distinguish fixture evidence
from provider/broker/live acceptance. Run full regressions, required existing
CI gates and the new workflow; clean only this worktree's test side effects.

## Review and delivery

Independent plan review and revision: maximum 3 rounds, final only when all
findings are resolved. Implement after the final plan. Independent code review
and remediation: maximum 3 rounds. Commit and create an English PR after all
findings and required local checks pass, then verify hosted checks. Do not
merge in this task.

## Review log

Round 1: two P2 findings addressed: enforce non-skipped critical obligations and fresh source/run-bound reports; isolate deliberate network-failure probes in nested logs. Numerical floors await the instrumented baseline before confirmation review.

## Measured baseline and enforced initial floors

Instrumented baseline on macOS, Python 3.11, coverage 7.17.3: 734 passed,
4 skipped. Line/branch percentages below include subprocess measurement.
Floors round down measured coverage to leave small cross-platform margins;
they prevent regression, not a claim that those percentages establish safety.
The required non-skipped critical selectors independently enforce safety paths.

| File | Baseline line / branch | Initial floor line / branch |
|---|---:|---:|
| fmp_data_fetcher.py | 27.1 / 11.3 | 25 / 10 |
| market_dates.py | 95.6 / 90.9 | 90 / 85 |
| market_breadth.py | 84.0 / 69.4 | 80 / 65 |
| feed_contract.py | 95.8 / 100.0 | 90 / 95 |
| backtest/backtest.py | 78.7 / 66.8 | 75 / 60 |
| backtest/run_multi_etf_backtest.py | 33.7 / 25.0 | 30 / 20 |
| scripts/compare_tv_python_trades.py | 66.8 / 65.8 | 60 / 60 |
| trigger_market_breadth.py | 81.1 / 82.0 | 75 / 75 |
| scripts/publish_market_breadth.py | 89.6 / 97.6 | 85 / 90 |
| trade/order_state.py | 87.0 / 70.5 | 80 / 65 |
| trade/run_market_breadth_trade.py | 83.6 / 78.3 | 80 / 70 |

The baseline shows broad provider acquisition and multi-ETF orchestration are
weakly covered. Add fake-transport acquisition success/failure tests and a real
multi-ETF synthetic run, then raise those floors if final measurements permit.
Provider-specific endpoints, interactive/browser/platform UI, live broker
responses and current external data remain explicit gaps in the report.

Required test obligations will include acquisition retry/error redaction,
new analysis/single/multi/trigger E2Es, existing comparison mismatch PASS/FAIL,
publication invalid artifacts/low coverage/nonzero CLI, and restart/partial
fill/duplicate/corrupt-state recovery tests. Use exact pytest selectors in the
checked-in policy, reject absent/skipped/failed JUnit cases, and demonstrate
that removing or skipping an obligation fails despite unchanged coverage.

Coverage JSON will list the actual missing lines and arcs; no executable
critical module is omitted to improve percentages. Exclude tests and independent
untracked subprojects from the policy, not from the raw measurement. Keep the
low-baseline scopes visible and trace gaps rather than averaging them away.

Round 2: no remaining actionable findings. Plan FINAL. Use temporary copies for existing security-test expiry fixtures so source identity remains unchanged.

## Implementation/code review log

Round 1: P2 source identity omitted the policy and pytest config; added policy, pyproject, runtime/tool requirements and workflow configuration fingerprints with mutation tests. P2 direct pytest did not inject bootstrap for env=None children; normalized inherited environment on install and added a pristine -S/conftest/default-child probe.

Initial validation: 765 passed, 4 skipped; 22 subprocess data files combined (28 duplicates skipped), no network attempts, all gates passed. Provider line/branch improved to 45.4/32.3 and multi-ETF to 88.1/54.2. Tightened floors to 40/30 and 85/50 respectively. Added a child-only sentinel with 95/95 floors to reject lost child instrumentation.

Round 2 code review: original P2 fixes confirmed; collection string/import errors fixed before the successful final run. Round 3: optional missing-distribution provenance could abort report generation; record absence explicitly and test it. Final independent review: no remaining actionable findings. Final local run: 768 passed, 7 skipped, all coverage/obligation/network gates PASS.
