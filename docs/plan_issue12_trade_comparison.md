# Issue #12: trustworthy TradingView trade and mode comparisons

## Selection and boundaries

- Issue #12 (P2) can start independently; no assignee/claim comments or competing implementation branch/PR found on selection.
- Outstanding P1 work is in PRs #25–#29 or depends on unfinished #7/#8 (#9). PR #30 owns backtest liquidation. This work changes only the two comparison scripts, new comparison tests, focused comparison documentation and this plan; no backtest engine, breadth, live trading, CI/security or dependency changes.
- Base `9eebcfe00aee81df5d8e63157e071c206cd0d651`; new worktree `/private/tmp/market-breadth-issue12-20261016`, branch `fix/issue-12-trade-comparison`.
- Visible evidence is open Issues/PR file lists, remote branches and local worktrees (including the isolated clone's worktrees); invisible external-agent activity cannot be excluded.

## Acceptance policy

1. Compare every closed trade after stable chronological sorting by entry date, exit date and normalized exit reason. Require nonempty, equal counts, every paired entry/exit within a nonnegative integer bar tolerance, and 100% reason agreement. First mismatch is diagnostic only. Expose max entry/exit/overall bar differences and failing trade indexes; never accept based on the first mismatch alone.
2. Exact trade diagnostics preserve multiplicity with counters rather than sets. Matching duplicate closed-trade occurrences are legitimate and must pass when every paired row meets the date/reason policy; excess/missing occurrences remain visible and fail count/per-trade criteria. Insertion/removal causes count and/or per-trade failures; do not invent fuzzy matching to force a PASS. Multiplicity diagnostics do not replace the tolerance-based acceptance rule.
3. Price calendar must be nonempty, valid, unique daily dates; sorted before use. All trade dates must be valid, ordered entry<=exit, timezone-naive normalized dates present in that calendar. No calendar-day fallback may establish acceptance. Invalid inputs produce a clear FAIL/error and nonzero CLI exit. Equal empty files do not establish equivalence. Reject negative tolerance.
4. Validate required CSV fields without dropping malformed closed rows. Normalized TV schema requires entry/exit dates and reason. Python requires entry/exit dates and exit_reason. Missing/blank reasons cannot match as successful evidence. Raw TV schema must preserve intentional open-trade exclusion while rejecting ambiguous duplicated entry/exit rows and malformed closed trades rather than choosing first/last silently. Document the supported raw long-entry/exit schema and unsupported partial/multiple-fill grouping.
5. TV mode runs explicitly with `tv_mode=True`; Chart mode with `tv_mode=False, chart_mode=True`; reject unknown modes. Remove the duplicate visualization call because `Backtest.run()` already renders. Do not change engine semantics or cached-data/network behavior (Issue #14 owns that).
6. Mode batch attempts both modes for every requested ETF, records per-symbol/per-mode success/error, and returns nonzero if any mode failed (including all failures). A successful mode requires nonempty price data and equity observations plus required metric attributes after `Backtest.run()`; silent early returns and zero-metric/empty-equity runs are failures. Zero trades/returns with actual equity observations remain valid. Reports and raw result CSVs retain partial diagnostics, label batch status/failed modes, create their output directory, and never turn missing results into a normal successful comparison. Do not claim numerical TV/Chart equivalence from successful execution alone.

## TDD, review and delivery

1. Write deterministic API-free fixtures first: exact/tolerated date matches; first mismatch 1 bar then 27; later reason mismatch even when >85% agree; count/insertion/removal; matching duplicates and unequal multiplicity; empty/invalid rows; bad calendars/missing date; raw TV closed/open/malformed groups; negative tolerance. Include real subprocess CLI checks for PASS=0 and FAIL/error!=0, output fields and verdict consistency.
2. Mode tests must instantiate the real Backtest constructor to catch incompatible mode combinations, inject synthetic data or fake the acquisition boundary, execute TV and Chart paths without real API requests, and verify each has a result. Separate batch tests inject one-mode/all-mode failures, silent returns and empty-equity results; check failure-labelled report/CSV output and returned/CLI nonzero status. Verify rendering happens once.
3. Implement the smallest validation/comparison changes satisfying the policy. Preserve current CLI arguments, normalization aliases and useful existing result keys where possible; append explicit diagnostics. No real workflows, broker orders or provider calls.
4. Run new targeted tests, full API-free suite and repository-compatible Ruff lint/format, mypy, codespell, Bandit, changed-file secrets hook and diff checks. Keep generated artifacts outside tracked paths. Record skips/limits separately.
5. Independent plan review maximum 3 rounds before implementation; revise actionable findings, finalize only when clear. Independent code review maximum 3 rounds after implementation; fix/recheck all findings. If blockers remain after round 3, stop without claiming completion or creating a ready PR.
6. Recheck Issue/PR overlap and exact main before submission. Verify Git author separately from active GitHub identity, commit only scoped files in this worktree, push via the active account's fork and open a PR with `Closes #12`, plan/review evidence and validation. Check CI; do not merge/deploy.

## Review log

- Plan round 1: independent reviewer identified over-restrictive duplicate rejection and insufficient mode-completion criteria. Revised to preserve/match duplicate occurrences and require observed equity/price results; added explicit corresponding fixtures.
- Plan round 2: independent reviewer confirmed both findings resolved and no remaining actionable findings. Finalized before implementation.
- Code round 1: reviewer reproduced raw exception credentials persisted in the newly added failure artifacts. Applied shared redaction once at each exception boundary and added query/header/registered-secret regressions over stdout and all three artifacts (3 tests failed before fix). No other actionable findings.
- Code round 2: independent reviewer confirmed the finding resolved, independently reran all 38 targeted tests successfully, and found no remaining actionable findings. Implementation review complete in two rounds.

## Validation evidence

- TDD: original regressions 24 failed / 9 passed before implementation. Additional numeric-date and erroneous-result-row tests failed before their fixes; the three redaction regressions failed before the review fix.
- Final focused suite: 38 passed, including real subprocess CLI and actual TV/Chart backtest execution with synthetic acquisition data. Network connections forbidden in the test process; no live APIs or workflow dispatches.
- Final full API-free suite (Python 3.11): **278 passed, 4 skipped**. Ruff 0.9.6 lint/format, codespell 2.3.0, mypy 1.14.1, Bandit 1.8.3, changed-file detect-secrets 1.5.0 hook and diff checks passed. The submitted revision is recorded in the implementation PR.
- Main/Issue/open PR overlap rechecked before submission: main remains the pinned base; no competing Issue #12 implementation appeared. Git author verified independently from active GitHub authentication; push uses the existing `takusaotome` fork, without alternate account credentials.
