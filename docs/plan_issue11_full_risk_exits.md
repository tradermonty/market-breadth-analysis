# Issue #11: separate partial profit exits from full liquidation

## Selection and scope

- Issue: https://github.com/tradermonty/market-breadth-analysis/issues/11 (P2, independently startable).
- Base: `9eebcfe00aee81df5d8e63157e071c206cd0d651`.
- Worktree: `/private/tmp/market-breadth-issue11-20261016`; branch: `fix/issue-11-full-risk-exits`.
- On selection, open PRs #25/#29 own live trading; #26 owns breadth/coverage; #27/#28 own CI/security. Existing local worktrees/remote branches have the same scopes. Issue #11 has no assignee, implementation PR, or claim comment. Other agents outside visible GitHub/local state cannot be ruled out.
- P1 #9 depends on unfinished #7 and #8; other outstanding P1 issues have existing implementations. #11 is a high-priority independent risk-accounting correction without changes to those owned files.
- Change only backtest exit behavior, its CLI help/design documentation, dedicated regressions, and this plan. Do not alter live trading, market date/coverage logic, workflow/dependency setup, or execution timing/price models.

## Behavior contract

1. `partial_exit=True` continues to halve discretionary profit/signal exits (integer floor for quantities >1; one share closes fully). Peak/background signal behavior stays compatible.
2. Fixed, trailing and volatility stops always liquidate all remaining shares in every supported runner (legacy, TV same-bar, Pine-compatible). Preserve current stop thresholds/fill timing, gap fills and costs. No partial-stop option is introduced.
3. Backtest-end liquidation always closes all remaining shares, regardless of `partial_exit` or two-stage state. Pending weekly end exits keep the `weekly trailing` reason; existing next-bar weekly exits remain full exits at their current fill price.
4. Two-stage peak exit keeps its explicit half-sale; stage-two trend-break liquidates the entire remainder, including with `partial_exit=True`.
5. Use the existing `_execute_exit(..., force_full_exit=True)` mechanism at risk/final/stage-two call sites. Retain the generic helper default and discretionary callers for compatibility. Do not infer exit policy by matching human-readable reason strings.
6. FIFO logs, proportional lot costs, cash, commissions, equity and exit-state reset must reconcile. After final/risk exit: `current_position==0`, `open_positions==[]`, cleared entry prices/high-water/two-stage state. Final equity is cash and equals initial capital plus realized trade-log P&L.

## Implementation and verification

1. Inspect every `_execute_exit` call and current partial/stage-one/weekly accounting. Write API-free synthetic regressions before production changes; capture the expected failures (RED).
2. Tests drive actual runner stop paths for legacy fixed/trailing, TV fixed/trailing/volatility, Pine-compatible fixed; supported combinations only. Use known prices rather than mocking the exit helper. Check full liquidation and no same-bar reentry.
3. Test 1/odd/even shares, multiple FIFO entry lots at distinct prices, nonzero commission/slippage, normal partial profit exits, stage-one then stop/stage-two/final liquidation, weekly next-open exits and last-bar pending weekly exits. For final liquidation run real `execute_trades()` on an injected frame with signals disabled; verify last equity and logs. Use fixed dates and forbidden network access.
4. Apply minimal explicit full-exit flags. If tests expose an accounting problem directly caused by these exits, fix within this scope; do not change metrics or unrelated strategy behavior.
5. Update `--partial_exit` help and the backtest design's partial-exit contract to distinguish profit exits from stop/final liquidation. Preserve CSV fields and existing reason strings (full exits do not gain `(partial)`).
6. Run targeted exit/FIFO/Pine/weekly regressions and the full API-free suite; run Ruff lint/format, mypy, codespell, Bandit, secret scan and diff checks with repository-compatible tools. Live broker/data tests remain out of scope and are explicitly skipped.
7. Independent subagent plan review: at most three review rounds, revise all actionable findings and record outcomes below. Finalize before implementation. Independent implementation review: at most three rounds, fix and recheck findings. Only commit/create PR when no actionable findings remain. If unresolved at round three, report remaining blockers and do not claim completion.
8. Recheck open PRs/Issue ownership before pushing; inspect exact staged diff and Git author separately from authenticated GitHub identity. Commit only scoped files, push new branch and create PR with `Closes #11`, validation, review evidence and model limitations. Do not merge or deploy.

## Review log

- Plan round 1: independent subagent reviewed Issue #11, every exit caller, FIFO/costs, supported modes, weekly scheduling and scope. No actionable findings. Preserve Pine-compatible defaults (`partial_exit=False`, fixed stop, Pine costs); no artificial flag overrides in tests. Plan finalized after round 1, before implementation.
- Code round 1: independent reviewer found a failing new background-exit fixture: both bars were bearish, so the transition-based signal did not fire. Corrected the preceding bar to bullish. No production-code defects identified.
- Code round 2: independent reviewer rechecked the final diff and reran all 51 exit-policy cases successfully. Prior finding resolved; no remaining actionable findings. Implementation review complete in two rounds.

## Execution evidence

- Before implementation: existing FIFO/Pine/weekly baseline 55 passed; added regressions initially 26 failed / 24 passed for the expected residual-position defects.
- Final exit-policy suite: 51 passed (Pine defaults preserved); mock weekly signal only, real scheduler/fills/FIFO; network connections forbidden in this suite.
- Ruff lint/format, codespell, mypy, Bandit and changed-file detect-secrets hook checks passed. An accidental scanner baseline rewrite was restored; `.secrets.baseline` is not part of the change.
- Final full API-free suite (Python 3.11): **291 passed, 4 skipped**. The exact submitted revision will be recorded in the implementation PR. Live orders/data acquisition and historical strategy equivalence were not tested or claimed.
- Rechecked Issue #11 and all open PR file lists before submission: no competing implementation or overlapping changed file found. The active `takusaotome` GitHub account has read-only access upstream; PR submission uses its fork rather than another saved account's credentials. Git author/email checked independently: `takusaotome <takusaotome@fujisoft.com>`.
