# Issue #17: Reproducible dependency environments

## Selection and baseline

- Base: `6af9e4a0e7d3a1620b2952ff88f172154a59a287` (main after PR #37).
- No open PRs, no Issue #17 assignee/comments, and no unmerged remote branches were found on 2026-11-14. Existing worktrees cover merged #13/#22, with no changes in the #22 worktree. Preserve the original checkout's unrelated order-state file.
- #17 is a highest-priority remaining implementation task (P2); #18 depends on it. Prerequisites #15/#16 are merged. #4 is a tracker and #36 requires owner/provider policy answers.
- New worktree: `/private/tmp/market-breadth-issue17-20261114`, branch `feat/issue-17-locked-environments`.

## Intended contract

1. Define Python 3.11 as the supported interpreter (`>=3.11,<3.12`). Validate Linux x86_64 in hosted CI and macOS Apple Silicon locally/hosted CI. Other Python/OS combinations are not claimed tested. Separate core headless analysis/backtests/HTML/CSV from optional PNG (`kaleido`), broker client (`alpaca-trade-api`), and Windows Qt. Retain the Alpaca version constraints and dated vulnerability exceptions; do not change trading behavior.
2. Introduce uv project metadata and a committed universal `uv.lock` resolving all runtime extras and test/lint/security dependency groups. Keep the existing tool-version pins and pre-commit parity contract; validate that direct tool group pins match their requirements files. Pin the uv version used by commands and CI.
3. Generate hash-bearing `requirements.txt` from the lock for core + all extras + test dependencies as a pip-compatible compatibility/audit surface. Source dependency ranges live in pyproject; exported pins are never edited by hand. CI verifies `uv lock --check` and an exact regeneration comparison (stable header without command-dependent content). Security scans this resolved export through the existing fail-closed gate, preserving the dated allowlist. Lint/security direct pins retain their parity checks; their transitives are locked but remain outside Gate 1, matching the existing audit boundary; no security bypasses or new exemptions.
4. Use `uv sync --locked` in CI test matrix and producer/monitor workflows, and `uv sync --locked --only-group lint` / `--only-group security` in tool jobs; activate the selected environment for every subsequent Python/tool command; producer/monitor install core only, full regression CI adds trading + test. Use the same Python 3.11 and uv version everywhere. Guard lock/input/export changes with CI paths. Retain publication validation and permissions.
5. Document install/activate/run/update/export/audit commands in English/Japanese installation sections and a dependency runbook. Document that saved-data examples need actual cache fixtures, Chrome is needed for optional modern Kaleido, Qt is Windows-only/best-effort, and runtime credentials and live trading are outside this validation.

## Tests and evidence

- Fix the live S&P500 test guard to honor `SKIP_API_TESTS=true` even when a credential is present, with a regression using a dummy key.
- First add failing contract tests for profiles, locked export freshness, workflow installation and source pin parity; update the existing monitor dependency test to validate shared lock installation instead of old inline ranges.
- Demonstrate a stale pyproject/lock or changed requirements export fails validation; use uv's native lock check rather than a custom resolver.
- Create two clean Linux CI installs of the same regression profile and compare installed distribution/version manifests. Locally create a clean macOS environment, run the full API-free test suite and a core-only synthetic HTML/CSV smoke path. Ensure no Qt/Kaleido/broker package is installed in core. PNG failure must remain nonfatal with usable HTML/CSV.
- Run relevant/all regressions, Ruff/format, configured mypy, codespell, Bandit, tool parity, lock/export checks, and the existing dependency/security gates. Disable real API tests explicitly for clean local runs; do not use credentials, submit trades, dispatch producer workflows, or publish Pages. Record local versus hosted evidence separately.

## Review and delivery

- Obtain independent subagent plan review (maximum three rounds), address all findings, then mark this plan final.
- Implement the final plan in this worktree. Obtain independent code review (maximum three rounds), address findings and rerun affected checks. If findings remain after the cap, do not claim completion.
- Commit and create an English PR with exact validation/limitations and a reviewed lock diff, linked to #17. Do not merge or deploy in this task.

## Review log

Round 1: reviewer found two P2 gaps: tool jobs could bypass the lock/environment, and SKIP_API_TESTS was ignored by one credential-dependent test. Both are explicitly addressed above; audit-boundary wording corrected.

Round 2: no remaining actionable findings. Plan revision 2 is FINAL.

## Implementation decisions from validation

- Separate lint/trading resolutions explicitly because their urllib3 constraints conflict.
- Upgrade pytest 9.0.2 to 9.0.3 consistently in the existing pin files and groups: the security scan found PYSEC-2026-1845. No new vulnerability exception.
- Noninteractive backtest rendering must select Agg; a clean macOS environment exposed a Tk initialization failure. Added a failing regression before the fix.

## Implementation review log

- Round 1: P1 workflow-level runner context was unavailable; moved environment setup into a runner step and added regression. P2 production core urllib3 fork was outside the all-extras audit export; added a checked core-only export and an independent fail-closed audit with no accepted exceptions.
- Round 2: independent reviewer confirmed no remaining actionable findings. Native lock/export checks and focused tests passed.
- Final local full suite: 738 passed, 9 skipped. See `validation_issue17_locked_environments.md` for evidence boundaries.
