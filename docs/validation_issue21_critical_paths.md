# Issue #21 evidence

Base main: `6af9e4a0e7d3a1620b2952ff88f172154a59a287`. New branch/worktree
is independent of open PR #38. No assignee/comments/PR or unmerged remote
branch for #21 was found during selection on 2026-11-14.

Plan review took two rounds; two P2 findings were resolved: machine-enforced
fresh run/source identity and non-skipped critical tests, plus isolated nested
network-fault logs. The second plan review found no actionable issues.

Baseline: macOS Apple Silicon, Python 3.11.12, coverage 7.17.3, **734 passed,
4 skipped**. The baseline table and initial regression floors are in
`plan_issue21_critical_validation.md`. Raw measured gaps are generated in each
new run; they are not described as live/provider acceptance.

Implementation validation and independent code review results will be recorded
in the PR after the final run. Required separate evidence: full local suite,
new API-free/freshness/obligation gates, configured critical/FMP mypy checks,
existing lint/security gates, and hosted CI on the final commit.

## Final local verification

- Exact validation tools: pytest 9.0.3, coverage 7.17.3, mypy 1.14.1. macOS
  Apple Silicon, Python 3.11.12. Runtime versions are recorded in run.json;
  the local environment references existing packages rather than claiming a
  clean locked installation. The broker SDK is not installed locally and is
  explicitly recorded as such; broker tests use injected fakes. Hosted CI
  installs the complete runtime requirements separately.
- Full instrumented run: **768 passed, 7 skipped**, all critical required
  selectors passed and the verdict was PASS. Skips are optional saved-data/API/PNG
  cases and unavailable Ruff commands in the validation-tool profile; pinned
  Ruff gates were run separately. No required critical obligation was skipped.
- Fresh source/test/config/report identity checks passed for 145 input files
  in this local tree. Network attempt log: **zero bytes**. Coverage combined
  23 distinct process files (32 duplicate files skipped).
- Provider line/branch coverage: **45.43% / 32.26%**, up from 27.1% / 11.3%.
  Multi-ETF: **88.12% / 54.17%**, up from 33.7% / 25.0%. Corresponding floors
  tightened to 40/30 and 85/50. Child-only sentinel: **100% / 100%** against
  95/95 floors, proving child instrumentation was not silently lost.
- **17 probe diagnostics** retained in fault-probes, including rejected stale
  run/source/policy/config, omitted/skipped tests, missing/low coverage and
  swallowed/import/default-child/explicit-child network attempts; healthy
  controls succeeded. The outer network log stayed empty.
- Ruff/format, codespell, existing FMP and new critical mypy gates, Bandit,
  secret baseline gate and git diff check passed. Existing pip-audit gate
  passed with only the existing dated Alpaca msgpack/urllib3 exceptions; no
  exceptions were added.

## Code review and delivery boundary

Three independent code-review rounds were completed. Round 1's two P2 findings
(policy/config fingerprinting and default-env child bootstrap) were fixed with
regressions. Round 2 confirmed those fixes; an embedded-code string/import edit
error was corrected before successful validation. Round 3 found an unconditional
missing-distribution metadata lookup; optional absence now records 'not installed',
with a regression. Final review reported no remaining actionable findings.

The final local evidence directory is `/private/tmp/issue21-verified-run`. Earlier
collection/provenance-failed runs are not counted as PASS. Final hosted CI and
source commit are recorded in the PR. Source hash binding establishes fixture
evidence for this code; live provider data, broker behavior, GUI/browser outputs,
all vendor endpoint variants and complete static validation of dynamic SDK/JSON
records remain unverified. No workflow dispatch, live trade or deployment was
performed.

## Combined-base PR inventory review

After #38 merged as `eb5c661b29e20cebeba7f29039676dcc56b2922d`, the
critical workflow needed the same pinned uv for shared lock regression tests.
A restricted-PATH probe reproduced the missing-tool failure. The workflow now
installs uv 0.12.17 and syncs the locked trading/test profile before adding the
supplementary validation tools. The source fingerprint additionally includes
`uv.lock` and `.python-version`; two new fault probes failed before this fix
and passed afterward. Focused combined-base regressions: 24 passed.
Final combined validation and hosted CI are recorded on the PR review comment.

Combined-base local result: 777 passed, 9 skipped; all mandatory critical
selectors, input identity, branch floors and empty-network-log gates passed.
Existing and critical mypy, Ruff/format, codespell, Bandit and diff checks
passed. The initial default-cache permission failures were local sandbox
setup failures; the verified run used an explicit writable UV_CACHE_DIR.
