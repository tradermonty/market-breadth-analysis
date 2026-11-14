# Issue #17 validation evidence

Base: `6af9e4a0e7d3a1620b2952ff88f172154a59a287`. Tested locally on
macOS Apple Silicon, CPython 3.11.15, uv 0.12.17, on 2026-11-14.
The implementation PR records the final source commit and hosted CI state.

`uv.lock` SHA-256:
`530afb5629dd4d1463fe060a3b9fe3f06ad1dc3393790dbdac8a578b697d0672`.

## Local results

- Full regression profile (`trading` + `test`): **738 passed, 9 skipped**.
  `SKIP_API_TESTS=true`, empty FMP key; no live broker/provider validation.
  Skips include optional PNG, unavailable lint/security tools in the regression
  profile, rate-limiter API tests and the real constituent API check.
- Native lock check and both generated hash-bearing exports match. Tests reject
  a modified export in either profile and a new unresolved dependency. A
  subprocess regression confirms the real API test is skipped with a dummy key.
- Two fresh independent regression environments each installed the same **47**
  distributions; `uv pip freeze` manifests compared exactly equal (including
  pytest 9.0.3). Exact commands are implemented in the CI test job.
- Clean core-only environment: **28** distributions, with Kaleido, Qt and the
  broker SDK absent. `scripts/smoke_headless.py` passed with sockets blocked,
  producing real Plotly HTML, nonempty data CSV and summary CSV. Missing PNG
  support was logged and did not prevent required outputs.
- Ruff lint and format, configured mypy, codespell, Bandit and `git diff --check`
  passed. Tool/pre-commit parity checks also ran in their separate locked
  lint/security environments.
- Detect-secrets output passed the existing fail-closed baseline gate.
- pip-audit core-only profile: **no known vulnerabilities**, empty allowlist,
  `CI-GATE-PASS`. Trading/test profile: only the existing dated Alpaca
  msgpack/urllib3 findings, `CI-GATE-PASS`. No new exemptions were introduced.
  Initial audit rejected pytest 9.0.2 (PYSEC-2026-1845); upgrading to 9.0.3
  resolved that finding across every group/direct-pin file.
- A clean macOS run exposed Tk initialization during noninteractive backtests.
  The failing regression was reproduced before selecting Agg for `show_plot=False`;
  the three affected offline backtest E2E cases pass in the final full suite.

## Independent reviews and remaining evidence

Plan review: two rounds, two P2 findings resolved, final round no actionable
findings. Code review: two rounds, P1/P2 findings resolved, final round no
remaining actionable findings. Details are retained in the plan's review log.

Ubuntu and macOS hosted matrix results must be verified on the PR. Local macOS
results do not prove Linux/hosted acceptance. The workflow repeats fresh
installs, compares manifests, uploads evidence, runs core smoke and full tests.
Windows Qt, Intel macOS, actual PNG/Chrome rendering, OS/build-tool binary
reproducibility, live broker behavior and current provider data remain
unvalidated. No producer workflow was dispatched and no Pages deployment or
trade was performed as part of this work. Lint/security tool transitives are
locked but remain outside Gate 1, as documented in the dependency runbook.
