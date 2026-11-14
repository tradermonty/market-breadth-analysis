# Critical API-free validation (Issue #21)

This workflow measures fixture evidence for analysis, acquisition, backtests,
publication, comparisons and order recovery. It does not establish live broker,
provider availability or strategy equivalence. Project dependency locking and
installation changes belong to #17 / PR #38; this PR introduces a separate
workflow without editing that work.

## Reproduce

From a clean Python 3.11 environment with the repository's runtime dependencies:

```sh
python -m pip install -r requirements.txt
python -m pip install -r requirements-validation-tools.txt
python -m mypy --config-file mypy-critical.ini
python scripts/run_critical_validation.py --output-dir /tmp/breadth-validation-NEW
```

The output directory must be new; previous evidence is never reused or erased.
The separate `Critical API-free Validation` workflow runs these commands on
Ubuntu 24.04 and uploads all diagnostics even if tests or gates fail. Existing
CI still runs its original lint/type/security/test checks. Tool direct pins
for this supplementary workflow are explicit; #17 owns the broader lock.

Outputs include `tests.log`, JUnit, line/branch coverage JSON, HTML annotated
source, `critical-gaps.json` (actual missing lines/arcs), source/test/config file
hashes and interpreter/tool versions in `run.json`, a network-attempt log and
`verdict.json`. The checker binds report digests to a fresh random run ID and
compares pre-run input hashes against the current checkout. Editing source or
a report, reusing another run ID, omitting a module, dropping below a floor,
or omitting/skipping a required critical test fails validation. This is an
integrity/regression check for trusted test execution, not tamper-proof attestation.

## Network and integration behavior

`SKIP_API_TESTS=true python -m pytest tests/` blocks Python socket connection,
DNS and datagram events before collection and through normal Python children,
including calls that explicitly replace their environment/PYTHONPATH. The
real constituent API test is skipped even when a key is present. A CPython
audit hook records event names only, without hostnames, URLs or credentials;
the run fails even when application code catches the connection exception.

The validation launcher installs the same guard through `sitecustomize` and
sets `DOTENV_DISABLED=1`. It does not need credentials or cached real market
data. Negative guard tests use their own nested log and assert that the inner
launcher fails; there is no exemption for a real communication attempt in the
outer run. This guard protects Python fixture regressions, not arbitrary hostile
code/native network clients or a Python process deliberately started with `-S`.
For explicitly authorized real integration testing, use a new shell/process
without `BREADTH_OFFLINE_LOG`, unset `SKIP_API_TESTS`, and run the individual
integration test with appropriate credentials. Real integration is not part
of ordinary CI and must never be inferred from fixture PASS.

## Critical acceptance checklist

`validation/critical_policy.json` is the machine-enforced source of module
floors and mandatory pytest selectors. Every matching parametrized JUnit case
must pass; a missing selector or any failure/error/skip cannot be compensated
by higher aggregate coverage elsewhere.

| Critical path | Evidence |
|---|---|
| Provider transient retry and successful recovery | `test_acquisition_contract::test_01_transport_retry_and_recovery` with fake transport |
| Exhausted transport and redacted diagnostics | `test_02_exhausted_transport_is_redacted` |
| Non-data HTTP/empty/error payloads; historical shapes | Acquisition contract tests, fixed time/sleep |
| Real cached analysis including 50/200 coverage, HTML and CSV/summary | `test_critical_e2e::test_01_cached_analysis_emits_required_outputs` |
| Real single and multi-ETF cached backtests, finite summary for every symbol | `test_02_single_etf_and_multi_etf_actual_runs` |
| Trigger CLI failure propagates nonzero status | `test_03_trigger_cli_failed_download_is_nonzero` with fake HTTP |
| All-trade comparison mismatch versus matching/tolerated cases | Existing real comparison CLI PASS/FAIL tests |
| Missing/broken artifacts and latest-row coverage prevent publication | Existing publication artifact/coverage/CLI fault tests |
| Restart/lost response recovers without second submission | `TestReconciliation` exact-once/lost-response tests |
| Canceled partial fill credited once | Partial-fill reconciliation test |
| Corrupt persisted state blocks execution; duplicate ID adopts existing order | `TestBlockedStates` / `TestSubmission` |
| Invalid boundary arguments fail static gate | `test_critical_typing` runs the actual mypy checker |
| Stale/low/missing coverage, skipped obligation or caught network cannot pass | `test_critical_coverage_gate` fault injections |

The baseline plan records measured coverage and reviewed initial floors for
11 production modules. Floors are local module regression budgets, not safety
scores and not an arbitrary 100% target. FMP endpoint/fallback coverage and
multi-ETF orchestration were particularly low at baseline. New tests improve
those paths while preserving visible gaps. Inspect missing lines/arcs in each
run; provider earnings/screener endpoint variations, GUI/browser renderers,
live broker SDK responses and platform-specific lock races remain incomplete.
No critical production module is excluded to improve its reported percentage.
The raw report also measures repository Python files; tests, fixture harnesses
and independently managed subprojects are not given production coverage floors.

## Incremental static contracts

The existing FMP mypy check remains intact. `mypy-critical.ini` additionally
checks all definitions in `feed_contract.py` and `trade/order_state.py`, including
CSV input/result types, persisted-state loader shape, order ID parameters,
file paths, status-code helpers and execution-lock methods. Dynamic JSON record
values and unvalidated lock-file payloads remain explicitly `Any`, and calls
into the existing calendar helper are not made fully strict. Runtime validation
and broker simulations remain necessary; annotations do not validate external
payloads. The invalid-call probe proves a wrong CSV argument is rejected.

The security-gate tests now use temporary fixtures and a fixed injected date,
so test execution no longer rewrites source-controlled expiry fixtures. New
E2Es use fixed seeds, explicit dates and isolated cwd/cache/report directories.
See the plan and validation evidence in this PR for review findings and actual
local/hosted results. The use of subprocess coverage follows the [coverage.py
process documentation](https://coverage.readthedocs.io/en/latest/subprocess.html).
