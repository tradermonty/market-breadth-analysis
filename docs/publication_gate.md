# Validated daily publication (Issue #9)

`python scripts/publish_market_breadth.py --output-dir /absolute/new/run-directory`
creates a fresh artifact directory, runs the analysis in a captured subprocess, and exits
zero only after validating fresh chart HTML, both stable CSVs and market-date provenance.
PNG is optional. Existing output directories are refused without modification.
`validation.json` records PASS/FAIL, redacted diagnostics, generation/run timestamps,
Git source revision, market dates and SHA-256 hashes of required validated files.

The scheduled workflow uses a directory unique to the run ID and attempt under
`RUNNER_TEMP`. Only successful generation/validation reaches the Pages artifact upload.
An always-uploaded diagnostic contains only `validation.json`; a failed build cannot
start the dependent deploy job, so the previous public site remains in place.
The legacy research CLI is unchanged; use the publication runner for Pages.

## Prerequisites and activation

This change depends on Issue #7 / PR #26 (coverage columns) and Issue #8 / PR #32
(observed-date provenance and offline exchange calendar). On current main without
those producers, the runner fails closed with PREREQUISITE before provider acquisition.
Keep this PR **Draft** until both prerequisites have merged and their independent
review findings have been resolved. Rebase and repeat the combined integration tests
against the actual merged revisions before enabling publication. Unit tests passing
on current main do not demonstrate production readiness. No workflow dispatch or
live deployment is part of this implementation.

## Checks and settings

Both 200-day and 50-day coverage must meet `--min-coverage` (default 0.9).
`--short-ma` defaults to 8; `--data-arrival-grace-minutes` defaults to 30.
The producer's existing coverage floor also applies. Analysis always uses observed
provider dates and includes 50-day data. The offline XNYS resolver recomputes the
expected session at validation time, including runs crossing the delivery-grace boundary.
Explicit/research dates, calendar-only estimates, unavailable calendars, an older
intersection of price and breadth dates, malformed CSVs and inconsistent summary
counts prevent publication. Schema validation does not independently audit the
underlying constituent universe or prove live data quality beyond these contracts.

Only the checked-out index and two allowlisted historical forecast pages are copied
as static site assets. Their historical content is not treated as current analysis.
The chart and CSV files must be generated within the new run directory. Secret values
registered by the shared redactor and URL/header credentials are sanitized across
captured stdout/stderr (including swallowed errors), timeout output and diagnostics.
Raw provider logs are not stored in the report. Git hashes identify the checkout;
they are not external authenticity attestations.

## Local verification

Run `FMP_API_KEY='' DOTENV_DISABLED=1 SKIP_API_TESTS=true python -m pytest tests/ -q`.
`test_real_producer_contract_and_restored_paths` intentionally skips on main without
PR #32 and must run with both prerequisite producer changes present. It exercises
real calculation, CSV export and Plotly HTML generation with synthetic acquisition,
stubbing only optional PNG rendering. All new tests prohibit provider network calls.
