# Trade and mode comparison contracts

## Closed-trade acceptance

```sh
python scripts/compare_tv_python_trades.py --python_csv python.csv --tv_csv tv.csv --price_csv prices.csv --tolerance_bars 1
```

A PASS requires nonempty, equal closed-trade counts, every paired entry and exit date within the bar tolerance, and **100% normalized exit-reason agreement**. The maximum bar difference across all trades determines acceptance; the first mismatch is only diagnostic. A later excessive mismatch always fails. Exit status is 0 for PASS, 1 for a valid comparison that fails, and 2 for invalid input.

Pairing uses stable chronological order by entry date, exit date, then normalized reason. It does not infer identities, align by monetary P&L or search for a more favorable fuzzy match. Matching duplicate occurrences are retained (for example, FIFO split fills); exact-match and unmatched-tuple diagnostics preserve their counts. Exact tuple differences can be nonempty on a tolerated PASS, since those diagnostics require identical dates while acceptance allows the configured bar tolerance. Changed duplicate multiplicities are never silently deduplicated.

Python CSV requires `entry_date`, `exit_date`, `exit_reason`. Normalized TV CSV requires `entry_date`, `exit_date` and `reason` (or `exit_reason`). Dates must be parseable, present and timezone-naive; trade timestamps are reduced to daily dates, with entry not after exit. Blank reasons and invalid rows cause failure, rather than being discarded. Existing reason aliases and Unicode dash normalization are retained.

Raw TV CSV supports `Trade #`, `Type`, `Date and time`, `Signal`: one `Entry long` and one `Exit long` row per closed trade number. A valid entry without an exit, or a valid entry/exit pair whose exit signal is `Open`, is excluded as open. Missing trade IDs, unsupported short/unknown row types, multiple entries/exits in one trade number, invalid timestamps or reversed chronology fail validation. Normalize multi-fill raw groups into one row per closed fill before comparison; this script does not infer their allocation.

The price CSV's first column supplies the trading calendar: nonempty, unique, timezone-naive daily dates, sorted on load. Every compared trade date must be present. Missing dates never fall back to calendar-day arithmetic. Use the actual market-session index for the tested instrument; no exchange calendar or holiday dates are fetched by this tool. Tolerance must be a nonnegative integer.

## TV versus Chart mode batch

```sh
python scripts/compare_tv_chart_modes.py
```

TV uses `tv_mode=True`; Chart uses `tv_mode=False, chart_mode=True`. Each backtest runs and renders once. A successful result requires price data, actual equity observations and the metrics used by the report. A valid zero-trade/zero-return run with observations is allowed; silent early returns and empty-equity runs fail.

The batch attempts both modes for every configured ETF. The report at `reports/tv_vs_chart_mode_comparison.md` has an explicit execution `Status`; `tv_mode_results.csv` and `chart_mode_results.csv` include per-symbol `Status` and failure `Error` fields. Exception messages pass through the shared credential-redaction helper before console/report/CSV output. Any failed mode makes the batch return 1; complete execution returns 0. Partial results remain available for diagnosis, and the output directory is created when absent. An empty ETF configuration also fails.

These statuses indicate execution completeness, not numerical equivalence, investment performance validation, or proof that the inputs are current. The batch retains its existing symbol/period configuration and saved-data behavior; strict offline acquisition is tracked by Issue #14. API-free regression tests replace acquisition/rendering boundaries and forbid network connections.
