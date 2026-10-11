# Work Plan — Issue #6: Prevent duplicate orders and reconcile broker state after restart or lost responses (rev 4, FINAL)

Ticket: https://github.com/tradermonty/market-breadth-analysis/issues/6 (P1, effort L)
Dependency: issue #5 (its open implementation PR is #25 — both referenced consistently).
Base: `main` at branch time. Scope: `trade/run_market_breadth_trade.py` + new `trade/order_state.py`.
Does NOT touch CI workflows, breadth fetching, or PR #25's fill-matching logic.
Plan review: 3 review rounds completed (rounds 1–3); round-3 findings 1–12 are all incorporated
below as fix tags [R3-n]. Max 3 review rounds reached per session policy → implementing rev 4.

## Problem Statement

- Crash after `submit_order` acceptance (or lost broker response) leaves no durable record of the
  accepted order; on restart, the same signal can trigger a second order.
- `acted_signals` persists only after a confirmed fill; submissions are otherwise unrecorded.
- No deterministic `client_order_id` is sent, so unknown-outcome retries create duplicate orders.
- No execution lock for the same symbol.

## Design (rev 4)

### 1. Deterministic client_order_id and duplicate-ID recovery
- `build_client_order_id(symbol, signal_type, signal_date, side, retry_count=0, stop_time=None)`
  in `trade/order_state.py`:
  - Non-stop: `mbt-{symbol}-{signal_type}-{signal_date}-{side}{retry_suffix}` (`R{n}` suffix,
    empty initially).
  - Stop: `mbt-{symbol}-stop-{signal_date}-{HHMM}-{side}` where HHMM is the order submission
    time (ET) — day-unique component so a same-day re-triggered stop after a resolved/partial
    record does not collide on COID ([R3-3] fix). A per-day monotonically increasing sequence
    (from existing same-day records) is appended if HHMM collides (`-1`, `-2`, ...).
  - Charset `[A-Za-z0-9-]`; **Alpaca cap is 48 chars** → overflow (`len > 48`) raises
    `ValueError` at build time → fail-closed BEFORE intent persist ([R3-10] fix).
- Alpaca v2 REJECTS resubmission of an existing active COID with 422
  "client_order_id must be unique" (does NOT return the existing order). Sequence:
  - a. Persist intent record (atomic `os.replace`) BEFORE submit. **Fail-closed: intent-persist
     failure aborts submission, logs CRITICAL, raises `ExecutionBlocked` (see §7).**
  - b. `submit_order(client_order_id=...)`.
  - c. On 422 "client_order_id must be unique": `get_order_by_client_order_id(id)` and ADOPT —
     never resubmit. Ambiguous outcomes (timeout/reset): mark `submitted-unknown`; resolution at
     reconcile via COID lookup.
  - d. Retry with `R{n}` is reconcile-side only ([R3-4] fix): eligibility = signal_type in
     {long_trough, short_trough} (plus `peak` only when broker position > 0); re-check signal
     still in lookback window AND not in acted_signals; re-derive qty from current cash via
     `_calculate_shares` (record `qty` is informational only); `stop` is excluded from retries
     (price logic naturally re-triggers next run, and the HHMM component makes it collision-free).
     Incremented retry_count is persisted BEFORE the retry submit.

### 2. Durability model: credit marker colocated with entry lots
- State file `trade/order_state_{SYMBOL}.json` (explicit enum + `schema_version`):
  ```json
  {"schema_version": 1, "records": [{
      "client_order_id": "mbt-SSO-long_trough-2026-03-10-B",
      "signal_type": "long_trough|short_trough|peak|stop",
      "signal_date": "2026-03-10", "side": "B|S", "qty": 100, "retry_count": 0,
      "status": "intent|submitted-unknown|accepted|resolved-no-order|resolved-filled|
                 resolved-canceled|resolved-unverified-testmode",
      "submitted_at": "...", "resolved_at": null, "fill_watermark": 0}]}
  ```
- **Idempotence authority lives with the LOTS, not in order_state.json** ([R3-1] fix):
  `entry_prices_{SYMBOL}.json` gains `applied_credit: {client_order_id: credited_qty}` written in
  the SAME atomic `os.replace` as the lots. Credit delta = `final_filled_qty − applied_credit[coid]`
  (0 if absent) → each fill coins exactly once regardless of where a crash lands. Crash windows
  tested at EVERY boundary (before/after acted write, before/after lots+credit write, before
  record resolve): re-run reconcile always converges to exactly-once crediting
  (`watermark` in the record remains for observability only and is not trusted for accounting).
- Fill-path repair order (identical for the LIVE fill path and reconcile — the live path writes
  at lines 777–780/817–820 are reordered to match):
  1. acted_signals upsert (atomic; skipped for `signal_type == 'stop'` — no acted key by design;
     position reaching 0 makes re-trigger impossible),
  2. entry-lots + `applied_credit` in one atomic `os.replace`,
  3. record -> `resolved-*` (last).
- `_save_acted_signals` / `_save_entry_prices` become atomic writes (tmp + `os.replace`).
- Corrupt acted_signals JSON: instead of silently returning an empty set, raise to
  `exec_blocked`.
- COID uniqueness: state file must not contain duplicate non-terminated COIDs; assert on write
  ([R3-3] companion).

### 3. Startup reconciliation (once, before the trading loop; pure functions in order_state.py over an injected API)
- Per record in `intent|submitted-unknown|accepted|resolved-unverified-testmode`
  ([R3-6] fix: testmode records are swept too):
  - `get_order_by_client_order_id(coid)`:
    - not found / final zero-fill canceled|expired|rejected: `resolved-no-order|canceled`.
    - `filled` or canceled/expired with `filled_qty > 0`: idempotent §2 credit repair, then
      `resolved-filled`. A canceled order with partial fill is `resolved-filled` and NEVER takes
      the `R{n}` retry path.
    - open: adopt via `_wait_for_fill`, then repair.
  - Lookup raises (network/auth/5xx): record stays; trader `exec_blocked`; exit nonzero (§7).
- `exec_blocked` triggers: corrupt order_state JSON, corrupt acted_signals JSON, intent-persist
  failure, COID overflow, reconcile lookup failure.
- **`--reconcile-only` performs NO new submissions and NO retry submissions** ([R3-5] fix); it
  only adopts/in-progress resolves existing orders and reports. Retry submission is a full
  trading run's job (§1d). Documented in the runbook.
- Structured error classification: by exception TYPE and response-status attributes
  (`getattr(exc, 'response', None)` → `.status_code`), never by matching logged strings
  (PR #10 redaction-safe). `execute_buy`/`execute_sell` catch-all returning `None` is removed so
  submissions raise through to the classifier (no silent swallow).
- Testmode: shares the state file; DOES write intents (persist moved to the shared pre-submit
  point inside `execute_buy`/`execute_sell`, before the testmode early-return). In-session
  testmode "fill" resolves the record to `resolved-unverified-testmode`. Live reconcile sweeps
  those records via COID lookup (§3 first bullet).

### 4. Per-symbol execution lock
- Module-level `logging.basicConfig` is MOVED into `main()`.
- Lock (`trade/.lock_{SYMBOL}`, `os.open(O_CREAT|O_EXCL)`, content `{pid, acquired_at}`) is the
  **first state-changing action: after argparse (COID/lock derive from `--symbol`), before
  logging setup and before `MarketBreadthTrader.__init__`** ([R3-9] fix).
- Stale detection: `os.kill(pid, 0)` → ProcessLookupError: stale, remove, proceed;
  PermissionError: treat ALIVE (foreign-user), block; alive: block CRITICAL, exit nonzero.
- Released in shared `finally`, including `ExecutionBlocked` paths.

### 5. CLI additions
- `--reconcile-only`: lock → reconciliation (no new orders) → machine-readable JSON to stdout;
  all logging to stderr; test asserts `json.loads(stdout)` via `run_reconcile_only(symbol, args,
  api=None)` seam (facts injected, no Alpaca import).
- Exit codes: 0 = clean, 1 = blocked ([R3-2] fix asserted in tests).
- No other CLI changes.

### 7. Fail-impaired execution: `ExecutionBlocked`
- New exception `ExecutionBlocked(reason)` (order_state.py). Raised by: intent-persist failure,
  COID overflow at build time,   reconcile lookup failure, corrupt state/acted files, live-lock contention.
- `run()` does NOT swallow it in its broad `except Exception` handler — re-raise ([R3-2] fix);
  `main()` catches → log CRITICAL + `sys.exit(1)`; lock released in `finally`. Reconcile-only
  mode uses the same handling. Tests assert nonzero exit codes (tests 6, 8, 9).

## Fault-injection test matrix (`tests/test_order_reconciliation.py`, API-free fakes)

1. Crash before submission (intent persisted, broker has no order) → `resolved-no-order`.
2. Crash after acceptance (intent, broker filled) → exactly-once lot+credited; same signal cannot
   re-order (acted guard + resolved record).
3. **Response-loss restart**: record pre-seeded `submitted-unknown`, broker order found via COID
   → repair + `resolved-filled`, no second submit ([R3-11] fix).
4. Duplicate-ID 422 on resubmit → adopt via lookup; `submit_count == 1`.
5. Zero-fill cancel then reconcile-side retry: `R1` ID; `retry_count`/intent persisted before
   retry submit (call-order spy). Entry signal re-derives qty from cash.
6. Partial fill then cancel: `canceled, filled_qty=5` of 10 → lots credited with exactly 5 once
   (credit-marker authority), acted_signals upsert (non-stop), resolved-filled — NO retry.
7. **Crash-window exhaustive repair (rev4)**: inject crash at every window boundary of the
   3-step repair (acted only / acted+lots+credit / none) for both live path and reconcile →
   re-run converges to exactly-once crediting ([R3-1] fix — applied_credit authority).
8. Temp-file atomicity: no `.tmp` residue; state parseable after every interleaving.
9. Corrupt order_state JSON → blocked nonzero; corrupt acted_signals JSON → blocked nonzero.
10. Reconcile lookup failure → blocked nonzero.
11. Lock tests: dead-pid stale removed; alive blocked nonzero; foreign (EPERM) blocked; a second
    process exiting on a live lock writes NOTHING to `trade/market_breadth_trade.log`;
    argparse-before-lock ordering asserted ([R3-9]).
12. COID: charset, 48-char cap (overflow raises, no intent persisted, no submit), retry-suffix
    uniqueness (`R1`, `R10`, `R100`), stop-HHMM same-day uniqueness.
13. Duplicate-signal guard: acted_signals entry prevents resubmission (existing behavior kept).
14. Testmode: intent on shared pre-submit path; testmode reconcile → `resolved-unverified-testmode`
    with zero broker calls; live reconcile SWEETS those records via COID lookup.
15. `--reconcile-only`: never calls `submit_order` even with retry-eligible zero-fill canceled
    record; stdout parses as JSON while logs go to stderr ([R3-5]).

## Acceptance criteria mapping (issue #6)

- No second order for same signal after acceptance/response-loss restart → tests 2, 3, 4.
- Fault injection before submission / after acceptance / after partial fill / during
  persistence → tests 1, 2, 6, 7, 8.
- Corrupt state or failed reconciliation blocks new submissions + documented recovery →
  tests 9, 10 + `trade/recovery_runbook.md` appended in this PR. Runbook enumerates per-case
  recovery ([R3-12] fix): order_state corrupt → quarantine file, resume from COID lookups;
  acted_signals corrupt → restore/delete + note the weakened duplicate guard; intent persist
  failure → check disk/permissions, then reconcile-only.
- CI regression + lint + tests pass; revision hash, evidence, residual limitations in PR body.

## Execution steps

1. Worktree `feat/issue-6-order-reconciliation` from `main`.
2. TDD: failing tests → `trade/order_state.py` → wire trader (reordered fill path, intent
   persist + `ExecutionBlocked`, reconcile call, lock, logging relocation into main()).
3. Full suite `python -m pytest tests/ -q`; `ruff check --fix`; `ruff format`; `codespell`.
4. Code-review subagent loop (max 3 rounds) → fix until clean.
5. Commit and open PR referencing issue #6.

## Risks / limitations (PR body)

- COID dedupe verified against the paper endpoint only.
- PID-liveness lock cannot detect a resurrected PID; manual recovery via runbook.
- Rebase interplay with issue #5 / PR #25 fill semantics (follow-up adapt inside this PR if it
  merges first).
- Real orders and live data feeds out of scope (per issue text).
- Testmode may mark real pending records `resolved-unverified-testmode`, bounded by COID
  422-adopt on resubmit (runbook note).
