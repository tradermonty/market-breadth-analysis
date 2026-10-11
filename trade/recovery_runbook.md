# Recovery Runbook — Order-State Reconciliation (Issue #6)

This runbook documents recovery procedures for the duplicate-order protection
introduced by issue #6. The trader persists every order intent to
`trade/order_state_{SYMBOL}.json` BEFORE submitting it to the broker, sends a
deterministic `client_order_id`, and reconciles pending records against the
broker at every startup.

## State files

| File | Purpose |
|---|---|
| `trade/order_state_{SYMBOL}.json` | Order records (intent → accepted → resolved), single source of truth for reconciliation |
| `trade/entry_prices_{SYMBOL}.json` | Entry lots + `applied_credit` (fill idempotence marker keyed by client_order_id) |
| `trade/acted_signals_{SYMBOL}.json` | Signals already acted on (duplicate-signal guard) |
| `trade/.lock_{SYMBOL}` | Execution lock (pid + acquired_at) |

## When trading is BLOCKED (exit code 1, no submissions)

The trader blocks (raises `ExecutionBlocked` → exits nonzero) on any of:

- Corrupt `order_state_{SYMBOL}.json` (invalid JSON/schema) — `StateCorruptionError`.
- Corrupt `acted_signals_{SYMBOL}.json` — duplicate-signal guard cannot be trusted.
- Failed broker lookup during reconciliation (network/auth/5xx).
- Order-intent persistence failure before submission (disk full/permissions).
- `client_order_id` overflow (symbol/id too long).

## Recovery procedures

### 1. Reconciliation-only check (always try first)

```bash
python trade/run_market_breadth_trade.py --symbol SSO --reconcile_only
```

-Checks pending order records against the broker; **never submits new orders**.
- Prints a machine-readable JSON summary to stdout (`blocked` flag + per-record outcomes).
- Exit 0 = clean; exit 1 = blocked (inspect `reason`).

### 2. Corrupt `order_state_{SYMBOL}.json`

1. Quarantine (rename, do not delete for evidence):
   `mv trade/order_state_SSO.json trade/order_state_SSO.json.corrupt.$(date +%s)`
2. Run `--reconcile_only`: with no state file, reconciliation cannot act further;
   the broker remains the source of truth — positions/orders must be verified
   manually in the Alpaca paper dashboard.
3. A fresh full run rebuilds state from scratch. Because resubmission replays new
   client_order_ids and signals previously acted on are already in
   `acted_signals_*.json` (if intact), duplicates remain prevented.

### 3. Corrupt `acted_signals_{SYMBOL}.json`

1. Quarantine it (`mv ... .corrupt.$(date +%s)`).
2. Run `--reconcile_only`: fills recorded in order_state repair
   acted-signals keys up via `resolved-filled` records (for signal types other
   than `stop`).
3. NOTE: while the acted-signals file is missing, the duplicate-SIGNAL guard is
   weakened (order-level duplicate protection via `client_order_id` still works).
   Retrospectively verify no double position exists before the next full run.

### 4. Order-intent persistence failure before submission (disk full/permissions)

- No broker order was sent (submission ABORTED by design).
- Check disk space/permissions for `trade/`.
- Then `--reconcile_only` to confirm clean, followed by a normal run.

### 5. Stale execution lock

- Locks contain `{pid, acquired_at}`; a dead pid is removed automatically.
- A corrupt (pid-less) lock file means the holder died between creation and
  payload write — it is safe to remove manually (`rm trade/.lock_{SYMBOL}`).
- If a live pid still holds the lock when it should not: verify the process
  (`ps -p <pid>`), kill it if it is genuinely dead/hung, then rerun.
- If the pid is owned by another user (`PermissionError`), verify ownership
  manually — the trader will not remove a foreign live lock.

## CLI reference

| Command | Behavior |
|---|---|
| normal run | reconcile pending records → reconcile-side retries only → trading loop |
| `--reconcile_only` | lock + reconciliation + JSON summary to stdout; NO new orders |
| exit 0 | clean |
| exit 1 | blocked (see `reason` in stderr/JSON) |

## Known limitations

- Broker-side dedupe (422 on an already-active client_order_id → adopt) was
  verified against the paper endpoint only.
- Testmode shares the state file with live mode; a testmode run may mark real
  pending records `resolved-unverified-testmode` — residual risk is bounded by
  the deterministic client_order_id (a real resubmit then adopts via the 422
  lookup path). Prefer reconciling real pending orders BEFORE running testmode.
- PID-liveness detection cannot detect a resurrected (recycled) PID.
