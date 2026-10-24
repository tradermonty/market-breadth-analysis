"""Order-state persistence and reconciliation primitives (Issue #6).

Prevents duplicate orders after a process crash or a lost broker response by
persisting order intents atomically, deriving deterministic client_order_ids,
and reconciling pending broker orders at startup.
"""

import json
import os
import re
import tempfile
from contextlib import suppress


class ExecutionBlocked(Exception):
    """Trading cannot proceed safely; must never be swallowed by run()."""


class StateCorruptionError(ExecutionBlocked):
    """Local state files are unreadable/corrupt; submissions must be blocked."""


# Status enum (schema_version 1)
STATUS_INTENT = 'intent'
STATUS_SUBMITTED_UNKNOWN = 'submitted-unknown'
STATUS_ACCEPTED = 'accepted'
STATUS_RESOLVED_NO_ORDER = 'resolved-no-order'
STATUS_RESOLVED_FILLED = 'resolved-filled'
STATUS_RESOLVED_CANCELED = 'resolved-canceled'
STATUS_RESOLVED_TESTMODE = 'resolved-unverified-testmode'

ACTIVE_STATUSES = {STATUS_INTENT, STATUS_SUBMITTED_UNKNOWN, STATUS_ACCEPTED}
SWEEP_STATUSES = set(ACTIVE_STATUSES)
TERMINAL_STATUSES = {STATUS_RESOLVED_NO_ORDER, STATUS_RESOLVED_FILLED, STATUS_RESOLVED_CANCELED}
FINAL_STATUSES = TERMINAL_STATUSES | {STATUS_RESOLVED_TESTMODE}

REQUIRED_RECORD_KEYS = {
    'client_order_id',
    'signal_type',
    'signal_date',
    'side',
    'qty',
    'retry_count',
    'status',
    'submitted_at',
    'resolved_at',
    'fill_watermark',
}

SCHEMA_VERSION = 1
COID_MAX_LEN = 48
_COID_RE = re.compile(r'^[A-Za-z0-9_-]+$')

# Alpaca order lifecycle: statuses that can NEVER fill again (review R2).
# Anything unknown/nonfinal (pending_cancel, pending_replace, held, suspended,
# done_for_day, ...) must stay active — fills can still occur before
# cancellation completes.
FINAL_ORDER_STATUSES = {'filled', 'canceled', 'expired', 'rejected'}


def build_client_order_id(symbol, signal_type, signal_date, side, retry_count=0, stop_time=None, stop_seq=None):
    """Deterministic client_order_id. Raises ValueError on overflow (fail-closed)."""
    side_str = str(side).lower()
    if side_str not in ('buy', 'sell'):
        raise ValueError(f'invalid side: {side!r}')
    side_char = 'B' if side_str == 'buy' else 'S'
    retry_suffix = f'R{retry_count}' if retry_count else ''
    stop_suffix = ''
    if signal_type == 'stop':
        stop_suffix = f'-{stop_time or "0000"}'
        if stop_seq:
            stop_suffix += f'-{stop_seq}'
    coid = f'mbt-{symbol}-{signal_type}-{signal_date}-{side_char}{retry_suffix}{stop_suffix}'
    if not _COID_RE.match(coid):
        raise ValueError(f'client_order_id contains invalid characters: {coid}')
    if len(coid) > COID_MAX_LEN:
        raise ValueError(f'client_order_id exceeds Alpaca limit of {COID_MAX_LEN} chars: {len(coid)}')
    return coid


def atomic_write_json(path, data):
    """Write JSON atomically via tempfile + os.replace in the same directory."""
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp-state-', dir=directory)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            with suppress(OSError):
                os.remove(tmp_path)


def validate_order_state(data):
    """Validate parsed order-state JSON. Raises StateCorruptionError on bad shape."""
    if not isinstance(data, dict):
        raise StateCorruptionError('order state must be a dict')
    if data.get('schema_version') != SCHEMA_VERSION:
        raise StateCorruptionError(f'unsupported order-state schema_version: {data.get("schema_version")}')
    records = data.get('records')
    if not isinstance(records, list):
        raise StateCorruptionError('order state missing records list')
    for rec in records:
        if not isinstance(rec, dict) or any(k not in rec for k in REQUIRED_RECORD_KEYS):
            raise StateCorruptionError('order-state record missing required keys')
        if rec['status'] not in FINAL_STATUSES and rec['status'] not in ACTIVE_STATUSES:
            raise StateCorruptionError(f'unknown record status: {rec["status"]}')
    active_coids = [rec['client_order_id'] for rec in records if rec['status'] in ACTIVE_STATUSES]
    if len(active_coids) != len(set(active_coids)):
        raise StateCorruptionError('duplicate active client_order_id')
    return data


def load_order_state(path):
    """Load order state; missing file yields an empty state. Corrupt file raises."""
    if not os.path.exists(path):
        return {'schema_version': SCHEMA_VERSION, 'records': []}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise StateCorruptionError(f'corrupt order state file {path}: {exc}') from exc
    return validate_order_state(data)


def broker_status_code(exc):
    """Best-effort HTTP status from an SDK exception (structured, redaction-safe).

    alpaca_trade_api.rest.APIError keeps the HTTP code on `_status_code`;
    alpaca-py style errors expose `.response.status_code`; some types set `.status_code`.
    """
    resp = getattr(exc, 'response', None)
    status_code = getattr(resp, 'status_code', None)
    if status_code is None:
        status_code = getattr(exc, '_status_code', None) or getattr(exc, 'status_code', None)
        code_attr = getattr(exc, 'code', None)
        if status_code is None and code_attr is not None and str(code_attr) == '404':
            return 404
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def order_is_duplicate_rejection(exc):
    """True for HTTP 422 duplicate-client_order_id rejections.

    Structured first; the message fallback stays because legacy SDK paths may
    not expose a status. The phrase is a protocol constant, not a redacted log line.
    """
    if broker_status_code(exc) == 422:
        return True
    return 'client_order_id must be unique' in str(exc).lower()


def order_lookup_not_found(exc):
    """True when the broker definitively reports no order (HTTP 404 family).

    Message fallback covers "order not found"/"position does not exist" bodies.
    """
    if broker_status_code(exc) == 404:
        return True
    msg = str(exc).lower()
    return 'not found' in msg or 'does not exist' in msg


class ExecutionLock:
    """Per-symbol execution lockfile with stale-PID detection.

    PID liveness: os.kill(pid, 0) -> ProcessLookupError means stale (removed);
    PermissionError means a foreign-user process is alive (blocked).
    """

    def __init__(self, symbol, state_dir='trade'):
        self.symbol = symbol
        self.state_dir = state_dir

    @property
    def path(self):
        return os.path.join(self.state_dir, f'.lock_{self.symbol}')

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def acquire(self):
        """Acquire the lock; raises ExecutionBlocked if a live process holds it."""
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                payload = self._read_lock_payload()
                pid = payload.get('pid')
                if isinstance(pid, int) and not self._pid_alive(pid):
                    logger = _lock_logger()
                    logger.warning('Removing stale execution lock %s (pid %s dead)', self.path, pid)
                    with suppress(FileNotFoundError):
                        os.remove(self.path)
                    continue
                raise ExecutionBlocked(
                    f'Another trading process holds the lock {self.path} (pid={pid}); '
                    'exiting without starting a second execution for the same symbol.'
                ) from None
            except OSError as exc:
                raise ExecutionBlocked(f'Cannot acquire execution lock {self.path}: {exc}') from exc
            with os.fdopen(fd, 'w') as f:
                json.dump({'pid': os.getpid(), 'acquired_at': str(datetime_now_iso())}, f)
            return
        raise ExecutionBlocked(f'Could not acquire execution lock {self.path}')

    def _read_lock_payload(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def release(self):
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except OSError:
            pass


def datetime_now_iso():
    import datetime as _dt

    return _dt.datetime.now().isoformat()


_lock_logger_cache = None


def _lock_logger():
    global _lock_logger_cache
    if _lock_logger_cache is None:
        import logging

        _lock_logger_cache = logging.getLogger('market_breadth_trade')
    return _lock_logger_cache
