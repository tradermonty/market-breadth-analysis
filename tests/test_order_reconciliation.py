"""Tests for Issue #6: duplicate-order prevention and broker-state reconciliation.

Covers the plan in docs/plan_issue6_order_reconciliation.md (rev 4):
- deterministic client_order_id + 42 2-adopt recovery
- order-state persistence / fail-closed intent persist
- credit-marker idempotence for entry/exit fills
- reconcile (startup + --reconcile-only) and blocking
- execution lock and stale lock handling
- all tests are API-free (FakeApi fakes)
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from trade.order_state import (
    COID_MAX_LEN,
    STATUS_ACCEPTED,
    STATUS_INTENT,
    STATUS_RESOLVED_CANCELED,
    STATUS_RESOLVED_FILLED,
    STATUS_RESOLVED_NO_ORDER,
    STATUS_RESOLVED_TESTMODE,
    STATUS_SUBMITTED_UNKNOWN,
)
from trade.run_market_breadth_trade import (
    ExecutionBlocked,
    ExecutionLock,
    MarketBreadthTrader,
    StateCorruptionError,
    build_client_order_id,
    run_reconcile_only,
)


class DuplicateCOIDError(Exception):
    """Mimics Alpaca v2 rejecting an already-active client_order_id (HTTP 422)."""

    def __init__(self):
        self.response = SimpleNamespace(status_code=422)
        super().__init__('client_order_id must be unique')


class FakeApi:
    """API-free fake of the Alpaca REST surface used by the trader."""

    def __init__(self):
        self.orders = {}  # client_order_id -> SimpleNamespace order
        self.submit_calls = []

    def submit_order(self, symbol, qty, side, type, time_in_force, client_order_id=None):
        self.submit_calls.append(client_order_id)
        if client_order_id in self.orders:
            raise DuplicateCOIDError()
        order = SimpleNamespace(
            id=f'broker-{client_order_id}',
            client_order_id=client_order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            status='new',
            filled_qty=0,
            filled_avg_price=None,
        )
        self.orders[client_order_id] = order
        return order

    def get_order_by_client_order_id(self, coid):
        return self.orders.get(coid)

    def get_order(self, order_id):
        for o in self.orders.values():
            if o.id == order_id:
                return o
        return None

    def get_position(self, symbol):
        raise Exception('position does not exist')


class OrderReconciliationTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self._tmpdir, 'trade'), exist_ok=True)
        self._old_cwd = os.getcwd()
        os.chdir(self._tmpdir)
        self.fake_api = FakeApi()
        self.trader = MarketBreadthTrader(alpaca_api=self.fake_api, symbol='SSO')

    def tearDown(self):
        os.chdir(self._old_cwd)

    # --- helpers ---------------------------------------------------------
    def _state(self):
        with open(self.trader._order_state_path()) as f:
            return json.load(f)

    def _seed_record(
        self,
        coid,
        signal_type='long_trough',
        status=STATUS_INTENT,
        signal_date='2026-03-10',
        side='B',
        qty=100,
        retry_count=0,
    ):
        rec = {
            'client_order_id': coid,
            'signal_type': signal_type,
            'signal_date': signal_date,
            'side': side,
            'qty': qty,
            'retry_count': retry_count,
            'status': status,
            'submitted_at': None,
            'resolved_at': None,
            'fill_watermark': 0,
        }
        self.trader._write_order_record(rec)
        return rec

    def _filled(self, coid, filled_qty, filled_avg_price=52.5):
        order = self.fake_api.orders.get(coid)
        if order is not None:
            order.status = 'filled'
            order.filled_qty = filled_qty
            order.filled_avg_price = filled_avg_price
        else:
            self.fake_api.orders[coid] = SimpleNamespace(
                id=f'broker-{coid}',
                client_order_id=coid,
                symbol='SSO',
                qty=filled_qty,
                side='sell',
                status='filled',
                filled_qty=filled_qty,
                filled_avg_price=filled_avg_price,
            )
        return self.fake_api.orders[coid]


class TestClientOrderId(OrderReconciliationTestBase):
    def test_01_format_charset_retry_suffix_and_48_cap(self):
        coid = build_client_order_id('SSO', 'long_trough', '2026-03-10', 'buy')
        self.assertEqual(coid, 'mbt-SSO-long_trough-2026-03-10-B')
        self.assertLessEqual(len(coid), COID_MAX_LEN)
        r1 = build_client_order_id('SSO', 'long_trough', '2026-03-10', 'buy', retry_count=1)
        r10 = build_client_order_id('SSO', 'long_trough', '2026-03-10', 'buy', retry_count=10)
        self.assertNotEqual(r1, r10)
        self.assertTrue(r1.endswith('-BR1'))
        stop_a = build_client_order_id('SSO', 'stop', '2026-03-10', 'sell', stop_time='1432')
        stop_b = build_client_order_id('SSO', 'stop', '2026-03-10', 'sell', stop_time='1433')
        self.assertNotEqual(stop_a, stop_b)
        # 48-char overflow must fail at build time
        with self.assertRaises(ValueError):
            build_client_order_id('VERYLONGSYMBOLTHATISLONG', 'long_trough', '2026-03-10', 'buy')

    def test_11_co_id_stop_same_day_collision_free(self):
        a = build_client_order_id('SSO', 'stop', '2026-03-10', 'sell', stop_time='1400')
        b = build_client_order_id('SSO', 'stop', '2026-03-10', 'sell', stop_time='1400', stop_seq=1)
        self.assertNotEqual(a, b)
        self.assertEqual(b, a + '-1')


class BrokerOrderNotFound(Exception):
    """Mimics Alpaca raising (404) for an unknown client_order_id lookup."""

    def __init__(self):
        self.response = SimpleNamespace(status_code=404)
        self._status_code = 404
        super().__init__('order not found for client_order_id')


class TestForbiddenLookups(OrderReconciliationTestBase):
    def test_27_lookup_404_resolves_no_order_not_blocked(self):
        coid = 'mbt-SSO-long_trough-2026-03-20-B'
        self._seed_record(coid, signal_date='2026-03-20', status=STATUS_INTENT)

        def raise_404(coid):
            raise BrokerOrderNotFound()

        self.fake_api.get_order_by_client_order_id = raise_404
        report = self.trader.reconcile_pending_orders()
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], STATUS_RESOLVED_NO_ORDER)
        self.assertEqual(report[0]['outcome'], 'resolved-no-order')
        self.assertEqual(self.fake_api.submit_calls, [])

    def test_28_lookup_5xx_still_blocks(self):
        coid = 'mbt-SSO-long_trough-2026-03-21-B'
        self._seed_record(coid, signal_date='2026-03-21', status=STATUS_INTENT)
        self.fake_api.get_order_by_client_order_id = lambda coid: (_ for _ in ()).throw(ConnectionError('server error'))
        with self.assertRaises(ExecutionBlocked):
            self.trader.reconcile_pending_orders()


class TestReconciliation(OrderReconciliationTestBase):
    def test_02_crash_before_submission_resolves_no_order(self):
        self._seed_record('mbt-SSO-long_trough-2026-03-10-B', status=STATUS_INTENT)
        report = self.trader.reconcile_pending_orders()
        state = self._state()
        rec = state['records'][0]
        self.assertEqual(rec['status'], STATUS_RESOLVED_NO_ORDER)
        self.assertEqual(self.fake_api.submit_calls, [])
        self.assertEqual(report[0]['outcome'], 'resolved-no-order')

    def test_03_crash_after_acceptance_repairs_entry_fill_exactly_once(self):
        coid = 'mbt-SSO-long_trough-2026-03-10-B'
        self._seed_record(coid)
        self._filled(coid, filled_qty=10, filled_avg_price=52.5)
        self.trader.reconcile_pending_orders()
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], STATUS_RESOLVED_FILLED)
        self.assertEqual(self.trader.entry_lots, [{'price': 52.5, 'shares': 10}])
        self.assertEqual(self.trader.applied_credit.get(coid), 10)
        self.assertIn(('long_trough', '2026-03-10'), self.trader._acted_signals)
        # idempotence: reconcile again must not double-credit
        self.trader.reconcile_pending_orders()
        self.assertEqual(self.trader.entry_lots, [{'price': 52.5, 'shares': 10}])

    def test_04_response_loss_reconcile_finds_and_repairs_no_second_submit(self):
        coid = 'mbt-SSO-short_trough-2026-03-11-B'
        self._seed_record(coid, signal_type='short_trough', signal_date='2026-03-11', status=STATUS_SUBMITTED_UNKNOWN)
        self._filled(coid, filled_qty=7, filled_avg_price=51.0)
        self.trader.reconcile_pending_orders()
        self.assertEqual(self.fake_api.submit_calls, [])
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], STATUS_RESOLVED_FILLED)
        self.assertEqual(self.trader.entry_lots, [{'price': 51.0, 'shares': 7}])

    def test_07_partial_fill_then_cancel_credits_delta_exactly_once(self):
        coid = 'mbt-SSO-long_trough-2026-03-12-B'
        self._seed_record(coid, signal_date='2026-03-12', qty=10)
        order = self._filled(coid, filled_qty=5, filled_avg_price=50.0)
        order.status = 'canceled'
        self.trader.reconcile_pending_orders()
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], STATUS_RESOLVED_FILLED)
        self.assertEqual(self.trader.entry_lots, [{'price': 50.0, 'shares': 5}])
        # NO retry for a resolved-filled (partial) record
        self.trader.reconcile_pending_orders()
        self.assertEqual(self.trader.entry_lots, [{'price': 50.0, 'shares': 5}])
        self.assertEqual(self.fake_api.submit_calls, [])

    def test_16_peak_exit_sell_credit_idempotent(self):
        coid = 'mbt-SSO-peak-2026-03-13-S'
        self._seed_record(coid, signal_type='peak', signal_date='2026-03-13', side='S', qty=20)
        self.trader.entry_lots = [{'price': 51.0, 'shares': 20}]
        self._filled(coid, filled_qty=20, filled_avg_price=49.0)
        self.trader.reconcile_pending_orders()
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], STATUS_RESOLVED_FILLED)
        self.assertEqual(self.trader.entry_lots, [])
        self.assertIn(('peak', '2026-03-13'), self.trader._acted_signals)
        self.trader.reconcile_pending_orders()
        self.assertEqual(self.trader.entry_lots, [])

    def test_16b_stop_repair_skips_acted_upsert(self):
        coid = 'mbt-SSO-stop-2026-03-13-1432-S'
        self._seed_record(coid, signal_type='stop', signal_date='2026-03-13', side='S', qty=20)
        self.trader.entry_lots = [{'price': 51.0, 'shares': 20}]
        self._filled(coid, filled_qty=20, filled_avg_price=49.0)
        self.trader.reconcile_pending_orders()
        self.assertEqual(self.trader.entry_lots, [])
        self.assertNotIn(('stop', '2026-03-13'), self.trader._acted_signals)

    def _entry_prices_file(self):
        with open(self.trader._entry_prices_path()) as f:
            return json.load(f)

    def test_19_partial_sell_fill_reconcile_keeps_remaining_lots(self):
        coid = 'mbt-SSO-peak-2026-03-14-S'
        self._seed_record(coid, signal_type='peak', signal_date='2026-03-14', side='S', qty=30)
        self.trader.entry_lots = [{'price': 50.0, 'shares': 30}]
        self.trader.entry_prices = [50.0]
        # crash after broker accepted a PARTIAL sell of 12 (broker then holds 18)
        self._filled(coid, filled_qty=12, filled_avg_price=49.0)
        order = self.fake_api.orders[coid]
        order.status = 'canceled'
        self.fake_api.get_position = lambda symbol: SimpleNamespace(qty='18')

        self.trader.reconcile_pending_orders()

        saved = self._entry_prices_file()
        self.assertEqual(saved['lots'], [{'price': 50.0, 'shares': 18}])
        self.assertEqual(saved['applied_credit'][coid], 12)
        # broker re-synced at the end of reconcile
        self.assertEqual(self.trader.current_position, 18)

    def test_20_crash_window_exhaustive_repair_exactly_once(self):
        # Crash right after the acted_signals write (step 1): acted key present,
        # no lots/credit, record still accepted.
        coid = 'mbt-SSO-long_trough-2026-03-15-B'
        self._seed_record(coid, signal_date='2026-03-15')
        self._filled(coid, filled_qty=5, filled_avg_price=52.0)
        self.trader._acted_signals.add(('long_trough', '2026-03-15'))
        self.trader._save_acted_signals()
        # remove acted file to prove the repair rebuilds from order state, not memory
        self.trader.reconcile_pending_orders()
        saved = self._entry_prices_file()
        self.assertEqual(saved['lots'], [{'price': 52.0, 'shares': 5}])
        self.assertEqual(saved['applied_credit'][coid], 5)
        # every crash-window re-run must be a no-op
        for _ in range(3):
            self.trader.reconcile_pending_orders()
        saved = self._entry_prices_file()
        self.assertEqual(saved['lots'], [{'price': 52.0, 'shares': 5}])

    def test_21_no_tmp_residue_and_state_parseable_after_reconcile(self):
        self._seed_record('mbt-SSO-long_trough-2026-03-16-B', signal_date='2026-03-16')
        self._filled('mbt-SSO-long_trough-2026-03-16-B', filled_qty=3, filled_avg_price=50.0)
        self.trader.reconcile_pending_orders()
        trade_dir = os.path.dirname(self.trader._order_state_path())
        leftovers = [f for f in os.listdir(trade_dir) if f.startswith('.tmp-state-')]
        self.assertEqual(leftovers, [])
        self._state()  # parseable

    def test_22_reconcile_adopts_open_order_and_repairs_fill(self):
        coid = 'mbt-SSO-long_trough-2026-03-17-B'
        self._seed_record(coid, signal_date='2026-03-17')
        open_order = SimpleNamespace(
            id='broker-open',
            client_order_id=coid,
            symbol='SSO',
            qty=9,
            side='buy',
            status='new',
            filled_qty=0,
            filled_avg_price=None,
        )
        self.fake_api.orders[coid] = open_order
        calls = []

        fake_get_order = self.fake_api.get_order

        def get_order(order_id):
            calls.append(order_id)
            if len(calls) >= 2:
                open_order.status = 'filled'
                open_order.filled_qty = 9
                open_order.filled_avg_price = 53.0
            return fake_get_order(order_id)

        self.fake_api.get_order = get_order
        self.trader.reconcile_pending_orders()
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], STATUS_RESOLVED_FILLED)
        self.assertEqual(self.trader.entry_lots, [{'price': 53.0, 'shares': 9}])

    def test_23_entry_fill_without_price_is_blocked(self):
        coid = 'mbt-SSO-long_trough-2026-03-18-B'
        self._seed_record(coid, signal_date='2026-03-18')
        self._filled(coid, filled_qty=5, filled_avg_price=None)
        with self.assertRaises(ExecutionBlocked):
            self.trader.reconcile_pending_orders()
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], 'intent')

    def test_24_record_metadata_survives_partial_updates(self):
        coid = 'mbt-SSO-long_trough-2026-03-19-B'
        self._seed_record(coid, signal_date='2026-03-19', retry_count=2, qty=44)
        self.trader._update_order_record(coid, status='accepted', submitted_at='now')
        self._filled(coid, filled_qty=44, filled_avg_price=50.0)
        self.trader.reconcile_pending_orders()
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], STATUS_RESOLVED_FILLED)
        self.assertEqual(rec['signal_type'], 'long_trough')
        self.assertEqual(rec['signal_date'], '2026-03-19')
        self.assertEqual(rec['retry_count'], 2)
        self.assertEqual(rec['qty'], 44)
        # a same-day re-signal therefore produces a new R suffix, not the same COID
        self.assertEqual(
            build_client_order_id('SSO', 'long_trough', '2026-03-19', 'buy', retry_count=1),
            coid + 'R1',
        )

    def test_29_repair_broker_order_without_client_order_id_attr(self):
        # Regression (review round 3): a broker object with no client_order_id
        # attribute must not crash the reconcile path with AttributeError, and
        # credit must key on the persisted record's COID.
        coid = 'mbt-SSO-long_trough-2026-03-20-B'
        self._seed_record(coid, signal_date='2026-03-20')
        self.fake_api.orders[coid] = SimpleNamespace(
            id='broker-x',
            signal='x',
            symbol='SSO',
            qty=6,
            side='buy',
            status='filled',
            filled_qty=6,
            filled_avg_price=50.0,
        )
        self.fake_api.orders[coid].__dict__.pop('client_order_id', None)
        self.assertFalse(hasattr(self.fake_api.orders[coid], 'client_order_id'))
        self.trader.reconcile_pending_orders()
        saved = self._entry_prices_file()
        self.assertEqual(saved['applied_credit'][coid], 6)
        self.assertEqual(self._state()['records'][0]['status'], STATUS_RESOLVED_FILLED)

    def test_30_foreign_lock_permission_error_treated_alive(self):
        lock = ExecutionLock('SSO')
        with open(lock.path, 'w') as f:
            f.write(json.dumps({'pid': 12345, 'acquired_at': 'now'}))
        try:
            with (
                patch('os.kill', side_effect=PermissionError('not yours')),
                self.assertRaises(ExecutionBlocked),
            ):
                lock.acquire()
        finally:
            if os.path.exists(lock.path):
                os.remove(lock.path)

    def test_31_blocked_reconcile_only_keeps_partial_report(self):
        import io
        from contextlib import redirect_stdout

        coid_ok = 'mbt-SSO-long_trough-2026-03-22-A'
        coid_bad = 'mbt-SSO-long_trough-2026-03-22-B'
        self._seed_record(coid_ok, signal_date='2026-03-22')
        self._seed_record(coid_bad, signal_date='2026-03-22', status=STATUS_ACCEPTED)
        self.fake_api.get_order_by_client_order_id = lambda c: (
            None if c == coid_ok else (_ for _ in ()).throw(ConnectionError('down'))
        )
        buf = io.StringIO()
        args = SimpleNamespace(symbol='SSO', testmode=False, debug=False)
        with redirect_stdout(buf):
            code = run_reconcile_only(args, api=self.fake_api)
        payload = json.loads(buf.getvalue())
        self.assertEqual(code, 1)
        self.assertTrue(payload['blocked'])
        self.assertTrue(payload['records'])  # partial report is preserved
        self.assertEqual(payload['records'][0]['outcome'], 'resolved-no-order')


class TestCliParsing(unittest.TestCase):
    def test_25_reconcile_only_dash_alias_and_blocked_json_stdout(self):
        tmp = tempfile.mkdtemp()
        subprocess.run(['mkdir', '-p', os.path.join(tmp, 'trade')], check=True)
        with open(os.path.join(tmp, 'trade', 'order_state_TST.json'), 'w') as f:
            f.write('corrupt{')
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade', 'run_market_breadth_trade.py')
        result = subprocess.run(
            [sys.executable, script, '--symbol', 'TST', '--reconcile-only'],
            capture_output=True,
            text=True,
            cwd=tmp,
            timeout=60,
        )
        self.assertIn(result.returncode, (0, 1))
        payload = json.loads(result.stdout)
        self.assertIn('records', payload)

    def test_26_reconcile_only_alias_underscore_form(self):
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, 'trade'), exist_ok=True)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade', 'run_market_breadth_trade.py')
        result = subprocess.run(
            [sys.executable, script, '--symbol', 'TST', '--reconcile_only'],
            capture_output=True,
            text=True,
            cwd=tmp,
            timeout=60,
        )
        self.assertIn(result.returncode, (0, 1))
        json.loads(result.stdout)


class TestBlockedStates(OrderReconciliationTestBase):
    def test_08_corrupt_order_state_blocks(self):
        with open(self.trader._order_state_path(), 'w') as f:
            f.write('not json{')
        with self.assertRaises(ExecutionBlocked):
            self.trader.reconcile_pending_orders()

    def test_08b_corrupt_acted_signals_blocks(self):
        with open(self.trader._acted_signals_path(), 'w') as f:
            f.write('corrupt{')
        with self.assertRaises(StateCorruptionError):
            MarketBreadthTrader(alpaca_api=self.fake_api, symbol='SSO')

    def test_09_reconcile_lookup_failure_blocks(self):
        self._seed_record('mbt-SSO-long_trough-2026-03-10-B')
        self.trader.api.get_order_by_client_order_id = lambda coid: (_ for _ in ()).throw(Exception('network down'))
        with self.assertRaises(ExecutionBlocked):
            self.trader.reconcile_pending_orders()


class TestSubmission(OrderReconciliationTestBase):
    def test_06_intent_persist_failure_is_fail_closed(self):
        def broken_write(path, data):
            raise OSError('disk full')

        with (
            patch('trade.run_market_breadth_trade.atomic_write_json', side_effect=broken_write),
            self.assertRaises(ExecutionBlocked),
        ):
            self.trader.execute_buy(
                100, reason='test', signal_type='long_trough', signal_date=datetime.fromisoformat('2026-03-10')
            )
        self.assertEqual(self.fake_api.submit_calls, [])

    def test_05_duplicate_coid_adopts_existing_order(self):
        coid = build_client_order_id('SSO', 'long_trough', '2026-03-10', 'buy')
        existing = self.fake_api.orders[coid] = SimpleNamespace(
            id='broker-1',
            client_order_id=coid,
            symbol='SSO',
            qty=100,
            side='buy',
            status='new',
            filled_qty=0,
            filled_avg_price=None,
        )
        order = self.trader.execute_buy(
            100, reason='test', signal_type='long_trough', signal_date=datetime.fromisoformat('2026-03-10')
        )
        self.assertEqual(len(self.fake_api.submit_calls), 1)
        self.assertEqual(order, existing)
        rec = self._state()['records'][0]
        self.assertEqual(rec['status'], 'accepted')

    def test_10_retry_count_persisted_before_retry_submit(self):
        first = build_client_order_id('SSO', 'long_trough', '2026-03-10', 'buy')
        self._seed_record(first, status=STATUS_RESOLVED_CANCELED)

        snapshots = []
        orig_submit = self.fake_api.submit_order

        def spy_submit(**kwargs):
            snapshots.append(self._state())
            return orig_submit(**kwargs)

        self.fake_api.submit_order = spy_submit
        self.trader.execute_buy(
            100, reason='retry test', signal_type='long_trough', signal_date=datetime.fromisoformat('2026-03-10')
        )
        self.assertEqual(self.fake_api.submit_calls[0], first + 'R1')
        rec = snapshots[0]['records'][-1]
        self.assertEqual(rec['retry_count'], 1)
        self.assertEqual(rec['status'], STATUS_INTENT)

    def test_10b_zero_fill_canceled_then_reconcile_no_direct_send(self):
        self._seed_record('mbt-SSO-long_trough-2026-03-10-B', status=STATUS_RESOLVED_CANCELED)
        stdout, code = self._capture_reconcile_only()
        self.assertEqual(code, 0)
        self.assertEqual(self.fake_api.submit_calls, [])

    def _capture_reconcile_only(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        args = SimpleNamespace(symbol='SSO', testmode=False, debug=False)
        with redirect_stdout(buf):
            code = run_reconcile_only(args, api=self.fake_api)
        return buf.getvalue(), code

    def test_15_reconcile_only_outputs_parseable_json(self):
        self._seed_record('mbt-SSO-long_trough-2026-03-10-B', status=STATUS_INTENT)
        out, code = self._capture_reconcile_only()
        payload = json.loads(out)
        self.assertIn('records', payload)
        self.assertEqual(code, 0)


class TestTestmode(OrderReconciliationTestBase):
    def test_15b_testmode_persists_intent_and_resolves_unverified(self):
        with patch('trade.run_market_breadth_trade.MarketBreadthTrader._initialize_alpaca'):
            t = MarketBreadthTrader(alpaca_api=self.fake_api, symbol='SSO', testmode=True, test_date='2026-03-10')
        t.prepare_test_dt()
        order = t.execute_buy(
            100, reason='test', signal_type='long_trough', signal_date=datetime.fromisoformat('2026-03-10')
        )
        self.assertEqual(order.id, 'TEST')
        state = self._state()
        self.assertEqual(len(state['records']), 1)
        self.assertEqual(state['records'][0]['status'], STATUS_RESOLVED_TESTMODE)
        self.assertEqual(self.fake_api.submit_calls, [])


class TestExecutionBlockedException(OrderReconciliationTestBase):
    def test_17_run_propagates_execution_blocked(self):
        self._seed_record('mbt-SSO-long_trough-2026-03-10-B')
        self.trader.api.get_order_by_client_order_id = lambda coid: (_ for _ in ()).throw(Exception('network down'))
        with (
            patch.object(self.trader, 'is_market_open', return_value=True),
            patch.object(self.trader, 'is_closing_time_range', return_value=True),
            patch.object(self.trader, 'analyze_market'),
            patch.object(self.trader, 'check_signals_and_trade'),
            self.assertRaises(ExecutionBlocked),
        ):
            self.trader.run()


class TestExecutionLock(OrderReconciliationTestBase):
    def test_13_stale_lock_removed_live_lock_blocks(self):
        lock = ExecutionLock('SSO')
        lock_path = lock.path
        with open(lock_path, 'w') as f:
            f.write(json.dumps({'pid': 999999999, 'acquired_at': '2026-01-01T00:00:00'}))
        # dead pid -> stale -> removed, acquire succeeds
        lock.acquire()
        self.assertTrue(os.path.exists(lock_path))
        lock.release()
        # live pid (this process) -> blocked
        with open(lock_path, 'w') as f:
            f.write(json.dumps({'pid': os.getpid(), 'acquired_at': 'now'}))
        with self.assertRaises(ExecutionBlocked):
            lock.acquire()
        os.remove(lock_path)

    def test_14_second_process_blocked_before_log_write(self):
        lock_content = json.dumps({'pid': os.getpid(), 'acquired_at': 'now'})
        with open(os.path.join('trade', '.lock_TST'), 'w') as f:
            f.write(lock_content)
        result = subprocess.run(
            [sys.executable, os.path.join(self._old_cwd, 'trade', 'run_market_breadth_trade.py'), '--symbol', 'TST'],
            capture_output=True,
            text=True,
            cwd=self._tmpdir,
            timeout=60,
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse(
            os.path.exists(os.path.join('trade', 'market_breadth_trade.log')),
            'log file must not be created when lock rejection occurs before logging setup',
        )


if __name__ == '__main__':
    unittest.main()
