"""API-free acceptance regressions for Issue #12's comparison tools."""

import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import secret_redaction
from scripts import compare_tv_chart_modes as modes
from scripts import compare_tv_python_trades as trades

ROOT = Path(__file__).resolve().parents[1]
DAYS = pd.bdate_range('2024-01-02', periods=90)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Comparison tests must not access the network')

    monkeypatch.setattr(socket.socket, 'connect', blocked)


def frame(rows=None):
    if rows is None:
        rows = [(0, 3, 'peak_exit'), (10, 15, 'stop_loss')]
    return pd.DataFrame(
        [(DAYS[a], DAYS[b], reason) for a, b, reason in rows],
        columns=['entry_date', 'exit_date', 'exit_reason_norm'],
    )


def cli(tmp_path, py, tv, calendar=DAYS, tolerance=1):
    py = py.rename(columns={'exit_reason_norm': 'exit_reason'})
    tv = tv.rename(columns={'exit_reason_norm': 'reason'})
    py.to_csv(tmp_path / 'py.csv', index=False)
    tv.to_csv(tmp_path / 'tv.csv', index=False)
    pd.DataFrame({'price': 100.0}, index=calendar).to_csv(tmp_path / 'price.csv')
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/compare_tv_python_trades.py'),
            '--python_csv',
            str(tmp_path / 'py.csv'),
            '--tv_csv',
            str(tmp_path / 'tv.csv'),
            '--price_csv',
            str(tmp_path / 'price.csv'),
            '--tolerance_bars',
            str(tolerance),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_01_later_27_bar_mismatch_fails_api_and_real_cli(tmp_path):
    py = frame()
    tv = frame([(1, 4, 'peak_exit'), (37, 42, 'stop_loss')])
    result = trades.compare_trades(py, tv, DAYS, 1)
    assert not result['pass']
    assert result['first_mismatch_bar_diff'] == 1
    assert result['max_bar_diff'] == 27
    proc = cli(tmp_path, py, tv)
    assert proc.returncode != 0
    assert 'Verdict: FAIL' in proc.stdout
    assert '27' in proc.stdout


@pytest.mark.parametrize('offset', [0, 1])
def test_02_exact_and_tolerated_matches_pass_cli(tmp_path, offset):
    proc = cli(tmp_path, frame(), frame([(offset, 3 + offset, 'peak_exit'), (10 + offset, 15 + offset, 'stop_loss')]))
    assert proc.returncode == 0, proc.stderr
    assert 'Verdict: PASS' in proc.stdout


def test_03_late_reason_mismatch_is_not_hidden_by_85_percent_agreement():
    py = frame([(i * 3, i * 3 + 1, 'peak_exit') for i in range(10)])
    tv = py.copy()
    tv.loc[9, 'exit_reason_norm'] = 'stop_loss'
    result = trades.compare_trades(py, tv, DAYS, 1)
    assert not result['pass']
    assert result['reason_match_rate'] == 0.9


def test_04_duplicates_preserve_multiplicity_and_order():
    py = frame([(0, 3, 'peak_exit'), (0, 3, 'peak_exit'), (10, 15, 'stop_loss')])
    result = trades.compare_trades(py.iloc[::-1], py, DAYS, 0)
    assert result['pass']
    assert result['exact_matches'] == 3
    changed = frame([(0, 3, 'peak_exit'), (10, 15, 'stop_loss'), (10, 15, 'stop_loss')])
    result = trades.compare_trades(py, changed, DAYS, 0)
    assert not result['pass']
    assert len(result['python_only']) == len(result['tv_only']) == 1


@pytest.mark.parametrize(
    'other', [[], [(0, 3, 'peak_exit')], [(0, 3, 'peak_exit'), (5, 7, 'stop_loss'), (10, 15, 'stop_loss')]]
)
def test_05_missing_inserted_or_empty_trades_fail(other):
    assert not trades.compare_trades(frame(), frame(other), DAYS, 1)['pass']


def test_06_both_empty_do_not_claim_equivalence(tmp_path):
    proc = cli(tmp_path, frame([]), frame([]))
    assert proc.returncode != 0
    assert 'Verdict: FAIL' in proc.stdout


@pytest.mark.parametrize('calendar', [DAYS[:0], DAYS.insert(1, DAYS[0]), DAYS.delete(0)])
def test_07_invalid_or_incomplete_calendar_never_passes(tmp_path, calendar):
    proc = cli(tmp_path, frame(), frame(), calendar)
    assert proc.returncode != 0
    assert 'Verdict: PASS' not in proc.stdout


def test_08_negative_tolerance_rejected(tmp_path):
    assert cli(tmp_path, frame(), frame(), tolerance=-1).returncode != 0


@pytest.mark.parametrize('bad_date', ['not-a-date', '', '2024-01-02T00:00:00Z'])
def test_09_malformed_closed_tv_rows_not_silently_dropped(tmp_path, bad_date):
    path = tmp_path / 'tv.csv'
    pd.DataFrame(
        {'entry_date': ['2024-01-02', bad_date], 'exit_date': ['2024-01-05'] * 2, 'reason': ['peak exit'] * 2}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError):
        trades.load_tv_trades(str(path))


@pytest.mark.parametrize('reason', ['', None])
def test_10_blank_reason_cannot_be_successful_evidence(tmp_path, reason):
    path = tmp_path / 'py.csv'
    pd.DataFrame({'entry_date': ['2024-01-02'], 'exit_date': ['2024-01-05'], 'exit_reason': [reason]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError):
        trades.load_python_trades(str(path))


def raw_rows():
    return [[1, 'Entry long', '2024-01-02 10:00', 'Buy'], [1, 'Exit long', '2024-01-05 14:00', 'Stop-loss']]


def raw_csv(tmp_path, rows):
    path = tmp_path / 'raw.csv'
    pd.DataFrame(rows, columns=['Trade #', 'Type', 'Date and time', 'Signal']).to_csv(path, index=False)
    return str(path)


def test_11_raw_tv_closed_and_open_trade_contract(tmp_path):
    rows = [*raw_rows(), [2, 'Entry long', '2024-01-08', 'Buy'], [2, 'Exit long', '2024-01-09', 'Open']]
    result = trades.load_tv_trades(raw_csv(tmp_path, rows))
    assert len(result) == 1
    assert result.iloc[0]['exit_reason_norm'] == 'stop_loss'


@pytest.mark.parametrize('bad', ['duplicate', 'malformed', 'missing-entry', 'missing-id', 'reverse', 'short'])
def test_12_ambiguous_or_invalid_raw_closed_trades_fail(tmp_path, bad):
    rows = raw_rows()
    if bad == 'duplicate':
        rows.append(rows[1].copy())
    elif bad == 'malformed':
        rows[1][2] = 'broken'
    elif bad == 'missing-entry':
        rows = rows[1:]
    elif bad == 'missing-id':
        rows[1][0] = None
    elif bad == 'reverse':
        rows[0][2] = '2024-01-08'
    else:
        rows[0][1] = 'Entry short'
        rows[1][1] = 'Exit short'
    with pytest.raises(ValueError):
        trades.load_tv_trades(raw_csv(tmp_path, rows))


def test_13_real_tv_and_chart_runs_with_only_acquisition_and_render_boundaries_mocked(monkeypatch):
    dates = pd.bdate_range('2024-01-02', periods=260)
    stocks = pd.DataFrame({'A': np.linspace(100, 200, len(dates))}, index=dates)
    prices = pd.DataFrame({'adjusted_close': np.linspace(100, 120, len(dates))}, index=dates)
    monkeypatch.setattr(modes.Backtest, '_get_sp500_data', lambda self: stocks.copy())
    monkeypatch.setattr('backtest.backtest.get_stock_price_ohlc', lambda *a, **k: prices.copy())
    monkeypatch.setattr('backtest.backtest.get_stock_price_data', lambda *a, **k: prices.copy())
    renders = []
    monkeypatch.setattr(
        modes.Backtest, 'visualize_results', lambda self, **kwargs: renders.append((self.tv_mode, self.chart_mode))
    )
    for mode in ['tv', 'chart']:
        result = modes.run_single('SPY', mode)
        assert 'Total Return' in result
    assert renders == [(True, False), (False, True)]


@pytest.mark.parametrize('empty_equity', [False, True])
def test_14_silent_or_empty_run_is_failure(monkeypatch, empty_equity):
    def empty(self):
        if empty_equity:
            self.price_data = pd.DataFrame({'adjusted_close': [100.0]}, index=DAYS[:1])
            self.calculate_performance()

    monkeypatch.setattr(modes.Backtest, 'run', empty)
    with pytest.raises(ValueError, match='observations'):
        modes.run_single('SPY', 'tv')


def test_15_unknown_mode_rejected():
    with pytest.raises(ValueError, match='mode'):
        modes.run_single('SPY', 'unknown')


@pytest.mark.parametrize('failed_modes', [set(), {'chart'}, {'tv', 'chart'}])
def test_16_batch_status_and_artifacts_reflect_partial_or_total_failure(monkeypatch, tmp_path, failed_modes):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modes, 'ETFS', ['SPY'])
    monkeypatch.setattr(modes.time, 'sleep', lambda _: None)
    calls = []

    def run(symbol, mode):
        calls.append(mode)
        if mode in failed_modes:
            raise ValueError('fixture failure')
        return dict.fromkeys([*modes.METRICS, 'B&H Return', 'B&H Max DD'], 0.0)

    monkeypatch.setattr(modes, 'run_single', run)
    code = modes.main()
    assert calls == ['tv', 'chart']
    assert (code == 0) == (not failed_modes)
    report = (tmp_path / 'reports/tv_vs_chart_mode_comparison.md').read_text()
    assert ('Status: FAIL' if failed_modes else 'Status: PASS') in report
    for mode in ['tv', 'chart']:
        result = pd.read_csv(tmp_path / f'reports/{mode}_mode_results.csv')
        assert result.iloc[0]['Status'] == ('FAIL' if mode in failed_modes else 'PASS')


def test_17_failed_batch_real_cli_returns_nonzero(tmp_path):
    code = (
        'from scripts import compare_tv_chart_modes as m; import sys; '
        'm.ETFS=["SPY"]; m.time.sleep=lambda _:None; '
        'm.run_single=lambda *a: (_ for _ in ()).throw(ValueError("fixture failure")); '
        'sys.exit(m.main())'
    )
    env = dict(__import__('os').environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, '-c', code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 1, proc.stderr
    assert 'Status: FAIL' in proc.stdout


def test_18_numeric_trade_dates_are_not_silently_interpreted_as_epoch(tmp_path):
    path = tmp_path / 'py.csv'
    pd.DataFrame({'entry_date': [0], 'exit_date': [0], 'exit_reason': ['peak exit']}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        trades.load_python_trades(str(path))


def test_19_bad_mode_metrics_do_not_create_duplicate_success_rows(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modes, 'ETFS', ['SPY', 'QQQ'])
    monkeypatch.setattr(modes.time, 'sleep', lambda _: None)
    monkeypatch.setattr(modes, 'run_single', lambda *args: {})
    assert modes.main() == 1
    for mode in ['tv', 'chart']:
        results = pd.read_csv(tmp_path / f'reports/{mode}_mode_results.csv')
        assert results['Symbol'].tolist() == ['SPY', 'QQQ']
        assert results['Status'].tolist() == ['FAIL', 'FAIL']


@pytest.mark.parametrize('source', ['query', 'header', 'registered'])
def test_20_batch_failures_redact_credentials_before_output(monkeypatch, tmp_path, capsys, source):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modes, 'ETFS', ['SPY'])
    monkeypatch.setattr(modes.time, 'sleep', lambda _: None)
    monkeypatch.setattr(secret_redaction, '_registry', set())
    dummy = 'COMPARISON_DUMMY_VALUE_20261016'  # pragma: allowlist secret
    message = {
        'query': f'https://example.invalid/prices?apikey={dummy}',
        'header': f'Authorization: Bearer {dummy}',
        'registered': f'Provider error: {dummy}',
    }[source]
    if source == 'registered':
        secret_redaction.register_secret(dummy)

    def fail(*args):
        raise ValueError(message)

    monkeypatch.setattr(modes, 'run_single', fail)
    assert modes.main() == 1
    outputs = [capsys.readouterr().out, *(path.read_text() for path in (tmp_path / 'reports').iterdir())]
    assert len(outputs) == 4
    for output in outputs:
        assert dummy not in output
        assert secret_redaction.REDACTED in output
