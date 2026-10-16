"""Issue #11: risk and final exits must not inherit optional half profit exits."""

import socket

import pandas as pd
import pytest

from backtest.backtest import Backtest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Exit-policy regressions must not access the network')

    monkeypatch.setattr(socket.socket, 'connect', blocked)


def make_backtest(**kwargs):
    options = dict(
        start_date='2024-01-04',
        end_date='2024-01-09',
        initial_capital=10000,
        slippage=0.01,
        commission=0.002,
        partial_exit=True,
        use_saved_data=True,
        no_show_plot=True,
    )
    options.update(kwargs)
    bt = Backtest(**options)
    dates = pd.bdate_range('2024-01-04', periods=4)
    bt.price_data = pd.DataFrame(
        {'adjusted_close': [110.0] * 4, 'adjusted_open': [110.0] * 4, 'adjusted_low': [109.0] * 4},
        index=dates,
    )
    bt.breadth_index = pd.Series(0.5, index=dates)
    bt.short_ma_line = pd.Series(0.5, index=dates)
    bt.long_ma_line = pd.Series(0.5, index=dates)
    bt.long_ma_trend = pd.Series(1, index=dates)
    bt.short_ma_bottoms = []
    bt.long_ma_bottoms = []
    bt.peaks = []
    bt._tv_peak_signals = {}
    bt._tv_long_trough_signals = {}
    bt._tv_short_trough_signals = {}
    return bt


def seed_lots(bt, qty):
    """Use real entry bookkeeping, including distinct cost bases for FIFO."""
    date = bt.price_data.index[0]
    first = qty // 3
    if first:
        assert bt._execute_entry(date, 98.0, first, 'first lot')
    assert bt._execute_entry(date, 101.0, qty - first, 'second lot')
    bt.highest_price = 120.0


def assert_liquidated(bt, qty, *, final_equity=False):
    assert bt.current_position == 0
    assert bt.open_positions == []
    assert bt.entry_prices == []
    assert bt.highest_price is None
    assert not bt._half_exited
    assert bt._stage1_exit_date is None
    assert not bt._pending_trend_break
    buys = [t for t in bt.trades if t['action'] == 'BUY']
    sells = [t for t in bt.trades if t['action'] == 'SELL']
    assert sum(t['shares'] for t in sells) == qty
    assert sum(t['exit_shares'] for t in bt.trade_log) == qty
    total_cost = sum(t['total_cost'] for t in buys)
    proceeds = sum(t['total_proceeds'] for t in sells)
    assert sum(t['entry_cost'] for t in bt.trade_log) == pytest.approx(total_cost)
    assert sum(t['exit_proceeds'] for t in bt.trade_log) == pytest.approx(proceeds)
    assert bt.current_capital == pytest.approx(bt.initial_capital - total_cost + proceeds)
    assert bt.current_capital == pytest.approx(bt.initial_capital + sum(t['pnl_dollar'] for t in bt.trade_log))
    assert bt.trade_log[-1]['cumulative_pnl'] == pytest.approx(bt.current_capital - bt.initial_capital)
    for sell in sells:
        assert sell['commission'] == pytest.approx(sell['price'] * sell['shares'] * bt.commission)
        assert sell['total_proceeds'] == pytest.approx(sell['price'] * sell['shares'] - sell['commission'])
    if final_equity:
        assert bt.equity_curve[-1]['equity'] == pytest.approx(bt.current_capital)


@pytest.mark.parametrize('qty', [1, 5, 10])
@pytest.mark.parametrize(
    'options',
    [
        {'tv_mode': False},
        {'tv_mode': False, 'use_trailing_stop': True},
        {'tv_mode': False, 'chart_mode': True},
        {'tv_mode': True},
        {'tv_mode': True, 'use_trailing_stop': True},
        {'tv_mode': True, 'use_volatility_stop': True, 'vol_atr_period': 2, 'vol_atr_multiplier': 0.5},
        {
            'tv_mode': True,
            'use_volatility_stop': True,
            'vol_trailing_mode': False,
            'vol_atr_period': 2,
            'vol_atr_multiplier': 0.5,
        },
        {'tv_pine_compat': True},
    ],
    ids=[
        'legacy-fixed',
        'legacy-trailing',
        'chart-fixed',
        'tv-fixed',
        'tv-trailing',
        'tv-vol-trail',
        'tv-vol-fixed',
        'pine-fixed',
    ],
)
def test_01_stop_fully_liquidates_and_does_not_reenter(options, qty):
    bt = make_backtest(**options)
    seed_lots(bt, qty)
    date = bt.price_data.index[-1]
    bt.price_data['adjusted_close'] = [100.0, 104.0, 102.0, 85.0]
    bt.price_data.loc[date, ['adjusted_open', 'adjusted_low']] = [84.0, 80.0]
    # An entry signal on the same bar must not reopen the stopped position.
    bt.long_ma_bottoms = [date]
    bt._tv_long_trough_signals = {date: (date, 0.1)}
    runner = bt._execute_tv_mode_trades if bt.tv_mode else bt._execute_legacy_trades
    runner(3, date, 85.0)
    assert_liquidated(bt, qty)
    sell = bt.trades[-1]
    assert sell['reason'] == 'stop loss'
    raw_fill = 84.0 if bt.tv_mode else 85.0
    assert sell['price'] == pytest.approx(raw_fill * (1 - bt.slippage))
    assert bt._pending_entry is None


@pytest.mark.parametrize('qty', [1, 5, 10])
@pytest.mark.parametrize('tv_mode', [False, True])
def test_02_end_of_period_closes_all_lots_with_costs(tv_mode, qty):
    bt = make_backtest(tv_mode=tv_mode)
    seed_lots(bt, qty)
    bt.execute_trades()
    assert_liquidated(bt, qty, final_equity=True)
    assert bt.trades[-1]['reason'] == 'backtest_end'
    assert bt.trades[-1]['date'] == bt.price_data.index[-1]
    assert bt.trades[-1]['price'] == pytest.approx(110 * (1 - bt.slippage))


@pytest.mark.parametrize('tv_mode', [False, True])
def test_03_peak_profit_exit_still_halves_odd_quantity_fifo(tv_mode):
    bt = make_backtest(tv_mode=tv_mode)
    seed_lots(bt, 5)
    date = bt.price_data.index[1]
    bt.peaks = [date]
    bt._tv_peak_signals = {date: (date, 0.8)}
    runner = bt._execute_tv_mode_trades if bt.tv_mode else bt._execute_legacy_trades
    runner(1, date, 110.0)
    assert bt.current_position == 3
    assert len(bt.open_positions) == 1
    assert bt.open_positions[0]['entry_shares'] == 3
    assert bt.open_positions[0]['entry_reason'] == 'second lot'
    assert bt.trades[-1]['shares'] == 2
    assert bt.trades[-1]['reason'] == 'peak exit (partial)'
    assert bt.trade_log[0]['entry_reason'] == 'first lot'
    assert bt.trade_log[1]['entry_reason'] == 'second lot'
    assert bt.open_positions[0]['entry_cost'] == pytest.approx(3 * 101.0 * (1 + bt.slippage) * (1 + bt.commission))


def test_06_background_profit_exit_still_halves_position():
    bt = make_backtest(tv_mode=False, use_background_color_signals=True)
    seed_lots(bt, 5)
    bt.short_ma_line[:] = 0.4
    bt.long_ma_line[:] = 0.6
    bt.long_ma_trend[:] = -1
    bt.long_ma_trend.iloc[1] = 1  # Transition from non-bearish to bearish at bar 2.
    date = bt.price_data.index[2]
    bt._execute_legacy_trades(2, date, 110.0)
    assert bt.current_position == 3
    assert bt.trades[-1]['reason'] == 'background color change (partial)'


@pytest.mark.parametrize('qty', [1, 5, 10])
@pytest.mark.parametrize('outcome', ['stop', 'trend_break', 'ma_cross', 'final'])
def test_04_after_stage_one_the_remainder_is_fully_liquidated(outcome, qty):
    bt = make_backtest(two_stage_exit=True, stage2_exit_mode='ma_cross' if outcome == 'ma_cross' else 'trend_break')
    seed_lots(bt, qty)
    peak_date = bt.price_data.index[1]
    last_date = bt.price_data.index[-1]
    bt._tv_peak_signals = {peak_date: (peak_date, 0.8)}
    if outcome == 'stop':
        bt.price_data.loc[last_date, ['adjusted_close', 'adjusted_open', 'adjusted_low']] = [85.0, 84.0, 80.0]
    elif outcome == 'trend_break':
        bt.long_ma_trend.loc[last_date] = -1
    elif outcome == 'ma_cross':
        bt.short_ma_line.loc[last_date] = 0.4
    bt.execute_trades()
    assert_liquidated(bt, qty, final_equity=True)
    sells = [t for t in bt.trades if t['action'] == 'SELL']
    if qty == 1:
        assert len(sells) == 1
        assert sells[0]['reason'] == 'peak exit (stage 1, full)'
    else:
        assert len(sells) == 2
        assert sells[0]['shares'] == qty // 2
        assert sells[0]['reason'] == 'peak exit (stage 1)'
        assert sells[1]['shares'] == qty - qty // 2
        expected = {'stop': 'stop loss', 'final': 'backtest_end'}.get(outcome, 'trend break exit (stage 2)')
        assert sells[1]['reason'] == expected
        assert sells[1]['date'] == last_date


@pytest.mark.parametrize('qty', [1, 5, 10])
@pytest.mark.parametrize('pending_on_last_bar', [False, True])
def test_05_weekly_exit_preserves_next_open_or_final_close(monkeypatch, pending_on_last_bar, qty):
    bt = make_backtest(enable_weekly_trailing=True)
    seed_lots(bt, qty)
    dates = bt.price_data.index
    signal_date = dates[-1] if pending_on_last_bar else dates[1]  # Friday -> Monday open
    bt.price_data.loc[dates[2], 'adjusted_open'] = 120.0
    bt.price_data.loc[dates[-1], 'adjusted_close'] = 130.0
    # Inject the signal only; exercise the real weekly scheduler, fills and FIFO.
    monkeypatch.setattr(
        'backtest.backtest.check_weekly_trailing_stop', lambda **kwargs: kwargs['current_date'] == signal_date
    )
    bt.execute_trades()
    assert_liquidated(bt, qty, final_equity=True)
    sell = bt.trades[-1]
    assert sell['reason'] == 'weekly trailing'
    assert sell['date'] == (dates[-1] if pending_on_last_bar else dates[2])
    assert sell['price'] == pytest.approx((130.0 if pending_on_last_bar else 120.0) * (1 - bt.slippage))
