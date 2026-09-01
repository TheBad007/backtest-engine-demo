# -*- coding: utf-8 -*-
"""Тесты-инварианты бэктест-движка.

Каждый тест — это одна из проверок «честности» из портфолио. Тесты запускаются
на синтетических данных, без внешних API.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Bar, Config, DataError, run_backtest  # noqa: E402


def flat_bars(n=30, price=100.0):
    return [Bar(ts=i * 3600, open=price, high=price, low=price, close=price, volume=1000.0)
            for i in range(n)]


def sine_bars(n=60):
    bars = []
    for i in range(n):
        px = 100 + 10 * math.sin(i / 3.0)
        bars.append(Bar(ts=i * 3600, open=px, high=px * 1.01, low=px * 0.99,
                        close=px, volume=1000.0))
    return bars


# ---------- 1. инвариант консервации ----------

def test_conservation_flat_market():
    """На плоском рынке без движения цены капитал не меняется от комиссий не-торговли."""
    res = run_backtest(flat_bars(30, price=100.0))
    # сделок быть не должно: цена равна SMA
    assert res['n_trades'] == 0
    assert res['final_equity'] == pytest.approx(10_000.0, abs=1e-9)


def hump_bars():
    """Разгон 100→111, спад 111→100, плато. Гарантированный цикл buy→sell,
    без открытой позиции в конце."""
    px = [100.0 + i for i in range(12)]
    px += [111.0 - i for i in range(1, 12)]
    px += [100.0] * 6
    return [Bar(ts=i * 3600, open=p, high=p, low=p, close=p, volume=1000.0)
            for i, p in enumerate(px)]


def test_conservation_after_roundtrip():
    """Полный цикл buy→sell: финал = старт − покупка + продажа ± funding,
    пересчитанный из лога сделок независимо от внутренней equity-кривой."""
    res = run_backtest(hump_bars())
    assert res['n_trades'] >= 2, 'должны быть и покупка, и продажа'
    cash = 10_000.0
    for tr in res['trades']:
        if tr['side'] == 'buy':
            cash -= tr['qty'] * tr['px'] + tr['fee']
        else:
            cash += tr['qty'] * tr['px'] - tr['fee']
    for f in res['funding_paid']:
        cash += f['amount']
    assert res['final_equity'] == pytest.approx(cash, abs=1e-6)


# ---------- 2. funding: платит только держатель ----------

def test_funding_only_when_holding():
    """Без позиции funding не платится вообще."""
    res = run_backtest(flat_bars(30))
    assert res['funding_paid'] == []


def test_funding_sign_long_pays_when_positive():
    """Long при положительном rate платит (abs-формула, не copysign)."""
    cfg = Config(funding_rate=0.0001, funding_interval_bars=1)
    # всегда в позиции: buy на баре 1, держим до конца
    bars = [Bar(ts=i * 3600, open=100.0 + i, high=100.0 + i, low=100.0 + i,
                close=100.0 + i, volume=1000.0) for i in range(10)]
    res = run_backtest(bars, cfg)
    assert res['funding_paid'], 'должны быть funding-платежи'
    for f in res['funding_paid']:
        assert f['amount'] < 0, 'long платит: amount отрицательный'
        expected = 0.0001 * f['notional']
        assert abs(f['amount'] + expected) < 1e-12


def test_funding_negative_rate_long_receives():
    """При отрицательном rate long ПОЛУЧАЕТ (проверка знака на противоположном случае)."""
    cfg = Config(funding_rate=-0.0001, funding_interval_bars=1)
    bars = [Bar(ts=i * 3600, open=100.0 + i, high=100.0 + i, low=100.0 + i,
                close=100.0 + i, volume=1000.0) for i in range(10)]
    res = run_backtest(bars, cfg)
    assert res['funding_paid']
    for f in res['funding_paid']:
        assert f['amount'] > 0, 'long получает при отрицательном rate'


# ---------- 3. lookahead ----------

def test_no_lookahead_shifted_data_degrades():
    """Если сдвинуть цены на 1 бар вперёд, результат обязан измениться
    (движок не может «видеть» будущее, иначе сдвиг был бы бесплатным)."""
    bars = sine_bars(60)
    res_a = run_backtest(bars)
    shifted = [Bar(ts=b.ts, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
               for b in bars[1:]]
    res_b = run_backtest(shifted)
    assert res_a['return_pct'] != res_b['return_pct']


def test_signal_at_t_executes_at_t_plus_1():
    """Сделка исполняется по open(t+1): растущий рынок -> сигнал на баре 4
    (close 104 > SMA5 102), сделка на баре 5 по open=105."""
    bars = [Bar(ts=i * 3600, open=100.0 + i, high=100.0 + i, low=100.0 + i,
                close=100.0 + i, volume=1000.0) for i in range(10)]
    res = run_backtest(bars)
    assert res['trades'], 'должна быть сделка'
    first = res['trades'][0]
    assert first['side'] == 'buy'
    assert first['ts'] == 5 * 3600, first['ts']
    assert first['px'] == 105.0, f'buy должен быть по open(5)=105, получено {first["px"]}'


# ---------- 4. fail-closed ----------

def test_nan_close_fails_closed():
    bars = flat_bars(10)
    bars[5].close = math.nan
    with pytest.raises(DataError):
        run_backtest(bars)


def test_zero_volume_fails_closed():
    bars = flat_bars(10)
    bars[3].volume = 0.0
    with pytest.raises(DataError):
        run_backtest(bars)


def test_empty_bars_fails_closed():
    with pytest.raises(DataError):
        run_backtest([])


# ---------- 5. capacity ----------

def test_capacity_clip_limits_size():
    """Заявка больше 5% объёма бара урезается до лимита, урезание логируется.
    Растущий рынок: заявка ~94 шт при allowed = 50 обязана урезаться."""
    bars = [Bar(ts=i * 3600, open=100.0 + i, high=100.0 + i, low=100.0 + i,
                close=100.0 + i, volume=1000.0) for i in range(10)]
    res = run_backtest(bars)
    assert res['n_clips'] >= 1, 'заявка ~94 шт > allowed 50 — обязана урезаться'
    for c in res['clips']:
        assert c['wanted'] > c['allowed']
        assert c['allowed'] <= 1000.0 * Config().max_volume_fraction + 1e-9


def test_capacity_fail_closed_on_none_volume():
    bars = flat_bars(10)
    bars[2].volume = None
    with pytest.raises(DataError):
        run_backtest(bars)


# ---------- 6. независимый пересчёт ----------

def test_independent_pnl_recompute():
    """PnL пересчитывается вторым способом: весь cash ведём из лога сделок и
    funding, минуя equity-кривую движка. Синусоида даёт несколько циклов и
    открытую позицию в конце — она оценивается mark-to-market по последнему close."""
    bars = sine_bars(60)
    res = run_backtest(bars)
    assert res['n_trades'] >= 4, 'на синусоиде обязано быть несколько циклов'
    cash = 10_000.0
    pos_qty, pos_px = 0.0, 0.0
    for tr in res['trades']:
        if tr['side'] == 'buy':
            cash -= tr['qty'] * tr['px'] + tr['fee']
            pos_qty, pos_px = tr['qty'], tr['px']
        else:
            cash += tr['qty'] * tr['px'] - tr['fee']
            pos_qty -= tr['qty']
    for f in res['funding_paid']:
        cash += f['amount']
    if pos_qty > 1e-12:
        cash += pos_qty * bars[-1].close  # открытая позиция — mark-to-market
    assert res['final_equity'] == pytest.approx(cash, abs=1e-6)


# ---------- 7. prereg-стиль: жёсткие критерии ----------

def test_prereg_style_no_soft_pass():
    """Prereg-кейс: критерий зафиксирован до запуска — на этом рынке стратегия
    ОБЯЗАНА потерять, и движок не имеет права «смягчить» результат.
    Разгон вверх (вход long по правилу), затем обвал −6 за бар:
    long обязан зафиксировать убыток + комиссии + funding."""
    px = [100.0 + i for i in range(12)]             # разгон: вход по правилу
    px += [105.0 - 6.0 * k for k in range(1, 18)]   # обвал до 3
    bars = [Bar(ts=i * 3600, open=p, high=p, low=p, close=p, volume=1000.0)
            for i, p in enumerate(px)]
    cfg = Config(funding_rate=0.0001, funding_interval_bars=1)
    res = run_backtest(bars, cfg)
    assert res['trades'], 'должен быть вход'
    assert res['trades'][0]['side'] == 'buy'
    assert res['n_trades'] >= 2, 'должны быть вход и выход'
    assert res['final_equity'] < 10_000.0, 'long в обвале обязан терять'
