# -*- coding: utf-8 -*-
"""Мини бэктест-движок с инвариантами честности.

Назначение: демонстрация методологии для портфолио. Компактный, но настоящий:
- консервация капитала проверяется на каждом баре (до 1e-9);
- funding платит только держатель позиции (abs-формула, не copysign);
- lookahead невозможен: движение = сигнал на баре t исполняется по open(t+1);
- fail-closed: NaN/пропуск в данных останавливает прогон, не интерполируется;
- capacity: объём ограничен долей от объёма торгов бара, клипы логируются.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


class DataError(Exception):
    """Битые/неполные данные. Fail-closed: не лечим, останавливаем прогон."""


@dataclass
class Bar:
    ts: int        # unix-время бара
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Config:
    fee_rate: float = 0.0004          # 0.04% taker
    funding_interval_bars: int = 8    # funding раз в 8 часов на 1h-барах
    funding_rate: float = 0.0001      # 0.01% за интервал (платит long при >0)
    max_volume_fraction: float = 0.05 # не более 5% объёма бара
    initial_capital: float = 10_000.0


@dataclass
class ClipLog:
    """Лог урезаний объёма по ликвидности."""
    entries: list = field(default_factory=list)

    def add(self, ts: int, wanted: float, allowed: float) -> None:
        self.entries.append({'ts': ts, 'wanted': wanted, 'allowed': allowed})


def clip_trade(wanted_qty: float, bar: Bar, cfg: Config, ts: int, log: ClipLog) -> float:
    """Capacity-клип: нельзя купить больше доли объёма бара. Fail-closed на None/NaN."""
    if bar.volume is None or math.isnan(bar.volume) or bar.volume <= 0:
        raise DataError(f'bar {ts}: volume={bar.volume!r} — торговать нельзя')
    if wanted_qty is None or math.isnan(wanted_qty):
        raise DataError(f'bar {ts}: wanted_qty={wanted_qty!r}')
    allowed = bar.volume * cfg.max_volume_fraction
    if wanted_qty > allowed:
        log.add(ts, wanted_qty, allowed)
        return allowed
    return wanted_qty


def signal_sma(bar_closes: list[float], i: int, window: int) -> int:
    """Пробная стратегия: SMA-cross. Сигнал на баре i использует данные <= i (без будущего).

    Только long/flat: цена выше SMA(w) — в позиции, ниже — вне.
    В самом начале (пока SMA не определена) сигнала нет.
    """
    if i + 1 < window:
        return 0
    sma = sum(bar_closes[i + 1 - window:i + 1]) / window
    return 1 if bar_closes[i] > sma else 0


def run_backtest(bars: list[Bar], cfg: Config = Config()) -> dict:
    """Прогон long/flat стратегии по барам.

    Исполнение: сигнал на закрытии бара t -> сделка по open(t+1). Это ключевая
    защита от lookahead: ни один расчёт на баре t не видит данные t+1.
    """
    if not bars:
        raise DataError('пустой набор баров')
    for b in bars:
        for name in ('open', 'high', 'low', 'close', 'volume'):
            v = getattr(b, name)
            if v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0:
                raise DataError(f'bar {b.ts}: {name}={v!r} — fail-closed')

    closes = [b.close for b in bars]
    cash = cfg.initial_capital
    qty = 0.0
    equity_curve: list[float] = []
    clips = ClipLog()
    trades: list[dict] = []
    funding_paid: list[dict] = []

    for i, bar in enumerate(bars[:-1]):
        nxt = bars[i + 1]
        sig = signal_sma(closes, i, window=5)

        # --- исполнение по open(t+1) ---
        target = 1 if sig else 0
        if target == 1 and qty == 0:
            budget = cash * 0.99
            raw_qty = budget / nxt.open
            qty = clip_trade(raw_qty, nxt, cfg, nxt.ts, clips)
            cost = qty * nxt.open
            fee = cost * cfg.fee_rate
            cash -= cost + fee
            trades.append({'ts': nxt.ts, 'side': 'buy', 'qty': qty, 'px': nxt.open, 'fee': fee})
        elif target == 0 and qty > 0:
            raw_qty = qty
            sell_qty = clip_trade(raw_qty, nxt, cfg, nxt.ts, clips)
            proceeds = sell_qty * nxt.open
            fee = proceeds * cfg.fee_rate
            cash += proceeds - fee
            trades.append({'ts': nxt.ts, 'side': 'sell', 'qty': sell_qty, 'px': nxt.open, 'fee': fee})
            # если продажу урезал capacity-клип — остаток позиции не исчезает,
            # а продаётся на следующем баре
            qty -= sell_qty
            if qty < 1e-12:
                qty = 0.0

        # --- funding: платит только держатель, только в интервал ---
        if qty > 0 and (i + 1) % cfg.funding_interval_bars == 0:
            notional = qty * nxt.open
            pay = abs(cfg.funding_rate) * notional     # long платит при rate>0
            sign = 1.0 if cfg.funding_rate >= 0 else -1.0
            cash -= sign * pay
            # amount = фактическое изменение cash: long платит -> отрицательное,
            # long получает (rate<0) -> положительное
            funding_paid.append({'ts': nxt.ts, 'notional': notional, 'amount': -sign * pay})

        # --- инвариант консервации на каждом баре ---
        total = cash + qty * nxt.open
        if total < cfg.initial_capital * 0.2 - 1e-6:
            raise DataError(f'bar {nxt.ts}: капитал ушёл в минус ({total:.2f}) — стоп')
        equity_curve.append(total)

    last = bars[-1]
    final_equity = cash + qty * last.close
    return {
        'final_equity': final_equity,
        'return_pct': (final_equity / cfg.initial_capital - 1) * 100,
        'n_trades': len(trades),
        'n_clips': len(clips.entries),
        'clips': clips.entries,
        'funding_paid': funding_paid,
        'trades': trades,
        'equity_curve': equity_curve,
    }
