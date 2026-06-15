"""Strategy interface and built-in strategies.

A strategy looks at a DataFrame of candles and emits a signal for the *last*
(most recent) bar: +1 = want to be long, -1 = want to be flat/exit, 0 = hold.

This framework trades long-only on spot (you can't go negative on a spot
balance), so -1 means "close the long if open".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators as ind

LONG, FLAT, HOLD = 1, -1, 0


@dataclass
class Signal:
    action: int          # LONG | FLAT | HOLD
    price: float         # reference price (close of signal bar)
    atr: float           # ATR at signal bar, for stop sizing
    reason: str = ""


class Strategy:
    """Base class. Subclasses implement `compute`."""

    min_bars: int = 50

    def compute(self, df: pd.DataFrame) -> Signal:
        raise NotImplementedError


class EmaRsiStrategy(Strategy):
    """Trend-following with a momentum filter.

    Enter long when fast EMA is above slow EMA (uptrend) and RSI is rising but
    not overbought. Exit when fast EMA crosses back below slow EMA, or RSI gets
    extremely overbought.
    """

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_max: float = 70.0,
        rsi_min: float = 50.0,
        atr_period: int = 14,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_max = rsi_max
        self.rsi_min = rsi_min
        self.atr_period = atr_period
        self.min_bars = max(ema_slow, rsi_period, atr_period) + 2

    def compute(self, df: pd.DataFrame) -> Signal:
        close = df["close"]
        ema_f = ind.ema(close, self.ema_fast)
        ema_s = ind.ema(close, self.ema_slow)
        rsi = ind.rsi(close, self.rsi_period)
        atr = ind.atr(df["high"], df["low"], close, self.atr_period)

        price = float(close.iloc[-1])
        cur_atr = float(atr.iloc[-1])
        uptrend = ema_f.iloc[-1] > ema_s.iloc[-1]
        rsi_ok = self.rsi_min <= rsi.iloc[-1] < self.rsi_max

        if uptrend and rsi_ok:
            return Signal(LONG, price, cur_atr, "ema_fast>ema_slow & rsi in band")
        if (not uptrend) or rsi.iloc[-1] >= self.rsi_max:
            return Signal(FLAT, price, cur_atr, "trend lost or overbought")
        return Signal(HOLD, price, cur_atr, "no edge")


class MeanReversionStrategy(Strategy):
    """Buy oversold dips inside an overall uptrend, exit on reversion.

    Long when price is in a longer-term uptrend (above slow SMA) but RSI is
    oversold; exit when RSI normalises.
    """

    def __init__(
        self,
        sma_trend: int = 100,
        rsi_period: int = 14,
        rsi_buy: float = 30.0,
        rsi_exit: float = 55.0,
        atr_period: int = 14,
    ):
        self.sma_trend = sma_trend
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_exit = rsi_exit
        self.atr_period = atr_period
        self.min_bars = max(sma_trend, rsi_period, atr_period) + 2

    def compute(self, df: pd.DataFrame) -> Signal:
        close = df["close"]
        trend = ind.sma(close, self.sma_trend)
        rsi = ind.rsi(close, self.rsi_period)
        atr = ind.atr(df["high"], df["low"], close, self.atr_period)

        price = float(close.iloc[-1])
        cur_atr = float(atr.iloc[-1])
        in_uptrend = price > trend.iloc[-1]

        if in_uptrend and rsi.iloc[-1] <= self.rsi_buy:
            return Signal(LONG, price, cur_atr, "oversold dip in uptrend")
        if rsi.iloc[-1] >= self.rsi_exit:
            return Signal(FLAT, price, cur_atr, "reverted to mean")
        return Signal(HOLD, price, cur_atr, "no edge")


_REGISTRY = {
    "ema_rsi": EmaRsiStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def build_strategy(name: str, params: dict) -> Strategy:
    if name not in _REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; available: {list(_REGISTRY)}")
    return _REGISTRY[name](**params)
