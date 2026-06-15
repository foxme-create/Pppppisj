"""Strategy interface and built-in strategies.

Each strategy is *vectorized*: `signals(df)` computes, for every bar, the
desired action (+1 long, -1 flat/exit, 0 hold) and the ATR at that bar, all in
one pass. This is both fast and free of look-ahead, because every indicator we
use is causal (value at bar i depends only on bars <= i). `compute(df)` returns
the signal for the most recent bar and is what live trading uses.

The framework trades long-only on spot (you can't go negative on a spot
balance), so FLAT means "close the long if open".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    """Base class. Subclasses implement `signals`."""

    min_bars: int = 50

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like `df` with integer column 'action' and
        float column 'atr'."""
        raise NotImplementedError

    def compute(self, df: pd.DataFrame) -> Signal:
        """Signal for the most recent bar (used by live/paper trading)."""
        sig = self.signals(df)
        return Signal(
            action=int(sig["action"].iloc[-1]),
            price=float(df["close"].iloc[-1]),
            atr=float(sig["atr"].iloc[-1]),
        )

    @staticmethod
    def _frame(index, action, atr) -> pd.DataFrame:
        return pd.DataFrame({"action": np.asarray(action, dtype=int), "atr": atr}, index=index)


class EmaRsiStrategy(Strategy):
    """Trend-following with a momentum filter.

    Long when fast EMA is above slow EMA (uptrend) and RSI sits in a healthy
    band. Exit when the trend is lost or RSI becomes overbought.
    """

    def __init__(self, ema_fast=20, ema_slow=50, rsi_period=14,
                 rsi_max=70.0, rsi_min=50.0, atr_period=14):
        self.ema_fast, self.ema_slow = ema_fast, ema_slow
        self.rsi_period, self.rsi_max, self.rsi_min = rsi_period, rsi_max, rsi_min
        self.atr_period = atr_period
        self.min_bars = max(ema_slow, rsi_period, atr_period) + 2

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        ema_f = ind.ema(close, self.ema_fast)
        ema_s = ind.ema(close, self.ema_slow)
        rsi = ind.rsi(close, self.rsi_period)
        atr = ind.atr(df["high"], df["low"], close, self.atr_period)

        uptrend = ema_f > ema_s
        rsi_ok = (rsi >= self.rsi_min) & (rsi < self.rsi_max)
        exit_cond = (~uptrend) | (rsi >= self.rsi_max)
        action = np.where(uptrend & rsi_ok, LONG, np.where(exit_cond, FLAT, HOLD))
        return self._frame(df.index, action, atr)


class MeanReversionStrategy(Strategy):
    """Buy oversold RSI dips inside a longer-term uptrend; exit on reversion."""

    def __init__(self, sma_trend=100, rsi_period=14, rsi_buy=30.0, rsi_exit=55.0, atr_period=14):
        self.sma_trend, self.rsi_period = sma_trend, rsi_period
        self.rsi_buy, self.rsi_exit = rsi_buy, rsi_exit
        self.atr_period = atr_period
        self.min_bars = max(sma_trend, rsi_period, atr_period) + 2

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        trend = ind.sma(close, self.sma_trend)
        rsi = ind.rsi(close, self.rsi_period)
        atr = ind.atr(df["high"], df["low"], close, self.atr_period)

        in_uptrend = close > trend
        action = np.where(in_uptrend & (rsi <= self.rsi_buy), LONG,
                          np.where(rsi >= self.rsi_exit, FLAT, HOLD))
        return self._frame(df.index, action, atr)


class DonchianBreakoutStrategy(Strategy):
    """Turtle-style breakout: enter on N-bar high break, exit on M-bar low break."""

    def __init__(self, entry=20, exit=10, atr_period=14):
        self.entry, self.exit, self.atr_period = entry, exit, atr_period
        self.min_bars = max(entry, exit, atr_period) + 2

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        _, upper_entry = ind.donchian(df["high"], df["low"], self.entry)
        lower_exit, _ = ind.donchian(df["high"], df["low"], self.exit)
        atr = ind.atr(df["high"], df["low"], df["close"], self.atr_period)
        close = df["close"]
        action = np.where(close > upper_entry, LONG,
                          np.where(close < lower_exit, FLAT, HOLD))
        return self._frame(df.index, action, atr)


class SupertrendStrategy(Strategy):
    """Long while the Supertrend direction is up."""

    def __init__(self, period=10, multiplier=3.0, atr_period=14):
        self.period, self.multiplier, self.atr_period = period, multiplier, atr_period
        self.min_bars = max(period, atr_period) + 2

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        _, direction = ind.supertrend(df["high"], df["low"], df["close"],
                                      self.period, self.multiplier)
        atr = ind.atr(df["high"], df["low"], df["close"], self.atr_period)
        action = np.where(direction == 1, LONG, FLAT)
        return self._frame(df.index, action, atr)


class MacdTrendStrategy(Strategy):
    """MACD histogram positive AND price above a long EMA -> long."""

    def __init__(self, fast=12, slow=26, signal=9, trend_ema=200, atr_period=14):
        self.fast, self.slow, self.signal = fast, slow, signal
        self.trend_ema, self.atr_period = trend_ema, atr_period
        self.min_bars = max(slow + signal, trend_ema, atr_period) + 2

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        _, _, hist = ind.macd(close, self.fast, self.slow, self.signal)
        trend = ind.ema(close, self.trend_ema)
        atr = ind.atr(df["high"], df["low"], close, self.atr_period)
        action = np.where((hist > 0) & (close > trend), LONG, FLAT)
        return self._frame(df.index, action, atr)


class BollingerReversionStrategy(Strategy):
    """Buy below the lower Bollinger band, exit at the middle band."""

    def __init__(self, period=20, num_std=2.0, atr_period=14):
        self.period, self.num_std, self.atr_period = period, num_std, atr_period
        self.min_bars = max(period, atr_period) + 2

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        lower, middle, _ = ind.bollinger(close, self.period, self.num_std)
        atr = ind.atr(df["high"], df["low"], close, self.atr_period)
        action = np.where(close < lower, LONG,
                          np.where(close >= middle, FLAT, HOLD))
        return self._frame(df.index, action, atr)


_REGISTRY = {
    "ema_rsi": EmaRsiStrategy,
    "mean_reversion": MeanReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "supertrend": SupertrendStrategy,
    "macd_trend": MacdTrendStrategy,
    "bollinger_reversion": BollingerReversionStrategy,
}

# Sensible default parameter grids for optimization / strategy selection.
_DEFAULT_GRIDS = {
    "ema_rsi": {"ema_fast": [10, 20, 30], "ema_slow": [50, 100, 150], "rsi_max": [65, 70, 75]},
    "mean_reversion": {"sma_trend": [50, 100, 150], "rsi_buy": [25, 30, 35], "rsi_exit": [50, 55, 60]},
    "donchian_breakout": {"entry": [20, 40, 55], "exit": [10, 20]},
    "supertrend": {"period": [7, 10, 14], "multiplier": [2.0, 3.0, 4.0]},
    "macd_trend": {"fast": [12], "slow": [26], "trend_ema": [100, 200]},
    "bollinger_reversion": {"period": [15, 20, 30], "num_std": [2.0, 2.5, 3.0]},
}


def available_strategies() -> list[str]:
    return list(_REGISTRY)


def default_grid(name: str) -> dict:
    """Default parameter grid for a strategy (used by the selector)."""
    return _DEFAULT_GRIDS.get(name, {})


def build_strategy(name: str, params: dict) -> Strategy:
    if name not in _REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; available: {list(_REGISTRY)}")
    return _REGISTRY[name](**params)
