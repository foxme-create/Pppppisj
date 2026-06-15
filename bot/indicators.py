"""Technical indicators implemented on top of pandas/numpy.

Kept dependency-free (no TA-Lib / pandas-ta) so the project installs cleanly
everywhere. All functions take and return pandas Series aligned to the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing is an EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100 - (100 / (1 + rs))
    # When avg_loss is 0 the asset only went up -> RSI 100.
    return out.fillna(100.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — volatility measure used for stop sizing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Per-bar True Range (used by ATR/Supertrend/ADX)."""
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: (lower, middle, upper)."""
    middle = sma(series, period)
    std = series.rolling(window=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return lower, middle, upper


def donchian(
    high: pd.Series, low: pd.Series, period: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Donchian channel: (lower, upper) over the trailing `period` bars,
    shifted by one so the current bar's own extreme doesn't leak into the
    breakout level it is tested against."""
    upper = high.rolling(window=period).max().shift(1)
    lower = low.rolling(window=period).min().shift(1)
    return lower, upper


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Supertrend indicator.

    Returns (trend_line, direction) where direction is +1 (uptrend) or -1
    (downtrend). Computed with the standard iterative final-band carry-over.
    """
    atr_ = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr_
    lower = hl2 - multiplier * atr_

    n = len(close)
    fu = upper.to_numpy(dtype=float, copy=True)
    fl = lower.to_numpy(dtype=float, copy=True)
    u = upper.to_numpy(dtype=float)
    lo = lower.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    for i in range(1, n):
        fu[i] = u[i] if (u[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lo[i] if (lo[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]

    dir_arr = np.ones(n, dtype=int)
    trend_arr = fl.copy()
    for i in range(1, n):
        if c[i] > fu[i]:
            dir_arr[i] = 1
        elif c[i] < fl[i]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = dir_arr[i - 1]
        trend_arr[i] = fl[i] if dir_arr[i] == 1 else fu[i]

    return pd.Series(trend_arr, index=close.index), pd.Series(dir_arr, index=close.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — trend *strength* (not direction). High ADX
    (>~25) = strong trend; low ADX = chop. Useful as a regime filter."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, pd.NA)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, pd.NA)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)
