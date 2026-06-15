"""Tests for the extended indicators, strategies and the auto-selector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot import indicators as ind
from bot.config import Config
from bot.selector import select_strategy
from bot.strategy import (
    FLAT,
    HOLD,
    LONG,
    available_strategies,
    build_strategy,
    default_grid,
)


def _df(n=600, seed=3, drift=0.0004):
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.012, n)
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts.astype("int64") // 10**6,
            "open": open_,
            "high": np.maximum.reduce([high, close, open_]),
            "low": np.minimum.reduce([low, close, open_]),
            "close": close,
            "volume": rng.uniform(1, 100, n),
            "datetime": ts,
        }
    )


# ---- indicators ----------------------------------------------------------

def test_macd_shapes_and_hist():
    df = _df(200)
    line, signal, hist = ind.macd(df["close"])
    assert len(line) == len(signal) == len(hist) == len(df)
    # histogram is line - signal by definition
    pd.testing.assert_series_equal(hist, line - signal, check_names=False)


def test_bollinger_ordering():
    df = _df(200)
    lower, middle, upper = ind.bollinger(df["close"], 20, 2.0)
    valid = middle.notna()
    assert (lower[valid] <= middle[valid]).all()
    assert (middle[valid] <= upper[valid]).all()


def test_donchian_no_lookahead():
    df = _df(100)
    lower, upper = ind.donchian(df["high"], df["low"], 20)
    # Upper at bar i is the max high of the PREVIOUS 20 bars (shifted), so the
    # current bar's high must not define its own breakout level.
    for i in range(25, len(df)):
        prev_max = df["high"].iloc[i - 20 : i].max()
        assert abs(upper.iloc[i] - prev_max) < 1e-9


def test_supertrend_direction_values():
    df = _df(300)
    trend, direction = ind.supertrend(df["high"], df["low"], df["close"])
    assert set(direction.unique()).issubset({-1, 1})
    assert len(trend) == len(df)


def test_adx_non_negative():
    df = _df(300)
    a = ind.adx(df["high"], df["low"], df["close"], 14)
    assert (a >= 0).all()
    assert (a <= 100).all()


# ---- strategies ----------------------------------------------------------

def test_all_strategies_build_and_emit_valid_signal():
    df = _df(700)
    for name in available_strategies():
        strat = build_strategy(name, {})
        sig = strat.compute(df)
        assert sig.action in (LONG, FLAT, HOLD)
        assert sig.price > 0
        assert sig.atr >= 0


def test_default_grids_exist_for_all():
    for name in available_strategies():
        grid = default_grid(name)
        assert isinstance(grid, dict)
        assert len(grid) >= 1


def test_donchian_breakout_signal_logic():
    # Construct a clean breakout: flat then a jump above the channel.
    n = 60
    close = np.concatenate([np.full(50, 100.0), np.linspace(101, 130, 10)])
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        }
    )
    strat = build_strategy("donchian_breakout", {"entry": 20, "exit": 10})
    assert strat.compute(df).action == LONG


# ---- selector ------------------------------------------------------------

def test_selector_ranks_and_reports():
    df = _df(1000, seed=11, drift=0.0006)
    cfg = Config(start_equity=200.0)
    result = select_strategy(df, cfg, n_splits=3, min_trades=1)
    assert len(result.ranked) >= 1
    # Ranking is stable: tradeable ones come before non-tradeable.
    flags = [e.tradeable for e in result.ranked]
    assert flags == sorted(flags, reverse=True)
    # Summary renders without error and mentions a verdict.
    assert "VERDICT" in result.summary()


def test_selector_best_is_tradeable_or_none():
    df = _df(1000, seed=5)
    cfg = Config(start_equity=200.0)
    result = select_strategy(df, cfg, n_splits=3, min_trades=1)
    best = result.best
    assert best is None or best.tradeable
