"""Tests for metrics and the optimizer / walk-forward validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot import metrics
from bot.config import Config
from bot.optimize import expand_grid, grid_search, sharpe_scorer, walk_forward


def _synthetic_df(n=900, seed=7):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.012, n)
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


# ---- metrics -------------------------------------------------------------

def test_periods_per_year_known_and_default():
    assert metrics.periods_per_year("1h") == 8_760
    assert metrics.periods_per_year("1d") == 365
    assert metrics.periods_per_year("weird") == 8_760  # default


def test_profit_factor_and_expectancy():
    trades = [
        {"side": "buy", "price": 100},
        {"side": "sell", "price": 110, "pnl": 10.0},
        {"side": "sell", "price": 95, "pnl": -5.0},
        {"side": "sell", "price": 120, "pnl": 20.0},
    ]
    # gross win 30, gross loss 5 -> PF 6.0; expectancy (10-5+20)/3
    assert abs(metrics.profit_factor(trades) - 6.0) < 1e-9
    assert abs(metrics.expectancy(trades) - 25 / 3) < 1e-9


def test_profit_factor_no_losses_is_inf():
    trades = [{"side": "sell", "pnl": 5.0}]
    assert metrics.profit_factor(trades) == float("inf")


def test_sharpe_zero_on_flat_curve():
    curve = pd.Series([100.0] * 50)
    assert metrics.sharpe(curve, "1h") == 0.0


def test_sharpe_positive_on_rising_curve():
    curve = pd.Series(100 * (1.001 ** np.arange(200)))
    assert metrics.sharpe(curve, "1h") > 0


# ---- optimizer -----------------------------------------------------------

def test_expand_grid_cartesian():
    grid = {"a": [1, 2], "b": [3, 4, 5]}
    combos = expand_grid(grid)
    assert len(combos) == 6
    assert {"a": 1, "b": 3} in combos
    assert {"a": 2, "b": 5} in combos


def test_expand_grid_empty():
    assert expand_grid({}) == [{}]


def test_grid_search_returns_best_and_leaderboard():
    df = _synthetic_df()
    cfg = Config(start_equity=200.0)
    grid = {"ema_fast": [10, 20], "ema_slow": [50, 100]}
    opt = grid_search(df, "ema_rsi", grid, cfg, sharpe_scorer(min_trades=1))
    assert opt.best_params in expand_grid(grid)
    assert len(opt.leaderboard) == 4
    # Leaderboard sorted descending.
    scores = [s for _, s in opt.leaderboard]
    assert scores == sorted(scores, reverse=True)


def test_walk_forward_runs_out_of_sample():
    df = _synthetic_df(900)
    cfg = Config(start_equity=200.0)
    grid = {"ema_fast": [10, 20], "ema_slow": [50, 100]}
    wf = walk_forward(df, "ema_rsi", grid, cfg, n_splits=3, scorer=sharpe_scorer(min_trades=1))
    assert len(wf.windows) >= 1
    assert wf.final_equity > 0
    # Each window records the params chosen and an OOS return.
    for w in wf.windows:
        assert "params" in w and "test_return" in w
