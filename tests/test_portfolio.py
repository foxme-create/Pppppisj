"""Tests for the multi-symbol portfolio backtester and plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot.backtest import BacktestResult
from bot.config import Config, RiskConfig
from bot.portfolio import int_or_nan, run_portfolio_backtest
from bot.strategy import build_strategy


def _df(n=500, seed=0, drift=0.0005):
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


def test_int_or_nan():
    assert int_or_nan(1) == 1
    assert int_or_nan(-1) == -1
    assert int_or_nan(float("nan")) == 0
    assert int_or_nan(None) == 0


def test_portfolio_runs_and_returns_backtestresult():
    data = {"A/USDT": _df(seed=1), "B/USDT": _df(seed=2), "C/USDT": _df(seed=3)}
    cfg = Config(start_equity=200.0)
    strat = build_strategy("ema_rsi", {})
    res = run_portfolio_backtest(data, strat, cfg)
    assert isinstance(res, BacktestResult)
    assert len(res.equity_curve) > 0
    assert res.final_equity > 0
    assert 0.0 <= res.win_rate <= 1.0


def test_portfolio_respects_max_positions():
    # With max_positions=1, never more than one position's worth of cash should
    # be committed -> cash never goes negative and final equity is finite.
    data = {f"S{i}/USDT": _df(seed=i, drift=0.001) for i in range(4)}
    cfg = Config(start_equity=200.0, risk=RiskConfig(max_positions=1))
    strat = build_strategy("supertrend", {})
    res = run_portfolio_backtest(data, strat, cfg)
    assert res.final_equity > 0
    # Equity should never be NaN/negative.
    assert (res.equity_curve > 0).all()


def test_portfolio_never_overspends_cash():
    # Even with many symbols all wanting to buy, shared-cash accounting must
    # keep equity positive (no buying on credit).
    data = {f"S{i}/USDT": _df(seed=10 + i, drift=0.002) for i in range(5)}
    cfg = Config(start_equity=100.0, risk=RiskConfig(max_positions=5))
    strat = build_strategy("supertrend", {})
    res = run_portfolio_backtest(data, strat, cfg)
    assert (res.equity_curve > 0).all()


def test_portfolio_empty_raises():
    cfg = Config(start_equity=200.0)
    strat = build_strategy("ema_rsi", {})
    try:
        run_portfolio_backtest({}, strat, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty portfolio")


def test_plot_equity_writes_png(tmp_path):
    from bot.plotting import plot_equity
    curve = pd.Series(
        100 * (1.001 ** np.arange(200)),
        index=pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
    )
    out = str(tmp_path / "equity.png")
    plot_equity(curve, out, title="test")
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_equity_empty_raises(tmp_path):
    from bot.plotting import plot_equity
    try:
        plot_equity(pd.Series(dtype=float), str(tmp_path / "x.png"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty curve")
