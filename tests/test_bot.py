"""Unit tests for indicators, risk sizing, the paper broker and the backtester.

These run fully offline (no ccxt / network) on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot import indicators as ind
from bot.backtest import run_backtest
from bot.broker import PaperBroker
from bot.config import Config, RiskConfig
from bot.risk import RiskManager
from bot.strategy import EmaRsiStrategy, build_strategy


def _synthetic_df(n=400, seed=0):
    """Trending + noisy price series with OHLC derived from close."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.01, n)  # slight upward drift
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": (ts.astype("int64") // 10**6),
            "open": open_,
            "high": np.maximum.reduce([high, close, open_]),
            "low": np.minimum.reduce([low, close, open_]),
            "close": close,
            "volume": rng.uniform(1, 100, n),
            "datetime": ts,
        }
    )


# ---- indicators ----------------------------------------------------------

def test_ema_matches_pandas():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    expected = s.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(ind.ema(s, 3), expected)


def test_rsi_bounds():
    s = pd.Series(np.linspace(1, 100, 100))
    r = ind.rsi(s, 14)
    assert (r >= 0).all() and (r <= 100).all()
    # Monotonic increase -> RSI should be very high.
    assert r.iloc[-1] > 90


def test_atr_positive():
    df = _synthetic_df(120)
    a = ind.atr(df["high"], df["low"], df["close"], 14)
    assert (a.dropna() >= 0).all()


# ---- risk ----------------------------------------------------------------

def test_position_sizing_respects_risk():
    rc = RiskConfig(risk_per_trade=0.01, stop_atr_mult=2.0, fee_rate=0.0)
    rm = RiskManager(rc, 1000)
    plan = rm.plan_trade(equity=1000, price=100, atr=1.0)
    # stop distance = 2.0; risk amount = 10; size = 10 / 2 = 5
    assert abs(plan.size - 5.0) < 1e-9
    assert abs(plan.stop_price - 98.0) < 1e-9
    assert abs(plan.take_price - 104.0) < 1e-9  # 2 * stop_dist * RR(2) above


def test_position_sizing_caps_at_equity():
    rc = RiskConfig(risk_per_trade=0.1, stop_atr_mult=0.1, fee_rate=0.0)
    rm = RiskManager(rc, 1000)
    plan = rm.plan_trade(equity=1000, price=100, atr=1.0)
    # Uncapped size would be huge; spot cap = equity/price = 10.
    assert plan.size <= 10.0 + 1e-9


def test_kill_switch_trips():
    rc = RiskConfig(max_drawdown=0.2)
    rm = RiskManager(rc, 1000)
    assert not rm.check_kill_switch(900)   # 10% dd, ok
    assert rm.check_kill_switch(750)       # 25% dd -> halt
    assert rm.halted


# ---- paper broker --------------------------------------------------------

def test_paper_broker_roundtrip_pnl():
    rc = RiskConfig(fee_rate=0.0, slippage=0.0)
    b = PaperBroker(1000, rc)
    b.buy(price=100, size=5, stop=90, take=120)
    assert b.position is not None
    assert abs(b.cash - 500) < 1e-9
    b.sell(price=110)
    assert b.position is None
    assert abs(b.realized_pnl - 50) < 1e-9   # (110-100)*5
    assert abs(b.equity(110) - 1050) < 1e-9


def test_paper_broker_applies_fees():
    rc = RiskConfig(fee_rate=0.001, slippage=0.0)
    b = PaperBroker(1000, rc)
    b.buy(price=100, size=1, stop=90, take=120)
    # cost 100 + fee 0.1 -> cash 899.9
    assert abs(b.cash - 899.9) < 1e-6


# ---- backtest end to end -------------------------------------------------

def test_backtest_runs_and_is_consistent():
    df = _synthetic_df(400, seed=42)
    cfg = Config(start_equity=200.0)
    strat = EmaRsiStrategy()
    res = run_backtest(df, strat, cfg)
    assert len(res.equity_curve) > 0
    assert res.final_equity > 0
    assert 0.0 <= res.win_rate <= 1.0
    assert 0.0 <= res.max_drawdown <= 1.0


def test_build_strategy_registry():
    assert isinstance(build_strategy("ema_rsi", {}), EmaRsiStrategy)
    try:
        build_strategy("does_not_exist", {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown strategy")
