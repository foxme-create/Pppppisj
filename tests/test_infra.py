"""Tests for persistence, notifications, the live broker reconcile, and the
regime ensemble strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot import notify
from bot.broker import LiveBroker
from bot.config import RiskConfig
from bot.risk import Position
from bot.state import BotState, load_state, save_state
from bot.strategy import (
    LONG,
    FLAT,
    HOLD,
    available_strategies,
    build_strategy,
    default_grid,
)


def _df(n=600, seed=3):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.012, n)
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


# ---- persistence ---------------------------------------------------------

def test_state_roundtrip_with_position(tmp_path):
    path = str(tmp_path / "state.json")
    pos = Position(entry_price=100.0, size=2.0, stop_price=95.0, take_price=110.0)
    save_state(path, mode="paper", symbol="BTC/USDT", peak_equity=250.0,
               last_bar_ts=1234, position=pos)
    st = load_state(path)
    assert isinstance(st, BotState)
    assert st.mode == "paper"
    assert st.symbol == "BTC/USDT"
    assert st.peak_equity == 250.0
    assert st.last_bar_ts == 1234
    restored = st.to_position()
    assert restored.entry_price == 100.0 and restored.size == 2.0
    assert restored.stop_price == 95.0 and restored.take_price == 110.0


def test_state_roundtrip_no_position(tmp_path):
    path = str(tmp_path / "state.json")
    save_state(path, mode="live", symbol="ETH/USDT", peak_equity=200.0,
               last_bar_ts=None, position=None)
    st = load_state(path)
    assert st.position is None
    assert st.to_position() is None


def test_load_missing_state_returns_none(tmp_path):
    assert load_state(str(tmp_path / "nope.json")) is None


def test_state_persists_cash_and_pnl(tmp_path):
    from bot.broker import PaperBroker
    path = str(tmp_path / "state.json")
    rc = RiskConfig(fee_rate=0.0, slippage=0.0)
    b = PaperBroker(1000, rc)
    b.buy(price=100, size=5, stop=90, take=120)   # cash -> 500
    save_state(path, mode="paper", symbol="BTC/USDT", peak_equity=1000.0,
               last_bar_ts=1, position=b.position, cash=b.cash,
               realized_pnl=b.realized_pnl)
    st = load_state(path)
    assert st.cash == 500.0
    assert st.realized_pnl == 0.0


def test_paper_resume_does_not_double_count_equity(tmp_path):
    """Regression: a restored paper position must not be added on top of a full
    fresh cash balance."""
    from bot.broker import PaperBroker
    path = str(tmp_path / "state.json")
    rc = RiskConfig(fee_rate=0.0, slippage=0.0)

    # Session 1: buy, then persist.
    b1 = PaperBroker(1000, rc)
    b1.buy(price=100, size=5, stop=90, take=120)   # cash 500, holds 5 @ 100
    assert abs(b1.equity(100) - 1000) < 1e-9
    save_state(path, mode="paper", symbol="BTC/USDT", peak_equity=1000.0,
               last_bar_ts=1, position=b1.position, cash=b1.cash,
               realized_pnl=b1.realized_pnl)

    # Session 2: fresh broker + restore exactly like engine._restore does.
    b2 = PaperBroker(1000, rc)
    st = load_state(path)
    b2.position = st.to_position()
    b2.cash = st.cash
    b2.realized_pnl = st.realized_pnl
    # Equity must still be 1000 (500 cash + 5*100), NOT 1500.
    assert abs(b2.equity(100) - 1000) < 1e-9


def test_save_state_is_atomic_no_leftover_tmp(tmp_path):
    path = str(tmp_path / "state.json")
    save_state(path, mode="paper", symbol="BTC/USDT", peak_equity=1.0,
               last_bar_ts=1, position=None)
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ---- notifications -------------------------------------------------------

def test_make_notifier_noop_when_unconfigured():
    n = notify.make_notifier("", "")
    assert type(n) is notify.Notifier
    n.send("hi")  # must not raise


def test_make_notifier_telegram_when_configured():
    n = notify.make_notifier("token", "chat")
    assert isinstance(n, notify.TelegramNotifier)


def test_telegram_send_swallows_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
    notify.TelegramNotifier("t", "c").send("hello")  # must not raise


def test_telegram_send_posts_payload(monkeypatch):
    captured = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"{}"

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        return FakeResp()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    notify.TelegramNotifier("TOKEN", "CHAT").send("hi there")
    assert "TOKEN" in captured["url"]
    assert b"chat_id=CHAT" in captured["data"]


# ---- live broker reconcile ----------------------------------------------

class FakeExchange:
    def __init__(self, base_free):
        self._base_free = base_free

    def fetch_balance(self):
        return {"BTC": {"free": self._base_free}, "USDT": {"free": 1000.0}}


def test_reconcile_drops_stale_position_when_no_balance():
    b = LiveBroker(FakeExchange(0.0), "BTC/USDT", RiskConfig())
    b.position = Position(entry_price=100, size=1.0, stop_price=90, take_price=120)
    b.reconcile(price=100.0)
    assert b.position is None


def test_reconcile_keeps_position_when_balance_present():
    b = LiveBroker(FakeExchange(1.0), "BTC/USDT", RiskConfig())
    b.position = Position(entry_price=100, size=1.0, stop_price=90, take_price=120)
    b.reconcile(price=100.0)
    assert b.position is not None and b.position.size == 1.0


def test_reconcile_shrinks_to_held_amount():
    b = LiveBroker(FakeExchange(0.6), "BTC/USDT", RiskConfig())
    b.position = Position(entry_price=100, size=1.0, stop_price=90, take_price=120)
    b.reconcile(price=100.0)
    assert abs(b.position.size - 0.6) < 1e-9


# ---- regime ensemble -----------------------------------------------------

def test_regime_ensemble_registered():
    assert "regime_ensemble" in available_strategies()
    assert "adx_threshold" in default_grid("regime_ensemble")


def test_regime_ensemble_signals_valid():
    df = _df(700)
    strat = build_strategy("regime_ensemble", {})
    sig = strat.signals(df)
    assert set(np.unique(sig["action"])).issubset({LONG, FLAT, HOLD})
    assert len(sig) == len(df)
    assert (sig["atr"].dropna() >= 0).all()


def test_regime_ensemble_switches_subs():
    # With threshold 0 it should always use the trend sub; with a huge
    # threshold it should always use the range sub. Different thresholds must
    # be able to produce different actions.
    df = _df(700, seed=9)
    always_trend = build_strategy("regime_ensemble", {"adx_threshold": 0.0})
    always_range = build_strategy("regime_ensemble", {"adx_threshold": 1000.0})
    trend_only = build_strategy("supertrend", {})
    range_only = build_strategy("bollinger_reversion", {})
    assert (always_trend.signals(df)["action"].to_numpy()
            == trend_only.signals(df)["action"].to_numpy()).all()
    assert (always_range.signals(df)["action"].to_numpy()
            == range_only.signals(df)["action"].to_numpy()).all()
