"""Live / paper trading loop.

Polls the exchange each `poll_seconds`, runs the strategy on the latest closed
candles, and routes orders through the configured broker. Paper and live use
the same logic — only the broker differs.

State is persisted after every action so a restart resumes safely (it does not
forget an open position or its stop). In live mode the saved position is
reconciled against the exchange's real balance at startup.
"""

from __future__ import annotations

import logging
import time

from .broker import LiveBroker, PaperBroker
from .config import Config
from .data import fetch_ohlcv, make_exchange
from .notify import make_notifier
from .risk import RiskManager
from .state import load_state, save_state
from .strategy import FLAT, LONG, build_strategy

log = logging.getLogger("engine")


def _restore(cfg: Config, broker, risk: RiskManager, price: float):
    """Load persisted state and apply it to broker/risk. Returns last_bar_ts."""
    st = load_state(cfg.state_file)
    if st is None:
        return None
    if st.symbol != cfg.symbol or st.mode != cfg.mode:
        log.warning("Ignoring state file: symbol/mode mismatch (%s/%s vs %s/%s)",
                    st.symbol, st.mode, cfg.symbol, cfg.mode)
        return None

    risk.peak_equity = max(risk.peak_equity, st.peak_equity)
    saved_pos = st.to_position()

    if isinstance(broker, LiveBroker):
        # Exchange is the source of truth. Trust the saved stop/take, but only
        # if the exchange actually still holds the base asset.
        broker.position = saved_pos
        broker.reconcile(price)
    else:
        # Paper: restore the *full* accounting state. Otherwise a fresh broker
        # would hold full starting cash AND the restored position, double-
        # counting equity.
        broker.position = saved_pos
        if st.cash is not None:
            broker.cash = st.cash
        if st.realized_pnl is not None:
            broker.realized_pnl = st.realized_pnl

    if broker.position:
        log.info("Restored open position: size=%.6f entry=%.2f stop=%.2f take=%.2f",
                 broker.position.size, broker.position.entry_price,
                 broker.position.stop_price, broker.position.take_price)
    return st.last_bar_ts


def run_live(cfg: Config) -> None:
    cfg.validate()
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
    notifier = make_notifier(cfg.telegram_token, cfg.telegram_chat_id)

    exchange = make_exchange(cfg.exchange, cfg.testnet, cfg.api_key, cfg.api_secret)
    quote_ccy = cfg.symbol.split("/")[1]

    # Seed an initial price for equity/restore.
    first = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, limit=strategy.min_bars + 2)
    seed_price = float(first["close"].iloc[-1])

    if cfg.mode == "paper":
        broker = PaperBroker(cfg.start_equity, cfg.risk)
        start_eq = cfg.start_equity
    else:  # live
        broker = LiveBroker(exchange, cfg.symbol, cfg.risk, quote_ccy)
        start_eq = broker.equity(seed_price)

    risk = RiskManager(cfg.risk, start_eq)
    last_bar_ts = _restore(cfg, broker, risk, seed_price)

    msg = (f"🤖 Bot started: {cfg.mode} {cfg.symbol} {cfg.timeframe} on {cfg.exchange}\n"
           f"Strategy: {cfg.strategy.name} | equity≈{broker.equity(seed_price):.2f} {quote_ccy}")
    log.info(msg.replace("\n", " | "))
    notifier.send(msg)

    def persist():
        cash = getattr(broker, "cash", None)
        realized = getattr(broker, "realized_pnl", None)
        save_state(cfg.state_file, mode=cfg.mode, symbol=cfg.symbol,
                   peak_equity=risk.peak_equity, last_bar_ts=last_bar_ts,
                   position=broker.position, cash=cash, realized_pnl=realized)

    while True:
        try:
            df = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, limit=strategy.min_bars + 6)
            # The exchange's last candle is the *forming* (incomplete) one. Act
            # on the last CLOSED candle to avoid signals that repaint before the
            # bar closes. Use the latest price for execution and stop checks.
            closed = df.iloc[:-1]
            bar_ts = int(closed["timestamp"].iloc[-1])
            price = float(df["close"].iloc[-1])

            # Manage open position stops/takes every poll (not just on new bars).
            if broker.position:
                pos = broker.position
                if price <= pos.stop_price or price >= pos.take_price:
                    broker.sell(price)
                    persist()
                    note = f"🔵 Closed {cfg.symbol} at {price:.2f} (stop/take hit)"
                    log.info(note)
                    notifier.send(note)

            equity = broker.equity(price)
            if risk.check_kill_switch(equity):
                broker.sell(price)
                persist()
                note = (f"🛑 KILL-SWITCH: drawdown limit hit. Flattened and halting. "
                        f"Equity {equity:.2f}, peak {risk.peak_equity:.2f}")
                log.warning(note)
                notifier.send(note)
                break

            if bar_ts != last_bar_ts:
                last_bar_ts = bar_ts
                signal = strategy.compute(closed)
                log.info("Signal=%s price=%.2f equity=%.2f", signal.action, price, equity)
                if signal.action == FLAT and broker.position:
                    broker.sell(price)
                    note = f"🔵 Exit signal -> closed {cfg.symbol} at {price:.2f}"
                    log.info(note)
                    notifier.send(note)
                elif signal.action == LONG and not broker.position:
                    plan = risk.plan_trade(equity, price, signal.atr)
                    if plan.size > 0:
                        broker.buy(price, plan.size, plan.stop_price, plan.take_price)
                        note = (f"🟢 Entered long {cfg.symbol} size={plan.size:.6f} "
                                f"@~{price:.2f} stop={plan.stop_price:.2f} "
                                f"take={plan.take_price:.2f}")
                        log.info(note)
                        notifier.send(note)
                persist()

        except Exception as exc:  # keep the loop alive on transient errors
            log.exception("Loop error: %s", exc)
            notifier.send(f"⚠️ Loop error: {type(exc).__name__}: {exc}")

        time.sleep(cfg.poll_seconds)
