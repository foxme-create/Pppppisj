"""Live / paper trading loop.

Polls the exchange each `poll_seconds`, runs the strategy on the latest closed
candles, and routes orders through the configured broker. Paper and live use
the same logic — only the broker differs.
"""

from __future__ import annotations

import logging
import time

from .broker import LiveBroker, PaperBroker
from .config import Config
from .data import fetch_ohlcv, make_exchange
from .risk import RiskManager
from .strategy import FLAT, LONG, build_strategy

log = logging.getLogger("engine")


def run_live(cfg: Config) -> None:
    cfg.validate()
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)

    exchange = make_exchange(cfg.exchange, cfg.testnet, cfg.api_key, cfg.api_secret)
    quote_ccy = cfg.symbol.split("/")[1]

    if cfg.mode == "paper":
        broker = PaperBroker(cfg.start_equity, cfg.risk)
        start_eq = cfg.start_equity
    else:  # live
        broker = LiveBroker(exchange, cfg.symbol, cfg.risk, quote_ccy)
        # Seed equity from a first price read.
        first = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, limit=strategy.min_bars + 2)
        start_eq = broker.equity(float(first["close"].iloc[-1]))

    risk = RiskManager(cfg.risk, start_eq)
    log.info("Starting %s loop on %s %s (%s)", cfg.mode, cfg.symbol, cfg.timeframe, cfg.exchange)

    last_bar_ts = None
    while True:
        try:
            df = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, limit=strategy.min_bars + 5)
            # Only act on a newly *closed* candle to avoid acting on a partial bar.
            bar_ts = int(df["timestamp"].iloc[-1])
            price = float(df["close"].iloc[-1])

            # Manage open position stops/takes every poll (not just on new bars).
            if broker.position:
                pos = broker.position
                if price <= pos.stop_price or price >= pos.take_price:
                    broker.sell(price)
                    log.info("Closed position at %.2f (stop/take)", price)

            equity = broker.equity(price)
            if risk.check_kill_switch(equity):
                log.warning("Kill-switch tripped (drawdown). Flattening and halting.")
                broker.sell(price)
                break

            if bar_ts != last_bar_ts:
                last_bar_ts = bar_ts
                signal = strategy.compute(df)
                log.info("Signal=%s price=%.2f equity=%.2f reason=%s",
                         signal.action, price, equity, signal.reason)
                if signal.action == FLAT and broker.position:
                    broker.sell(price)
                    log.info("Exit signal -> closed at %.2f", price)
                elif signal.action == LONG and not broker.position:
                    plan = risk.plan_trade(equity, price, signal.atr)
                    if plan.size > 0:
                        broker.buy(price, plan.size, plan.stop_price, plan.take_price)
                        log.info("Entered long size=%.6f stop=%.2f take=%.2f",
                                 plan.size, plan.stop_price, plan.take_price)

        except Exception as exc:  # keep the loop alive on transient errors
            log.exception("Loop error: %s", exc)

        time.sleep(cfg.poll_seconds)
