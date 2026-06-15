#!/usr/bin/env python3
"""Backtest one strategy across several symbols from a shared cash pool.

Symbols come from `symbols:` in the config (falls back to the single `symbol`).
Capital is shared and capped at `risk.max_positions` concurrent positions.
"""

from __future__ import annotations

import argparse

from bot.config import load_config
from bot.data import fetch_ohlcv, make_exchange
from bot.portfolio import run_portfolio_backtest
from bot.strategy import build_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-symbol portfolio backtest.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--plot", default="", help="optional PNG path for the equity chart")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = cfg.symbols or [cfg.symbol]
    if len(symbols) < 1:
        raise SystemExit("Set `symbols:` (a list) in the config for a portfolio backtest.")

    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
    exchange = make_exchange(cfg.exchange)

    data = {}
    for sym in symbols:
        print(f"Fetching {cfg.history_limit} x {cfg.timeframe} of {sym} ...")
        data[sym] = fetch_ohlcv(exchange, sym, cfg.timeframe, cfg.history_limit)

    print(f"\nPortfolio: {symbols}  | max concurrent positions: {cfg.risk.max_positions}\n")
    result = run_portfolio_backtest(data, strategy, cfg)
    print(result.summary())

    if args.plot:
        from bot.plotting import plot_equity
        plot_equity(result.equity_curve, args.plot,
                    title=f"Portfolio: {cfg.strategy.name} on {', '.join(symbols)}")
        print(f"\nChart saved to {args.plot}")


if __name__ == "__main__":
    main()
