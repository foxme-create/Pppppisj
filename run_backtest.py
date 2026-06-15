#!/usr/bin/env python3
"""Backtest a strategy on historical data. No API keys required."""

from __future__ import annotations

import argparse

from bot.backtest import run_backtest
from bot.config import load_config
from bot.data import fetch_ohlcv, make_exchange
from bot.strategy import build_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a trading strategy.")
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    parser.add_argument("--save-curve", default="", help="optional CSV path for equity curve")
    args = parser.parse_args()

    cfg = load_config(args.config)
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)

    print(f"Fetching {cfg.history_limit} x {cfg.timeframe} candles of {cfg.symbol} "
          f"from {cfg.exchange} ...")
    exchange = make_exchange(cfg.exchange)  # public data, no keys
    df = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, cfg.history_limit)
    print(f"Got {len(df)} candles "
          f"({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})\n")

    result = run_backtest(df, strategy, cfg)
    print(result.summary())

    # Buy-and-hold benchmark for context.
    bh = df["close"].iloc[-1] / df["close"].iloc[strategy.min_bars] - 1
    print(f"Buy & hold:     {bh * 100:.2f}%  (benchmark)")

    if args.save_curve:
        result.equity_curve.to_csv(args.save_curve, header=["equity"])
        print(f"\nEquity curve saved to {args.save_curve}")


if __name__ == "__main__":
    main()
