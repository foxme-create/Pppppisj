#!/usr/bin/env python3
"""Automatically find the best strategy for a market, validated out-of-sample.

This is the "implement the best variant" entry point: it walk-forward tests
EVERY strategy and tells you which one (if any) actually has an edge on the
data the optimizer never saw.
"""

from __future__ import annotations

import argparse

from bot.config import load_config
from bot.data import fetch_ohlcv, make_exchange
from bot.selector import select_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best strategy (OOS validated).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--splits", type=int, default=4, help="walk-forward folds")
    parser.add_argument("--min-trades", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Fetching {cfg.history_limit} x {cfg.timeframe} candles of {cfg.symbol} "
          f"from {cfg.exchange} ...")
    exchange = make_exchange(cfg.exchange)
    df = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, cfg.history_limit)
    print(f"Got {len(df)} candles "
          f"({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})\n")
    print("Walk-forward testing every strategy (this takes a moment) ...\n")

    result = select_strategy(df, cfg, n_splits=args.splits, min_trades=args.min_trades)
    print(result.summary())

    best = result.best
    if best is not None:
        print(f"\nTo deploy: set strategy.name='{best.name}' in config.yaml, run paper mode,")
        print("and only go live after a clean paper run.")


if __name__ == "__main__":
    main()
