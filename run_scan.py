#!/usr/bin/env python3
"""Scan many symbols x timeframes for any out-of-sample edge.

Runs the strategy selector across a grid of markets and prints a one-line
verdict per (symbol, timeframe). Use this to thoroughly check whether ANY
edge exists before concluding there is none.

Example:
    python run_scan.py --symbols BTC/USDT,ETH/USDT,SOL/USDT \
                       --timeframes 1h,4h,1d
"""

from __future__ import annotations

import argparse

from bot.config import load_config
from bot.data import fetch_ohlcv, make_exchange
from bot.selector import select_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan symbols x timeframes for edge.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", default="", help="comma list, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", default="", help="comma list, e.g. 1h,4h,1d")
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--min-trades", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or \
        (cfg.symbols or [cfg.symbol])
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()] or [cfg.timeframe]

    exchange = make_exchange(cfg.exchange)
    print(f"Scanning {len(symbols)} symbol(s) x {len(timeframes)} timeframe(s) "
          f"= {len(symbols) * len(timeframes)} markets ...\n")

    header = f"{'symbol':<12}{'tf':<6}{'best strategy':<22}{'OOS ret':>10}  verdict"
    print(header)
    print("-" * len(header))

    found = []
    for tf in timeframes:
        for sym in symbols:
            cfg.symbol = sym
            cfg.timeframe = tf
            try:
                df = fetch_ohlcv(exchange, sym, tf, cfg.history_limit)
                res = select_strategy(df, cfg, n_splits=args.splits, min_trades=args.min_trades)
            except Exception as exc:
                print(f"{sym:<12}{tf:<6}{'-':<22}{'-':>10}  error: {type(exc).__name__}")
                continue
            best = res.best
            if best is not None:
                print(f"{sym:<12}{tf:<6}{best.name:<22}{best.oos_return * 100:>9.2f}%  "
                      f"TRADEABLE (Sharpe {best.oos_sharpe:.2f})")
                found.append((sym, tf, best))
            else:
                # Show the least-bad strategy for context.
                top = res.ranked[0] if res.ranked else None
                ret = f"{top.oos_return * 100:.2f}%" if top else "n/a"
                name = top.name if top else "-"
                print(f"{sym:<12}{tf:<6}{name:<22}{ret:>10}  no edge")

    print()
    if found:
        print(f"Found {len(found)} market(s) with a positive out-of-sample edge:")
        for sym, tf, b in found:
            print(f"  {sym} {tf}: {b.name} ({b.oos_return * 100:.2f}% OOS) — "
                  f"validate on paper before risking money.")
    else:
        print("No market in this scan shows a positive out-of-sample edge.")
        print("That is a genuine result: do not trade these with real money.")


if __name__ == "__main__":
    main()
