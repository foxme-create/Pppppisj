#!/usr/bin/env python3
"""Optimize strategy parameters with walk-forward (out-of-sample) validation.

Reads the parameter grid from the `optimize:` section of the config. The number
you should trust is the WALK-FORWARD result, not the in-sample grid search —
the former is measured on data the optimizer never saw.
"""

from __future__ import annotations

import argparse

from bot.config import load_config
from bot.data import fetch_ohlcv, make_exchange
from bot.optimize import grid_search, sharpe_scorer, walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize + walk-forward validate.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--splits", type=int, default=4, help="walk-forward folds")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="reject param sets with fewer trades than this")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.optimize:
        raise SystemExit("No `optimize:` grid in config. Add one (see config.example.yaml).")

    print(f"Fetching {cfg.history_limit} x {cfg.timeframe} candles of {cfg.symbol} ...")
    exchange = make_exchange(cfg.exchange)
    df = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, cfg.history_limit)
    print(f"Got {len(df)} candles\n")

    scorer = sharpe_scorer(min_trades=args.min_trades)

    print("In-sample grid search (for reference — can be overfit) ...")
    opt = grid_search(df, cfg.strategy.name, cfg.optimize, cfg, scorer)
    print(f"  Best in-sample params: {opt.best_params}  (Sharpe={opt.best_score:.2f})")
    print("  Top 5:")
    for params, score in opt.leaderboard[:5]:
        s = "n/a" if score == float("-inf") else f"{score:.2f}"
        print(f"    {s:>6}  {params}")

    print("\nWalk-forward validation (TRUST THIS) ...")
    wf = walk_forward(df, cfg.strategy.name, cfg.optimize, cfg, n_splits=args.splits, scorer=scorer)
    print()
    print(wf.summary())

    print("\nReminder: positive OOS return + Sharpe > ~1 across most windows is")
    print("encouraging. Negative or wildly unstable OOS = do NOT trade this live.")


if __name__ == "__main__":
    main()
