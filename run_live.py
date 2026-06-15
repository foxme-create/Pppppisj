#!/usr/bin/env python3
"""Run the bot in paper or live mode (controlled by config `mode`)."""

from __future__ import annotations

import argparse
import logging

from bot.config import load_config
from bot.engine import run_live


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trading bot (paper/live).")
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if cfg.mode == "backtest":
        raise SystemExit("config mode is 'backtest' — use run_backtest.py instead.")

    if cfg.mode == "live":
        print("\n*** LIVE MODE: this will trade REAL money. Ctrl-C now to abort. ***\n")

    run_live(cfg)


if __name__ == "__main__":
    main()
