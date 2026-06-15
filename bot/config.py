"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01      # fraction of equity risked per trade
    stop_atr_mult: float = 2.0        # stop distance = N * ATR
    take_profit_rr: float = 2.0       # take-profit = RR * stop distance
    max_drawdown: float = 0.20        # halt if equity falls this fraction from peak
    max_positions: int = 1            # max concurrent open positions
    fee_rate: float = 0.001           # exchange taker fee per side (0.1%)
    slippage: float = 0.0005          # assumed slippage per side


@dataclass
class StrategyConfig:
    name: str = "ema_rsi"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    mode: str = "backtest"            # backtest | paper | live
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    symbols: list[str] = field(default_factory=list)  # for portfolio backtests
    timeframe: str = "1h"
    start_equity: float = 200.0       # used for backtest/paper accounting
    history_limit: int = 1500         # candles to pull for backtest
    poll_seconds: int = 60            # live loop sleep between checks

    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""

    # Persistence: where to save live/paper state so a restart resumes safely.
    state_file: str = "state.json"

    # Optional Telegram alerts (trades, kill-switch, errors). Leave empty to
    # disable. Get a token from @BotFather; chat_id from @userinfobot.
    telegram_token: str = ""
    telegram_chat_id: str = ""

    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # Optional parameter grid for run_optimize.py, e.g.
    # {"ema_fast": [10, 20], "ema_slow": [50, 100]}
    optimize: dict[str, list] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in {"backtest", "paper", "live"}:
            raise ValueError(f"mode must be backtest|paper|live, got {self.mode!r}")
        if self.mode == "live" and self.testnet:
            raise ValueError(
                "mode=live with testnet=true is contradictory. "
                "Set testnet=false for real trading (be careful!)."
            )
        if self.mode in {"paper", "live"} and not (self.api_key and self.api_secret):
            raise ValueError(f"mode={self.mode} requires api_key and api_secret")
        if not 0 < self.risk.risk_per_trade <= 0.1:
            raise ValueError("risk.risk_per_trade should be in (0, 0.1]")
        if not 0 < self.risk.max_drawdown < 1:
            raise ValueError("risk.max_drawdown should be in (0, 1)")


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    risk = RiskConfig(**(raw.pop("risk", {}) or {}))
    strat_raw = raw.pop("strategy", {}) or {}
    strategy = StrategyConfig(
        name=strat_raw.get("name", "ema_rsi"),
        params=strat_raw.get("params", {}) or {},
    )
    optimize = raw.pop("optimize", {}) or {}
    cfg = Config(risk=risk, strategy=strategy, optimize=optimize, **raw)
    cfg.validate()
    return cfg
