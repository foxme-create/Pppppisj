"""Risk management: position sizing, stops, and a drawdown kill-switch."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RiskConfig


@dataclass
class Position:
    entry_price: float
    size: float            # base-asset quantity
    stop_price: float
    take_price: float

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.size


@dataclass
class TradePlan:
    """Computed entry plan; size==0 means 'do not take this trade'."""
    size: float
    stop_price: float
    take_price: float


class RiskManager:
    def __init__(self, cfg: RiskConfig, start_equity: float):
        self.cfg = cfg
        self.peak_equity = start_equity
        self.halted = False

    def update_peak(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity

    def drawdown(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity

    def check_kill_switch(self, equity: float) -> bool:
        """Returns True (and latches halted) if drawdown breaches the limit."""
        self.update_peak(equity)
        if self.drawdown(equity) >= self.cfg.max_drawdown:
            self.halted = True
        return self.halted

    def plan_trade(self, equity: float, price: float, atr: float) -> TradePlan:
        """Size a long so that hitting the stop loses ~risk_per_trade of equity.

        stop distance = stop_atr_mult * ATR (floored to avoid div-by-zero on
        flat markets). size = risk_amount / stop_distance, capped so notional
        never exceeds available equity (spot, no leverage).
        """
        stop_dist = self.cfg.stop_atr_mult * atr
        if stop_dist <= 0 or price <= 0:
            return TradePlan(0.0, 0.0, 0.0)

        risk_amount = equity * self.cfg.risk_per_trade
        size = risk_amount / stop_dist

        # Spot cap: can't buy more than equity allows (minus a fee buffer).
        max_size = (equity * (1 - self.cfg.fee_rate)) / price
        size = min(size, max_size)
        if size <= 0:
            return TradePlan(0.0, 0.0, 0.0)

        stop_price = price - stop_dist
        take_price = price + stop_dist * self.cfg.take_profit_rr
        return TradePlan(size, stop_price, take_price)
