"""Event-driven backtester.

Walks the candle history one bar at a time (no look-ahead): the strategy only
ever sees data up to the current bar, and orders fill at that bar's close.
Intrabar stop/take are checked against each bar's high/low.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import metrics
from .broker import PaperBroker
from .config import Config
from .risk import RiskManager
from .strategy import FLAT, LONG, Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[dict]
    start_equity: float
    final_equity: float
    timeframe: str = "1h"

    @property
    def total_return(self) -> float:
        if self.start_equity == 0:
            return 0.0
        return self.final_equity / self.start_equity - 1

    @property
    def num_trades(self) -> int:
        return sum(1 for t in self.trades if t["side"] == "sell")

    @property
    def win_rate(self) -> float:
        return metrics.win_rate(self.trades)

    @property
    def max_drawdown(self) -> float:
        return metrics.max_drawdown(self.equity_curve)

    @property
    def sharpe(self) -> float:
        return metrics.sharpe(self.equity_curve, self.timeframe)

    @property
    def sortino(self) -> float:
        return metrics.sortino(self.equity_curve, self.timeframe)

    @property
    def cagr(self) -> float:
        return metrics.cagr(self.equity_curve, self.timeframe)

    @property
    def profit_factor(self) -> float:
        return metrics.profit_factor(self.trades)

    @property
    def expectancy(self) -> float:
        return metrics.expectancy(self.trades)

    def summary(self) -> str:
        pf = self.profit_factor
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        return (
            f"Start equity:   {self.start_equity:.2f}\n"
            f"Final equity:   {self.final_equity:.2f}\n"
            f"Total return:   {self.total_return * 100:.2f}%\n"
            f"CAGR:           {self.cagr * 100:.2f}%\n"
            f"Sharpe:         {self.sharpe:.2f}\n"
            f"Sortino:        {self.sortino:.2f}\n"
            f"Profit factor:  {pf_str}\n"
            f"Expectancy:     {self.expectancy:.4f} / trade\n"
            f"Trades:         {self.num_trades}\n"
            f"Win rate:       {self.win_rate * 100:.1f}%\n"
            f"Max drawdown:   {self.max_drawdown * 100:.2f}%\n"
        )


def run_backtest(df: pd.DataFrame, strategy: Strategy, cfg: Config) -> BacktestResult:
    broker = PaperBroker(cfg.start_equity, cfg.risk)
    risk = RiskManager(cfg.risk, cfg.start_equity)
    equity_points: list[float] = []

    n = len(df)
    start = strategy.min_bars
    for i in range(start, n):
        window = df.iloc[: i + 1]
        bar = df.iloc[i]
        price = float(bar["close"])

        # 1) Manage an open position: check intrabar stop/take first.
        if broker.position:
            pos = broker.position
            if bar["low"] <= pos.stop_price:
                broker.sell(pos.stop_price)
            elif bar["high"] >= pos.take_price:
                broker.sell(pos.take_price)

        # 2) Drawdown kill-switch: if tripped, flatten and stop trading.
        equity = broker.equity(price)
        if risk.check_kill_switch(equity):
            if broker.position:
                broker.sell(price)
            equity_points.append(broker.equity(price))
            break

        # 3) Strategy signal on closed data only.
        signal = strategy.compute(window)
        if signal.action == FLAT and broker.position:
            broker.sell(price)
        elif signal.action == LONG and not broker.position:
            plan = risk.plan_trade(equity, price, signal.atr)
            if plan.size > 0:
                broker.buy(price, plan.size, plan.stop_price, plan.take_price)

        equity_points.append(broker.equity(price))

    # Close any residual position at the last price. Update (don't append) the
    # final equity point so the curve stays aligned 1:1 with the bars walked.
    if broker.position:
        last_price = float(df.iloc[-1]["close"])
        broker.sell(last_price)
        if equity_points:
            equity_points[-1] = broker.equity(last_price)

    idx = df["datetime"].iloc[start : start + len(equity_points)]
    curve = pd.Series(equity_points, index=idx.values)
    return BacktestResult(
        equity_curve=curve,
        trades=broker.trades,
        start_equity=cfg.start_equity,
        final_equity=broker.equity(float(df.iloc[-1]["close"])),
        timeframe=cfg.timeframe,
    )
