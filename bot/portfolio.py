"""Multi-symbol portfolio backtester.

Trades one strategy across several symbols out of a *single shared* cash pool,
capped at `risk.max_positions` concurrent positions. This is closer to how you
would actually run a small account across a few pairs: capital is finite and
shared, so the bot can only hold as many positions as cash and the cap allow.

Returns a `BacktestResult` (same metrics as the single-symbol backtester), so
all reporting/plotting works unchanged.
"""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult
from .config import Config
from .risk import Position, RiskManager
from .strategy import FLAT, LONG, Strategy


def _aligned_frames(data: dict[str, pd.DataFrame], strategy: Strategy):
    """Per-symbol frames (high/low/close/action/atr) indexed by timestamp,
    restricted to timestamps common to all symbols."""
    frames: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        sig = strategy.signals(df)
        frames[sym] = pd.DataFrame(
            {
                "high": df["high"].to_numpy(),
                "low": df["low"].to_numpy(),
                "close": df["close"].to_numpy(),
                "action": sig["action"].to_numpy(),
                "atr": sig["atr"].to_numpy(),
            },
            index=df["timestamp"].to_numpy(),
        )

    common = None
    for f in frames.values():
        common = f.index if common is None else common.intersection(f.index)
    common = common.sort_values()
    return {sym: f.reindex(common) for sym, f in frames.items()}, common


def _close_position(pos: Position, price: float, fee: float, slip: float,
                    trades: list, sym: str) -> float:
    """Close a position at `price`; record the trade; return net cash freed."""
    fill = price * (1 - slip)
    proceeds = fill * pos.size
    f = proceeds * fee
    pnl = (fill - pos.entry_price) * pos.size - f - pos.entry_fee
    trades.append({"side": "sell", "symbol": sym, "price": fill,
                   "size": pos.size, "fee": f, "pnl": pnl})
    return proceeds - f


def run_portfolio_backtest(data: dict[str, pd.DataFrame], strategy: Strategy,
                           cfg: Config) -> BacktestResult:
    if not data:
        raise ValueError("no symbols provided")

    frames, common = _aligned_frames(data, strategy)
    symbols = list(frames)
    n = len(common)
    start = strategy.min_bars
    if n <= start + 1:
        raise ValueError("not enough overlapping history across symbols")

    fee, slip = cfg.risk.fee_rate, cfg.risk.slippage
    max_pos = cfg.risk.max_positions
    cash = cfg.start_equity
    positions: dict[str, Position] = {}
    risk = RiskManager(cfg.risk, cfg.start_equity)
    trades: list[dict] = []
    equity_points: list[float] = []
    ts_points: list = []

    def total_equity(prices: dict[str, float]) -> float:
        return cash + sum(positions[s].size * prices[s] for s in positions)

    for i in range(start, n):
        prices = {s: float(frames[s]["close"].iloc[i]) for s in symbols}

        # 1) Intrabar stop/take per open position.
        for sym in list(positions):
            pos = positions[sym]
            lo = float(frames[sym]["low"].iloc[i])
            hi = float(frames[sym]["high"].iloc[i])
            exit_price = pos.stop_price if lo <= pos.stop_price else (
                pos.take_price if hi >= pos.take_price else None)
            if exit_price is not None:
                cash += _close_position(pos, exit_price, fee, slip, trades, sym)
                del positions[sym]

        equity = total_equity(prices)

        # 2) Portfolio-level drawdown kill-switch.
        if risk.check_kill_switch(equity):
            for sym in list(positions):
                cash += _close_position(positions[sym], prices[sym], fee, slip, trades, sym)
                del positions[sym]
            equity_points.append(cash)
            ts_points.append(common[i])
            break

        # 3) Exit signals first (frees cash and a position slot for entries).
        for sym in symbols:
            if int_or_nan(frames[sym]["action"].iloc[i]) == FLAT and sym in positions:
                cash += _close_position(positions[sym], prices[sym], fee, slip, trades, sym)
                del positions[sym]

        # 4) Entry signals, respecting the global position cap and shared cash.
        for sym in symbols:
            if len(positions) >= max_pos:
                break
            if int_or_nan(frames[sym]["action"].iloc[i]) != LONG or sym in positions:
                continue
            atr = float(frames[sym]["atr"].iloc[i])
            plan = risk.plan_trade(equity, prices[sym], atr)
            if plan.size <= 0:
                continue
            fill = prices[sym] * (1 + slip)
            affordable = cash / (fill * (1 + fee))
            size = min(plan.size, affordable)
            cost = fill * size
            f = cost * fee
            if size > 0 and cost + f <= cash + 1e-9:
                cash -= cost + f
                positions[sym] = Position(entry_price=fill, size=size,
                                          stop_price=plan.stop_price,
                                          take_price=plan.take_price, entry_fee=f)
                trades.append({"side": "buy", "symbol": sym, "price": fill,
                               "size": size, "fee": f})

        equity_points.append(total_equity(prices))
        ts_points.append(common[i])

    # Close residual positions at the final bar's prices.
    if positions:
        last = {s: float(frames[s]["close"].iloc[-1]) for s in symbols}
        for sym in list(positions):
            cash += _close_position(positions[sym], last[sym], fee, slip, trades, sym)
            del positions[sym]
        if equity_points:
            equity_points[-1] = cash

    curve = pd.Series(equity_points, index=pd.to_datetime(ts_points, unit="ms", utc=True))
    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        start_equity=cfg.start_equity,
        final_equity=cash,
        timeframe=cfg.timeframe,
    )


def int_or_nan(value) -> int:
    """Treat NaN (warmup/missing) actions as HOLD (0)."""
    try:
        if value != value:  # NaN
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0
