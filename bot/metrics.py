"""Performance metrics for an equity curve and a trade list.

All functions are pure and operate on plain pandas/python objects so they can
be unit-tested without running a full backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate number of bars per year for common timeframes, used to
# annualise Sharpe/Sortino/CAGR.
_TF_PER_YEAR = {
    "1m": 525_600,
    "3m": 175_200,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "12h": 730,
    "1d": 365,
    "1w": 52,
}


def periods_per_year(timeframe: str) -> float:
    return _TF_PER_YEAR.get(timeframe, 8_760)


def bar_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def sharpe(equity_curve: pd.Series, timeframe: str, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio (risk-free assumed ~0 for crypto)."""
    r = bar_returns(equity_curve)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    ann = periods_per_year(timeframe)
    excess = r - rf / ann
    return float(excess.mean() / r.std(ddof=1) * np.sqrt(ann))


def sortino(equity_curve: pd.Series, timeframe: str) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    r = bar_returns(equity_curve)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return 0.0
    ann = periods_per_year(timeframe)
    return float(r.mean() / dd * np.sqrt(ann))


def cagr(equity_curve: pd.Series, timeframe: str) -> float:
    """Compound annual growth rate implied by the curve length."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods_per_year(timeframe)
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    return float(((peak - equity_curve) / peak).max())


def _closed_pnls(trades: list[dict]) -> list[float]:
    return [t["pnl"] for t in trades if t["side"] == "sell" and "pnl" in t]


def profit_factor(trades: list[dict]) -> float:
    """Gross profit / gross loss. >1 means the system makes money."""
    pnls = _closed_pnls(trades)
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expectancy(trades: list[dict]) -> float:
    """Average PnL per closed trade (your edge per trade, in quote currency)."""
    pnls = _closed_pnls(trades)
    return float(np.mean(pnls)) if pnls else 0.0


def win_rate(trades: list[dict]) -> float:
    pnls = _closed_pnls(trades)
    if not pnls:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)
