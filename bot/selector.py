"""Automatic strategy selection.

Runs walk-forward (out-of-sample) validation for every registered strategy
using its default parameter grid, then ranks them by *out-of-sample*
performance. The winner is the strategy whose edge survived on data the
optimizer never saw — the closest honest answer to "the best variant".

Crucially, if NO strategy shows a positive out-of-sample result, the selector
says so. Not trading is the correct, capital-preserving decision when there is
no demonstrable edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .optimize import WalkForwardResult, sharpe_scorer, walk_forward
from .strategy import available_strategies, default_grid


@dataclass
class StrategyEvaluation:
    name: str
    oos_return: float          # out-of-sample total return (compounded)
    oos_sharpe: float          # mean OOS Sharpe across windows
    num_windows: int
    total_trades: int
    wf: WalkForwardResult

    @property
    def tradeable(self) -> bool:
        """A minimum bar to even consider trading: positive OOS return with a
        meaningful number of trades (not one lucky shot)."""
        return self.oos_return > 0 and self.total_trades >= 10


@dataclass
class SelectionResult:
    ranked: list[StrategyEvaluation]

    @property
    def best(self) -> StrategyEvaluation | None:
        tradeable = [e for e in self.ranked if e.tradeable]
        return tradeable[0] if tradeable else None

    def summary(self) -> str:
        lines = ["Strategy selection — OUT-OF-SAMPLE ranking", "=" * 48]
        lines.append(f"{'strategy':<22}{'OOS ret':>10}{'OOS Shrp':>10}{'trades':>8}  status")
        for e in self.ranked:
            status = "tradeable" if e.tradeable else "skip (no edge)"
            lines.append(
                f"{e.name:<22}{e.oos_return * 100:>9.2f}%{e.oos_sharpe:>10.2f}"
                f"{e.total_trades:>8}  {status}"
            )
        lines.append("")
        if self.best is None:
            lines.append("VERDICT: no strategy shows a positive out-of-sample edge on this")
            lines.append("market/timeframe. Do NOT trade. Try another symbol/timeframe, or")
            lines.append("accept that there is no exploitable edge here right now.")
        else:
            b = self.best
            lines.append(f"VERDICT: best variant = '{b.name}' "
                         f"(OOS return {b.oos_return * 100:.2f}%, Sharpe {b.oos_sharpe:.2f}).")
            lines.append(f"Recommended params per window: {[w['params'] for w in b.wf.windows]}")
            lines.append("Still validate with paper trading on testnet before risking real money.")
        return "\n".join(lines)


def _mean_oos_sharpe(wf: WalkForwardResult) -> float:
    sharpes = [w["test_sharpe"] for w in wf.windows if w["test_trades"] > 0]
    return float(sum(sharpes) / len(sharpes)) if sharpes else 0.0


def select_strategy(
    df: pd.DataFrame,
    cfg: Config,
    n_splits: int = 4,
    min_trades: int = 5,
    strategies: list[str] | None = None,
) -> SelectionResult:
    """Evaluate every (or the given) strategies and rank by OOS performance."""
    names = strategies or available_strategies()
    scorer = sharpe_scorer(min_trades=min_trades)
    evals: list[StrategyEvaluation] = []

    for name in names:
        grid = default_grid(name)
        try:
            wf = walk_forward(df, name, grid, cfg, n_splits=n_splits, scorer=scorer)
        except ValueError:
            # Not enough data for this strategy's warmup at this fold size.
            continue
        total_trades = sum(w["test_trades"] for w in wf.windows)
        evals.append(
            StrategyEvaluation(
                name=name,
                oos_return=wf.total_return,
                oos_sharpe=_mean_oos_sharpe(wf),
                num_windows=len(wf.windows),
                total_trades=total_trades,
                wf=wf,
            )
        )

    # Rank: tradeable first, then by OOS return.
    evals.sort(key=lambda e: (e.tradeable, e.oos_return), reverse=True)
    return SelectionResult(ranked=evals)
