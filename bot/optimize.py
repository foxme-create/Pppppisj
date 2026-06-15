"""Parameter optimization with out-of-sample (walk-forward) validation.

Why this module exists: it is trivially easy to find strategy parameters that
look amazing on past data and then lose money live. That is *overfitting*. The
only honest defence is to choose parameters on one slice of history and measure
them on a *different* slice the optimizer never saw. `walk_forward` does exactly
that and is the number you should trust — not the in-sample grid-search score.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import Config
from .strategy import build_strategy

# A scorer turns a backtest result into a single number to maximise.
Scorer = Callable[[BacktestResult], float]


def sharpe_scorer(min_trades: int = 10) -> Scorer:
    """Maximise Sharpe, but reject samples with too few trades (luck, not edge)."""
    def score(res: BacktestResult) -> float:
        if res.num_trades < min_trades:
            return float("-inf")
        return res.sharpe
    return score


def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a {param: [values]} grid into a list of param dicts."""
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())]


@dataclass
class OptResult:
    best_params: dict
    best_score: float
    best_result: BacktestResult
    leaderboard: list[tuple[dict, float]] = field(default_factory=list)


def grid_search(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list],
    cfg: Config,
    scorer: Scorer | None = None,
) -> OptResult:
    """Exhaustively score every parameter combination on `df` (in-sample)."""
    scorer = scorer or sharpe_scorer()
    board: list[tuple[dict, float]] = []
    best = None  # (score, params, result)

    for params in expand_grid(param_grid):
        strategy = build_strategy(strategy_name, params)
        if len(df) <= strategy.min_bars + 5:
            continue
        res = run_backtest(df, strategy, cfg)
        score = scorer(res)
        board.append((params, score))
        if best is None or score > best[0]:
            best = (score, params, res)

    if best is None:
        raise ValueError("no parameter combination could be evaluated (data too short?)")

    board.sort(key=lambda x: x[1], reverse=True)
    return OptResult(best_params=best[1], best_score=best[0], best_result=best[2], leaderboard=board)


@dataclass
class WalkForwardResult:
    stitched_curve: pd.Series          # out-of-sample equity, all windows joined
    windows: list[dict]                # per-window {params, train_score, test_*}
    start_equity: float
    final_equity: float
    timeframe: str = "1h"

    @property
    def total_return(self) -> float:
        if self.start_equity == 0:
            return 0.0
        return self.final_equity / self.start_equity - 1

    def summary(self) -> str:
        lines = [
            "Walk-forward (OUT-OF-SAMPLE) result",
            "=" * 40,
            f"Windows:           {len(self.windows)}",
            f"OOS total return:  {self.total_return * 100:.2f}%",
            f"OOS final equity:  {self.final_equity:.2f}",
            "",
            "Per-window out-of-sample returns:",
        ]
        for i, w in enumerate(self.windows, 1):
            lines.append(
                f"  #{i}: test_return={w['test_return'] * 100:7.2f}%  "
                f"trades={w['test_trades']:3d}  params={w['params']}"
            )
        return "\n".join(lines)


def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list],
    cfg: Config,
    n_splits: int = 4,
    scorer: Scorer | None = None,
) -> WalkForwardResult:
    """Anchored walk-forward: for each test fold, optimise on all prior data,
    then trade the chosen params on the unseen fold. Returns are compounded
    across folds so the final equity reflects what this *process* would have
    produced live, not a cherry-picked parameter set."""
    scorer = scorer or sharpe_scorer()

    # Warmup must cover the most history-hungry param combo.
    warmup = max(build_strategy(strategy_name, p).min_bars for p in expand_grid(param_grid))
    n = len(df)
    fold = n // (n_splits + 1)
    if fold <= warmup + 5:
        raise ValueError(
            f"not enough data for {n_splits} folds: fold size {fold} <= warmup {warmup}. "
            "Increase history_limit or reduce n_splits."
        )

    equity = cfg.start_equity
    pieces: list[pd.Series] = []
    windows: list[dict] = []

    for k in range(1, n_splits + 1):
        train = df.iloc[: k * fold].reset_index(drop=True)
        # Test slice includes a warmup prefix so indicators are primed, but the
        # backtester only starts trading after the warmup -> truly OOS.
        test = df.iloc[k * fold - warmup : (k + 1) * fold].reset_index(drop=True)
        if len(test) <= warmup + 5:
            break

        opt = grid_search(train, strategy_name, param_grid, cfg, scorer)

        # Run chosen params on the unseen fold, carrying equity forward.
        fold_cfg = Config(**{**cfg.__dict__})
        fold_cfg.start_equity = equity
        strat = build_strategy(strategy_name, opt.best_params)
        test_res = run_backtest(test, strat, fold_cfg)

        equity = test_res.final_equity
        pieces.append(test_res.equity_curve)
        windows.append(
            {
                "params": opt.best_params,
                "train_score": opt.best_score,
                "test_return": test_res.total_return,
                "test_trades": test_res.num_trades,
                "test_sharpe": test_res.sharpe,
            }
        )

    stitched = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return WalkForwardResult(
        stitched_curve=stitched,
        windows=windows,
        start_equity=cfg.start_equity,
        final_equity=equity,
        timeframe=cfg.timeframe,
    )
