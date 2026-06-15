"""Equity-curve and drawdown plots.

matplotlib is imported lazily (with the headless 'Agg' backend) so the rest of
the package has no hard dependency on it — only plotting needs it.
"""

from __future__ import annotations

import pandas as pd


def plot_equity(curve: pd.Series, path: str, *, benchmark: pd.Series | None = None,
                title: str = "Equity curve") -> str:
    """Save a two-panel PNG: equity over time and the drawdown underneath.

    Returns the path written. Raises ImportError with a helpful message if
    matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Plotting needs matplotlib. Install it with: pip install matplotlib"
        ) from exc

    if curve.empty:
        raise ValueError("cannot plot an empty equity curve")

    peak = curve.cummax()
    drawdown = (curve - peak) / peak

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(curve.index, curve.values, label="Strategy equity", color="#1f77b4")
    if benchmark is not None and not benchmark.empty:
        # Scale buy & hold to the same starting equity for a fair visual compare.
        scaled = benchmark / benchmark.iloc[0] * curve.iloc[0]
        ax1.plot(scaled.index, scaled.values, label="Buy & hold",
                 color="#999999", linestyle="--")
    ax1.set_title(title)
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                     color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
