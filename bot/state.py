"""Crash-safe persistence for the live/paper loop.

The bot saves its state (open position, peak equity for the drawdown
kill-switch, last processed candle) to a JSON file after every action. On
restart it reloads that file so it doesn't, for example, forget an open
position and its stop-loss.

For LIVE mode the saved position is additionally reconciled against the
exchange's real balance at startup — the exchange is the source of truth.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass

from .risk import Position

log = logging.getLogger("state")


@dataclass
class BotState:
    mode: str
    symbol: str
    peak_equity: float
    last_bar_ts: int | None = None
    position: dict | None = None   # serialized Position or None

    def to_position(self) -> Position | None:
        if not self.position:
            return None
        return Position(**self.position)


def save_state(path: str, *, mode: str, symbol: str, peak_equity: float,
               last_bar_ts: int | None, position: Position | None) -> None:
    """Atomically write state to `path` (temp file + rename)."""
    state = BotState(
        mode=mode,
        symbol=symbol,
        peak_equity=peak_equity,
        last_bar_ts=last_bar_ts,
        position=asdict(position) if position else None,
    )
    try:
        d = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(asdict(state), fh, indent=2)
        os.replace(tmp, path)
    except Exception as exc:  # persistence must not crash trading
        log.warning("Could not save state to %s: %s", path, exc)


def load_state(path: str) -> BotState | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return BotState(**raw)
    except Exception as exc:
        log.warning("Could not load state from %s: %s", path, exc)
        return None
