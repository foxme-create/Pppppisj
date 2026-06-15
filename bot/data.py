"""Market data fetching via ccxt."""

from __future__ import annotations

import time

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_exchange(name: str, testnet: bool = False, api_key: str = "", api_secret: str = ""):
    """Create a ccxt exchange instance. Import is local so unit tests that
    don't touch the network don't require ccxt to be installed."""
    import ccxt

    klass = getattr(ccxt, name)
    exchange = klass(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )
    if testnet:
        # ccxt exposes sandbox mode for most major exchanges.
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 1500) -> pd.DataFrame:
    """Fetch up to `limit` candles, paging backwards if the exchange caps a
    single request (usually 1000)."""
    all_rows: list[list] = []
    per_call = min(limit, 1000)
    since = None
    # Page from oldest to newest by walking forward in time.
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    if limit > per_call:
        since = exchange.milliseconds() - limit * tf_ms

    while len(all_rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=per_call)
        if not batch:
            break
        all_rows += batch
        since = batch[-1][0] + tf_ms
        if len(batch) < per_call:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=OHLCV_COLUMNS).drop_duplicates(subset="timestamp")
    df = df.tail(limit).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
