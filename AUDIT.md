# Code audit

A correctness-focused pass over the trading bot. In a trading system, bugs cost
real money, so this prioritizes accounting/look-ahead/live-execution issues over
style. Each finding below was fixed and covered by a test.

## Findings & fixes

### 1. Paper-mode resume double-counted equity — HIGH
`PaperBroker.__init__` starts with full cash. On restart, `engine._restore`
set `broker.position` directly without restoring the reduced cash, so equity =
full cash + restored position value (overstated), and the next sell inflated
cash further.
**Fix:** persist and restore `cash` and `realized_pnl` for paper mode
(`bot/state.py`, `bot/engine.py`). Regression test:
`test_paper_resume_does_not_double_count_equity`.

### 2. Per-trade PnL ignored the entry fee — MEDIUM
Cash accounting was correct, but the `pnl` recorded on each closed trade only
subtracted the exit fee. This made win-rate, profit factor and expectancy
optimistic.
**Fix:** store `entry_fee` on `Position`; net both fees in
`PaperBroker.sell` (`bot/risk.py`, `bot/broker.py`). Test:
`test_paper_broker_pnl_nets_both_fees`.

### 3. Live loop acted on the forming candle (repaint risk) — MEDIUM/HIGH (live)
`fetch_ohlcv` returns the in-progress candle as the last row. The live engine
computed signals on it, so a signal could appear and then vanish before the bar
closed.
**Fix:** the engine now computes signals on the last *closed* candle
(`df.iloc[:-1]`) and detects a new bar from its timestamp, while still using the
latest price for execution and stop checks (`bot/engine.py`).

### 4. Spot market-buy order could be rejected live — MEDIUM (live)
On Binance-style exchanges a spot market BUY defaults to requiring a quote
amount/price, but `LiveBroker.buy` passes a base-asset amount.
**Fix:** set `createMarketBuyOrderRequiresPrice=False` (and
`adjustForTimeDifference=True` to avoid clock-skew errors) in `make_exchange`
(`bot/data.py`).

## Verified correct (no change needed)

- **No look-ahead in backtests.** All indicators are causal (EMA/RSI/ATR/MACD
  use `ewm`; SMA/Bollinger use trailing `rolling`; Donchian uses `shift(1)`;
  Supertrend/ADX are iterative-causal). Computing signals once over the full
  series therefore equals per-bar computation — confirmed by the vectorization
  matching the previous per-bar loop and by the indicator tests.
- **Backtest entry timing.** A bar's stop/take is checked before that bar's
  entry, so a position is never stopped out on its own entry bar.
- **Equity-curve alignment** after early kill-switch exit and residual close.
- **Walk-forward is genuinely out-of-sample**: the test fold includes only a
  warmup prefix, and trading starts after it.

## Known limitations (by design, not bugs)

- Long-only spot (no shorting, no leverage) — deliberate for a small account.
- `LiveBroker` keeps a local view of the position for stop/take; it assumes
  full fills on market orders (reconciled against balance on restart).
- No guarantee of profit. The selector reports out-of-sample edge honestly and
  recommends *not* trading when none exists.
