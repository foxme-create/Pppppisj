# cryptobot

A small, **safety-first** crypto trading bot framework in Python.

> ⚠️ **Read this first.** Trading is risky. With a small account ($50–200),
> high-frequency / scalping strategies are *mathematically* losing games once
> you account for fees (~0.1% per side), spread and slippage. This project is
> built so you **validate a strategy on historical data and on a testnet with
> fake money first**, and only switch to real funds once you have evidence it
> works. No strategy here is a guaranteed money printer — they are starting
> points you must test.

## There is no "guaranteed instant profit" bot

If a bot could *guarantee* immediate profit, nobody would share or sell it —
they would quietly run it. This project does the honest, maximally-useful thing
instead: it tests many proven strategies on your market, **automatically picks
the one with a real out-of-sample edge**, and tells you to **not trade** when
none of them has an edge. Capital you don't lose is the first profit.

## What it does

1. **Backtest** a strategy on historical OHLCV data (free, via `ccxt`).
2. **Select** the best strategy automatically, validated out-of-sample
   (`run_select.py`) — this is the "implement the best variant" step.
3. **Optimize** a strategy's parameters with walk-forward validation
   (`run_optimize.py`).
4. **Paper-trade** the chosen strategy on an exchange **testnet** (real market
   data, fake money).
5. **Live-trade** with real funds — only by flipping one flag in the config.

Every mode runs the *exact same* strategy + risk code, so what you test is
what you trade.

## Strategies included

Each is a documented, real-world approach — not a magic indicator:

| Name | Family | Idea |
|------|--------|------|
| `donchian_breakout` | Trend (Turtle) | Buy N-bar high breakout, exit M-bar low |
| `supertrend` | Trend | Long while ATR-based Supertrend points up |
| `macd_trend` | Trend | MACD momentum up + above long EMA |
| `ema_rsi` | Trend | Fast>slow EMA with an RSI momentum band |
| `bollinger_reversion` | Mean reversion | Buy below lower band, exit at middle |
| `mean_reversion` | Mean reversion | Buy oversold RSI dips inside an uptrend |

## Finding the best strategy (recommended first step)

```bash
python run_select.py --config config.yaml
```

This walk-forward tests **every** strategy and prints an out-of-sample ranking
plus a verdict: which variant to trade, or "no edge — don't trade".

## Risk controls (always on)

- Per-trade **stop-loss** (ATR-based).
- **Position sizing** by % of equity risked per trade.
- **Take-profit** target (risk/reward multiple).
- **Max drawdown kill-switch** — bot halts if equity drops past a threshold.
- **Max concurrent positions** cap.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml   # then edit

# 1) Backtest on history (no keys needed)
python run_backtest.py --config config.yaml

# 2) Paper trade on testnet (needs free testnet API keys)
python run_live.py --config config.yaml         # mode: paper in config
```

## Getting free testnet keys

Binance Spot Testnet: https://testnet.binance.vision/ — log in with GitHub,
generate an API key/secret, paste into `config.yaml`. No real money involved.

## Going live (real money)

Only after your backtest **and** paper run show a positive, stable result over
a meaningful sample:

1. Create real API keys on your exchange with **trade** permission only
   (never enable withdrawals).
2. Set `mode: live` and real keys in `config.yaml`.
3. Start with the smallest size possible and watch it.

## Project layout

```
bot/
  config.py      load + validate YAML config
  indicators.py  EMA/RSI/ATR/MACD/Bollinger/Donchian/Supertrend/ADX (pure)
  data.py        fetch OHLCV via ccxt
  strategy.py    strategy interface + 6 built-in strategies + default grids
  risk.py        position sizing, stops, drawdown kill-switch
  broker.py      paper broker + live ccxt broker (same interface)
  backtest.py    event-driven backtester (no look-ahead)
  metrics.py     Sharpe / Sortino / CAGR / profit factor / expectancy
  optimize.py    grid search + walk-forward (out-of-sample) validation
  selector.py    auto-pick the best strategy, OOS-validated
  engine.py      live/paper trading loop
run_backtest.py  CLI: backtest one strategy
run_select.py    CLI: auto-select the best strategy
run_optimize.py  CLI: optimize params with walk-forward
run_live.py      CLI: paper/live
tests/           unit tests
```
