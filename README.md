# cryptobot

A small, **safety-first** crypto trading bot framework in Python.

> ⚠️ **Read this first.** Trading is risky. With a small account ($50–200),
> high-frequency / scalping strategies are *mathematically* losing games once
> you account for fees (~0.1% per side), spread and slippage. This project is
> built so you **validate a strategy on historical data and on a testnet with
> fake money first**, and only switch to real funds once you have evidence it
> works. No strategy here is a guaranteed money printer — they are starting
> points you must test.

## What it does

1. **Backtest** a strategy on historical OHLCV data (free, via `ccxt`).
2. **Paper-trade** the same strategy live on an exchange **testnet** (real
   market data, fake money).
3. **Live-trade** with real funds — only by flipping one flag in the config.

Every mode runs the *exact same* strategy + risk code, so what you test is
what you trade.

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
  indicators.py  EMA / RSI / ATR (pure numpy/pandas)
  data.py        fetch OHLCV via ccxt
  strategy.py    strategy interface + built-in strategies
  risk.py        position sizing, stops, drawdown kill-switch
  broker.py      paper broker + live ccxt broker (same interface)
  backtest.py    event-driven backtester
  engine.py      live/paper trading loop
run_backtest.py  CLI: backtest
run_live.py      CLI: paper/live
tests/           unit tests
```
