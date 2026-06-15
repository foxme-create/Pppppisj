"""Brokers: a paper broker for simulation and a live ccxt broker.

Both expose the same tiny interface so the engine code is identical for
paper and live trading:

    broker.equity(price) -> float
    broker.position -> Position | None
    broker.buy(price, size, stop, take)
    broker.sell(price)            # close the whole position
"""

from __future__ import annotations

from .config import RiskConfig
from .risk import Position


class PaperBroker:
    """Simulated broker with cash + a single long position. Applies fees and
    slippage so paper results don't flatter you versus reality."""

    def __init__(self, start_equity: float, risk: RiskConfig):
        self.cash = start_equity
        self.risk = risk
        self.position: Position | None = None
        self.realized_pnl = 0.0
        self.trades: list[dict] = []

    def equity(self, price: float) -> float:
        eq = self.cash
        if self.position:
            eq += self.position.size * price
        return eq

    def _fill_price(self, price: float, side: str) -> float:
        slip = self.risk.slippage
        return price * (1 + slip) if side == "buy" else price * (1 - slip)

    def buy(self, price: float, size: float, stop: float, take: float) -> None:
        if self.position or size <= 0:
            return
        fill = self._fill_price(price, "buy")
        cost = fill * size
        fee = cost * self.risk.fee_rate
        if cost + fee > self.cash:
            # Trim to fit available cash.
            size = (self.cash / (fill * (1 + self.risk.fee_rate)))
            cost = fill * size
            fee = cost * self.risk.fee_rate
            if size <= 0:
                return
        self.cash -= cost + fee
        self.position = Position(entry_price=fill, size=size, stop_price=stop, take_price=take)
        self.trades.append({"side": "buy", "price": fill, "size": size, "fee": fee})

    def sell(self, price: float) -> None:
        if not self.position:
            return
        fill = self._fill_price(price, "sell")
        proceeds = fill * self.position.size
        fee = proceeds * self.risk.fee_rate
        self.cash += proceeds - fee
        pnl = (fill - self.position.entry_price) * self.position.size - fee
        self.realized_pnl += pnl
        self.trades.append(
            {"side": "sell", "price": fill, "size": self.position.size, "fee": fee, "pnl": pnl}
        )
        self.position = None


class LiveBroker:
    """Live trading through a ccxt exchange. Mirrors PaperBroker's interface.

    Note: real fills, balances and partial fills are handled by the exchange;
    we keep a lightweight local view of the open position for stop/take logic.
    """

    def __init__(self, exchange, symbol: str, risk: RiskConfig, quote_ccy: str = "USDT"):
        self.exchange = exchange
        self.symbol = symbol
        self.risk = risk
        self.quote_ccy = quote_ccy
        self.position: Position | None = None

    def equity(self, price: float) -> float:
        bal = self.exchange.fetch_balance()
        quote = bal.get(self.quote_ccy, {}).get("free", 0.0) or 0.0
        eq = float(quote)
        if self.position:
            eq += self.position.size * price
        return eq

    def reconcile(self, price: float) -> None:
        """Reconcile a restored position against the exchange's real balance.

        The exchange is the source of truth. If we think we hold a position but
        the base-asset balance is effectively zero (it was sold elsewhere, or
        the order never filled), drop the stale position. If the held amount is
        smaller than recorded, shrink the position to match.
        """
        if not self.position:
            return
        base_ccy = self.symbol.split("/")[0]
        bal = self.exchange.fetch_balance()
        held = float(bal.get(base_ccy, {}).get("free", 0.0) or 0.0)

        # Treat dust (worth < 1 unit of quote) as zero.
        if held * price < 1.0:
            self.position = None
            return
        if held < self.position.size:
            self.position.size = held

    def buy(self, price: float, size: float, stop: float, take: float) -> None:
        if self.position or size <= 0:
            return
        amount = float(self.exchange.amount_to_precision(self.symbol, size))
        if amount <= 0:
            return
        order = self.exchange.create_market_buy_order(self.symbol, amount)
        fill = float(order.get("average") or order.get("price") or price)
        filled = float(order.get("filled") or amount)
        self.position = Position(entry_price=fill, size=filled, stop_price=stop, take_price=take)

    def sell(self, price: float) -> None:
        if not self.position:
            return
        amount = float(self.exchange.amount_to_precision(self.symbol, self.position.size))
        if amount > 0:
            self.exchange.create_market_sell_order(self.symbol, amount)
        self.position = None
