"""Alpaca API client wrapper for market data and trading."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from loguru import logger


class AlpacaClient:
    """Wrapper for Alpaca Trading and Data APIs."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Whether to use paper trading (default: True)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        # Initialize clients
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        logger.info(f"Initialized Alpaca client (paper={paper})")

    def get_account(self) -> dict:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "currency": account.currency,
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": pos.symbol,
                "qty": int(pos.qty),
                "side": "long" if int(pos.qty) > 0 else "short",
                "market_value": float(pos.market_value),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
            }
            for pos in positions
        ]

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol."""
        try:
            pos = self.trading_client.get_open_position(symbol)
            return {
                "symbol": pos.symbol,
                "qty": int(pos.qty),
                "side": "long" if int(pos.qty) > 0 else "short",
                "market_value": float(pos.market_value),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
            }
        except Exception:
            return None

    def get_historical_bars(
        self,
        symbol: str,
        days: int = 252,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """
        Get historical price bars.

        Args:
            symbol: Stock symbol
            days: Number of trading days of history
            timeframe: Bar timeframe (default: daily)

        Returns:
            DataFrame with OHLCV data
        """
        end = datetime.now()
        start = end - timedelta(days=int(days * 1.5))  # Account for non-trading days

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        bars = self.data_client.get_stock_bars(request)

        if symbol not in bars.data or not bars.data[symbol]:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        data = []
        for bar in bars.data[symbol]:
            data.append(
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index)

        # Return most recent 'days' bars
        return df.tail(days)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        try:
            bars = self.get_historical_bars(symbol, days=5)
            if not bars.empty:
                return float(bars["close"].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def get_prices_for_pair(
        self,
        symbol1: str,
        symbol2: str,
        days: int = 252,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Get aligned price series for a pair of symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            days: Number of trading days

        Returns:
            Tuple of aligned close price series
        """
        bars1 = self.get_historical_bars(symbol1, days)
        bars2 = self.get_historical_bars(symbol2, days)

        if bars1.empty or bars2.empty:
            raise ValueError(f"Could not get data for pair ({symbol1}, {symbol2})")

        # Align the series on common dates
        prices1 = bars1["close"]
        prices2 = bars2["close"]

        common_idx = prices1.index.intersection(prices2.index)
        prices1 = prices1.loc[common_idx]
        prices2 = prices2.loc[common_idx]

        return prices1, prices2

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
    ) -> dict:
        """
        Submit a market order.

        Args:
            symbol: Stock symbol
            qty: Number of shares (positive)
            side: 'buy' or 'sell'

        Returns:
            Order details
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = self.trading_client.submit_order(request)

        logger.info(f"Submitted {side} order for {qty} shares of {symbol}")

        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": int(order.qty),
            "side": order.side.value,
            "status": order.status.value,
            "submitted_at": str(order.submitted_at),
        }

    def close_position(self, symbol: str) -> Optional[dict]:
        """
        Close an open position.

        Args:
            symbol: Stock symbol

        Returns:
            Order details or None if no position
        """
        try:
            order = self.trading_client.close_position(symbol)
            logger.info(f"Closed position for {symbol}")
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "status": order.status.value,
            }
        except Exception as e:
            logger.warning(f"Could not close position for {symbol}: {e}")
            return None

    def close_all_positions(self) -> list[dict]:
        """Close all open positions."""
        orders = self.trading_client.close_all_positions(cancel_orders=True)
        logger.info(f"Closed {len(orders)} positions")
        return [
            {"id": str(o.id), "symbol": o.symbol, "status": o.status.value}
            for o in orders
        ]

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.trading_client.get_clock()
        return clock.is_open
