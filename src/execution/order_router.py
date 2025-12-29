"""Order routing for executing spread trades."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from loguru import logger

from ..data.alpaca_client import AlpacaClient
from ..signal_generation.signals import SignalType, TradingSignal


@dataclass
class OrderResult:
    """Result of an order execution."""

    success: bool
    order_id: Optional[str]
    symbol: str
    side: str
    qty: int
    filled_price: Optional[float]
    status: str
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "filled_price": self.filled_price,
            "status": self.status,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SpreadTradeResult:
    """Result of a spread trade (both legs)."""

    success: bool
    signal: TradingSignal
    leg1_result: OrderResult
    leg2_result: OrderResult
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "signal": self.signal.to_dict(),
            "leg1": self.leg1_result.to_dict(),
            "leg2": self.leg2_result.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


class OrderRouter:
    """Routes orders to Alpaca for execution."""

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        dry_run: bool = False,
    ):
        """
        Initialize order router.

        Args:
            alpaca_client: AlpacaClient instance
            dry_run: If True, simulate orders without executing
        """
        self.client = alpaca_client
        self.dry_run = dry_run
        self._order_history: List[OrderResult] = []

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
    ) -> OrderResult:
        """
        Submit a single order.

        Args:
            symbol: Stock symbol
            qty: Number of shares (positive)
            side: 'buy' or 'sell'

        Returns:
            OrderResult object
        """
        if qty <= 0:
            return OrderResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side,
                qty=qty,
                filled_price=None,
                status="rejected",
                error="Quantity must be positive",
            )

        if self.dry_run:
            logger.info(f"[DRY RUN] Would {side} {qty} shares of {symbol}")
            result = OrderResult(
                success=True,
                order_id=f"dry_run_{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                side=side,
                qty=qty,
                filled_price=self.client.get_latest_price(symbol),
                status="simulated",
            )
        else:
            try:
                order = self.client.submit_market_order(symbol, qty, side)
                result = OrderResult(
                    success=True,
                    order_id=order["id"],
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    filled_price=None,  # Market orders fill at market price
                    status=order["status"],
                )
            except Exception as e:
                logger.error(f"Order failed for {symbol}: {e}")
                result = OrderResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    filled_price=None,
                    status="failed",
                    error=str(e),
                )

        self._order_history.append(result)
        return result

    def close_position(self, symbol: str) -> OrderResult:
        """
        Close an existing position.

        Args:
            symbol: Stock symbol

        Returns:
            OrderResult object
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would close position for {symbol}")
            return OrderResult(
                success=True,
                order_id=f"dry_run_close_{symbol}",
                symbol=symbol,
                side="close",
                qty=0,
                filled_price=None,
                status="simulated",
            )

        try:
            result = self.client.close_position(symbol)
            if result:
                return OrderResult(
                    success=True,
                    order_id=result["id"],
                    symbol=symbol,
                    side="close",
                    qty=0,
                    filled_price=None,
                    status=result["status"],
                )
            else:
                return OrderResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    side="close",
                    qty=0,
                    filled_price=None,
                    status="no_position",
                    error="No position to close",
                )
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side="close",
                qty=0,
                filled_price=None,
                status="failed",
                error=str(e),
            )

    def execute_spread_trade(
        self,
        signal: TradingSignal,
        shares_asset1: int,
        shares_asset2: int,
    ) -> SpreadTradeResult:
        """
        Execute a spread trade based on signal.

        Args:
            signal: TradingSignal object
            shares_asset1: Shares of first asset (positive = buy, negative = sell)
            shares_asset2: Shares of second asset (positive = buy, negative = sell)

        Returns:
            SpreadTradeResult object
        """
        asset1, asset2 = signal.pair

        if signal.signal_type == SignalType.LONG_SPREAD:
            # Long spread: buy asset1, sell asset2
            side1 = "buy"
            side2 = "sell"
        elif signal.signal_type == SignalType.SHORT_SPREAD:
            # Short spread: sell asset1, buy asset2
            side1 = "sell"
            side2 = "buy"
        elif signal.signal_type in (SignalType.EXIT, SignalType.STOP_LOSS):
            # Close both positions
            leg1_result = self.close_position(asset1)
            leg2_result = self.close_position(asset2)

            return SpreadTradeResult(
                success=leg1_result.success and leg2_result.success,
                signal=signal,
                leg1_result=leg1_result,
                leg2_result=leg2_result,
            )
        else:
            # HOLD - no action needed
            return SpreadTradeResult(
                success=True,
                signal=signal,
                leg1_result=OrderResult(
                    success=True,
                    order_id=None,
                    symbol=asset1,
                    side="hold",
                    qty=0,
                    filled_price=None,
                    status="no_action",
                ),
                leg2_result=OrderResult(
                    success=True,
                    order_id=None,
                    symbol=asset2,
                    side="hold",
                    qty=0,
                    filled_price=None,
                    status="no_action",
                ),
            )

        # Execute both legs
        leg1_result = self.submit_order(asset1, abs(shares_asset1), side1)
        leg2_result = self.submit_order(asset2, abs(shares_asset2), side2)

        success = leg1_result.success and leg2_result.success

        if not success:
            logger.warning(
                f"Spread trade partially failed: "
                f"leg1={leg1_result.status}, leg2={leg2_result.status}"
            )

        return SpreadTradeResult(
            success=success,
            signal=signal,
            leg1_result=leg1_result,
            leg2_result=leg2_result,
        )

    def calculate_position_sizes(
        self,
        pair: Tuple[str, str],
        hedge_ratio: float,
        account_value: float,
        risk_per_trade: float,
        spread_volatility: float,
        price1: float,
        price2: float,
    ) -> Tuple[int, int]:
        """
        Calculate position sizes for a spread trade.

        Args:
            pair: Tuple of (asset1, asset2)
            hedge_ratio: Current hedge ratio
            account_value: Total account value
            risk_per_trade: Fraction of account to risk (e.g., 0.02 for 2%)
            spread_volatility: Annualized spread volatility
            price1: Current price of asset1
            price2: Current price of asset2

        Returns:
            Tuple of (shares_asset1, shares_asset2)
        """
        # Dollar amount to risk
        dollar_risk = account_value * risk_per_trade

        # Assume ~10 day holding period
        holding_period_vol = spread_volatility * (10 / 252) ** 0.5

        if holding_period_vol < 0.01:
            holding_period_vol = 0.01  # Minimum volatility floor

        # Position size in dollar terms
        dollar_position = dollar_risk / holding_period_vol

        # Limit to 20% of account per leg
        max_per_leg = account_value * 0.20
        dollar_position = min(dollar_position, max_per_leg)

        # Convert to shares
        shares_asset1 = int(dollar_position / price1)
        shares_asset2 = int(dollar_position * hedge_ratio / price2)

        # Ensure at least 1 share each
        shares_asset1 = max(1, shares_asset1)
        shares_asset2 = max(1, shares_asset2)

        logger.debug(
            f"Position sizes for {pair}: {shares_asset1} / {shares_asset2} shares "
            f"(${dollar_position:.0f} notional, hedge_ratio={hedge_ratio:.3f})"
        )

        return shares_asset1, shares_asset2

    def get_order_history(self) -> List[OrderResult]:
        """Get order history."""
        return self._order_history.copy()

    def clear_history(self) -> None:
        """Clear order history."""
        self._order_history.clear()
