"""Position management for pairs trading."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class PairPosition:
    """Represents a position in a trading pair."""

    pair: Tuple[str, str]
    direction: int  # 1 for long spread, -1 for short spread
    shares_asset1: int
    shares_asset2: int
    entry_price1: float
    entry_price2: float
    entry_hedge_ratio: float
    entry_zscore: float
    entry_time: datetime
    entry_spread: float

    # Updated values
    current_price1: Optional[float] = None
    current_price2: Optional[float] = None
    current_spread: Optional[float] = None
    current_zscore: Optional[float] = None
    unrealized_pnl: float = 0.0

    def update_prices(
        self,
        price1: float,
        price2: float,
        spread: float,
        zscore: float,
    ) -> None:
        """Update current prices and calculate P&L."""
        self.current_price1 = price1
        self.current_price2 = price2
        self.current_spread = spread
        self.current_zscore = zscore

        # Calculate unrealized P&L
        # Long spread: profit when spread increases (asset1 up, asset2 down)
        # Short spread: profit when spread decreases (asset1 down, asset2 up)
        pnl_asset1 = self.shares_asset1 * (price1 - self.entry_price1)
        pnl_asset2 = self.shares_asset2 * (price2 - self.entry_price2)

        self.unrealized_pnl = pnl_asset1 + pnl_asset2

    @property
    def market_value(self) -> float:
        """Get current market value of position."""
        if self.current_price1 is None or self.current_price2 is None:
            return 0.0

        val1 = abs(self.shares_asset1) * self.current_price1
        val2 = abs(self.shares_asset2) * self.current_price2
        return val1 + val2

    @property
    def entry_value(self) -> float:
        """Get entry value of position."""
        val1 = abs(self.shares_asset1) * self.entry_price1
        val2 = abs(self.shares_asset2) * self.entry_price2
        return val1 + val2

    @property
    def holding_time(self) -> float:
        """Get holding time in days."""
        delta = datetime.now() - self.entry_time
        return delta.total_seconds() / 86400

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "direction": self.direction,
            "shares": {
                self.pair[0]: self.shares_asset1,
                self.pair[1]: self.shares_asset2,
            },
            "entry_prices": {
                self.pair[0]: self.entry_price1,
                self.pair[1]: self.entry_price2,
            },
            "current_prices": {
                self.pair[0]: self.current_price1,
                self.pair[1]: self.current_price2,
            },
            "entry_hedge_ratio": self.entry_hedge_ratio,
            "entry_zscore": self.entry_zscore,
            "current_zscore": self.current_zscore,
            "entry_spread": self.entry_spread,
            "current_spread": self.current_spread,
            "entry_time": self.entry_time.isoformat(),
            "holding_time_days": self.holding_time,
            "unrealized_pnl": self.unrealized_pnl,
            "market_value": self.market_value,
        }


class PositionManager:
    """Manages all pair positions."""

    def __init__(self):
        """Initialize position manager."""
        self._positions: Dict[Tuple[str, str], PairPosition] = {}
        self._closed_positions: List[dict] = []
        self._realized_pnl: float = 0.0
        self._starting_equity: Optional[float] = None
        self._current_equity: Optional[float] = None

    def open_position(
        self,
        pair: Tuple[str, str],
        direction: int,
        shares_asset1: int,
        shares_asset2: int,
        price1: float,
        price2: float,
        hedge_ratio: float,
        zscore: float,
        spread: float,
    ) -> PairPosition:
        """
        Open a new pair position.

        Args:
            pair: Tuple of (asset1, asset2)
            direction: 1 for long spread, -1 for short spread
            shares_asset1: Shares of asset1 (positive for long, negative for short)
            shares_asset2: Shares of asset2 (positive for long, negative for short)
            price1: Entry price for asset1
            price2: Entry price for asset2
            hedge_ratio: Hedge ratio at entry
            zscore: Z-score at entry
            spread: Spread value at entry

        Returns:
            Created PairPosition object
        """
        if pair in self._positions:
            logger.warning(f"Position already exists for {pair}, closing first")
            self.close_position(pair, price1, price2)

        position = PairPosition(
            pair=pair,
            direction=direction,
            shares_asset1=shares_asset1,
            shares_asset2=shares_asset2,
            entry_price1=price1,
            entry_price2=price2,
            entry_hedge_ratio=hedge_ratio,
            entry_zscore=zscore,
            entry_time=datetime.now(),
            entry_spread=spread,
        )

        self._positions[pair] = position

        logger.info(
            f"Opened {'LONG' if direction > 0 else 'SHORT'} spread position: "
            f"{pair} - {shares_asset1} / {shares_asset2} shares"
        )

        return position

    def close_position(
        self,
        pair: Tuple[str, str],
        price1: float,
        price2: float,
    ) -> Optional[dict]:
        """
        Close a pair position and calculate realized P&L.

        Args:
            pair: Tuple of (asset1, asset2)
            price1: Exit price for asset1
            price2: Exit price for asset2

        Returns:
            Closed position details or None if no position
        """
        if pair not in self._positions:
            logger.warning(f"No position to close for {pair}")
            return None

        position = self._positions[pair]
        position.update_prices(price1, price2, 0, 0)

        # Calculate realized P&L
        realized_pnl = position.unrealized_pnl
        self._realized_pnl += realized_pnl

        closed_info = {
            "pair": pair,
            "direction": position.direction,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "holding_time_days": position.holding_time,
            "entry_prices": {
                pair[0]: position.entry_price1,
                pair[1]: position.entry_price2,
            },
            "exit_prices": {pair[0]: price1, pair[1]: price2},
            "shares": {pair[0]: position.shares_asset1, pair[1]: position.shares_asset2},
            "realized_pnl": realized_pnl,
            "entry_zscore": position.entry_zscore,
        }

        self._closed_positions.append(closed_info)
        del self._positions[pair]

        logger.info(f"Closed position for {pair}: P&L = ${realized_pnl:.2f}")

        return closed_info

    def get_position(self, pair: Tuple[str, str]) -> Optional[PairPosition]:
        """Get position for a pair."""
        return self._positions.get(pair)

    def has_position(self, pair: Tuple[str, str]) -> bool:
        """Check if position exists for a pair."""
        return pair in self._positions

    def get_all_positions(self) -> Dict[Tuple[str, str], PairPosition]:
        """Get all open positions."""
        return self._positions.copy()

    def update_position(
        self,
        pair: Tuple[str, str],
        price1: float,
        price2: float,
        spread: float,
        zscore: float,
    ) -> Optional[PairPosition]:
        """Update position with current prices."""
        if pair not in self._positions:
            return None

        self._positions[pair].update_prices(price1, price2, spread, zscore)
        return self._positions[pair]

    def update_all_positions(
        self,
        prices: Dict[str, float],
        spreads: Dict[Tuple[str, str], float],
        zscores: Dict[Tuple[str, str], float],
    ) -> None:
        """
        Update all positions with current market data.

        Args:
            prices: Dictionary mapping symbol to price
            spreads: Dictionary mapping pair to spread
            zscores: Dictionary mapping pair to zscore
        """
        for pair, position in self._positions.items():
            price1 = prices.get(pair[0])
            price2 = prices.get(pair[1])
            spread = spreads.get(pair, 0)
            zscore = zscores.get(pair, 0)

            if price1 is not None and price2 is not None:
                position.update_prices(price1, price2, spread, zscore)

    @property
    def num_positions(self) -> int:
        """Get number of open positions."""
        return len(self._positions)

    @property
    def total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._realized_pnl

    @property
    def total_market_value(self) -> float:
        """Get total market value of all positions."""
        return sum(p.market_value for p in self._positions.values())

    def set_equity(self, equity: float, is_starting: bool = False) -> None:
        """Set current equity value."""
        self._current_equity = equity
        if is_starting or self._starting_equity is None:
            self._starting_equity = equity

    @property
    def daily_pnl(self) -> float:
        """Get daily P&L (requires setting starting equity)."""
        if self._starting_equity is None or self._current_equity is None:
            return 0.0
        return self._current_equity - self._starting_equity

    def get_summary(self) -> dict:
        """Get summary of all positions and P&L."""
        return {
            "num_positions": self.num_positions,
            "positions": [p.to_dict() for p in self._positions.values()],
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "total_market_value": self.total_market_value,
            "starting_equity": self._starting_equity,
            "current_equity": self._current_equity,
            "daily_pnl": self.daily_pnl,
            "num_closed_trades": len(self._closed_positions),
        }

    def get_closed_positions(self) -> List[dict]:
        """Get list of closed position records."""
        return self._closed_positions.copy()

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        self._starting_equity = self._current_equity
        self._closed_positions = []
