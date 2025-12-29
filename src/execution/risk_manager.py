"""Risk management for the pairs trading bot."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..signal_generation.signals import TradingSignal


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    passed: bool
    checks: Dict[str, bool]
    reasons: List[str]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checks": self.checks,
            "reasons": self.reasons,
            "timestamp": self.timestamp.isoformat(),
        }


class RiskManager:
    """
    Manages risk limits and pre-trade checks.

    Enforces position limits, exposure limits, and loss limits
    to protect the trading account.
    """

    def __init__(
        self,
        max_pairs: int = 3,
        max_pair_exposure: float = 0.40,
        max_gross_exposure: float = 1.50,
        max_trade_loss: float = 0.03,
        max_daily_loss: float = 0.05,
        min_buying_power_ratio: float = 0.10,
    ):
        """
        Initialize risk manager.

        Args:
            max_pairs: Maximum number of pairs to trade simultaneously
            max_pair_exposure: Maximum exposure per pair as fraction of account
            max_gross_exposure: Maximum total gross exposure as fraction
            max_trade_loss: Maximum loss per trade as fraction of account
            max_daily_loss: Maximum daily loss as fraction of account
            min_buying_power_ratio: Minimum buying power to maintain
        """
        self.max_pairs = max_pairs
        self.max_pair_exposure = max_pair_exposure
        self.max_gross_exposure = max_gross_exposure
        self.max_trade_loss = max_trade_loss
        self.max_daily_loss = max_daily_loss
        self.min_buying_power_ratio = min_buying_power_ratio

        # Tracking
        self._daily_start_equity: Optional[float] = None
        self._trading_halted: bool = False
        self._halt_reason: Optional[str] = None

    def set_daily_start(self, equity: float) -> None:
        """Set the starting equity for daily loss tracking."""
        self._daily_start_equity = equity
        self._trading_halted = False
        self._halt_reason = None
        logger.info(f"Daily risk reset. Starting equity: ${equity:,.2f}")

    def check_position_limit(self, current_positions: int) -> Tuple[bool, str]:
        """Check if we can open a new position."""
        can_open = current_positions < self.max_pairs
        reason = (
            ""
            if can_open
            else f"Max pairs limit reached ({current_positions}/{self.max_pairs})"
        )
        return can_open, reason

    def check_pair_exposure(
        self,
        proposed_value: float,
        account_value: float,
    ) -> Tuple[bool, str]:
        """Check if proposed trade exceeds pair exposure limit."""
        exposure_ratio = proposed_value / account_value if account_value > 0 else 1.0
        within_limit = exposure_ratio <= self.max_pair_exposure
        reason = (
            ""
            if within_limit
            else f"Pair exposure {exposure_ratio:.1%} exceeds limit {self.max_pair_exposure:.1%}"
        )
        return within_limit, reason

    def check_gross_exposure(
        self,
        current_gross: float,
        proposed_addition: float,
        account_value: float,
    ) -> Tuple[bool, str]:
        """Check if total gross exposure is within limits."""
        new_gross = current_gross + proposed_addition
        exposure_ratio = new_gross / account_value if account_value > 0 else 1.0
        within_limit = exposure_ratio <= self.max_gross_exposure
        reason = (
            ""
            if within_limit
            else f"Gross exposure {exposure_ratio:.1%} exceeds limit {self.max_gross_exposure:.1%}"
        )
        return within_limit, reason

    def check_buying_power(
        self,
        buying_power: float,
        account_value: float,
        required_margin: float,
    ) -> Tuple[bool, str]:
        """Check if sufficient buying power available."""
        min_required = account_value * self.min_buying_power_ratio
        available_for_trade = buying_power - min_required

        has_power = available_for_trade >= required_margin
        reason = (
            ""
            if has_power
            else f"Insufficient buying power: ${buying_power:,.0f} - ${min_required:,.0f} reserve < ${required_margin:,.0f} needed"
        )
        return has_power, reason

    def check_daily_loss(
        self,
        current_equity: float,
    ) -> Tuple[bool, str]:
        """Check if daily loss limit has been breached."""
        if self._daily_start_equity is None:
            return True, ""

        daily_pnl = current_equity - self._daily_start_equity
        daily_return = daily_pnl / self._daily_start_equity

        within_limit = daily_return > -self.max_daily_loss
        reason = (
            ""
            if within_limit
            else f"Daily loss {daily_return:.2%} exceeds limit {self.max_daily_loss:.1%}"
        )
        return within_limit, reason

    def check_trade_risk(
        self,
        position_value: float,
        account_value: float,
    ) -> Tuple[bool, str]:
        """Check if single trade risk is within limits."""
        risk_amount = position_value * 0.1  # Assume 10% adverse move
        risk_ratio = risk_amount / account_value if account_value > 0 else 1.0

        within_limit = risk_ratio <= self.max_trade_loss
        reason = (
            ""
            if within_limit
            else f"Trade risk {risk_ratio:.2%} exceeds limit {self.max_trade_loss:.1%}"
        )
        return within_limit, reason

    def pre_trade_check(
        self,
        signal: TradingSignal,
        proposed_value: float,
        current_positions: int,
        current_gross_exposure: float,
        account_value: float,
        buying_power: float,
        current_equity: float,
    ) -> RiskCheckResult:
        """
        Perform comprehensive pre-trade risk checks.

        Args:
            signal: The trading signal to evaluate
            proposed_value: Dollar value of proposed trade
            current_positions: Number of open positions
            current_gross_exposure: Current total gross exposure
            account_value: Total account value
            buying_power: Available buying power
            current_equity: Current equity value

        Returns:
            RiskCheckResult with pass/fail and details
        """
        checks = {}
        reasons = []

        # Check if trading is halted
        if self._trading_halted:
            return RiskCheckResult(
                passed=False,
                checks={"trading_halted": False},
                reasons=[f"Trading halted: {self._halt_reason}"],
            )

        # Run all checks
        passed, reason = self.check_position_limit(current_positions)
        checks["position_limit"] = passed
        if not passed:
            reasons.append(reason)

        passed, reason = self.check_pair_exposure(proposed_value, account_value)
        checks["pair_exposure"] = passed
        if not passed:
            reasons.append(reason)

        passed, reason = self.check_gross_exposure(
            current_gross_exposure, proposed_value, account_value
        )
        checks["gross_exposure"] = passed
        if not passed:
            reasons.append(reason)

        # Estimate required margin (simplified)
        required_margin = proposed_value * 0.5  # 50% margin requirement
        passed, reason = self.check_buying_power(
            buying_power, account_value, required_margin
        )
        checks["buying_power"] = passed
        if not passed:
            reasons.append(reason)

        passed, reason = self.check_daily_loss(current_equity)
        checks["daily_loss"] = passed
        if not passed:
            reasons.append(reason)
            self._trading_halted = True
            self._halt_reason = reason

        passed, reason = self.check_trade_risk(proposed_value, account_value)
        checks["trade_risk"] = passed
        if not passed:
            reasons.append(reason)

        all_passed = all(checks.values())

        result = RiskCheckResult(
            passed=all_passed,
            checks=checks,
            reasons=reasons,
        )

        if not all_passed:
            logger.warning(f"Risk check failed: {reasons}")
        else:
            logger.debug("All risk checks passed")

        return result

    def check_stop_loss(
        self,
        position_pnl: float,
        account_value: float,
    ) -> bool:
        """
        Check if a position should be stopped out.

        Args:
            position_pnl: Current P&L of the position
            account_value: Total account value

        Returns:
            True if stop loss should be triggered
        """
        loss_ratio = -position_pnl / account_value if account_value > 0 else 0
        return loss_ratio > self.max_trade_loss

    def get_position_size_limit(
        self,
        account_value: float,
        current_gross_exposure: float,
    ) -> float:
        """
        Get maximum allowed position size.

        Args:
            account_value: Total account value
            current_gross_exposure: Current gross exposure

        Returns:
            Maximum dollar value for new position
        """
        # Limit by pair exposure
        max_by_pair = account_value * self.max_pair_exposure

        # Limit by remaining gross exposure
        max_by_gross = (
            account_value * self.max_gross_exposure - current_gross_exposure
        )

        return max(0, min(max_by_pair, max_by_gross))

    def halt_trading(self, reason: str) -> None:
        """Halt all trading."""
        self._trading_halted = True
        self._halt_reason = reason
        logger.warning(f"Trading halted: {reason}")

    def resume_trading(self) -> None:
        """Resume trading."""
        self._trading_halted = False
        self._halt_reason = None
        logger.info("Trading resumed")

    @property
    def is_trading_halted(self) -> bool:
        """Check if trading is halted."""
        return self._trading_halted

    def get_status(self) -> dict:
        """Get risk manager status."""
        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "daily_start_equity": self._daily_start_equity,
            "limits": {
                "max_pairs": self.max_pairs,
                "max_pair_exposure": self.max_pair_exposure,
                "max_gross_exposure": self.max_gross_exposure,
                "max_daily_loss": self.max_daily_loss,
                "max_trade_loss": self.max_trade_loss,
            },
        }
