"""Signal generation modules for pairs trading."""

from .kalman_filter import KalmanFilterHedgeRatio
from .signals import SignalGenerator, TradingSignal
from .spread_calculator import SpreadCalculator
from .zscore import ZScoreCalculator

__all__ = [
    "KalmanFilterHedgeRatio",
    "SpreadCalculator",
    "ZScoreCalculator",
    "SignalGenerator",
    "TradingSignal",
]
