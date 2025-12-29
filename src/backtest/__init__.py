"""Backtesting framework for pairs trading strategies."""

from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .visualization import BacktestVisualizer

__all__ = ["BacktestEngine", "PerformanceMetrics", "BacktestVisualizer"]
