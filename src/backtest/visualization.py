"""Visualization tools for backtest results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .engine import BacktestResult
from .metrics import PerformanceMetrics


class BacktestVisualizer:
    """Visualization tools for backtest analysis."""

    def __init__(self, output_dir: str = "output/charts"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save chart images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_equity_curve(
        self,
        result: BacktestResult,
        benchmark: Optional[pd.Series] = None,
        save: bool = False,
    ) -> Figure:
        """
        Plot equity curve.

        Args:
            result: BacktestResult object
            benchmark: Optional benchmark equity curve
            save: Whether to save the chart

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Normalize to starting value
        equity_normalized = result.equity_curve / result.initial_capital

        ax.plot(equity_normalized.index, equity_normalized, label="Strategy", linewidth=1.5)

        if benchmark is not None:
            bench_normalized = benchmark / benchmark.iloc[0]
            common_idx = equity_normalized.index.intersection(bench_normalized.index)
            ax.plot(
                common_idx,
                bench_normalized.loc[common_idx],
                label="Benchmark",
                linewidth=1.5,
                alpha=0.7,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (Normalized)")
        ax.set_title(f"Equity Curve - {result.pair}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.output_dir / f"equity_{result.pair[0]}_{result.pair[1]}.png"
            fig.savefig(path, dpi=150)

        return fig

    def plot_drawdown(
        self,
        result: BacktestResult,
        save: bool = False,
    ) -> Figure:
        """
        Plot drawdown chart.

        Args:
            result: BacktestResult object
            save: Whether to save the chart

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

        # Equity curve on top
        equity = result.equity_curve
        axes[0].plot(equity.index, equity, linewidth=1.5)
        axes[0].set_ylabel("Equity ($)")
        axes[0].set_title(f"Equity and Drawdown - {result.pair}")
        axes[0].grid(True, alpha=0.3)

        # Drawdown on bottom
        returns = equity.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.5, color="red")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.output_dir / f"drawdown_{result.pair[0]}_{result.pair[1]}.png"
            fig.savefig(path, dpi=150)

        return fig

    def plot_spread_and_signals(
        self,
        result: BacktestResult,
        save: bool = False,
    ) -> Figure:
        """
        Plot spread, z-score, and trading signals.

        Args:
            result: BacktestResult object
            save: Whether to save the chart

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        signals_df = result.signals_df

        # Price ratio
        axes[0].plot(
            signals_df.index,
            signals_df["price1"] / signals_df["price2"],
            label="Price Ratio",
            linewidth=1,
        )
        axes[0].set_ylabel("Price Ratio")
        axes[0].set_title(f"Spread Analysis - {result.pair}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Spread
        axes[1].plot(
            signals_df.index,
            signals_df["spread"],
            label="Spread",
            linewidth=1,
        )
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Spread")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Z-score with signals
        axes[2].plot(
            signals_df.index,
            signals_df["zscore"],
            label="Z-Score",
            linewidth=1,
        )

        # Entry/exit thresholds
        axes[2].axhline(y=2, color="green", linestyle="--", alpha=0.5, label="Entry")
        axes[2].axhline(y=-2, color="green", linestyle="--", alpha=0.5)
        axes[2].axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Exit")
        axes[2].axhline(y=-0.5, color="orange", linestyle="--", alpha=0.5)
        axes[2].axhline(y=3.5, color="red", linestyle="--", alpha=0.5, label="Stop")
        axes[2].axhline(y=-3.5, color="red", linestyle="--", alpha=0.5)

        # Mark trades
        for trade in result.trades:
            if trade.exit_date:
                color = "green" if trade.pnl > 0 else "red"
                axes[2].axvspan(
                    trade.entry_date,
                    trade.exit_date,
                    alpha=0.2,
                    color=color,
                )

        axes[2].set_xlabel("Date")
        axes[2].set_ylabel("Z-Score")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.output_dir / f"signals_{result.pair[0]}_{result.pair[1]}.png"
            fig.savefig(path, dpi=150)

        return fig

    def plot_trade_analysis(
        self,
        result: BacktestResult,
        save: bool = False,
    ) -> Figure:
        """
        Plot trade analysis charts.

        Args:
            result: BacktestResult object
            save: Whether to save the chart

        Returns:
            Matplotlib Figure
        """
        trades = [t for t in result.trades if t.exit_date is not None]

        if not trades:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trades to analyze", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Trade P&L distribution
        pnls = [t.pnl for t in trades]
        axes[0, 0].hist(pnls, bins=20, edgecolor="black", alpha=0.7)
        axes[0, 0].axvline(x=0, color="red", linestyle="--")
        axes[0, 0].axvline(x=np.mean(pnls), color="green", linestyle="-", label=f"Mean: ${np.mean(pnls):.0f}")
        axes[0, 0].set_xlabel("P&L ($)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Trade P&L Distribution")
        axes[0, 0].legend()

        # Return distribution
        returns = [t.return_pct * 100 for t in trades]
        axes[0, 1].hist(returns, bins=20, edgecolor="black", alpha=0.7)
        axes[0, 1].axvline(x=0, color="red", linestyle="--")
        axes[0, 1].set_xlabel("Return (%)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Trade Return Distribution")

        # Holding period vs P&L
        holding = [t.holding_days for t in trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        axes[1, 0].scatter(holding, pnls, c=colors, alpha=0.6)
        axes[1, 0].axhline(y=0, color="gray", linestyle="--")
        axes[1, 0].set_xlabel("Holding Period (days)")
        axes[1, 0].set_ylabel("P&L ($)")
        axes[1, 0].set_title("Holding Period vs P&L")

        # Cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        axes[1, 1].plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl, marker="o", markersize=4)
        axes[1, 1].axhline(y=0, color="gray", linestyle="--")
        axes[1, 1].set_xlabel("Trade Number")
        axes[1, 1].set_ylabel("Cumulative P&L ($)")
        axes[1, 1].set_title("Cumulative P&L by Trade")

        plt.tight_layout()

        if save:
            path = self.output_dir / f"trades_{result.pair[0]}_{result.pair[1]}.png"
            fig.savefig(path, dpi=150)

        return fig

    def create_report(
        self,
        result: BacktestResult,
        metrics: PerformanceMetrics,
        save: bool = True,
    ) -> None:
        """
        Create a full visual report.

        Args:
            result: BacktestResult object
            metrics: PerformanceMetrics object
            save: Whether to save charts
        """
        # Generate all charts
        self.plot_equity_curve(result, save=save)
        self.plot_drawdown(result, save=save)
        self.plot_spread_and_signals(result, save=save)
        self.plot_trade_analysis(result, save=save)

        # Save metrics summary
        if save:
            summary_path = self.output_dir / f"summary_{result.pair[0]}_{result.pair[1]}.txt"
            with open(summary_path, "w") as f:
                f.write(metrics.summary())

        plt.close("all")
