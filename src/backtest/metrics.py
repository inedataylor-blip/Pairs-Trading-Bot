"""Performance metrics for backtest analysis."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .engine import BacktestResult, BacktestTrade


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a backtest."""

    # Returns
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int  # in days
    avg_drawdown: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_return: float
    avg_holding_period: float

    # Risk metrics
    calmar_ratio: float
    avg_win_loss_ratio: float
    expectancy: float

    # Correlation
    correlation_to_benchmark: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "avg_drawdown": self.avg_drawdown,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "avg_holding_period": self.avg_holding_period,
            "calmar_ratio": self.calmar_ratio,
            "avg_win_loss_ratio": self.avg_win_loss_ratio,
            "expectancy": self.expectancy,
            "correlation_to_benchmark": self.correlation_to_benchmark,
        }

    def summary(self) -> str:
        """Generate text summary of metrics."""
        return f"""
Performance Summary
==================
Total Return: {self.total_return:.2%}
CAGR: {self.cagr:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}

Trade Statistics
----------------
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.1%}
Profit Factor: {self.profit_factor:.2f}
Avg Win/Loss: {self.avg_win_loss_ratio:.2f}
Avg Holding: {self.avg_holding_period:.1f} days
"""


def calculate_metrics(
    result: BacktestResult,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest result.

    Args:
        result: BacktestResult object
        benchmark_returns: Optional benchmark returns series
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics object
    """
    equity = result.equity_curve
    trades = [t for t in result.trades if t.exit_date is not None]

    # Calculate returns
    returns = equity.pct_change().dropna()

    # Total return
    total_return = result.total_return

    # CAGR
    days = (result.end_date - result.start_date).days
    years = days / 365
    if years > 0 and result.final_capital > 0:
        cagr = (result.final_capital / result.initial_capital) ** (1 / years) - 1
    else:
        cagr = 0

    # Volatility (annualized)
    annualized_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

    # Sharpe ratio
    if annualized_vol > 0:
        excess_return = cagr - risk_free_rate
        sharpe = excess_return / annualized_vol
    else:
        sharpe = 0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    if downside_vol > 0:
        sortino = (cagr - risk_free_rate) / downside_vol
    else:
        sortino = 0

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    avg_dd = abs(drawdown.mean()) if len(drawdown) > 0 else 0

    # Max drawdown duration
    is_in_drawdown = drawdown < 0
    dd_periods = []
    current_dd_length = 0
    for in_dd in is_in_drawdown:
        if in_dd:
            current_dd_length += 1
        else:
            if current_dd_length > 0:
                dd_periods.append(current_dd_length)
            current_dd_length = 0
    if current_dd_length > 0:
        dd_periods.append(current_dd_length)

    max_dd_duration = max(dd_periods) if dd_periods else 0

    # Trade statistics
    num_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]

    num_winners = len(winning_trades)
    num_losers = len(losing_trades)
    win_rate = num_winners / num_trades if num_trades > 0 else 0

    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0

    # Profit factor
    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = abs(sum(t.pnl for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average trade
    avg_trade_return = (
        np.mean([t.return_pct for t in trades]) if trades else 0
    )

    # Average holding period
    avg_holding = (
        np.mean([t.holding_days for t in trades]) if trades else 0
    )

    # Calmar ratio
    calmar = cagr / max_dd if max_dd > 0 else 0

    # Win/loss ratio
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Expectancy (per trade)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Correlation to benchmark
    correlation = None
    if benchmark_returns is not None:
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 10:
            correlation = returns.loc[common_idx].corr(benchmark_returns.loc[common_idx])

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        avg_drawdown=avg_dd,
        total_trades=num_trades,
        winning_trades=num_winners,
        losing_trades=num_losers,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_trade_return=avg_trade_return,
        avg_holding_period=avg_holding,
        calmar_ratio=calmar,
        avg_win_loss_ratio=win_loss_ratio,
        expectancy=expectancy,
        correlation_to_benchmark=correlation,
    )


def aggregate_walk_forward_metrics(
    results: List[BacktestResult],
) -> PerformanceMetrics:
    """
    Aggregate metrics from walk-forward backtest results.

    Args:
        results: List of BacktestResult from walk-forward test

    Returns:
        Aggregated PerformanceMetrics
    """
    # Combine all trades
    all_trades = []
    for r in results:
        all_trades.extend([t for t in r.trades if t.exit_date is not None])

    # Combine equity curves
    all_equity = pd.concat([r.equity_curve for r in results])
    all_equity = all_equity[~all_equity.index.duplicated(keep="last")]
    all_equity = all_equity.sort_index()

    # Calculate overall returns
    total_return = (all_equity.iloc[-1] / all_equity.iloc[0]) - 1 if len(all_equity) > 1 else 0

    returns = all_equity.pct_change().dropna()

    # Time span
    start_date = all_equity.index[0]
    end_date = all_equity.index[-1]
    days = (end_date - start_date).days
    years = days / 365

    # CAGR
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility
    annualized_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

    # Sharpe
    risk_free_rate = 0.02
    sharpe = (cagr - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    avg_dd = abs(drawdown.mean()) if len(drawdown) > 0 else 0

    # Trade stats
    num_trades = len(all_trades)
    winners = [t for t in all_trades if t.pnl > 0]
    losers = [t for t in all_trades if t.pnl <= 0]

    win_rate = len(winners) / num_trades if num_trades > 0 else 0
    avg_win = np.mean([t.pnl for t in winners]) if winners else 0
    avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_trade = np.mean([t.return_pct for t in all_trades]) if all_trades else 0
    avg_holding = np.mean([t.holding_days for t in all_trades]) if all_trades else 0

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=0,  # Not calculated for aggregate
        avg_drawdown=avg_dd,
        total_trades=num_trades,
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_trade_return=avg_trade,
        avg_holding_period=avg_holding,
        calmar_ratio=cagr / max_dd if max_dd > 0 else 0,
        avg_win_loss_ratio=avg_win / avg_loss if avg_loss > 0 else float("inf"),
        expectancy=(win_rate * avg_win) - ((1 - win_rate) * avg_loss),
        correlation_to_benchmark=None,
    )
