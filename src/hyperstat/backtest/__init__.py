# src/hyperstat/backtest/__init__.py
from __future__ import annotations

from .costs import FeeModel, SlippageModel, CostModel, TradeCostBreakdown
from .engine import BacktestConfig, BacktestEngine, run_backtest
from .metrics import PerformanceMetrics, compute_performance_metrics
from .reports import BacktestReport, save_report_csv, save_report_html

__all__ = [
    "FeeModel",
    "SlippageModel",
    "CostModel",
    "TradeCostBreakdown",
    "BacktestConfig",
    "BacktestEngine",
    "run_backtest",
    "PerformanceMetrics",
    "compute_performance_metrics",
    "BacktestReport",
    "save_report_csv",
    "save_report_html",
]
