# src/hyperstat/backtest/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    start: pd.Timestamp
    end: pd.Timestamp
    n_steps: int

    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float

    avg_gross: float
    avg_net: float
    avg_turnover: float

    pnl_gross: float
    pnl_funding: float
    pnl_fees: float
    pnl_slippage: float
    pnl_net: float


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _infer_steps_per_year(index: pd.DatetimeIndex) -> float:
    """
    Heuristic: estimate steps/year from median time delta.
    Works for 1m/5m/15m/1h etc.
    """
    if len(index) < 3:
        return 0.0
    dt = index.to_series().diff().dropna()
    if dt.empty:
        return 0.0
    median_s = float(dt.median().total_seconds())
    if median_s <= 0:
        return 0.0
    return float((365.25 * 24 * 3600) / median_s)


def compute_performance_metrics(
    equity_curve: pd.DataFrame,
    weights_curve: pd.DataFrame,
    turnover_curve: pd.Series,
    breakdown: Dict[str, float],
) -> PerformanceMetrics:
    """
    equity_curve columns: ["equity"]
    weights_curve: index ts, columns symbols, values weights
    turnover_curve: index ts, value = sum(abs(delta_w))
    breakdown keys: pnl_gross, pnl_funding, fees, slippage, pnl_net
    """
    eq = equity_curve["equity"].astype(float)
    idx = eq.index
    n = int(eq.shape[0])

    if n < 2:
        return PerformanceMetrics(
            start=idx[0] if n else pd.Timestamp("1970-01-01", tz="UTC"),
            end=idx[-1] if n else pd.Timestamp("1970-01-01", tz="UTC"),
            n_steps=n,
            total_return=float("nan"),
            cagr=float("nan"),
            ann_vol=float("nan"),
            sharpe=float("nan"),
            max_drawdown=float("nan"),
            avg_gross=float("nan"),
            avg_net=float("nan"),
            avg_turnover=float("nan"),
            pnl_gross=float(breakdown.get("pnl_gross", 0.0)),
            pnl_funding=float(breakdown.get("pnl_funding", 0.0)),
            pnl_fees=float(breakdown.get("fees", 0.0)),
            pnl_slippage=float(breakdown.get("slippage", 0.0)),
            pnl_net=float(breakdown.get("pnl_net", 0.0)),
        )

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    steps_per_year = _infer_steps_per_year(idx)
    if steps_per_year > 0:
        years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 24 * 3600)
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / max(years, 1e-12)) - 1.0) if years > 0 else float("nan")
    else:
        cagr = float("nan")

    # per-step returns
    r = eq.pct_change().dropna()
    if steps_per_year > 0 and not r.empty:
        ann_vol = float(r.std(ddof=1) * np.sqrt(steps_per_year))
        ann_ret = float(r.mean() * steps_per_year)
        sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else float("nan")
    else:
        ann_vol, sharpe = float("nan"), float("nan")

    mdd = _max_drawdown(eq)

    # exposures
    gross = weights_curve.abs().sum(axis=1) if not weights_curve.empty else pd.Series(index=idx, dtype=float)
    net = weights_curve.sum(axis=1) if not weights_curve.empty else pd.Series(index=idx, dtype=float)

    avg_gross = float(gross.mean()) if not gross.empty else float("nan")
    avg_net = float(net.mean()) if not net.empty else float("nan")
    avg_turnover = float(turnover_curve.mean()) if not turnover_curve.empty else float("nan")

    return PerformanceMetrics(
        start=idx[0],
        end=idx[-1],
        n_steps=n,
        total_return=total_ret,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        avg_gross=avg_gross,
        avg_net=avg_net,
        avg_turnover=avg_turnover,
        pnl_gross=float(breakdown.get("pnl_gross", 0.0)),
        pnl_funding=float(breakdown.get("pnl_funding", 0.0)),
        pnl_fees=float(breakdown.get("fees", 0.0)),
        pnl_slippage=float(breakdown.get("slippage", 0.0)),
        pnl_net=float(breakdown.get("pnl_net", 0.0)),
    )
