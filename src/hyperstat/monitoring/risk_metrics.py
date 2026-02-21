# src/hyperstat/monitoring/risk_metrics.py
from __future__ import annotations

import io
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _tail_csv(path: Path, n_lines: int) -> str:
    """
    Efficiently read the last n_lines of an append-only CSV, keeping header.
    This avoids reading huge files in Streamlit refresh loops.
    """
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return ""
        buf = deque(maxlen=n_lines)
        for line in f:
            buf.append(line)
    return header + "".join(buf)


def load_equity_df(run_dir: str, tail: int = 50000) -> pd.DataFrame:
    p = Path(run_dir) / "equity.csv"
    txt = _tail_csv(p, tail)
    if not txt.strip():
        return pd.DataFrame(columns=["equity"]).astype(float)

    df = pd.read_csv(io.StringIO(txt))
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_mids_df(run_dir: str, tail: int = 200000) -> pd.DataFrame:
    p = Path(run_dir) / "mids.csv"
    txt = _tail_csv(p, tail)
    if not txt.strip():
        return pd.DataFrame(columns=["ts", "symbol", "mid"])

    df = pd.read_csv(io.StringIO(txt))
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df = df.dropna(subset=["mid"])
    return df


def load_weights_df(run_dir: str, tail: int = 200000) -> pd.DataFrame:
    p = Path(run_dir) / "weights.csv"
    txt = _tail_csv(p, tail)
    if not txt.strip():
        return pd.DataFrame(columns=["ts", "symbol", "weight"])

    df = pd.read_csv(io.StringIO(txt))
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"])
    return df


def infer_steps_per_year(index: pd.DatetimeIndex) -> float:
    """
    Estimate steps/year from median timestep (works for 1m/5m/15m/h).
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


def compute_drawdown(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    eq = equity.astype(float).dropna()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    dd.name = "drawdown"
    return dd


def compute_equity_metrics(equity_df: pd.DataFrame, window: int = 2000) -> Dict[str, float]:
    """
    Mean return, vol, Sharpe from the equity curve.
    Equity is assumed net of fees if it comes from accountValue.
    """
    if equity_df.empty or "equity" not in equity_df.columns:
        return {}

    eq = equity_df["equity"].astype(float).dropna()
    if len(eq) < 5:
        return {"equity": float(eq.iloc[-1])} if len(eq) else {}

    r = eq.pct_change().dropna()
    steps_per_year = infer_steps_per_year(eq.index)
    if steps_per_year <= 0:
        steps_per_year = 365.25 * 24  # fallback

    r_win = r.tail(window) if len(r) > window else r

    mean_step = float(r_win.mean())
    vol_step = float(r_win.std(ddof=1))

    ann_return = float(mean_step * steps_per_year)
    ann_vol = float(vol_step * np.sqrt(steps_per_year))
    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else float("nan")

    dd = compute_drawdown(eq)
    max_dd = float(dd.min()) if not dd.empty else float("nan")

    # extra helpful metrics
    last_equity = float(eq.iloc[-1])
    ret_total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    # Sortino (downside vol seulement)
    neg_r = r_win[r_win < 0]
    if not neg_r.empty:
        down_vol = float(neg_r.std(ddof=1) * np.sqrt(steps_per_year))
        sortino = float(ann_return / down_vol) if down_vol > 1e-12 else float("nan")
    else:
        sortino = float("nan")

    # Calmar = CAGR / |max drawdown|
    calmar = float(ann_return / abs(max_dd)) if abs(max_dd) > 1e-12 else float("nan")

    # Win rate (fraction des barres positives)
    win_rate = float((r_win > 0).mean())

    return {
        "equity": last_equity,
        "total_return": ret_total,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "steps_per_year": float(steps_per_year),
    }


def compute_var_cvar(
    equity_df: pd.DataFrame,
    alpha: float = 0.05,
    window: int = 5000,
    horizon_steps: int = 1,
) -> Dict[str, float]:
    """
    Historical VaR/CVaR on equity returns.
    - alpha=0.05 => VaR 95%
    - horizon_steps: approximate multi-step by aggregating log-returns.
    """
    if equity_df.empty or "equity" not in equity_df.columns:
        return {}

    eq = equity_df["equity"].astype(float).dropna()
    if len(eq) < max(50, horizon_steps + 5):
        return {}

    r = eq.pct_change().dropna()
    r = r.tail(window) if len(r) > window else r

    if horizon_steps > 1:
        lr = np.log1p(r.values)
        agg = pd.Series(lr).rolling(horizon_steps).sum().dropna()
        r_h = np.expm1(agg.values)
    else:
        r_h = r.values

    q = float(np.nanquantile(r_h, alpha))
    tail = r_h[r_h <= q]
    cvar = float(np.nanmean(tail)) if tail.size else float("nan")

    return {
        f"var_{int((1 - alpha) * 100)}": q,
        f"cvar_{int((1 - alpha) * 100)}": cvar,
        "alpha": float(alpha),
        "window": float(window),
        "horizon_steps": float(horizon_steps),
    }


def compute_correlation_matrix(mids_df: pd.DataFrame, window: int = 2000) -> pd.DataFrame:
    """
    Correlation of asset returns from mids long-format (ts,symbol,mid).
    """
    if mids_df.empty:
        return pd.DataFrame()

    px = mids_df.pivot_table(index="ts", columns="symbol", values="mid", aggfunc="last").sort_index()
    px = px.ffill().dropna(how="all")
    if px.shape[0] < 10 or px.shape[1] < 2:
        return pd.DataFrame()

    px = px.tail(window) if len(px) > window else px
    r = px.pct_change().dropna(how="all")
    if r.empty:
        return pd.DataFrame()

    return r.corr()


def compute_portfolio_var_from_weights(
    mids_df: pd.DataFrame,
    weights_last: Dict[str, float],
    window: int = 2000,
) -> Dict[str, float]:
    """
    Compute instantaneous portfolio variance estimate from:
      - last target weights (dimensionless notional/equity)
      - covariance of returns from mids

    Returns:
      - port_var_step, port_vol_step
    """
    if not weights_last:
        return {}

    px = mids_df.pivot_table(index="ts", columns="symbol", values="mid", aggfunc="last").sort_index()
    px = px.ffill().dropna(how="all")
    if px.shape[0] < 10:
        return {}

    px = px.tail(window) if len(px) > window else px
    r = px.pct_change().dropna(how="all")
    if r.empty:
        return {}

    cov = r.cov()
    syms = [s for s in cov.columns if s in weights_last]
    if len(syms) < 2:
        return {}

    w = np.asarray([float(weights_last[s]) for s in syms], dtype=float)
    C = cov.loc[syms, syms].values.astype(float)

    var = float(w @ C @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    return {"port_var_step": var, "port_vol_step": vol, "n_assets": float(len(syms))}
