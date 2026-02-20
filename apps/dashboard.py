# apps/dashboard.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plotly is optional but recommended
try:
    import plotly.graph_objects as go
except Exception:
    go = None

from hyperstat.monitoring.risk_metrics import (
    load_equity_df,
    load_mids_df,
    load_weights_df,
    compute_drawdown,
    compute_equity_metrics,
    compute_var_cvar,
    compute_correlation_matrix,
    compute_portfolio_var_from_weights,
)

Json = Dict[str, Any]


# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(
    page_title="HyperStat — Live Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Utilities
# -----------------------------
def _safe_read_json(path: Path) -> Json:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _human_pct(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * float(x):.{digits}f}%"


def _human_num(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):,.{digits}f}"


def _pick_last_weights(weights_df: pd.DataFrame) -> Dict[str, float]:
    """
    weights_df is long-format: ts,symbol,weight
    This function returns weights at last timestamp.
    """
    if weights_df.empty:
        return {}
    if "ts" not in weights_df.columns:
        return {}
    ts_last = weights_df["ts"].max()
    w = weights_df[weights_df["ts"] == ts_last].copy()
    if w.empty:
        return {}
    out = {}
    for _, r in w.iterrows():
        s = str(r["symbol"])
        try:
            out[s] = float(r["weight"])
        except Exception:
            continue
    return out


def _plot_line(x, y, name: str, height: int = 340):
    if go is None:
        st.line_chart(pd.DataFrame({name: y}, index=x))
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _plot_area(x, y, name: str, height: int = 260):
    if go is None:
        st.area_chart(pd.DataFrame({name: y}, index=x))
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy", name=name))
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _plot_heatmap(mat: pd.DataFrame, title: str, height: int = 520):
    if mat.empty:
        st.info("Pas assez de données pour afficher une matrice.")
        return
    if go is None:
        st.dataframe(mat, use_container_width=True)
        return
    fig = go.Figure(data=go.Heatmap(z=mat.values, x=mat.columns, y=mat.index))
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=35, b=10), title=title)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.title("HyperStat UI")

    st.subheader("Source")
    run_dir = st.text_input("Run dir", value="artifacts/live/default")
    run_path = Path(run_dir)

    st.subheader("Refresh")
    refresh_s = st.number_input("Refresh (sec)", min_value=1, max_value=60, value=3, step=1)
    st.caption("Auto-refresh via meta tag (simple et robuste).")

    st.subheader("Performance window")
    eq_tail = st.number_input("Equity rows (tail)", min_value=1000, max_value=250000, value=50000, step=5000)
    mids_tail = st.number_input("Mids rows (tail)", min_value=5000, max_value=1000000, value=200000, step=50000)
    corr_window = st.number_input("Corr window bars", min_value=200, max_value=50000, value=2000, step=200)

    st.subheader("VaR settings")
    var_window = st.number_input("VaR window (steps)", min_value=500, max_value=200000, value=5000, step=500)
    horizon_steps = st.number_input("VaR horizon (steps)", min_value=1, max_value=200, value=1, step=1)

    st.divider()
    st.subheader("Actions")
    show_raw = st.checkbox("Show raw tables", value=False)
    st.caption("Tu peux ouvrir plusieurs onglets: dashboard + logs.")

# basic auto-refresh without extra deps
st.markdown(f"<meta http-equiv='refresh' content='{int(refresh_s)}'>", unsafe_allow_html=True)


# -----------------------------
# Load data
# -----------------------------
equity_df = load_equity_df(str(run_path), tail=int(eq_tail))
mids_df = load_mids_df(str(run_path), tail=int(mids_tail))
weights_df = load_weights_df(str(run_path), tail=int(mids_tail))  # same tail is fine
snap = _safe_read_json(run_path / "snapshot_latest.json")


# -----------------------------
# Header + run status
# -----------------------------
st.title("HyperStat — Live Dashboard")

status_cols = st.columns(4)
status_cols[0].metric("Run dir", run_dir)

latest_ts = snap.get("ts") if isinstance(snap, dict) else None
status_cols[1].metric("Latest snapshot", str(latest_ts) if latest_ts else "—")

equity_latest = float(snap.get("equity")) if isinstance(snap, dict) and snap.get("equity") is not None else np.nan
status_cols[2].metric("Equity (latest)", _human_num(equity_latest))

n_pos = len((snap.get("positions") or {})) if isinstance(snap, dict) else 0
status_cols[3].metric("#Positions", str(n_pos))

st.divider()


# -----------------------------
# Top metrics from equity curve
# -----------------------------
metrics = compute_equity_metrics(equity_df, window=2000)
var95 = compute_var_cvar(equity_df, alpha=0.05, window=int(var_window), horizon_steps=int(horizon_steps))
var99 = compute_var_cvar(equity_df, alpha=0.01, window=int(var_window), horizon_steps=int(horizon_steps))

mcols = st.columns(6)
mcols[0].metric("Ann. Return", _human_pct(metrics.get("ann_return", np.nan), 1))
mcols[1].metric("Ann. Vol", _human_pct(metrics.get("ann_vol", np.nan), 1))
mcols[2].metric("Sharpe", _human_num(metrics.get("sharpe", np.nan), 2))
mcols[3].metric("Max DD", _human_pct(metrics.get("max_drawdown", np.nan), 1))
mcols[4].metric("VaR 95%", _human_pct(var95.get("var_95", np.nan), 2))
mcols[5].metric("CVaR 95%", _human_pct(var95.get("cvar_95", np.nan), 2))

st.caption(
    "Equity est supposée *net* si tu la lis depuis `accountValue` (fees inclus). "
    "VaR/CVaR ici = historique sur returns de l’equity."
)

st.divider()


# -----------------------------
# Equity + Drawdown
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Equity curve (PnL net)")
    if equity_df.empty or "equity" not in equity_df.columns:
        st.warning("Aucune donnée `equity.csv` (lance le runner live/paper avec SnapshotSink).")
    else:
        _plot_line(equity_df.index, equity_df["equity"].astype(float), "Equity", height=360)

        # If pnl_step/fees_est exist, show mini chart
        optional_cols = [c for c in ["pnl_step", "fees_est", "slip_est"] if c in equity_df.columns]
        if optional_cols:
            st.caption("Breakdown (si le runner le renseigne)")
            small = equity_df[optional_cols].copy()
            # show last 500 points for readability
            if len(small) > 500:
                small = small.tail(500)
            if go is None:
                st.line_chart(small)
            else:
                fig = go.Figure()
                for c in optional_cols:
                    fig.add_trace(go.Scatter(x=small.index, y=small[c], mode="lines", name=c))
                fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Drawdown")
    if not equity_df.empty and "equity" in equity_df.columns:
        dd = compute_drawdown(equity_df["equity"])
        _plot_area(dd.index, dd.values, "Drawdown", height=260)

    st.subheader("VaR / CVaR")
    st.write(f"VaR 99%: **{_human_pct(var99.get('var_99', np.nan), 2)}**")
    st.write(f"CVaR 99%: **{_human_pct(var99.get('cvar_99', np.nan), 2)}**")
    st.write(f"Horizon: **{int(horizon_steps)} step(s)**")
    st.write(f"Window: **{int(var_window)} step(s)**")

st.divider()


# -----------------------------
# Positions table (from snapshot_latest.json)
# -----------------------------
st.subheader("Positions (mark-to-market)")

if not snap:
    st.info("Pas de `snapshot_latest.json`. Le runner live doit appeler `sink.write_latest_snapshot(...)`.")
else:
    mids = snap.get("mids") or {}
    positions = snap.get("positions") or {}
    tgt = snap.get("target_weights") or {}

    rows = []
    eq = equity_latest if np.isfinite(equity_latest) else np.nan

    for sym, p in positions.items():
        qty = float(p.get("qty", 0.0))
        entry = float(p.get("entry_px", 0.0))
        mid = float(mids.get(sym, np.nan)) if sym in mids else np.nan

        notional = qty * mid if np.isfinite(mid) else np.nan
        upnl = (mid - entry) * qty if (np.isfinite(mid) and entry > 0.0) else np.nan

        w_cur = (notional / eq) if (np.isfinite(notional) and np.isfinite(eq) and abs(eq) > 1e-12) else np.nan
        w_tgt = float(tgt.get(sym, 0.0)) if sym in tgt else 0.0

        rows.append(
            {
                "symbol": sym,
                "qty": qty,
                "entry_px": entry,
                "mid": mid,
                "notional": notional,
                "uPnL": upnl,
                "weight_cur": w_cur,
                "weight_target": w_tgt,
            }
        )

    pos_df = pd.DataFrame(rows)
    if not pos_df.empty:
        pos_df = pos_df.sort_values("notional", key=lambda s: s.abs(), ascending=False)
        st.dataframe(pos_df, use_container_width=True)
    else:
        st.info("Snapshot présent, mais aucune position non-nulle.")

st.divider()


# -----------------------------
# Correlation matrix (from mids)
# -----------------------------
st.subheader("Correlation matrix (returns from mids)")
corr = compute_correlation_matrix(mids_df, window=int(corr_window))
_plot_heatmap(corr, title="Corr(returns)", height=520)

st.divider()


# -----------------------------
# Portfolio risk from weights + cov
# -----------------------------
st.subheader("Portfolio variance (weights × cov)")
w_last = _pick_last_weights(weights_df)
port = compute_portfolio_var_from_weights(mids_df, weights_last=w_last, window=int(corr_window))

c1, c2, c3 = st.columns(3)
c1.metric("Port vol/step", _human_num(port.get("port_vol_step", np.nan), 6))
c2.metric("Port var/step", _human_num(port.get("port_var_step", np.nan), 10))
c3.metric("#assets used", str(int(port.get("n_assets", 0))) if port else "0")

st.caption("Vol/step = volatilité sur un pas de temps (bar). Annualisation possible via steps_per_year.")

st.divider()


# -----------------------------
# Raw tables (optional)
# -----------------------------
if show_raw:
    st.subheader("Raw — equity.csv (tail)")
    st.dataframe(equity_df.tail(2000), use_container_width=True)

    st.subheader("Raw — mids.csv (tail)")
    st.dataframe(mids_df.tail(2000), use_container_width=True)

    st.subheader("Raw — weights.csv (tail)")
    st.dataframe(weights_df.tail(2000), use_container_width=True)
