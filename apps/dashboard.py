# apps/dashboard.py
"""
Dashboard Streamlit HyperStat v2 — Monitoring temps réel.

Sections (navigation sidebar) :
    📡 System Status   — Exchange, agents ON/OFF/HALT, stratégies
    🤖 Agents IA       — Score de chaque agent en temps réel
    📈 Portfolio       — Equity curve, PnL, positions mark-to-market
    ⚠️  Risque          — VaR/CVaR, drawdown, corrélation, variance portfolio
    🌊 Régime          — Régime détecté, Q_t, Fear & Greed
    🔮 Prédictions ML  — Probabilités directionnelles par coin
    📊 Equity Curves   — Courbe v1 baseline vs v2 multi-agents
    🔢 Raw Tables      — Tables brutes (optionnel)

Lancement :
    streamlit run apps/dashboard.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Plotly est optionnel mais recommandé
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


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HyperStat v2 — Live Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }

    .badge-active   { background:#1a7a3c; color:#fff; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-halted   { background:#8b1a1a; color:#fff; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-degraded { background:#8b6e1a; color:#fff; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-warming  { background:#2a3a6e; color:#fff; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-off      { background:#3a3a3a; color:#aaa; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }

    .section-header {
        color: #7289da; font-size: 14px; font-weight: 600;
        letter-spacing: 1.5px; text-transform: uppercase;
        border-bottom: 1px solid #2a2d3a; padding-bottom: 6px; margin: 16px 0 12px 0;
    }
    .kill-switch-warning {
        background: #4a1010; border: 2px solid #e74c3c; border-radius: 8px;
        padding: 12px 16px; color: #ff6b6b; font-weight: 700; font-size: 16px; text-align: center;
    }
    .metric-positive { color: #2ecc71; }
    .metric-negative { color: #e74c3c; }
    .metric-neutral  { color: #95a5a6; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES VISUELS
# ─────────────────────────────────────────────────────────────────────────────

def status_badge(status: str) -> str:
    css = {"active": "badge-active", "halted": "badge-halted",
           "degraded": "badge-degraded", "warming_up": "badge-warming", "off": "badge-off"}
    emoji = {"active": "🟢", "halted": "🔴", "degraded": "🟡", "warming_up": "🔵", "off": "⚫"}
    cls = css.get(status, "badge-off")
    ico = emoji.get(status, "⚫")
    return f'<span class="{cls}">{ico} {status.upper().replace("_"," ")}</span>'


def pnl_color(val: float) -> str:
    if val > 0:  return f'<span class="metric-positive">+{val:.2f}$</span>'
    if val < 0:  return f'<span class="metric-negative">{val:.2f}$</span>'
    return f'<span class="metric-neutral">0.00$</span>'


def score_bar(score: float, width: int = 120) -> str:
    pct   = int((score + 1) / 2 * 100)
    color = "#2ecc71" if score > 0.1 else "#e74c3c" if score < -0.1 else "#95a5a6"
    return (f'<div style="background:#1a1d27;border-radius:4px;height:8px;width:{width}px;">'
            f'<div style="background:{color};height:8px;border-radius:4px;width:{pct}%;"></div></div>'
            f'<small style="color:#8892a4">{score:+.3f}</small>')


def _human_pct(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x): return "—"
    return f"{100.0 * float(x):.{digits}f}%"


def _human_num(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x): return "—"
    return f"{float(x):,.{digits}f}"


def _safe_read_json(path: Path) -> Json:
    if not path.exists(): return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_last_weights(weights_df: pd.DataFrame) -> Dict[str, float]:
    if weights_df.empty or "ts" not in weights_df.columns: return {}
    ts_last = weights_df["ts"].max()
    w = weights_df[weights_df["ts"] == ts_last]
    out: Dict[str, float] = {}
    for _, r in w.iterrows():
        try:
            out[str(r["symbol"])] = float(r["weight"])
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
        st.info("Pas assez de données pour afficher la matrice.")
        return
    if go is None:
        st.dataframe(mat, use_container_width=True)
        return
    fig = go.Figure(data=go.Heatmap(z=mat.values, x=mat.columns, y=mat.index))
    fig.update_layout(template="plotly_dark", height=height,
                      margin=dict(l=10, r=10, t=35, b=10), title=title)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES AGENTS (state.json) — avec demo fallback
# ─────────────────────────────────────────────────────────────────────────────

def _demo_state() -> Json:
    """Données de démonstration si aucun live runner n'est actif."""
    return {
        "exchange": {
            "connected": True, "mode": "paper",
            "latency_ms": 45, "rate_limit_pct": 12,
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        },
        "supervisor": {
            "scale_factor": 0.85, "regime": "mean_reverting",
            "kill_switch": False, "composite_score": 0.42, "confidence": 0.73,
            "reason": "Régime mean_reverting — exposition nominale",
        },
        "agents": {
            "TechnicalAgent" : {"status": "active",  "score": 0.38, "confidence": 0.80, "ic_recent": 0.042},
            "SentimentAgent" : {"status": "active",  "score": 0.25, "confidence": 0.65, "ic_recent": 0.028,
                                "fg_raw": 52, "fg_label": "Neutral"},
            "PredictionAgent": {"status": "active",  "score": 0.51, "confidence": 0.71, "ic_recent": 0.055},
            "RegimeAgent"    : {"status": "active",  "score": 0.60, "confidence": 0.90, "ic_recent": 0.038,
                                "current_regime": "mean_reverting", "current_qt": 1.0, "fg_signal": "neutral"},
        },
        "strategies": {
            "stat_arb_bucket_A": {"status": "active", "pnl": 124.5,  "n_positions": 4, "gross_exposure": 0.82},
            "stat_arb_bucket_B": {"status": "active", "pnl": -23.1,  "n_positions": 3, "gross_exposure": 0.61},
            "funding_carry"    : {"status": "halted", "pnl": 0.0,    "n_positions": 0,
                                  "halt_reason": "Funding instable — DELIST suspicion"},
        },
        "predictions": {
            "BTC": 0.62, "ETH": 0.58, "SOL": 0.71,
            "AVAX": 0.43, "LINK": 0.55, "DOT": 0.38,
        },
    }


@st.cache_data(ttl=5)
def load_state(run_dir: str) -> Json:
    p = Path(run_dir) / "state.json"
    if p.exists():
        data = _safe_read_json(p)
        return data if data else _demo_state()
    return _demo_state()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ HyperStat v2")
    st.divider()

    st.subheader("Source")
    run_dir = st.text_input("Run dir", value="artifacts/live/default")
    run_path = Path(run_dir)

    st.subheader("Refresh")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_s = st.number_input("Intervalle (sec)", min_value=5, max_value=120, value=10, step=5,
                                disabled=not auto_refresh)

    st.subheader("Performance window")
    eq_tail    = st.number_input("Equity rows (tail)",  min_value=1000,  max_value=250000,  value=50000,  step=5000)
    mids_tail  = st.number_input("Mids rows (tail)",    min_value=5000,  max_value=1000000, value=200000, step=50000)
    corr_window= st.number_input("Corr window bars",    min_value=200,   max_value=50000,   value=2000,   step=200)

    st.subheader("VaR settings")
    var_window    = st.number_input("VaR window (steps)",  min_value=500,  max_value=200000, value=5000, step=500)
    horizon_steps = st.number_input("VaR horizon (steps)", min_value=1,    max_value=200,    value=1,    step=1)

    st.divider()
    st.subheader("Navigation")
    section = st.radio("", [
        "📡 System Status",
        "🤖 Agents IA",
        "📈 Portfolio",
        "⚠️ Risque",
        "🌊 Régime",
        "🔮 Prédictions ML",
        "📊 Equity Curves",
        "🔢 Raw Tables",
    ], label_visibility="collapsed")

    st.divider()
    show_raw = st.checkbox("Show raw tables", value=False)
    if st.button("🔴 FLAT ALL (Emergency)", type="primary"):
        st.error("⚠️ Flatten d'urgence — non implémenté en démo")

    st.markdown(f"<small style='color:#8892a4'>UTC: {datetime.now(timezone.utc).strftime('%H:%M:%S')}</small>",
                unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

state      = load_state(run_dir)
equity_df  = load_equity_df(str(run_path), tail=int(eq_tail))
mids_df    = load_mids_df(str(run_path), tail=int(mids_tail))
weights_df = load_weights_df(str(run_path), tail=int(mids_tail))
snap       = _safe_read_json(run_path / "snapshot_latest.json")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

col_title, col_exchange, col_mode, col_ts = st.columns([3, 2, 2, 2])

with col_title:
    st.markdown("## 📊 HyperStat v2 — Live Dashboard")

exch = state.get("exchange", {})
with col_exchange:
    conn = exch.get("connected", False)
    st.markdown(f"**Exchange** &nbsp; {status_badge('active' if conn else 'halted')}",
                unsafe_allow_html=True)
    if conn:
        st.caption(f"Latence: {exch.get('latency_ms','?')}ms | Rate limit: {exch.get('rate_limit_pct','?')}%")

with col_mode:
    mode = exch.get("mode", "unknown").upper()
    color = "#f39c12" if mode == "PAPER" else "#2ecc71" if mode == "LIVE" else "#95a5a6"
    st.markdown(f"**Mode** &nbsp; <span style='color:{color};font-weight:700'>{mode}</span>",
                unsafe_allow_html=True)

with col_ts:
    latest_ts = snap.get("ts") if isinstance(snap, dict) else None
    st.markdown(f"**Snapshot** &nbsp; `{str(latest_ts)[:19] if latest_ts else '—'}`")

# Kill-switch warning global
sup = state.get("supervisor", {})
if sup.get("kill_switch"):
    st.markdown(
        '<div class="kill-switch-warning">🚨 KILL-SWITCH ACTIVÉ — Toutes positions fermées</div>',
        unsafe_allow_html=True
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION : SYSTEM STATUS
# ─────────────────────────────────────────────────────────────────────────────

if "System Status" in section:
    st.markdown('<div class="section-header">📡 System Status — Exchange & Stratégies</div>',
                unsafe_allow_html=True)

    # Exchange + Supervisor
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        eq_latest = float(snap.get("equity", np.nan)) if isinstance(snap, dict) and snap.get("equity") is not None else np.nan
        st.metric("Run dir", run_dir)
    with c2:
        st.metric("Equity (latest)", _human_num(eq_latest))
    with c3:
        st.metric("Scale Factor", f"{sup.get('scale_factor', 0):.0%}")
    with c4:
        st.metric("Composite Score", f"{sup.get('composite_score', 0):+.3f}")

    st.caption(f"🧠 {sup.get('reason', '')}")
    st.divider()

    # Stratégies
    st.markdown("#### Stratégies")
    strats = state.get("strategies", {})
    for strat_name, info in strats.items():
        status = info.get("status", "off")
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 3])
        with c1:
            st.markdown(f"**{strat_name}** &nbsp; {status_badge(status)}", unsafe_allow_html=True)
        with c2:
            st.markdown(f"PnL: {pnl_color(info.get('pnl', 0.0))}", unsafe_allow_html=True)
        with c3:
            st.metric("Positions", info.get("n_positions", 0), label_visibility="collapsed")
        with c4:
            if status == "active":
                st.metric("Gross Exp", f"{info.get('gross_exposure', 0):.0%}", label_visibility="collapsed")
        with c5:
            if status == "halted": st.caption(f"🔴 {info.get('halt_reason', '')}")
            elif status == "off":  st.caption(f"⚫ {info.get('off_reason', '')}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION : AGENTS IA
# ─────────────────────────────────────────────────────────────────────────────

elif "Agents IA" in section:
    st.markdown('<div class="section-header">🤖 Agents IA — Scores & Statuts</div>',
                unsafe_allow_html=True)
    agents = state.get("agents", {})

    agent_list = list(agents.items())
    for i in range(0, len(agent_list), 2):
        cols = st.columns(2)
        for j, (ag_name, ag_info) in enumerate(agent_list[i:i+2]):
            with cols[j]:
                with st.container(border=True):
                    status = ag_info.get("status", "off")
                    st.markdown(f"**{ag_name}** &nbsp; {status_badge(status)}", unsafe_allow_html=True)
                    score = ag_info.get("score", 0.0)
                    conf  = ag_info.get("confidence", 0.0)
                    ic    = ag_info.get("ic_recent", 0.0)
                    st.markdown(f"Score: {score_bar(score)}", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("Confidence", f"{conf:.0%}")
                    with m2: st.metric("IC récent",  f"{ic:.4f}")
                    if ag_name == "SentimentAgent":
                        with m3: st.metric("F&G", ag_info.get("fg_raw", "?"))
                        st.caption(f"📰 {ag_info.get('fg_label', '?')}")
                    elif ag_name == "RegimeAgent":
                        with m3: st.metric("Q_t", f"{ag_info.get('current_qt', 0):.1f}")
                        st.caption(f"🌊 Régime: **{ag_info.get('current_regime', 'unknown')}**")

    st.divider()
    rows = [{"Agent": k, "Statut": v.get("status","off").upper(),
             "Score": f"{v.get('score',0):+.3f}",
             "Confidence": f"{v.get('confidence',0):.0%}",
             "IC récent": f"{v.get('ic_recent',0):.4f}"}
            for k, v in agents.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION : PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

elif "Portfolio" in section:
    st.markdown('<div class="section-header">📈 Portfolio — Equity & Positions</div>',
                unsafe_allow_html=True)

    # ── Top metrics depuis equity_df (données réelles) ──
    metrics = compute_equity_metrics(equity_df, window=2000)
    var95   = compute_var_cvar(equity_df, alpha=0.05, window=int(var_window), horizon_steps=int(horizon_steps))

    mcols = st.columns(5)
    mcols[0].metric("Ann. Return", _human_pct(metrics.get("ann_return", np.nan), 1))
    mcols[1].metric("Ann. Vol",    _human_pct(metrics.get("ann_vol", np.nan), 1))
    mcols[2].metric("Sharpe",      _human_num(metrics.get("sharpe", np.nan), 2))
    mcols[3].metric("Max DD",      _human_pct(metrics.get("max_drawdown", np.nan), 1))
    mcols[4].metric("VaR 95%",     _human_pct(var95.get("var_95", np.nan), 2))

    st.divider()

    # ── Equity curve ──
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Equity curve (PnL net)")
        if equity_df.empty or "equity" not in equity_df.columns:
            st.warning("Aucune donnée `equity.csv`. Lance le runner live/paper.")
        else:
            _plot_line(equity_df.index, equity_df["equity"].astype(float), "Equity", height=340)
            opt_cols = [c for c in ["pnl_step", "fees_est", "slip_est"] if c in equity_df.columns]
            if opt_cols:
                st.caption("Breakdown PnL/fees/slippage")
                small = equity_df[opt_cols].tail(500)
                if go is None:
                    st.line_chart(small)
                else:
                    fig = go.Figure()
                    for c in opt_cols:
                        fig.add_trace(go.Scatter(x=small.index, y=small[c], mode="lines", name=c))
                    fig.update_layout(template="plotly_dark", height=240,
                                      margin=dict(l=10, r=10, t=25, b=10))
                    st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Drawdown")
        if not equity_df.empty and "equity" in equity_df.columns:
            dd = compute_drawdown(equity_df["equity"])
            _plot_area(dd.index, dd.values, "Drawdown", height=240)
        st.subheader("VaR / CVaR")
        var99 = compute_var_cvar(equity_df, alpha=0.01, window=int(var_window), horizon_steps=int(horizon_steps))
        st.write(f"VaR 99%:  **{_human_pct(var99.get('var_99', np.nan), 2)}**")
        st.write(f"CVaR 99%: **{_human_pct(var99.get('cvar_99', np.nan), 2)}**")
        st.write(f"Horizon:  **{int(horizon_steps)} step(s)**")

    st.divider()

    # ── Positions mark-to-market (depuis snapshot_latest.json) ──
    st.subheader("Positions (mark-to-market)")
    if not snap:
        st.info("Pas de `snapshot_latest.json`. Le runner live doit écrire ce fichier.")
    else:
        mids      = snap.get("mids") or {}
        positions = snap.get("positions") or {}
        tgt       = snap.get("target_weights") or {}
        eq_latest = float(snap.get("equity", np.nan)) if snap.get("equity") is not None else np.nan

        rows = []
        for sym, p in positions.items():
            qty    = float(p.get("qty", 0.0))
            entry  = float(p.get("entry_px", 0.0))
            mid    = float(mids.get(sym, np.nan)) if sym in mids else np.nan
            notional = qty * mid if np.isfinite(mid) else np.nan
            upnl     = (mid - entry) * qty if (np.isfinite(mid) and entry > 0.0) else np.nan
            w_cur    = (notional / eq_latest) if (np.isfinite(notional) and np.isfinite(eq_latest) and abs(eq_latest) > 1e-12) else np.nan
            w_tgt    = float(tgt.get(sym, 0.0)) if sym in tgt else 0.0
            rows.append({"symbol": sym, "qty": qty, "entry_px": entry, "mid": mid,
                         "notional": notional, "uPnL": upnl,
                         "weight_cur": w_cur, "weight_target": w_tgt})

        pos_df = pd.DataFrame(rows)
        if not pos_df.empty:
            pos_df = pos_df.sort_values("notional", key=lambda s: s.abs(), ascending=False)
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("Snapshot présent, aucune position non-nulle.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION : RISQUE
# ─────────────────────────────────────────────────────────────────────────────

elif "Risque" in section:
    st.markdown('<div class="section-header">⚠️ Risk Management</div>', unsafe_allow_html=True)

    metrics = compute_equity_metrics(equity_df, window=2000)
    var95   = compute_var_cvar(equity_df, alpha=0.05, window=int(var_window), horizon_steps=int(horizon_steps))
    var99   = compute_var_cvar(equity_df, alpha=0.01, window=int(var_window), horizon_steps=int(horizon_steps))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ann. Return", _human_pct(metrics.get("ann_return", np.nan), 1))
    c2.metric("Ann. Vol",    _human_pct(metrics.get("ann_vol", np.nan), 1))
    c3.metric("Sharpe",      _human_num(metrics.get("sharpe", np.nan), 2))
    c4.metric("Max DD",      _human_pct(metrics.get("max_drawdown", np.nan), 1))
    c5.metric("VaR 95%",     _human_pct(var95.get("var_95", np.nan), 2))
    c6.metric("CVaR 95%",    _human_pct(var95.get("cvar_95", np.nan), 2))

    st.caption("VaR/CVaR = historique sur returns equity. Horizon: "
               f"{int(horizon_steps)} step(s) | Fenêtre: {int(var_window)} steps.")

    st.divider()

    # Barre de progression drawdown vers kill-switch
    st.markdown("#### Drawdown courant vs Kill-Switch")
    if not equity_df.empty and "equity" in equity_df.columns:
        eq_s = equity_df["equity"].dropna()
        if len(eq_s) > 0:
            peak    = eq_s.cummax()
            dd_cur  = float(((eq_s - peak) / peak).iloc[-1] * 100)
            max_dd  = float(metrics.get("max_drawdown", 0) * 100)
            ks_thr  = 3.0  # seuil kill-switch (3%)
            progress= min(1.0, abs(dd_cur) / ks_thr)
            st.progress(progress, text=f"DD courant: {dd_cur:.2f}% | Max: {max_dd:.2f}% | Kill-switch à −{ks_thr:.0f}%")

    st.divider()

    # Corrélation
    st.subheader("Correlation matrix (returns from mids)")
    corr = compute_correlation_matrix(mids_df, window=int(corr_window))
    _plot_heatmap(corr, title="Corr(returns)", height=500)

    st.divider()

    # Portfolio variance
    st.subheader("Portfolio variance (weights × cov)")
    w_last = _pick_last_weights(weights_df)
    port   = compute_portfolio_var_from_weights(mids_df, weights_last=w_last, window=int(corr_window))
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Port vol/step", _human_num(port.get("port_vol_step", np.nan), 6))
    pc2.metric("Port var/step", _human_num(port.get("port_var_step", np.nan), 10))
    pc3.metric("#assets used",  str(int(port.get("n_assets", 0))))
    st.caption("Vol/step = volatilité sur un pas de temps (bar).")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION : RÉGIME
# ─────────────────────────────────────────────────────────────────────────────

elif "Régime" in section:
    st.markdown('<div class="section-header">🌊 Régime de Marché</div>', unsafe_allow_html=True)

    ag_regime = state.get("agents", {}).get("RegimeAgent", {})
    ag_sent   = state.get("agents", {}).get("SentimentAgent", {})
    regime    = ag_regime.get("current_regime", "unknown")
    qt        = ag_regime.get("current_qt", 0.5)

    regime_desc = {
        "mean_reverting" : ("🟢 Mean Reverting",  "Conditions idéales pour stat-arb. Q_t = 1.0"),
        "carry_favorable": ("🟢 Carry Favorable",  "Funding stable et élevé. Boost overlay carry."),
        "trending"       : ("🟡 Trending",          "Momentum fort. Exposition réduite. Q_t = 0.3"),
        "high_vol"       : ("🔴 High Vol",          "Vol BTC > p90. Stratégie stoppée. Q_t = 0.0"),
        "crisis"         : ("🚨 CRISIS",            "Liquidations massives. Kill-switch activé."),
        "unknown"        : ("⚪ Unknown",            "Données insuffisantes pour classifier."),
    }
    label, desc = regime_desc.get(regime, ("⚪ Unknown", ""))
    st.markdown(f"## {label}")
    st.info(desc)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Q_t appliqué", f"{qt:.1f}")
    with c2: st.metric("Fear & Greed",  f"{ag_sent.get('fg_raw', 50)} / 100")
    with c3: st.metric("Vol Score",     f"{ag_regime.get('vol_score', 0):.3f}")
    with c4: st.metric("Momentum",      f"{ag_regime.get('momentum_score', 0):.3f}")

    st.divider()
    st.markdown("#### Composantes du score régime")
    df_comp = pd.DataFrame({
        "Composante"  : ["Volatilité", "Momentum", "Liquidations", "Funding"],
        "Score"       : [ag_regime.get("vol_score", 0), ag_regime.get("momentum_score", 0),
                         ag_regime.get("liq_score", 0), ag_regime.get("funding_score", 0)],
        "Poids"       : [0.35, 0.40, 0.15, 0.10],
    })
    df_comp["Contribution"] = df_comp["Score"] * df_comp["Poids"]
    st.dataframe(df_comp.round(4), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION : PRÉDICTIONS ML
# ─────────────────────────────────────────────────────────────────────────────

elif "Prédictions" in section:
    st.markdown('<div class="section-header">🔮 Prédictions ML — PredictionAgent</div>',
                unsafe_allow_html=True)

    probas = state.get("predictions", {})
    if probas:
        st.markdown("#### Probabilités directionnelles (horizon 1h)")
        st.caption("P > 0.55 → haussier | P < 0.45 → baissier")
        rows = []
        for sym, p in sorted(probas.items(), key=lambda x: -x[1]):
            direction = "🟢 BULLISH" if p > 0.55 else "🔴 BEARISH" if p < 0.45 else "⚪ NEUTRAL"
            rows.append({"Symbole": sym, "P(hausse)": f"{p:.3f}",
                         "Score": f"{2*(p-0.5):+.3f}", "Signal": direction})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Pas de prédictions disponibles dans state.json.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION : EQUITY CURVES (v1 vs v2)
# ─────────────────────────────────────────────────────────────────────────────

elif "Equity Curves" in section:
    st.markdown('<div class="section-header">📊 Equity Curves — v1 Baseline vs v2 Multi-Agents</div>',
                unsafe_allow_html=True)

    if not equity_df.empty and "equity" in equity_df.columns:
        df_plot = equity_df[["equity"]].copy()
        # Normaliser à base 100
        df_plot["equity_norm"] = df_plot["equity"] / df_plot["equity"].iloc[0] * 100

        _plot_line(df_plot.index, df_plot["equity_norm"], "Equity (base 100)", height=400)

        # Métriques
        metrics = compute_equity_metrics(equity_df, window=2000)
        eq_s    = equity_df["equity"].dropna()
        total_ret = (eq_s.iloc[-1] / eq_s.iloc[0] - 1) * 100 if len(eq_s) > 1 else np.nan

        with st.container(border=True):
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Retour total",   f"{total_ret:+.2f}%" if np.isfinite(total_ret) else "—")
            cc2.metric("Sharpe",         _human_num(metrics.get("sharpe", np.nan), 2))
            cc3.metric("Max Drawdown",   _human_pct(metrics.get("max_drawdown", np.nan), 2))
            cc4.metric("Ann. Vol",       _human_pct(metrics.get("ann_vol", np.nan), 2))
    else:
        st.info("Données equity non disponibles. Lance le backtest ou le paper trading d'abord.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION : RAW TABLES
# ─────────────────────────────────────────────────────────────────────────────

elif "Raw Tables" in section or show_raw:
    st.markdown('<div class="section-header">🔢 Raw Tables</div>', unsafe_allow_html=True)
    st.subheader("equity.csv (tail)")
    st.dataframe(equity_df.tail(2000), use_container_width=True)
    st.subheader("mids.csv (tail)")
    st.dataframe(mids_df.tail(2000), use_container_width=True)
    st.subheader("weights.csv (tail)")
    st.dataframe(weights_df.tail(2000), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Show raw si checkbox cochée (depuis n'importe quelle section)
# ─────────────────────────────────────────────────────────────────────────────
if show_raw and "Raw Tables" not in section:
    st.divider()
    st.subheader("Raw — equity.csv (tail)")
    st.dataframe(equity_df.tail(2000), use_container_width=True)
    st.subheader("Raw — mids.csv (tail)")
    st.dataframe(mids_df.tail(2000), use_container_width=True)
    st.subheader("Raw — weights.csv (tail)")
    st.dataframe(weights_df.tail(2000), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH (st.rerun = delta WebSocket, pas de rechargement navigateur)
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(int(refresh_s))
    st.rerun()