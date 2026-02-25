# apps/dashboard.py
"""
Dashboard HyperStat v2 — Monitoring temps réel (Dash + thème sombre CYBORG)
===========================================================================
Sections (navigation sidebar) :
    📡 System Status   — Exchange, agents ON/OFF/HALT, stratégies
    🤖 Agents IA       — Score de chaque agent en temps réel
    📈 Portfolio       — Equity curve, PnL, positions mark-to-market
    ⚠️  Risque          — VaR/CVaR, drawdown, corrélation, variance portfolio
    🌊 Régime          — Régime détecté, Q_t, Fear & Greed
    🔮 Prédictions ML  — Probabilités directionnelles par coin
    📊 Equity Curves   — Courbe equity normalisée
    🔢 Raw Tables      — Tables brutes

Lancement :
    python apps/dashboard.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, no_update
import dash_bootstrap_components as dbc

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:
    go = None
    _PLOTLY = False

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

_DARK_BG  = "#0d0d1a"
_CARD_BG  = "#111122"
_PLOT_BG  = "#0f0f23"
_SIDEBAR_W = "260px"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_read_json(path: Path) -> Json:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _human_pct(x, digits: int = 2) -> str:
    if x is None or not np.isfinite(float(x)): return "—"
    return f"{100.0 * float(x):.{digits}f}%"


def _human_num(x, digits: int = 2) -> str:
    if x is None or not np.isfinite(float(x)): return "—"
    return f"{float(x):,.{digits}f}"


def _pick_last_weights(weights_df: pd.DataFrame) -> Dict[str, float]:
    if weights_df.empty or "ts" not in weights_df.columns:
        return {}
    ts_last = weights_df["ts"].max()
    w = weights_df[weights_df["ts"] == ts_last]
    out: Dict[str, float] = {}
    for _, r in w.iterrows():
        try:
            out[str(r["symbol"])] = float(r["weight"])
        except Exception:
            continue
    return out


def _demo_state() -> Json:
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
                                "current_regime": "mean_reverting", "current_qt": 1.0,
                                "fg_signal": "neutral", "vol_score": 0.2,
                                "momentum_score": 0.4, "liq_score": 0.1, "funding_score": 0.3},
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


def _load_state(run_dir: str) -> Json:
    p = Path(run_dir) / "state.json"
    if p.exists():
        data = _safe_read_json(p)
        return data if data else _demo_state()
    return _demo_state()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _empty_fig(height=340, msg="Données non disponibles"):
    if not _PLOTLY:
        return {}
    return go.Figure().update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
        annotations=[{"text": msg, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"color": "#666", "size": 13}}],
        margin=dict(l=10, r=10, t=30, b=10),
    )


def _fig_line(x, y, name: str, height: int = 340, color: str = "#2ecc71"):
    if not _PLOTLY:
        return _empty_fig(height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                             line=dict(color=color, width=2)))
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
    )
    return fig


def _fig_area(x, y, name: str, height: int = 260, color: str = "#e74c3c"):
    if not _PLOTLY:
        return _empty_fig(height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy", name=name,
                             line=dict(color=color, width=1.5),
                             fillcolor="rgba(231,76,60,0.15)"))
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
    )
    return fig


def _fig_heatmap(mat: pd.DataFrame, title: str, height: int = 520):
    if not _PLOTLY or mat.empty:
        return _empty_fig(height, "Pas assez de données")
    fig = go.Figure(data=go.Heatmap(
        z=mat.values, x=mat.columns, y=mat.index,
        colorscale="RdYlGn", zmid=0,
    ))
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        title=title,
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
    )
    return fig


def _fig_score_bar(score: float, width: int = 140):
    pct   = int((score + 1) / 2 * 100)
    color = "#2ecc71" if score > 0.1 else "#e74c3c" if score < -0.1 else "#95a5a6"
    return html.Div([
        html.Div(style={
            "background": "#1a1d27", "borderRadius": "4px",
            "height": "8px", "width": f"{width}px",
        }, children=[
            html.Div(style={
                "background": color, "height": "8px",
                "borderRadius": "4px", "width": f"{pct}%",
            }),
        ]),
        html.Small(f"{score:+.3f}", style={"color": "#8892a4", "fontSize": "11px"}),
    ])


def _status_badge(status: str):
    _map = {
        "active"    : ("#1a7a3c", "🟢 ACTIVE"),
        "halted"    : ("#7a1a1a", "🔴 HALTED"),
        "degraded"  : ("#7a5a0a", "🟡 DEGRADED"),
        "warming_up": ("#2a3a6e", "🔵 WARMING UP"),
        "off"       : ("#2a2a2a", "⚫ OFF"),
    }
    bg, label = _map.get(status, ("#2a2a2a", status.upper()))
    return html.Span(label, style={
        "background": bg, "color": "#fff",
        "padding": "2px 8px", "borderRadius": "10px",
        "fontSize": "10px", "fontWeight": "700",
    })


def _kpi(label: str, value: str, color: str = "#ddd") -> dbc.Col:
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, style={"fontSize": "10px", "color": "#666"}),
        html.Div(value, style={"fontSize": "14px", "fontWeight": "700", "color": color}),
    ], className="p-2"), style={"background": _CARD_BG, "border": "1px solid #2a2d3a"}))


# ─────────────────────────────────────────────────────────────────────────────
# DASH APP
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="HyperStat v2 — Monitoring",
)

_SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": _SIDEBAR_W, "overflowY": "auto", "padding": "15px",
    "backgroundColor": "#0a0a14",
    "borderRight": "1px solid #1e1e30",
    "zIndex": 1000,
}
_CONTENT_STYLE = {
    "marginLeft": f"calc({_SIDEBAR_W} + 10px)",
    "padding": "18px 22px",
    "minHeight": "100vh",
    "backgroundColor": _DARK_BG,
}

_SECTIONS = [
    "📡 System Status",
    "🤖 Agents IA",
    "📈 Portfolio",
    "⚠️ Risque",
    "🌊 Régime",
    "🔮 Prédictions ML",
    "📊 Equity Curves",
    "🔢 Raw Tables",
]

app.layout = html.Div([
    dcc.Interval(id="interval", interval=10_000, n_intervals=0),

    # Sidebar
    html.Div([
        html.H5("📊 HyperStat v2", style={"color": "#7289da", "fontSize": "14px"}),
        html.Hr(style={"borderColor": "#1e1e30"}),

        html.Div("SOURCE", style={"color": "#555", "fontSize": "10px",
                                   "letterSpacing": "1px", "marginBottom": "4px"}),
        dbc.Input(id="input-run-dir", type="text",
                  value="artifacts/live/default", size="sm",
                  style={"backgroundColor": "#111", "color": "#ddd",
                         "border": "1px solid #2a2d3a", "fontSize": "11px"},
                  className="mb-2"),

        html.Div("REFRESH", style={"color": "#555", "fontSize": "10px",
                                    "letterSpacing": "1px", "marginBottom": "4px"}),
        dbc.Row([
            dbc.Col(dbc.Switch(id="toggle-refresh", value=True,
                               label=html.Span("Auto", style={"fontSize": "11px"})), width=5),
            dbc.Col(dbc.Input(id="input-refresh-s", type="number", value=10, min=5, max=120,
                              size="sm",
                              style={"backgroundColor": "#111", "color": "#ddd",
                                     "border": "1px solid #2a2d3a"}), width=7),
        ], className="mb-3 align-items-center g-1"),

        html.Div("FENÊTRES", style={"color": "#555", "fontSize": "10px",
                                     "letterSpacing": "1px", "marginBottom": "4px"}),
        html.Div("Equity rows (tail)", style={"fontSize": "10px", "color": "#888"}),
        dbc.Input(id="input-eq-tail", type="number", value=50000, min=1000, step=5000,
                  size="sm", style={"backgroundColor": "#111", "color": "#ddd",
                                     "border": "1px solid #2a2d3a"}, className="mb-1"),
        html.Div("Corr window (bars)", style={"fontSize": "10px", "color": "#888"}),
        dbc.Input(id="input-corr-window", type="number", value=2000, min=200, step=200,
                  size="sm", style={"backgroundColor": "#111", "color": "#ddd",
                                     "border": "1px solid #2a2d3a"}, className="mb-1"),
        html.Div("VaR window (steps)", style={"fontSize": "10px", "color": "#888"}),
        dbc.Input(id="input-var-window", type="number", value=5000, min=500, step=500,
                  size="sm", style={"backgroundColor": "#111", "color": "#ddd",
                                     "border": "1px solid #2a2d3a"}, className="mb-1"),
        html.Div("VaR horizon (steps)", style={"fontSize": "10px", "color": "#888"}),
        dbc.Input(id="input-var-horizon", type="number", value=1, min=1, max=200,
                  size="sm", style={"backgroundColor": "#111", "color": "#ddd",
                                     "border": "1px solid #2a2d3a"}, className="mb-3"),

        html.Hr(style={"borderColor": "#1e1e30"}),
        html.Div("NAVIGATION", style={"color": "#555", "fontSize": "10px",
                                       "letterSpacing": "1px", "marginBottom": "4px"}),
        dbc.RadioItems(
            id="input-section",
            options=[{"label": s, "value": s} for s in _SECTIONS],
            value=_SECTIONS[0],
            labelStyle={"display": "block", "fontSize": "11px",
                         "padding": "3px 0", "cursor": "pointer"},
            className="mb-3",
        ),

        html.Hr(style={"borderColor": "#1e1e30"}),
        dbc.Checklist(
            id="toggle-raw",
            options=[{"label": html.Span("Afficher tables brutes",
                                          style={"fontSize": "11px"}),
                      "value": "show"}],
            value=[], className="mb-2",
        ),
        dbc.Button("🚨 FLAT ALL (Emergency)", id="btn-flat-all",
                   color="danger", size="sm", className="w-100", n_clicks=0),

        html.Hr(style={"borderColor": "#1e1e30"}),
        html.Small(id="sidebar-clock", style={"color": "#555", "fontSize": "10px"}),
    ], style=_SIDEBAR_STYLE),

    # Main content
    html.Div([
        # Header
        dbc.Row([
            dbc.Col(html.H4("📊 HyperStat v2 — Monitoring",
                            style={"color": "#7289da", "fontSize": "17px", "margin": 0}), width=5),
            dbc.Col(html.Div(id="header-exchange"), width=4),
            dbc.Col(html.Div(id="header-snapshot"), width=3),
        ], className="mb-2 align-items-center"),

        html.Div(id="kill-switch-banner", className="mb-2"),
        html.Hr(style={"borderColor": "#1e1e30"}),

        # Main section content
        html.Div(id="main-content"),

        # Raw tables (always visible if toggled)
        html.Div(id="raw-tables-section"),
    ], style=_CONTENT_STYLE),
], style={"backgroundColor": _DARK_BG, "minHeight": "100vh", "fontFamily": "monospace"})


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("interval", "disabled"),
    Output("interval", "interval"),
    Output("sidebar-clock", "children"),
    Input("toggle-refresh", "value"),
    Input("input-refresh-s", "value"),
    Input("interval", "n_intervals"),
)
def update_interval(auto, refresh_s, _n):
    ts = datetime.now(timezone.utc).strftime("UTC: %H:%M:%S")
    return (not bool(auto), max(5, int(refresh_s or 10)) * 1000, ts)


@app.callback(
    Output("header-exchange",    "children"),
    Output("header-snapshot",    "children"),
    Output("kill-switch-banner", "children"),
    Output("main-content",       "children"),
    Output("raw-tables-section", "children"),
    Input("interval",          "n_intervals"),
    Input("input-section",     "value"),
    Input("input-run-dir",     "value"),
    Input("input-eq-tail",     "value"),
    Input("input-corr-window", "value"),
    Input("input-var-window",  "value"),
    Input("input-var-horizon", "value"),
    Input("toggle-raw",        "value"),
)
def update_dashboard(_n, section, run_dir, eq_tail, corr_window,
                     var_window, var_horizon, show_raw):
    run_dir     = run_dir or "artifacts/live/default"
    eq_tail     = int(eq_tail or 50000)
    corr_window = int(corr_window or 2000)
    var_window  = int(var_window or 5000)
    var_horizon = int(var_horizon or 1)
    run_path    = Path(run_dir)

    # Load data
    state      = _load_state(run_dir)
    equity_df  = load_equity_df(run_dir, tail=eq_tail)
    mids_df    = load_mids_df(run_dir, tail=eq_tail * 4)
    weights_df = load_weights_df(run_dir, tail=eq_tail * 4)
    snap       = _safe_read_json(run_path / "snapshot_latest.json")

    exch = state.get("exchange", {})
    sup  = state.get("supervisor", {})

    # ── Header Exchange ───────────────────────────────────────────────────────
    conn  = exch.get("connected", False)
    mode  = exch.get("mode", "unknown").upper()
    mode_color = "#f39c12" if mode == "PAPER" else "#2ecc71" if mode == "LIVE" else "#95a5a6"
    hdr_exch = dbc.Row([
        dbc.Col([
            _status_badge("active" if conn else "halted"),
            html.Small(
                f"  {exch.get('latency_ms', '?')}ms | {exch.get('rate_limit_pct', '?')}%",
                style={"color": "#666", "fontSize": "10px"},
            ),
        ], width=7),
        dbc.Col(
            html.Span(mode, style={"color": mode_color, "fontWeight": "700", "fontSize": "13px"}),
            width=5,
        ),
    ], className="align-items-center")

    # ── Header Snapshot ───────────────────────────────────────────────────────
    latest_ts = snap.get("ts") if isinstance(snap, dict) else None
    hdr_snap  = html.Small(
        f"Snapshot: {str(latest_ts)[:19] if latest_ts else '—'}",
        style={"color": "#555", "fontSize": "10px"},
    )

    # ── Kill-switch banner ────────────────────────────────────────────────────
    ks_banner = html.Div()
    if sup.get("kill_switch"):
        ks_banner = dbc.Alert(
            "🚨 KILL-SWITCH ACTIVÉ — Toutes positions fermées",
            color="danger", className="fw-bold text-center p-2",
        )

    # ── Emergency flat handler ─────────────────────────────────────────────── (no-op in demo)

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION : System Status
    # ──────────────────────────────────────────────────────────────────────────
    if "System Status" in (section or ""):
        eq_latest = float(snap.get("equity", float("nan"))) if isinstance(snap, dict) and snap.get("equity") else float("nan")
        strats    = state.get("strategies", {})

        strat_rows = []
        for sn, info in strats.items():
            s = info.get("status", "off")
            pnl = info.get("pnl", 0.0)
            pnl_color = "#2ecc71" if pnl > 0 else "#e74c3c" if pnl < 0 else "#888"
            strat_rows.append(html.Tr([
                html.Td([_status_badge(s), html.Span(f"  {sn}", style={"fontSize": "12px", "marginLeft": "6px"})]),
                html.Td(html.Span(f"{pnl:+.2f}$", style={"color": pnl_color, "fontSize": "12px"})),
                html.Td(str(info.get("n_positions", 0)), style={"fontSize": "12px"}),
                html.Td(f"{info.get('gross_exposure', 0):.0%}" if s == "active" else "—",
                        style={"fontSize": "12px"}),
                html.Td(html.Small(info.get("halt_reason", info.get("off_reason", "")),
                                   style={"color": "#e74c3c" if s == "halted" else "#888",
                                          "fontSize": "10px"})),
            ]))

        content = html.Div([
            html.H6("📡 System Status", style={"color": "#7289da", "letterSpacing": "1px"}),
            dbc.Row([
                _kpi("Run dir",        run_dir, "#aaa"),
                _kpi("Equity",         _human_num(eq_latest)),
                _kpi("Scale Factor",   f"{sup.get('scale_factor', 0):.0%}"),
                _kpi("Composite Score",f"{sup.get('composite_score', 0):+.3f}"),
            ], className="mb-3 g-2"),
            dbc.Alert(sup.get("reason", ""), color="dark",
                      className="p-2", style={"fontSize": "11px"}),
            html.H6("Stratégies", style={"color": "#aaa", "fontSize": "12px", "marginTop": "12px"}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Stratégie", style={"fontSize": "11px"}),
                    html.Th("PnL",       style={"fontSize": "11px"}),
                    html.Th("Positions", style={"fontSize": "11px"}),
                    html.Th("Gross Exp", style={"fontSize": "11px"}),
                    html.Th("Info",      style={"fontSize": "11px"}),
                ])),
                html.Tbody(strat_rows),
            ], dark=True, hover=True, size="sm"),
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION : Agents IA
    # ──────────────────────────────────────────────────────────────────────────
    elif "Agents IA" in (section or ""):
        agents = state.get("agents", {})
        agent_cards = []
        for ag_name, ag_info in agents.items():
            st     = ag_info.get("status", "off")
            score  = ag_info.get("score", 0.0)
            conf   = ag_info.get("confidence", 0.0)
            ic     = ag_info.get("ic_recent", 0.0)
            extras = []
            if ag_name == "SentimentAgent":
                extras = [html.Small(
                    f"F&G: {ag_info.get('fg_raw', '?')} — {ag_info.get('fg_label', '?')}",
                    style={"color": "#f39c12"},
                )]
            elif ag_name == "RegimeAgent":
                extras = [html.Small(
                    f"Régime: {ag_info.get('current_regime', 'unknown')} | Q_t: {ag_info.get('current_qt', 0):.1f}",
                    style={"color": "#3498db"},
                )]
            agent_cards.append(dbc.Col(
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong(ag_name, style={"fontSize": "12px"}), width=7),
                        dbc.Col(_status_badge(st), width=5),
                    ], className="mb-2 align-items-center"),
                    html.Div("Score:", style={"fontSize": "10px", "color": "#888", "marginBottom": "3px"}),
                    _fig_score_bar(score),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.Div("Confidence", style={"fontSize": "10px", "color": "#666"}),
                            html.Div(f"{conf:.0%}", style={"fontSize": "13px", "fontWeight": "700"}),
                        ]), width=6),
                        dbc.Col(html.Div([
                            html.Div("IC récent", style={"fontSize": "10px", "color": "#666"}),
                            html.Div(f"{ic:.4f}", style={"fontSize": "13px", "fontWeight": "700"}),
                        ]), width=6),
                    ], className="mt-2"),
                ] + extras), style={"background": _CARD_BG, "border": "1px solid #2a2d3a"}),
                width=6, className="mb-2",
            ))

        rows_data = [
            {"Agent": k, "Statut": v.get("status","off").upper(),
             "Score": f"{v.get('score',0):+.3f}",
             "Confidence": f"{v.get('confidence',0):.0%}",
             "IC récent": f"{v.get('ic_recent',0):.4f}"}
            for k, v in agents.items()
        ]
        content = html.Div([
            html.H6("🤖 Agents IA", style={"color": "#7289da", "letterSpacing": "1px"}),
            dbc.Row(agent_cards, className="mb-3"),
            html.Hr(style={"borderColor": "#1e1e30"}),
            dbc.Table.from_dataframe(pd.DataFrame(rows_data), dark=True, hover=True, size="sm"),
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION : Portfolio
    # ──────────────────────────────────────────────────────────────────────────
    elif "Portfolio" in (section or ""):
        metrics = compute_equity_metrics(equity_df, window=2000)
        var95   = compute_var_cvar(equity_df, alpha=0.05, window=var_window, horizon_steps=var_horizon)
        var99   = compute_var_cvar(equity_df, alpha=0.01, window=var_window, horizon_steps=var_horizon)

        # Equity chart
        if not equity_df.empty and "equity" in equity_df.columns:
            eq_fig = _fig_line(equity_df.index, equity_df["equity"].astype(float), "Equity")
            dd     = compute_drawdown(equity_df["equity"])
            dd_fig = _fig_area(dd.index, dd.values, "Drawdown", height=240)
        else:
            eq_fig = _empty_fig(340, "Lancer le runner pour avoir les données equity.csv")
            dd_fig = _empty_fig(240)

        # Positions table
        mids_snap = snap.get("mids") or {}
        positions = snap.get("positions") or {}
        tgt       = snap.get("target_weights") or {}
        eq_lat    = float(snap.get("equity", float("nan"))) if snap.get("equity") is not None else float("nan")
        pos_rows  = []
        for sym, p in positions.items():
            qty    = float(p.get("qty", 0.0))
            entry  = float(p.get("entry_px", 0.0))
            mid    = float(mids_snap.get(sym, float("nan"))) if sym in mids_snap else float("nan")
            notional = qty * mid if np.isfinite(mid) else float("nan")
            upnl     = (mid - entry) * qty if (np.isfinite(mid) and entry > 0) else float("nan")
            w_cur    = (notional / eq_lat) if (np.isfinite(notional) and np.isfinite(eq_lat) and abs(eq_lat) > 1e-12) else float("nan")
            w_tgt    = float(tgt.get(sym, 0.0))
            pos_rows.append({"Symbol": sym, "Qty": qty, "Entry": entry, "Mid": mid,
                              "Notional": notional, "uPnL": upnl,
                              "W_cur": w_cur, "W_tgt": w_tgt})
        pos_df = pd.DataFrame(pos_rows)
        if not pos_df.empty:
            pos_df = pos_df.sort_values("Notional", key=lambda s: s.abs(), ascending=False)

        content = html.Div([
            html.H6("📈 Portfolio — Equity & Positions", style={"color": "#7289da", "letterSpacing": "1px"}),
            dbc.Row([
                _kpi("Ann. Return", _human_pct(metrics.get("ann_return"))),
                _kpi("Ann. Vol",    _human_pct(metrics.get("ann_vol"))),
                _kpi("Sharpe",      _human_num(metrics.get("sharpe"))),
                _kpi("Max DD",      _human_pct(metrics.get("max_drawdown"))),
                _kpi("VaR 95%",     _human_pct(var95.get("var_95"))),
            ], className="mb-3 g-2"),
            dbc.Row([
                dbc.Col([
                    html.H6("Equity curve (PnL net)",
                            style={"fontSize": "12px", "color": "#aaa"}),
                    dcc.Graph(figure=eq_fig, config={"displayModeBar": False}),
                ], width=8),
                dbc.Col([
                    html.H6("Drawdown", style={"fontSize": "12px", "color": "#aaa"}),
                    dcc.Graph(figure=dd_fig, config={"displayModeBar": False}),
                    html.H6("VaR / CVaR", style={"fontSize": "12px", "color": "#aaa", "marginTop": "8px"}),
                    dbc.Table([html.Tbody([
                        html.Tr([html.Td("VaR 99%"),  html.Td(_human_pct(var99.get("var_99")))]),
                        html.Tr([html.Td("CVaR 99%"), html.Td(_human_pct(var99.get("cvar_99")))]),
                        html.Tr([html.Td("Horizon"),  html.Td(f"{var_horizon} step(s)")]),
                    ])], dark=True, size="sm"),
                ], width=4),
            ], className="mb-3"),
            html.H6("Positions (mark-to-market)", style={"fontSize": "12px", "color": "#aaa"}),
            (dbc.Table.from_dataframe(pos_df, dark=True, hover=True, size="sm", striped=True)
             if not pos_df.empty
             else dbc.Alert("Pas de snapshot_latest.json ou aucune position.",
                            color="secondary")),
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION : Risque
    # ──────────────────────────────────────────────────────────────────────────
    elif "Risque" in (section or ""):
        metrics = compute_equity_metrics(equity_df, window=2000)
        var95   = compute_var_cvar(equity_df, alpha=0.05, window=var_window, horizon_steps=var_horizon)
        var99   = compute_var_cvar(equity_df, alpha=0.01, window=var_window, horizon_steps=var_horizon)

        # Drawdown progress bar
        dd_progress = html.Div()
        if not equity_df.empty and "equity" in equity_df.columns:
            eq_s = equity_df["equity"].dropna()
            if len(eq_s) > 0:
                peak    = eq_s.cummax()
                dd_cur  = float(((eq_s - peak) / peak).iloc[-1] * 100)
                ks_thr  = 3.0
                prog    = min(100, abs(dd_cur) / ks_thr * 100)
                dd_progress = dbc.Progress(
                    value=prog,
                    label=f"DD courant: {dd_cur:.2f}% | Kill-switch à −{ks_thr:.0f}%",
                    color="danger" if prog > 80 else "warning" if prog > 50 else "success",
                    style={"height": "22px"}, className="mb-3",
                )

        # Correlation heatmap
        corr = compute_correlation_matrix(mids_df, window=corr_window)
        corr_fig = _fig_heatmap(corr, "Corr(returns)", height=500)

        # Portfolio variance
        w_last = _pick_last_weights(weights_df)
        port   = compute_portfolio_var_from_weights(mids_df, weights_last=w_last, window=corr_window)

        content = html.Div([
            html.H6("⚠️ Risk Management", style={"color": "#7289da", "letterSpacing": "1px"}),
            dbc.Row([
                _kpi("Ann. Return", _human_pct(metrics.get("ann_return"))),
                _kpi("Ann. Vol",    _human_pct(metrics.get("ann_vol"))),
                _kpi("Sharpe",      _human_num(metrics.get("sharpe"))),
                _kpi("Max DD",      _human_pct(metrics.get("max_drawdown"))),
                _kpi("VaR 95%",     _human_pct(var95.get("var_95"))),
                _kpi("CVaR 95%",    _human_pct(var95.get("cvar_95"))),
            ], className="mb-3 g-2"),
            dd_progress,
            html.H6("Correlation matrix", style={"fontSize": "12px", "color": "#aaa"}),
            dcc.Graph(figure=corr_fig, config={"displayModeBar": False}),
            html.Hr(style={"borderColor": "#1e1e30"}),
            html.H6("Portfolio variance (weights × cov)",
                    style={"fontSize": "12px", "color": "#aaa"}),
            dbc.Row([
                _kpi("Port vol/step", _human_num(port.get("port_vol_step"), 6)),
                _kpi("Port var/step", _human_num(port.get("port_var_step"), 10)),
                _kpi("#assets",       str(int(port.get("n_assets", 0)))),
            ], className="g-2"),
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION : Régime
    # ──────────────────────────────────────────────────────────────────────────
    elif "Régime" in (section or ""):
        ag_regime = state.get("agents", {}).get("RegimeAgent", {})
        ag_sent   = state.get("agents", {}).get("SentimentAgent", {})
        regime    = ag_regime.get("current_regime", "unknown")
        qt        = ag_regime.get("current_qt", 0.5)

        _regime_desc = {
            "mean_reverting" : ("🟢 Mean Reverting",  "info",   "Conditions idéales pour stat-arb. Q_t = 1.0"),
            "carry_favorable": ("🟢 Carry Favorable",  "success","Funding stable et élevé. Boost overlay carry."),
            "trending"       : ("🟡 Trending",          "warning","Momentum fort. Exposition réduite. Q_t = 0.3"),
            "high_vol"       : ("🔴 High Vol",          "danger", "Vol BTC > p90. Stratégie stoppée. Q_t = 0.0"),
            "crisis"         : ("🚨 CRISIS",            "danger", "Liquidations massives. Kill-switch activé."),
            "unknown"        : ("⚪ Unknown",            "secondary","Données insuffisantes pour classifier."),
        }
        label, color_r, desc = _regime_desc.get(regime, ("⚪ Unknown", "secondary", ""))

        comp_df = pd.DataFrame({
            "Composante"  : ["Volatilité", "Momentum", "Liquidations", "Funding"],
            "Score"       : [ag_regime.get("vol_score", 0), ag_regime.get("momentum_score", 0),
                             ag_regime.get("liq_score", 0), ag_regime.get("funding_score", 0)],
            "Poids"       : [0.35, 0.40, 0.15, 0.10],
        })
        comp_df["Contribution"] = comp_df["Score"] * comp_df["Poids"]

        content = html.Div([
            html.H6("🌊 Régime de Marché", style={"color": "#7289da", "letterSpacing": "1px"}),
            dbc.Alert([html.H5(label, className="mb-1"), html.Small(desc)], color=color_r,
                      className="p-3 mb-3"),
            dbc.Row([
                _kpi("Q_t appliqué",  f"{qt:.1f}"),
                _kpi("Fear & Greed",  f"{ag_sent.get('fg_raw', 50)} / 100"),
                _kpi("Vol Score",     f"{ag_regime.get('vol_score', 0):.3f}"),
                _kpi("Momentum",      f"{ag_regime.get('momentum_score', 0):.3f}"),
            ], className="mb-3 g-2"),
            html.H6("Composantes du score régime",
                    style={"fontSize": "12px", "color": "#aaa"}),
            dbc.Table.from_dataframe(comp_df.round(4), dark=True, hover=True, size="sm"),
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION : Prédictions ML
    # ──────────────────────────────────────────────────────────────────────────
    elif "Prédictions" in (section or ""):
        probas = state.get("predictions", {})
        if probas:
            rows_pred = []
            for sym, p in sorted(probas.items(), key=lambda x: -x[1]):
                direction = "🟢 BULLISH" if p > 0.55 else "🔴 BEARISH" if p < 0.45 else "⚪ NEUTRAL"
                rows_pred.append({
                    "Symbole": sym,
                    "P(hausse)": f"{p:.3f}",
                    "Score": f"{2*(p-0.5):+.3f}",
                    "Signal": direction,
                })
            pred_table = dbc.Table.from_dataframe(
                pd.DataFrame(rows_pred), dark=True, hover=True, size="sm", striped=True,
            )
        else:
            pred_table = dbc.Alert("Pas de prédictions disponibles dans state.json.",
                                   color="secondary")
        content = html.Div([
            html.H6("🔮 Prédictions ML — PredictionAgent",
                    style={"color": "#7289da", "letterSpacing": "1px"}),
            dbc.Alert("P > 0.55 → haussier | P < 0.45 → baissier",
                      color="dark", className="p-2 mb-3", style={"fontSize": "11px"}),
            pred_table,
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION : Equity Curves
    # ──────────────────────────────────────────────────────────────────────────
    elif "Equity Curves" in (section or ""):
        if not equity_df.empty and "equity" in equity_df.columns:
            eq_s = equity_df["equity"].dropna()
            eq_norm = eq_s / eq_s.iloc[0] * 100 if len(eq_s) > 0 else eq_s
            eq_fig  = _fig_line(equity_df.index, eq_norm, "Equity (base 100)", height=400)
            metrics = compute_equity_metrics(equity_df, window=2000)
            total_ret = (eq_s.iloc[-1] / eq_s.iloc[0] - 1) * 100 if len(eq_s) > 1 else float("nan")
            kpi_row = dbc.Row([
                _kpi("Retour total", f"{total_ret:+.2f}%" if np.isfinite(total_ret) else "—"),
                _kpi("Sharpe",       _human_num(metrics.get("sharpe"))),
                _kpi("Max Drawdown", _human_pct(metrics.get("max_drawdown"))),
                _kpi("Ann. Vol",     _human_pct(metrics.get("ann_vol"))),
            ], className="mt-3 g-2")
        else:
            eq_fig  = _empty_fig(400, "Lancer backtest ou paper trading pour avoir equity.csv")
            kpi_row = html.Div()

        content = html.Div([
            html.H6("📊 Equity Curve", style={"color": "#7289da", "letterSpacing": "1px"}),
            dcc.Graph(figure=eq_fig, config={"displayModeBar": False}),
            kpi_row,
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION : Raw Tables
    # ──────────────────────────────────────────────────────────────────────────
    elif "Raw Tables" in (section or ""):
        content = html.Div([
            html.H6("🔢 Raw Tables", style={"color": "#7289da", "letterSpacing": "1px"}),
            html.H6("equity.csv (tail)", style={"fontSize": "12px", "color": "#aaa"}),
            dbc.Table.from_dataframe(
                equity_df.tail(500), dark=True, hover=True, size="sm", striped=True,
            ) if not equity_df.empty else dbc.Alert("Vide", color="dark"),
            html.H6("mids.csv (tail)", style={"fontSize": "12px", "color": "#aaa", "marginTop": "12px"}),
            dbc.Table.from_dataframe(
                mids_df.tail(500), dark=True, hover=True, size="sm", striped=True,
            ) if not mids_df.empty else dbc.Alert("Vide", color="dark"),
            html.H6("weights.csv (tail)", style={"fontSize": "12px", "color": "#aaa", "marginTop": "12px"}),
            dbc.Table.from_dataframe(
                weights_df.tail(500), dark=True, hover=True, size="sm", striped=True,
            ) if not weights_df.empty else dbc.Alert("Vide", color="dark"),
        ])

    else:
        content = dbc.Alert("Sélectionne une section dans le sidebar.", color="dark")

    # ── Raw tables toggle (any section) ──────────────────────────────────────
    if "show" in (show_raw or []) and "Raw Tables" not in (section or ""):
        raw_section = html.Div([
            html.Hr(style={"borderColor": "#1e1e30"}),
            html.H6("🔢 Raw — equity.csv",  style={"fontSize": "12px", "color": "#aaa"}),
            dbc.Table.from_dataframe(equity_df.tail(500), dark=True, hover=True,
                                     size="sm") if not equity_df.empty else html.Div(),
        ])
    else:
        raw_section = html.Div()

    return hdr_exch, hdr_snap, ks_banner, content, raw_section


@app.callback(
    Output("btn-flat-all", "children"),
    Input("btn-flat-all", "n_clicks"),
    prevent_initial_call=True,
)
def flat_all(_n):
    return "⚠️ Non implémenté en démo"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8051)