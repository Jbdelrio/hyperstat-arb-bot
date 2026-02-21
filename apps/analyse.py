#!/usr/bin/env python3
# apps/analyse.py
"""
HyperStat — Analyse & Backtest sur données réelles Hyperliquid
Lance avec : streamlit run apps/analyse.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.error("pip install plotly")
    st.stop()

from hyperstat.backtest.engine import run_backtest
from hyperstat.strategy.stat_arb import StatArbStrategy, StatArbConfig
from hyperstat.strategy.regime import RegimeModel, RegimeConfig
from hyperstat.strategy.allocator import PortfolioAllocator, AllocatorConfig
from hyperstat.strategy.funding_overlay import FundingOverlayModel, FundingOverlayConfig

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HyperStat · Analyse",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #080b10;
    color: #c9d1d9;
}
.stApp { background: #080b10; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1c2128;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #58a6ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1c2128;
    padding-bottom: 6px;
    margin-top: 18px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #1c2128;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #6e7681;
    padding: 10px 22px;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

/* KPI cards */
.kpi-row { display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }
.kpi {
    background: #0d1117;
    border: 1px solid #1c2128;
    border-radius: 6px;
    padding: 14px 20px;
    min-width: 140px;
    flex: 1;
    position: relative;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 6px 6px 0 0;
}
.kpi.positive::before { background: #3fb950; }
.kpi.negative::before { background: #f85149; }
.kpi.neutral::before  { background: #58a6ff; }
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6e7681;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    margin-top: 4px;
    color: #e6edf3;
}
.kpi-value.pos { color: #3fb950; }
.kpi-value.neg { color: #f85149; }

/* Section titles */
.section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #58a6ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-left: 3px solid #58a6ff;
    padding-left: 10px;
    margin: 24px 0 12px;
}

/* Header */
.page-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.page-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    color: #6e7681;
    margin-bottom: 20px;
}

/* Alert / info box */
.info-strip {
    background: #111d2b;
    border: 1px solid #1f3958;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
    color: #8b949e;
    margin-bottom: 16px;
}
.warn-strip {
    background: #1e1a0e;
    border: 1px solid #3d2e0a;
    border-left: 3px solid #d29922;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
    color: #8b949e;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
PALETTE = ["#58a6ff","#3fb950","#d29922","#f85149","#bc8cff","#39d353","#ff7b72","#79c0ff"]
BG, GRID, PAPER = "#080b10", "#161b22", "#0d1117"

def _pct(x, d=2):
    if x is None or not np.isfinite(float(x)): return "—"
    return f"{100*float(x):+.{d}f}%"

def _num(x, d=2):
    if x is None or not np.isfinite(float(x)): return "—"
    return f"{float(x):,.{d}f}"

def kpi(label, value, cls="neutral"):
    val_cls = "pos" if cls == "positive" else ("neg" if cls == "negative" else "")
    st.markdown(f"""
    <div class="kpi {cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {val_cls}">{value}</div>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)

def dark_fig(height=340):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        margin=dict(l=12, r=12, t=36, b=12),
        font=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID, showgrid=True, zeroline=False),
    )
    return fig

def pline(fig, x, y, name, color=None, width=1.5, fill=None):
    kw = dict(x=x, y=y, mode="lines", name=name,
              line=dict(color=color or PALETTE[0], width=width))
    if fill:
        kw["fill"] = fill
        kw["fillcolor"] = (color or PALETTE[0]).replace(")", ",0.12)").replace("rgb","rgba") if color else "rgba(88,166,255,0.10)"
    fig.add_trace(go.Scatter(**kw))

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(data_dir: str, tf: str):
    root = Path(data_dir)
    candles_dir = root / "candles"
    funding_dir = root / "funding"

    candles, funding = {}, {}

    if not candles_dir.exists():
        return candles, funding

    for sym_dir in sorted(candles_dir.iterdir()):
        if not sym_dir.is_dir():
            continue
        f = sym_dir / f"{tf}.parquet"
        if not f.exists():
            # try csv fallback
            f = sym_dir / f"{tf}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
        else:
            df = pd.read_parquet(f)

        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        for col in ("open","high","low","close","volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        candles[sym_dir.name] = df

    if funding_dir.exists():
        for sym_dir in sorted(funding_dir.iterdir()):
            if not sym_dir.is_dir():
                continue
            for ext in ("parquet","csv"):
                f = sym_dir / f"8h.{ext}"
                if f.exists():
                    df = pd.read_parquet(f) if ext == "parquet" else pd.read_csv(f)
                    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
                    df = df.dropna(subset=["ts","rate"]).sort_values("ts").reset_index(drop=True)
                    funding[sym_dir.name] = df
                    break

    return candles, funding


@st.cache_data(show_spinner=False)
def run_full_backtest(
    _candles_pkl, _funding_pkl, _coins, _buckets_raw,
    initial_equity, gross_target, max_w_coin, max_w_bucket,
    dollar_neutral, beta_neutral,
    z_in, z_out, z_max, horizon_bars, tf_min,
    fee_bps, slip_base, slip_k,
    max_dd, cooldown,
    fund_enabled, fund_eta,
):
    candles  = _candles_pkl
    funding  = _funding_pkl

    # Reconstruire ts
    for d in (candles, funding):
        for k in d:
            d[k]["ts"] = pd.to_datetime(d[k]["ts"], utc=True, errors="coerce")

    buckets = {}
    for b in _buckets_raw:
        bname, syms = b
        buckets[bname] = [s for s in syms if s in candles]

    buckets = {k: v for k, v in buckets.items() if len(v) >= 3}
    if not buckets:
        buckets = {"all": [c for c in _coins if c in candles]}

    cfg = {
        "data": {"timeframe": f"{tf_min}m", "base_factor_symbol": "BTC"},
        "portfolio": {
            "initial_equity_eur": initial_equity, "gross_target": gross_target,
            "max_weight_per_coin": max_w_coin, "max_weight_per_bucket": max_w_bucket,
            "dollar_neutral": dollar_neutral, "beta_neutral": beta_neutral,
        },
        "execution": {
            "mode": "taker", "fill_model": "close",
            "fees":     {"taker_bps": fee_bps, "maker_bps": 2},
            "slippage": {"base_bps": slip_base, "k_bps_per_1pct_rv1h": slip_k},
        },
        "risk": {
            "max_intraday_drawdown_pct": max_dd,
            "cooldown_minutes": cooldown,
            "z_emergency_flat": 3.5,
        },
    }

    report = run_backtest(
        cfg=cfg,
        candles_by_symbol={c: candles[c] for c in _coins if c in candles},
        funding_by_symbol={c: funding[c] for c in _coins if c in funding},
        buckets=buckets,
        stat_arb=StatArbStrategy(StatArbConfig(
            timeframe_minutes=tf_min, horizon_bars=horizon_bars,
            z_in=z_in, z_out=z_out, z_max=z_max,
        )),
        regime_model=RegimeModel(RegimeConfig(timeframe_minutes=tf_min), base_factor_symbol="BTC"),
        allocator=PortfolioAllocator(AllocatorConfig(
            gross_target_stat=gross_target, max_weight_per_coin=max_w_coin,
            max_weight_per_bucket=max_w_bucket,
            dollar_neutral=dollar_neutral, beta_neutral=beta_neutral,
        )),
        funding_overlay=FundingOverlayModel(FundingOverlayConfig(
            enabled=fund_enabled, eta=fund_eta,
        )) if fund_enabled else None,
    )
    return report, buckets

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⬡ HyperStat")

    st.markdown("## Données")
    data_dir = st.text_input("Dossier data/", value="data")
    tf_opts = {"1m":1,"5m":5,"15m":15,"1h":60}
    tf_label = st.selectbox("Timeframe chargé", list(tf_opts.keys()), index=1)
    tf_min = tf_opts[tf_label]

    with st.spinner("Chargement…"):
        candles, funding = load_data(data_dir, tf_label)

    coins_avail = sorted(candles.keys())
    if not coins_avail:
        st.error(f"Aucun parquet trouvé dans {data_dir}/candles/*/{{tf}}.parquet")
        st.stop()

    st.caption(f"✅ {len(coins_avail)} coins · {len(funding)} funding")

    coins_sel = st.multiselect("Coins actifs", coins_avail, default=coins_avail)
    if len(coins_sel) < 3:
        st.warning("Sélectionne au moins 3 coins.")
        coins_sel = coins_avail[:max(3, len(coins_avail))]

    # Date range
    all_ts = pd.concat([candles[c]["ts"] for c in coins_sel], ignore_index=True)
    ts_min, ts_max = all_ts.min().date(), all_ts.max().date()
    date_range = st.date_input("Période backtest", value=(ts_min, ts_max),
                               min_value=ts_min, max_value=ts_max)
    if len(date_range) == 2:
        d_start, d_end = date_range
    else:
        d_start, d_end = ts_min, ts_max

    st.markdown("## Buckets")
    bucket_mode = st.selectbox("Mode", ["Auto (2 groupes)", "Tout dans 1 bucket", "3 groupes"])

    st.markdown("## Stratégie")
    horizon_bars = st.slider("Horizon (barres)", 4, 48, 12)
    z_in  = st.slider("z_in",  0.5, 3.0, 1.5, 0.1)
    z_out = st.slider("z_out", 0.1, 2.0, 0.5, 0.1)
    z_max = st.slider("z_max", 1.5, 5.0, 3.0, 0.5)

    st.markdown("## Portfolio")
    initial_equity   = st.number_input("Capital (€)", 100, 500000, 1500, 100)
    gross_target     = st.slider("Gross target", 0.3, 2.0, 1.2, 0.1)
    max_w_coin       = st.slider("Max poids/coin", 0.03, 0.30, 0.12, 0.01)
    max_w_bucket     = st.slider("Max poids/bucket", 0.10, 0.80, 0.35, 0.05)
    dollar_neutral   = st.checkbox("Dollar neutral", True)
    beta_neutral     = st.checkbox("Beta neutral", False)

    st.markdown("## Coûts")
    fee_bps   = st.slider("Fees taker (bps)", 1, 20, 6)
    slip_base = st.slider("Slippage base (bps)", 0, 30, 8)
    slip_k    = st.slider("Slippage k", 0, 30, 10)

    st.markdown("## Risk")
    max_dd   = st.slider("Max DD intraday (%)", 1, 15, 3) / 100
    cooldown = st.slider("Cooldown (min)", 60, 1440, 720, 60)

    st.markdown("## Funding Overlay")
    fund_enabled = st.checkbox("Activer", True)
    fund_eta     = st.slider("Eta", 0.0, 0.5, 0.10, 0.01) if fund_enabled else 0.0

    st.markdown("---")
    run_bt = st.button("▶  Lancer le backtest", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════
# FILTRAGE PAR DATE
# ══════════════════════════════════════════════════════════════
def filter_by_date(df, d_start, d_end):
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    mask = (ts.dt.date >= d_start) & (ts.dt.date <= d_end)
    return df[mask].reset_index(drop=True)

candles_f = {c: filter_by_date(candles[c], d_start, d_end) for c in coins_sel}
funding_f = {c: filter_by_date(funding[c], d_start, d_end) for c in coins_sel if c in funding}

# ══════════════════════════════════════════════════════════════
# BUCKETS
# ══════════════════════════════════════════════════════════════
L1 = ["BTC","ETH","SOL","AVAX","BNB","NEAR","ATOM"]
def make_buckets(coins, mode):
    if mode == "Tout dans 1 bucket":
        return [("all", coins)]
    if mode == "3 groupes":
        n = len(coins); t = n // 3
        groups = [coins[:t+1], coins[t:2*t+1], coins[2*t:]]
        return [(f"G{i+1}", g) for i, g in enumerate(groups) if len(g) >= 3]
    # Auto 2 groupes
    g1 = [c for c in coins if c in L1]
    g2 = [c for c in coins if c not in L1]
    result = []
    if len(g1) >= 3: result.append(("L1", g1))
    if len(g2) >= 3: result.append(("Alts", g2))
    if not result:   result = [("all", coins)]
    return result

buckets_raw = make_buckets(coins_sel, bucket_mode)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="page-header">HyperStat · Analyse</div>', unsafe_allow_html=True)
n_bars_total = sum(len(candles_f[c]) for c in coins_sel)
days_span = (d_end - d_start).days
st.markdown(
    f'<div class="page-sub">{len(coins_sel)} coins · {days_span}j de données réelles Hyperliquid · '
    f'{n_bars_total:,} barres total · timeframe {tf_label}</div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_market, tab_corr, tab_signal, tab_backtest = st.tabs([
    "📈  MARCHÉ",
    "🔗  CORRÉLATIONS",
    "⚡  SIGNAUX Z-SCORE",
    "🏦  BACKTEST",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — MARCHÉ
# ──────────────────────────────────────────────────────────────
with tab_market:
    section("Prix normalisés (base 100)")

    # Prix normalisés
    close_df = {}
    for c in coins_sel:
        df = candles_f[c]
        if df.empty: continue
        s = df.set_index("ts")["close"].astype(float)
        if s.iloc[0] > 0:
            close_df[c] = s / s.iloc[0] * 100

    if close_df:
        fig = dark_fig(340)
        for i, (c, s) in enumerate(close_df.items()):
            step = max(1, len(s) // 2000)
            pline(fig, s.index[::step], s.values[::step], c, PALETTE[i % len(PALETTE)])
        fig.update_layout(title="Prix normalisés base 100")
        st.plotly_chart(fig, use_container_width=True)

    # Candle chart pour 1 coin
    section("Chandeliers OHLCV")
    coin_ohlcv = st.selectbox("Coin", coins_sel, key="ohlcv_coin")
    df_ohlcv = candles_f[coin_ohlcv]
    if not df_ohlcv.empty:
        step = max(1, len(df_ohlcv) // 1000)
        dfs = df_ohlcv.iloc[::step]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=dfs["ts"], open=dfs["open"], high=dfs["high"],
            low=dfs["low"], close=dfs["close"], name=coin_ohlcv,
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950", decreasing_fillcolor="#f85149",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=dfs["ts"], y=dfs["volume"], name="Volume",
            marker_color=["#3fb950" if c >= o else "#f85149"
                          for c, o in zip(dfs["close"], dfs["open"])],
            opacity=0.6,
        ), row=2, col=1)
        fig.update_layout(
            template="plotly_dark", height=460,
            paper_bgcolor=PAPER, plot_bgcolor=BG,
            margin=dict(l=12, r=12, t=36, b=12),
            font=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
            xaxis_rangeslider_visible=False, showlegend=False,
            title=dict(text=f"{coin_ohlcv} · OHLCV", font=dict(size=13, color="#58a6ff")),
        )
        fig.update_xaxes(gridcolor=GRID)
        fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, use_container_width=True)

    # Stats descriptives
    section("Stats descriptives")
    rows = []
    for c in coins_sel:
        df = candles_f[c]
        if df.empty: continue
        cl = df["close"].astype(float)
        ret = cl.pct_change().dropna()
        rows.append({
            "Coin": c,
            "Dernier prix": f"{cl.iloc[-1]:,.4f}",
            "Return période": f"{(cl.iloc[-1]/cl.iloc[0]-1)*100:+.2f}%",
            "Vol annualisée": f"{ret.std() * np.sqrt(365*24*60/tf_min) * 100:.1f}%",
            "Max": f"{cl.max():,.4f}",
            "Min": f"{cl.min():,.4f}",
            "Barres": f"{len(df):,}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Coin"), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 2 — CORRÉLATIONS
# ──────────────────────────────────────────────────────────────
with tab_corr:
    section("Matrice de corrélation des returns")

    close_all = {}
    for c in coins_sel:
        df = candles_f[c]
        if not df.empty:
            close_all[c] = df.set_index("ts")["close"].astype(float)

    if len(close_all) >= 2:
        px_df = pd.DataFrame(close_all).sort_index().ffill().dropna(how="all")
        ret_df = px_df.pct_change().dropna(how="all")
        corr = ret_df.corr().round(3)

        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale=[[0,"#f85149"],[0.5,"#161b22"],[1,"#3fb950"]],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, family="IBM Plex Mono"),
            hovertemplate="%{y} / %{x} : %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", height=500,
            paper_bgcolor=PAPER, plot_bgcolor=BG,
            margin=dict(l=12, r=12, t=36, b=12),
            font=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
            title=dict(text="Corrélation returns", font=dict(size=13, color="#58a6ff")),
        )
        st.plotly_chart(fig, use_container_width=True)

        section("Corrélation glissante vs BTC (30 périodes)")
        if "BTC" in ret_df.columns:
            fig2 = dark_fig(280)
            for i, c in enumerate([x for x in coins_sel if x != "BTC" and x in ret_df.columns]):
                rc = ret_df[c].rolling(30).corr(ret_df["BTC"]).dropna()
                step = max(1, len(rc) // 2000)
                pline(fig2, rc.index[::step], rc.values[::step], c, PALETTE[i % len(PALETTE)])
            fig2.add_hline(y=0, line_dash="dot", line_color="#444")
            fig2.update_layout(title="Rolling corr vs BTC (30 barres)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Ajoute BTC à la sélection pour voir les corrélations glissantes.")

        section("Distribution des returns par coin")
        fig3 = dark_fig(300)
        for i, c in enumerate(coins_sel):
            if c not in ret_df.columns: continue
            r = ret_df[c].dropna() * 100
            fig3.add_trace(go.Violin(
                y=r.values, name=c, box_visible=True, meanline_visible=True,
                line_color=PALETTE[i % len(PALETTE)], opacity=0.8,
            ))
        fig3.update_layout(title="Distribution returns (%)", violinmode="overlay")
        st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 3 — SIGNAUX Z-SCORE
# ──────────────────────────────────────────────────────────────
with tab_signal:
    section("Z-scores cross-sectionnels par bucket")

    from hyperstat.data.features import compute_returns, compute_ewma_vol

    for bname, bcoins in buckets_raw:
        valid = [c for c in bcoins if c in candles_f and not candles_f[c].empty]
        if len(valid) < 2:
            continue

        st.markdown(f'<div class="section">Bucket · {bname} — {", ".join(valid)}</div>',
                    unsafe_allow_html=True)

        h = horizon_bars
        zscore_series = {}
        for c in valid:
            df = candles_f[c]
            cl = df.set_index("ts")["close"].astype(float)
            lr = np.log(cl).diff(h)
            zscore_series[c] = lr

        zdf = pd.DataFrame(zscore_series).dropna(how="all")
        if zdf.empty:
            continue

        z_cs = pd.DataFrame(index=zdf.index)
        for c in zdf.columns:
            col = zdf[c].values.astype(float)
            med = np.nanmedian(zdf.values, axis=1)
            mad = np.nanmedian(np.abs(zdf.values - med[:, None]), axis=1)
            mad = np.where(mad < 1e-10, 1e-10, mad)
            z_cs[c] = (col - med) / (1.4826 * mad)

        fig = dark_fig(300)
        for i, c in enumerate(z_cs.columns):
            s = z_cs[c].dropna()
            step = max(1, len(s) // 2000)
            pline(fig, s.index[::step], s.values[::step], c, PALETTE[i % len(PALETTE)])
        for level, color, dash in [(z_in,"#d29922","dash"),(-z_in,"#d29922","dash"),
                                    (z_out,"#3fb950","dot"),(-z_out,"#3fb950","dot")]:
            fig.add_hline(y=level, line_dash=dash, line_color=color, opacity=0.6)
        fig.add_hline(y=0, line_dash="dot", line_color="#444")
        fig.update_layout(title=f"Z-scores cross-section · {bname}")
        st.plotly_chart(fig, use_container_width=True)

        last_z = z_cs.iloc[-1].dropna().sort_values()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Derniers z-scores**")
            for c, z in last_z.items():
                arrow = "↑ SHORT" if z > z_in else ("↓ LONG" if z < -z_in else "· neutre")
                color = "#f85149" if z > z_in else ("#3fb950" if z < -z_in else "#6e7681")
                st.markdown(
                    f'<span style="font-family:IBM Plex Mono;font-size:12px;color:{color}">'
                    f'{c:8s} {z:+.3f}  {arrow}</span>', unsafe_allow_html=True)
        with col2:
            fig_hist = dark_fig(200)
            all_z = z_cs.values.flatten()
            all_z = all_z[np.isfinite(all_z)]
            fig_hist.add_trace(go.Histogram(x=all_z, nbinsx=60,
                marker_color="#58a6ff", opacity=0.75, name="z-scores"))
            fig_hist.add_vline(x=z_in,  line_dash="dash", line_color="#d29922")
            fig_hist.add_vline(x=-z_in, line_dash="dash", line_color="#d29922")
            fig_hist.update_layout(title="Distribution z-scores", showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

    section("Volatilité EWMA annualisée par coin")
    fig_vol = dark_fig(300)
    for i, c in enumerate(coins_sel):
        df = candles_f[c]
        if df.empty: continue
        r = compute_returns(df)
        vol = compute_ewma_vol(r, lam=0.94, min_periods=20) * np.sqrt(365*24*60/tf_min) * 100
        vol = vol.dropna()
        step = max(1, len(vol) // 2000)
        pline(fig_vol, vol.index[::step], vol.values[::step], c, PALETTE[i % len(PALETTE)])
    fig_vol.update_layout(title="Vol EWMA annualisée (%)")
    st.plotly_chart(fig_vol, use_container_width=True)

    if funding_f:
        section("Funding rates (8h)")
        fig_fund = dark_fig(260)
        for i, c in enumerate([x for x in coins_sel if x in funding_f]):
            df_f = funding_f[c]
            if df_f.empty: continue
            pline(fig_fund, df_f["ts"], df_f["rate"] * 100, c, PALETTE[i % len(PALETTE)])
        fig_fund.add_hline(y=0, line_dash="dot", line_color="#444")
        fig_fund.update_layout(title="Funding rate (%) par coin")
        st.plotly_chart(fig_fund, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 4 — BACKTEST
# ──────────────────────────────────────────────────────────────
with tab_backtest:

    if not run_bt and "bt_report" not in st.session_state:
        st.markdown(
            '<div class="info-strip">👈 Configure les paramètres dans la sidebar puis clique '
            '<b>▶ Lancer le backtest</b> pour exécuter sur tes données réelles Hyperliquid.</div>',
            unsafe_allow_html=True)
        st.stop()

    if run_bt:
        with st.spinner("⏳ Backtest en cours sur données réelles…"):
            try:
                report, buckets_used = run_full_backtest(
                    _candles_pkl=candles_f, _funding_pkl=funding_f,
                    _coins=tuple(coins_sel), _buckets_raw=tuple(buckets_raw),
                    initial_equity=initial_equity, gross_target=gross_target,
                    max_w_coin=max_w_coin, max_w_bucket=max_w_bucket,
                    dollar_neutral=dollar_neutral, beta_neutral=beta_neutral,
                    z_in=z_in, z_out=z_out, z_max=z_max,
                    horizon_bars=horizon_bars, tf_min=tf_min,
                    fee_bps=fee_bps, slip_base=slip_base, slip_k=slip_k,
                    max_dd=max_dd, cooldown=cooldown,
                    fund_enabled=fund_enabled, fund_eta=fund_eta,
                )
                st.session_state["bt_report"]  = report
                st.session_state["bt_buckets"] = buckets_used
            except Exception as e:
                st.error(f"Erreur backtest : {e}")
                st.exception(e)
                st.stop()
    else:
        report       = st.session_state["bt_report"]
        buckets_used = st.session_state["bt_buckets"]

    m = report.metrics

    # KPIs
    section("Performance")
    col1,col2,col3,col4,col5,col6,col7 = st.columns(7)
    with col1: kpi("Return total",  _pct(m.total_return), "positive" if m.total_return>=0 else "negative")
    with col2: kpi("CAGR",          _pct(m.cagr,1),       "positive" if m.cagr>=0 else "negative")
    with col3: kpi("Vol ann.",       _pct(m.ann_vol,1),    "neutral")
    with col4: kpi("Sharpe",         _num(m.sharpe,2),     "positive" if m.sharpe>=1 else ("neutral" if m.sharpe>=0 else "negative"))
    with col5: kpi("Max Drawdown",   _pct(m.max_drawdown), "negative")
    with col6: kpi("Barres",         f"{m.n_steps:,}",     "neutral")
    with col7: kpi("Coins",          str(len(coins_sel)),  "neutral")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1: kpi("PnL brut",    _num(m.pnl_gross),              "positive" if m.pnl_gross>=0 else "negative")
    with col2: kpi("PnL funding", _num(m.pnl_funding),            "positive" if m.pnl_funding>=0 else "negative")
    with col3: kpi("Fees",        f"−{_num(abs(m.pnl_fees))}",    "negative")
    with col4: kpi("Slippage",    f"−{_num(abs(m.pnl_slippage))}", "negative")
    with col5: kpi("PnL net",     _num(m.pnl_net),                "positive" if m.pnl_net>=0 else "negative")

    # Equity + Drawdown
    section("Equity curve")
    eq   = report.equity_curve["equity"].astype(float)
    peak = eq.cummax()
    dd   = (eq - peak) / peak

    fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.68, 0.32], vertical_spacing=0.03)
    fig_eq.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines", name="Equity",
        line=dict(color="#58a6ff", width=1.8),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
    ), row=1, col=1)
    fig_eq.add_hline(y=initial_equity, line_dash="dot", line_color="#444",
                     annotation_text=f"Départ {initial_equity}€", row=1, col=1)
    fig_eq.add_trace(go.Scatter(
        x=dd.index, y=dd.values*100, mode="lines", name="Drawdown (%)",
        line=dict(color="#f85149", width=1.2),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.10)",
    ), row=2, col=1)
    fig_eq.add_hline(y=-max_dd*100, line_dash="dash", line_color="#d29922",
                     annotation_text=f"Kill-switch {-max_dd*100:.0f}%", row=2, col=1)
    fig_eq.update_layout(
        template="plotly_dark", height=440,
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        margin=dict(l=12, r=12, t=36, b=12),
        font=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Equity & Drawdown", font=dict(size=13, color="#58a6ff")),
    )
    fig_eq.update_xaxes(gridcolor=GRID)
    fig_eq.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig_eq, use_container_width=True)

    # Décomposition PnL
    section("Décomposition PnL cumulé")
    col1, col2 = st.columns(2)
    with col1:
        pnl = report.pnl_curve
        fig_pnl = dark_fig(300)
        for col_name, color, label in [
            ("pnl_net_step","#58a6ff","Net"),
            ("pnl_price",   "#3fb950","Prix"),
            ("pnl_funding", "#d29922","Funding"),
        ]:
            if col_name in pnl.columns:
                s = pnl[col_name].cumsum()
                step = max(1, len(s) // 2000)
                pline(fig_pnl, s.index[::step], s.values[::step], label, color)
        fig_pnl.add_hline(y=0, line_dash="dot", line_color="#444")
        fig_pnl.update_layout(title="PnL cumulé par composante")
        st.plotly_chart(fig_pnl, use_container_width=True)

    with col2:
        labels = ["PnL brut","Funding","Fees","Slippage","PnL net"]
        values = [m.pnl_gross, m.pnl_funding, -abs(m.pnl_fees), -abs(m.pnl_slippage), m.pnl_net]
        colors = ["#3fb950" if v>=0 else "#f85149" for v in values]
        fig_bar = dark_fig(300)
        fig_bar.add_trace(go.Bar(
            x=labels, y=values, marker_color=colors,
            text=[f"{v:+.1f}€" for v in values], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11),
        ))
        fig_bar.add_hline(y=0, line_dash="dot", line_color="#444")
        fig_bar.update_layout(title="Bilan coûts / PnL (€)", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Poids & Exposition
    section("Poids & Exposition")
    w_df = report.weights
    if not w_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_w = dark_fig(280)
            step = max(1, len(w_df) // 1000)
            for i, c in enumerate(w_df.columns):
                s = w_df[c].iloc[::step]
                pline(fig_w, s.index, s.values, c, PALETTE[i % len(PALETTE)], width=1.2)
            fig_w.add_hline(y=0, line_dash="dot", line_color="#444")
            fig_w.update_layout(title="Poids cibles par coin")
            st.plotly_chart(fig_w, use_container_width=True)
        with col2:
            gross_ts = w_df.abs().sum(axis=1)
            net_ts   = w_df.sum(axis=1)
            step = max(1, len(gross_ts) // 1000)
            fig_exp = dark_fig(280)
            pline(fig_exp, gross_ts.index[::step], gross_ts.values[::step], "Gross", "#58a6ff")
            pline(fig_exp, net_ts.index[::step],   net_ts.values[::step],   "Net",   "#d29922")
            fig_exp.add_hline(y=gross_target, line_dash="dash", line_color="#3fb950",
                              annotation_text=f"Target {gross_target}")
            fig_exp.add_hline(y=0, line_dash="dot", line_color="#444")
            fig_exp.update_layout(title="Exposition gross / net")
            st.plotly_chart(fig_exp, use_container_width=True)

    # Turnover
    section("Turnover & coûts dynamiques")
    col1, col2 = st.columns(2)
    with col1:
        to = report.turnover.rolling(20, min_periods=1).mean()
        step = max(1, len(to) // 1000)
        fig_to = dark_fig(240)
        pline(fig_to, to.index[::step], to.values[::step], "Turnover MA20", "#bc8cff")
        fig_to.update_layout(title=f"Turnover moyen (avg={m.avg_turnover:.4f}/barre)")
        st.plotly_chart(fig_to, use_container_width=True)
    with col2:
        if "pnl_fees" in report.pnl_curve.columns:
            fees_cum = report.pnl_curve["pnl_fees"].abs().cumsum()
            step = max(1, len(fees_cum) // 1000)
            fig_cost = dark_fig(240)
            pline(fig_cost, fees_cum.index[::step], fees_cum.values[::step], "Fees cumulés", "#f85149")
            if "pnl_slippage" in report.pnl_curve.columns:
                slip_cum = report.pnl_curve["pnl_slippage"].abs().cumsum()
                pline(fig_cost, slip_cum.index[::step], slip_cum.values[::step], "Slippage cumulé", "#d29922")
            fig_cost.update_layout(title="Coûts cumulés (€)")
            st.plotly_chart(fig_cost, use_container_width=True)

    # ── PnL Estimé — avec fees réels Hyperliquid Futures ───────────────────
    section("Estimation PnL réalisé · Fees Hyperliquid Futures")

    st.markdown(
        '<div style="font-family:IBM Plex Sans;font-size:12px;color:#6e7681;'
        'padding:8px 12px;border-left:2px solid #1c2128;margin:4px 0 16px;">'
        '📌 Fees Hyperliquid Futures : <b>Maker −0.2 bps (rebate)</b> · <b>Taker +2.5 bps</b> · '
        'Funding 8h payé/reçu selon position. '
        'Scénarios : 100% taker (worst case), 50/50, 100% maker (best case).</div>',
        unsafe_allow_html=True,
    )

    pnl = report.pnl_curve.copy()
    eq_series   = report.equity_curve["equity"].astype(float)
    to_series   = report.turnover.reindex(pnl.index).fillna(0.0)
    eq_at_trade = eq_series.reindex(pnl.index).ffill().fillna(initial_equity)
    notional_traded = to_series * eq_at_trade

    HL_TAKER_BPS = 2.5
    HL_MAKER_BPS = -0.2
    HL_MIXED_BPS = (HL_TAKER_BPS + HL_MAKER_BPS) / 2

    fees_taker = notional_traded * (HL_TAKER_BPS / 1e4)
    fees_mixed = notional_traded * (HL_MIXED_BPS / 1e4)
    fees_maker = notional_traded * (HL_MAKER_BPS / 1e4)

    pnl_gross_step   = pnl["pnl_price"]
    pnl_funding_step = pnl["pnl_funding"] if "pnl_funding" in pnl.columns else pd.Series(0.0, index=pnl.index)

    pnl_taker = (pnl_gross_step + pnl_funding_step - fees_taker).cumsum()
    pnl_mixed = (pnl_gross_step + pnl_funding_step - fees_mixed).cumsum()
    pnl_maker = (pnl_gross_step + pnl_funding_step - fees_maker).cumsum()
    pnl_brut  = (pnl_gross_step + pnl_funding_step).cumsum()

    eq_taker = initial_equity + pnl_taker
    eq_mixed = initial_equity + pnl_mixed
    eq_maker = initial_equity + pnl_maker
    eq_brut  = initial_equity + pnl_brut

    step_s = max(1, len(eq_taker) // 2000)
    idx    = eq_taker.index[::step_s]

    fig_pnl_est = dark_fig(380)
    fig_pnl_est.add_trace(go.Scatter(
        x=idx, y=eq_brut.iloc[::step_s].values,
        mode="lines", name="Brut (sans fees)",
        line=dict(color="#6e7681", width=1.2, dash="dot"),
    ))
    fig_pnl_est.add_trace(go.Scatter(
        x=idx, y=eq_maker.iloc[::step_s].values,
        mode="lines", name=f"Maker only (−{abs(HL_MAKER_BPS)} bps rebate)",
        line=dict(color="#3fb950", width=1.5),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.04)",
    ))
    fig_pnl_est.add_trace(go.Scatter(
        x=idx, y=eq_mixed.iloc[::step_s].values,
        mode="lines", name=f"50/50 (+{HL_MIXED_BPS:.2f} bps)",
        line=dict(color="#d29922", width=1.8),
    ))
    fig_pnl_est.add_trace(go.Scatter(
        x=idx, y=eq_taker.iloc[::step_s].values,
        mode="lines", name=f"Taker only (+{HL_TAKER_BPS} bps) — worst case",
        line=dict(color="#f85149", width=1.5),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.04)",
    ))
    fig_pnl_est.add_hline(
        y=initial_equity, line_dash="dot", line_color="#444",
        annotation_text=f"Capital initial {initial_equity}€",
        annotation_font=dict(color="#6e7681", size=10),
    )
    fig_pnl_est.update_layout(
        title=dict(text="Equity estimée selon scénario de fees — données réelles Hyperliquid",
                   font=dict(size=13, color="#58a6ff")),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
    )
    st.plotly_chart(fig_pnl_est, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    total_notional = float(notional_traded.sum())
    final_brut  = float(pnl_brut.iloc[-1])
    final_taker = float(pnl_taker.iloc[-1])
    final_mixed = float(pnl_mixed.iloc[-1])
    final_maker = float(pnl_maker.iloc[-1])
    with col1: kpi("Brut (sans fees)",          f"{final_brut:+.1f}€",  "positive" if final_brut>=0  else "negative")
    with col2: kpi(f"Taker +{HL_TAKER_BPS} bps", f"{final_taker:+.1f}€","positive" if final_taker>=0 else "negative")
    with col3: kpi(f"50/50 +{HL_MIXED_BPS:.2f} bps", f"{final_mixed:+.1f}€","positive" if final_mixed>=0 else "negative")
    with col4: kpi(f"Maker −{abs(HL_MAKER_BPS)} bps", f"{final_maker:+.1f}€","positive" if final_maker>=0 else "negative")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_fees = dark_fig(260)
        fig_fees.add_trace(go.Scatter(
            x=idx, y=fees_taker.cumsum().reindex(eq_taker.index).iloc[::step_s].values,
            mode="lines", name=f"Taker +{HL_TAKER_BPS} bps", line=dict(color="#f85149", width=1.5)))
        fig_fees.add_trace(go.Scatter(
            x=idx, y=fees_mixed.cumsum().reindex(eq_taker.index).iloc[::step_s].values,
            mode="lines", name=f"50/50 +{HL_MIXED_BPS:.2f} bps", line=dict(color="#d29922", width=1.5)))
        fig_fees.add_trace(go.Scatter(
            x=idx, y=fees_maker.cumsum().reindex(eq_taker.index).iloc[::step_s].values,
            mode="lines", name=f"Maker −{abs(HL_MAKER_BPS)} bps", line=dict(color="#3fb950", width=1.5)))
        fig_fees.add_hline(y=0, line_dash="dot", line_color="#444")
        fig_fees.update_layout(title="Fees cumulés (€) par scénario",
                               legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
        st.plotly_chart(fig_fees, use_container_width=True)
    with col2:
        notional_cum  = notional_traded.cumsum()
        days_elapsed  = max((pnl.index[-1] - pnl.index[0]).total_seconds() / 86400, 1)
        daily_notional = float(notional_traded.sum()) / days_elapsed
        fee_drag_taker = daily_notional * HL_TAKER_BPS / 1e4
        fee_drag_mixed = daily_notional * HL_MIXED_BPS / 1e4
        fig_notional = dark_fig(260)
        fig_notional.add_trace(go.Scatter(
            x=notional_cum.index[::step_s], y=notional_cum.values[::step_s],
            mode="lines", name="Notionnel cumulé (€)",
            line=dict(color="#79c0ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(121,192,255,0.06)",
        ))
        fig_notional.update_layout(title=f"Notionnel tradé cumulé | {daily_notional:.0f}€/jour avg")
        st.plotly_chart(fig_notional, use_container_width=True)
        st.markdown(
            f'<div style="font-family:IBM Plex Mono;font-size:11px;color:#6e7681;'
            f'padding:8px 12px;background:#0d1117;border-radius:4px;margin-top:4px;">'
            f'Notionnel total : <b style="color:#e6edf3">{total_notional:,.0f}€</b><br>'
            f'Drag fees/jour · Taker : <b style="color:#f85149">−{fee_drag_taker:.2f}€</b> '
            f'· 50/50 : <b style="color:#d29922">−{fee_drag_mixed:.2f}€</b><br>'
            f'→ Break-even quotidien : <b style="color:#58a6ff">{fee_drag_mixed:.2f}€</b> de PnL brut min.</div>',
            unsafe_allow_html=True,
        )

    # Interprétation
    section("Interprétation")
    total_cost = abs(m.pnl_fees) + abs(m.pnl_slippage)
    cost_ratio = total_cost / max(abs(m.pnl_gross), 0.01)
    msgs = []
    if m.sharpe >= 1.5:
        msgs.append(("#3fb950", f"✅ Sharpe {m.sharpe:.2f} — excellent signal sur données réelles."))
    elif m.sharpe >= 0.5:
        msgs.append(("#d29922", f"⚠️ Sharpe {m.sharpe:.2f} — signal présent, optimise les coûts."))
    elif m.sharpe >= 0:
        msgs.append(("#d29922", f"⚠️ Sharpe {m.sharpe:.2f} — signal faible. Essaie de baisser z_in ou augmenter l'horizon."))
    else:
        msgs.append(("#f85149", f"❌ Sharpe {m.sharpe:.2f} — stratégie non rentable sur cette période."))
    if cost_ratio > 0.6:
        msgs.append(("#f85149", f"💸 Coûts = {cost_ratio*100:.0f}% du PnL brut — sur-trading. Augmente z_in ou z_out."))
    elif cost_ratio > 0.3:
        msgs.append(("#d29922", f"💸 Coûts = {cost_ratio*100:.0f}% du PnL brut — ratio acceptable."))
    else:
        msgs.append(("#3fb950", f"💚 Coûts = {cost_ratio*100:.0f}% du PnL brut — très efficace."))
    if abs(m.max_drawdown) > 0.15:
        msgs.append(("#f85149", f"📉 Drawdown max {m.max_drawdown*100:.1f}% — risque élevé pour {initial_equity}€."))
    if m.pnl_funding > 0:
        msgs.append(("#3fb950", f"💰 Funding overlay positif : +{m.pnl_funding:.1f}€ — carry exploitable."))

    for color, text in msgs:
        st.markdown(
            f'<div style="font-family:IBM Plex Sans;font-size:13px;color:{color};'
            f'padding:8px 12px;border-left:2px solid {color};margin:4px 0;">{text}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.caption("HyperStat · Backtest sur données réelles Hyperliquid · Ne constitue pas un conseil financier")
