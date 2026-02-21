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
from hyperstat.strategy.funding_divergence_signal import FundingDivergenceSignal, FDSConfig, FDSDiagnostics

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


@st.cache_data(show_spinner=False)
def compute_fds_analysis(_candles, _funding, coins, tf_min, fds_params):
    """
    Calcule le FDS sur données réelles et retourne les DataFrames
    nécessaires pour la validation IC et la décomposition des composantes.

    Retourne:
        gate_df        : DataFrame (T, N) du signal FDS ∈ [-1, 1]
        ic_series      : Series de l'IC de Spearman par timestamp
        breakdown_df   : DataFrame des 3 composantes moyennées
        returns_df     : DataFrame des log-returns
        funding_df     : DataFrame des funding rates
    """
    # ── Aligner les returns sur un index commun ─────────────────────────────
    close_dict = {}
    for c in coins:
        df = _candles.get(c)
        if df is None or df.empty:
            continue
        dfi = df.set_index("ts")["close"].astype(float)
        dfi.index = pd.to_datetime(dfi.index, utc=True)
        close_dict[c] = dfi

    if len(close_dict) < 3:
        return None, None, None, None, None

    close_df = pd.DataFrame(close_dict).sort_index().ffill()
    returns_df = np.log(close_df / close_df.shift(1)).dropna(how="all")

    # ── Aligner les funding rates ───────────────────────────────────────────
    fund_dict = {}
    for c in coins:
        df = _funding.get(c)
        if df is None or df.empty:
            continue
        dfi = df.set_index("ts")["rate"].astype(float)
        dfi.index = pd.to_datetime(dfi.index, utc=True)
        fund_dict[c] = dfi

    if len(fund_dict) < 3:
        return None, None, None, None, None

    # Resampler le funding sur le même index que les returns (ffill entre ticks)
    fund_raw = pd.DataFrame(fund_dict).reindex(returns_df.index, method="ffill")
    funding_df = fund_raw[returns_df.columns.intersection(fund_raw.columns)]

    # Garder seulement les coins présents dans les deux
    common = returns_df.columns.intersection(funding_df.columns).tolist()
    if len(common) < 3:
        return None, None, None, None, None

    returns_df = returns_df[common]
    funding_df = funding_df[common]

    # ── Calcul FDS batch ────────────────────────────────────────────────────
    cfg = FDSConfig(
        span_funding_fast=int(fds_params["span_fast"]),
        span_funding_slow=int(fds_params["span_slow"]),
        divergence_window=int(fds_params["div_window"]),
        w_carry=float(fds_params["w_carry"]),
        w_divergence=float(fds_params["w_div"]),
        w_velocity=float(fds_params["w_vel"]),
        gate_scale=float(fds_params["gate_scale"]),
        min_obs=int(fds_params["min_obs"]),
    )
    fds = FundingDivergenceSignal(cfg)
    diag = FDSDiagnostics(fds)

    gate_df      = fds.compute(returns_df, funding_df)
    ic_series    = diag.signal_ic(returns_df, funding_df,
                                  forward_horizon=int(fds_params["fwd_horizon"]))
    breakdown_df = diag.component_breakdown(returns_df, funding_df)

    return gate_df, ic_series, breakdown_df, returns_df, funding_df


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

    st.markdown("## FDS — Paramètres")
    fds_span_fast  = st.slider("span_fast (τf barres)", 2, 32, 8)
    fds_span_slow  = st.slider("span_slow (τs barres)", 12, 144, 72)
    fds_div_window = st.slider("divergence_window (W barres)", 8, 72, 24)
    fds_fwd_h      = st.slider("IC forward horizon (barres)", 4, 48, 12)
    fds_gate_scale = st.slider("gate_scale (α_fds)", 0.0, 1.0, 0.6, 0.05)
    fds_min_obs    = st.slider("min_obs (warm-up)", 12, 96, 48)
    w_carry = st.slider("w_carry",      0.0, 1.0, 0.35, 0.05)
    w_div   = st.slider("w_divergence", 0.0, 1.0, 0.40, 0.05)
    w_vel   = st.slider("w_velocity",   0.0, 1.0, 0.25, 0.05)
    total_w = w_carry + w_div + w_vel
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠ Poids FDS = {total_w:.2f} ≠ 1.0")

    fds_params = dict(
        span_fast=fds_span_fast, span_slow=fds_span_slow,
        div_window=fds_div_window, fwd_horizon=fds_fwd_h,
        gate_scale=fds_gate_scale, min_obs=fds_min_obs,
        w_carry=w_carry, w_div=w_div, w_vel=w_vel,
    )

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
# TABS  — ajout de l'onglet FDS
# ══════════════════════════════════════════════════════════════
tab_market, tab_corr, tab_signal, tab_fds, tab_backtest = st.tabs([
    "📈  MARCHÉ",
    "🔗  CORRÉLATIONS",
    "⚡  SIGNAUX Z-SCORE",
    "🧠  FDS · VALIDATION",
    "🏦  BACKTEST",
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — MARCHÉ
# ──────────────────────────────────────────────────────────────
with tab_market:
    section("Prix normalisés (base 100)")

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
# TAB 4 — FDS VALIDATION  (NOUVEAU)
# ──────────────────────────────────────────────────────────────
with tab_fds:
    st.markdown("""
    <div class="info-strip">
    🧠 <b>Funding Divergence Signal (FDS)</b> — validation sur données réelles Hyperliquid.<br>
    L'<b>IC (Information Coefficient)</b> mesure la corrélation de Spearman entre le signal FDS calculé
    à l'instant <i>t</i> et les returns réels qui arrivent à <i>t + H</i>.
    Un IC moyen > 0.03 avec t-stat > 2 confirme que le signal est prédictif et mérite d'être activé.
    </div>
    """, unsafe_allow_html=True)

    coins_with_funding = [c for c in coins_sel if c in funding_f and not funding_f[c].empty]
    if len(coins_with_funding) < 3:
        st.warning("⚠ Il faut au moins 3 coins avec des données de funding pour valider le FDS. "
                   "Vérifie que le dossier data/funding/ contient des fichiers 8h.parquet.")
        st.stop()

    with st.spinner("Calcul du FDS sur données réelles…"):
        gate_df, ic_series, breakdown_df, returns_df, funding_df = compute_fds_analysis(
            _candles=candles_f,
            _funding=funding_f,
            coins=coins_with_funding,
            tf_min=tf_min,
            fds_params=fds_params,
        )

    if gate_df is None:
        st.error("Impossible de calculer le FDS — vérifie que les données funding sont alignées avec les candles.")
        st.stop()

    ic_clean = ic_series.dropna()

    # ── KPIs IC ─────────────────────────────────────────────────────────────
    section("Information Coefficient (IC) — résumé")

    ic_mean  = float(ic_clean.mean()) if len(ic_clean) > 0 else float("nan")
    ic_std   = float(ic_clean.std())  if len(ic_clean) > 1 else float("nan")
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_clean))) if ic_std > 0 else float("nan")
    ic_pos_pct = float((ic_clean > 0).mean()) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        verdict = "positive" if ic_mean > 0.03 else ("neutral" if ic_mean > 0 else "negative")
        kpi("IC moyen", f"{ic_mean:+.4f}", verdict)
    with col2:
        verdict = "positive" if abs(ic_tstat) > 2 else ("neutral" if abs(ic_tstat) > 1 else "negative")
        kpi("t-stat IC", f"{ic_tstat:+.2f}", verdict)
    with col3:
        verdict = "positive" if ic_pos_pct > 52 else ("neutral" if ic_pos_pct > 48 else "negative")
        kpi("IC > 0 (%)", f"{ic_pos_pct:.1f}%", verdict)
    with col4:
        kpi("Observations", f"{len(ic_clean):,}", "neutral")

    # Interprétation automatique
    if ic_mean > 0.03 and abs(ic_tstat) > 2:
        st.markdown(
            '<div style="border-left:3px solid #3fb950;padding:8px 14px;background:#0d1f12;'
            'font-family:IBM Plex Sans;font-size:13px;color:#3fb950;margin:12px 0;">'
            f'✅ Signal FDS prédictif confirmé — IC={ic_mean:.4f}, t-stat={ic_tstat:.2f}. '
            'Tu peux activer le gate_scale et procéder à la calibration des poids.</div>',
            unsafe_allow_html=True)
    elif ic_mean > 0 and abs(ic_tstat) > 1:
        st.markdown(
            '<div style="border-left:3px solid #d29922;padding:8px 14px;background:#1a1505;'
            'font-family:IBM Plex Sans;font-size:13px;color:#d29922;margin:12px 0;">'
            f'⚠ Signal FDS faible — IC={ic_mean:.4f}, t-stat={ic_tstat:.2f}. '
            'Essaie d\'ajuster divergence_window ou span_fast dans la sidebar.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="border-left:3px solid #f85149;padding:8px 14px;background:#1a0505;'
            'font-family:IBM Plex Sans;font-size:13px;color:#f85149;margin:12px 0;">'
            f'❌ FDS non prédictif sur cette période — IC={ic_mean:.4f}, t-stat={ic_tstat:.2f}. '
            'Garde gate_scale=0 jusqu\'à obtenir plus de données ou d\'autres paramètres.</div>',
            unsafe_allow_html=True)

    # ── IC série temporelle ──────────────────────────────────────────────────
    section("IC par timestamp (rolling 20 barres)")
    col1, col2 = st.columns(2)

    with col1:
        fig_ic = dark_fig(300)
        ic_ma = ic_clean.rolling(20, min_periods=5).mean()
        step = max(1, len(ic_clean) // 2000)
        fig_ic.add_trace(go.Scatter(
            x=ic_clean.index[::step], y=ic_clean.values[::step],
            mode="lines", name="IC brut",
            line=dict(color="#6e7681", width=0.8), opacity=0.5,
        ))
        pline(fig_ic, ic_ma.index[::step], ic_ma.values[::step], "IC MA20", "#58a6ff", width=2.0)
        fig_ic.add_hline(y=0.03, line_dash="dash", line_color="#3fb950",
                         annotation_text="seuil 0.03")
        fig_ic.add_hline(y=0,    line_dash="dot",  line_color="#444")
        fig_ic.add_hline(y=-0.03, line_dash="dash", line_color="#f85149")
        fig_ic.update_layout(title=f"IC · forward {fds_fwd_h} barres")
        st.plotly_chart(fig_ic, use_container_width=True)

    with col2:
        fig_ic_hist = dark_fig(300)
        fig_ic_hist.add_trace(go.Histogram(
            x=ic_clean.values, nbinsx=50,
            marker_color="#58a6ff", opacity=0.75, name="IC distribution",
        ))
        fig_ic_hist.add_vline(x=0,    line_dash="dot",  line_color="#444")
        fig_ic_hist.add_vline(x=0.03, line_dash="dash", line_color="#3fb950",
                              annotation_text="0.03")
        fig_ic_hist.add_vline(x=ic_mean, line_dash="solid", line_color="#d29922",
                              annotation_text=f"mean={ic_mean:.4f}")
        fig_ic_hist.update_layout(title="Distribution IC", showlegend=False)
        st.plotly_chart(fig_ic_hist, use_container_width=True)

    # ── Décomposition des 3 composantes ─────────────────────────────────────
    section("Décomposition des composantes FDS")
    st.markdown(
        '<div style="font-family:IBM Plex Sans;font-size:12px;color:#6e7681;margin-bottom:12px;">'
        '📌 Carry = z-score cross-sectionnel du funding lent (contrarian funding élevé). '
        'Divergence = désalignement prix/funding (tension maximale quand ρ ≈ 0). '
        'Velocity = accélération du funding (overcrowding détection).</div>',
        unsafe_allow_html=True)

    if breakdown_df is not None and not breakdown_df.empty:
        fig_bd = dark_fig(320)
        colors_bd = {"carry_mean": "#58a6ff", "divergence_mean": "#3fb950",
                     "velocity_mean": "#d29922", "fds_gate_mean": "#e6edf3"}
        widths_bd = {"carry_mean": 1.2, "divergence_mean": 1.2,
                     "velocity_mean": 1.2, "fds_gate_mean": 2.2}
        labels_bd = {"carry_mean": "Carry (35%)", "divergence_mean": "Divergence (40%)",
                     "velocity_mean": "Velocity (25%)", "fds_gate_mean": "FDS final"}
        for col_name in ["carry_mean", "divergence_mean", "velocity_mean", "fds_gate_mean"]:
            if col_name not in breakdown_df.columns:
                continue
            s = breakdown_df[col_name].dropna()
            step = max(1, len(s) // 2000)
            pline(fig_bd, s.index[::step], s.values[::step],
                  labels_bd[col_name], colors_bd[col_name], widths_bd[col_name])
        fig_bd.add_hline(y=0, line_dash="dot", line_color="#444")
        fig_bd.update_layout(title="Composantes FDS moyennées cross-section")
        st.plotly_chart(fig_bd, use_container_width=True)

    # ── Heatmap du gate FDS par coin ─────────────────────────────────────────
    section("Gate FDS par coin (dernières 100 barres)")
    gate_tail = gate_df.tail(100).T
    if not gate_tail.empty:
        fig_heat = go.Figure(data=go.Heatmap(
            z=gate_tail.values,
            x=[str(c)[:10] for c in gate_tail.columns],
            y=list(gate_tail.index),
            colorscale=[[0,"#f85149"],[0.5,"#161b22"],[1,"#3fb950"]],
            zmid=0, zmin=-1, zmax=1,
            hovertemplate="t=%{x}<br>coin=%{y}<br>FDS=%{z:.3f}<extra></extra>",
            colorbar=dict(title="FDS", tickfont=dict(family="IBM Plex Mono", size=10)),
        ))
        fig_heat.update_layout(
            template="plotly_dark", height=max(300, len(gate_tail) * 22),
            paper_bgcolor=PAPER, plot_bgcolor=BG,
            margin=dict(l=12, r=12, t=36, b=12),
            font=dict(family="IBM Plex Mono", size=10, color="#8b949e"),
            title=dict(text="Gate FDS par coin · vert=renforce · rouge=atténue",
                       font=dict(size=13, color="#58a6ff")),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Guide calibration ───────────────────────────────────────────────────
    section("Guide de calibration")
    st.markdown("""
    <div style="font-family:IBM Plex Mono;font-size:11px;color:#6e7681;
    background:#0d1117;border:1px solid #1c2128;border-radius:6px;padding:14px 18px;line-height:2;">
    <b style="color:#58a6ff">Ordre recommandé (walk-forward, fixer les autres à leur valeur par défaut) :</b><br>
    1. <b style="color:#e6edf3">divergence_window</b> — teste 12, 24, 48 barres → prend celui qui maximise t-stat<br>
    2. <b style="color:#e6edf3">span_fast (τf)</b> — teste 4, 8, 16 barres<br>
    3. <b style="color:#e6edf3">w_carry / w_divergence / w_velocity</b> — grid search autour de (0.35, 0.40, 0.25)<br>
    4. <b style="color:#e6edf3">gate_scale (α_fds)</b> — commence à 0.4, augmente seulement si IC > 0.03 et t-stat > 2<br><br>
    <b style="color:#3fb950">Seuils de décision :</b><br>
    IC > 0.03 et t-stat > 2.0 → signal prédictif → activer gate_scale ≥ 0.4<br>
    IC > 0.05 et t-stat > 3.0 → signal fort → gate_scale jusqu'à 0.8<br>
    IC ≤ 0.01 ou t-stat ≤ 1.0 → garder gate_scale = 0 (ne pas activer)
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 5 — BACKTEST
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

    # ── Ligne 1 : Rendement & Ratios performance/risque ──────────────────────
    section("Rendement & Ratios")
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    with col1: kpi("Return total", _pct(m.total_return),
                   "positive" if m.total_return >= 0 else "negative")
    with col2: kpi("CAGR",        _pct(m.cagr, 1),
                   "positive" if m.cagr >= 0 else "negative")
    with col3: kpi("Sharpe",      _num(m.sharpe, 2),
                   "positive" if m.sharpe >= 1 else ("neutral" if m.sharpe >= 0 else "negative"))
    with col4: kpi("Sortino",     _num(m.sortino, 2),
                   "positive" if m.sortino >= 1.5 else ("neutral" if m.sortino >= 0 else "negative"))
    with col5: kpi("Calmar",      _num(m.calmar, 2),
                   "positive" if m.calmar >= 1 else ("neutral" if m.calmar >= 0 else "negative"))
    with col6: kpi("Vol ann.",    _pct(m.ann_vol, 1), "neutral")

    # ── Ligne 2 : Risque ─────────────────────────────────────────────────────
    section("Risque")
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1: kpi("Max Drawdown",     _pct(m.max_drawdown),     "negative")
    with col2: kpi("DD moyen",         _pct(m.avg_drawdown),     "negative")
    with col3: kpi("Durée DD max",
                   f"{m.max_dd_duration_bars:,} bar",             "neutral")
    with col4: kpi("VaR 95%",          _pct(m.var_95),
                   "negative" if np.isfinite(m.var_95) else "neutral")
    with col5: kpi("Barres",           f"{m.n_steps:,}",          "neutral")

    # ── Ligne 3 : Robustesse ─────────────────────────────────────────────────
    section("Robustesse")
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1: kpi("Win Rate",       _pct(m.win_rate, 1),
                   "positive" if m.win_rate >= 0.52 else ("neutral" if m.win_rate >= 0.48 else "negative"))
    with col2: kpi("Profit Factor",  _num(m.profit_factor, 2),
                   "positive" if m.profit_factor >= 1.5 else ("neutral" if m.profit_factor >= 1 else "negative"))
    with col3: kpi("Gain/Loss",      _num(m.avg_gain_loss_ratio, 2), "neutral")
    with col4: kpi("Kelly ★",        _num(m.kelly_fraction, 3),      "neutral")
    with col5: kpi("Coins actifs",   str(len(coins_sel)),             "neutral")

    # ── Ligne 4 : PnL décomposé ──────────────────────────────────────────────
    section("PnL décomposé")
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1: kpi("PnL brut",    _num(m.pnl_gross),
                   "positive" if m.pnl_gross >= 0 else "negative")
    with col2: kpi("PnL funding", _num(m.pnl_funding),
                   "positive" if m.pnl_funding >= 0 else "negative")
    with col3: kpi("Fees",        f"−{_num(abs(m.pnl_fees))}",     "negative")
    with col4: kpi("Slippage",    f"−{_num(abs(m.pnl_slippage))}", "negative")
    with col5: kpi("PnL net",     _num(m.pnl_net),
                   "positive" if m.pnl_net >= 0 else "negative")

    # ── Equity curve + Drawdown ───────────────────────────────────────────────
    section("Equity curve & Drawdown")
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
        x=dd.index, y=dd.values * 100, mode="lines", name="Drawdown (%)",
        line=dict(color="#f85149", width=1.2),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.10)",
    ), row=2, col=1)
    fig_eq.add_hline(y=-max_dd * 100, line_dash="dash", line_color="#d29922",
                     annotation_text=f"Kill-switch {-max_dd*100:.0f}%", row=2, col=1)
    if np.isfinite(m.avg_drawdown):
        fig_eq.add_hline(y=m.avg_drawdown * 100, line_dash="dot", line_color="#bc8cff",
                         annotation_text=f"DD moyen {m.avg_drawdown*100:.1f}%", row=2, col=1)
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

    # ── Distribution des returns + Drawdown durée ────────────────────────────
    section("Analyse de la distribution des returns")
    r_series = eq.pct_change().dropna() * 100  # en %
    col1, col2 = st.columns(2)

    with col1:
        fig_ret = dark_fig(280)
        fig_ret.add_trace(go.Histogram(
            x=r_series.values, nbinsx=80,
            name="Returns",
            marker_color="#58a6ff", opacity=0.7,
        ))
        # VaR line
        if np.isfinite(m.var_95):
            fig_ret.add_vline(x=-m.var_95 * 100, line_dash="dash", line_color="#f85149",
                              annotation_text=f"VaR 95% = {m.var_95*100:.2f}%")
        fig_ret.add_vline(x=0, line_dash="dot", line_color="#444")
        wins_pct = float((r_series > 0).mean()) * 100
        fig_ret.update_layout(
            title=f"Distribution returns — Win {wins_pct:.1f}% / Loss {100-wins_pct:.1f}%",
            showlegend=False,
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with col2:
        # Drawdown profondeur au fil du temps
        fig_dd2 = dark_fig(280)
        step = max(1, len(dd) // 2000)
        fig_dd2.add_trace(go.Scatter(
            x=dd.index[::step], y=dd.values[::step] * 100,
            mode="lines", name="Drawdown (%)",
            line=dict(color="#f85149", width=1.2),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.10)",
        ))
        if np.isfinite(m.avg_drawdown):
            fig_dd2.add_hline(y=m.avg_drawdown * 100, line_dash="dot", line_color="#bc8cff",
                              annotation_text=f"DD moyen {m.avg_drawdown*100:.1f}%")
        fig_dd2.add_hline(y=m.max_drawdown * 100, line_dash="dash", line_color="#f85149",
                          annotation_text=f"Max DD {m.max_drawdown*100:.1f}%")
        fig_dd2.update_layout(title=f"Drawdown — durée max {m.max_dd_duration_bars:,} barres")
        st.plotly_chart(fig_dd2, use_container_width=True)

    # ── PnL décomposé ─────────────────────────────────────────────────────────
    section("Décomposition PnL cumulé")
    col1, col2 = st.columns(2)
    with col1:
        pnl = report.pnl_curve
        fig_pnl = dark_fig(300)
        for col_name, color, label in [
            ("pnl_net_step", "#58a6ff", "Net"),
            ("pnl_price",    "#3fb950", "Prix"),
            ("pnl_funding",  "#d29922", "Funding"),
        ]:
            if col_name in pnl.columns:
                s = pnl[col_name].cumsum()
                step = max(1, len(s) // 2000)
                pline(fig_pnl, s.index[::step], s.values[::step], label, color)
        fig_pnl.add_hline(y=0, line_dash="dot", line_color="#444")
        fig_pnl.update_layout(title="PnL cumulé par composante")
        st.plotly_chart(fig_pnl, use_container_width=True)

    with col2:
        labels = ["PnL brut", "Funding", "Fees", "Slippage", "PnL net"]
        values = [m.pnl_gross, m.pnl_funding, -abs(m.pnl_fees),
                  -abs(m.pnl_slippage), m.pnl_net]
        colors_bar = ["#3fb950" if v >= 0 else "#f85149" for v in values]
        fig_bar = dark_fig(300)
        fig_bar.add_trace(go.Bar(
            x=labels, y=values, marker_color=colors_bar,
            text=[f"{v:+.1f}€" for v in values], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11),
        ))
        fig_bar.add_hline(y=0, line_dash="dot", line_color="#444")
        fig_bar.update_layout(title="Bilan coûts / PnL (€)", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Poids & Exposition ────────────────────────────────────────────────────
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

    # ── Tableau récapitulatif toutes métriques ────────────────────────────────
    section("Tableau complet des métriques")
    _metric_rows = [
        ("Rendement",     [("Return total",       _pct(m.total_return)),
                           ("CAGR",               _pct(m.cagr, 1))]),
        ("Ratios",        [("Sharpe",             _num(m.sharpe, 3)),
                           ("Sortino",            _num(m.sortino, 3)),
                           ("Calmar",             _num(m.calmar, 3)),
                           ("Treynor",            _num(m.treynor, 3))]),
        ("Risque",        [("Vol annualisée",     _pct(m.ann_vol, 1)),
                           ("Max Drawdown",       _pct(m.max_drawdown)),
                           ("DD moyen",           _pct(m.avg_drawdown)),
                           ("Durée DD max (bar)", f"{m.max_dd_duration_bars:,}"),
                           ("VaR 95%",            _pct(m.var_95))]),
        ("Robustesse",    [("Win Rate",           _pct(m.win_rate, 1)),
                           ("Loss Rate",          _pct(m.loss_rate, 1)),
                           ("Profit Factor",      _num(m.profit_factor, 3)),
                           ("Gain/Loss moyen",    _num(m.avg_gain_loss_ratio, 3)),
                           ("Kelly ★",            _num(m.kelly_fraction, 4))]),
        ("Exposition",    [("Gross moyen",        _num(m.avg_gross, 3)),
                           ("Net moyen",          _num(m.avg_net, 4)),
                           ("Turnover moyen",     _num(m.avg_turnover, 4))]),
        ("Corr. marché",  [("Bêta (BTC)",         _num(m.beta, 3)),
                           ("R²",                 _num(m.r_squared, 3)),
                           ("Alpha Jensen",       _pct(m.jensen_alpha, 2))]),
        ("PnL (€)",       [("PnL brut",           _num(m.pnl_gross)),
                           ("PnL funding",        _num(m.pnl_funding)),
                           ("Fees",               f"−{_num(abs(m.pnl_fees))}"),
                           ("Slippage",           f"−{_num(abs(m.pnl_slippage))}"),
                           ("PnL net",            _num(m.pnl_net))]),
    ]
    col_left, col_right = st.columns(2)
    split = len(_metric_rows) // 2 + 1
    for col_side, rows_slice in [(col_left, _metric_rows[:split]),
                                  (col_right, _metric_rows[split:])]:
        with col_side:
            for group_name, fields in rows_slice:
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono;font-size:10px;color:#58a6ff;'
                    f'letter-spacing:1.5px;text-transform:uppercase;margin:14px 0 4px;">'
                    f'{group_name}</div>', unsafe_allow_html=True)
                for label, val in fields:
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono;font-size:12px;'
                        f'display:flex;justify-content:space-between;padding:3px 0;'
                        f'border-bottom:1px solid #1c2128;">'
                        f'<span style="color:#6e7681">{label}</span>'
                        f'<span style="color:#e6edf3">{val}</span></div>',
                        unsafe_allow_html=True)

    st.markdown(
        '<div class="warn-strip" style="margin-top:10px;">★ Kelly Fraction = indicative only. '
        'Apply at most 25% of this value in practice. '
        'Corr. marché = NaN si BTC n\'est pas utilisé comme market_returns dans compute_performance_metrics().</div>',
        unsafe_allow_html=True)

    # ── Interprétation automatique enrichie ──────────────────────────────────
    section("Interprétation automatique")
    total_cost = abs(m.pnl_fees) + abs(m.pnl_slippage)
    cost_ratio = total_cost / max(abs(m.pnl_gross), 0.01)
    msgs = []

    # Sharpe
    if m.sharpe >= 1.5:
        msgs.append(("#3fb950", f"✅ Sharpe {m.sharpe:.2f} — excellent sur données réelles."))
    elif m.sharpe >= 0.5:
        msgs.append(("#d29922", f"⚠ Sharpe {m.sharpe:.2f} — signal présent, optimise les coûts."))
    elif m.sharpe >= 0:
        msgs.append(("#d29922", f"⚠ Sharpe {m.sharpe:.2f} — signal faible. Baisse z_in ou augmente l'horizon."))
    else:
        msgs.append(("#f85149", f"❌ Sharpe {m.sharpe:.2f} — stratégie non rentable sur cette période."))

    # Sortino (downside uniquement)
    if np.isfinite(m.sortino):
        if m.sortino >= 2.0:
            msgs.append(("#3fb950", f"✅ Sortino {m.sortino:.2f} — bonne asymétrie gain/perte."))
        elif m.sortino < 0.5 and m.sortino >= 0:
            msgs.append(("#d29922", f"⚠ Sortino {m.sortino:.2f} — downside vol élevée, pertes concentrées."))
        elif m.sortino < 0:
            msgs.append(("#f85149", f"❌ Sortino {m.sortino:.2f} — rendement net négatif."))

    # Calmar
    if np.isfinite(m.calmar):
        if m.calmar >= 1.0:
            msgs.append(("#3fb950", f"✅ Calmar {m.calmar:.2f} — CAGR > MDD, profil risque sain."))
        elif m.calmar < 0.5:
            msgs.append(("#d29922", f"⚠ Calmar {m.calmar:.2f} — drawdown trop grand vs CAGR. Réduis le gross_target."))

    # Coûts
    if cost_ratio > 0.6:
        msgs.append(("#f85149", f"💸 Coûts = {cost_ratio*100:.0f}% du PnL brut — sur-trading. Augmente z_in ou z_out."))
    elif cost_ratio > 0.3:
        msgs.append(("#d29922", f"💸 Coûts = {cost_ratio*100:.0f}% du PnL brut — acceptable."))
    else:
        msgs.append(("#3fb950", f"💚 Coûts = {cost_ratio*100:.0f}% du PnL brut — très efficace."))

    # Drawdown
    if abs(m.max_drawdown) > 0.15:
        msgs.append(("#f85149", f"📉 Max Drawdown {m.max_drawdown*100:.1f}% — risque élevé pour {initial_equity}€."))

    # Win rate & profit factor
    if np.isfinite(m.win_rate) and np.isfinite(m.profit_factor):
        if m.profit_factor >= 1.5 and m.win_rate >= 0.50:
            msgs.append(("#3fb950",
                f"✅ Win {m.win_rate*100:.1f}% · Profit Factor {m.profit_factor:.2f} — robustesse confirmée."))
        elif m.profit_factor < 1.0:
            msgs.append(("#f85149",
                f"❌ Profit Factor {m.profit_factor:.2f} < 1 — pertes > gains en absolu."))
        elif m.win_rate < 0.45:
            msgs.append(("#d29922",
                f"⚠ Win Rate {m.win_rate*100:.1f}% faible — la stratégie gagne surtout par ampleur des gains (vérifier avg_gain_loss_ratio)."))

    # VaR
    if np.isfinite(m.var_95):
        msgs.append(("#6e7681",
            f"📊 VaR 95% = {m.var_95*100:.2f}% par barre — perte max attendue sur 1 barre dans 95% des cas."))

    # Funding
    if m.pnl_funding > 0:
        msgs.append(("#3fb950", f"💰 Funding overlay positif : +{m.pnl_funding:.1f}€ — carry exploitable."))

    for color, text in msgs:
        st.markdown(
            f'<div style="font-family:IBM Plex Sans;font-size:13px;color:{color};'
            f'padding:8px 12px;border-left:2px solid {color};margin:4px 0;">{text}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.caption("HyperStat · Backtest sur données réelles Hyperliquid · Ne constitue pas un conseil financier")
