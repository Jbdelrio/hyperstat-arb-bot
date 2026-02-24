# apps/live_dashboard.py
"""
Live Paper Trading Dashboard — Hyperliquid Exchange
====================================================
Connecte aux flux temps réel Hyperliquid (WebSocket allMids).
Simule la stratégie stat-arb en paper trading — aucun argent réel.

Architecture :
    BackgroundEngine (thread daemon + asyncio)
        └── WebSocket Hyperliquid (stream_all_mids)
        └── StatArbStrategy.update() à chaque barre
        └── PaperPortfolio (simule fills au mid)
        └── SharedState (thread-safe)
                ↓
    Streamlit UI — lit SharedState, affiche métriques / signaux / positions
                ↓
    st.rerun() toutes les N secondes (delta WebSocket, sans flash)

Lancement :
    streamlit run apps/live_dashboard.py
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

from hyperstat.exchange.hyperliquid.endpoints import HyperliquidEndpoints
from hyperstat.exchange.hyperliquid.rest_client import HyperliquidRestClient
from hyperstat.exchange.hyperliquid.rate_limiter import RateLimiter, RateLimiterConfig
from hyperstat.exchange.hyperliquid.ws_client import HyperliquidWsClient
from hyperstat.exchange.hyperliquid.market_data import HyperliquidMarketData
from hyperstat.strategy.stat_arb import StatArbStrategy, StatArbConfig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HyperStat — Live Paper Trading",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }

    .badge-live        { background:#1a7a3c; color:#fff; padding:4px 14px; border-radius:14px;
                         font-size:13px; font-weight:700; animation:pulse 2s infinite; }
    .badge-connecting  { background:#2a3a6e; color:#fff; padding:4px 14px; border-radius:14px;
                         font-size:13px; font-weight:700; }
    .badge-warming_up  { background:#8b6e1a; color:#fff; padding:4px 14px; border-radius:14px;
                         font-size:13px; font-weight:700; }
    .badge-error       { background:#8b1a1a; color:#fff; padding:4px 14px; border-radius:14px;
                         font-size:13px; font-weight:700; }
    .badge-idle        { background:#3a3a3a; color:#aaa; padding:4px 14px; border-radius:14px;
                         font-size:13px; font-weight:700; }

    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.65; } }

    .section-header {
        color: #7289da; font-size: 13px; font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; border-bottom: 1px solid #2a2d3a;
        padding-bottom: 6px; margin: 16px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EQUITY    = 10_000.0
DEFAULT_GROSS_LEV = 1.5
FEE_RATE          = 0.00035    # ~3.5 bps taker Hyperliquid — no market impact assumed
MAX_EQUITY_PTS    = 5_000      # Taille max de l'historique equity en mémoire
MIN_BAR_COINS     = 2          # Minimum de coins avec prix pour déclencher une barre

# Timeframes disponibles (label → secondes)
TF_OPTIONS: Dict[str, int] = {
    "10s": 10, "30s": 30,
    "1m": 60, "3m": 180, "5m": 300,
    "15m": 900, "30m": 1800, "1h": 3600,
}
TF_DEFAULT        = "5m"       # index par défaut
HORIZON_BARS_MIN  = 12         # jamais moins de 12 barres de warmup
HORIZON_BARS_MAX  = 90         # cap pour éviter une attente infinie sur petits TF

def _horizon_bars(tf_seconds: int) -> int:
    """Cible ~1h de lookback horizon, encadré entre MIN et MAX."""
    return max(HORIZON_BARS_MIN, min(HORIZON_BARS_MAX, 3600 // tf_seconds))

FALLBACK_COINS    = [          # Si la récupération de l'univers échoue
    "BTC", "ETH", "SOL", "AVAX", "LINK", "ARB", "OP",
    "MATIC", "DOT", "ADA", "BNB", "NEAR", "FTM", "ATOM",
]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (thread-safe entre BackgroundEngine et Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

class SharedState:
    """
    État partagé entre le thread asyncio (BackgroundEngine) et le thread Streamlit.
    Toutes les écritures passent par _lock ; snapshot() retourne une copie sûre.
    """

    def __init__(self) -> None:
        self._lock           = threading.RLock()
        self.status: str     = "idle"   # idle | connecting | warming_up | live | error
        self.error: str      = ""

        # Config runtime
        self.tf_seconds: int             = TF_OPTIONS[TF_DEFAULT]
        self.horizon_bars: int           = _horizon_bars(self.tf_seconds)

        # Données de marché
        self.mids: Dict[str, float]      = {}
        self.universe: List[str]         = []
        self.selected: List[str]         = []
        self.last_update: Optional[datetime] = None

        # Sorties stratégie
        self.zscores: Dict[str, float]   = {}
        self.weights: Dict[str, float]   = {}
        self.signal_ts: Optional[datetime] = None

        # Portfolio paper
        self.equity: float               = DEFAULT_EQUITY
        self.initial_equity: float       = DEFAULT_EQUITY
        self.equity_history: deque       = deque(maxlen=MAX_EQUITY_PTS)  # [(ts_iso, equity)]
        self.positions: Dict[str, Dict]  = {}
        self.n_bars: int                 = 0
        self.n_wins: int                 = 0
        self.n_losses: int               = 0
        self.realized_pnl: float         = 0.0
        self.total_fees: float           = 0.0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status"        : self.status,
                "error"         : self.error,
                "tf_seconds"    : self.tf_seconds,
                "horizon_bars"  : self.horizon_bars,
                "mids"          : dict(self.mids),
                "universe"      : list(self.universe),
                "selected"      : list(self.selected),
                "last_update"   : self.last_update,
                "zscores"       : dict(self.zscores),
                "weights"       : dict(self.weights),
                "signal_ts"     : self.signal_ts,
                "equity"        : self.equity,
                "initial_equity": self.initial_equity,
                "equity_history": list(self.equity_history),
                "positions"     : {k: dict(v) for k, v in self.positions.items()},
                "n_bars"        : self.n_bars,
                "n_wins"        : self.n_wins,
                "n_losses"      : self.n_losses,
                "realized_pnl"  : self.realized_pnl,
                "total_fees"    : self.total_fees,
            }


# ─────────────────────────────────────────────────────────────────────────────
# PAPER PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

class PaperPortfolio:
    """
    Simule les trades au mid price avec commission FEE_RATE × notional.
    Pas d'execution réelle — paper trading uniquement.
    """

    def __init__(self, initial_equity: float, gross_leverage: float) -> None:
        self.equity        = initial_equity
        self.gross_leverage = gross_leverage
        self._pos: Dict[str, float]      = {}   # sym -> qty
        self._entry: Dict[str, float]    = {}   # sym -> avg entry price
        self.realized_pnl = 0.0
        self.total_fees   = 0.0
        self.n_wins       = 0
        self.n_losses     = 0

    def rebalance(self, target_weights: Dict[str, float], mids: Dict[str, float]) -> None:
        """Rebalance le portfolio vers target_weights (déjà normalisés au gross leverage)."""
        total_eq = self.equity + self._unrealized(mids)

        for sym, tgt_w in target_weights.items():
            px = mids.get(sym)
            if not px or not math.isfinite(px) or px <= 0:
                continue

            tgt_qty = (tgt_w * total_eq) / px
            cur_qty = self._pos.get(sym, 0.0)
            delta   = tgt_qty - cur_qty

            if abs(delta * px) < 0.5:           # Ignore les trades < 50 cts
                continue

            fee = abs(delta * px) * FEE_RATE

            # Réalisation du PnL sur fermetures partielles / totales
            if cur_qty != 0.0 and math.copysign(1, delta) != math.copysign(1, cur_qty):
                close_qty = min(abs(delta), abs(cur_qty)) * math.copysign(1, cur_qty)
                rpnl      = close_qty * (px - self._entry.get(sym, px))
                self.realized_pnl += rpnl
                self.equity += rpnl   # crédit du PnL réalisé dans le cash
                if rpnl > 0:
                    self.n_wins += 1
                else:
                    self.n_losses += 1

            new_qty = cur_qty + delta
            if abs(new_qty) < 1e-9:
                self._pos.pop(sym, None)
                self._entry.pop(sym, None)
            else:
                self._pos[sym] = new_qty
                if cur_qty == 0.0 or math.copysign(1, new_qty) != math.copysign(1, cur_qty):
                    # Nouvelle position ou flip de direction → entry = prix courant
                    self._entry[sym] = px
                elif math.copysign(1, delta) == math.copysign(1, cur_qty):
                    # Ajout à la position (même sens) → prix moyen pondéré
                    self._entry[sym] = (
                        abs(cur_qty) * self._entry.get(sym, px) + abs(delta) * px
                    ) / abs(new_qty)
                # else: réduction de position (même sens restant) → entry inchangé
                # Raison: les parts restantes ont toujours le même prix d'entrée original.

            self.equity   -= fee
            self.total_fees += fee

    def _unrealized(self, mids: Dict[str, float]) -> float:
        return sum(
            qty * (mids[sym] - self._entry.get(sym, mids[sym]))
            for sym, qty in self._pos.items()
            if sym in mids and math.isfinite(mids[sym])
        )

    def total_equity(self, mids: Dict[str, float]) -> float:
        return self.equity + self._unrealized(mids)

    def positions_snapshot(
        self,
        mids: Dict[str, float],
        zscores: Dict[str, float],
        weights: Dict[str, float],
    ) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for sym, qty in self._pos.items():
            if abs(qty) < 1e-9:
                continue
            px    = mids.get(sym, 0.0)
            entry = self._entry.get(sym, px)
            out[sym] = {
                "qty"       : qty,
                "entry_px"  : entry,
                "current_px": px,
                "upnl"      : qty * (px - entry) if math.isfinite(px) else 0.0,
                "notional"  : abs(qty * px),
                "zscore"    : zscores.get(sym, 0.0),
                "weight"    : weights.get(sym, 0.0),
            }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND ENGINE  (thread daemon + asyncio)
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundEngine:
    """
    Tourne dans un thread daemon avec son propre event loop asyncio.
    - Connecte au WebSocket Hyperliquid (stream_all_mids)
    - Forme des barres toutes les tf_minutes secondes
    - Lance StatArbStrategy.update() à chaque barre
    - Met à jour SharedState en temps réel
    """

    def __init__(self, state: SharedState, cfg: Dict[str, Any]) -> None:
        self.state  = state
        self.cfg    = cfg
        self._loop  = asyncio.new_event_loop()
        self._stop  = False
        self._thread = threading.Thread(
            target=self._run_thread, daemon=True, name="HL-PaperEngine"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ── Thread entry point ──────────────────────────────────────────────────

    def _run_thread(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async())
        except Exception as exc:
            with self.state._lock:
                self.state.status = "error"
                self.state.error  = str(exc)

    # ── Async main ──────────────────────────────────────────────────────────

    async def _run_async(self) -> None:
        with self.state._lock:
            self.state.status = "connecting"
            self.state.error  = ""

        # ── Clients Hyperliquid ──
        ep  = (HyperliquidEndpoints.mainnet()
               if self.cfg["network"] == "mainnet"
               else HyperliquidEndpoints.testnet())
        rl   = RateLimiter(RateLimiterConfig(weight_budget_per_minute=600))
        rest = HyperliquidRestClient(endpoints=ep, rate_limiter=rl)
        ws_c = HyperliquidWsClient(endpoints=ep)
        mkt  = HyperliquidMarketData(rest, ws_c)

        await ws_c.start()

        # ── Fetch univers (REST) ──
        try:
            meta     = await mkt.meta()
            universe = sorted([a["name"] for a in meta.get("universe", [])])
        except Exception:
            universe = FALLBACK_COINS

        selected = self.cfg["selected"]
        tf_sec   = self.cfg["tf_seconds"]
        h_bars   = _horizon_bars(tf_sec)

        with self.state._lock:
            self.state.universe      = universe
            self.state.selected      = selected
            self.state.initial_equity = self.cfg["initial_equity"]
            self.state.equity         = self.cfg["initial_equity"]
            self.state.tf_seconds     = tf_sec
            self.state.horizon_bars   = h_bars
            self.state.status         = "warming_up"

        # ── Stratégie + portfolio ──
        strategy  = StatArbStrategy(StatArbConfig(
            timeframe_minutes=max(1, tf_sec // 60),
            horizon_bars=h_bars,
        ))
        buckets   = {"live": selected}
        portfolio = PaperPortfolio(
            initial_equity=self.cfg["initial_equity"],
            gross_leverage=self.cfg["gross_leverage"],
        )

        # ── État interne du bar builder ──
        latest_mids: Dict[str, float]   = {}
        last_bar_ts: Optional[datetime] = None

        # ── Callback WebSocket allMids ──
        def _on_mids(msg: Any) -> None:
            nonlocal latest_mids, last_bar_ts

            # Extraction du dict de prix depuis les différentes formes possibles
            if isinstance(msg, dict):
                data = msg.get("data", msg)
                raw  = data.get("mids", data) if isinstance(data, dict) else {}
            else:
                return

            for sym, px_raw in raw.items():
                try:
                    latest_mids[sym] = float(px_raw)
                except (ValueError, TypeError):
                    pass

            now = datetime.now(timezone.utc)
            with self.state._lock:
                self.state.mids        = {s: latest_mids[s] for s in selected if s in latest_mids}
                self.state.last_update = now

            # ── Déclenchement d'une barre ──
            if last_bar_ts is not None and (now - last_bar_ts).total_seconds() < tf_sec:
                return
            last_bar_ts = now

            bar_mids = {
                s: latest_mids[s]
                for s in selected
                if s in latest_mids and math.isfinite(latest_mids[s]) and latest_mids[s] > 0
            }
            if len(bar_mids) < MIN_BAR_COINS:
                return

            # ── Signal stratégie ──
            signal = strategy.update(now, bar_mids, buckets)

            # Normalise les poids au gross leverage cible
            raw_w   = signal.weights_raw
            total_w = sum(abs(w) for w in raw_w.values())
            if total_w > 1e-9:
                scale = portfolio.gross_leverage / total_w
                norm_w = {k: v * scale for k, v in raw_w.items()}
            else:
                norm_w = raw_w

            # ── Rebalance portfolio paper ──
            portfolio.rebalance(norm_w, bar_mids)

            total_eq   = portfolio.total_equity(bar_mids)
            n_bars_new = (self.state.n_bars or 0) + 1
            new_status = "live" if n_bars_new >= h_bars else "warming_up"
            positions  = portfolio.positions_snapshot(bar_mids, signal.zscores, norm_w)

            with self.state._lock:
                self.state.equity_history.append((now.isoformat(), total_eq))
                self.state.equity       = total_eq
                self.state.zscores      = dict(signal.zscores)
                self.state.weights      = dict(norm_w)
                self.state.signal_ts    = now
                self.state.positions    = positions
                self.state.n_bars       = n_bars_new
                self.state.n_wins       = portfolio.n_wins
                self.state.n_losses     = portfolio.n_losses
                self.state.realized_pnl = portfolio.realized_pnl
                self.state.total_fees   = portfolio.total_fees
                self.state.status       = new_status

        # ── Souscription WebSocket ──
        await mkt.stream_all_mids(_on_mids)

        # ── Boucle principale (maintient le thread vivant) ──
        while not self._stop:
            await asyncio.sleep(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (sync fetch univers + utilitaires visuels)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_universe(network: str) -> List[str]:
    """Appel HTTP synchrone pour récupérer l'univers depuis Hyperliquid (mis en cache 1h)."""
    try:
        import httpx
        url  = ("https://api.hyperliquid.xyz/info"
                if network == "mainnet"
                else "https://api.hyperliquid-testnet.xyz/info")
        resp = httpx.post(url, json={"type": "meta"}, timeout=8.0)
        return sorted([a["name"] for a in resp.json().get("universe", [])])
    except Exception:
        return FALLBACK_COINS


def _badge(status: str) -> str:
    icons = {
        "live"       : "🟢 LIVE",
        "connecting" : "🔵 CONNECTING",
        "warming_up" : "🟡 WARMING UP",
        "error"      : "🔴 ERROR",
        "idle"       : "⚫ IDLE",
    }
    return (f'<span class="badge-{status}">'
            f'{icons.get(status, status.upper())}</span>')


def _pct(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f}%" if math.isfinite(v) else "—"


def _fmt(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}" if math.isfinite(v) else "—"


def _compute_metrics(equity_history: list, initial: float, snap: Dict) -> Dict:
    if len(equity_history) < 2 or initial <= 0:
        return {}
    eq   = np.array([e for _, e in equity_history], dtype=float)
    rets = np.diff(np.log(np.clip(eq, 1e-9, None)))

    peak   = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / (peak + 1e-9)).min()) * 100

    bars_per_year = 365 * 24 * 3600 / max(snap.get("tf_seconds", 300), 1)
    sharpe = (float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
              if len(rets) > 1 and rets.std() > 0 else 0.0)

    n_trades = snap.get("n_wins", 0) + snap.get("n_losses", 0)
    return {
        "total_return_pct": float((eq[-1] / initial - 1) * 100),
        "max_dd_pct"      : float(max_dd),
        "sharpe"          : float(sharpe),
        "win_rate"        : snap.get("n_wins", 0) / n_trades if n_trades > 0 else 0.0,
        "n_trades"        : n_trades,
    }


def _plot_pnl_chart(
    equity_history: list,
    initial: float,
    total_fees: float,
) -> None:
    """
    Courbe PnL live :
      - PnL net  = equity(t) - initial  (après frais)
      - PnL brut = PnL net + total_fees cumulés (proxy sans market impact)
      - Drawdown (subplot bas)
    """
    if len(equity_history) < 2:
        st.info("En attente de données — warmup en cours...")
        return

    ts_   = [datetime.fromisoformat(t) for t, _ in equity_history]
    eq_   = np.array([e for _, e in equity_history], dtype=float)
    pnl_net   = eq_ - initial
    pnl_gross = pnl_net + total_fees          # frais ré-ajoutés = gross
    peak      = np.maximum.accumulate(eq_)
    dd        = (eq_ - peak) / (peak + 1e-9) * 100

    if go is None or make_subplots is None:
        df = pd.DataFrame({"PnL net ($)": pnl_net, "PnL brut ($)": pnl_gross}, index=ts_)
        st.line_chart(df)
        return

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.04,
        subplot_titles=("PnL ($)", "Drawdown (%)"),
    )
    fig.add_trace(
        go.Scatter(x=ts_, y=pnl_gross, name="PnL brut (avant frais)",
                   line=dict(color="#7289da", width=1.5, dash="dot")), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ts_, y=pnl_net, name="PnL net (après frais)",
                   line=dict(color="#2ecc71", width=2),
                   fill="tozeroy",
                   fillcolor="rgba(46,204,113,0.08)"), row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=1, col=1)
    fig.add_trace(
        go.Scatter(x=ts_, y=dd, name="Drawdown %", fill="tozeroy",
                   line=dict(color="#e74c3c", width=1),
                   fillcolor="rgba(231,76,60,0.15)"), row=2, col=1
    )
    fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.07),
        yaxis=dict(tickprefix="$"),
        yaxis2=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_zscores(zscores: Dict[str, float]) -> None:
    if not zscores:
        st.info("Z-scores disponibles après le warmup...")
        return
    syms   = sorted(zscores, key=lambda s: abs(zscores[s]), reverse=True)[:20]
    vals   = [zscores[s] for s in syms]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

    if go is None:
        st.bar_chart(pd.Series(vals, index=syms))
        return

    fig = go.Figure(go.Bar(x=syms, y=vals, marker_color=colors))
    fig.add_hline(y=1.5,  line_dash="dash", line_color="#f39c12",
                  annotation_text="z_in = 1.5")
    fig.add_hline(y=-1.5, line_dash="dash", line_color="#f39c12")
    fig.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=10, r=10, t=30, b=40),
        title="Z-scores intra-panier (rouge → short candidat | vert → long candidat)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_weights(weights: Dict[str, float]) -> None:
    if not weights:
        return
    syms   = sorted(weights, key=lambda s: abs(weights[s]), reverse=True)[:20]
    vals   = [weights[s] for s in syms]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

    if go is None:
        st.bar_chart(pd.Series(vals, index=syms))
        return

    fig = go.Figure(go.Bar(x=syms, y=vals, marker_color=colors))
    fig.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=10, r=10, t=30, b=40),
        title="Poids alloués (rouge = short | vert = long)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "hl_state"  not in st.session_state:
    st.session_state.hl_state  = SharedState()
if "hl_engine" not in st.session_state:
    st.session_state.hl_engine = None
if "hl_cfg"    not in st.session_state:
    st.session_state.hl_cfg    = None
if "hl_tf_s"   not in st.session_state:
    st.session_state.hl_tf_s   = TF_OPTIONS[TF_DEFAULT]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Configuration & contrôles
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔴 HyperStat Live")
    st.divider()

    st.subheader("Réseau")
    network = st.radio("Réseau", ["mainnet", "testnet"], index=0,
                       horizontal=True, label_visibility="collapsed")

    st.subheader("Sélection des coins")
    with st.spinner("Chargement de l'univers..."):
        universe_list = _fetch_universe(network)

    defaults = [c for c in ["BTC", "ETH", "SOL", "AVAX", "LINK", "ARB", "OP"]
                if c in universe_list]
    selected_coins = st.multiselect(
        "Coins à inclure dans le panier",
        options=universe_list,
        default=defaults,
        help="Min. 2 coins. Z-scores calculés intra-panier.",
    )

    st.subheader("Stratégie")
    tf_label   = st.selectbox(
        "Timeframe barre", list(TF_OPTIONS.keys()),
        index=list(TF_OPTIONS.keys()).index(TF_DEFAULT),
        help="Durée d'une barre — de 10 secondes à 1 heure",
    )
    tf_seconds_sel = TF_OPTIONS[tf_label]
    h_bars_disp    = _horizon_bars(tf_seconds_sel)
    st.caption(f"Warmup : {h_bars_disp} barres ≈ "
               f"{h_bars_disp * tf_seconds_sel // 60} min")

    initial_equity  = st.number_input("Capital initial ($)", min_value=100.0,
                                      max_value=1_000_000.0, value=DEFAULT_EQUITY, step=1000.0)
    gross_leverage  = st.slider("Gross leverage", 0.1, 3.0, DEFAULT_GROSS_LEV, 0.1,
                                help="Exposition totale (1.5 = 150% du capital) — no market impact")

    st.subheader("Refresh UI")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_s    = st.number_input("Intervalle (sec)", min_value=2, max_value=30,
                                   value=3, step=1, disabled=not auto_refresh)

    st.divider()

    # ── Config courante ──
    cfg = {
        "network"       : network,
        "selected"      : selected_coins,
        "tf_seconds"    : tf_seconds_sel,
        "initial_equity": initial_equity,
        "gross_leverage": gross_leverage,
    }

    engine     = st.session_state.hl_engine
    is_running = engine is not None and engine.is_alive()

    if not is_running:
        if st.button("▶️ Lancer le paper trading", type="primary",
                     use_container_width=True, disabled=len(selected_coins) < 2):
            new_state  = SharedState()
            new_engine = BackgroundEngine(new_state, cfg)
            new_engine.start()
            st.session_state.hl_state  = new_state
            st.session_state.hl_engine = new_engine
            st.session_state.hl_cfg    = cfg
            st.session_state.hl_tf_s   = tf_seconds_sel
            st.rerun()
    else:
        col_stop, col_restart = st.columns(2)
        with col_stop:
            if st.button("⏹️ Stop", use_container_width=True):
                engine.stop()
                st.session_state.hl_engine = None
                st.session_state.hl_state  = SharedState()
                st.rerun()
        with col_restart:
            if st.button("🔄 Relancer", use_container_width=True,
                         help="Applique la nouvelle config"):
                engine.stop()
                new_state  = SharedState()
                new_engine = BackgroundEngine(new_state, cfg)
                new_engine.start()
                st.session_state.hl_state  = new_state
                st.session_state.hl_engine = new_engine
                st.session_state.hl_cfg    = cfg
                st.session_state.hl_tf_s   = tf_seconds_sel
                st.rerun()

    # Statut affiché dans la sidebar
    snap_sb = st.session_state.hl_state.snapshot()
    st.markdown(f"**Statut** &nbsp; {_badge(snap_sb['status'])}", unsafe_allow_html=True)
    if snap_sb["error"]:
        st.error(snap_sb["error"])
    if snap_sb["last_update"]:
        age = (datetime.now(timezone.utc) - snap_sb["last_update"]).total_seconds()
        st.caption(f"Dernière MAJ: {age:.0f}s | Barres: {snap_sb['n_bars']}/{snap_sb['horizon_bars']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

snap    = st.session_state.hl_state.snapshot()
tf_s    = snap["tf_seconds"]    # secondes/barre (depuis SharedState)
h_bars  = snap["horizon_bars"]  # barres warmup (depuis SharedState)
tf_label_disp = next((k for k, v in TF_OPTIONS.items() if v == tf_s), f"{tf_s}s")

# ── Header ────────────────��────────────���────────────────────────────────────
col_t, col_s, col_eq, col_ret, col_dd = st.columns([3, 2, 2, 2, 2])

with col_t:
    st.markdown("## 🔴 Live Paper Trading — Hyperliquid")

with col_s:
    st.markdown(f"**Statut** &nbsp; {_badge(snap['status'])}", unsafe_allow_html=True)
    if snap["last_update"]:
        st.caption(snap["last_update"].strftime("%H:%M:%S UTC"))

with col_eq:
    eq = snap["equity"]
    st.metric("Equity", f"${eq:,.2f}" if math.isfinite(eq) else "—")

with col_ret:
    init = snap["initial_equity"]
    ret  = (eq / init - 1) * 100 if (init > 0 and math.isfinite(eq)) else 0.0
    st.metric("Retour total", _pct(ret))

with col_dd:
    eq_h = snap["equity_history"]
    if len(eq_h) >= 2:
        eq_arr = np.array([e for _, e in eq_h])
        peak   = np.maximum.accumulate(eq_arr)
        cur_dd = float((eq_arr[-1] - peak[-1]) / (peak[-1] + 1e-9) * 100)
        st.metric("Drawdown courant", _pct(cur_dd))
    else:
        st.metric("Drawdown courant", "—")

# ── Idle screen ────────────────────────────────────���────────────────────────
if snap["status"] == "idle":
    st.divider()
    st.info("⬅️ Sélectionne tes coins et clique **Lancer le paper trading** dans le sidebar.")
    st.markdown("""
    **Fonctionnement :**
    1. Connexion au WebSocket Hyperliquid — récupération des prix en temps réel pour **n'importe quel coin listé**
    2. Formation de barres toutes les N minutes (paramétrable)
    3. Calcul des **z-scores stat-arb** intra-panier → signaux long/short contrarians
    4. **Paper trading** : simulation de trades au prix mid avec frais réalistes (3.5 bps)
    5. Suivi du PnL, drawdown, positions en temps réel
    """)
    st.stop()

# ── Barre de warmup ─────────────────────────────────────────────────────────
if snap["status"] == "warming_up":
    n        = snap["n_bars"]
    rem_sec  = max(0, (h_bars - n) * tf_s)
    rem_str  = f"{rem_sec // 60}min {rem_sec % 60}s" if rem_sec >= 60 else f"{rem_sec}s"
    st.progress(
        min(1.0, n / max(h_bars, 1)),
        text=f"🟡 WARMING UP — {n}/{h_bars} barres de {tf_label_disp} collectées "
             f"(~{rem_str} restantes)"
    )
elif snap["status"] == "connecting":
    st.progress(0.0, text="🔵 Connexion au WebSocket Hyperliquid...")

st.divider()

# ── Tabs ───────────────────���────────────────────────────────────────────────
tab_mkt, tab_sig, tab_pf, tab_pos = st.tabs([
    "📈 Marché Live",
    "📊 Signaux Stratégie",
    "💰 Portfolio",
    "📋 Positions",
])


# ── TAB : Marché ─────────────────────────────────────────────────────────────
with tab_mkt:
    st.markdown('<div class="section-header">📈 Prix en temps réel — Hyperliquid</div>',
                unsafe_allow_html=True)

    mids    = snap["mids"]
    zscores = snap["zscores"]
    weights = snap["weights"]

    if mids:
        rows = []
        for sym in snap["selected"]:
            px = mids.get(sym)
            if px is None:
                continue
            z = zscores.get(sym, 0.0)
            signal_label = ("🔴 SHORT" if z > 1.5 else "🟢 LONG" if z < -1.5 else "⚪ FLAT")
            rows.append({
                "Coin"    : sym,
                "Prix"    : f"${px:,.4f}" if px < 100 else f"${px:,.2f}",
                "Z-Score" : f"{z:+.3f}",
                "Poids"   : f"{weights.get(sym, 0):+.3f}",
                "Signal"  : signal_label,
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"{len(rows)} coins | Timeframe: {tf_label_disp} | "
                       f"MAJ: {snap['last_update'].strftime('%H:%M:%S') if snap['last_update'] else '—'} UTC")
    else:
        st.info("En attente des premiers prix Hyperliquid (quelques secondes)...")

    # Nombre de coins dans l'univers
    if snap["universe"]:
        st.caption(f"Univers total Hyperliquid : **{len(snap['universe'])} coins** disponibles")


# ── TAB : Signaux ────────────────────────────────────────────────────────────
with tab_sig:
    st.markdown('<div class="section-header">📊 Signaux stat-arb</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Z-Scores intra-panier")
        _plot_zscores(snap["zscores"])
    with col2:
        st.markdown("#### Poids alloués")
        _plot_weights(snap["weights"])

    if snap["signal_ts"]:
        st.caption(
            f"Dernier signal: {snap['signal_ts'].strftime('%H:%M:%S UTC')} | "
            f"Barres traitées: {snap['n_bars']} | "
            f"Gross leverage: {cfg['gross_leverage']:.1f}x"
        )
    else:
        st.info("Les signaux apparaîtront après le warmup.")


# ── TAB : Portfolio ──────────────────────────────────────────────────────────
with tab_pf:
    st.markdown('<div class="section-header">💰 Performance — Paper Trading (no market impact)</div>',
                unsafe_allow_html=True)

    # ── Calculs PnL ──────────────────────────────────────────────────────────
    upnl        = sum(p.get("upnl", 0) for p in snap["positions"].values())
    pnl_net     = snap["realized_pnl"] + upnl - snap["total_fees"]
    pnl_gross   = snap["realized_pnl"] + upnl
    total_notional = sum(p.get("notional", 0) for p in snap["positions"].values())
    init        = snap["initial_equity"]

    # ── Ligne 1 : métriques principales ──────────────────────────────────────
    r1 = st.columns(6)
    r1[0].metric("PnL net ($)",      f"${pnl_net:+,.2f}",
                 help="Réalisé + Unrealized − Frais")
    r1[1].metric("PnL brut ($)",     f"${pnl_gross:+,.2f}",
                 help="Réalisé + Unrealized (avant frais)")
    r1[2].metric("Frais cumulés",    f"${snap['total_fees']:.2f}",
                 help="3.5 bps par trade, taker Hyperliquid")
    r1[3].metric("Retour net (%)",   _pct(pnl_net / init * 100) if init > 0 else "—")
    r1[4].metric("uPnL ouvert",      f"${upnl:+,.2f}")
    r1[5].metric("Notional total",   f"${total_notional:,.0f}",
                 help="Somme |qty × mark| toutes positions")

    # ── Ligne 2 : métriques risque / perf ────────────────────────────────────
    metrics = _compute_metrics(snap["equity_history"], init, snap)
    r2 = st.columns(5)
    if metrics:
        r2[0].metric("Sharpe annualisé", _fmt(metrics["sharpe"]))
        r2[1].metric("Max Drawdown",     _pct(metrics["max_dd_pct"]))
        r2[2].metric("Win Rate",         _pct(metrics["win_rate"] * 100, 1))
        r2[3].metric("Nb trades",        str(metrics["n_trades"]))
    r2[4].metric("Barres traitées",  str(snap["n_bars"]))

    st.divider()

    # ── Graphique PnL live (net + brut + drawdown) ───────────────────────────
    st.markdown("#### PnL live — net vs brut (avant frais)")
    _plot_pnl_chart(snap["equity_history"], init, snap["total_fees"])

    # ── Réalisé vs Unrealized ─────────────────────────────────────────────────
    with st.expander("Détail PnL réalisé / unrealized / frais"):
        dc = st.columns(3)
        dc[0].metric("PnL réalisé",   f"${snap['realized_pnl']:+,.2f}")
        dc[1].metric("uPnL (open)",   f"${upnl:+,.2f}")
        dc[2].metric("Gains / Pertes", f"{snap['n_wins']} ✅ / {snap['n_losses']} ❌")


# ── TAB : Positions ──────────────────────────────────────────────────────────
with tab_pos:
    st.markdown('<div class="section-header">📋 Positions courantes</div>',
                unsafe_allow_html=True)

    positions = snap["positions"]
    if positions:
        rows = []
        for sym, p in sorted(positions.items(),
                             key=lambda x: abs(x[1].get("upnl", 0)), reverse=True):
            qty  = p.get("qty", 0)
            upnl = p.get("upnl", 0)
            rows.append({
                "Coin"     : sym,
                "Side"     : "🟢 LONG" if qty > 0 else "🔴 SHORT",
                "Qté"      : f"{qty:.6f}",
                "Entry"    : f"${p.get('entry_px', 0):,.4f}",
                "Mark"     : f"${p.get('current_px', 0):,.4f}",
                "Notional" : f"${p.get('notional', 0):,.1f}",
                "uPnL"     : f"${upnl:+,.2f}",
                "Z-Score"  : f"{p.get('zscore', 0):+.3f}",
                "Poids"    : f"{p.get('weight', 0):+.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        total_upnl   = sum(p.get("upnl", 0)      for p in positions.values())
        total_notional = sum(p.get("notional", 0) for p in positions.values())
        pc = st.columns(3)
        pc[0].metric("uPnL total",     f"${total_upnl:+,.2f}")
        pc[1].metric("Notional total", f"${total_notional:,.0f}")
        pc[2].metric("Nb positions",   str(len(positions)))
    else:
        st.info(
            "Aucune position ouverte.\n\n"
            "Les positions s'ouvrent quand un z-score dépasse **z_in = 1.5** "
            "après le warmup."
        )


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH (st.rerun = delta WebSocket, sans rechargement navigateur)
# ─────────────────────────────────────────────────────────────────────────────

if auto_refresh and snap["status"] != "idle":
    time.sleep(int(refresh_s))
    st.rerun()