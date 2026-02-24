# apps/live_dashboard.py
"""
Live Multi-Strategy Paper Trading Dashboard — Hyperliquid Exchange
==================================================================
Supporte 7 stratégies indépendantes, chacune avec son propre PaperPortfolio.

Architecture :
    BackgroundEngine (thread daemon + asyncio)
        ├── SharedData          : données marché partagées (mids, price_history, funding, ob, trades)
        ├── StrategySlot × 7    : état par stratégie (agent + portfolio + historiques)
        └── CostAwareRebalancer : filtre edge > 2*(fee+slip)+buffer avant toute exécution
                ↓  SharedState (thread-safe)
    Streamlit UI — lit SharedState.snapshot()
        ├── Sidebar : start/stop/restart par stratégie + Reload Config
        └── Tabs    : Marché | Signaux | Portfolio (sous-onglet par strat + Agrégé) | Config

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
from typing import Any, Dict, List, Optional, Tuple

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
from hyperstat.strategy.base_signal_agent import AgentContext, BaseSignalAgent
from hyperstat.strategy.momentum import CrossSectionalMomentumAgent
from hyperstat.strategy.funding_carry_pure import FundingCarryPureAgent
from hyperstat.strategy.liquidation_reversion import LiquidationReversionAgent
from hyperstat.strategy.ob_imbalance import OrderFlowImbalanceAgent
from hyperstat.strategy.pca_residual_mr import PCAResiduaMRAgent
from hyperstat.strategy.quality_liquidity import QualityLiquidityAgent


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HyperStat — Multi-Strategy Live",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .badge-live        { background:#1a7a3c; color:#fff; padding:3px 10px; border-radius:12px;
                         font-size:12px; font-weight:700; animation:pulse 2s infinite; }
    .badge-connecting  { background:#2a3a6e; color:#fff; padding:3px 10px; border-radius:12px;
                         font-size:12px; font-weight:700; }
    .badge-warming_up  { background:#8b6e1a; color:#fff; padding:3px 10px; border-radius:12px;
                         font-size:12px; font-weight:700; }
    .badge-stopped     { background:#444; color:#aaa; padding:3px 10px; border-radius:12px;
                         font-size:12px; font-weight:700; }
    .badge-error       { background:#8b1a1a; color:#fff; padding:3px 10px; border-radius:12px;
                         font-size:12px; font-weight:700; }
    .badge-idle        { background:#3a3a3a; color:#aaa; padding:3px 10px; border-radius:12px;
                         font-size:12px; font-weight:700; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.65; } }
    .section-header {
        color: #7289da; font-size: 13px; font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; border-bottom: 1px solid #2a2d3a;
        padding-bottom: 6px; margin: 16px 0 12px 0;
    }
    .strat-row { border-bottom: 1px solid #2a2d3a; padding: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EQUITY    = 10_000.0
DEFAULT_GROSS_LEV = 1.5
FEE_RATE          = 0.00035    # 3.5 bps taker Hyperliquid
MAX_EQUITY_PTS    = 5_000
MIN_BAR_COINS     = 2
FUNDING_POLL_S    = 300.0      # poll funding rates every 5 min

TF_OPTIONS: Dict[str, int] = {
    "10s": 10, "30s": 30,
    "1m": 60, "3m": 180, "5m": 300,
    "15m": 900, "30m": 1800, "1h": 3600,
}
TF_DEFAULT = "5m"
HORIZON_BARS_MIN = 12
HORIZON_BARS_MAX = 90

FALLBACK_COINS = [
    "BTC", "ETH", "SOL", "AVAX", "LINK", "ARB", "OP",
    "MATIC", "DOT", "ADA", "BNB", "NEAR", "FTM", "ATOM",
]

# ── Cost-aware filter constants ────────────────────────────────────────────────
FEE_BPS           = 3.5
SLIP_BPS          = 10.0
ROUND_TRIP_BPS    = 2.0 * (FEE_BPS + SLIP_BPS)   # 27 bps
BUFFER_BPS        = 3.0
THRESHOLD_BPS     = ROUND_TRIP_BPS + BUFFER_BPS   # 30 bps = 0.003
DELTA_W_MIN       = 0.003   # no-trade zone: ignore deltas < 0.3%
MIN_TRADE_NOTIONAL = 0.50   # skip trades < $0.50 notional

# ── Strategy registry ──────────────────────────────────────────────────────────
# name → (factory_fn, enabled_by_default)
_STRATEGY_REGISTRY: Dict[str, Tuple] = {
    "Stat-Arb MR + FDS" : (lambda: _StatArbMRWrapper(), True),
    "Cross-Section Momentum": (lambda: CrossSectionalMomentumAgent(), True),
    "Funding Carry Pure": (lambda: FundingCarryPureAgent(), True),
    "PCA Residual MR"   : (lambda: PCAResiduaMRAgent(), True),
    "Quality / Liquidity": (lambda: QualityLiquidityAgent(), True),
    "Liquidation Reversion": (lambda: LiquidationReversionAgent(), False),
    "OB Imbalance"      : (lambda: OrderFlowImbalanceAgent(), False),
}

STRATEGY_COLORS: Dict[str, str] = {
    "Stat-Arb MR + FDS"     : "#2ecc71",
    "Cross-Section Momentum": "#3498db",
    "Funding Carry Pure"    : "#f39c12",
    "PCA Residual MR"       : "#9b59b6",
    "Quality / Liquidity"   : "#1abc9c",
    "Liquidation Reversion" : "#e74c3c",
    "OB Imbalance"          : "#e67e22",
}


def _horizon_bars(tf_seconds: int) -> int:
    return max(HORIZON_BARS_MIN, min(HORIZON_BARS_MAX, 3600 // tf_seconds))


# ─────────────────────────────────────────────────────────────────────────────
# STAT-ARB WRAPPER  (wraps existing StatArbStrategy as a BaseSignalAgent)
# ─────────────────────────────────────────────────────────────────────────────

class _StatArbMRWrapper(BaseSignalAgent):
    """Thin wrapper around the existing StatArbStrategy."""
    name = "Stat-Arb MR + FDS"
    warmup_bars = HORIZON_BARS_MIN
    enabled_by_default = True

    def __init__(self, tf_seconds: int = 300) -> None:
        super().__init__()
        self._tf_seconds = tf_seconds
        self._strategy: Optional[StatArbStrategy] = None
        self._h_bars: int = _horizon_bars(tf_seconds)
        self._init_strategy()

    def _init_strategy(self) -> None:
        self._h_bars = _horizon_bars(self._tf_seconds)
        self.warmup_bars = self._h_bars
        self._strategy = StatArbStrategy(StatArbConfig(
            timeframe_minutes=max(1, self._tf_seconds // 60),
            horizon_bars=self._h_bars,
        ))

    def reset(self) -> None:
        self._bars_seen = 0
        self._init_strategy()

    def update(self, ts, mids, context: AgentContext):
        from hyperstat.strategy.base_signal_agent import AgentOutput
        signal = self._strategy.update(ts, mids, context.buckets)
        self._bars_seen += 1
        if not self.is_warmed_up:
            return AgentOutput()
        return AgentOutput(
            weights=dict(signal.weights_raw),
            zscores=dict(signal.zscores),
            meta=dict(signal.meta),
        )


# ─────────────────────────────────────────────────────────────────────────────
# COST-AWARE REBALANCER
# ─────────────────────────────────────────────────────────────────────────────

class CostAwareRebalancer:
    """
    Filters target weights before execution.

    Global rule: trade only if edge > 2*(fee+slip)+buffer  (THRESHOLD_BPS = 30 bps)

    Edge proxy per trade:
        edge_bps ≈ |delta_w| × 10 000
        → skip if edge_bps < THRESHOLD_BPS   (= |delta_w| < 0.003)

    Additional guard: skip if |delta_notional| < MIN_TRADE_NOTIONAL ($0.50)
    """

    @staticmethod
    def filter_trades(
        target_w: Dict[str, float],
        current_w: Dict[str, float],
        mids: Dict[str, float],
        equity: float,
    ) -> Dict[str, float]:
        result: Dict[str, float] = dict(current_w)  # start from current (hold)
        for sym, tw in target_w.items():
            cw = current_w.get(sym, 0.0)
            delta_w = tw - cw

            # ── Edge filter: trade only if edge > 2*(fee+slip)+buffer ─────────
            # Proxy: expected edge ≈ |delta_w| × 10 000 bps
            # Equivalent to: |delta_w| ≥ THRESHOLD_BPS × 1e-4 = 0.003
            edge_bps_proxy = abs(delta_w) * 10_000
            if edge_bps_proxy < THRESHOLD_BPS:   # 30 bps = 2*(3.5+10)+3
                continue   # expected edge < round-trip cost — skip

            px = mids.get(sym, 0.0)
            if not (math.isfinite(px) and px > 0):
                continue
            delta_notional = abs(delta_w * equity)
            if delta_notional < MIN_TRADE_NOTIONAL:
                continue   # trade too small in absolute $ — skip
            result[sym] = tw

        # Also close symbols present in current_w but absent from target_w
        for sym in list(current_w.keys()):
            if sym not in target_w and abs(current_w.get(sym, 0.0)) > 1e-12:
                result[sym] = 0.0
        return result

    @staticmethod
    def compute_bar_turnover(
        prev_w: Dict[str, float],
        new_w: Dict[str, float],
    ) -> float:
        """Turnover = sum of |w_new - w_prev| across all symbols."""
        all_syms = set(list(prev_w.keys()) + list(new_w.keys()))
        return float(sum(abs(new_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_syms))


# ─────────────────────────────────────────────────────────────────────────────
# PAPER PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

class PaperPortfolio:
    """Simulates trades at mid price with FEE_RATE. Independent per strategy."""

    def __init__(self, initial_equity: float, gross_leverage: float) -> None:
        self.initial_equity  = initial_equity
        self.equity          = initial_equity
        self.gross_leverage  = gross_leverage
        self._pos: Dict[str, float]   = {}
        self._entry: Dict[str, float] = {}
        self.realized_pnl = 0.0
        self.total_fees   = 0.0
        self.fees_this_bar = 0.0   # reset each bar
        self.n_wins   = 0
        self.n_losses = 0

    def rebalance(
        self,
        target_weights: Dict[str, float],
        mids: Dict[str, float],
    ) -> None:
        self.fees_this_bar = 0.0
        total_eq = self.equity + self._unrealized(mids)

        for sym, tgt_w in target_weights.items():
            px = mids.get(sym)
            if not (px and math.isfinite(px) and px > 0):
                continue
            tgt_qty = (tgt_w * total_eq) / px
            cur_qty = self._pos.get(sym, 0.0)
            delta   = tgt_qty - cur_qty

            if abs(delta * px) < 0.5:
                continue

            fee = abs(delta * px) * FEE_RATE

            # Realize PnL on closures
            if cur_qty != 0.0 and math.copysign(1, delta) != math.copysign(1, cur_qty):
                close_qty = min(abs(delta), abs(cur_qty)) * math.copysign(1, cur_qty)
                rpnl = close_qty * (px - self._entry.get(sym, px))
                self.realized_pnl += rpnl
                self.equity += rpnl
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
                    self._entry[sym] = px
                elif math.copysign(1, delta) == math.copysign(1, cur_qty):
                    self._entry[sym] = (
                        abs(cur_qty) * self._entry.get(sym, px) + abs(delta) * px
                    ) / abs(new_qty)
                # else: reducing position → entry unchanged

            self.equity        -= fee
            self.total_fees    += fee
            self.fees_this_bar += fee

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

    def current_weights(self, mids: Dict[str, float]) -> Dict[str, float]:
        """Current position weights (notional / total_equity)."""
        teq = self.total_equity(mids)
        if teq <= 0:
            return {}
        return {
            sym: (qty * mids.get(sym, 0.0)) / teq
            for sym, qty in self._pos.items()
            if sym in mids and math.isfinite(mids.get(sym, float("nan")))
        }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY SLOT  (per-strategy state container)
# ─────────────────────────────────────────────────────────────────────────────

class StrategySlot:
    """
    Holds the live state of one strategy instance.
    Thread-safe reads via snapshot().
    """

    def __init__(
        self,
        name: str,
        agent: BaseSignalAgent,
        portfolio: PaperPortfolio,
        enabled: bool = True,
    ) -> None:
        self.name      = name
        self.agent     = agent
        self.portfolio = portfolio
        self.enabled   = enabled
        self.status: str = "warming_up"

        # Per-strategy historical data
        self.equity_history: deque   = deque(maxlen=MAX_EQUITY_PTS)
        self.turnover_history: deque = deque(maxlen=MAX_EQUITY_PTS)  # (ts, turnover)
        self.fee_history: deque      = deque(maxlen=MAX_EQUITY_PTS)  # (ts, fee_this_bar)

        # Last output from agent
        self.last_weights: Dict[str, float] = {}
        self.last_zscores: Dict[str, float] = {}
        self.last_meta: Dict[str, Any]      = {}
        self.last_bar_weights: Dict[str, float] = {}  # for turnover tracking
        self.last_bar_ts: Optional[datetime]    = None
        self.n_bars = 0
        self.positions: Dict[str, Dict] = {}

    def snapshot(self) -> Dict[str, Any]:
        """Return a thread-safe dict copy (called under SharedState lock)."""
        p = self.portfolio
        return {
            "name"            : self.name,
            "status"          : self.status,
            "enabled"         : self.enabled,
            "n_bars"          : self.n_bars,
            "warmup_bars"     : self.agent.warmup_bars,
            "bars_seen"       : self.agent.bars_seen,
            "equity"          : p.equity,
            "initial_equity"  : p.initial_equity,
            "equity_history"  : list(self.equity_history),
            "turnover_history": list(self.turnover_history),
            "fee_history"     : list(self.fee_history),
            "realized_pnl"    : p.realized_pnl,
            "total_fees"      : p.total_fees,
            "n_wins"          : p.n_wins,
            "n_losses"        : p.n_losses,
            "weights"         : dict(self.last_weights),
            "zscores"         : dict(self.last_zscores),
            "meta"            : dict(self.last_meta),
            "positions"       : {k: dict(v) for k, v in self.positions.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (thread-safe global container)
# ─────────────────────────────────────────────────────────────────────────────

class SharedState:
    """
    Thread-safe shared state between BackgroundEngine and Streamlit UI.
    Holds market data + per-strategy slots.
    """

    def __init__(self, initial_equity: float = DEFAULT_EQUITY, gross_leverage: float = DEFAULT_GROSS_LEV) -> None:
        self._lock = threading.RLock()
        self.global_status: str = "idle"
        self.error: str = ""

        # Config
        self.tf_seconds: int  = TF_OPTIONS[TF_DEFAULT]
        self.horizon_bars: int = _horizon_bars(self.tf_seconds)
        self.initial_equity   = initial_equity
        self.gross_leverage   = gross_leverage

        # Market data (shared)
        self.mids: Dict[str, float]         = {}
        self.universe: List[str]            = []
        self.selected: List[str]            = []
        self.last_update: Optional[datetime] = None
        self.funding_rates: Dict[str, float] = {}

        # Strategy slots
        self.strategies: Dict[str, StrategySlot] = {}

        # Config dict (for "Reload Config" button)
        self.config: Dict[str, Any] = {}

    def get_active_strategies(self) -> List[str]:
        return [name for name, slot in self.strategies.items() if slot.enabled]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            strat_snaps = {
                name: slot.snapshot()
                for name, slot in self.strategies.items()
            }
            return {
                "global_status" : self.global_status,
                "error"         : self.error,
                "tf_seconds"    : self.tf_seconds,
                "horizon_bars"  : self.horizon_bars,
                "initial_equity": self.initial_equity,
                "mids"          : dict(self.mids),
                "universe"      : list(self.universe),
                "selected"      : list(self.selected),
                "last_update"   : self.last_update,
                "funding_rates" : dict(self.funding_rates),
                "strategies"    : strat_snaps,
                "config"        : dict(self.config),
            }


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND ENGINE  (thread daemon + asyncio)
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundEngine:
    """
    Runs in a daemon thread with its own asyncio event loop.
    - Single WS connection for allMids, optional l2Book + trades
    - Dispatches each bar to all enabled StrategySlots
    - Applies cost-aware filter before each portfolio rebalance
    - Logs turnover + fees per strategy per bar
    """

    def __init__(self, state: SharedState, cfg: Dict[str, Any]) -> None:
        self.state  = state
        self.cfg    = cfg
        self._loop  = asyncio.new_event_loop()
        self._stop  = False
        self._thread = threading.Thread(
            target=self._run_thread, daemon=True, name="HL-MultiStratEngine"
        )
        self._rebalancer = CostAwareRebalancer()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run_thread(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async())
        except Exception as exc:
            with self.state._lock:
                self.state.global_status = "error"
                self.state.error         = str(exc)

    async def _run_async(self) -> None:
        with self.state._lock:
            self.state.global_status = "connecting"
            self.state.error = ""

        cfg      = self.cfg
        network  = cfg["network"]
        selected = cfg["selected"]
        tf_sec   = cfg["tf_seconds"]
        h_bars   = _horizon_bars(tf_sec)
        init_eq  = cfg["initial_equity"]
        gross_lev = cfg["gross_leverage"]
        enabled_strats = cfg.get("enabled_strategies", {})

        # ── Clients Hyperliquid ────────────────────────────────────────────────
        ep  = (HyperliquidEndpoints.mainnet()
               if network == "mainnet"
               else HyperliquidEndpoints.testnet())
        rl   = RateLimiter(RateLimiterConfig(weight_budget_per_minute=600))
        rest = HyperliquidRestClient(endpoints=ep, rate_limiter=rl)
        ws_c = HyperliquidWsClient(endpoints=ep)
        mkt  = HyperliquidMarketData(rest, ws_c)

        await ws_c.start()

        # ── Fetch universe ────────────────────────────────────────────────────
        try:
            meta     = await mkt.meta()
            universe = sorted([a["name"] for a in meta.get("universe", [])])
        except Exception:
            universe = FALLBACK_COINS

        # ── Initialize strategies ─────────────────────────────────────────────
        per_strat_eq = init_eq  # each strategy gets full capital (independent portfolios)

        slots: Dict[str, StrategySlot] = {}
        for strat_name, (factory_fn, default_on) in _STRATEGY_REGISTRY.items():
            should_enable = enabled_strats.get(strat_name, default_on)
            agent = factory_fn()
            # Propagate tf_seconds to StatArbMRWrapper
            if isinstance(agent, _StatArbMRWrapper):
                agent._tf_seconds = tf_sec
                agent._init_strategy()
            portfolio = PaperPortfolio(initial_equity=per_strat_eq, gross_leverage=gross_lev)
            slot = StrategySlot(name=strat_name, agent=agent, portfolio=portfolio, enabled=should_enable)
            slots[strat_name] = slot

        with self.state._lock:
            self.state.universe      = universe
            self.state.selected      = selected
            self.state.initial_equity = init_eq
            self.state.gross_leverage = gross_lev
            self.state.tf_seconds     = tf_sec
            self.state.horizon_bars   = h_bars
            self.state.global_status  = "warming_up"
            self.state.strategies     = slots
            self.state.config         = dict(cfg)

        # ── Shared data buffers ────────────────────────────────────────────────
        latest_mids: Dict[str, float]         = {}
        latest_ob: Dict[str, Dict]            = {}     # for OB imbalance
        latest_trades: Dict[str, deque]       = {}     # for liquidation reversion
        latest_funding: Dict[str, float]      = {}
        last_bar_ts: Optional[datetime]       = None
        buckets                               = {"live": selected}
        ob_subscriptions_done                 = False

        # ── Subscribe to l2Book & trades if needed ────────────────────────────
        def _maybe_subscribe_extra() -> None:
            """Called once to subscribe to l2Book/trades for OB/liq strats."""
            nonlocal ob_subscriptions_done
            if ob_subscriptions_done:
                return
            ob_needed  = slots.get("OB Imbalance", StrategySlot.__new__(StrategySlot)).enabled if "OB Imbalance" in slots else False
            liq_needed = slots.get("Liquidation Reversion", StrategySlot.__new__(StrategySlot)).enabled if "Liquidation Reversion" in slots else False

            if ob_needed or liq_needed:
                ob_subscriptions_done = True  # mark before scheduling to avoid double-subscribe

                async def _subscribe_extras_async() -> None:
                    for sym in selected[:10]:   # limit to 10 coins to avoid WS sub limit
                        if ob_needed:
                            try:
                                await mkt.stream_l2_book(sym, cb=lambda msg, s=sym: _on_l2book(msg, s))
                            except Exception:
                                pass
                        if liq_needed:
                            try:
                                await mkt.stream_trades(sym, cb=lambda msg, s=sym: _on_trades(msg, s))
                            except Exception:
                                pass

                asyncio.ensure_future(_subscribe_extras_async(), loop=self._loop)

        def _on_l2book(msg: Any, sym: str) -> None:
            if isinstance(msg, dict):
                data = msg.get("data", msg)
                latest_ob[sym] = data

        def _on_trades(msg: Any, sym: str) -> None:
            if isinstance(msg, dict):
                data = msg.get("data", msg)
                if isinstance(data, list):
                    if sym not in latest_trades:
                        latest_trades[sym] = deque(maxlen=50)
                    for t in data:
                        latest_trades[sym].append(t)

        # ── Callback WebSocket allMids ────────────────────────────────────────
        def _on_mids(msg: Any) -> None:
            nonlocal latest_mids, last_bar_ts

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

            # ── Subscribe extras on first good mids (avoids subscribe before connected) ──
            _maybe_subscribe_extra()

            # ── Trigger bar ───────────────────────────────────────────────────
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

            context = AgentContext(
                selected=selected,
                buckets=buckets,
                funding_rates=dict(latest_funding),
                ob_snapshots=dict(latest_ob),
                trade_history={s: deque(list(v)) for s, v in latest_trades.items()},
            )

            # ── Dispatch to all enabled strategy slots ────────────────────────
            any_live = False

            with self.state._lock:
                active_slots = [slot for slot in slots.values() if slot.enabled]

            for slot in active_slots:
                try:
                    output = slot.agent.update(now, bar_mids, context)
                    slot.n_bars += 1

                    if not slot.agent.is_warmed_up:
                        slot.status = "warming_up"
                        continue

                    # ── Cost-aware filter ─────────────────────────────────────
                    current_w = slot.portfolio.current_weights(bar_mids)
                    filtered_w = self._rebalancer.filter_trades(
                        output.weights, current_w, bar_mids, slot.portfolio.total_equity(bar_mids)
                    )

                    # Normalize to gross leverage
                    total_w = sum(abs(w) for w in filtered_w.values())
                    if total_w > 1e-9:
                        scale = gross_lev / total_w
                        norm_w = {k: v * scale for k, v in filtered_w.items()}
                    else:
                        norm_w = {}

                    # ── Rebalance portfolio ───────────────────────────────────
                    slot.portfolio.rebalance(norm_w, bar_mids)

                    total_eq = slot.portfolio.total_equity(bar_mids)

                    # ── Log turnover + fees per bar ───────────────────────────
                    turnover = self._rebalancer.compute_bar_turnover(
                        slot.last_bar_weights, norm_w
                    )
                    slot.turnover_history.append((now.isoformat(), turnover))
                    slot.fee_history.append((now.isoformat(), slot.portfolio.fees_this_bar))
                    slot.last_bar_weights = dict(norm_w)

                    # ── Update slot state ─────────────────────────────────────
                    slot.equity_history.append((now.isoformat(), total_eq))
                    slot.last_weights = dict(norm_w)
                    slot.last_zscores = dict(output.zscores)
                    slot.last_meta    = dict(output.meta)
                    slot.positions    = slot.portfolio.positions_snapshot(
                        bar_mids, output.zscores, norm_w
                    )
                    slot.status = "live"
                    any_live = True

                except Exception as exc:
                    slot.status = "error"
                    slot.last_meta = {"error": str(exc)}

            with self.state._lock:
                if any_live:
                    self.state.global_status = "live"
                elif self.state.global_status == "warming_up":
                    pass  # keep warming_up

        # ── Funding rate polling (async background task) ──────────────────────
        async def _poll_funding() -> None:
            while not self._stop:
                try:
                    resp = await mkt.meta_and_asset_ctxs()
                    # HL format: [meta_dict, [asset_ctx_dict, ...]]
                    if isinstance(resp, list) and len(resp) == 2:
                        meta_info = resp[0]
                        asset_ctxs = resp[1]
                        assets = meta_info.get("universe", [])
                        for i, asset in enumerate(assets):
                            sym = asset.get("name", "")
                            if sym in selected and i < len(asset_ctxs):
                                ctx = asset_ctxs[i]
                                funding = ctx.get("funding", "0")
                                try:
                                    latest_funding[sym] = float(funding)
                                except (ValueError, TypeError):
                                    pass
                    with self.state._lock:
                        self.state.funding_rates = dict(latest_funding)
                except Exception:
                    pass
                await asyncio.sleep(FUNDING_POLL_S)

        # ── Start async tasks ─────────────────────────────────────────────────
        asyncio.ensure_future(_poll_funding(), loop=self._loop)
        await mkt.stream_all_mids(_on_mids)

        # ── Keep-alive loop ───────────────────────────────────────────────────
        while not self._stop:
            await asyncio.sleep(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_universe(network: str) -> List[str]:
    try:
        import httpx
        url = ("https://api.hyperliquid.xyz/info"
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
        "stopped"    : "⚫ STOPPED",
        "error"      : "🔴 ERROR",
        "idle"       : "⚫ IDLE",
    }
    return f'<span class="badge-{status}">{icons.get(status, status.upper())}</span>'


def _pct(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f}%" if math.isfinite(v) else "—"


def _fmt(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}" if math.isfinite(v) else "—"


def _compute_metrics(equity_history: list, initial: float, tf_seconds: int,
                     n_wins: int = 0, n_losses: int = 0) -> Dict[str, float]:
    if len(equity_history) < 2 or initial <= 0:
        return {}
    eq   = np.array([e for _, e in equity_history], dtype=float)
    rets = np.diff(np.log(np.clip(eq, 1e-9, None)))
    peak   = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / (peak + 1e-9)).min()) * 100
    bars_per_year = 365 * 24 * 3600 / max(tf_seconds, 1)
    sharpe = (float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
              if len(rets) > 1 and rets.std() > 0 else 0.0)
    n_trades = n_wins + n_losses
    return {
        "total_return_pct": float((eq[-1] / initial - 1) * 100),
        "max_dd_pct"      : float(max_dd),
        "sharpe"          : float(sharpe),
        "win_rate"        : n_wins / n_trades if n_trades > 0 else 0.0,
        "n_trades"        : n_trades,
    }


def _plot_pnl_chart(equity_history: list, initial: float, total_fees: float,
                    color: str = "#2ecc71", label: str = "") -> None:
    if len(equity_history) < 2:
        st.info("En attente de données — warmup en cours...")
        return
    ts_   = [datetime.fromisoformat(t) for t, _ in equity_history]
    eq_   = np.array([e for _, e in equity_history], dtype=float)
    pnl_net   = eq_ - initial
    pnl_gross = pnl_net + total_fees
    peak      = np.maximum.accumulate(eq_)
    dd        = (eq_ - peak) / (peak + 1e-9) * 100

    title_suffix = f" — {label}" if label else ""

    if go is None or make_subplots is None:
        df = pd.DataFrame({"PnL net ($)": pnl_net}, index=ts_)
        st.line_chart(df)
        return

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.04,
        subplot_titles=(f"PnL ($){title_suffix}", "Drawdown (%)"),
    )
    fig.add_trace(
        go.Scatter(x=ts_, y=pnl_gross, name="PnL brut (avant frais)",
                   line=dict(color="#7289da", width=1.5, dash="dot")), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ts_, y=pnl_net, name="PnL net",
                   line=dict(color=color, width=2),
                   fill="tozeroy", fillcolor=f"rgba{_hex_to_rgba(color, 0.08)}"), row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555", row=1, col=1)
    fig.add_trace(
        go.Scatter(x=ts_, y=dd, name="Drawdown %", fill="tozeroy",
                   line=dict(color="#e74c3c", width=1),
                   fillcolor="rgba(231,76,60,0.15)"), row=2, col=1
    )
    fig.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.07),
        yaxis=dict(tickprefix="$"),
        yaxis2=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_turnover_chart(turnover_history: list, fee_history: list, color: str = "#f39c12") -> None:
    if len(turnover_history) < 2:
        return
    ts_  = [datetime.fromisoformat(t) for t, _ in turnover_history]
    to_  = [v for _, v in turnover_history]
    fee_ = [v for _, v in fee_history] if fee_history else [0.0] * len(to_)

    if go is None:
        df = pd.DataFrame({"Turnover": to_}, index=ts_)
        st.line_chart(df)
        return

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=ts_, y=to_, name="Turnover", marker_color=color, opacity=0.7))
    fig.add_trace(go.Bar(x=ts_, y=fee_, name="Frais ($)", marker_color="#e74c3c", opacity=0.6))
    fig.update_layout(
        template="plotly_dark", height=220, barmode="overlay",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.05),
        title="Turnover par barre (et frais $)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_zscores(zscores: Dict[str, float], color: str = "#e74c3c") -> None:
    if not zscores:
        st.info("Z-scores disponibles après le warmup...")
        return
    syms = sorted(zscores, key=lambda s: abs(zscores[s]), reverse=True)[:20]
    vals = [zscores[s] for s in syms]
    colors_bar = [color if v > 0 else "#2ecc71" for v in vals]
    if go is None:
        st.bar_chart(pd.Series(vals, index=syms))
        return
    fig = go.Figure(go.Bar(x=syms, y=vals, marker_color=colors_bar))
    fig.add_hline(y=1.5, line_dash="dash", line_color="#f39c12", annotation_text="z_in")
    fig.add_hline(y=-1.5, line_dash="dash", line_color="#f39c12")
    fig.update_layout(
        template="plotly_dark", height=270,
        margin=dict(l=10, r=10, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' → '(r,g,b,a)' string for plotly."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"({r},{g},{b},{alpha})"
    return f"(46,204,113,{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

if "hl_state"  not in st.session_state:
    st.session_state.hl_state  = SharedState()
if "hl_engine" not in st.session_state:
    st.session_state.hl_engine = None
if "hl_cfg"    not in st.session_state:
    st.session_state.hl_cfg    = None
# Per-strategy enabled flags (persists across reruns)
if "strat_enabled" not in st.session_state:
    st.session_state.strat_enabled = {
        name: default_on
        for name, (_, default_on) in _STRATEGY_REGISTRY.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Configuration & Strategy Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔴 HyperStat Multi-Strategy")
    st.divider()

    # ── Réseau ────────────────────────────────────────────────────────────────
    st.subheader("Réseau")
    network = st.radio("Réseau", ["mainnet", "testnet"], index=0,
                       horizontal=True, label_visibility="collapsed")

    # ── Coins ─────────────────────────────────────────────────────────────────
    st.subheader("Coins")
    with st.spinner("Chargement univers..."):
        universe_list = _fetch_universe(network)
    defaults = [c for c in ["BTC", "ETH", "SOL", "AVAX", "LINK", "ARB", "OP", "NEAR", "FTM", "ATOM"]
                if c in universe_list]
    selected_coins = st.multiselect(
        "Coins dans le panier",
        options=universe_list, default=defaults,
        help="Min. 2 coins. Z-scores calculés intra-panier.",
    )

    # ── Config globale ────────────────────────────────────────────────────────
    st.subheader("Configuration")
    tf_label = st.selectbox(
        "Timeframe barre", list(TF_OPTIONS.keys()),
        index=list(TF_OPTIONS.keys()).index(TF_DEFAULT),
        help="Durée d'une barre — de 10s à 1h",
    )
    tf_seconds_sel = TF_OPTIONS[tf_label]
    h_bars_disp    = _horizon_bars(tf_seconds_sel)
    st.caption(f"Warmup : {h_bars_disp} barres ≈ {h_bars_disp * tf_seconds_sel // 60} min")

    initial_equity = st.number_input("Capital / stratégie ($)", min_value=100.0,
                                     max_value=1_000_000.0, value=DEFAULT_EQUITY, step=1000.0)
    gross_leverage = st.slider("Gross leverage", 0.1, 3.0, DEFAULT_GROSS_LEV, 0.1,
                               help="Exposition totale par stratégie (1.5 = 150%)")

    # ── Stratégies ────────────────────────────────────────────────────────────
    st.subheader("Stratégies")
    st.caption("🟢 ON par défaut | ⚫ OFF par défaut (6-7) | ⏹ Kill | ▶ Start | 🔄 Restart")

    engine_for_strats = st.session_state.hl_engine
    is_running_strats = engine_for_strats is not None and engine_for_strats.is_alive()
    # Snapshot current slot statuses for button rendering (read once, thread-safe)
    _snap_for_buttons = st.session_state.hl_state.snapshot()

    for strat_name in _STRATEGY_REGISTRY:
        slot_snap   = _snap_for_buttons["strategies"].get(strat_name, {})
        slot_status = slot_snap.get("status", "idle")
        slot_active = slot_status in ("live", "warming_up")

        c1, c2, c3 = st.columns([4, 1, 1])
        with c1:
            enabled = st.toggle(
                strat_name,
                value=st.session_state.strat_enabled.get(strat_name, False),
                key=f"tog_{strat_name}",
            )
            st.session_state.strat_enabled[strat_name] = enabled

        if is_running_strats:
            with c2:
                # ⏹ Kill (if active) / ▶ Start (if stopped/idle)
                if slot_active:
                    if st.button("⏹", key=f"stop_{strat_name}",
                                 help=f"Kill {strat_name} — arrête les trades, conserve l'historique"):
                        with st.session_state.hl_state._lock:
                            slot = st.session_state.hl_state.strategies.get(strat_name)
                            if slot:
                                slot.enabled = False
                                slot.status  = "stopped"
                        st.session_state.strat_enabled[strat_name] = False
                        st.rerun()
                else:
                    if st.button("▶", key=f"start_{strat_name}",
                                 help=f"Démarrer {strat_name} — reprend sans reset de l'historique"):
                        with st.session_state.hl_state._lock:
                            slot = st.session_state.hl_state.strategies.get(strat_name)
                            if slot:
                                slot.enabled = True
                                if slot.status in ("stopped", "idle"):
                                    slot.status = "warming_up"
                        st.session_state.strat_enabled[strat_name] = True
                        st.rerun()

            with c3:
                if st.button("🔄", key=f"rst_{strat_name}",
                             help=f"Restart {strat_name} — reset état + nouveau PaperPortfolio"):
                    with st.session_state.hl_state._lock:
                        slot = st.session_state.hl_state.strategies.get(strat_name)
                        if slot:
                            slot.agent.reset()
                            slot.portfolio = PaperPortfolio(
                                initial_equity=st.session_state.hl_state.initial_equity,
                                gross_leverage=st.session_state.hl_state.gross_leverage,
                            )
                            slot.equity_history.clear()
                            slot.turnover_history.clear()
                            slot.fee_history.clear()
                            slot.last_weights    = {}
                            slot.last_bar_weights = {}
                            slot.n_bars  = 0
                            slot.status  = "warming_up"
                            slot.enabled = True   # re-enable if it was killed
                    st.session_state.strat_enabled[strat_name] = True
                    st.rerun()

    st.divider()

    # ── Refresh ───────────────────────────────────────────────────────────────
    st.subheader("Refresh UI")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_s    = st.number_input("Intervalle (sec)", min_value=2, max_value=30,
                                   value=3, step=1, disabled=not auto_refresh)

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = {
        "network"            : network,
        "selected"           : selected_coins,
        "tf_seconds"         : tf_seconds_sel,
        "initial_equity"     : initial_equity,
        "gross_leverage"     : gross_leverage,
        "enabled_strategies" : dict(st.session_state.strat_enabled),
    }

    is_running = st.session_state.hl_engine is not None and st.session_state.hl_engine.is_alive()

    st.divider()
    if not is_running:
        if st.button("▶️ Lancer", type="primary",
                     use_container_width=True, disabled=len(selected_coins) < 2):
            new_state  = SharedState(initial_equity=initial_equity, gross_leverage=gross_leverage)
            new_engine = BackgroundEngine(new_state, cfg)
            new_engine.start()
            st.session_state.hl_state  = new_state
            st.session_state.hl_engine = new_engine
            st.session_state.hl_cfg    = cfg
            st.rerun()
    else:
        c_stop, c_reload = st.columns(2)
        with c_stop:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.hl_engine.stop()
                st.session_state.hl_engine = None
                st.session_state.hl_state  = SharedState()
                st.rerun()
        with c_reload:
            if st.button("⟳ Reload Config", use_container_width=True,
                         help="Relance avec la nouvelle config (timeframe, coins, capital, stratégies actives)"):
                st.session_state.hl_engine.stop()
                new_state  = SharedState(initial_equity=initial_equity, gross_leverage=gross_leverage)
                new_engine = BackgroundEngine(new_state, cfg)
                new_engine.start()
                st.session_state.hl_state  = new_state
                st.session_state.hl_engine = new_engine
                st.session_state.hl_cfg    = cfg
                st.rerun()

    # Statut sidebar
    snap_sb = st.session_state.hl_state.snapshot()
    st.markdown(f"**Statut** &nbsp; {_badge(snap_sb['global_status'])}", unsafe_allow_html=True)
    if snap_sb["error"]:
        st.error(snap_sb["error"])
    if snap_sb["last_update"]:
        age = (datetime.now(timezone.utc) - snap_sb["last_update"]).total_seconds()
        active_cnt = sum(1 for s in snap_sb["strategies"].values() if s["enabled"])
        live_cnt   = sum(1 for s in snap_sb["strategies"].values() if s["status"] == "live")
        st.caption(f"MAJ: {age:.0f}s | {live_cnt}/{active_cnt} strats LIVE")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

snap = st.session_state.hl_state.snapshot()
tf_s = snap["tf_seconds"]
tf_label_disp = next((k for k, v in TF_OPTIONS.items() if v == tf_s), f"{tf_s}s")

# ── Header ────────────────────────────────────────────────────────────────────
col_t, col_s, col_active, col_live = st.columns([3, 2, 2, 2])
with col_t:
    st.markdown("## 🔴 HyperStat — Live Multi-Strategy")
with col_s:
    st.markdown(f"**Statut** &nbsp; {_badge(snap['global_status'])}", unsafe_allow_html=True)
    if snap["last_update"]:
        st.caption(snap["last_update"].strftime("%H:%M:%S UTC"))
with col_active:
    active_strats = [s for s, v in snap["strategies"].items() if v["enabled"]]
    st.metric("Strats actives", str(len(active_strats)))
with col_live:
    live_strats = [s for s, v in snap["strategies"].items() if v["status"] == "live"]
    st.metric("Strats LIVE", str(len(live_strats)))

# ── Idle screen ───────────────────────────────────────────────────────────────
if snap["global_status"] == "idle":
    st.divider()
    st.info("⬅️ Sélectionne les stratégies à activer et clique **Lancer** dans le sidebar.")
    st.markdown("""
**Architecture Multi-Stratégie :**
- Chaque stratégie a son propre **PaperPortfolio indépendant** (capital séparé)
- Toutes les stratégies partagent un **seul flux WebSocket Hyperliquid** (pas de re-téléchargement)
- Filtre **cost-aware** global : `trade only if edge > 2*(fee+slip)+buffer` (30 bps)
- **Start/Stop/Restart** par stratégie sans interruption du flux de données

**Stratégies disponibles :**
| # | Nom | Type | Turnover | Défaut |
|---|-----|------|----------|--------|
| 1 | Stat-Arb MR + FDS | Mean-Reversion | Moyen | ON |
| 2 | Cross-Section Momentum | Momentum | Faible | ON |
| 3 | Funding Carry Pure | Carry | Très faible | ON |
| 4 | PCA Residual MR | Mean-Reversion | Moyen | ON |
| 5 | Quality / Liquidity | Factor | Très faible | ON |
| 6 | Liquidation Reversion | Event-driven | Rare | **OFF** |
| 7 | OB Imbalance | Microstructure | Élevé | **OFF** |
    """)
    st.stop()

# ── Warmup progress ────────────────────────────────────────────────────────────
if snap["global_status"] in ("warming_up", "connecting"):
    warming_strats = {
        name: s for name, s in snap["strategies"].items()
        if s["enabled"] and s["status"] in ("warming_up", "connecting")
    }
    if snap["global_status"] == "connecting":
        st.progress(0.0, text="🔵 Connexion au WebSocket Hyperliquid...")
    elif warming_strats:
        # Show worst progress
        min_pct = min(
            (s["bars_seen"] / max(s["warmup_bars"], 1))
            for s in warming_strats.values()
        )
        slowest = min(warming_strats, key=lambda n: warming_strats[n]["bars_seen"] / max(warming_strats[n]["warmup_bars"], 1))
        s = warming_strats[slowest]
        rem_sec = max(0, (s["warmup_bars"] - s["bars_seen"]) * tf_s)
        rem_str = f"{rem_sec // 60}min {rem_sec % 60}s" if rem_sec >= 60 else f"{rem_sec}s"
        st.progress(
            min(1.0, min_pct),
            text=f"🟡 WARMING UP — {slowest}: {s['bars_seen']}/{s['warmup_bars']} barres (~{rem_str} restantes)"
        )

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_mkt, tab_sig, tab_pf, tab_cfg = st.tabs([
    "📈 Marché Live",
    "📊 Signaux",
    "💼 Portfolio",
    "⚙️ Config",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB : Marché Live
# ─────────────────────────────────────────────────────────────────────────────
with tab_mkt:
    st.markdown('<div class="section-header">📈 Prix en temps réel — Hyperliquid</div>',
                unsafe_allow_html=True)
    mids = snap["mids"]
    if mids:
        # Collect z-scores from first live strategy for signal column
        ref_zscores: Dict[str, float] = {}
        for s in snap["strategies"].values():
            if s["status"] == "live" and s["zscores"]:
                ref_zscores = s["zscores"]
                break

        rows = []
        for sym in snap["selected"]:
            px = mids.get(sym)
            if px is None:
                continue
            z = ref_zscores.get(sym, 0.0)
            rows.append({
                "Coin"   : sym,
                "Prix"   : f"${px:,.4f}" if px < 100 else f"${px:,.2f}",
                "Z-Score": f"{z:+.3f}" if z else "—",
                "Signal" : ("🔴 SHORT" if z > 1.5 else "🟢 LONG" if z < -1.5 else "⚪ FLAT") if z else "—",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"{len(rows)} coins | TF: {tf_label_disp} | "
                       f"MAJ: {snap['last_update'].strftime('%H:%M:%S UTC') if snap['last_update'] else '—'}")
    else:
        st.info("En attente des premiers prix Hyperliquid...")

    if snap["universe"]:
        st.caption(f"Univers Hyperliquid : **{len(snap['universe'])} coins**")


# ─────────────────────────────────────────────────────────────────────────────
# TAB : Signaux
# ─────────────────────────────────────────────────────────────────────────────
with tab_sig:
    st.markdown('<div class="section-header">📊 Signaux par stratégie</div>', unsafe_allow_html=True)

    live_strat_names = [name for name, s in snap["strategies"].items()
                        if s["enabled"] and s["status"] == "live"]

    if not live_strat_names:
        st.info("Aucune stratégie LIVE pour le moment (warmup en cours).")
    else:
        sel_strat = st.selectbox("Stratégie", live_strat_names, key="sig_sel_strat")
        if sel_strat:
            s = snap["strategies"][sel_strat]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### Z-Scores — {sel_strat}")
                _plot_zscores(s["zscores"], color=STRATEGY_COLORS.get(sel_strat, "#e74c3c"))
            with col2:
                st.markdown("#### Poids alloués")
                weights = s["weights"]
                if weights:
                    syms_w = sorted(weights, key=lambda x: abs(weights[x]), reverse=True)[:20]
                    vals_w = [weights[sym] for sym in syms_w]
                    colors_w = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals_w]
                    if go:
                        fig = go.Figure(go.Bar(x=syms_w, y=vals_w, marker_color=colors_w))
                        fig.update_layout(template="plotly_dark", height=270,
                                          margin=dict(l=10, r=10, t=30, b=40))
                        st.plotly_chart(fig, use_container_width=True)

            if s.get("meta"):
                st.caption(f"Méta: {s['meta']}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB : Portfolio
# ─────────────────────────────────────────────────────────────────────────────
with tab_pf:
    st.markdown('<div class="section-header">💼 Portfolio — Paper Trading (no market impact)</div>',
                unsafe_allow_html=True)

    enabled_strat_names = [name for name, s in snap["strategies"].items() if s["enabled"]]
    if not enabled_strat_names:
        st.info("Aucune stratégie active.")
    else:
        # Sub-tabs: one per strategy + Aggregate
        subtab_names  = enabled_strat_names + ["📊 Agrégé"]
        subtabs = st.tabs(subtab_names)

        # ── Per-strategy sub-tabs ─────────────────────────────────────────────
        for i, strat_name in enumerate(enabled_strat_names):
            s    = snap["strategies"][strat_name]
            init = snap["initial_equity"]
            color = STRATEGY_COLORS.get(strat_name, "#2ecc71")

            with subtabs[i]:
                # Status + bar counter
                st.markdown(
                    f'{_badge(s["status"])} &nbsp; '
                    f'**{strat_name}** — {s["n_bars"]} barres | '
                    f'Warmup: {s["bars_seen"]}/{s["warmup_bars"]}',
                    unsafe_allow_html=True
                )

                # ── Métriques principales ──────────────────────────────────────
                upnl = sum(p.get("upnl", 0) for p in s["positions"].values())
                pnl_net   = s["realized_pnl"] + upnl - s["total_fees"]
                pnl_gross = s["realized_pnl"] + upnl
                total_notional = sum(p.get("notional", 0) for p in s["positions"].values())

                r1 = st.columns(6)
                r1[0].metric("PnL net ($)",   f"${pnl_net:+,.2f}", help="Réalisé+uPnL−Frais")
                r1[1].metric("PnL brut ($)",  f"${pnl_gross:+,.2f}")
                r1[2].metric("Frais cumulés", f"${s['total_fees']:.2f}", help="3.5 bps taker")
                r1[3].metric("Retour net",    _pct(pnl_net / init * 100) if init > 0 else "—")
                r1[4].metric("uPnL ouvert",   f"${upnl:+,.2f}")
                r1[5].metric("Notional",      f"${total_notional:,.0f}")

                # ── Métriques risque ──────────────────────────────────────────
                metrics = _compute_metrics(
                    s["equity_history"], init, tf_s, s["n_wins"], s["n_losses"]
                )
                r2 = st.columns(5)
                if metrics:
                    r2[0].metric("Sharpe",    _fmt(metrics["sharpe"]))
                    r2[1].metric("MaxDD",     _pct(metrics["max_dd_pct"]))
                    r2[2].metric("Win Rate",  _pct(metrics["win_rate"] * 100, 1))
                    r2[3].metric("Trades",    str(metrics["n_trades"]))
                r2[4].metric("Barres",        str(s["n_bars"]))

                # ── Turnover + fees average ───────────────────────────────────
                if s["turnover_history"]:
                    avg_to  = float(np.mean([v for _, v in s["turnover_history"]]))
                    avg_fee = float(np.mean([v for _, v in s["fee_history"]])) if s["fee_history"] else 0.0
                    cum_to  = float(sum(v for _, v in s["turnover_history"]))
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Turnover moyen/barre", f"{avg_to:.4f}")
                    tc2.metric("Frais moyen/barre",    f"${avg_fee:.4f}")
                    tc3.metric("Turnover cumulé",       f"{cum_to:.2f}")

                st.divider()

                # ── PnL chart ─────────────────────────────────────────────────
                st.markdown("#### Courbe PnL live")
                _plot_pnl_chart(s["equity_history"], init, s["total_fees"],
                                color=color, label=strat_name)

                # ── Turnover chart ────────────────────────────────────────────
                if s["turnover_history"]:
                    with st.expander("📊 Turnover & Frais par barre"):
                        _plot_turnover_chart(s["turnover_history"], s["fee_history"], color=color)

                # ── Positions ─────────────────────────────────────────────────
                positions = s["positions"]
                if positions:
                    with st.expander(f"📋 Positions ouvertes ({len(positions)})"):
                        rows = []
                        for sym, p in sorted(positions.items(),
                                             key=lambda x: abs(x[1].get("upnl", 0)), reverse=True):
                            qty = p.get("qty", 0)
                            rows.append({
                                "Coin"    : sym,
                                "Side"    : "🟢 LONG" if qty > 0 else "🔴 SHORT",
                                "Qté"     : f"{qty:.6f}",
                                "Entry"   : f"${p.get('entry_px', 0):,.4f}",
                                "Mark"    : f"${p.get('current_px', 0):,.4f}",
                                "Notional": f"${p.get('notional', 0):,.1f}",
                                "uPnL"    : f"${p.get('upnl', 0):+,.2f}",
                                "Z-Score" : f"{p.get('zscore', 0):+.3f}",
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Aggregate sub-tab ─────────────────────────────────────────────────
        with subtabs[-1]:
            st.markdown("#### Vue agrégée — toutes stratégies actives")

            if not enabled_strat_names:
                st.info("Aucune stratégie active.")
            else:
                # Combine equity histories (align by timestamp)
                total_eq_by_ts: Dict[str, float] = {}
                total_init = 0.0

                for strat_name in enabled_strat_names:
                    s    = snap["strategies"][strat_name]
                    init = snap["initial_equity"]
                    total_init += init
                    for ts_iso, eq in s["equity_history"]:
                        total_eq_by_ts[ts_iso] = total_eq_by_ts.get(ts_iso, 0.0) + eq

                if total_eq_by_ts:
                    agg_hist = sorted(total_eq_by_ts.items())

                    # Aggregate metrics
                    total_pnl_net = sum(
                        s["realized_pnl"]
                        + sum(p.get("upnl", 0) for p in s["positions"].values())
                        - s["total_fees"]
                        for s in [snap["strategies"][n] for n in enabled_strat_names]
                    )
                    total_fees = sum(snap["strategies"][n]["total_fees"] for n in enabled_strat_names)
                    total_trades = sum(snap["strategies"][n]["n_wins"] + snap["strategies"][n]["n_losses"]
                                       for n in enabled_strat_names)

                    ag1, ag2, ag3, ag4 = st.columns(4)
                    ag1.metric("PnL net agrégé ($)",  f"${total_pnl_net:+,.2f}")
                    ag2.metric("Frais totaux ($)",     f"${total_fees:.2f}")
                    ag3.metric("Retour net (%)",        _pct(total_pnl_net / total_init * 100) if total_init > 0 else "—")
                    ag4.metric("Trades totaux",         str(total_trades))

                    # Multi-curve chart
                    if go:
                        fig = go.Figure()
                        for strat_name in enabled_strat_names:
                            s = snap["strategies"][strat_name]
                            if s["equity_history"]:
                                ts_p = [datetime.fromisoformat(t) for t, _ in s["equity_history"]]
                                pnl_ = [e - snap["initial_equity"] for _, e in s["equity_history"]]
                                fig.add_trace(go.Scatter(
                                    x=ts_p, y=pnl_,
                                    name=strat_name,
                                    line=dict(color=STRATEGY_COLORS.get(strat_name, "#fff"), width=2),
                                ))
                        if agg_hist:
                            ts_agg = [datetime.fromisoformat(t) for t, _ in agg_hist]
                            pnl_agg = [e - total_init for _, e in agg_hist]
                            fig.add_trace(go.Scatter(
                                x=ts_agg, y=pnl_agg,
                                name="Agrégé",
                                line=dict(color="#fff", width=3, dash="dash"),
                            ))
                        fig.update_layout(
                            template="plotly_dark", height=380,
                            margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(orientation="h"),
                            yaxis=dict(tickprefix="$"),
                            title="PnL net par stratégie + agrégé",
                        )
                        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB : Config
# ─────────────────────────────────────────────────────────────────────────────
with tab_cfg:
    st.markdown('<div class="section-header">⚙️ Configuration courante</div>', unsafe_allow_html=True)

    config = snap.get("config", {})
    if config:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Paramètres globaux**")
            st.json({
                "network"       : config.get("network"),
                "timeframe"     : tf_label_disp,
                "tf_seconds"    : config.get("tf_seconds"),
                "initial_equity": config.get("initial_equity"),
                "gross_leverage": config.get("gross_leverage"),
                "n_coins"       : len(config.get("selected", [])),
                "coins"         : config.get("selected", []),
            })
        with col_c2:
            st.markdown("**Stratégies configurées**")
            strat_cfg = {
                name: ("🟢 ON" if v else "⚫ OFF")
                for name, v in config.get("enabled_strategies", {}).items()
            }
            st.json(strat_cfg)

        st.markdown("**Paramètres cost-aware filter**")
        st.json({
            "fee_bps"           : FEE_BPS,
            "slip_bps"          : SLIP_BPS,
            "round_trip_bps"    : ROUND_TRIP_BPS,
            "buffer_bps"        : BUFFER_BPS,
            "threshold_bps"     : THRESHOLD_BPS,
            "delta_w_min"       : DELTA_W_MIN,
            "min_trade_notional": MIN_TRADE_NOTIONAL,
        })
    else:
        st.info("Démarrez une session pour voir la configuration.")

    if is_running:
        st.warning("⟳ Pour appliquer une nouvelle config, cliquez **Reload Config** dans le sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────

if auto_refresh and snap["global_status"] != "idle":
    time.sleep(int(refresh_s))
    st.rerun()
