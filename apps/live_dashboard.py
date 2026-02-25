# apps/live_dashboard.py
"""
Live Multi-Strategy Paper Trading Dashboard — Hyperliquid Exchange
==================================================================
Interface : Dash + thème sombre CYBORG (Bootstrap)

Architecture :
    BackgroundEngine (thread daemon + asyncio)
        ├── SharedData       : données marché partagées (mids, funding, ob, trades)
        ├── StrategySlot × 7 : état par stratégie (agent + portfolio + historiques)
        └── CostAwareRebalancer : filtre edge > 2*(fee+slip)+buffer
                ↓  SharedState (thread-safe)
    Dash UI — lit SharedState.snapshot() via dcc.Interval
        ├── Sidebar fixe : config + stratégies + lancer/stop
        └── Tabs : Marché | Signaux | Portfolio (sub-tabs par strat + Agrégé) | Config

Lancement :
    python apps/live_dashboard.py
    # ou
    python -m apps.live_dashboard
"""
from __future__ import annotations

import asyncio
import io
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, no_update
import dash_bootstrap_components as dbc

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    _PLOTLY = False

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
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EQUITY    = 10_000.0
DEFAULT_GROSS_LEV = 1.5
FEE_RATE          = 0.00035
MAX_EQUITY_PTS    = 5_000
MIN_BAR_COINS     = 2
FUNDING_POLL_S    = 300.0

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

FEE_BPS           = 3.5
SLIP_BPS          = 10.0
ROUND_TRIP_BPS    = 2.0 * (FEE_BPS + SLIP_BPS)
BUFFER_BPS        = 3.0
THRESHOLD_BPS     = ROUND_TRIP_BPS + BUFFER_BPS
DELTA_W_MIN       = 0.003
MIN_TRADE_NOTIONAL = 0.50

_STRATEGY_REGISTRY: Dict[str, Tuple] = {
    "Stat-Arb MR + FDS"     : (lambda: _StatArbMRWrapper(), True),
    "Cross-Section Momentum": (lambda: CrossSectionalMomentumAgent(), True),
    "Funding Carry Pure"    : (lambda: FundingCarryPureAgent(), True),
    "PCA Residual MR"       : (lambda: PCAResiduaMRAgent(), True),
    "Quality / Liquidity"   : (lambda: QualityLiquidityAgent(), True),
    "Liquidation Reversion" : (lambda: LiquidationReversionAgent(), False),
    "OB Imbalance"          : (lambda: OrderFlowImbalanceAgent(), False),
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

STRAT_NAMES = list(_STRATEGY_REGISTRY.keys())
_DEFAULT_COINS = ["BTC", "ETH", "SOL", "AVAX", "LINK", "ARB", "OP", "NEAR", "FTM", "ATOM"]


def _horizon_bars(tf_seconds: int) -> int:
    return max(HORIZON_BARS_MIN, min(HORIZON_BARS_MAX, 3600 // tf_seconds))


def _strat_slug(name: str) -> str:
    """Convert strategy name to a DOM-safe slug."""
    return (name.replace(" ", "_").replace("/", "_")
                .replace("+", "").replace("-", "_").replace(".", ""))


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS LOGIC  (identique à la version Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

class _StatArbMRWrapper(BaseSignalAgent):
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


class CostAwareRebalancer:

    @staticmethod
    def filter_trades(
        target_w: Dict[str, float],
        current_w: Dict[str, float],
        mids: Dict[str, float],
        equity: float,
    ) -> Dict[str, float]:
        result: Dict[str, float] = dict(current_w)
        for sym, tw in target_w.items():
            cw = current_w.get(sym, 0.0)
            delta_w = tw - cw
            if abs(delta_w) * 10_000 < THRESHOLD_BPS:
                continue
            px = mids.get(sym, 0.0)
            if not (math.isfinite(px) and px > 0):
                continue
            if abs(delta_w * equity) < MIN_TRADE_NOTIONAL:
                continue
            result[sym] = tw
        for sym in list(current_w.keys()):
            if sym not in target_w and abs(current_w.get(sym, 0.0)) > 1e-12:
                result[sym] = 0.0
        return result

    @staticmethod
    def compute_bar_turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
        all_syms = set(list(prev_w.keys()) + list(new_w.keys()))
        return float(sum(abs(new_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_syms))


class PaperPortfolio:
    def __init__(self, initial_equity: float, gross_leverage: float) -> None:
        self.initial_equity  = initial_equity
        self.equity          = initial_equity
        self.gross_leverage  = gross_leverage
        self._pos: Dict[str, float]   = {}
        self._entry: Dict[str, float] = {}
        self.realized_pnl = 0.0
        self.total_fees   = 0.0
        self.fees_this_bar = 0.0
        self.n_wins   = 0
        self.n_losses = 0

    def rebalance(self, target_weights: Dict[str, float], mids: Dict[str, float]) -> None:
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

    def positions_snapshot(self, mids, zscores, weights):
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
        teq = self.total_equity(mids)
        if teq <= 0:
            return {}
        return {
            sym: (qty * mids.get(sym, 0.0)) / teq
            for sym, qty in self._pos.items()
            if sym in mids and math.isfinite(mids.get(sym, float("nan")))
        }


class StrategySlot:
    def __init__(self, name, agent, portfolio, enabled=True):
        self.name      = name
        self.agent     = agent
        self.portfolio = portfolio
        self.enabled   = enabled
        self.status: str = "warming_up"
        self.equity_history: deque   = deque(maxlen=MAX_EQUITY_PTS)
        self.turnover_history: deque = deque(maxlen=MAX_EQUITY_PTS)
        self.fee_history: deque      = deque(maxlen=MAX_EQUITY_PTS)
        self.last_weights: Dict[str, float] = {}
        self.last_zscores: Dict[str, float] = {}
        self.last_meta: Dict[str, Any]      = {}
        self.last_bar_weights: Dict[str, float] = {}
        self.last_bar_ts: Optional[datetime]    = None
        self.n_bars = 0
        self.positions: Dict[str, Dict] = {}

    def snapshot(self):
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


class SharedState:
    def __init__(self, initial_equity=DEFAULT_EQUITY, gross_leverage=DEFAULT_GROSS_LEV):
        self._lock = threading.RLock()
        self.global_status: str = "idle"
        self.error: str = ""
        self.tf_seconds: int   = TF_OPTIONS[TF_DEFAULT]
        self.horizon_bars: int = _horizon_bars(self.tf_seconds)
        self.initial_equity    = initial_equity
        self.gross_leverage    = gross_leverage
        self.mids: Dict[str, float]          = {}
        self.universe: List[str]             = []
        self.selected: List[str]             = []
        self.last_update: Optional[datetime] = None
        self.funding_rates: Dict[str, float] = {}
        self.strategies: Dict[str, StrategySlot] = {}
        self.config: Dict[str, Any] = {}

    def snapshot(self):
        with self._lock:
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
                "strategies"    : {n: s.snapshot() for n, s in self.strategies.items()},
                "config"        : dict(self.config),
            }


class BackgroundEngine:
    def __init__(self, state: SharedState, cfg: Dict[str, Any]) -> None:
        self.state  = state
        self.cfg    = cfg
        self._loop  = asyncio.new_event_loop()
        self._stop  = False
        self._thread = threading.Thread(
            target=self._run_thread, daemon=True, name="HL-MultiStratEngine"
        )
        self._rebalancer = CostAwareRebalancer()

    def start(self) -> None: self._thread.start()
    def stop(self)  -> None: self._stop = True
    def is_alive(self) -> bool: return self._thread.is_alive()

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

        cfg       = self.cfg
        network   = cfg["network"]
        selected  = cfg["selected"]
        tf_sec    = cfg["tf_seconds"]
        h_bars    = _horizon_bars(tf_sec)
        init_eq   = cfg["initial_equity"]
        gross_lev = cfg["gross_leverage"]
        enabled_strats = cfg.get("enabled_strategies", {})

        ep   = (HyperliquidEndpoints.mainnet() if network == "mainnet"
                else HyperliquidEndpoints.testnet())
        rl   = RateLimiter(RateLimiterConfig(weight_budget_per_minute=600))
        rest = HyperliquidRestClient(endpoints=ep, rate_limiter=rl)
        ws_c = HyperliquidWsClient(endpoints=ep)
        mkt  = HyperliquidMarketData(rest, ws_c)

        await ws_c.start()
        try:
            meta     = await mkt.meta()
            universe = sorted([a["name"] for a in meta.get("universe", [])])
        except Exception:
            universe = FALLBACK_COINS

        slots: Dict[str, StrategySlot] = {}
        for strat_name, (factory_fn, default_on) in _STRATEGY_REGISTRY.items():
            should_enable = enabled_strats.get(strat_name, default_on)
            agent = factory_fn()
            if isinstance(agent, _StatArbMRWrapper):
                agent._tf_seconds = tf_sec
                agent._init_strategy()
            portfolio = PaperPortfolio(initial_equity=init_eq, gross_leverage=gross_lev)
            slot = StrategySlot(name=strat_name, agent=agent, portfolio=portfolio,
                                enabled=should_enable)
            slots[strat_name] = slot

        with self.state._lock:
            self.state.universe       = universe
            self.state.selected       = selected
            self.state.initial_equity = init_eq
            self.state.gross_leverage = gross_lev
            self.state.tf_seconds     = tf_sec
            self.state.horizon_bars   = h_bars
            self.state.global_status  = "warming_up"
            self.state.strategies     = slots
            self.state.config         = dict(cfg)

        latest_mids: Dict[str, float]   = {}
        latest_ob: Dict[str, Dict]      = {}
        latest_trades: Dict[str, deque] = {}
        latest_funding: Dict[str, float]= {}
        last_bar_ts: Optional[datetime] = None
        buckets = {"live": selected}
        ob_subscriptions_done = False

        def _maybe_subscribe_extra():
            nonlocal ob_subscriptions_done
            if ob_subscriptions_done:
                return
            ob_needed  = slots.get("OB Imbalance") and slots["OB Imbalance"].enabled
            liq_needed = slots.get("Liquidation Reversion") and slots["Liquidation Reversion"].enabled
            if ob_needed or liq_needed:
                ob_subscriptions_done = True
                async def _sub():
                    for sym in selected[:10]:
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
                asyncio.ensure_future(_sub(), loop=self._loop)

        def _on_l2book(msg, sym):
            if isinstance(msg, dict):
                latest_ob[sym] = msg.get("data", msg)

        def _on_trades(msg, sym):
            if isinstance(msg, dict):
                data = msg.get("data", msg)
                if isinstance(data, list):
                    if sym not in latest_trades:
                        latest_trades[sym] = deque(maxlen=50)
                    for t in data:
                        latest_trades[sym].append(t)

        def _on_mids(msg) -> None:
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
            _maybe_subscribe_extra()
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
                    current_w  = slot.portfolio.current_weights(bar_mids)
                    filtered_w = self._rebalancer.filter_trades(
                        output.weights, current_w, bar_mids,
                        slot.portfolio.total_equity(bar_mids)
                    )
                    total_w = sum(abs(w) for w in filtered_w.values())
                    norm_w  = ({k: v * (gross_lev / total_w) for k, v in filtered_w.items()}
                               if total_w > 1e-9 else {})
                    slot.portfolio.rebalance(norm_w, bar_mids)
                    total_eq = slot.portfolio.total_equity(bar_mids)
                    turnover = self._rebalancer.compute_bar_turnover(slot.last_bar_weights, norm_w)
                    slot.turnover_history.append((now.isoformat(), turnover))
                    slot.fee_history.append((now.isoformat(), slot.portfolio.fees_this_bar))
                    slot.last_bar_weights = dict(norm_w)
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

        async def _poll_funding():
            while not self._stop:
                try:
                    resp = await mkt.meta_and_asset_ctxs()
                    if isinstance(resp, list) and len(resp) == 2:
                        meta_info  = resp[0]
                        asset_ctxs = resp[1]
                        for i, asset in enumerate(meta_info.get("universe", [])):
                            sym = asset.get("name", "")
                            if sym in selected and i < len(asset_ctxs):
                                try:
                                    latest_funding[sym] = float(asset_ctxs[i].get("funding", "0"))
                                except (ValueError, TypeError):
                                    pass
                    with self.state._lock:
                        self.state.funding_rates = dict(latest_funding)
                except Exception:
                    pass
                await asyncio.sleep(FUNDING_POLL_S)

        asyncio.ensure_future(_poll_funding(), loop=self._loop)
        await mkt.stream_all_mids(_on_mids)
        while not self._stop:
            await asyncio.sleep(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE  (module-level, partagé entre tous les callbacks Dash)
# ─────────────────────────────────────────────────────────────────────────────

_g: Dict[str, Any] = {
    "engine"      : None,
    "state"       : SharedState(),
    "launch_time" : None,
    "strat_enabled": {n: d for n, (_, d) in _STRATEGY_REGISTRY.items()},
    "cfg"         : None,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
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


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(46,204,113,{alpha})"


def _compute_metrics(equity_history, initial, tf_seconds, n_wins=0, n_losses=0):
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


def _build_metrics_df(snap: Dict, launch_time: Optional[datetime]) -> pd.DataFrame:
    """Construit un DataFrame avec toutes les métriques de chaque stratégie (pour CSV)."""
    rows = []
    tf_s    = snap.get("tf_seconds", 300)
    init_eq = snap.get("initial_equity", DEFAULT_EQUITY)
    for strat_name, s in snap.get("strategies", {}).items():
        upnl          = sum(p.get("upnl", 0) for p in s.get("positions", {}).values())
        pnl_net       = s.get("realized_pnl", 0) + upnl - s.get("total_fees", 0)
        pnl_gross     = s.get("realized_pnl", 0) + upnl
        total_notional= sum(p.get("notional", 0) for p in s.get("positions", {}).values())
        metrics       = _compute_metrics(
            s.get("equity_history", []), init_eq, tf_s,
            s.get("n_wins", 0), s.get("n_losses", 0)
        )
        to_hist  = s.get("turnover_history", [])
        fee_hist = s.get("fee_history", [])
        rows.append({
            "strategy_name"      : strat_name,
            "launch_date"        : launch_time.strftime("%Y-%m-%d") if launch_time else "",
            "launch_time"        : launch_time.strftime("%H:%M:%S") if launch_time else "",
            "status"             : s.get("status", ""),
            "n_bars"             : s.get("n_bars", 0),
            "initial_equity"     : init_eq,
            "current_equity"     : round(s.get("equity", init_eq), 2),
            "realized_pnl"       : round(s.get("realized_pnl", 0), 4),
            "total_fees"         : round(s.get("total_fees", 0), 4),
            "unrealized_pnl"     : round(upnl, 4),
            "pnl_net"            : round(pnl_net, 4),
            "pnl_gross"          : round(pnl_gross, 4),
            "total_return_pct"   : round(metrics.get("total_return_pct", 0), 4),
            "max_dd_pct"         : round(metrics.get("max_dd_pct", 0), 4),
            "sharpe"             : round(metrics.get("sharpe", 0), 4),
            "win_rate_pct"       : round(metrics.get("win_rate", 0) * 100, 2),
            "n_trades"           : metrics.get("n_trades", 0),
            "n_wins"             : s.get("n_wins", 0),
            "n_losses"           : s.get("n_losses", 0),
            "avg_turnover_bar"   : round(float(np.mean([v for _, v in to_hist])) if to_hist else 0, 6),
            "cumulative_turnover": round(float(sum(v for _, v in to_hist)) if to_hist else 0, 4),
            "avg_fee_per_bar"    : round(float(np.mean([v for _, v in fee_hist])) if fee_hist else 0, 6),
            "open_positions"     : len(s.get("positions", {})),
            "open_notional"      : round(total_notional, 2),
            "open_upnl"          : round(upnl, 4),
            "timeframe_s"        : tf_s,
            "network"            : snap.get("config", {}).get("network", ""),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

_DARK_BG  = "#0d0d1a"
_CARD_BG  = "#111122"
_PLOT_BG  = "#0f0f23"

def _empty_fig(height=300, msg="En attente de données..."):
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


def _make_pnl_figure(equity_history, initial, total_fees, color="#2ecc71", label=""):
    if not _PLOTLY or len(equity_history) < 2:
        return _empty_fig(380, "Warmup en cours...")
    ts_  = [datetime.fromisoformat(t) for t, _ in equity_history]
    eq_  = np.array([e for _, e in equity_history], dtype=float)
    pnl_net   = eq_ - initial
    pnl_gross = pnl_net + total_fees
    peak      = np.maximum.accumulate(eq_)
    dd        = (eq_ - peak) / (peak + 1e-9) * 100
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.04,
        subplot_titles=(f"PnL ($){' — ' + label if label else ''}", "Drawdown (%)"),
    )
    fig.add_trace(
        go.Scatter(x=ts_, y=pnl_gross, name="PnL brut",
                   line=dict(color="#7289da", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=ts_, y=pnl_net, name="PnL net",
                   line=dict(color=color, width=2),
                   fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.08)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#444", row=1, col=1)
    fig.add_trace(
        go.Scatter(x=ts_, y=dd, name="Drawdown %", fill="tozeroy",
                   line=dict(color="#e74c3c", width=1),
                   fillcolor="rgba(231,76,60,0.15)"), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08),
        yaxis=dict(tickprefix="$"), yaxis2=dict(ticksuffix="%"),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
    )
    return fig


def _make_turnover_figure(turnover_history, fee_history, color="#f39c12"):
    if not _PLOTLY or len(turnover_history) < 2:
        return _empty_fig(220)
    ts_  = [datetime.fromisoformat(t) for t, _ in turnover_history]
    to_  = [v for _, v in turnover_history]
    fee_ = [v for _, v in fee_history] if fee_history else [0.0] * len(to_)
    fig  = go.Figure()
    fig.add_trace(go.Bar(x=ts_, y=to_,  name="Turnover",  marker_color=color,     opacity=0.7))
    fig.add_trace(go.Bar(x=ts_, y=fee_, name="Frais ($)", marker_color="#e74c3c", opacity=0.6))
    fig.update_layout(
        template="plotly_dark", height=220, barmode="overlay",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.05),
        title="Turnover / barre (et frais $)",
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
    )
    return fig


def _make_zscores_figure(zscores: Dict[str, float], color="#e74c3c"):
    if not _PLOTLY or not zscores:
        return _empty_fig(270, "Z-scores disponibles après warmup")
    syms = sorted(zscores, key=lambda s: abs(zscores[s]), reverse=True)[:20]
    vals = [zscores[s] for s in syms]
    colors_bar = [color if v > 0 else "#2ecc71" for v in vals]
    fig = go.Figure(go.Bar(x=syms, y=vals, marker_color=colors_bar))
    fig.add_hline(y=1.5,  line_dash="dash", line_color="#f39c12", annotation_text="z_in")
    fig.add_hline(y=-1.5, line_dash="dash", line_color="#f39c12")
    fig.update_layout(
        template="plotly_dark", height=270,
        margin=dict(l=10, r=10, t=30, b=40),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
    )
    return fig


def _make_weights_figure(weights: Dict[str, float], color="#2ecc71"):
    if not _PLOTLY or not weights:
        return _empty_fig(270, "Poids disponibles après warmup")
    syms_w = sorted(weights, key=lambda x: abs(weights[x]), reverse=True)[:20]
    vals_w = [weights[s] for s in syms_w]
    colors_w = [color if v > 0 else "#e74c3c" for v in vals_w]
    fig = go.Figure(go.Bar(x=syms_w, y=vals_w, marker_color=colors_w))
    fig.update_layout(
        template="plotly_dark", height=270,
        margin=dict(l=10, r=10, t=30, b=40),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
    )
    return fig


def _make_aggregate_figure(snap, enabled_names, init_eq):
    if not _PLOTLY:
        return _empty_fig(380)
    total_eq_by_ts: Dict[str, float] = {}
    total_init = 0.0
    for strat_name in enabled_names:
        s = snap["strategies"][strat_name]
        total_init += init_eq
        for ts_iso, eq in s["equity_history"]:
            total_eq_by_ts[ts_iso] = total_eq_by_ts.get(ts_iso, 0.0) + eq
    if not total_eq_by_ts:
        return _empty_fig(380, "Pas encore de données agrégées")
    agg_hist = sorted(total_eq_by_ts.items())
    fig = go.Figure()
    for strat_name in enabled_names:
        s = snap["strategies"][strat_name]
        if s["equity_history"]:
            ts_p = [datetime.fromisoformat(t) for t, _ in s["equity_history"]]
            pnl_ = [e - init_eq for _, e in s["equity_history"]]
            fig.add_trace(go.Scatter(
                x=ts_p, y=pnl_, name=strat_name,
                line=dict(color=STRATEGY_COLORS.get(strat_name, "#fff"), width=2),
            ))
    ts_agg  = [datetime.fromisoformat(t) for t, _ in agg_hist]
    pnl_agg = [e - total_init for _, e in agg_hist]
    fig.add_trace(go.Scatter(
        x=ts_agg, y=pnl_agg, name="Agrégé",
        line=dict(color="#ffffff", width=3, dash="dash"),
    ))
    fig.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
        yaxis=dict(tickprefix="$"),
        title="PnL net par stratégie + agrégé",
        paper_bgcolor=_DARK_BG, plot_bgcolor=_PLOT_BG,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> html.Span:
    _colors = {
        "live"       : ("#1a7a3c", "🟢 LIVE"),
        "connecting" : ("#2a3a6e", "🔵 CONNECTING"),
        "warming_up" : ("#7a5a0a", "🟡 WARMING UP"),
        "stopped"    : ("#333",    "⚫ STOPPED"),
        "error"      : ("#7a1a1a", "🔴 ERROR"),
        "idle"       : ("#2a2a2a", "⚫ IDLE"),
    }
    bg, label = _colors.get(status, ("#333", status.upper()))
    return html.Span(label, style={
        "background": bg, "color": "#fff",
        "padding": "2px 10px", "borderRadius": "12px",
        "fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.5px",
    })


def _kpi(label: str, value: str, color: str = "light") -> dbc.Col:
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div(label, style={"fontSize": "10px", "color": "#888", "marginBottom": "2px"}),
            html.Div(value, style={"fontSize": "14px", "fontWeight": "700",
                                   "color": {"success": "#2ecc71", "danger": "#e74c3c",
                                             "warning": "#f39c12", "info": "#3498db",
                                             "light": "#ddd"}.get(color, "#ddd")}),
        ], className="p-2"),
        style={"background": _CARD_BG, "border": "1px solid #2a2d3a"}),
        className="px-1",
    )


def _render_positions_table(positions: Dict[str, Dict]) -> html.Div:
    if not positions:
        return html.Div()
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
    return dbc.Accordion([
        dbc.AccordionItem([
            dbc.Table.from_dataframe(
                pd.DataFrame(rows), dark=True, hover=True, size="sm", striped=True,
            ),
        ], title=f"📋 Positions ouvertes ({len(positions)})"),
    ], start_collapsed=True, className="mt-2")


def _render_strat_tab_content(strat_name: str, s: Dict, init_eq: float,
                               tf_s: int) -> html.Div:
    color   = STRATEGY_COLORS.get(strat_name, "#2ecc71")
    metrics = _compute_metrics(s["equity_history"], init_eq, tf_s,
                                s["n_wins"], s["n_losses"])
    upnl          = sum(p.get("upnl", 0) for p in s["positions"].values())
    pnl_net       = s["realized_pnl"] + upnl - s["total_fees"]
    pnl_gross     = s["realized_pnl"] + upnl
    total_notional= sum(p.get("notional", 0) for p in s["positions"].values())
    to_hist       = s["turnover_history"]

    return html.Div([
        # Status header + CSV button
        dbc.Row([
            dbc.Col(_status_badge(s["status"]), width="auto", className="pe-2"),
            dbc.Col(html.Small(
                f"{strat_name} — {s['n_bars']} barres | "
                f"Warmup: {s['bars_seen']}/{s['warmup_bars']}",
                className="text-muted",
            )),
            dbc.Col(
                dbc.Button(
                    "📥 Exporter CSV",
                    id={"type": "btn-csv-strat", "index": strat_name},
                    size="sm", color="info", outline=True, n_clicks=0,
                ),
                width="auto",
            ),
        ], className="mb-3 align-items-center"),

        # PnL metrics
        dbc.Row([
            _kpi("PnL net ($)",   f"${pnl_net:+,.2f}",  "success" if pnl_net >= 0 else "danger"),
            _kpi("PnL brut ($)",  f"${pnl_gross:+,.2f}", "light"),
            _kpi("Frais cumulés", f"${s['total_fees']:.2f}", "warning"),
            _kpi("Retour net",    f"{pnl_net/init_eq*100:+.2f}%" if init_eq else "—", "info"),
            _kpi("uPnL ouvert",   f"${upnl:+,.2f}", "success" if upnl >= 0 else "danger"),
            _kpi("Notional",      f"${total_notional:,.0f}", "light"),
        ], className="mb-2 g-1"),

        # Risk metrics
        dbc.Row([
            _kpi("Sharpe",    f"{metrics.get('sharpe', 0):+.3f}" if metrics else "—", "info"),
            _kpi("Max DD",    f"{metrics.get('max_dd_pct', 0):.2f}%" if metrics else "—", "danger"),
            _kpi("Win Rate",  f"{metrics.get('win_rate', 0)*100:.1f}%" if metrics else "—", "success"),
            _kpi("Trades",    str(metrics.get("n_trades", 0)) if metrics else "0", "light"),
            _kpi("Barres",    str(s["n_bars"]), "light"),
        ] + ([_kpi("TO moyen/barre",
                   f"{np.mean([v for _, v in to_hist]):.4f}", "light")]
             if to_hist else []),
        className="mb-3 g-1"),

        html.Hr(style={"borderColor": "#2a2d3a"}),

        # PnL chart
        html.H6("Courbe PnL live", style={"color": "#aaa", "fontSize": "12px"}),
        dcc.Graph(
            figure=_make_pnl_figure(s["equity_history"], init_eq,
                                     s["total_fees"], color=color, label=strat_name),
            config={"displayModeBar": False},
        ),

        # Turnover accordion
        (dbc.Accordion([
            dbc.AccordionItem([
                dcc.Graph(
                    figure=_make_turnover_figure(s["turnover_history"],
                                                 s["fee_history"], color=color),
                    config={"displayModeBar": False},
                ),
            ], title="📊 Turnover & Frais par barre"),
        ], start_collapsed=True, className="mb-2") if to_hist else html.Div()),

        # Positions
        _render_positions_table(s["positions"]),
    ], style={"padding": "12px 0"})


def _render_aggregate_tab(snap, enabled_names, init_eq, tf_s):
    total_pnl_net = sum(
        snap["strategies"][n]["realized_pnl"]
        + sum(p.get("upnl", 0) for p in snap["strategies"][n]["positions"].values())
        - snap["strategies"][n]["total_fees"]
        for n in enabled_names
    )
    total_fees   = sum(snap["strategies"][n]["total_fees"] for n in enabled_names)
    total_trades = sum(snap["strategies"][n]["n_wins"] + snap["strategies"][n]["n_losses"]
                       for n in enabled_names)
    total_init   = init_eq * len(enabled_names)

    return html.Div([
        html.H6("Vue agrégée — toutes stratégies actives",
                style={"color": "#aaa", "fontSize": "12px", "marginBottom": "12px"}),
        dbc.Row([
            _kpi("PnL net agrégé", f"${total_pnl_net:+,.2f}",
                 "success" if total_pnl_net >= 0 else "danger"),
            _kpi("Frais totaux",   f"${total_fees:.2f}", "warning"),
            _kpi("Retour net",
                 f"{total_pnl_net/total_init*100:+.2f}%" if total_init else "—", "info"),
            _kpi("Trades totaux",  str(total_trades), "light"),
        ], className="mb-3 g-1"),
        dcc.Graph(
            figure=_make_aggregate_figure(snap, enabled_names, init_eq),
            config={"displayModeBar": False},
        ),
    ], style={"padding": "12px 0"})


# ─────────────────────────────────────────────────────────────────────────────
# DASH APP
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="HyperStat — Multi-Strategy Live",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

_SIDEBAR_W = "290px"
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


def _sidebar_layout():
    return html.Div([
        html.Div([
            html.Span("🔴", style={"fontSize": "18px"}),
            html.Span(" HyperStat Multi-Strategy",
                      style={"color": "#e74c3c", "fontWeight": "700",
                             "fontSize": "14px", "marginLeft": "6px"}),
        ], className="mb-2"),
        html.Hr(style={"borderColor": "#1e1e30", "margin": "8px 0"}),

        # Network
        html.Div("RÉSEAU", style={"color": "#555", "fontSize": "10px",
                                   "letterSpacing": "1px", "marginBottom": "4px"}),
        dbc.RadioItems(
            id="input-network", value="mainnet", inline=True,
            options=[{"label": "mainnet", "value": "mainnet"},
                     {"label": "testnet", "value": "testnet"}],
            className="mb-2",
        ),
        html.Hr(style={"borderColor": "#1e1e30", "margin": "8px 0"}),

        # Coins
        html.Div("COINS", style={"color": "#555", "fontSize": "10px",
                                  "letterSpacing": "1px", "marginBottom": "4px"}),
        dcc.Dropdown(
            id="input-coins",
            options=[{"label": c, "value": c} for c in _DEFAULT_COINS],
            value=_DEFAULT_COINS, multi=True, clearable=False,
            style={"backgroundColor": "#111", "fontSize": "12px"},
            className="mb-3",
        ),
        html.Hr(style={"borderColor": "#1e1e30", "margin": "8px 0"}),

        # Config
        html.Div("CONFIGURATION", style={"color": "#555", "fontSize": "10px",
                                          "letterSpacing": "1px", "marginBottom": "6px"}),
        html.Div("Timeframe", style={"fontSize": "11px", "color": "#aaa", "marginBottom": "3px"}),
        dcc.Dropdown(
            id="input-tf",
            options=[{"label": k, "value": k} for k in TF_OPTIONS],
            value=TF_DEFAULT, clearable=False,
            style={"backgroundColor": "#111", "fontSize": "12px"},
            className="mb-1",
        ),
        html.Div(id="tf-caption", style={"fontSize": "10px", "color": "#666",
                                          "marginBottom": "8px"}),
        html.Div("Capital / stratégie ($)",
                 style={"fontSize": "11px", "color": "#aaa", "marginBottom": "3px"}),
        dbc.Input(
            id="input-equity", type="number",
            value=DEFAULT_EQUITY, min=100, max=1_000_000, step=1000,
            style={"backgroundColor": "#111", "color": "#ddd", "fontSize": "12px",
                   "border": "1px solid #2a2d3a"},
            className="mb-2",
        ),
        html.Div(id="lev-caption",
                 style={"fontSize": "11px", "color": "#aaa", "marginBottom": "3px"}),
        dcc.Slider(
            id="input-leverage", min=0.1, max=3.0, step=0.1, value=DEFAULT_GROSS_LEV,
            marks={0.5: "0.5", 1.0: "1×", 1.5: "1.5×", 2.0: "2×", 3.0: "3×"},
            className="mb-3",
        ),

        html.Hr(style={"borderColor": "#1e1e30", "margin": "8px 0"}),
        html.Div("STRATÉGIES", style={"color": "#555", "fontSize": "10px",
                                       "letterSpacing": "1px", "marginBottom": "6px"}),
        html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Switch(
                        id={"type": "toggle-strat", "index": n},
                        label=html.Span(n, style={"fontSize": "11px"}),
                        value=d,
                    ), width=7,
                ),
                dbc.Col([
                    dbc.Button("⏹", id={"type": "btn-strat-stop",    "index": n},
                               size="sm", color="danger",  outline=True,
                               className="me-1 px-1 py-0", n_clicks=0),
                    dbc.Button("▶", id={"type": "btn-strat-start",   "index": n},
                               size="sm", color="success", outline=True,
                               className="me-1 px-1 py-0", n_clicks=0),
                    dbc.Button("↺", id={"type": "btn-strat-restart", "index": n},
                               size="sm", color="warning", outline=True,
                               className="px-1 py-0", n_clicks=0),
                ], width=5, className="d-flex align-items-center"),
            ], className="mb-1 align-items-center g-1")
            for n, (_, d) in _STRATEGY_REGISTRY.items()
        ]),

        html.Hr(style={"borderColor": "#1e1e30", "margin": "8px 0"}),
        html.Div("AUTO-REFRESH", style={"color": "#555", "fontSize": "10px",
                                         "letterSpacing": "1px", "marginBottom": "4px"}),
        dbc.Row([
            dbc.Col(dbc.Switch(id="toggle-autorefresh", value=True,
                               label=html.Span("Auto", style={"fontSize": "11px"})), width=5),
            dbc.Col(dbc.Input(
                id="input-refresh-s", type="number", value=3, min=2, max=60,
                style={"backgroundColor": "#111", "color": "#ddd", "fontSize": "12px",
                       "border": "1px solid #2a2d3a"},
            ), width=7),
        ], className="mb-3 align-items-center g-1"),

        html.Hr(style={"borderColor": "#1e1e30", "margin": "8px 0"}),
        dbc.Button("▶  Lancer", id="btn-launch", color="success",
                   className="w-100 mb-2", n_clicks=0, size="sm"),
        dbc.Row([
            dbc.Col(dbc.Button("⏹ Stop",  id="btn-stop",   color="danger",
                               className="w-100", n_clicks=0, size="sm"), width=6),
            dbc.Col(dbc.Button("↺ Reload", id="btn-reload", color="secondary",
                               className="w-100", n_clicks=0, size="sm"), width=6),
        ], className="mb-2 g-1"),

        html.Hr(style={"borderColor": "#1e1e30", "margin": "8px 0"}),
        html.Div(id="sidebar-status"),
    ], style=_SIDEBAR_STYLE)


app.layout = html.Div([
    # Hidden helpers
    dcc.Interval(id="interval-main", interval=3_000, n_intervals=0, disabled=True),
    dcc.Download(id="download-all-csv"),
    dcc.Download(id="download-strat-csv"),
    dcc.Store(id="store-dummy", data=0),   # forces first render

    # Layout
    _sidebar_layout(),

    # Main content
    html.Div([
        # Header
        dbc.Row([
            dbc.Col(
                html.H4("🔴 HyperStat — Live Multi-Strategy",
                        style={"color": "#e74c3c", "margin": 0, "fontSize": "18px"}),
                width=5,
            ),
            dbc.Col(html.Div(id="header-status"), width=4),
            dbc.Col(html.Div(id="header-kpis"), width=3),
        ], className="mb-3 align-items-center"),

        html.Div(id="warmup-progress", className="mb-2"),

        # Export all CSV
        dbc.Row([
            dbc.Col(
                dbc.Button("📥 Exporter CSV — toutes stratégies",
                           id="btn-export-all", color="info", outline=True,
                           size="sm", n_clicks=0),
                width="auto",
            ),
        ], className="mb-2"),

        html.Hr(style={"borderColor": "#1e1e30"}),

        # Main tabs
        dbc.Tabs([
            dbc.Tab(html.Div(id="tab-market"),    label="📈 Marché Live", tab_id="mkt"),
            dbc.Tab(html.Div(id="tab-signals"),   label="📊 Signaux",     tab_id="sig"),
            dbc.Tab(html.Div(id="tab-portfolio"), label="💼 Portfolio",   tab_id="pf"),
            dbc.Tab(html.Div(id="tab-config"),    label="⚙️ Config",      tab_id="cfg"),
        ], id="tabs-main", active_tab="mkt",
           style={"borderBottom": "1px solid #2a2d3a"}),
    ], style=_CONTENT_STYLE),

], style={"backgroundColor": _DARK_BG, "minHeight": "100vh", "fontFamily": "monospace"})


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("input-coins", "options"),
    Output("input-coins", "value"),
    Input("input-network", "value"),
)
def populate_coins(network):
    universe = _fetch_universe(network or "mainnet")
    options  = [{"label": c, "value": c} for c in universe]
    defaults = [c for c in _DEFAULT_COINS if c in universe]
    return options, defaults


@app.callback(
    Output("tf-caption",  "children"),
    Output("lev-caption", "children"),
    Input("input-tf",       "value"),
    Input("input-leverage", "value"),
)
def update_captions(tf_label, leverage):
    tf_s   = TF_OPTIONS.get(tf_label or TF_DEFAULT, 300)
    h      = _horizon_bars(tf_s)
    return (f"Warmup: {h} barres ≈ {h * tf_s // 60} min",
            f"Gross leverage: {float(leverage or DEFAULT_GROSS_LEV):.1f}×")


@app.callback(
    Output("interval-main", "disabled"),
    Output("interval-main", "interval"),
    Output("sidebar-status", "children"),
    Input("btn-launch",         "n_clicks"),
    Input("btn-stop",           "n_clicks"),
    Input("btn-reload",         "n_clicks"),
    Input("toggle-autorefresh", "value"),
    Input("input-refresh-s",    "value"),
    State("input-network",   "value"),
    State("input-coins",     "value"),
    State("input-tf",        "value"),
    State("input-equity",    "value"),
    State("input-leverage",  "value"),
    State({"type": "toggle-strat", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def control_engine(n_launch, n_stop, n_reload,
                   auto_refresh, refresh_s,
                   network, coins, tf_label, equity, leverage,
                   strat_toggle_values):
    triggered = ctx.triggered_id

    if triggered in ("btn-launch", "btn-reload"):
        strat_enabled = {
            name: bool(v)
            for name, v in zip(STRAT_NAMES, strat_toggle_values or [])
        }
        cfg = {
            "network"            : network or "mainnet",
            "selected"           : list(coins) if coins else _DEFAULT_COINS,
            "tf_seconds"         : TF_OPTIONS.get(tf_label or TF_DEFAULT, 300),
            "initial_equity"     : float(equity or DEFAULT_EQUITY),
            "gross_leverage"     : float(leverage or DEFAULT_GROSS_LEV),
            "enabled_strategies" : strat_enabled,
        }
        if _g["engine"] is not None:
            _g["engine"].stop()
        new_state  = SharedState(initial_equity=cfg["initial_equity"],
                                 gross_leverage=cfg["gross_leverage"])
        new_engine = BackgroundEngine(new_state, cfg)
        new_engine.start()
        _g["engine"]      = new_engine
        _g["state"]       = new_state
        _g["launch_time"] = datetime.now(timezone.utc)
        _g["cfg"]         = cfg

    elif triggered == "btn-stop":
        if _g["engine"] is not None:
            _g["engine"].stop()
            _g["engine"] = None
        _g["state"]       = SharedState()
        _g["launch_time"] = None

    interval_ms = max(2, int(refresh_s or 3)) * 1000
    disabled    = not bool(auto_refresh)

    # Build sidebar status
    snap   = _g["state"].snapshot()
    status = snap["global_status"]
    err    = snap["error"]
    last_u = snap["last_update"]

    status_items = [_status_badge(status)]
    if err:
        status_items.append(
            dbc.Alert(err, color="danger", className="mt-1 p-1", style={"fontSize": "11px"})
        )
    if last_u:
        age = (datetime.now(timezone.utc) - last_u).total_seconds()
        ac  = sum(1 for s in snap["strategies"].values() if s["enabled"])
        lv  = sum(1 for s in snap["strategies"].values() if s["status"] == "live")
        status_items.append(html.Div(
            f"MAJ: {age:.0f}s | {lv}/{ac} LIVE",
            style={"fontSize": "10px", "color": "#666", "marginTop": "4px"},
        ))
    if _g["launch_time"]:
        status_items.append(html.Div(
            f"Lancé: {_g['launch_time'].strftime('%H:%M:%S UTC')}",
            style={"fontSize": "10px", "color": "#555"},
        ))

    return disabled, interval_ms, html.Div(status_items)


@app.callback(
    Output("header-status",   "children"),
    Output("header-kpis",     "children"),
    Output("warmup-progress", "children"),
    Output("tab-market",      "children"),
    Output("tab-signals",     "children"),
    Output("tab-portfolio",   "children"),
    Output("tab-config",      "children"),
    Input("interval-main", "n_intervals"),
    Input("store-dummy",   "data"),   # triggers once on load
)
def update_display(_n, _dummy):
    snap   = _g["state"].snapshot()
    status = snap["global_status"]
    tf_s   = snap["tf_seconds"]
    tf_lbl = next((k for k, v in TF_OPTIONS.items() if v == tf_s), f"{tf_s}s")
    last_u = snap["last_update"]

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_status = html.Div([
        _status_badge(status),
        html.Span(
            f"  {last_u.strftime('%H:%M:%S UTC') if last_u else '—'}",
            style={"fontSize": "11px", "color": "#666"},
        ),
    ])
    ac = sum(1 for s in snap["strategies"].values() if s["enabled"])
    lv = sum(1 for s in snap["strategies"].values() if s["status"] == "live")
    hdr_kpis = dbc.Row([
        dbc.Col(dbc.Badge(f"{ac} actives", color="secondary", className="me-1"), width="auto"),
        dbc.Col(dbc.Badge(f"{lv} LIVE",    color="success"  if lv else "dark"), width="auto"),
    ], className="justify-content-end")

    # ── Warmup progress ───────────────────────────────────────────────────────
    warmup_comp = html.Div()
    if status in ("warming_up", "connecting"):
        if status == "connecting":
            warmup_comp = dbc.Progress(
                value=0, label="🔵 Connexion WebSocket Hyperliquid...",
                animated=True, color="info", style={"height": "20px"},
            )
        else:
            ws = {n: s for n, s in snap["strategies"].items()
                  if s["enabled"] and s["status"] in ("warming_up", "connecting")}
            if ws:
                min_pct = min(s["bars_seen"] / max(s["warmup_bars"], 1)
                              for s in ws.values()) * 100
                slowest = min(ws, key=lambda n: ws[n]["bars_seen"] / max(ws[n]["warmup_bars"], 1))
                sv = ws[slowest]
                rem = max(0, (sv["warmup_bars"] - sv["bars_seen"]) * tf_s)
                rem_s = f"{rem // 60}min {rem % 60}s" if rem >= 60 else f"{rem}s"
                warmup_comp = dbc.Progress(
                    value=min(100, min_pct),
                    label=f"🟡 {slowest}: {sv['bars_seen']}/{sv['warmup_bars']} barres (~{rem_s})",
                    animated=True, color="warning", style={"height": "20px"},
                )

    # ── IDLE screen ───────────────────────────────────────────────────────────
    if status == "idle":
        idle_screen = dbc.Alert([
            html.H5("⬅️ Configure et clique Lancer dans le sidebar", className="mb-3"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("#"), html.Th("Stratégie"),
                                    html.Th("Type"), html.Th("Turnover"), html.Th("Défaut")])),
                html.Tbody([
                    html.Tr([html.Td(str(i+1)), html.Td(n), html.Td(t), html.Td(to), html.Td(d)])
                    for i, (n, t, to, d) in enumerate([
                        ("Stat-Arb MR + FDS",      "Mean-Rev",     "Moyen",     "🟢 ON"),
                        ("Cross-Section Momentum",  "Momentum",     "Faible",    "🟢 ON"),
                        ("Funding Carry Pure",       "Carry",        "Très faible","🟢 ON"),
                        ("PCA Residual MR",          "Mean-Rev",     "Moyen",     "🟢 ON"),
                        ("Quality / Liquidity",      "Factor",       "Très faible","🟢 ON"),
                        ("Liquidation Reversion",    "Event-driven", "Rare",      "⚫ OFF"),
                        ("OB Imbalance",             "Microstructure","Élevé",    "⚫ OFF"),
                    ])
                ]),
            ], dark=True, hover=True, size="sm"),
        ], color="dark")
        return hdr_status, hdr_kpis, warmup_comp, idle_screen, \
               html.Div(), html.Div(), html.Div()

    # ── Tab: Marché Live ──────────────────────────────────────────────────────
    mids = snap["mids"]
    ref_z: Dict[str, float] = {}
    for s in snap["strategies"].values():
        if s["status"] == "live" and s["zscores"]:
            ref_z = s["zscores"]
            break
    if mids:
        rows = []
        for sym in snap["selected"]:
            px = mids.get(sym)
            if px is None:
                continue
            z = ref_z.get(sym, 0.0)
            rows.append({
                "Coin"   : sym,
                "Prix"   : f"${px:,.4f}" if px < 100 else f"${px:,.2f}",
                "Z-Score": f"{z:+.3f}" if z else "—",
                "Signal" : ("🔴 SHORT" if z > 1.5 else
                             "🟢 LONG"  if z < -1.5 else "⚪ FLAT") if z else "—",
            })
        mkt_tab = html.Div([
            html.H6(f"📈 Prix temps réel — Hyperliquid | TF: {tf_lbl}",
                    style={"color": "#3498db", "fontSize": "12px"}),
            dbc.Table.from_dataframe(
                pd.DataFrame(rows), dark=True, hover=True, size="sm", striped=True,
            ) if rows else dbc.Alert("En attente des prix...", color="secondary"),
            html.Small(
                f"{len(rows)} coins | "
                f"MAJ: {snap['last_update'].strftime('%H:%M:%S UTC') if snap['last_update'] else '—'}",
                style={"color": "#555"},
            ),
        ])
    else:
        mkt_tab = dbc.Alert("En attente des premiers prix Hyperliquid...", color="secondary")

    # ── Tab: Signaux ──────────────────────────────────────────────────────────
    live_names = [n for n, s in snap["strategies"].items()
                  if s["enabled"] and s["status"] == "live"]
    if not live_names:
        sig_tab = dbc.Alert("Aucune stratégie LIVE (warmup en cours).", color="warning")
    else:
        sig_subtabs = []
        for sn in live_names:
            s     = snap["strategies"][sn]
            color = STRATEGY_COLORS.get(sn, "#e74c3c")
            slug  = _strat_slug(sn)
            sig_subtabs.append(dbc.Tab(
                dbc.Row([
                    dbc.Col([
                        html.H6(f"Z-Scores", style={"fontSize": "12px", "color": "#aaa"}),
                        dcc.Graph(figure=_make_zscores_figure(s["zscores"], color),
                                  config={"displayModeBar": False}),
                    ], width=6),
                    dbc.Col([
                        html.H6("Poids alloués", style={"fontSize": "12px", "color": "#aaa"}),
                        dcc.Graph(figure=_make_weights_figure(s["weights"], color),
                                  config={"displayModeBar": False}),
                    ], width=6),
                ], style={"padding": "12px 0"}),
                label=sn[:22], tab_id=f"sig-{slug}",
            ))
        sig_tab = dbc.Tabs(sig_subtabs, active_tab=f"sig-{_strat_slug(live_names[0])}")

    # ── Tab: Portfolio ────────────────────────────────────────────────────────
    enabled_names = [n for n, s in snap["strategies"].items() if s["enabled"]]
    init_eq       = snap["initial_equity"]
    if not enabled_names:
        pf_tab = dbc.Alert("Aucune stratégie active.", color="warning")
    else:
        pf_subtabs = []
        for sn in enabled_names:
            s    = snap["strategies"][sn]
            slug = _strat_slug(sn)
            pf_subtabs.append(dbc.Tab(
                _render_strat_tab_content(sn, s, init_eq, tf_s),
                label=sn[:22], tab_id=f"pf-{slug}",
            ))
        pf_subtabs.append(dbc.Tab(
            _render_aggregate_tab(snap, enabled_names, init_eq, tf_s),
            label="📊 Agrégé", tab_id="pf-agrege",
        ))
        pf_tab = dbc.Tabs(
            pf_subtabs,
            active_tab=f"pf-{_strat_slug(enabled_names[0])}",
        )

    # ── Tab: Config ───────────────────────────────────────────────────────────
    cfg = snap.get("config", {})
    if cfg:
        cfg_tab = dbc.Row([
            dbc.Col([
                html.H6("Paramètres", style={"color": "#aaa", "fontSize": "12px"}),
                dbc.Table([
                    html.Tbody([
                        html.Tr([html.Td("Réseau"),    html.Td(cfg.get("network", "—"))]),
                        html.Tr([html.Td("Timeframe"), html.Td(tf_lbl)]),
                        html.Tr([html.Td("Capital"),   html.Td(f"${cfg.get('initial_equity', 0):,.0f}")]),
                        html.Tr([html.Td("Levier"),    html.Td(f"{cfg.get('gross_leverage', 0):.1f}×")]),
                        html.Tr([html.Td("Coins"),     html.Td(str(len(cfg.get("selected", []))))]),
                    ])
                ], dark=True, size="sm", className="mb-3"),
                html.H6("Cost-Aware Filter", style={"color": "#aaa", "fontSize": "12px"}),
                dbc.Table([
                    html.Tbody([
                        html.Tr([html.Td("Fee bps"),        html.Td(f"{FEE_BPS}")]),
                        html.Tr([html.Td("Slippage bps"),   html.Td(f"{SLIP_BPS}")]),
                        html.Tr([html.Td("Threshold bps"),  html.Td(f"{THRESHOLD_BPS}")]),
                        html.Tr([html.Td("Min notional"),   html.Td(f"${MIN_TRADE_NOTIONAL}")]),
                    ])
                ], dark=True, size="sm"),
            ], width=6),
            dbc.Col([
                html.H6("Stratégies actives", style={"color": "#aaa", "fontSize": "12px"}),
                dbc.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td(n, style={"fontSize": "11px"}),
                            html.Td(html.Span("🟢 ON", style={"color": "#2ecc71"})
                                    if v else html.Span("⚫ OFF", style={"color": "#555"})),
                        ])
                        for n, v in cfg.get("enabled_strategies", {}).items()
                    ])
                ], dark=True, size="sm"),
            ], width=6),
        ], className="mt-3")
    else:
        cfg_tab = dbc.Alert("Démarrez une session pour voir la configuration.",
                            color="secondary")

    return (hdr_status, hdr_kpis, warmup_comp,
            mkt_tab, sig_tab, pf_tab, cfg_tab)


@app.callback(
    Output("store-dummy", "data"),   # just forces re-render via a no-op store update
    Input({"type": "btn-strat-stop",    "index": ALL}, "n_clicks"),
    Input({"type": "btn-strat-start",   "index": ALL}, "n_clicks"),
    Input({"type": "btn-strat-restart", "index": ALL}, "n_clicks"),
    Input({"type": "toggle-strat",      "index": ALL}, "value"),
    State("store-dummy", "data"),
    prevent_initial_call=True,
)
def handle_strat_actions(stop_clicks, start_clicks, restart_clicks,
                          toggle_values, dummy_val):
    triggered = ctx.triggered_id
    if triggered is None:
        return no_update

    t = triggered if isinstance(triggered, dict) else {}
    action     = t.get("type", "")
    strat_name = t.get("index", "")

    state = _g["state"]

    if action == "toggle-strat":
        # Update enabled flags from toggles
        for name, val in zip(STRAT_NAMES, toggle_values or []):
            if val is not None:
                _g["strat_enabled"][name] = bool(val)
                with state._lock:
                    slot = state.strategies.get(name)
                    if slot:
                        slot.enabled = bool(val)
                        if not val and slot.status == "live":
                            slot.status = "stopped"
                        elif val and slot.status == "stopped":
                            slot.status = "warming_up"

    elif action == "btn-strat-stop" and strat_name:
        with state._lock:
            slot = state.strategies.get(strat_name)
            if slot:
                slot.enabled = False
                slot.status  = "stopped"
        _g["strat_enabled"][strat_name] = False

    elif action == "btn-strat-start" and strat_name:
        with state._lock:
            slot = state.strategies.get(strat_name)
            if slot:
                slot.enabled = True
                if slot.status in ("stopped", "idle"):
                    slot.status = "warming_up"
        _g["strat_enabled"][strat_name] = True

    elif action == "btn-strat-restart" and strat_name:
        with state._lock:
            slot = state.strategies.get(strat_name)
            if slot:
                slot.agent.reset()
                slot.portfolio = PaperPortfolio(
                    initial_equity=state.initial_equity,
                    gross_leverage=state.gross_leverage,
                )
                slot.equity_history.clear()
                slot.turnover_history.clear()
                slot.fee_history.clear()
                slot.last_weights     = {}
                slot.last_bar_weights = {}
                slot.n_bars  = 0
                slot.status  = "warming_up"
                slot.enabled = True
        _g["strat_enabled"][strat_name] = True

    return (dummy_val or 0) + 1


# ── CSV Export — global (toutes stratégies) ───────────────────────────────────

@app.callback(
    Output("download-all-csv", "data"),
    Input("btn-export-all", "n_clicks"),
    prevent_initial_call=True,
)
def export_all_csv(n):
    snap        = _g["state"].snapshot()
    launch_time = _g.get("launch_time")
    df = _build_metrics_df(snap, launch_time)
    if df.empty:
        return no_update
    ts_str   = (launch_time.strftime("%Y-%m-%d_%H%M%S")
                if launch_time
                else datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S"))
    filename = f"hyperstat_all_{ts_str}.csv"
    return dcc.send_data_frame(df.to_csv, filename=filename, index=False)


# ── CSV Export — par stratégie (bouton dans chaque sub-tab Portfolio) ─────────

@app.callback(
    Output("download-strat-csv", "data"),
    Input({"type": "btn-csv-strat", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def export_strat_csv(n_clicks_list):
    triggered = ctx.triggered_id
    if triggered is None or not any(n for n in (n_clicks_list or []) if n):
        return no_update

    strat_name  = triggered["index"]
    snap        = _g["state"].snapshot()
    launch_time = _g.get("launch_time")
    s = snap["strategies"].get(strat_name)
    if not s:
        return no_update

    tf_s    = snap.get("tf_seconds", 300)
    init_eq = snap.get("initial_equity", DEFAULT_EQUITY)
    upnl    = sum(p.get("upnl", 0) for p in s.get("positions", {}).values())
    pnl_net = s.get("realized_pnl", 0) + upnl - s.get("total_fees", 0)
    metrics = _compute_metrics(s.get("equity_history", []), init_eq, tf_s,
                                s.get("n_wins", 0), s.get("n_losses", 0))
    to_hist  = s.get("turnover_history", [])
    fee_hist = s.get("fee_history", [])

    row = {
        "strategy_name"      : strat_name,
        "launch_date"        : launch_time.strftime("%Y-%m-%d") if launch_time else "",
        "launch_time"        : launch_time.strftime("%H:%M:%S") if launch_time else "",
        "status"             : s.get("status", ""),
        "n_bars"             : s.get("n_bars", 0),
        "initial_equity"     : init_eq,
        "current_equity"     : round(s.get("equity", init_eq), 2),
        "realized_pnl"       : round(s.get("realized_pnl", 0), 4),
        "total_fees"         : round(s.get("total_fees", 0), 4),
        "unrealized_pnl"     : round(upnl, 4),
        "pnl_net"            : round(pnl_net, 4),
        "total_return_pct"   : round(metrics.get("total_return_pct", 0), 4),
        "max_dd_pct"         : round(metrics.get("max_dd_pct", 0), 4),
        "sharpe"             : round(metrics.get("sharpe", 0), 4),
        "win_rate_pct"       : round(metrics.get("win_rate", 0) * 100, 2),
        "n_trades"           : metrics.get("n_trades", 0),
        "n_wins"             : s.get("n_wins", 0),
        "n_losses"           : s.get("n_losses", 0),
        "avg_turnover_bar"   : round(float(np.mean([v for _, v in to_hist])) if to_hist else 0, 6),
        "cumulative_turnover": round(float(sum(v for _, v in to_hist)) if to_hist else 0, 4),
        "avg_fee_per_bar"    : round(float(np.mean([v for _, v in fee_hist])) if fee_hist else 0, 6),
        "open_positions"     : len(s.get("positions", {})),
        "open_notional"      : round(sum(p.get("notional", 0)
                                         for p in s.get("positions", {}).values()), 2),
        "timeframe_s"        : tf_s,
        "network"            : snap.get("config", {}).get("network", ""),
    }
    df = pd.DataFrame([row])

    slug     = _strat_slug(strat_name)
    ts_str   = (launch_time.strftime("%Y-%m-%d_%H%M%S")
                if launch_time
                else datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S"))
    filename = f"{slug}_{ts_str}.csv"

    return dcc.send_data_frame(df.to_csv, filename=filename, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)