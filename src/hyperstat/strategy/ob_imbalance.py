# src/hyperstat/strategy/ob_imbalance.py
"""
Order Book Imbalance Agent (SignalAgent 5)  [OFF by default]
=============================================================
Uses the public `l2Book` WebSocket feed for order flow signals.

Signal:
  I_i = (Q_bid - Q_ask) / (Q_bid + Q_ask + eps)    ∈ [-1, 1]
  OFI_i = ΔQ_bid - ΔQ_ask   (order flow imbalance)
  E[r_i] = a * OFI_i + b * I_i

Anti-frais (strict):
  Trade ONLY if |E[r_i]| > ROUND_TRIP_BPS + BUFFER_BPS
  This is the harshest filter — microstructure signals are very noisy.

Note: coefficient a, b are empirically calibrated defaults; no fitting is done here
(walk-forward fitting out of scope for paper trading).

OFF by default — requires l2Book WS subscription per coin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from .base_signal_agent import AgentContext, AgentOutput, BaseSignalAgent

# Strict edge filter for microstructure (bps threshold as fraction)
_ROUND_TRIP_BPS = 27.0
_BUFFER_BPS = 3.0
_OFI_EDGE_THRESHOLD = (_ROUND_TRIP_BPS + _BUFFER_BPS) * 1e-4  # fraction of price move needed


@dataclass(frozen=True)
class OBImbalanceConfig:
    # OFI model coefficients (empirical defaults — calibrate from data)
    a_ofi: float = 0.4            # OFI coefficient
    b_imbalance: float = 0.3      # Imbalance coefficient
    n_levels: int = 5             # number of OB levels to aggregate
    # Signal parameters
    z_max: float = 2.0
    vol_ewma_lam: float = 0.94
    min_vol_floor: float = 1e-6
    min_position_size: float = 0.005   # minimum weight per symbol
    max_position_size: float = 0.15    # maximum weight per symbol
    # Edge filter (as fraction of mid price, not bps)
    edge_threshold: float = _OFI_EDGE_THRESHOLD


class OrderFlowImbalanceAgent(BaseSignalAgent):
    """
    Microstructure order-flow imbalance strategy.

    Uses order book snapshots to predict short-horizon price direction.
    Trades only when predicted edge > round-trip cost + buffer.

    OFF by default — activating this strategy automatically subscribes
    to l2Book feeds per coin in BackgroundEngine.
    """

    name = "OB Imbalance"
    warmup_bars = 6
    enabled_by_default = False   # OFF: requires l2Book stream

    def __init__(self, cfg: Optional[OBImbalanceConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or OBImbalanceConfig()
        # Previous OB state for OFI delta
        self._prev_bid_qty: Dict[str, float] = {}
        self._prev_ask_qty: Dict[str, float] = {}
        # EWMA vol
        self._ewma_var: Dict[str, float] = {}
        self._prev_px: Dict[str, float] = {}
        self._last_weights: Dict[str, float] = {}

    def reset(self) -> None:
        self._bars_seen = 0
        self._prev_bid_qty.clear()
        self._prev_ask_qty.clear()
        self._ewma_var.clear()
        self._prev_px.clear()
        self._last_weights.clear()

    def update(
        self,
        ts: datetime,
        mids: Dict[str, float],
        context: AgentContext,
    ) -> AgentOutput:
        cfg = self.cfg
        ob = context.ob_snapshots

        # ── Update vol & parse OB ─────────────────────────────────────────────
        imbalance: Dict[str, float] = {}
        ofi: Dict[str, float] = {}
        predicted_ret: Dict[str, float] = {}

        for sym in context.selected:
            px = mids.get(sym, 0.0)
            if math.isfinite(px) and px > 0:
                if sym in self._prev_px and self._prev_px[sym] > 0:
                    r = self._log_return(self._prev_px[sym], px)
                    prev_var = self._ewma_var.get(sym, r * r)
                    self._ewma_var[sym] = self._ewma_update(prev_var, r * r, cfg.vol_ewma_lam)
                self._prev_px[sym] = px

            # Parse order book snapshot
            snap = ob.get(sym)
            if not snap:
                continue

            # HL l2Book format: {"coin": sym, "levels": [[bids, asks], ...]}
            # Each level: [price_str, qty_str]
            levels = snap.get("levels", [[], []])
            if len(levels) < 2:
                continue

            bid_levels = levels[0][:cfg.n_levels] if levels[0] else []
            ask_levels = levels[1][:cfg.n_levels] if levels[1] else []

            q_bid = sum(float(lv[1]) for lv in bid_levels if len(lv) >= 2)
            q_ask = sum(float(lv[1]) for lv in ask_levels if len(lv) >= 2)

            total_qty = q_bid + q_ask
            if total_qty < 1e-12:
                continue

            # Imbalance ∈ [-1, 1]
            I = (q_bid - q_ask) / (total_qty + 1e-12)
            imbalance[sym] = I

            # OFI = delta(Q_bid) - delta(Q_ask)
            prev_bid = self._prev_bid_qty.get(sym, q_bid)
            prev_ask = self._prev_ask_qty.get(sym, q_ask)
            ofi_raw = (q_bid - prev_bid) - (q_ask - prev_ask)
            # Normalize OFI by total qty
            ofi_norm = ofi_raw / (total_qty + 1e-12)
            ofi[sym] = ofi_norm

            self._prev_bid_qty[sym] = q_bid
            self._prev_ask_qty[sym] = q_ask

            # Predicted return (linear model)
            E_r = cfg.a_ofi * ofi_norm + cfg.b_imbalance * I
            predicted_ret[sym] = E_r

        self._bars_seen += 1

        if not self.is_warmed_up or not predicted_ret:
            return AgentOutput(zscores=imbalance)

        # ── Edge filter: only trade if |E[r]| > threshold ────────────────────
        weights: Dict[str, float] = {}
        for sym, E_r in predicted_ret.items():
            if abs(E_r) < cfg.edge_threshold:
                continue  # expected move doesn't cover round-trip cost

            vol = math.sqrt(max(self._ewma_var.get(sym, cfg.min_vol_floor), cfg.min_vol_floor))
            # Weight proportional to predicted return, scaled by vol
            w = float(np.clip(E_r / (vol + 1e-12), -cfg.z_max, cfg.z_max))
            if abs(w) > 1e-12:
                weights[sym] = w

        if not weights:
            return AgentOutput(zscores=imbalance, meta={"reason": "edge_filter_blocked_all"})

        # Normalize + cap
        total = sum(abs(v) for v in weights.values())
        if total < 1e-12:
            return AgentOutput(zscores=imbalance)

        norm_w = {
            s: float(np.clip(v / total, -cfg.max_position_size, cfg.max_position_size))
            for s, v in weights.items()
        }

        self._last_weights = dict(norm_w)

        return AgentOutput(
            weights=norm_w,
            zscores=imbalance,
            meta={
                "n_with_ob": len(predicted_ret),
                "n_passed_edge_filter": len(weights),
            },
        )
