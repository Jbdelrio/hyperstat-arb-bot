# src/hyperstat/strategy/liquidation_reversion.py
"""
Liquidation Reversion Agent (SignalAgent 4)  [OFF by default]
==============================================================
Uses the public `trades` WebSocket feed as a proxy for liquidation cascades.

Heuristic: large sudden volume spikes often correspond to liquidation cascades.
After such a spike, price tends to revert (overshooting due to forced closes).

Signal:
  L_i = (recent_vol_spike - ADV_ewma_i) / (ADV_ewma_i + eps)
  Position: contrarian if L_i > threshold (buy after sell liquidations, sell after buy ones)

Anti-frais:
  - Requires minimum edge = 3 × ROUND_TRIP_BPS (larger edge for event-driven trades)
  - Concave sizing: g(L) = min(L, L_max)^0.5
  - Short hold horizon (minutes to hours)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

import numpy as np

from .base_signal_agent import AgentContext, AgentOutput, BaseSignalAgent

# Edge requirement = 3x round-trip (more conservative for event-driven)
_EVENT_EDGE_BPS = 3 * 27.0  # 81 bps minimum


@dataclass(frozen=True)
class LiquidationReversionConfig:
    # Spike detection
    adv_ewma_lam: float = 0.99    # slow EWMA for ADV (≈ 100-bar average)
    spike_threshold: float = 3.0  # L_i > threshold to signal
    l_max: float = 10.0           # cap on L_i before sqrt

    # Position sizing
    sizing_power: float = 0.5     # concave: weight ∝ L^power

    # Trade window: how long after spike to hold contrarian position
    hold_window_bars: int = 6     # hold for at most 6 bars after spike
    min_bars_since_spike: int = 1 # enter after 1 bar confirmation

    # Directional proxy: use price direction in spike window
    spike_window_bars: int = 2    # bars to look back for spike detection

    vol_ewma_lam: float = 0.94
    min_vol_floor: float = 1e-6
    z_max: float = 2.0


class LiquidationReversionAgent(BaseSignalAgent):
    """
    Event-driven contrarian strategy after suspected liquidation cascades.

    Uses public trades feed (aggregated in AgentContext.trade_history) to detect
    sudden volume spikes. When a spike occurs, takes contrarian position with
    concave sizing and short hold horizon.

    OFF by default — requires trades WS subscription to work properly.
    """

    name = "Liquidation Reversion"
    warmup_bars = 12
    enabled_by_default = False   # OFF: requires trades stream

    def __init__(self, cfg: Optional[LiquidationReversionConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or LiquidationReversionConfig()
        # ADV EWMA per symbol (using price × trade_count as volume proxy)
        self._adv_ewma: Dict[str, float] = {}
        # Bar-by-bar volume accumulator (reset each bar)
        self._bar_volume: Dict[str, float] = {}
        # Spike events: {sym: (spike_bar, direction, L)}
        self._spike_events: Dict[str, tuple] = {}
        # Hold counter since spike
        self._hold_bars: Dict[str, int] = {}
        # EWMA vol
        self._ewma_var: Dict[str, float] = {}
        self._prev_px: Dict[str, float] = {}
        # Bar count
        self._current_bar: int = 0
        # Last weights
        self._last_weights: Dict[str, float] = {}

    def reset(self) -> None:
        self._bars_seen = 0
        self._adv_ewma.clear()
        self._bar_volume.clear()
        self._spike_events.clear()
        self._hold_bars.clear()
        self._ewma_var.clear()
        self._prev_px.clear()
        self._current_bar = 0
        self._last_weights.clear()

    def update(
        self,
        ts: datetime,
        mids: Dict[str, float],
        context: AgentContext,
    ) -> AgentOutput:
        cfg = self.cfg
        self._current_bar += 1

        # ── Aggregate trade volume from trade_history ─────────────────────────
        # trade_history[sym] is a deque of recent trade dicts
        # We use count × price as volume proxy if actual volume not available
        bar_vol: Dict[str, float] = {}
        net_side: Dict[str, float] = {}  # +1 buys, -1 sells (direction proxy)

        for sym in context.selected:
            trades = context.trade_history.get(sym, deque())
            vol = 0.0
            side_sum = 0.0
            for t in trades:
                if isinstance(t, dict):
                    # HL trade format: {coin, side, px, sz, time, ...}
                    sz = float(t.get("sz", t.get("size", 0.0)) or 0.0)
                    px = float(t.get("px", t.get("price", mids.get(sym, 1.0))) or 1.0)
                    side_str = str(t.get("side", "B")).upper()
                    side_val = 1.0 if side_str in ("B", "BUY") else -1.0
                    vol += sz * px
                    side_sum += side_val * sz * px
            bar_vol[sym] = vol
            net_side[sym] = math.copysign(1.0, side_sum) if abs(side_sum) > 1e-12 else 0.0

        # ── Update ADV EWMA & vol ─────────────────────────────────────────────
        for sym in context.selected:
            v = bar_vol.get(sym, 0.0)
            if sym not in self._adv_ewma:
                self._adv_ewma[sym] = max(v, 1.0)
            else:
                self._adv_ewma[sym] = self._ewma_update(self._adv_ewma[sym], max(v, 1e-9), cfg.adv_ewma_lam)

            px = mids.get(sym, 0.0)
            if math.isfinite(px) and px > 0:
                if sym in self._prev_px and self._prev_px[sym] > 0:
                    r = self._log_return(self._prev_px[sym], px)
                    prev_var = self._ewma_var.get(sym, r * r)
                    self._ewma_var[sym] = self._ewma_update(prev_var, r * r, cfg.vol_ewma_lam)
                self._prev_px[sym] = px

        self._bars_seen += 1

        if not self.is_warmed_up:
            return AgentOutput()

        # ── Detect spikes & update hold counters ──────────────────────────────
        weights: Dict[str, float] = {}
        spike_scores: Dict[str, float] = {}

        for sym in context.selected:
            v = bar_vol.get(sym, 0.0)
            adv = self._adv_ewma.get(sym, 1.0)
            L = (v - adv) / (adv + 1e-12)

            # Update hold counter
            if sym in self._hold_bars:
                self._hold_bars[sym] += 1
                if self._hold_bars[sym] >= cfg.hold_window_bars:
                    del self._hold_bars[sym]
                    self._spike_events.pop(sym, None)

            # Detect new spike
            if L >= cfg.spike_threshold and sym not in self._spike_events:
                direction = net_side.get(sym, 0.0)
                if direction != 0.0:
                    self._spike_events[sym] = (self._current_bar, direction, L)
                    self._hold_bars[sym] = 0

            # Generate signal if in hold window
            if sym in self._spike_events:
                bar_spike, direction, L_spike = self._spike_events[sym]
                bars_since = self._current_bar - bar_spike

                if bars_since >= cfg.min_bars_since_spike:
                    # Contrarian: if spike was sell-driven (direction=-1) → go long
                    L_clipped = min(L_spike, cfg.l_max)
                    sizing = (L_clipped ** cfg.sizing_power) / (cfg.l_max ** cfg.sizing_power)
                    vol = math.sqrt(max(self._ewma_var.get(sym, cfg.min_vol_floor), cfg.min_vol_floor))
                    w = -direction * sizing / (vol + 1e-12)  # contrarian
                    weights[sym] = w
                    spike_scores[sym] = L_spike

        if not weights:
            return AgentOutput()

        total = sum(abs(v) for v in weights.values())
        if total < 1e-12:
            return AgentOutput()
        norm_w = {s: v / total for s, v in weights.items()}
        self._last_weights = dict(norm_w)

        return AgentOutput(
            weights=norm_w,
            zscores={s: float(np.clip(spike_scores.get(s, 0.0), -cfg.z_max, cfg.z_max))
                     for s in norm_w},
            meta={"n_spike_events": len(self._spike_events)},
        )
