# src/hyperstat/strategy/funding_carry_pure.py
"""
Funding Carry Pure Agent (SignalAgent 3)
=========================================
Signal : c_i = -zscore_cross(EWMA_slow(f_i))
         — short high-funding (market pays you), long low-funding

Break-even check: expected carry over hold horizon > round-trip cost
Anti-frais: max 2 rebalances per day (low churn), stability gate on funding std

Math:
  f_slow_i = EWMA(f_i, span=72)   (slow smooth over ~3 days of 1h bars)
  c_i = -zscore_mad_cross(f_slow_i)   (negative → sell what market overpays for)
  stability gate: only trade if rolling_std(f_i) < theta_f
  carry break-even: |f_avg_bps| * horizon_h > round_trip_bps

Funding on Hyperliquid: paid every 1h at 1/8 of the 8h rate
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Optional

import numpy as np

from .base_signal_agent import AgentContext, AgentOutput, BaseSignalAgent


@dataclass(frozen=True)
class FundingCarryConfig:
    ewma_slow_span: int = 72          # EWMA span for slow funding (bars)
    ewma_fast_span: int = 8           # EWMA span for fast funding (for volatility)
    stability_window: int = 24        # bars for funding std gate
    stability_threshold: float = 5e-4 # max funding std (raw rate) to qualify as "stable"
    hold_horizon_hours: float = 12.0  # expected hold horizon (hours)
    round_trip_bps: float = 27.0      # 2*(3.5 fee + 10 slip)
    buffer_bps: float = 3.0
    vol_ewma_lam: float = 0.94
    min_vol_floor: float = 1e-6
    z_max: float = 2.0                # clip carry z-score
    min_rebal_seconds: float = 43200.0  # 12h between rebalances (low churn)


class FundingCarryPureAgent(BaseSignalAgent):
    """
    Pure funding carry strategy.
    Short coins with persistently high positive funding (market is paying for longs).
    Long coins with persistently negative funding (market is paying for shorts).

    Very low turnover — hold 12h minimum, only rebalance when funding shifts.
    """

    name = "Funding Carry Pure"
    warmup_bars = 48
    enabled_by_default = True

    def __init__(self, cfg: Optional[FundingCarryConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or FundingCarryConfig()
        # EWMA states for funding
        self._ewma_slow: Dict[str, float] = {}   # slow EWMA of funding rate
        self._ewma_fast: Dict[str, float] = {}   # fast EWMA (for variance)
        # Funding history for std gate
        self._f_hist: Dict[str, Deque[float]] = {}
        # EWMA vol of price for vol-scaling
        self._ewma_var: Dict[str, float] = {}
        self._prev_px: Dict[str, float] = {}
        # Last rebalance
        self._last_rebal_ts: Optional[datetime] = None
        self._last_weights: Dict[str, float] = {}

    def reset(self) -> None:
        self._bars_seen = 0
        self._ewma_slow.clear()
        self._ewma_fast.clear()
        self._f_hist.clear()
        self._ewma_var.clear()
        self._prev_px.clear()
        self._last_rebal_ts = None
        self._last_weights.clear()

    def update(
        self,
        ts: datetime,
        mids: Dict[str, float],
        context: AgentContext,
    ) -> AgentOutput:
        cfg = self.cfg
        lam_slow = 1.0 - 2.0 / (cfg.ewma_slow_span + 1)
        lam_fast = 1.0 - 2.0 / (cfg.ewma_fast_span + 1)
        funding = context.funding_rates

        # ── Update EWMA states ────────────────────────────────────────────────
        for sym in context.selected:
            f = funding.get(sym, 0.0)

            # Funding EWMA updates
            if sym not in self._ewma_slow:
                self._ewma_slow[sym] = f
                self._ewma_fast[sym] = f
            else:
                self._ewma_slow[sym] = self._ewma_update(self._ewma_slow[sym], f, lam_slow)
                self._ewma_fast[sym] = self._ewma_update(self._ewma_fast[sym], f, lam_fast)

            # Funding history for stability gate
            if sym not in self._f_hist:
                self._f_hist[sym] = deque(maxlen=cfg.stability_window)
            self._f_hist[sym].append(f)

            # Price EWMA vol
            px = mids.get(sym, 0.0)
            if math.isfinite(px) and px > 0:
                if sym in self._prev_px:
                    r = self._log_return(self._prev_px[sym], px)
                    prev_var = self._ewma_var.get(sym, r * r)
                    self._ewma_var[sym] = self._ewma_update(prev_var, r * r, cfg.vol_ewma_lam)
                self._prev_px[sym] = px

        self._bars_seen += 1

        if not self.is_warmed_up:
            return AgentOutput()

        # ── Anti-churn: respect min rebalance interval ────────────────────────
        if self._last_rebal_ts is not None:
            elapsed = (ts - self._last_rebal_ts).total_seconds()
            if elapsed < cfg.min_rebal_seconds:
                return AgentOutput(
                    weights=dict(self._last_weights),
                    meta={"reason": "throttled", "elapsed_s": elapsed},
                )

        # ── Build carry signal ────────────────────────────────────────────────
        carry_scores: Dict[str, float] = {}
        stable_syms: Dict[str, float] = {}

        for sym in context.selected:
            f_hist = list(self._f_hist.get(sym, []))
            if len(f_hist) < cfg.stability_window // 2:
                continue

            # Stability gate: skip unstable funding
            f_std = float(np.std(f_hist))
            if f_std > cfg.stability_threshold:
                continue

            f_slow = self._ewma_slow.get(sym, 0.0)

            # Break-even check: expected carry over hold horizon > round-trip cost
            # f_slow in raw rate (e.g. 0.0001 = 1 bp per 8h = ~0.125 bps/h on HL)
            # carry per hour (HL pays 1/8 of 8h rate every hour)
            carry_per_h_bps = abs(f_slow) * 1e4 / 8.0
            expected_carry_bps = carry_per_h_bps * cfg.hold_horizon_hours
            if expected_carry_bps < cfg.round_trip_bps + cfg.buffer_bps:
                continue  # not worth trading

            carry_scores[sym] = f_slow
            stable_syms[sym] = f_slow

        if len(carry_scores) < 2:
            return AgentOutput(weights={}, meta={"reason": "insufficient_stable_symbols"})

        # Cross-sectional z-score of slow EWMA funding
        z_carry = self._zscore_cross(carry_scores)

        # Signal: -z_carry (short overpriced funding, long underpriced)
        raw_w: Dict[str, float] = {}
        for sym, z in z_carry.items():
            vol = math.sqrt(max(self._ewma_var.get(sym, cfg.min_vol_floor), cfg.min_vol_floor))
            signal = -float(np.clip(z, -cfg.z_max, cfg.z_max))
            raw_w[sym] = signal / (vol + 1e-12)

        if not raw_w:
            return AgentOutput()

        # Normalize to unit gross
        total = sum(abs(v) for v in raw_w.values())
        if total < 1e-12:
            return AgentOutput()
        norm_w = {s: v / total for s, v in raw_w.items()}

        self._last_weights = dict(norm_w)
        self._last_rebal_ts = ts

        return AgentOutput(
            weights=norm_w,
            zscores={sym: -z_carry.get(sym, 0.0) for sym in norm_w},
            meta={
                "n_stable_symbols": len(stable_syms),
                "avg_funding_bps": float(np.mean([abs(v) * 1e4 for v in stable_syms.values()])),
            },
        )
