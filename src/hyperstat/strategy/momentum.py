# src/hyperstat/strategy/momentum.py
"""
Cross-Sectional Momentum Agent (SignalAgent 2)
================================================
Signal : m_i = rank_B(R^H1_i) - rank_B(R^H2_i)
         — long recent winners, short recent losers within each bucket

Anti-frais :
  - Rebalance threshold  : |delta_w| > DELTA_W_MIN (0.02)
  - Min rebalance interval: 30 min (avoids micro-churn on short TFs)
  - Low vol-scaling
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .base_signal_agent import AgentContext, AgentOutput, BaseSignalAgent


@dataclass(frozen=True)
class MomentumConfig:
    horizon_fast: int = 4        # H2 — short lookback (bars)
    horizon_slow: int = 12       # H1 — long lookback (bars)
    m_max: float = 1.0           # clip momentum score
    vol_ewma_lam: float = 0.94   # EWMA lambda for vol estimate
    min_vol_floor: float = 1e-6
    delta_w_min: float = 0.02    # no-trade zone: ignore deltas below this
    min_rebal_seconds: float = 1800.0  # 30 min between rebalances (anti-churn)


class CrossSectionalMomentumAgent(BaseSignalAgent):
    """
    Bucket-neutral cross-sectional momentum:
      m_i = rank_B(R^slow_i) - rank_B(R^fast_i)   ∈ [-1, 1]
      w_i = clip(m_i, ±m_max) / ewma_vol_i

    Uses existing price deques — no separate data fetch.
    Low-churn: min 30min between rebalances.
    """

    name = "Cross-Section Momentum"
    warmup_bars = 24
    enabled_by_default = True

    def __init__(self, cfg: Optional[MomentumConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MomentumConfig()

        # Rolling price history: deque of (ts, price)
        self._price_hist: Dict[str, Deque[Tuple[datetime, float]]] = {}
        # EWMA variance per symbol
        self._ewma_var: Dict[str, float] = {}
        # Last weights (to compute delta for no-trade zone)
        self._last_weights: Dict[str, float] = {}
        # Last rebalance timestamp
        self._last_rebal_ts: Optional[datetime] = None

    def reset(self) -> None:
        self._bars_seen = 0
        self._price_hist.clear()
        self._ewma_var.clear()
        self._last_weights.clear()
        self._last_rebal_ts = None

    def update(
        self,
        ts: datetime,
        mids: Dict[str, float],
        context: AgentContext,
    ) -> AgentOutput:
        cfg = self.cfg
        max_hist = cfg.horizon_slow + 2

        # ── Update price history & vol ────────────────────────────────────────
        for sym, px in mids.items():
            if not (math.isfinite(px) and px > 0):
                continue
            if sym not in self._price_hist:
                self._price_hist[sym] = deque(maxlen=max_hist)
            q = self._price_hist[sym]
            # Compute log-return for EWMA vol update
            if q:
                r = self._log_return(q[-1][1], px)
                prev_var = self._ewma_var.get(sym, r * r)
                self._ewma_var[sym] = self._ewma_update(prev_var, r * r, cfg.vol_ewma_lam)
            q.append((ts, px))

        self._bars_seen += 1

        if not self.is_warmed_up:
            return AgentOutput()

        # ── Anti-churn: respect min rebalance interval ────────────────────────
        if self._last_rebal_ts is not None:
            elapsed = (ts - self._last_rebal_ts).total_seconds()
            if elapsed < cfg.min_rebal_seconds:
                # Return last weights unchanged (hold)
                return AgentOutput(
                    weights=dict(self._last_weights),
                    meta={"reason": "throttled", "elapsed_s": elapsed},
                )

        # ── Compute momentum signals per bucket ───────────────────────────────
        raw_signals: Dict[str, float] = {}

        for bucket_syms in context.buckets.values():
            bucket_signals: Dict[str, float] = {}

            for sym in bucket_syms:
                if sym not in self._price_hist:
                    continue
                hist = list(self._price_hist[sym])
                prices = np.array([p for _, p in hist])
                if len(prices) < cfg.horizon_slow + 1:
                    continue

                # Log returns over slow horizon
                r_slow = math.log(prices[-1] / prices[-(cfg.horizon_slow + 1)])
                # Log returns over fast horizon
                r_fast = math.log(prices[-1] / prices[-(cfg.horizon_fast + 1)])

                bucket_signals[sym] = (r_slow, r_fast)

            if len(bucket_signals) < 2:
                continue

            # Cross-sectional ranks within bucket
            slow_vals = {s: v[0] for s, v in bucket_signals.items()}
            fast_vals = {s: v[1] for s, v in bucket_signals.items()}
            syms_b = list(bucket_signals.keys())

            slow_arr = np.array([slow_vals[s] for s in syms_b])
            fast_arr = np.array([fast_vals[s] for s in syms_b])

            # Percentile ranks ∈ [0, 1]
            from scipy.stats import rankdata
            slow_ranks = rankdata(slow_arr) / len(slow_arr)
            fast_ranks = rankdata(fast_arr) / len(fast_arr)

            for i, sym in enumerate(syms_b):
                # m = rank_slow - rank_fast: positive = improving momentum
                m = float(np.clip(slow_ranks[i] - fast_ranks[i], -cfg.m_max, cfg.m_max))
                vol = math.sqrt(max(self._ewma_var.get(sym, cfg.min_vol_floor), cfg.min_vol_floor))
                raw_signals[sym] = m / (vol + 1e-12)

        if not raw_signals:
            return AgentOutput()

        # ── Apply no-trade zone ───────────────────────────────────────────────
        # Normalize raw_signals to unit gross first for delta comparison
        total = sum(abs(v) for v in raw_signals.values())
        if total < 1e-12:
            return AgentOutput()
        norm_signals = {s: v / total for s, v in raw_signals.items()}

        filtered: Dict[str, float] = {}
        for sym, tw in norm_signals.items():
            cw = self._last_weights.get(sym, 0.0)
            if abs(tw - cw) >= cfg.delta_w_min:
                filtered[sym] = tw
            else:
                filtered[sym] = cw   # hold current weight

        self._last_weights = dict(filtered)
        self._last_rebal_ts = ts

        # Z-scores = cross-sectional (re-rank for display)
        zscores = self._zscore_cross(raw_signals)

        return AgentOutput(
            weights=filtered,
            zscores=zscores,
            meta={"n_signals": len(raw_signals)},
        )
