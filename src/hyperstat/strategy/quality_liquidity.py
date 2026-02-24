# src/hyperstat/strategy/quality_liquidity.py
"""
Quality / Liquidity Premium Agent (SignalAgent 7)
==================================================
Long high-quality (high DV, low illiquid, stable funding) coins.
Short low-quality coins. Very low turnover — rebalances 1-2× per day.

Score:
  q_i = α·rank(DV_i) + β·(1 - rank(ILLIQ_i)) - γ·rank(funding_instability_i)

Anti-frais: max 2 rebalances per day — this strategy EARNS the fee budget
            by holding stable positions with carry and liquidity premium.
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
class QualityLiquidityConfig:
    # Score weights (must sum to ≤ 1.0, relative)
    w_dv: float = 0.40            # dollar volume rank weight
    w_illiq: float = 0.35         # illiquidity rank weight (inverted)
    w_fund_instab: float = 0.25   # funding instability weight (inverted)

    # Rolling windows
    dv_window: int = 20           # bars for rolling DV
    illiq_window: int = 20        # bars for Amihud illiq
    funding_window: int = 24      # bars for funding std

    # Signal clip
    z_max: float = 2.0
    vol_ewma_lam: float = 0.94
    min_vol_floor: float = 1e-6
    min_rebal_seconds: float = 43200.0  # 12h between rebalances


class QualityLiquidityAgent(BaseSignalAgent):
    """
    Quality/Liquidity premium strategy.
    Long: high volume, low illiquidity, stable funding.
    Short: low volume, high illiquidity, erratic funding.

    Very low turnover by design — provides diversification vs MR strategies.
    """

    name = "Quality / Liquidity"
    warmup_bars = 20
    enabled_by_default = True

    def __init__(self, cfg: Optional[QualityLiquidityConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or QualityLiquidityConfig()
        # Rolling price history for volume proxy (|r| * px as Amihud proxy)
        self._price_hist: Dict[str, Deque[float]] = {}   # recent prices
        self._ret_hist: Dict[str, Deque[float]] = {}     # recent |returns|
        self._fund_hist: Dict[str, Deque[float]] = {}    # recent funding rates
        self._ewma_var: Dict[str, float] = {}
        self._prev_px: Dict[str, float] = {}
        self._last_rebal_ts: Optional[datetime] = None
        self._last_weights: Dict[str, float] = {}

    def reset(self) -> None:
        self._bars_seen = 0
        self._price_hist.clear()
        self._ret_hist.clear()
        self._fund_hist.clear()
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
        funding = context.funding_rates

        # ── Update histories ──────────────────────────────────────────────────
        for sym in context.selected:
            px = mids.get(sym, 0.0)
            f = funding.get(sym, 0.0)

            if math.isfinite(px) and px > 0:
                if sym not in self._price_hist:
                    self._price_hist[sym] = deque(maxlen=cfg.dv_window)
                    self._ret_hist[sym] = deque(maxlen=cfg.illiq_window)
                self._price_hist[sym].append(px)

                if sym in self._prev_px and self._prev_px[sym] > 0:
                    r = self._log_return(self._prev_px[sym], px)
                    self._ret_hist[sym].append(abs(r))
                    prev_var = self._ewma_var.get(sym, r * r)
                    self._ewma_var[sym] = self._ewma_update(prev_var, r * r, cfg.vol_ewma_lam)
                self._prev_px[sym] = px

            if sym not in self._fund_hist:
                self._fund_hist[sym] = deque(maxlen=cfg.funding_window)
            self._fund_hist[sym].append(f)

        self._bars_seen += 1

        if not self.is_warmed_up:
            return AgentOutput()

        # ── Anti-churn ────────────────────────────────────────────────────────
        if self._last_rebal_ts is not None:
            elapsed = (ts - self._last_rebal_ts).total_seconds()
            if elapsed < cfg.min_rebal_seconds:
                return AgentOutput(
                    weights=dict(self._last_weights),
                    meta={"reason": "throttled"},
                )

        # ── Compute quality scores ────────────────────────────────────────────
        syms_with_data = [
            s for s in context.selected
            if (s in self._price_hist and len(self._price_hist[s]) >= cfg.dv_window // 2
                and s in self._ret_hist and len(self._ret_hist[s]) >= 2)
        ]

        if len(syms_with_data) < 4:
            return AgentOutput()

        # DV proxy: average price * avg |return| * window (rough proxy for turnover)
        dv_scores: Dict[str, float] = {}
        illiq_scores: Dict[str, float] = {}  # lower is better (Amihud-like)
        fund_instab: Dict[str, float] = {}   # lower is better

        for sym in syms_with_data:
            prices = list(self._price_hist[sym])
            rets = list(self._ret_hist[sym])
            f_hist = list(self._fund_hist.get(sym, []))

            avg_px = float(np.mean(prices)) if prices else 0.0
            avg_abs_ret = float(np.mean(rets)) if rets else 0.0

            # DV proxy: avg_px * (1 / avg_abs_ret) — higher price stability = more DV-like
            # More practical: just use average price as DV proxy (higher-priced coins tend to be larger)
            dv_scores[sym] = avg_px

            # Illiquidity proxy: avg(|r| / price) — Amihud-like (lower = more liquid)
            illiq_scores[sym] = avg_abs_ret / (avg_px + 1e-12) if avg_px > 0 else 1.0

            # Funding instability: std of recent funding rates
            fund_instab[sym] = float(np.std(f_hist)) if len(f_hist) >= 2 else 0.0

        # Cross-sectional ranks ∈ [0, 1]
        syms = list(dv_scores.keys())
        n = len(syms)
        if n < 2:
            return AgentOutput()

        def rank_pct(d: Dict[str, float], ascending: bool = True) -> Dict[str, float]:
            vals = [d[s] for s in syms]
            order = np.argsort(vals)
            ranks = np.empty(n)
            for rank_i, idx in enumerate(order):
                ranks[idx] = rank_i / max(n - 1, 1)
            if not ascending:
                ranks = 1.0 - ranks
            return {syms[i]: float(ranks[i]) for i in range(n)}

        dv_rank      = rank_pct(dv_scores, ascending=True)      # high DV = high rank
        illiq_rank   = rank_pct(illiq_scores, ascending=False)  # low illiq = high rank (inverted)
        fund_rank    = rank_pct(fund_instab, ascending=False)   # low instab = high rank (inverted)

        # Quality score: weighted combination
        quality: Dict[str, float] = {}
        for sym in syms:
            quality[sym] = (cfg.w_dv * dv_rank[sym]
                            + cfg.w_illiq * illiq_rank[sym]
                            + cfg.w_fund_instab * fund_rank[sym])

        # Cross-sectional z-score of quality
        z_quality = self._zscore_cross(quality)

        # Vol-scaled weights
        raw_w: Dict[str, float] = {}
        for sym, z in z_quality.items():
            vol = math.sqrt(max(self._ewma_var.get(sym, cfg.min_vol_floor), cfg.min_vol_floor))
            signal = float(np.clip(z, -cfg.z_max, cfg.z_max))
            raw_w[sym] = signal / (vol + 1e-12)

        # Normalize
        total = sum(abs(v) for v in raw_w.values())
        if total < 1e-12:
            return AgentOutput()
        norm_w = {s: v / total for s, v in raw_w.items()}

        self._last_weights = dict(norm_w)
        self._last_rebal_ts = ts

        return AgentOutput(
            weights=norm_w,
            zscores=z_quality,
            meta={"n_scored": len(syms)},
        )
