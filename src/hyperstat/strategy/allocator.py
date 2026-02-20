# src/hyperstat/strategy/allocator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from hyperstat.core.math import neutralize_weights, normalize_to_gross
from hyperstat.core.risk import apply_bucket_caps, apply_weight_caps, emergency_flatten_by_z
from hyperstat.core.types import PortfolioWeights, RegimeScore, Signal
from .funding_overlay import FundingOverlayModel


@dataclass(frozen=True)
class AllocatorConfig:
    """
    Combines:
      - stat-arb raw signal
      - regime gating Q_t
      - optional funding overlay
    and enforces:
      - vol scaling
      - neutralization
      - gross targets
      - per-coin & per-bucket caps
      - emergency flatten by z
    """
    gross_target_stat: float = 1.20
    gross_target_fund: float = 0.20  # max overlay gross BEFORE eta*q_fund (already in overlay model)
    max_weight_per_coin: float = 0.12
    max_weight_per_bucket: float = 0.35

    dollar_neutral: bool = True
    beta_neutral: bool = True

    z_emergency_flat: float = 3.5
    eps: float = 1e-12


class PortfolioAllocator:
    def __init__(self, cfg: AllocatorConfig, funding_overlay: Optional[FundingOverlayModel] = None) -> None:
        self.cfg = cfg
        self.funding_overlay = funding_overlay

    def allocate(
        self,
        ts,
        signal: Signal,
        regime: RegimeScore,
        buckets: Dict[str, List[str]],
        features: Dict[str, Dict[str, float]],
        betas: Optional[Dict[str, float]] = None,
    ) -> PortfolioWeights:
        """
        features expected keys per symbol (best-effort):
          - ewma_vol OR vol (used for scaling)
        """
        symbols = list(signal.weights_raw.keys())

        # --- 1) vol scaling
        w = {}
        for s in symbols:
            raw = float(signal.weights_raw.get(s, 0.0))
            vol = features.get(s, {}).get("ewma_vol")
            if vol is None or not np.isfinite(vol) or float(vol) <= self.cfg.eps:
                # if missing vol, keep raw (conservative: could also set 0)
                w[s] = raw
            else:
                w[s] = float(raw / float(vol))

        # --- 2) regime scaling
        q = float(regime.q_total)
        w = {s: float(q * v) for s, v in w.items()}

        # --- 3) neutralize + normalize stat layer gross
        w = neutralize_weights(w, betas=betas, dollar_neutral=self.cfg.dollar_neutral, beta_neutral=self.cfg.beta_neutral)
        w = normalize_to_gross(w, gross_target=float(self.cfg.gross_target_stat))

        # --- 4) apply caps (coin then bucket) + re-normalize stat layer
        w = apply_weight_caps(w, max_weight_per_coin=float(self.cfg.max_weight_per_coin), gross_target=None)
        w = apply_bucket_caps(w, buckets=buckets, max_weight_per_bucket=float(self.cfg.max_weight_per_bucket))
        w = neutralize_weights(w, betas=betas, dollar_neutral=self.cfg.dollar_neutral, beta_neutral=self.cfg.beta_neutral)
        w = normalize_to_gross(w, gross_target=float(self.cfg.gross_target_stat))

        # --- 5) emergency flatten (per-coin)
        w = emergency_flatten_by_z(w, signal.zscores, z_emergency_flat=float(self.cfg.z_emergency_flat))
        w = neutralize_weights(w, betas=betas, dollar_neutral=self.cfg.dollar_neutral, beta_neutral=self.cfg.beta_neutral)
        w = normalize_to_gross(w, gross_target=float(self.cfg.gross_target_stat))

        # --- 6) funding overlay
        w_fund: Dict[str, float] = {s: 0.0 for s in symbols}
        q_fund: float = 0.0

        if self.funding_overlay is not None:
            w_fund, q_fund = self.funding_overlay.compute_overlay(
                symbols=symbols,
                betas=betas,
                dollar_neutral=self.cfg.dollar_neutral,
                beta_neutral=self.cfg.beta_neutral,
            )

        # combine layers
        w_total = dict(w)
        for s in symbols:
            w_total[s] = float(w_total.get(s, 0.0) + w_fund.get(s, 0.0))

        # --- 7) final constraint enforcement + cap gross if needed
        w_total = neutralize_weights(
            w_total, betas=betas, dollar_neutral=self.cfg.dollar_neutral, beta_neutral=self.cfg.beta_neutral
        )

        # gross budget: stat gross target + overlay gross target (upper bound)
        gross_cap = float(self.cfg.gross_target_stat + self.cfg.gross_target_fund)
        gross_now = float(sum(abs(v) for v in w_total.values()))
        if gross_now > self.cfg.eps and gross_now > gross_cap:
            scale = gross_cap / gross_now
            w_total = {s: float(scale * v) for s, v in w_total.items()}

        # re-apply caps after overlay (important)
        w_total = apply_weight_caps(w_total, max_weight_per_coin=float(self.cfg.max_weight_per_coin), gross_target=None)
        w_total = apply_bucket_caps(w_total, buckets=buckets, max_weight_per_bucket=float(self.cfg.max_weight_per_bucket))
        w_total = neutralize_weights(
            w_total, betas=betas, dollar_neutral=self.cfg.dollar_neutral, beta_neutral=self.cfg.beta_neutral
        )

        gross = float(sum(abs(v) for v in w_total.values()))
        net = float(sum(v for v in w_total.values()))
        beta = 0.0
        if betas is not None:
            beta = float(sum(w_total[s] * float(betas.get(s, 0.0)) for s in w_total))

        meta = {
            "q_total": float(regime.q_total),
            "q_mr": float(regime.q_mr),
            "q_liq": float(regime.q_liq),
            "q_risk": float(regime.q_risk),
            "q_fund": float(q_fund),
            "gross_stat_target": float(self.cfg.gross_target_stat),
            "gross_cap_total": float(gross_cap),
        }

        return PortfolioWeights(ts=ts, weights=w_total, gross=gross, net=net, beta=beta, meta=meta)
