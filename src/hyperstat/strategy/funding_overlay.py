# src/hyperstat/strategy/funding_overlay.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from hyperstat.core.math import project_to_constraints, normalize_to_gross, clip


@dataclass(frozen=True)
class FundingOverlayConfig:
    """
    Funding overlay: small tilt to capture carry.

    You should update the model only on funding timestamps (low turnover).

    Convention:
      - funding rate f > 0 typically means longs pay shorts -> prefer SHORT => u = -mu
      - we use EWMA(mu) and EWMA(|f-mu|) as robust scale; SNR=|mu|/(mad+eps)

    Break-even check is in bps:
      expected_bps = |mu| * 1e4 * horizon_periods
      cost_bps = 2*(fee_bps + slip_bps_base) + buffer_bps  (conservative)
    """
    enabled: bool = True
    ewma_lambda: float = 0.15  # weight of new observation in EWMA
    snr_gate_min: float = 0.5

    # overlay sizing
    eta: float = 0.10          # mixing strength
    gross_target: float = 0.20 # overlay gross exposure

    # break-even
    horizon_funding_periods: int = 3
    fee_bps: float = 6.0
    slip_bps_base: float = 8.0
    buffer_bps: float = 5.0

    eps: float = 1e-12


@dataclass
class FundingOverlayModel:
    cfg: FundingOverlayConfig

    # per symbol EWMA stats
    mu: Dict[str, float] = field(default_factory=dict)
    mad: Dict[str, float] = field(default_factory=dict)
    snr: Dict[str, float] = field(default_factory=dict)

    last_update_ts: Optional[int] = None  # optional external gating by time

    def update(self, funding_rates: Dict[str, float]) -> None:
        """
        Update EWMA stats with latest funding rates:
          funding_rates: symbol -> rate (e.g. 0.0001)
        Call this only when you have a new funding tick/snapshot.
        """
        lam = float(self.cfg.ewma_lambda)
        for s, f in funding_rates.items():
            x = float(f)
            if not np.isfinite(x):
                continue

            prev_mu = self.mu.get(s)
            if prev_mu is None or not np.isfinite(prev_mu):
                new_mu = x
            else:
                new_mu = lam * x + (1.0 - lam) * prev_mu

            dev = abs(x - new_mu)
            prev_mad = self.mad.get(s)
            if prev_mad is None or not np.isfinite(prev_mad):
                new_mad = dev
            else:
                new_mad = lam * dev + (1.0 - lam) * prev_mad

            self.mu[s] = float(new_mu)
            self.mad[s] = float(new_mad)
            self.snr[s] = float(abs(new_mu) / (new_mad + self.cfg.eps))

    def _break_even_ok(self, mu: float) -> bool:
        expected_bps = abs(mu) * 1e4 * float(self.cfg.horizon_funding_periods)
        cost_bps = 2.0 * (self.cfg.fee_bps + self.cfg.slip_bps_base) + self.cfg.buffer_bps
        return expected_bps > cost_bps

    def compute_overlay(
        self,
        symbols: List[str],
        betas: Optional[Dict[str, float]] = None,
        dollar_neutral: bool = True,
        beta_neutral: bool = False,
    ) -> Tuple[Dict[str, float], float]:
        """
        Returns:
          overlay_weights (already gross-normalized to cfg.gross_target and projected to constraints)
          q_fund in [0,1] (global confidence score)
        """
        if not self.cfg.enabled:
            return {s: 0.0 for s in symbols}, 0.0

        u = []
        snrs = []

        for s in symbols:
            mu = self.mu.get(s, 0.0)
            snr = self.snr.get(s, 0.0)

            # gating by SNR + break-even
            if snr < self.cfg.snr_gate_min:
                u.append(0.0)
                snrs.append(float(snr))
                continue
            if not self._break_even_ok(mu):
                u.append(0.0)
                snrs.append(float(snr))
                continue

            # direction preference: u = -mu  (mu>0 => want short)
            u.append(float(-mu))
            snrs.append(float(snr))

        u_vec = np.asarray(u, dtype=float)

        # confidence score from cross-sectional SNR
        q_fund = float(np.nanmedian(snrs)) if snrs else 0.0
        # map to [0,1] softly
        q_fund = clip(q_fund / max(self.cfg.snr_gate_min, 1e-6), 0.0, 1.0)

        if np.allclose(u_vec, 0.0):
            return {s: 0.0 for s in symbols}, q_fund

        # projection to constraints
        rows = []
        if dollar_neutral:
            rows.append(np.ones_like(u_vec))
        if beta_neutral and betas is not None:
            rows.append(np.asarray([float(betas.get(s, 0.0)) for s in symbols], dtype=float))

        if rows:
            A = np.vstack(rows)
            w = project_to_constraints(u_vec, A)
        else:
            w = u_vec.copy()

        # normalize overlay gross, then apply eta and q_fund
        w_dict = {s: float(w[i]) for i, s in enumerate(symbols)}
        w_dict = normalize_to_gross(w_dict, gross_target=float(self.cfg.gross_target))

        # mix strength + confidence
        scale = float(self.cfg.eta) * float(q_fund)
        w_dict = {s: float(scale * v) for s, v in w_dict.items()}

        return w_dict, q_fund
