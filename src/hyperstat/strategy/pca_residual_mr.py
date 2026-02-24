# src/hyperstat/strategy/pca_residual_mr.py
"""
PCA Residual Mean-Reversion Agent (SignalAgent 6)
===================================================
Removes common factor moves before applying MR signal.

Step 1 : Collect returns matrix R (n_symbols × window_bars)
Step 2 : SVD → extract top-k eigenvectors (market + sector factors)
Step 3 : residual_i = r_i - B_i @ f   (factor-neutralized residual)
Step 4 : z_res_i = zscore_mad(residual_i, bucket)
Step 5 : signal = -clip(z_res_i, ±z_max)   (MR on cleaner residuals)

Refit PCA every pca_refit_bars (default 20).
Anti-frais: same z_in/z_out hysteresis as StatArbStrategy.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .base_signal_agent import AgentContext, AgentOutput, BaseSignalAgent


@dataclass(frozen=True)
class PCAResiduaMRConfig:
    window_bars: int = 30        # bars for PCA calibration + z-score window
    n_factors: int = 3           # number of PCA factors to remove
    z_in: float = 1.5            # entry threshold
    z_out: float = 0.5           # exit threshold
    z_max: float = 3.0           # signal clip
    min_hold_minutes: float = 20.0
    max_hold_minutes: float = 1440.0
    pca_refit_bars: int = 20     # refit PCA every N bars
    vol_ewma_lam: float = 0.94
    min_vol_floor: float = 1e-6
    eps: float = 1e-12


@dataclass
class _PCAResiduaLegState:
    active: bool = False
    entered_at: Optional[datetime] = None
    side: int = 0


class PCAResiduaMRAgent(BaseSignalAgent):
    """
    PCA-neutralized mean-reversion.
    Removes top-k market/sector factors before computing z-scores,
    giving cleaner signals with fewer false positives.
    """

    name = "PCA Residual MR"
    warmup_bars = 30
    enabled_by_default = True

    def __init__(self, cfg: Optional[PCAResiduaMRConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PCAResiduaMRConfig()
        # Price history: deque of (ts, price) per symbol
        self._price_hist: Dict[str, Deque[Tuple[datetime, float]]] = {}
        # Return history: deque of float per symbol
        self._ret_hist: Dict[str, Deque[float]] = {}
        # Hysteresis state
        self._leg: Dict[str, _PCAResiduaLegState] = {}
        # EWMA vol
        self._ewma_var: Dict[str, float] = {}
        # PCA loadings: left singular vectors U[:, :k], shape (n_symbols, n_factors)
        # U[:,i] is the i-th factor's symbol loadings (asset space, NOT time space)
        self._pca_loadings: Optional[np.ndarray] = None  # (n_symbols, k)
        self._pca_symbols: List[str] = []
        self._bars_since_refit: int = 0

    def reset(self) -> None:
        self._bars_seen = 0
        self._price_hist.clear()
        self._ret_hist.clear()
        self._leg.clear()
        self._ewma_var.clear()
        self._pca_loadings = None
        self._pca_symbols = []
        self._bars_since_refit = 0

    def update(
        self,
        ts: datetime,
        mids: Dict[str, float],
        context: AgentContext,
    ) -> AgentOutput:
        cfg = self.cfg
        max_hist = cfg.window_bars + 2

        # ── Update price/return history ───────────────────────────────────────
        for sym, px in mids.items():
            if not (math.isfinite(px) and px > 0):
                continue
            if sym not in self._price_hist:
                self._price_hist[sym] = deque(maxlen=max_hist)
                self._ret_hist[sym] = deque(maxlen=max_hist)

            q = self._price_hist[sym]
            if q:
                r = self._log_return(q[-1][1], px)
                self._ret_hist[sym].append(r)
                prev_var = self._ewma_var.get(sym, r * r)
                self._ewma_var[sym] = self._ewma_update(prev_var, r * r, cfg.vol_ewma_lam)
            q.append((ts, px))

        self._bars_seen += 1
        self._bars_since_refit += 1

        if not self.is_warmed_up:
            return AgentOutput()

        # ── Identify symbols with full history ────────────────────────────────
        selected = [
            sym for sym in context.selected
            if sym in self._ret_hist and len(self._ret_hist[sym]) >= cfg.window_bars
        ]
        if len(selected) < max(cfg.n_factors + 2, 4):
            return AgentOutput()

        # ── Refit PCA periodically ────────────────────────────────────────────
        if (self._pca_loadings is None
                or self._bars_since_refit >= cfg.pca_refit_bars
                or self._pca_symbols != selected):
            self._refit_pca(selected)
            self._bars_since_refit = 0

        if self._pca_loadings is None:
            return AgentOutput()

        # ── Compute factor residuals ──────────────────────────────────────────
        sym_to_idx = {s: i for i, s in enumerate(self._pca_symbols)}
        active_syms = [s for s in selected if s in sym_to_idx]

        # Latest return for each symbol
        latest_ret = {s: (list(self._ret_hist[s])[-1] if self._ret_hist[s] else 0.0)
                      for s in active_syms}
        r_vec = np.array([latest_ret[s] for s in active_syms], dtype=float)

        # Symbol-space factor loadings for active symbols: L_sub (n_sub, k)
        # L_sub[:,i] is the i-th factor's loading across active symbols
        L_sub = self._pca_loadings[[sym_to_idx[s] for s in active_syms], :]  # (n_sub, k)

        # Factor returns: f = L_sub.T @ r_vec  (k,)
        # = projection of returns onto the PCA factor directions
        f = L_sub.T @ r_vec   # (k,)

        # Reconstructed (factor-explained) return: r_hat = L_sub @ f  (n_sub,)
        r_hat = L_sub @ f     # (n_sub,)

        # Residuals: component of returns orthogonal to top-k factors
        resid = r_vec - r_hat  # (n_sub,)

        residuals = {s: float(resid[i]) for i, s in enumerate(active_syms)}

        # ── Cross-sectional z-score of residuals per bucket ───────────────────
        zscores: Dict[str, float] = {}
        for bucket_syms in context.buckets.values():
            b_syms = [s for s in bucket_syms if s in residuals]
            if len(b_syms) < 2:
                continue
            b_vals = np.array([residuals[s] for s in b_syms])
            med = float(np.nanmedian(b_vals))
            m = float(np.nanmedian(np.abs(b_vals - med))) * 1.4826
            for s in b_syms:
                zscores[s] = float((residuals[s] - med) / (m + cfg.eps)) if m > cfg.eps else 0.0

        # ── Hysteresis ────────────────────────────────────────────────────────
        weights: Dict[str, float] = {}

        for sym in active_syms:
            if sym not in zscores:
                continue
            z = zscores[sym]
            leg = self._leg.get(sym, _PCAResiduaLegState())

            if not leg.active:
                if abs(z) >= cfg.z_in:
                    leg = _PCAResiduaLegState(active=True, entered_at=ts, side=int(np.sign(-z)))
            else:
                held_min = (ts - leg.entered_at).total_seconds() / 60.0 if leg.entered_at else 0.0
                if held_min >= cfg.max_hold_minutes:
                    leg = _PCAResiduaLegState()
                elif held_min >= cfg.min_hold_minutes and abs(z) <= cfg.z_out:
                    leg = _PCAResiduaLegState()

            self._leg[sym] = leg

            if leg.active:
                vol = math.sqrt(max(self._ewma_var.get(sym, cfg.min_vol_floor), cfg.min_vol_floor))
                signal = -float(np.clip(z, -cfg.z_max, cfg.z_max))
                weights[sym] = signal / (vol + 1e-12)

        # Normalize to unit gross
        total = sum(abs(v) for v in weights.values())
        if total < cfg.eps:
            return AgentOutput(zscores=zscores)
        norm_w = {s: v / total for s, v in weights.items()}

        return AgentOutput(
            weights=norm_w,
            zscores=zscores,
            meta={"n_factors": cfg.n_factors, "n_active": len(weights)},
        )

    def _refit_pca(self, symbols: List[str]) -> None:
        """Fit PCA on recent return history for the given symbols."""
        cfg = self.cfg
        ret_matrix = []
        valid_syms = []
        for sym in symbols:
            hist = list(self._ret_hist.get(sym, []))
            if len(hist) >= cfg.window_bars:
                ret_matrix.append(hist[-cfg.window_bars:])
                valid_syms.append(sym)

        if len(valid_syms) < cfg.n_factors + 2:
            return

        R = np.array(ret_matrix, dtype=float)  # (n_syms, window)
        # Demean per symbol (remove mean return before PCA)
        R -= R.mean(axis=1, keepdims=True)

        # SVD: R = U @ diag(S) @ Vt
        #   U columns = LEFT singular vectors → SYMBOL-space factor loadings (n_syms, min)
        #   Vt rows   = RIGHT singular vectors → TIME-space factors (min, window)
        # We need U (symbol space), NOT Vt (time space).
        try:
            U, _, _ = np.linalg.svd(R, full_matrices=False)
        except np.linalg.LinAlgError:
            return

        # Top-k symbol-space loadings: U[:, :k] is (n_syms, k)
        # U[:, i] gives the loading of each symbol onto the i-th PCA factor
        k = min(cfg.n_factors, U.shape[1], len(valid_syms) - 1)
        self._pca_loadings = U[:, :k]  # (n_syms, k) — symbol factor loadings
        self._pca_symbols = valid_syms
