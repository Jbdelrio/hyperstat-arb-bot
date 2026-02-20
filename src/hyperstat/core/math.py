# src/hyperstat/core/math.py
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ----------------------------
# Robust statistics
# ----------------------------

def mad(x: np.ndarray, scale: float = 1.4826) -> float:
    """
    Median Absolute Deviation (scaled to match std under normality).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    m = np.nanmedian(x)
    d = np.nanmedian(np.abs(x - m))
    return float(scale * d)


def zscore_mad(x: float, x_bucket: np.ndarray, eps: float = 1e-12) -> float:
    """
    Robust z-score using bucket median and MAD.
    """
    med = float(np.nanmedian(x_bucket))
    s = mad(x_bucket)
    if not np.isfinite(s) or s < eps:
        return 0.0
    return float((x - med) / (s + eps))


def clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# ----------------------------
# EWMA / returns
# ----------------------------

def ewma(series: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    EWMA smoothing.
    lam close to 1 => more smoothing. Returns same shape.
    """
    s = np.asarray(series, dtype=float)
    out = np.empty_like(s)
    if s.size == 0:
        return out
    out[0] = s[0]
    for i in range(1, s.size):
        out[i] = lam * out[i - 1] + (1.0 - lam) * s[i]
    return out


def log_returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    if p.size < 2:
        return np.array([], dtype=float)
    return np.log(p[1:] / p[:-1])


def realized_vol(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    return float(np.sqrt(np.nansum(r * r)))


# ----------------------------
# Linear algebra helpers (neutralization / projection)
# ----------------------------

def project_to_constraints(
    u: np.ndarray,
    A: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Project vector u into null-space of constraints A w = 0 via:
      w = (I - A^T (A A^T)^{-1} A) u
    where A shape is (k, n), u shape is (n,).
    """
    u = np.asarray(u, dtype=float)
    A = np.asarray(A, dtype=float)

    if A.size == 0:
        return u.copy()

    # Compute M = (A A^T)^{-1}
    AAT = A @ A.T
    # Regularize for stability
    AAT = AAT + eps * np.eye(AAT.shape[0])
    M = np.linalg.inv(AAT)
    P = np.eye(u.shape[0]) - A.T @ M @ A
    return P @ u


def neutralize_weights(
    weights: Dict[str, float],
    betas: Optional[Dict[str, float]] = None,
    dollar_neutral: bool = True,
    beta_neutral: bool = False,
) -> Dict[str, float]:
    """
    Enforce constraints:
      sum(w)=0  (dollar neutral)
      sum(w*beta)=0 (beta neutral)
    via projection in weight-space.

    Note: After caps/scaling you may want to neutralize again.
    """
    syms = list(weights.keys())
    w = np.array([weights[s] for s in syms], dtype=float)

    rows = []
    if dollar_neutral:
        rows.append(np.ones_like(w))
    if beta_neutral and betas is not None:
        rows.append(np.array([betas.get(s, 0.0) for s in syms], dtype=float))

    if not rows:
        return dict(weights)

    A = np.vstack(rows)  # (k, n)
    w_proj = project_to_constraints(w, A)

    return {s: float(w_proj[i]) for i, s in enumerate(syms)}


# ----------------------------
# AR(1) + half-life utilities
# ----------------------------

def fit_ar1(x: np.ndarray) -> Tuple[float, float]:
    """
    Fit AR(1): x_t = a + b x_{t-1} + e_t
    Returns (a, b). Uses OLS, ignores NaNs.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0, float("nan")

    x0 = x[:-1]
    x1 = x[1:]
    mask = np.isfinite(x0) & np.isfinite(x1)
    x0 = x0[mask]
    x1 = x1[mask]
    if x0.size < 2:
        return 0.0, float("nan")

    X = np.vstack([np.ones_like(x0), x0]).T
    beta = np.linalg.lstsq(X, x1, rcond=None)[0]  # [a, b]
    return float(beta[0]), float(beta[1])


def half_life_minutes(b: float, dt_minutes: float) -> float:
    """
    Half-life for AR(1) coefficient b in (0,1):
      t_half = ln(2)/(-ln(b)) * dt
    """
    if not (0.0 < b < 1.0):
        return float("inf")
    return float(math.log(2.0) / (-math.log(b)) * dt_minutes)


def normalize_to_gross(weights: Dict[str, float], gross_target: float, eps: float = 1e-12) -> Dict[str, float]:
    g = float(sum(abs(v) for v in weights.values()))
    if g < eps:
        return {k: 0.0 for k in weights}
    scale = gross_target / g
    return {k: float(v * scale) for k, v in weights.items()}
