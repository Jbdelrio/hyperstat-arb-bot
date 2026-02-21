# src/hyperstat/data/features.py
"""
Feature computation from OHLCV / funding DataFrames.

Fonctions exportées (toutes attendues par data/__init__.py) :
  compute_returns(df)                          -> pd.Series  log-returns indexés par ts
  compute_ewma_vol(r, lam, min_periods)        -> pd.Series  vol EWMA
  compute_rv(df, window_bars)                  -> pd.Series  realized vol sur N barres
  compute_rv_1h_pct(df, timeframe_minutes)     -> pd.Series  realized vol 1h (fraction)
  compute_amihud_illiq(df, window_bars)        -> pd.Series  Amihud illiquidité rolling
  compute_beta_vs_factor(r, r_factor, window)  -> pd.Series  beta rolling vs facteur
  compute_residual_returns(r, r_factor, window)-> pd.Series  résidus après BTC-beta
  compute_funding_ewma_stats(f, lam)           -> Tuple[pd.Series, pd.Series, pd.Series]
                                                  (mu, mad, snr)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaire interne
# ─────────────────────────────────────────────────────────────────────────────

def _to_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne df avec index DatetimeIndex UTC sur la colonne ts."""
    dfi = df.copy()
    if "ts" in dfi.columns:
        dfi = dfi.set_index("ts").sort_index()
    if dfi.index.tzinfo is None:
        dfi.index = dfi.index.tz_localize("UTC")
    return dfi


# ─────────────────────────────────────────────────────────────────────────────
# Returns
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(df: pd.DataFrame, eps: float = 1e-12) -> pd.Series:
    """
    Log-returns depuis les prix close.
    Retourne une Series indexée par ts (premier élément = NaN).
    """
    dfi = _to_indexed(df)
    c = dfi["close"].astype(float)
    r = np.log(c / c.shift(1).replace(0, np.nan))
    return r.rename("log_return")


# ─────────────────────────────────────────────────────────────────────────────
# Volatilité
# ─────────────────────────────────────────────────────────────────────────────

def compute_ewma_vol(
    r: pd.Series,
    lam: float = 0.94,
    min_periods: int = 20,
) -> pd.Series:
    """
    Volatilité EWMA des log-returns (formule RiskMetrics).
        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_t
    Retourne l'écart-type (fraction, ex: 0.008).
    """
    r2 = r.astype(float) ** 2
    alpha = 1.0 - lam
    var = r2.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()
    return var.apply(
        lambda x: float(np.sqrt(x)) if np.isfinite(x) else np.nan
    ).rename("ewma_vol")


def compute_rv(df: pd.DataFrame, window_bars: int = 12) -> pd.Series:
    """
    Realized volatility sur une fenêtre glissante de N barres.
        RV = sqrt( sum_{k} r_k^2 )
    Retourne une Series en fraction (ex: 0.012).
    """
    dfi = _to_indexed(df)
    c = dfi["close"].astype(float)
    r = np.log(c / c.shift(1).replace(0, np.nan))
    r2 = r ** 2
    rv = r2.rolling(window=window_bars, min_periods=window_bars // 2).sum().apply(
        lambda x: float(np.sqrt(x)) if np.isfinite(x) else np.nan
    )
    return rv.rename("rv")


def compute_rv_1h_pct(
    df: pd.DataFrame,
    timeframe_minutes: int = 5,
) -> pd.Series:
    """
    Realized volatility sur une fenêtre glissante d'1 heure.
    H = 60 / timeframe_minutes barres.
    Retourne une Series en fraction (ex: 0.012 = 1.2%).
    """
    bars_per_hour = max(1, int(60 / timeframe_minutes))
    return compute_rv(df, window_bars=bars_per_hour).rename("rv_1h")


# ─────────────────────────────────────────────────────────────────────────────
# Liquidité
# ─────────────────────────────────────────────────────────────────────────────

def compute_amihud_illiq(
    df: pd.DataFrame,
    window_bars: int = 288,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Illiquidité d'Amihud rolling :
        ILLIQ_t = median_{k=t-W+1}^{t}( |r_k| / DV_k )
    où DV_k = close_k * volume_k.
    Retourne une Series (valeurs petites = liquide).
    """
    dfi = _to_indexed(df)
    c = dfi["close"].astype(float)
    v = dfi["volume"].astype(float)
    r = np.log(c / c.shift(1).replace(0, np.nan)).abs()
    dv = (c * v).replace(0, np.nan)
    ratio = r / (dv + eps)
    illiq = ratio.rolling(window=window_bars, min_periods=window_bars // 4).median()
    return illiq.rename("amihud_illiq")


# ─────────────────────────────────────────────────────────────────────────────
# Beta / résidus
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta_vs_factor(
    r: pd.Series,
    r_factor: pd.Series,
    window: int = 1440,
    min_periods: int = 100,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Beta rolling d'un actif contre un facteur (ex: BTC) :
        beta_t = Cov_W(r, r_factor) / Var_W(r_factor)
    """
    r = r.astype(float)
    rf = r_factor.astype(float).reindex(r.index)
    cov = r.rolling(window, min_periods=min_periods).cov(rf)
    var = rf.rolling(window, min_periods=min_periods).var()
    beta = cov / (var + eps)
    return beta.rename("beta")


def compute_residual_returns(
    r: pd.Series,
    r_factor: pd.Series,
    window: int = 1440,
    min_periods: int = 100,
) -> pd.Series:
    """
    Résidus après neutralisation beta :
        eps_t = r_t - beta_t * r_factor_t
    Utilisés pour le clustering sur résidus BTC-neutralisés.
    """
    beta = compute_beta_vs_factor(r, r_factor, window=window, min_periods=min_periods)
    rf = r_factor.astype(float).reindex(r.index)
    residuals = r.astype(float) - beta * rf
    return residuals.rename("residual_return")


# ─────────────────────────────────────────────────────────────────────────────
# Funding stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_funding_ewma_stats(
    f: pd.Series,
    lam: float = 0.15,
    eps: float = 1e-12,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    EWMA stats du funding rate (utilisées par FundingOverlayModel) :
        mu  = EWMA(f)
        mad = EWMA(|f - mu|)
        snr = |mu| / (mad + eps)
    Retourne (mu, mad, snr).
    """
    f = f.astype(float)
    alpha = float(lam)
    mu = f.ewm(alpha=alpha, adjust=False, min_periods=1).mean().rename("funding_mu")
    dev = (f - mu).abs()
    mad = dev.ewm(alpha=alpha, adjust=False, min_periods=1).mean().rename("funding_mad")
    snr = (mu.abs() / (mad + eps)).rename("funding_snr")
    return mu, mad, snr
