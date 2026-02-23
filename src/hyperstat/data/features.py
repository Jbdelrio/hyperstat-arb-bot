# src/hyperstat/data/features.py
"""
Feature computation from OHLCV / funding DataFrames.

Fonctions exportées (toutes attendues par data/__init__.py) :
  compute_returns(df)                          -> pd.Series  log-returns indexés par ts
  compute_ewma_vol(r, lam, min_periods)        -> pd.Series  vol EWMA (RiskMetrics lam=0.94)
  compute_rv(df, window_bars)                  -> pd.Series  realized vol sur N barres
  compute_rv_1h_pct(df, timeframe_minutes)     -> pd.Series  realized vol 1h (fraction)
  compute_amihud_illiq(df, window_bars)        -> pd.Series  Amihud illiquidité rolling
  compute_beta_vs_factor(r, r_factor, window)  -> pd.Series  beta rolling vs facteur
  compute_residual_returns(r, r_factor, window)-> pd.Series  résidus après BTC-beta
  compute_funding_ewma_stats(f, lam)           -> FundingStats(mu, mad, snr)

Fonctions additionnelles pour agents / pipeline :
  compute_rolling_returns, compute_rsi, compute_macd, compute_bollinger, compute_atr,
  compute_dollar_volume, compute_amihud, compute_order_book_imbalance,
  compute_rolling_beta, compute_btc_residuals,
  compute_all_features, compute_cross_sectional_features
"""
from __future__ import annotations

from collections import namedtuple
from typing import Dict, Optional, Tuple

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
    Accepte un DataFrame avec colonne ts (ou index DatetimeIndex).
    Retourne une Series indexée par ts (premier élément = NaN).
    """
    dfi = _to_indexed(df)
    c = dfi["close"].astype(float)
    r = np.log(c / c.shift(1).replace(0, np.nan))
    return r.rename("log_return")


def compute_rolling_returns(
    prices: pd.Series,
    horizon: int,
) -> pd.Series:
    """Retour log sur fenêtre glissante de `horizon` barres (depuis une Series de prix)."""
    return np.log(prices / prices.shift(horizon))


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

    Parameters
    ----------
    r           : log-returns (pd.Series)
    lam         : facteur de décroissance (0.94 = RiskMetrics daily)
    min_periods : minimum de barres valides avant de produire une valeur
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
    Illiquidité d'Amihud rolling depuis un DataFrame OHLCV :
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


def compute_dollar_volume(candles: pd.DataFrame) -> pd.Series:
    """Dollar volume proxy : close × volume."""
    return (candles["close"] * candles["volume"]).rename("dollar_volume")


def compute_amihud(
    returns: pd.Series,
    dollar_volume: pd.Series,
    window: int = 100,
) -> pd.Series:
    """
    Illiquidité d'Amihud (rolling médiane) depuis des Series pré-calculées.
    ILLIQ = median(|r| / DV).
    Pour l'API DataFrame, voir compute_amihud_illiq().
    """
    ratio = returns.abs() / (dollar_volume + 1e-9)
    return ratio.rolling(window).median().rename("amihud")


def compute_order_book_imbalance(
    bid_size: pd.Series,
    ask_size: pd.Series,
) -> pd.Series:
    """
    Order Book Imbalance (OBI) depuis données L2.
    OBI ∈ [-1, 1] : positif = pression achat, négatif = pression vente.
    """
    total = bid_size + ask_size + 1e-9
    return ((bid_size - ask_size) / total).rename("obi")


# ─────────────────────────────────────────────────────────────────────────────
# Beta / résidus  — API originale (utilisée par universe.py + __init__.py)
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
# Beta / résidus  — API simplifiée pour agents (fenêtre courte, sans min_periods)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_beta(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    window: int = 288,   # 24h en barres 5min
) -> pd.Series:
    """
    Beta rolling asset vs BTC (fenêtre courte, usage pipeline agent).
    Pour la version avec min_periods, voir compute_beta_vs_factor().
    """
    cov = asset_returns.rolling(window).cov(btc_returns)
    var = btc_returns.rolling(window).var()
    return (cov / (var + 1e-9)).rename("beta_btc")


def compute_btc_residuals(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Résidus après neutralisation BTC : ε = r_i - β_i × r_BTC."""
    return (asset_returns - beta * btc_returns).rename("btc_residual")


# ─────────────────────────────────────────────────────────────────────────────
# Funding stats
# ─────────────────────────────────────────────────────────────────────────────

FundingStats = namedtuple("FundingStats", ["mu", "mad", "snr"])


def compute_funding_ewma_stats(
    f,
    lam: float = 0.15,
    eps: float = 1e-12,
) -> FundingStats:
    """
    EWMA stats du funding rate (utilisées par FundingOverlayModel) :
        mu  = EWMA(f)
        mad = EWMA(|f - mu|)
        snr = |mu| / (mad + eps)

    Parameters
    ----------
    f   : pd.Series de funding rates OU DataFrame avec colonne "rate"
    lam : facteur de lissage EWMA

    Retourne FundingStats(mu, mad, snr) — accessible par attribut (.mu, .mad, .snr)
    ou par déstructuration (mu, mad, snr = compute_funding_ewma_stats(...)).
    """
    if isinstance(f, pd.DataFrame):
        if "rate" in f.columns:
            f = f["rate"]
        elif not f.empty:
            f = f.iloc[:, 0]
        else:
            empty = pd.Series(dtype=float)
            return FundingStats(mu=empty, mad=empty, snr=empty)

    f = f.astype(float)
    alpha = float(lam)
    mu = f.ewm(alpha=alpha, adjust=False, min_periods=1).mean().rename("funding_mu")
    dev = (f - mu).abs()
    mad = dev.ewm(alpha=alpha, adjust=False, min_periods=1).mean().rename("funding_mad")
    snr = (mu.abs() / (mad + eps)).rename("funding_snr")
    return FundingStats(mu=mu, mad=mad, snr=snr)


# ─────────────────────────────────────────────────────────────────────────────
# Indicateurs techniques (ajouts pour agents)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """RSI classique (Wilder)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / (loss + 1e-9)
    return (100 - 100 / (1 + rs)).rename("rsi")


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


def compute_bollinger(
    prices: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Bandes de Bollinger."""
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return pd.DataFrame({
        "bb_upper": mid + n_std * std,
        "bb_mid": mid,
        "bb_lower": mid - n_std * std,
        "bb_width": (2 * n_std * std) / (mid + 1e-9),
        "bb_pct": (prices - (mid - n_std * std)) / (2 * n_std * std + 1e-9),
    })


def compute_atr(candles: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range."""
    high = candles["high"]
    low = candles["low"]
    prev_close = candles["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, min_periods=window).mean().rename("atr")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet par symbole (pour agents)
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_features(
    candles: pd.DataFrame,
    btc_returns: Optional[pd.Series] = None,
    include_ta: bool = True,
) -> pd.DataFrame:
    """
    Calcule toutes les features pour un symbole donné (pipeline agent).
    Retourne un DataFrame enrichi avec toutes les colonnes.

    Parameters
    ----------
    candles      : OHLCV DataFrame (colonnes: ts, open, high, low, close, volume)
    btc_returns  : log-returns BTC (optionnel, pour beta)
    include_ta   : si True, calcule RSI, MACD, BB, ATR

    Returns
    -------
    pd.DataFrame avec colonnes features supplémentaires
    """
    df = candles.copy()
    close = df["close"].astype(float)

    # Returns (sur la Series close directement, sans _to_indexed)
    log_ret = np.log(close / close.shift(1)).rename("log_return")
    df["log_return"] = log_ret
    df["return_12b"] = compute_rolling_returns(close, horizon=12)    # ~1h  (5min bars)
    df["return_48b"] = compute_rolling_returns(close, horizon=48)    # ~4h
    df["return_288b"] = compute_rolling_returns(close, horizon=288)  # ~24h

    # Volatilité (span-based pour cohérence avec la convention "20 barres" / "96 barres")
    df["ewma_vol_20"] = log_ret.ewm(span=20, min_periods=10).std()
    df["ewma_vol_96"] = log_ret.ewm(span=96, min_periods=20).std()

    # RV 1h : via compute_rv_1h_pct si ts disponible, sinon approx rolling std
    if "ts" in df.columns:
        df["rv_1h_pct"] = compute_rv_1h_pct(df).values
    else:
        df["rv_1h_pct"] = log_ret.rolling(12).std() * 100

    # Microstructure
    df["dollar_volume"] = compute_dollar_volume(df)
    df["amihud"] = compute_amihud(log_ret, df["dollar_volume"])

    # Indicateurs techniques (optionnel)
    if include_ta:
        df["rsi_14"] = compute_rsi(close)
        macd_df = compute_macd(close)
        df = pd.concat([df, macd_df], axis=1)
        bb_df = compute_bollinger(close)
        df = pd.concat([df, bb_df], axis=1)
        df["atr_14"] = compute_atr(df)

    # Beta BTC (si fourni)
    if btc_returns is not None:
        df["beta_btc"] = compute_rolling_beta(log_ret, btc_returns)
        df["btc_residual"] = compute_btc_residuals(log_ret, btc_returns, df["beta_btc"])

    return df


def compute_cross_sectional_features(
    candles_by_symbol: Dict[str, pd.DataFrame],
    btc_symbol: str = "BTC",
) -> Dict[str, pd.DataFrame]:
    """
    Calcule les features pour tout l'univers en une passe (pipeline agent).

    Parameters
    ----------
    candles_by_symbol : {symbol: DataFrame OHLCV}
    btc_symbol        : symbole de référence pour le beta

    Returns
    -------
    {symbol: DataFrame enrichi}
    """
    btc_returns: Optional[pd.Series] = None
    if btc_symbol in candles_by_symbol:
        btc_close = candles_by_symbol[btc_symbol]["close"].astype(float)
        btc_returns = np.log(btc_close / btc_close.shift(1))

    result: Dict[str, pd.DataFrame] = {}
    for sym, df in candles_by_symbol.items():
        result[sym] = compute_all_features(df, btc_returns=btc_returns)

    return result