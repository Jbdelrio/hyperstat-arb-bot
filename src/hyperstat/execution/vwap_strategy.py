# src/hyperstat/execution/vwap_strategy.py
"""
Stratégies VWAP et TWAP pour l'exécution fragmentée des ordres.

Références :
- Genet (2025), "Deep Learning for VWAP Execution in Crypto Markets"
  arXiv:2502.13722 — optimisation directe de l'objectif VWAP sans
  prédiction intermédiaire de la courbe de volume.
- He & Manela (2024), "Fundamentals of Perpetual Futures"
  arXiv:2212.06888 — bornes no-arbitrage perp-spot avec coûts de transaction.

Usage minimal (live) :
    from hyperstat.execution import OrderSlicer, ExecutionConfig

    slicer = OrderSlicer(cfg=ExecutionConfig(mode="hybrid", n_slices=8))
    slices = slicer.slice_order(
        symbol="ARB",
        direction="BUY",
        total_notional=5000.0,
        df_recent=recent_candles_df,
    )
    for limit_px, size in slices:
        exchange.place_limit_order("ARB", "BUY", size, limit_px)

Usage intégration allocateur :
    from hyperstat.execution import execution_cost_adjustment

    # Dans allocator.allocate(), juste avant les caps :
    signal_weights = execution_cost_adjustment(signal_weights, df_candles_latest)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionConfig:
    """Paramètres de l'exécution fragmentée."""
    # Nombre de sous-ordres (tranches)
    n_slices: int = 8
    # Fenêtre de calcul VWAP en barres (ex : 8 barres de 1h = 8h)
    vwap_window: int = 8
    # Tolérance de prix : n'exécute pas si écart > tol vs VWAP pour un achat
    price_tolerance: float = 0.003    # 30 bps
    # Mode : "vwap" | "twap" | "hybrid"
    mode: str = "hybrid"
    # En mode hybrid : poids donné au signal VWAP (vs TWAP uniforme)
    vwap_weight: float = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Calculs de base
# ─────────────────────────────────────────────────────────────────────────────

def calculate_vwap(df: pd.DataFrame, window: int = 8) -> pd.Series:
    """
    VWAP roulant sur `window` barres.

    Formule :  VWAP_t = Σ(V_i × P_typ_i) / Σ(V_i)
    où P_typ = (high + low + close) / 3  (typical price).

    Args:
        df     : DataFrame avec colonnes high / low / close / volume.
        window : nombre de barres.

    Returns:
        pd.Series indexée comme df, nommée "vwap".
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cumvp = (typical * df["volume"]).rolling(window, min_periods=1).sum()
    cumv = df["volume"].rolling(window, min_periods=1).sum()
    return (cumvp / (cumv + 1e-10)).rename("vwap")


def calculate_twap(df: pd.DataFrame, window: int = 8) -> pd.Series:
    """
    TWAP roulant sur `window` barres (moyenne simple des typical prices).

    Args:
        df     : DataFrame avec colonnes high / low / close.
        window : nombre de barres.

    Returns:
        pd.Series indexée comme df, nommée "twap".
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    return typical.rolling(window, min_periods=1).mean().rename("twap")


def volume_profile_weights(df: pd.DataFrame, n_slices: int) -> np.ndarray:
    """
    Pondérations des tranches basées sur le profil de volume historique.

    Adapté de Genet (2025) arXiv:2502.13722 : au lieu de prédire la courbe
    de volume par deep learning, on utilise le profil historique rolling
    comme proxy (suffisant pour un horizon 1h-4h).

    Args:
        df       : DataFrame avec colonne volume.
        n_slices : nombre de tranches souhaitées.

    Returns:
        np.ndarray de poids normalisés (somme = 1), longueur n_slices.
    """
    vol = df["volume"].values
    if len(vol) < n_slices:
        return np.ones(n_slices) / n_slices
    profile = vol[-n_slices:].astype(float)
    total = profile.sum()
    if total <= 0.0:
        return np.ones(n_slices) / n_slices
    return profile / total


# ─────────────────────────────────────────────────────────────────────────────
# Slicer principal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderSlicer:
    """
    Fragmente un ordre parent en sous-ordres VWAP, TWAP ou hybrides.

    Chaque tranche est exprimée en (prix_limite, notionnel_tranche).
    Les tranches sont pondérées par le profil de volume historique (mode vwap/hybrid)
    ou uniformément (mode twap).

    Example::

        slicer = OrderSlicer(cfg=ExecutionConfig(mode="hybrid"))
        slices = slicer.slice_order(
            symbol="ARB",
            direction="BUY",
            total_notional=5000.0,
            df_recent=recent_candles_df,
        )
        for price_target, size in slices:
            exchange.place_limit_order(symbol, direction, size, price_target)
    """
    cfg: ExecutionConfig = field(default_factory=ExecutionConfig)

    def slice_order(
        self,
        symbol: str,
        direction: str,
        total_notional: float,
        df_recent: pd.DataFrame,
        current_price: float | None = None,
    ) -> List[Tuple[float, float]]:
        """
        Retourne une liste de (prix_limite, notionnel) par tranche.

        Args:
            symbol          : symbole (pour logging externe).
            direction       : "BUY" ou "SELL".
            total_notional  : montant total à exécuter en devise de cotation.
            df_recent       : DataFrame récent avec colonnes ts/open/high/low/close/volume.
                              Doit contenir au minimum cfg.vwap_window lignes pour un VWAP fiable.
            current_price   : prix actuel (optionnel — si None, utilise df_recent close[-1]).

        Returns:
            Liste de (limit_price, slice_notional). La somme des notionnels = total_notional.
        """
        if df_recent.empty or len(df_recent) < 2:
            p = current_price if current_price is not None else float("nan")
            if not np.isfinite(p):
                return []
            return [(p, total_notional)]

        w = self.cfg.vwap_window
        vwap_val = float(calculate_vwap(df_recent, w).iloc[-1])
        twap_val = float(calculate_twap(df_recent, w).iloc[-1])

        if self.cfg.mode == "vwap":
            ref_price = vwap_val
        elif self.cfg.mode == "twap":
            ref_price = twap_val
        else:  # hybrid
            ref_price = self.cfg.vwap_weight * vwap_val + (1.0 - self.cfg.vwap_weight) * twap_val

        cp = current_price if current_price is not None else float(df_recent["close"].iloc[-1])

        # Si le prix actuel dépasse le benchmark VWAP au-delà de la tolérance,
        # on exécute quand même mais au prix actuel (ne pas laisser l'ordre expirer).
        if direction.upper() == "BUY" and cp > ref_price * (1.0 + self.cfg.price_tolerance):
            ref_price = cp
        elif direction.upper() == "SELL" and cp < ref_price * (1.0 - self.cfg.price_tolerance):
            ref_price = cp

        # Pondérations des tranches
        if self.cfg.mode in ("vwap", "hybrid"):
            weights = volume_profile_weights(df_recent, self.cfg.n_slices)
        else:
            weights = np.ones(self.cfg.n_slices) / self.cfg.n_slices

        slices: List[Tuple[float, float]] = []
        for w_i in weights:
            slice_notional = round(float(total_notional * w_i), 2)
            # Prix limite légèrement agressif (0.5 bps) pour maximiser le fill
            if direction.upper() == "BUY":
                limit_px = round(ref_price * 1.00005, 6)
            else:
                limit_px = round(ref_price * 0.99995, 6)
            slices.append((limit_px, slice_notional))

        return slices


# ─────────────────────────────────────────────────────────────────────────────
# Ajustement de coût d'exécution pour l'allocateur
# ─────────────────────────────────────────────────────────────────────────────

def execution_cost_adjustment(
    signal_weights: "pd.Series",
    df_candles: Dict[str, pd.DataFrame],
    cfg: ExecutionConfig | None = None,
) -> "pd.Series":
    """
    Réduit les poids dont le coût d'exécution estimé dépasse l'alpha attendu.

    Logique : si le prix actuel s'écarte fortement du VWAP, l'exécution coûte
    plus cher que prévu. On pénalise le signal proportionnellement à l'excès
    de coût sur l'alpha proxy.

    Intégration dans allocator.py (step 4.5, après le FDS gate, avant les caps) ::

        from hyperstat.execution import execution_cost_adjustment

        w_pd = pd.Series(weights)
        w_pd = execution_cost_adjustment(w_pd, df_candles_latest)
        weights = w_pd.to_dict()

    Args:
        signal_weights : pd.Series indexée par symbole, valeurs ∈ [-1, 1].
        df_candles     : {symbol: DataFrame récent} avec colonnes OHLCV.
        cfg            : ExecutionConfig (optionnel, utilise les défauts sinon).

    Returns:
        pd.Series ajustée (mêmes index et signe, amplitudes réduites si nécessaire).
    """
    cfg = cfg or ExecutionConfig()
    adjusted = signal_weights.copy().astype(float)

    for sym in signal_weights.index:
        w = float(signal_weights[sym])
        if abs(w) < 1e-6:
            continue
        df = df_candles.get(sym)
        if df is None or df.empty or len(df) < cfg.vwap_window:
            continue

        vwap_val = float(calculate_vwap(df, cfg.vwap_window).iloc[-1])
        cp = float(df["close"].iloc[-1])

        if not (np.isfinite(vwap_val) and np.isfinite(cp) and vwap_val > 0):
            continue

        # Volatilité estimée via high-low (proxy Parkinson)
        tail = df.iloc[-5:]
        rv_frac = float((np.log(tail["high"] / tail["low"])).abs().mean())
        rv_pct = rv_frac * 100.0

        # Coût total estimé (cohérent avec backtest/costs.py)
        slip_bps = 8.0 + 10.0 * rv_pct
        vwap_dev_bps = abs(cp - vwap_val) / vwap_val * 10_000.0
        cost_bps = slip_bps + 0.3 * vwap_dev_bps

        # Alpha proxy : signal de magnitude 1.0 → ~50 bps d'alpha attendu
        alpha_bps = abs(w) * 50.0

        if cost_bps > alpha_bps:
            excess = cost_bps - alpha_bps
            scale = max(0.0, 1.0 - excess / max(alpha_bps, 1e-6))
            adjusted[sym] = w * scale

    return adjusted
