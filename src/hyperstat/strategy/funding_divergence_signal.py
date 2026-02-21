"""
FundingDivergenceSignal (FDS) — Alpha original pour HyperStat
==============================================================

Insight : Sur Hyperliquid, les traders on-chain ont un biais structurel long,
ce qui crée une asymétrie persistante dans le funding rate.

L'alpha vient de combiner 3 dimensions que la littérature ne croise pas :
  1. Z-score cross-sectionnel du funding (ton overlay normalisé comme les returns)
  2. Désalignement funding/return  → un coin dont le prix monte mais le funding
     baisse EST en tension — signal prédictif de retournement
  3. Vélocité du funding (dérivée de l'EWMA) → un funding qui accélère vers le haut
     précède un squeeze puis une correction

Le FDS résultant sert de *gate de confiance* multiplicatif sur ton signal MR existant
(exactement l'approche Okasová 2026 adaptée au contexte cross-sectionnel altcoin).

Usage :
    fds = FundingDivergenceSignal()
    gate = fds.compute(returns_df, funding_df)          # shape (T, N) ∈ [-1, 1]
    w_final = w_stat * (1 + alpha_fds * gate)           # combine avec signal existant
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class FDSConfig:
    # EWMA spans (en nombre de périodes)
    span_funding_slow: int = 72       # ~3 jours en 1h — trend long
    span_funding_fast: int = 8        # ~8h  — réagit vite
    span_return: int = 12             # horizon retour (H = 1h en 5m, ou 12 en 1h)

    # Clip du z-score pour robustesse
    z_clip: float = 3.0

    # Seuil de désalignement : |corr_rolling| < seuil → tension maximale
    divergence_window: int = 24       # fenêtre rolling pour corr(return, funding)

    # Vélocité : ratio fast/slow
    velocity_clip: float = 2.0        # saturation du signal vélocité

    # Poids des 3 composantes dans le FDS final
    w_carry:      float = 0.35        # composante carry cross-sectionnel
    w_divergence: float = 0.40        # composante désalignement (la plus originale)
    w_velocity:   float = 0.25        # composante vélocité

    # Gate final
    gate_scale:   float = 0.6         # intensité du gate (1.0 = remplace, <1 = additionnel)
    min_obs:      int   = 48          # nb min de périodes avant d'activer le signal


# ─────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────

def _zscore_cross(df: pd.DataFrame, clip: float) -> pd.DataFrame:
    """Z-score cross-sectionnel robuste (median/MAD) sur chaque ligne (timestamp)."""
    med = df.median(axis=1)
    mad = (df.subtract(med, axis=0)).abs().median(axis=1).replace(0, np.nan)
    z = df.subtract(med, axis=0).divide(mad + 1e-9, axis=0)
    return z.clip(-clip, clip)


def _ewma(df: pd.DataFrame, span: int) -> pd.DataFrame:
    return df.ewm(span=span, min_periods=span // 2).mean()


def _rolling_corr_rowwise(a: pd.DataFrame, b: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Corrélation rolling entre deux DataFrames de même shape, calculée colonne par colonne.
    Retourne un DataFrame de même shape.
    """
    result = pd.DataFrame(index=a.index, columns=a.columns, dtype=float)
    for col in a.columns:
        result[col] = a[col].rolling(window, min_periods=window // 2).corr(b[col])
    return result


# ─────────────────────────────────────────────
# Signal principal
# ─────────────────────────────────────────────

class FundingDivergenceSignal:
    """
    Calcule le Funding Divergence Score (FDS).

    Paramètres d'entrée :
        returns_df  : pd.DataFrame, shape (T, N), log-returns sur horizon H
                      index = DatetimeIndex, columns = coin symbols
        funding_df  : pd.DataFrame, même shape, funding rate instantané (ex: 0.0001 = 0.01%)
                      Doit être aligné sur le même index que returns_df.

    Sortie :
        gate : pd.DataFrame (T, N) ∈ [-1, 1] — multiplicateur sur le signal MR
               > 0 : renforce le signal contrarian
               < 0 : atténue ou inverse (rare, seulement en divergence extrême)
    """

    def __init__(self, config: Optional[FDSConfig] = None):
        self.cfg = config or FDSConfig()

    # ── Composante 1 : Carry cross-sectionnel ──────────────────────────────────

    def _carry_signal(self, funding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score cross-sectionnel du funding EWMA slow.
        Un funding très positif (foule de longs) → signal short = +1 (contrarian).
        On retourne -z car funding positif → longs payent → short capte le carry.
        """
        f_slow = _ewma(funding_df, self.cfg.span_funding_slow)
        z = _zscore_cross(f_slow, self.cfg.z_clip)
        # funding élevé cross-section → être short → signal contrarian = -z
        # (cohérent avec ton signal MR : s = -clip(z_return))
        return -z

    # ── Composante 2 : Désalignement funding/return ────────────────────────────

    def _divergence_signal(
        self, returns_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Détecte la tension entre direction prix et direction funding.

        Logique :
          - corr_rolling(return, funding) ≈ +1 : alignés → pas de tension
          - corr_rolling(return, funding) ≈ -1 : désalignés → forte tension
          - Tension = 1 - |corr|  ∈ [0, 1]

        Signe du signal : si funding monte mais prix baisse → short aggravé
                          si funding baisse mais prix monte → signal contrarian renforcé

        Le signal final encode DIRECTION de la divergence × intensité.
        """
        f_fast = _ewma(funding_df, self.cfg.span_funding_fast)
        f_direction = f_fast.diff()         # variation du funding (vélocité brute)
        r = returns_df

        # Corrélation rolling : mesure l'alignement local
        corr = _rolling_corr_rowwise(r, f_direction, self.cfg.divergence_window)

        # Tension = désaccord entre les deux
        tension = 1.0 - corr.abs()          # ∈ [0, 1], max quand corr = 0

        # Signe : si prix monte (r > 0) et funding baisse (f_direction < 0)
        # → marché sous-priced → signal long (≈ +1 contrarian)
        # Si prix monte et funding monte aussi → foule long → signal short
        sign = np.sign(r.multiply(-np.sign(f_direction)))  # -1 si alignés, +1 si divergent

        signal = sign * tension
        return signal.clip(-1, 1)

    # ── Composante 3 : Vélocité du funding ────────────────────────────────────

    def _velocity_signal(self, funding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ratio fast/slow du funding → mesure l'accélération.

        Si fast >> slow : funding s'accélère (bulle longue en formation)
        → signal contrarian fort (short)

        Si fast << slow : déaccélération / retournement possible
        → signal long (réversion à venir)
        """
        f_slow = _ewma(funding_df, self.cfg.span_funding_slow)
        f_fast = _ewma(funding_df, self.cfg.span_funding_fast)

        # Ratio centré : (fast - slow) / (|slow| + eps)
        ratio = (f_fast - f_slow) / (f_slow.abs() + 1e-8)
        ratio = ratio.clip(-self.cfg.velocity_clip, self.cfg.velocity_clip)
        ratio_norm = ratio / self.cfg.velocity_clip   # ∈ [-1, 1]

        # Funding qui accélère vers le haut → longs overcrowded → signal short (=contrarian)
        return -ratio_norm

    # ── Assemblage FDS ─────────────────────────────────────────────────────────

    def compute(
        self, returns_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcule le gate FDS final.

        Returns:
            gate_df : pd.DataFrame (T, N) ∈ [-1, 1]
                     À utiliser comme : w_final = w_stat * (1 + cfg.gate_scale * gate)
        """
        assert returns_df.shape == funding_df.shape, "returns et funding doivent avoir le même shape"
        assert (returns_df.index == funding_df.index).all(), "index non alignés"

        cfg = self.cfg

        s_carry      = self._carry_signal(funding_df)
        s_divergence = self._divergence_signal(returns_df, funding_df)
        s_velocity   = self._velocity_signal(funding_df)

        # Combine pondéré
        fds = (
            cfg.w_carry      * s_carry      +
            cfg.w_divergence * s_divergence +
            cfg.w_velocity   * s_velocity
        )

        # Normalisation finale cross-sectionnel
        fds_z = _zscore_cross(fds, clip=2.0) / 2.0   # ∈ [-1, 1]

        # Masque cold-start
        valid_mask = funding_df.expanding().count() >= cfg.min_obs
        fds_z = fds_z.where(valid_mask, 0.0)

        return fds_z.fillna(0.0)

    def apply_to_weights(
        self,
        w_stat: pd.DataFrame,
        returns_df: pd.DataFrame,
        funding_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Applique le gate FDS sur les poids existants.

        Formule :
            w_final[i,t] = w_stat[i,t] * (1 + gate_scale * fds[i,t])

        Si fds > 0 : renforce le signal contrarian (plus confiant)
        Si fds < 0 : atténue (moins confiant, ou signal contradictoire)

        Note : on ne neutralise pas ici (c'est déjà fait dans ton allocator).
        """
        gate = self.compute(returns_df, funding_df)
        multiplier = 1.0 + self.cfg.gate_scale * gate
        w_final = w_stat * multiplier
        return w_final


# ─────────────────────────────────────────────
# Diagnostic / analyse offline
# ─────────────────────────────────────────────

class FDSDiagnostics:
    """
    Outils pour comprendre et debugger le FDS sur des données historiques.
    Utile pour calibrer les poids w_carry / w_divergence / w_velocity.
    """

    def __init__(self, fds: FundingDivergenceSignal):
        self.fds = fds

    def component_breakdown(
        self, returns_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Retourne les 3 composantes + FDS final pour inspection."""
        s_carry      = self.fds._carry_signal(funding_df)
        s_divergence = self.fds._divergence_signal(returns_df, funding_df)
        s_velocity   = self.fds._velocity_signal(funding_df)
        gate         = self.fds.compute(returns_df, funding_df)

        # Moyenne cross-sectionnel à chaque timestep pour visualisation
        return pd.DataFrame({
            "carry_mean":      s_carry.mean(axis=1),
            "divergence_mean": s_divergence.mean(axis=1),
            "velocity_mean":   s_velocity.mean(axis=1),
            "fds_gate_mean":   gate.mean(axis=1),
        })

    def signal_ic(
        self,
        returns_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        forward_horizon: int = 12,
    ) -> pd.Series:
        """
        Calcule l'IC (Information Coefficient = corr de Spearman) entre le FDS
        et les returns forward, pour valider que le signal prédit bien la direction.

        forward_horizon : en nombre de périodes (ex: 12 = 12h si données 1h)
        """
        gate = self.fds.compute(returns_df, funding_df)
        fwd_returns = returns_df.shift(-forward_horizon)

        ics = []
        for t in gate.index:
            g = gate.loc[t]
            r = fwd_returns.loc[t]
            mask = g.notna() & r.notna()
            if mask.sum() < 5:
                ics.append(np.nan)
            else:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(g[mask], r[mask])
                ics.append(ic)

        return pd.Series(ics, index=gate.index, name=f"IC_fds_fwd{forward_horizon}")

    def turnover_impact(
        self,
        w_stat: pd.DataFrame,
        returns_df: pd.DataFrame,
        funding_df: pd.DataFrame,
    ) -> dict:
        """
        Compare le turnover avant/après application du FDS gate.
        Un bon gate ne doit pas exploser le turnover.
        """
        w_final = self.fds.apply_to_weights(w_stat, returns_df, funding_df)

        def _avg_turnover(w: pd.DataFrame) -> float:
            return w.diff().abs().sum(axis=1).mean()

        to_before = _avg_turnover(w_stat)
        to_after  = _avg_turnover(w_final)

        return {
            "turnover_before": round(to_before, 4),
            "turnover_after":  round(to_after, 4),
            "ratio":           round(to_after / (to_before + 1e-9), 3),
        }


# ─────────────────────────────────────────────
# Test / démo rapide
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(42)
    T, N = 500, 20
    coins = [f"COIN{i}" for i in range(N)]
    idx = pd.date_range("2024-01-01", periods=T, freq="1h")

    # Données synthétiques
    returns_df = pd.DataFrame(
        np.random.randn(T, N) * 0.01, index=idx, columns=coins
    )
    # Funding avec une structure réaliste : mean positif (biais long on-chain)
    # + un peu de corrélation avec les returns + du bruit
    funding_base = np.random.randn(T, N) * 0.0002 + 0.0001
    funding_df = pd.DataFrame(
        funding_base + returns_df.values * 0.3,
        index=idx, columns=coins
    )

    # Créer le signal
    cfg = FDSConfig(gate_scale=0.6)
    fds = FundingDivergenceSignal(config=cfg)
    gate = fds.compute(returns_df, funding_df)

    print("=" * 55)
    print("FundingDivergenceSignal — Test rapide")
    print("=" * 55)
    print(f"\nShape gate     : {gate.shape}")
    print(f"Gate mean      : {gate.mean().mean():.4f}  (proche 0 attendu)")
    print(f"Gate std       : {gate.std().mean():.4f}")
    print(f"Gate min/max   : {gate.values.min():.3f} / {gate.values.max():.3f}")
    print(f"NaN count      : {gate.isna().sum().sum()}")

    # Diagnostics
    diag = FDSDiagnostics(fds)
    breakdown = diag.component_breakdown(returns_df, funding_df)
    print("\nComposantes (moyennes cross-section sur dernier timestep) :")
    print(breakdown.tail(5).round(4).to_string())

    ic_series = diag.signal_ic(returns_df, funding_df, forward_horizon=12)
    ic_mean = ic_series.dropna().mean()
    ic_std  = ic_series.dropna().std()
    print(f"\nIC forward-12h : mean={ic_mean:.4f}, std={ic_std:.4f}")
    print(f"  (sur données synthétiques — attend ~0 sans vrai alpha injecté)")

    # Turnover
    w_stat = pd.DataFrame(
        np.random.randn(T, N) * 0.05, index=idx, columns=coins
    )
    to = diag.turnover_impact(w_stat, returns_df, funding_df)
    print(f"\nTurnover avant gate : {to['turnover_before']}")
    print(f"Turnover après gate : {to['turnover_after']}")
    print(f"Ratio turnover      : {to['ratio']}x")
    print("\n✅ FDS opérationnel — intégrez dans src/hyperstat/strategy/")
