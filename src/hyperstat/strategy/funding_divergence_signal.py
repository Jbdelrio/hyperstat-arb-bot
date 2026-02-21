"""
FundingDivergenceSignal (FDS) — Alpha original pour HyperStat
==============================================================

Insight : Sur Hyperliquid, les traders on-chain ont un biais structurel long,
ce qui crée une asymétrie persistante dans le funding rate.

L'alpha vient de combiner 3 dimensions que la littérature ne croise pas :
  1. Z-score cross-sectionnel du funding (carry normalisé comme les returns)
  2. Désalignement funding/return  → un coin dont le prix monte mais le funding
     baisse EST en tension — signal prédictif de retournement
  3. Vélocité du funding (dérivée de l'EWMA) → un funding qui accélère vers le haut
     précède un squeeze puis une correction

Le FDS sert de *gate de confiance* multiplicatif sur le signal MR existant
(approche Okasová 2026 adaptée au contexte cross-sectionnel altcoin Hyperliquid).

Deux interfaces disponibles
----------------------------
  FundingDivergenceSignal      — version batch pandas (backtest / calibration offline)
  FundingDivergenceSignalLive  — version step-by-step (live / paper / allocateur)

Usage allocateur (step-by-step) :
    fds = FundingDivergenceSignalLive(FDSConfig())
    # dans allocate(), entre step 3 et step 4 :
    fds_scores = fds.update_and_compute(signal.zscores, funding_rates)
    w = fds.apply_gate(w, fds_scores)
    w = neutralize_weights(w, ...)
    w = normalize_to_gross(w, ...)

Usage offline / IC validation :
    fds_batch = FundingDivergenceSignal(FDSConfig())
    gate_df   = fds_batch.compute(returns_df, funding_df)   # (T, N) ∈ [-1, 1]
    diag      = FDSDiagnostics(fds_batch)
    ic_series = diag.signal_ic(returns_df, funding_df, forward_horizon=8)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────
# Config partagée
# ─────────────────────────────────────────────

@dataclass
class FDSConfig:
    """
    Paramètres du Funding Divergence Signal.

    Ordre de calibration par walk-forward (fixer les autres d'abord) :
        1. divergence_window  12–48 bars
        2. span_funding_fast   4–16 bars
        3. w_carry / w_divergence / w_velocity
        4. gate_scale
    """
    # EWMA spans (en nombre de barres, à aligner sur votre timeframe)
    span_funding_slow: int = 72   # tau_s — carry level (~ 6h en 5min)
    span_funding_fast: int = 8    # tau_f — vélocité   (~ 40min en 5min)

    # Fenêtre rolling pour corr(return_z, Δfunding) — composante divergence
    divergence_window: int = 24   # W bars

    # Saturation des sous-signaux avant pondération
    z_clip: float = 2.0           # clip carry cross-sectionnel
    velocity_clip: float = 2.0    # clip vélocité

    # Poids des 3 composantes (doivent sommer à 1)
    w_carry:      float = 0.35
    w_divergence: float = 0.40
    w_velocity:   float = 0.25

    # Intensité du gate multiplicatif α_fds ∈ [0, 1]
    # 0 → FDS sans effet ; 1 → peut zéroter un poids
    gate_scale: float = 0.6

    # Nombre minimum de périodes avant d'activer le signal (cold-start)
    min_obs: int = 48

    eps: float = 1e-12


# ─────────────────────────────────────────────
# Version batch — pandas (backtest / offline)
# ─────────────────────────────────────────────

class FundingDivergenceSignal:
    """
    Version vectorisée pandas du FDS — pour backtest et calibration offline.

    Entrées :
        returns_df  : pd.DataFrame (T, N), log-returns sur horizon H
        funding_df  : pd.DataFrame (T, N), funding rate brut (ex: 0.0001)

    Sortie de compute() :
        gate_df : pd.DataFrame (T, N) ∈ [-1, 1]
        Utilisation : w_final = w_stat * (1 + cfg.gate_scale * gate_df)
    """

    def __init__(self, config: Optional[FDSConfig] = None) -> None:
        self.cfg = config or FDSConfig()

    # ── Composante 1 : Carry cross-sectionnel ────────────────────────────────

    def _carry_signal(self, funding_df) -> object:  # pd.DataFrame
        """Z-score cross-sectionnel du funding EWMA slow. Contrarian : high → short."""
        f_slow = funding_df.ewm(span=self.cfg.span_funding_slow,
                                min_periods=self.cfg.span_funding_slow // 2).mean()
        med = f_slow.median(axis=1)
        mad = (f_slow.subtract(med, axis=0)).abs().median(axis=1).replace(0, np.nan)
        z = f_slow.subtract(med, axis=0).divide(mad + self.cfg.eps, axis=0)
        return (-z).clip(-self.cfg.z_clip, self.cfg.z_clip)

    # ── Composante 2 : Désalignement funding/return ───────────────────────────

    def _divergence_signal(self, returns_df, funding_df) -> object:  # pd.DataFrame
        """
        Tension entre direction des prix et vélocité du funding.

        corr ≈ +1 → alignés, tension faible, pas de signal.
        corr ≈  0 → désaccord maximal → signal fort.

        Signe : si prix monte (r > 0) et funding baisse (Δf < 0)
                → signal contrarian renforcé (la hausse n'est pas crédible).
        """
        f_fast = funding_df.ewm(span=self.cfg.span_funding_fast,
                                min_periods=self.cfg.span_funding_fast // 2).mean()
        delta_f = f_fast.diff()  # Δf = vélocité brute du funding

        # Corrélation rolling colonne par colonne
        import pandas as pd
        corr = pd.DataFrame(index=returns_df.index,
                            columns=returns_df.columns, dtype=float)
        w = self.cfg.divergence_window
        for col in returns_df.columns:
            corr[col] = returns_df[col].rolling(w, min_periods=w // 2).corr(delta_f[col])

        tension = 1.0 - corr.abs()  # ∈ [0, 1]

        # sign = -sgn(return_z) * sgn(Δf)
        sign_r = np.sign(returns_df)
        sign_f = np.sign(delta_f)
        sign = -(sign_r * sign_f)

        return (sign * tension).clip(-1, 1)

    # ── Composante 3 : Vélocité du funding ───────────────────────────────────

    def _velocity_signal(self, funding_df) -> object:  # pd.DataFrame
        """
        Ratio (fast − slow) / |slow| — mesure l'accélération du funding.
        Funding qui accélère → overcrowding des longs → signal short contrarian.
        """
        f_slow = funding_df.ewm(span=self.cfg.span_funding_slow,
                                min_periods=self.cfg.span_funding_slow // 2).mean()
        f_fast = funding_df.ewm(span=self.cfg.span_funding_fast,
                                min_periods=self.cfg.span_funding_fast // 2).mean()
        ratio = (f_fast - f_slow) / (f_slow.abs() + self.cfg.eps)
        ratio = ratio.clip(-self.cfg.velocity_clip, self.cfg.velocity_clip)
        return -(ratio / self.cfg.velocity_clip)  # contrarian

    # ── Assemblage FDS ────────────────────────────────────────────────────────

    def compute(self, returns_df, funding_df) -> object:  # pd.DataFrame
        """
        Calcule le gate FDS final (T, N) ∈ [-1, 1].
        À utiliser : w_final = w_stat * (1 + cfg.gate_scale * gate)
        """
        assert returns_df.shape == funding_df.shape
        assert (returns_df.index == funding_df.index).all()

        cfg = self.cfg
        s_c = self._carry_signal(funding_df)
        s_d = self._divergence_signal(returns_df, funding_df)
        s_v = self._velocity_signal(funding_df)

        fds = cfg.w_carry * s_c + cfg.w_divergence * s_d + cfg.w_velocity * s_v

        # Normalisation cross-sectionnel finale
        med = fds.median(axis=1)
        mad = (fds.subtract(med, axis=0)).abs().median(axis=1).replace(0, np.nan)
        fds_z = fds.subtract(med, axis=0).divide(2.0 * mad + cfg.eps, axis=0)
        fds_z = fds_z.clip(-1, 1)

        # Masque cold-start
        valid = funding_df.expanding().count() >= cfg.min_obs
        fds_z = fds_z.where(valid, 0.0)

        return fds_z.fillna(0.0)

    def apply_to_weights(self, w_stat, returns_df, funding_df) -> object:  # pd.DataFrame
        """
        Applique le gate sur les poids existants.
            w_final = w_stat * (1 + gate_scale * gate)
        """
        gate = self.compute(returns_df, funding_df)
        return w_stat * (1.0 + self.cfg.gate_scale * gate)


# ─────────────────────────────────────────────
# Version live — step-by-step (allocateur)
# ─────────────────────────────────────────────

class _SymbolFDSState:
    """État mutable par symbole pour la version step-by-step."""

    __slots__ = ("ewma_fast", "ewma_slow", "prev_ewma_fast", "corr_hist")

    def __init__(self, corr_window: int) -> None:
        self.ewma_fast: float = float("nan")
        self.ewma_slow: float = float("nan")
        self.prev_ewma_fast: float = float("nan")
        # Historique roulant de (return_z, delta_f_fast) pour la corrélation
        self.corr_hist: Deque[Tuple[float, float]] = deque(maxlen=corr_window)


class FundingDivergenceSignalLive:
    """
    Version step-by-step du FDS — pour le pipeline live / paper / allocateur.

    Implémente exactement les mêmes formules que FundingDivergenceSignal (batch)
    mais maintient un état interne par symbole, mis à jour à chaque pas de temps.

    Appelée dans PortfolioAllocator.allocate() entre step 3 et step 4 :
        fds_scores = self.fds.update_and_compute(signal.zscores, funding_rates)
        w = self.fds.apply_gate(w, fds_scores)
        # Re-neutralisation obligatoire après le gate (il casse la neutralité)
        w = neutralize_weights(w, ...)
        w = normalize_to_gross(w, ...)

    Propriétés du gate w_final_i = w_stat_i * (1 + gate_scale * FDS_i) :
      - FDS_i = 0  → identité, poids inchangé
      - FDS_i > 0  → amplifie |w_i| (confiance haute)
      - FDS_i < 0  → atténue |w_i| (confiance basse), ne renverse jamais le signe
      - FDS_i = -1 avec gate_scale=1 → annule le poids
    """

    def __init__(self, cfg: FDSConfig) -> None:
        self.cfg = cfg
        self._alpha_fast = 2.0 / (cfg.span_funding_fast + 1.0)
        self._alpha_slow = 2.0 / (cfg.span_funding_slow + 1.0)
        self._state: Dict[str, _SymbolFDSState] = {}
        self._bar_count: Dict[str, int] = {}

    def _ensure(self, symbol: str) -> _SymbolFDSState:
        if symbol not in self._state:
            self._state[symbol] = _SymbolFDSState(self.cfg.divergence_window)
            self._bar_count[symbol] = 0
        return self._state[symbol]

    def update_and_compute(
        self,
        zscores: Dict[str, float],
        funding_rates: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Met à jour l'état interne et retourne les scores FDS ∈ [-1, 1].

        Args:
            zscores:       {symbol: z-score cross-sectionnel} depuis StatArbStrategy.
                           z > 0 = le coin a surperformé sa médiane de bucket.
            funding_rates: {symbol: funding rate brut} du pas de temps courant.

        Returns:
            {symbol: fds_score ∈ [-1, 1]}
        """
        cfg = self.cfg
        af = self._alpha_fast
        as_ = self._alpha_slow

        # ── 1. Mise à jour EWMA et historique de corrélation ─────────────────
        for s, f in funding_rates.items():
            if not np.isfinite(f):
                continue
            st = self._ensure(s)
            self._bar_count[s] = self._bar_count.get(s, 0) + 1

            # Fast EWMA — on sauvegarde t-1 avant mise à jour (pour Δf)
            if np.isfinite(st.ewma_fast):
                st.prev_ewma_fast = st.ewma_fast
                st.ewma_fast = af * f + (1.0 - af) * st.ewma_fast
            else:
                st.ewma_fast = f

            # Slow EWMA
            if np.isfinite(st.ewma_slow):
                st.ewma_slow = as_ * f + (1.0 - as_) * st.ewma_slow
            else:
                st.ewma_slow = f

            # Vélocité funding : Δf = ewma_fast_t − ewma_fast_{t-1}
            delta_f = (
                st.ewma_fast - st.prev_ewma_fast
                if np.isfinite(st.prev_ewma_fast) else 0.0
            )

            # Historique (z_score, Δf) pour corrélation rolling
            z = zscores.get(s, 0.0)
            if np.isfinite(z) and np.isfinite(delta_f):
                st.corr_hist.append((z, delta_f))

        # ── 2. Composante carry — normalisation cross-sectionnel ──────────────
        slow_ewmas: Dict[str, float] = {
            s: st.ewma_slow
            for s, st in self._state.items()
            if np.isfinite(st.ewma_slow)
        }

        s_carry: Dict[str, float] = {}
        if len(slow_ewmas) >= 4:
            vals = np.asarray(list(slow_ewmas.values()), dtype=float)
            med = float(np.nanmedian(vals))
            # MAD (non scalé pour rester cohérent avec _zscore_cross pandas)
            m_raw = float(np.nanmedian(np.abs(vals - med)))
            m = m_raw if m_raw > cfg.eps else cfg.eps
            for s, v in slow_ewmas.items():
                z = (v - med) / (m + cfg.eps)
                # Contrarian : funding élevé → short → signal négatif pour les longs
                s_carry[s] = float(np.clip(-z, -cfg.z_clip, cfg.z_clip))

        # ── 3. Composantes divergence & vélocité — par symbole ────────────────
        s_div: Dict[str, float] = {}
        s_vel: Dict[str, float] = {}
        min_corr_obs = max(4, cfg.divergence_window // 4)

        for s, st in self._state.items():
            # Filtre cold-start
            if self._bar_count.get(s, 0) < cfg.min_obs:
                continue

            # Vélocité : (fast − slow) / |slow|
            if np.isfinite(st.ewma_fast) and np.isfinite(st.ewma_slow):
                denom = abs(st.ewma_slow) + cfg.eps
                v = (st.ewma_fast - st.ewma_slow) / denom
                s_vel[s] = float(-np.clip(v / cfg.velocity_clip, -1.0, 1.0))

            # Divergence : corrélation rolling entre z_return et Δf_fast
            if len(st.corr_hist) >= min_corr_obs:
                hist = np.asarray(st.corr_hist, dtype=float)  # (N, 2)
                ret_z = hist[:, 0]
                delta_f = hist[:, 1]

                if np.std(ret_z) > cfg.eps and np.std(delta_f) > cfg.eps:
                    rho = float(np.corrcoef(ret_z, delta_f)[0, 1])
                    if np.isfinite(rho):
                        tension = 1.0 - abs(rho)
                        lr, lf = float(ret_z[-1]), float(delta_f[-1])
                        sign = float(-np.sign(lr) * np.sign(lf)) if (lr != 0 and lf != 0) else 0.0
                        s_div[s] = float(np.clip(sign * tension, -1.0, 1.0))

        # ── 4. Assemblage et normalisation cross-sectionnel finale ────────────
        all_syms = set(s_carry) | set(s_div) | set(s_vel)
        if not all_syms:
            return {}

        raw: Dict[str, float] = {
            s: (
                cfg.w_carry * s_carry.get(s, 0.0)
                + cfg.w_divergence * s_div.get(s, 0.0)
                + cfg.w_velocity * s_vel.get(s, 0.0)
            )
            for s in all_syms
        }

        vals = np.asarray(list(raw.values()), dtype=float)
        med = float(np.nanmedian(vals))
        m_raw = float(np.nanmedian(np.abs(vals - med)))
        m = m_raw if m_raw > cfg.eps else cfg.eps

        # Dispersion nulle → pas de signal cross-sectionnel
        if m < cfg.eps:
            return {s: 0.0 for s in all_syms}

        return {
            s: float(np.clip((r - med) / (2.0 * m + cfg.eps), -1.0, 1.0))
            for s, r in raw.items()
        }

    def apply_gate(
        self,
        weights: Dict[str, float],
        fds_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Applique le gate multiplicatif de confiance :
            w_final_i = w_stat_i * (1 + gate_scale * FDS_i)

        IMPORTANT : Appeler neutralize_weights() et normalize_to_gross()
                    immédiatement après, car le gate casse la neutralité.
        """
        alpha = self.cfg.gate_scale
        return {
            s: float(w * (1.0 + alpha * fds_scores.get(s, 0.0)))
            for s, w in weights.items()
        }


# ─────────────────────────────────────────────
# Diagnostics offline
# ─────────────────────────────────────────────

class FDSDiagnostics:
    """
    Outils de validation et calibration du FDS sur données historiques.

    Workflow recommandé sur tes vraies données Hyperliquid :
        fds   = FundingDivergenceSignal(FDSConfig())
        diag  = FDSDiagnostics(fds)
        ic    = diag.signal_ic(returns_df, funding_df, forward_horizon=8)
        print(f"IC moyen: {ic.mean():.4f}  t-stat: {ic.mean()/ic.std()*len(ic)**0.5:.2f}")
        # IC > 0.03 et t-stat > 2 → signal prédictif → calibrer les poids

    Ordre de calibration :
        1. corr_window / span_fast  (impact le plus fort)
        2. w_carry / w_divergence / w_velocity
        3. gate_scale
    """

    def __init__(self, fds: FundingDivergenceSignal) -> None:
        self.fds = fds

    def component_breakdown(self, returns_df, funding_df) -> object:  # pd.DataFrame
        """Retourne les 3 composantes + FDS final, moyennées cross-section."""
        import pandas as pd
        s_c = self.fds._carry_signal(funding_df)
        s_d = self.fds._divergence_signal(returns_df, funding_df)
        s_v = self.fds._velocity_signal(funding_df)
        gate = self.fds.compute(returns_df, funding_df)

        return pd.DataFrame({
            "carry_mean":      s_c.mean(axis=1),
            "divergence_mean": s_d.mean(axis=1),
            "velocity_mean":   s_v.mean(axis=1),
            "fds_gate_mean":   gate.mean(axis=1),
        })

    def signal_ic(self, returns_df, funding_df, forward_horizon: int = 12) -> object:  # pd.Series
        """
        IC de Spearman entre FDS_t et les returns forward r_{t+H}.
        IC > 0.03 avec t-stat > 2 indique un signal prédictif valide.
        forward_horizon en barres (ex: 12 = 1h si données en 5min).
        """
        import pandas as pd
        from scipy.stats import spearmanr

        gate = self.fds.compute(returns_df, funding_df)
        fwd = returns_df.shift(-forward_horizon)

        ics = []
        for t in gate.index:
            g = gate.loc[t]
            r = fwd.loc[t]
            mask = g.notna() & r.notna()
            if mask.sum() < 5:
                ics.append(float("nan"))
            else:
                ic, _ = spearmanr(g[mask].values, r[mask].values)
                ics.append(float(ic))

        return pd.Series(ics, index=gate.index, name=f"IC_fds_fwd{forward_horizon}")

    def turnover_impact(self, w_stat, returns_df, funding_df) -> dict:
        """Compare le turnover avant/après gate. Ratio ≈ 1 est idéal."""
        w_final = self.fds.apply_to_weights(w_stat, returns_df, funding_df)

        def _to(w) -> float:
            return float(w.diff().abs().sum(axis=1).mean())

        to_b = _to(w_stat)
        to_a = _to(w_final)
        return {
            "turnover_before": round(to_b, 4),
            "turnover_after":  round(to_a, 4),
            "ratio":           round(to_a / (to_b + 1e-9), 3),
        }


# ─────────────────────────────────────────────
# Test / démo rapide
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd

    np.random.seed(42)
    T, N = 500, 20
    coins = [f"COIN{i}" for i in range(N)]
    idx = pd.date_range("2024-01-01", periods=T, freq="1h")

    returns_df = pd.DataFrame(np.random.randn(T, N) * 0.01, index=idx, columns=coins)
    funding_df = pd.DataFrame(
        np.random.randn(T, N) * 0.0002 + 0.0001 + returns_df.values * 0.3,
        index=idx, columns=coins,
    )

    cfg = FDSConfig(gate_scale=0.6)

    # --- Batch (backtest) ---
    fds_batch = FundingDivergenceSignal(cfg)
    gate = fds_batch.compute(returns_df, funding_df)
    print("=" * 55)
    print("FundingDivergenceSignal — Batch (backtest)")
    print(f"  Gate shape : {gate.shape}")
    print(f"  Gate mean  : {gate.mean().mean():.4f}")
    print(f"  Gate std   : {gate.std().mean():.4f}")

    diag = FDSDiagnostics(fds_batch)
    ic = diag.signal_ic(returns_df, funding_df, forward_horizon=12)
    print(f"  IC fwd-12  : mean={ic.dropna().mean():.4f}  (synthétique ≈ 0)")

    to = diag.turnover_impact(
        pd.DataFrame(np.random.randn(T, N) * 0.05, index=idx, columns=coins),
        returns_df, funding_df,
    )
    print(f"  Turnover ratio : {to['ratio']}x  (bon si ≈ 1)")

    # --- Live (step-by-step) ---
    fds_live = FundingDivergenceSignalLive(cfg)
    last_scores: Dict[str, float] = {}
    for i in range(T):
        zs = {c: float(returns_df.iloc[i][c]) for c in coins}
        fs = {c: float(funding_df.iloc[i][c]) for c in coins}
        last_scores = fds_live.update_and_compute(zs, fs)

    print("\nFundingDivergenceSignalLive — Step-by-step")
    valid = [v for v in last_scores.values() if np.isfinite(v)]
    print(f"  Symboles actifs : {len(valid)}/{N}")
    if valid:
        print(f"  Score mean  : {np.mean(valid):.4f}")
        print(f"  Score range : [{min(valid):.3f}, {max(valid):.3f}]")

    print("\n✅ FDS opérationnel")
