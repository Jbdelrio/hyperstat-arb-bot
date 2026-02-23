"""
hyperstat.agents.regime_agent
===============================
RegimeAgent — Classifie le régime de marché courant.

Régimes :
    mean_reverting  : vol modérée, spreads stables → Q_t nominal
    trending        : momentum fort, spreads qui dérivent → Q_t réduit
    high_vol        : volatilité extrême BTC → Q_t = 0
    crisis          : liquidations massives → kill-switch global
    carry_favorable : funding élevé stable → boost overlay carry

Approche hybride :
    1. Règles déterministes pour les cas clairs (crisis, high_vol)
    2. Score composite pour trending vs mean_reverting
    3. LLM optionnel (Gemini Flash ou local) pour cas ambigus
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hyperstat.agents.base_agent import BaseAgent, AgentSignal, AgentStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeConfig:
    # Thresholds vol BTC
    vol_high_pct     : float = 90.0   # percentile vol → high_vol
    vol_crisis_pct   : float = 97.0   # percentile vol → crisis
    vol_window_days  : int   = 60     # fenêtre rolling percentiles

    # Thresholds liquidations
    liq_crisis_usd   : float = 50_000_000   # $50M liquidations → crisis
    liq_high_usd     : float = 10_000_000   # $10M → high stress

    # Momentum / trending
    trend_window     : int   = 48    # barres pour mesurer le momentum
    trend_threshold  : float = 1.5   # z-score momentum → trending

    # Half-life mean reversion
    halflife_mr_min  : float = 30.0  # minutes — en dessous = trop rapide
    halflife_mr_max  : float = 360.0 # minutes — au dessus = trending

    # Carry
    funding_high_pct : float = 70.0  # percentile funding → carry_favorable

    # LLM (optionnel)
    use_llm          : bool  = False
    llm_model        : str   = "gemini-flash"  # ou chemin local
    llm_api_key      : str   = ""

    # Mapping régime → Q_t
    regime_qt_map: Dict[str, float] = field(default_factory=lambda: {
        "mean_reverting" : 1.0,
        "trending"       : 0.3,
        "high_vol"       : 0.0,
        "crisis"         : 0.0,
        "carry_favorable": 1.0,
        "unknown"        : 0.5,
    })


# ─────────────────────────────────────────────────────────────────────────────
# REGIME AGENT
# ─────────────────────────────────────────────────────────────────────────────

class RegimeAgent(BaseAgent):
    """
    Agent de classification du régime de marché.

    Produit :
        - AgentSignal avec regime_hint = étiquette du régime
        - score ∈ [-1, 1] : -1 = très bearish/crisis, +1 = conditions favorables
        - metadata : Q_t recommandé, scores composantes
    """

    def __init__(self, cfg: Optional[RegimeConfig] = None, **kwargs):
        super().__init__(name="RegimeAgent", **kwargs)
        self.cfg  = cfg or RegimeConfig()

        # Buffer de données BTC pour les calculs de vol rolling
        self._btc_returns     : List[float] = []
        self._btc_vol_history : List[float] = []

        # État courant
        self._current_regime  : str   = "unknown"
        self._current_qt      : float = 0.5
        self._liq_total_usd   : float = 0.0
        self._funding_history : List[float] = []
        self._fg_signal       : str   = "neutral"

        # Composantes de score
        self._vol_score       : float = 0.0
        self._momentum_score  : float = 0.0
        self._liq_score       : float = 0.0
        self._funding_score   : float = 0.0

    # ── Interface BaseAgent ───────────────────────────────────────────────

    def warm_up(self, btc_returns: Optional[pd.Series] = None, **kwargs) -> bool:
        if btc_returns is not None:
            self._btc_returns = btc_returns.dropna().tolist()[-2000:]
            self._compute_vol_history()
        self._set_active()
        return True

    def observe(self, ts: datetime, data: Dict[str, Any]) -> None:
        """
        data attendu :
            btc_return   : float — log-return BTC du tick courant
            liq_total_usd: float — total liquidations USD
            avg_funding  : float — funding rate moyen de l'univers
            fg_signal    : str   — signal Fear & Greed ("extreme_fear", ...)
            momentum_zscore: float — z-score momentum cross-sectionnel
        """
        # BTC returns
        btc_ret = data.get("btc_return", 0.0)
        self._btc_returns.append(btc_ret)
        if len(self._btc_returns) > 10000:
            self._btc_returns.pop(0)

        # Mise à jour vol history (toutes les 12 barres = 1h)
        if len(self._btc_returns) % 12 == 0:
            self._compute_vol_history()

        # Liquidations
        self._liq_total_usd = data.get("liq_total_usd", 0.0)

        # Funding
        avg_funding = data.get("avg_funding", 0.0)
        self._funding_history.append(avg_funding)
        if len(self._funding_history) > 1000:
            self._funding_history.pop(0)

        # Fear & Greed
        self._fg_signal = data.get("fg_signal", "neutral")

        # Momentum
        self._last_momentum_zscore = data.get("momentum_zscore", 0.0)

    def act(self, ts: datetime) -> AgentSignal:
        try:
            regime = self._classify_regime()
            qt     = self.cfg.regime_qt_map.get(regime, 0.5)

            self._current_regime = regime
            self._current_qt     = qt

            # Score global : positif = conditions favorables
            score = self._regime_to_score(regime)

            # Confidence basée sur la clarté du régime
            confidence = self._compute_confidence(regime)

            return self._make_signal(
                ts          = ts,
                score       = score,
                confidence  = confidence,
                regime_hint = regime,
                qt          = qt,
                vol_score   = round(self._vol_score, 4),
                momentum_score = round(self._momentum_score, 4),
                liq_score   = round(self._liq_score, 4),
                funding_score  = round(self._funding_score, 4),
                fg_signal   = self._fg_signal,
            )

        except Exception as exc:
            return self._handle_error(exc)

    # ── Classification ────────────────────────────────────────────────────

    def _classify_regime(self) -> str:
        """
        Hiérarchie de classification :
        1. Crisis (règle dure)
        2. High Vol (règle dure)
        3. Carry Favorable (règle)
        4. Trending vs Mean Reverting (score composite)
        """
        # ── 1. Crisis ──
        if self._is_crisis():
            return "crisis"

        # ── 2. High Vol ──
        if self._is_high_vol():
            return "high_vol"

        # ── 3. Score composite ──
        self._vol_score      = self._compute_vol_score()
        self._momentum_score = self._compute_momentum_score()
        self._liq_score      = self._compute_liq_score()
        self._funding_score  = self._compute_funding_score()

        # Carry favorable : funding élevé + marché pas en crise
        if self._is_carry_favorable():
            return "carry_favorable"

        # Trending vs Mean Reverting
        composite = (
            0.35 * self._vol_score +
            0.40 * self._momentum_score +
            0.15 * self._liq_score +
            0.10 * self._funding_score
        )

        if composite > 0.4:
            return "trending"
        return "mean_reverting"

    def _is_crisis(self) -> bool:
        """Règle dure : liquidations massives OU vol BTC extrême."""
        if self._liq_total_usd > self.cfg.liq_crisis_usd:
            logger.warning(f"[RegimeAgent] CRISIS détecté: liq={self._liq_total_usd:.0f}$")
            return True
        if self._fg_signal == "extreme_fear":
            vol_pct = self._current_vol_percentile()
            if vol_pct > self.cfg.vol_crisis_pct:
                logger.warning(f"[RegimeAgent] CRISIS: extreme_fear + vol p{vol_pct:.0f}")
                return True
        return False

    def _is_high_vol(self) -> bool:
        vol_pct = self._current_vol_percentile()
        return vol_pct > self.cfg.vol_high_pct

    def _is_carry_favorable(self) -> bool:
        if len(self._funding_history) < 100:
            return False
        funding_arr = np.array(self._funding_history[-500:])
        current_avg = np.mean(self._funding_history[-10:])
        pct         = float(np.percentile(funding_arr, self.cfg.funding_high_pct))
        return current_avg > pct and current_avg > 0

    # ── Scores composantes ────────────────────────────────────────────────

    def _compute_vol_score(self) -> float:
        """0 = vol normale, 1 = vol très élevée."""
        pct = self._current_vol_percentile()
        return float(np.clip((pct - 50) / 50, 0.0, 1.0))

    def _compute_momentum_score(self) -> float:
        """0 = mean-reverting, 1 = trending fort."""
        mz = abs(getattr(self, "_last_momentum_zscore", 0.0))
        return float(np.clip(mz / self.cfg.trend_threshold, 0.0, 1.0))

    def _compute_liq_score(self) -> float:
        """0 = liquidations faibles, 1 = liquidations élevées."""
        return float(np.clip(
            self._liq_total_usd / self.cfg.liq_high_usd, 0.0, 1.0
        ))

    def _compute_funding_score(self) -> float:
        """Score funding : fort positif = carry favorable."""
        if not self._funding_history:
            return 0.0
        current = np.mean(self._funding_history[-10:])
        hist    = np.array(self._funding_history)
        return float(np.clip((current - hist.mean()) / (hist.std() + 1e-9), -1.0, 1.0))

    # ── Utilitaires ───────────────────────────────────────────────────────

    def _compute_vol_history(self):
        if len(self._btc_returns) < 12:
            return
        arr = np.array(self._btc_returns)
        # Vol réalisée 1h (12 barres)
        rv = np.std(arr[-12:]) * np.sqrt(12)
        self._btc_vol_history.append(rv)
        max_hist = self.cfg.vol_window_days * 24  # une valeur par heure
        if len(self._btc_vol_history) > max_hist:
            self._btc_vol_history.pop(0)

    def _current_vol_percentile(self) -> float:
        if len(self._btc_vol_history) < 10:
            return 50.0
        current = self._btc_vol_history[-1]
        return float(np.percentile(
            np.array(self._btc_vol_history) <= current, [0]
        )[0] * 100 + (
            np.mean(np.array(self._btc_vol_history) <= current) * 100
        )) / 2

    def _current_vol_percentile(self) -> float:
        if len(self._btc_vol_history) < 10:
            return 50.0
        arr     = np.array(self._btc_vol_history)
        current = arr[-1]
        return float(np.mean(arr <= current) * 100)

    @staticmethod
    def _regime_to_score(regime: str) -> float:
        """Convertit le régime en score ∈ [-1, 1] pour le SupervisorAgent."""
        mapping = {
            "mean_reverting" :  0.6,
            "carry_favorable":  0.8,
            "unknown"        :  0.0,
            "trending"       : -0.3,
            "high_vol"       : -0.7,
            "crisis"         : -1.0,
        }
        return mapping.get(regime, 0.0)

    @staticmethod
    def _compute_confidence(regime: str) -> float:
        """Confidence basée sur la clarté du régime."""
        high_confidence = {"crisis", "high_vol", "mean_reverting"}
        low_confidence  = {"unknown"}
        if regime in high_confidence:
            return 0.9
        if regime in low_confidence:
            return 0.3
        return 0.6

    # ── Dashboard ────────────────────────────────────────────────────────

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def current_qt(self) -> float:
        return self._current_qt

    def get_status_dict(self) -> Dict[str, Any]:
        base = super().get_status_dict()
        base.update({
            "current_regime" : self._current_regime,
            "current_qt"     : round(self._current_qt, 2),
            "vol_score"      : round(self._vol_score, 4),
            "momentum_score" : round(self._momentum_score, 4),
            "liq_score"      : round(self._liq_score, 4),
            "funding_score"  : round(self._funding_score, 4),
            "fg_signal"      : self._fg_signal,
        })
        return base
