"""
hyperstat.agents.sentiment_agent
==================================
SentimentAgent — Agrège Fear & Greed + news RSS + liquidations on-chain.
Produit un score de sentiment ∈ [-1, 1] utilisé par le SupervisorAgent.

Sources (toutes gratuites) :
    - Alternative.me Fear & Greed Index
    - RSS feeds (Coindesk, Cointelegraph)
    - Hyperliquid WebSocket (ratio liquidations long/short)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from hyperstat.agents.base_agent import BaseAgent, AgentSignal, AgentStatus
from hyperstat.agents.utils.fear_greed import FearGreedClient
from hyperstat.agents.utils.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """
    Agent de sentiment marché.

    Score final = 0.40 × Fear&Greed + 0.35 × news + 0.25 × on-chain liquidations

    Parameters
    ----------
    fg_weight     : poids Fear & Greed (défaut 0.40)
    news_weight   : poids news (défaut 0.35)
    onchain_weight: poids liquidations on-chain (défaut 0.25)
    news_lookback : fenêtre news en heures (défaut 4)
    cc_api_key    : clé CryptoCompare (optionnel, enrichit les news)
    """

    def __init__(
        self,
        fg_weight       : float = 0.40,
        news_weight     : float = 0.35,
        onchain_weight  : float = 0.25,
        news_lookback_h : int   = 4,
        cc_api_key      : Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name="SentimentAgent", **kwargs)
        self.fg_weight        = fg_weight
        self.news_weight      = news_weight
        self.onchain_weight   = onchain_weight
        self.news_lookback_h  = news_lookback_h

        self._fg_client    = FearGreedClient()
        self._news_fetcher = NewsFetcher(cc_api_key=cc_api_key)

        # État on-chain mis à jour via observe()
        self._liq_long_usd  : float = 0.0
        self._liq_short_usd : float = 0.0
        self._oi_total      : float = 0.0

        # Dernières valeurs calculées
        self._last_fg_score    : float = 0.0
        self._last_news_score  : float = 0.0
        self._last_onchain_score: float = 0.0

    # ── Interface BaseAgent ───────────────────────────────────────────────

    def warm_up(self, **kwargs) -> bool:
        """Warm-up : teste les connexions aux sources de données."""
        try:
            fg = self._fg_client.get_latest()
            logger.info(f"[SentimentAgent] Fear&Greed: {fg['value']} ({fg['value_classification']})")
            score = self._news_fetcher.get_sentiment_score(lookback_hours=24)
            logger.info(f"[SentimentAgent] News score 24h: {score}")
            self._set_active()
            return True
        except Exception as exc:
            logger.error(f"[SentimentAgent] warm_up échoué: {exc}")
            self._set_degraded(reason=str(exc))
            return False

    def observe(self, ts: datetime, data: Dict[str, Any]) -> None:
        """
        Reçoit les données on-chain du tick courant.

        data attendu :
            {
                "liq_long_usd"  : float,  # USD liquidés côté long
                "liq_short_usd" : float,  # USD liquidés côté short
                "oi_total"      : float,  # Open Interest total en USD
            }
        """
        self._liq_long_usd   = data.get("liq_long_usd", 0.0)
        self._liq_short_usd  = data.get("liq_short_usd", 0.0)
        self._oi_total       = data.get("oi_total", 0.0)

    def act(self, ts: datetime) -> AgentSignal:
        try:
            fg_score      = self._compute_fg_score()
            news_score    = self._compute_news_score()
            onchain_score = self._compute_onchain_score()

            self._last_fg_score     = fg_score
            self._last_news_score   = news_score
            self._last_onchain_score = onchain_score

            # Vérification de disponibilité des sources
            available_weight = 0.0
            composite = 0.0
            if fg_score is not None:
                composite      += self.fg_weight * fg_score
                available_weight += self.fg_weight
            if news_score is not None:
                composite       += self.news_weight * news_score
                available_weight += self.news_weight
            if onchain_score is not None:
                composite       += self.onchain_weight * onchain_score
                available_weight += self.onchain_weight

            if available_weight < 0.3:
                # Moins de 30% des sources disponibles → signal dégradé
                self._set_degraded(reason="sources insuffisantes")
                confidence = 0.2
            else:
                composite  = composite / available_weight
                confidence = min(1.0, available_weight * 0.8 + 0.2)
                if self.status != AgentStatus.ACTIVE:
                    self._set_active()

            return self._make_signal(
                ts          = ts,
                score       = float(np.clip(composite, -1.0, 1.0)),
                confidence  = confidence,
                regime_hint = self._fg_client.get_regime_signal(),
                fg_score    = round(fg_score or 0.0, 4),
                news_score  = round(news_score or 0.0, 4),
                onchain_score = round(onchain_score or 0.0, 4),
                fg_raw      = self._fg_client.get_latest().get("value", 50),
                available_w = round(available_weight, 2),
            )

        except Exception as exc:
            return self._handle_error(exc)

    # ── Calcul des composantes ────────────────────────────────────────────

    def _compute_fg_score(self) -> Optional[float]:
        """Fear & Greed → score ∈ [-1, 1]."""
        try:
            return self._fg_client.get_normalized_score()
        except Exception as exc:
            logger.debug(f"[SentimentAgent] FG score erreur: {exc}")
            return None

    def _compute_news_score(self) -> Optional[float]:
        """News RSS + CryptoCompare → score ∈ [-1, 1]."""
        try:
            return self._news_fetcher.get_sentiment_score(
                lookback_hours=self.news_lookback_h
            )
        except Exception as exc:
            logger.debug(f"[SentimentAgent] News score erreur: {exc}")
            return None

    def _compute_onchain_score(self) -> Optional[float]:
        """
        Score on-chain basé sur le ratio de liquidations.
        Logique :
            - Beaucoup de longs liquidés → pression vendeuse récente → bearish signal
            - Beaucoup de shorts liquidés → squeeze → bullish signal
        Score ∈ [-1, 1]
        """
        total_liq = self._liq_long_usd + self._liq_short_usd
        if total_liq < 1_000:   # Moins de $1k liquidé → pas de signal
            return 0.0
        # Ratio : > 0.5 = plus de shorts liquidés (bullish), < 0.5 = plus de longs (bearish)
        short_ratio = self._liq_short_usd / total_liq
        score = (short_ratio - 0.5) * 2.0   # ∈ [-1, 1]
        # Atténuer si OI faible (liquidations peu significatives)
        if self._oi_total > 0:
            liq_pct = total_liq / self._oi_total
            weight  = min(1.0, liq_pct * 100)   # significatif si liq > 1% de l'OI
            score  *= weight
        return float(np.clip(score, -1.0, 1.0))

    def get_status_dict(self) -> Dict[str, Any]:
        base = super().get_status_dict()
        base.update({
            "fg_score"      : round(self._last_fg_score, 4),
            "news_score"    : round(self._last_news_score, 4),
            "onchain_score" : round(self._last_onchain_score, 4),
            "fg_raw"        : self._fg_client.get_latest().get("value", "N/A"),
            "fg_label"      : self._fg_client.get_latest().get("value_classification", "N/A"),
        })
        return base
