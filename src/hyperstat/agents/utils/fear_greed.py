"""
hyperstat.agents.utils.fear_greed
==================================
Wrapper pour l'API Alternative.me Fear & Greed Index.
Complètement gratuit, sans clé API requise.
https://alternative.me/crypto/fear-and-greed-index/

Retourne un score normalisé ∈ [-1, 1] :
    -1 = Extreme Fear (vente panique)
    +1 = Extreme Greed (euphorie)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API_URL   = "https://api.alternative.me/fng/"
_CACHE_TTL = 3600   # 1 heure en secondes (le score change 1x/jour max)


class FearGreedClient:
    """
    Client Fear & Greed Index avec cache en mémoire.

    Usage
    -----
    >>> client = FearGreedClient()
    >>> score = client.get_normalized_score()  # float ∈ [-1, 1]
    >>> raw   = client.get_latest()            # dict complet
    """

    def __init__(self, timeout: int = 10):
        self.timeout    = timeout
        self._cache_val : Optional[dict] = None
        self._cache_ts  : float          = 0.0

    # ── API publique ──────────────────────────────────────────────────────

    def get_latest(self, force_refresh: bool = False) -> dict:
        """
        Retourne le dernier point Fear & Greed.

        Returns
        -------
        {
            "value"            : int (0-100),
            "value_classification": str,  # "Extreme Fear" | "Fear" | "Neutral" | ...
            "timestamp"        : datetime,
            "normalized_score" : float (-1 à 1),
        }
        """
        now = time.monotonic()
        if not force_refresh and self._cache_val and (now - self._cache_ts) < _CACHE_TTL:
            return self._cache_val

        try:
            resp = requests.get(
                _API_URL,
                params={"limit": 1, "format": "json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()["data"][0]

            result = {
                "value"               : int(data["value"]),
                "value_classification": data["value_classification"],
                "timestamp"           : datetime.utcfromtimestamp(int(data["timestamp"])),
                "normalized_score"    : self._normalize(int(data["value"])),
            }
            self._cache_val = result
            self._cache_ts  = now
            return result

        except Exception as exc:
            logger.warning(f"[FearGreed] Échec fetch: {exc}. Retourne cache ou 0.")
            return self._cache_val or {
                "value": 50,
                "value_classification": "Neutral",
                "timestamp": datetime.utcnow(),
                "normalized_score": 0.0,
            }

    def get_normalized_score(self) -> float:
        """Score normalisé ∈ [-1, 1]. -1 = peur extrême, +1 = euphorie."""
        return self.get_latest()["normalized_score"]

    def get_history(self, days: int = 30) -> list[dict]:
        """Retourne les N derniers jours de scores."""
        try:
            resp = requests.get(
                _API_URL,
                params={"limit": days, "format": "json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return [
                {
                    "value"            : int(d["value"]),
                    "classification"   : d["value_classification"],
                    "timestamp"        : datetime.utcfromtimestamp(int(d["timestamp"])),
                    "normalized_score" : self._normalize(int(d["value"])),
                }
                for d in resp.json()["data"]
            ]
        except Exception as exc:
            logger.error(f"[FearGreed] Erreur historique: {exc}")
            return []

    # ── Interne ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(value: int) -> float:
        """
        Convertit [0, 100] en [-1, 1] :
            0-25   → peur extrême → score < -0.5
            25-45  → peur → score ∈ [-0.5, 0]
            45-55  → neutre → score ≈ 0
            55-75  → greed → score ∈ [0, 0.5]
            75-100 → greed extrême → score > 0.5
        """
        return round((value - 50) / 50.0, 4)

    def get_regime_signal(self) -> str:
        """
        Retourne une étiquette de régime basée sur le score.
        Utilisée par le RegimeAgent comme input externe.
        """
        val = self.get_latest()["value"]
        if val <= 20:
            return "extreme_fear"
        elif val <= 40:
            return "fear"
        elif val <= 60:
            return "neutral"
        elif val <= 80:
            return "greed"
        else:
            return "extreme_greed"
