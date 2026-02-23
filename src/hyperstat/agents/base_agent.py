"""
hyperstat.agents.base_agent
============================
Interface abstraite commune à tous les agents IA.
Chaque agent hérite de BaseAgent et implémente : observe(), act(), score().

Protocol de communication inter-agents :
    AgentSignal  → sortie standardisée de chaque agent
    AgentBus     → bus de messages pour le SupervisorAgent
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TYPES PARTAGÉS
# ─────────────────────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    """État courant d'un agent."""
    ACTIVE   = "active"      # tourne normalement
    DEGRADED = "degraded"    # données partielles, signal moins fiable
    HALTED   = "halted"      # stoppé (maintenance, erreur critique)
    WARMING  = "warming_up"  # en cours d'initialisation (pas encore assez de données)


class SignalDirection(str, Enum):
    LONG     = "long"
    SHORT    = "short"
    NEUTRAL  = "neutral"


@dataclass
class AgentSignal:
    """
    Sortie standardisée de chaque agent, consommée par le SupervisorAgent.

    Attributes
    ----------
    agent_name   : nom de l'agent émetteur
    ts           : horodatage du signal
    direction    : LONG / SHORT / NEUTRAL
    confidence   : float ∈ [0, 1] — fiabilité du signal
    score        : float ∈ [-1, 1] — intensité directionnelle
                   > 0 = bullish, < 0 = bearish
    regime_hint  : régime détecté par cet agent (optionnel)
    metadata     : dict de métriques internes pour le dashboard
    """
    agent_name   : str
    ts           : datetime
    direction    : SignalDirection
    confidence   : float               # [0, 1]
    score        : float               # [-1, 1]
    regime_hint  : Optional[str] = None
    metadata     : Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.score      = float(np.clip(self.score, -1.0, 1.0))

    @property
    def weighted_score(self) -> float:
        """Score pondéré par la confiance."""
        return self.score * self.confidence


# Lazy import pour éviter dépendance circulaire
import numpy as np


@dataclass
class AgentPerformanceTracker:
    """
    Suit la performance historique d'un agent pour ajuster son poids
    dans le SupervisorAgent (vote pondéré par performance récente).
    """
    agent_name     : str
    window         : int = 100         # nombre de signaux à conserver
    _scores        : List[float] = field(default_factory=list, repr=False)
    _outcomes      : List[float] = field(default_factory=list, repr=False)

    def record(self, predicted_score: float, actual_return: float):
        """Enregistre un signal et son résultat réel."""
        self._scores.append(predicted_score)
        self._outcomes.append(actual_return)
        if len(self._scores) > self.window:
            self._scores.pop(0)
            self._outcomes.pop(0)

    @property
    def ic(self) -> float:
        """Information Coefficient (Spearman) sur la fenêtre courante."""
        if len(self._scores) < 10:
            return 0.0
        from scipy.stats import spearmanr
        corr, _ = spearmanr(self._scores, self._outcomes)
        return float(corr) if not np.isnan(corr) else 0.0

    @property
    def reliability_weight(self) -> float:
        """Poids de fiabilité ∈ [0.1, 1.0] basé sur l'IC récent."""
        ic = self.ic
        if ic <= 0:
            return 0.1
        return float(np.clip(0.1 + 0.9 * ic / 0.1, 0.1, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# ABC BASEAGENT
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Classe de base abstraite pour tous les agents HyperStat.

    Cycle de vie :
        1. __init__ : configuration
        2. warm_up  : initialisation avec données historiques
        3. observe  : reçoit les données du tick courant
        4. act      : produit un AgentSignal
        5. on_feedback : reçoit le résultat pour mettre à jour le tracker
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name    = name
        self.config  = config or {}
        self.status  = AgentStatus.WARMING
        self.tracker = AgentPerformanceTracker(agent_name=name)
        self._last_signal: Optional[AgentSignal] = None
        self._error_count: int = 0
        self._max_errors: int  = 5
        logger.info(f"[{self.name}] Agent initialisé — statut: {self.status}")

    # ── Interface obligatoire ──────────────────────────────────────────────

    @abstractmethod
    def warm_up(self, **kwargs) -> bool:
        """
        Initialise l'agent avec des données historiques.
        Retourne True si l'agent est prêt à produire des signaux.
        """

    @abstractmethod
    def observe(self, ts: datetime, data: Dict[str, Any]) -> None:
        """
        Reçoit les données du tick courant.
        Met à jour l'état interne sans produire de signal.
        """

    @abstractmethod
    def act(self, ts: datetime) -> AgentSignal:
        """
        Produit un AgentSignal à partir de l'état interne courant.
        Appelé après observe().
        """

    # ── Interface optionnelle ──────────────────────────────────────────────

    def on_feedback(self, actual_return: float) -> None:
        """
        Reçoit le retour réel pour mettre à jour le performance tracker.
        Appelé H barres après act().
        """
        if self._last_signal is not None:
            self.tracker.record(self._last_signal.score, actual_return)

    def get_status_dict(self) -> Dict[str, Any]:
        """Retourne un dict de statut pour le dashboard."""
        return {
            "name"               : self.name,
            "status"             : self.status.value,
            "last_signal"        : self._last_signal.score if self._last_signal else None,
            "last_direction"     : self._last_signal.direction.value if self._last_signal else None,
            "last_confidence"    : self._last_signal.confidence if self._last_signal else None,
            "ic_recent"          : round(self.tracker.ic, 4),
            "reliability_weight" : round(self.tracker.reliability_weight, 3),
            "error_count"        : self._error_count,
        }

    # ── Helpers internes ───────────────────────────────────────────────────

    def _set_active(self):
        self.status = AgentStatus.ACTIVE
        logger.debug(f"[{self.name}] → ACTIVE")

    def _set_halted(self, reason: str = ""):
        self.status = AgentStatus.HALTED
        logger.warning(f"[{self.name}] → HALTED. Raison: {reason}")

    def _set_degraded(self, reason: str = ""):
        self.status = AgentStatus.DEGRADED
        logger.warning(f"[{self.name}] → DEGRADED. Raison: {reason}")

    def _handle_error(self, exc: Exception) -> AgentSignal:
        """Gestion d'erreur avec compteur et fallback signal NEUTRAL."""
        self._error_count += 1
        logger.error(f"[{self.name}] Erreur #{self._error_count}: {exc}")
        if self._error_count >= self._max_errors:
            self._set_halted(reason=str(exc))
        else:
            self._set_degraded(reason=str(exc))
        return AgentSignal(
            agent_name  = self.name,
            ts          = datetime.utcnow(),
            direction   = SignalDirection.NEUTRAL,
            confidence  = 0.0,
            score       = 0.0,
            metadata    = {"error": str(exc), "error_count": self._error_count},
        )

    def _make_signal(
        self,
        ts: datetime,
        score: float,
        confidence: float,
        regime_hint: Optional[str] = None,
        **metadata,
    ) -> AgentSignal:
        """Helper pour construire un AgentSignal proprement."""
        score      = float(np.clip(score, -1.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        direction  = (
            SignalDirection.LONG  if score > 0.05  else
            SignalDirection.SHORT if score < -0.05 else
            SignalDirection.NEUTRAL
        )
        signal = AgentSignal(
            agent_name  = self.name,
            ts          = ts,
            direction   = direction,
            confidence  = confidence,
            score       = score,
            regime_hint = regime_hint,
            metadata    = metadata,
        )
        self._last_signal = signal
        self._error_count = 0   # reset erreur si succès
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# BUS DE MESSAGES
# ─────────────────────────────────────────────────────────────────────────────

class AgentBus:
    """
    Bus de messages centralisé.
    Les agents publient leurs signaux, le Supervisor s'abonne.
    """

    def __init__(self):
        self._signals: Dict[str, AgentSignal] = {}
        self._agents : Dict[str, BaseAgent]   = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        logger.info(f"[AgentBus] Agent enregistré: {agent.name}")

    def publish(self, signal: AgentSignal) -> None:
        self._signals[signal.agent_name] = signal

    def get_latest(self) -> Dict[str, AgentSignal]:
        return dict(self._signals)

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        return [a.get_status_dict() for a in self._agents.values()]

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)
