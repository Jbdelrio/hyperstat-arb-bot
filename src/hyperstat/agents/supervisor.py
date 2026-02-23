"""
hyperstat.agents.supervisor
=============================
SupervisorAgent — Orchestrateur central.

Responsabilités :
    - Agrège les signaux de tous les agents spécialisés
    - Vote pondéré par la performance récente de chaque agent (IC tracker)
    - Déclenche le kill-switch global si régime = crisis
    - Ajuste les poids des stratégies en temps réel
    - Alimente le dashboard avec l'état complet du système
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hyperstat.agents.base_agent import (
    AgentBus, AgentSignal, AgentStatus, BaseAgent, SignalDirection
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AllocationDecision:
    """
    Décision finale du SupervisorAgent :
        - scale_factor : multiplicateur global sur les poids stat-arb [0, 1]
        - regime       : régime détecté
        - kill_switch  : True → flatten tout et stopper
        - agents_scores: score par agent pour le dashboard
        - qt_override  : Q_t suggéré (peut surcharger le Q_t interne)
    """
    ts             : datetime
    scale_factor   : float                  # [0, 1]
    regime         : str
    kill_switch    : bool = False
    qt_override    : Optional[float] = None
    agents_scores  : Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    confidence     : float = 0.0
    reason         : str   = ""


@dataclass
class SupervisorConfig:
    # Poids des agents (normalisés automatiquement)
    weight_technical  : float = 0.40
    weight_sentiment  : float = 0.20
    weight_prediction : float = 0.25
    weight_regime     : float = 0.15

    # Seuils
    kill_switch_threshold: float = -0.85   # score composite → kill-switch
    reduce_threshold     : float = -0.40   # réduction de l'exposition
    boost_threshold      : float =  0.50   # renforcement (si signal fort)

    # Scale factors selon le composite score
    scale_crisis    : float = 0.0
    scale_bearish   : float = 0.3
    scale_neutral   : float = 0.7
    scale_bullish   : float = 1.0
    scale_strong    : float = 1.2   # max 120% de l'exposition nominale

    # Minimum de confiance pour agir
    min_confidence  : float = 0.3

    # Adaptation dynamique des poids par IC
    use_ic_weighting: bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorAgent(BaseAgent):
    """
    Orchestrateur central du système multi-agents.

    Usage
    -----
    bus = AgentBus()
    bus.register(technical_agent)
    bus.register(sentiment_agent)
    bus.register(prediction_agent)
    bus.register(regime_agent)

    supervisor = SupervisorAgent(bus=bus)
    decision   = supervisor.decide(ts)
    """

    def __init__(
        self,
        bus : AgentBus,
        cfg : Optional[SupervisorConfig] = None,
        **kwargs,
    ):
        super().__init__(name="SupervisorAgent", **kwargs)
        self.bus = bus
        self.cfg = cfg or SupervisorConfig()

        # Historique des décisions pour analytics
        self._decision_history : List[AllocationDecision] = []
        self._last_decision    : Optional[AllocationDecision] = None

    # ── Interface BaseAgent ───────────────────────────────────────────────

    def warm_up(self, **kwargs) -> bool:
        self._set_active()
        return True

    def observe(self, ts: datetime, data: Dict[str, Any]) -> None:
        """Le Supervisor n'observe pas directement — il lit le bus."""
        pass

    def act(self, ts: datetime) -> AgentSignal:
        decision = self.decide(ts)
        return self._make_signal(
            ts           = ts,
            score        = decision.composite_score,
            confidence   = decision.confidence,
            regime_hint  = decision.regime,
            scale_factor = decision.scale_factor,
            kill_switch  = decision.kill_switch,
        )

    # ── Décision principale ───────────────────────────────────────────────

    def decide(self, ts: datetime) -> AllocationDecision:
        """
        Agrège tous les signaux et produit une AllocationDecision.
        C'est la méthode centrale appelée par le live runner.
        """
        signals   = self.bus.get_latest()
        regime    = self._extract_regime(signals)

        # ── Kill-switch immédiat sur crisis ──
        if regime == "crisis":
            decision = AllocationDecision(
                ts              = ts,
                scale_factor    = self.cfg.scale_crisis,
                regime          = "crisis",
                kill_switch     = True,
                composite_score = -1.0,
                confidence      = 1.0,
                reason          = "Régime CRISIS — kill-switch activé",
                agents_scores   = {k: v.score for k, v in signals.items()},
            )
            self._record_decision(decision)
            logger.critical(f"[Supervisor] KILL-SWITCH activé à {ts}")
            return decision

        # ── Vote pondéré ──
        composite, confidence, scores_by_agent = self._weighted_vote(signals)

        # ── Détermination du scale factor ──
        scale_factor = self._composite_to_scale(composite)

        # ── Kill-switch si composite très négatif ──
        kill = composite < self.cfg.kill_switch_threshold

        # ── Qt override depuis le RegimeAgent ──
        qt_override = None
        if "RegimeAgent" in signals:
            regime_meta = signals["RegimeAgent"].metadata
            qt_override = regime_meta.get("qt")

        decision = AllocationDecision(
            ts              = ts,
            scale_factor    = scale_factor,
            regime          = regime,
            kill_switch     = kill,
            qt_override     = qt_override,
            composite_score = round(composite, 4),
            confidence      = round(confidence, 4),
            agents_scores   = scores_by_agent,
            reason          = self._explain_decision(composite, scale_factor, regime),
        )
        self._record_decision(decision)
        return decision

    # ── Vote pondéré ─────────────────────────────────────────────────────

    def _weighted_vote(
        self, signals: Dict[str, AgentSignal]
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Vote pondéré par :
            1. Poids configuré pour l'agent
            2. × Fiabilité IC récent (si use_ic_weighting)
            3. × Confidence du signal courant

        Retourne (composite_score, mean_confidence, scores_dict)
        """
        agent_weights = {
            "TechnicalAgent"  : self.cfg.weight_technical,
            "SentimentAgent"  : self.cfg.weight_sentiment,
            "PredictionAgent" : self.cfg.weight_prediction,
            "RegimeAgent"     : self.cfg.weight_regime,
        }

        weighted_sum  = 0.0
        total_weight  = 0.0
        confidences   = []
        scores_dict   = {}

        for agent_name, base_weight in agent_weights.items():
            if agent_name not in signals:
                continue
            signal = signals[agent_name]

            # Poids = base × IC reliability × confidence du signal
            reliability = 1.0
            if self.cfg.use_ic_weighting:
                agent = self.bus.get_agent(agent_name)
                if agent:
                    reliability = agent.tracker.reliability_weight

            effective_weight = base_weight * reliability * max(signal.confidence, 0.1)

            weighted_sum += signal.score * effective_weight
            total_weight += effective_weight
            confidences.append(signal.confidence)
            scores_dict[agent_name] = round(signal.score, 4)

        if total_weight < 1e-6:
            return 0.0, 0.0, {}

        composite  = weighted_sum / total_weight
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return float(composite), float(confidence), scores_dict

    def _composite_to_scale(self, composite: float) -> float:
        """Convertit le score composite [-1, 1] en scale_factor [0, 1.2]."""
        if composite < -0.85:
            return self.cfg.scale_crisis
        elif composite < -0.40:
            # Interpolation linéaire entre crisis et neutral
            t = (composite + 0.85) / 0.45
            return self.cfg.scale_crisis + t * (self.cfg.scale_neutral - self.cfg.scale_crisis)
        elif composite < 0.30:
            return self.cfg.scale_neutral
        elif composite < self.cfg.boost_threshold:
            return self.cfg.scale_bullish
        else:
            return self.cfg.scale_strong

    # ── Utilitaires ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_regime(signals: Dict[str, AgentSignal]) -> str:
        """Extrait le régime depuis le RegimeAgent (prioritaire)."""
        if "RegimeAgent" in signals:
            return signals["RegimeAgent"].regime_hint or "unknown"
        # Fallback : tous les agents qui donnent un regime_hint
        for sig in signals.values():
            if sig.regime_hint:
                return sig.regime_hint
        return "unknown"

    @staticmethod
    def _explain_decision(score: float, scale: float, regime: str) -> str:
        if scale == 0.0:
            return f"Exposition nulle — régime {regime}"
        if scale < 0.5:
            return f"Exposition réduite ({scale:.0%}) — score={score:.2f}, régime={regime}"
        if scale > 1.0:
            return f"Exposition renforcée ({scale:.0%}) — signal fort, régime={regime}"
        return f"Exposition nominale ({scale:.0%}) — régime={regime}"

    def _record_decision(self, decision: AllocationDecision):
        self._last_decision = decision
        self._decision_history.append(decision)
        if len(self._decision_history) > 1000:
            self._decision_history.pop(0)

    # ── Dashboard ────────────────────────────────────────────────────────

    @property
    def last_decision(self) -> Optional[AllocationDecision]:
        return self._last_decision

    def get_status_dict(self) -> Dict[str, Any]:
        base = super().get_status_dict()
        if self._last_decision:
            base.update({
                "scale_factor"   : round(self._last_decision.scale_factor, 3),
                "regime"         : self._last_decision.regime,
                "kill_switch"    : self._last_decision.kill_switch,
                "composite_score": self._last_decision.composite_score,
                "agents_scores"  : self._last_decision.agents_scores,
                "reason"         : self._last_decision.reason,
            })
        # Statut de tous les agents
        base["all_agents"] = self.bus.get_all_statuses()
        return base

    def get_decision_history_df(self):
        """Retourne l'historique des décisions sous forme de DataFrame."""
        if not self._decision_history:
            import pandas as pd
            return pd.DataFrame()
        import pandas as pd
        rows = []
        for d in self._decision_history:
            row = {
                "ts"             : d.ts,
                "scale_factor"   : d.scale_factor,
                "regime"         : d.regime,
                "kill_switch"    : d.kill_switch,
                "composite_score": d.composite_score,
                "confidence"     : d.confidence,
            }
            row.update({f"score_{k}": v for k, v in d.agents_scores.items()})
            rows.append(row)
        return pd.DataFrame(rows).set_index("ts")
