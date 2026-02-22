"""
HyperStat v2 — Orchestrateur Multi-Agents
Combine les signaux de l'Agent StatArb, l'Agent ML Predictor, et le gate Sentiment

Formule :
  signal_final_i = w_stat · s_stat_i + w_ml · s_ml_i · sent_gate

Pondération dynamique par régime :
  | Régime              | w_stat | w_ml |
  | MR stable (Q=1.0)   |  0.80  | 0.20 |
  | Vol élevée (Q=0.5)  |  0.60  | 0.40 |
  | Choc (Q_break=0)    |  0.00  | 0.00 |  → flat total

Boucle de verbal feedback (Singhi 2025, arXiv:2510.08068) :
  Le Reflect Agent produit chaque semaine une critique JSON des performances,
  injectée dans le prochain appel LLM pour ajuster les poids dynamiquement.

Contexte JSON passé aux LLM (Li et al. 2512.02227) :
  Jamais de P&L brut ni de returns tick-à-tick.
  Uniquement : Sharpe agrégé, DD, IC, régime, hit_rate.

Auteur : HyperStat v2
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorConfig:
    # Poids par défaut (ajustés dynamiquement par régime)
    w_stat_default: float = 0.80
    w_ml_default: float = 0.20

    # Poids par régime
    w_stat_high_vol: float = 0.60    # Q_risk ≤ 0.5
    w_ml_high_vol: float = 0.40

    w_stat_shock: float = 0.0        # Q_break = 0
    w_ml_shock: float = 0.0

    # Sentiment gate
    sent_gate_min: float = 0.0       # gate minimum (jamais négatif = sent_gate amplificateur)
    sent_gate_max: float = 1.0       # gate maximum

    # Reflect Agent
    enable_reflect: bool = True
    reflect_every_bars: int = 168    # ~1 semaine en 1h
    reflect_memory_file: str = "artifacts/reflect_memory.json"

    # LLM pour le Reflect Agent
    llm_provider: str = "anthropic"  # "anthropic" | "openai" | "local"
    llm_model: str = "claude-haiku-4-5-20251001"  # low-cost
    llm_max_tokens: int = 512


@dataclass
class AgentContext:
    """Contexte JSON passé entre agents. Jamais de P&L brut."""
    task_id: str
    ts: str                                    # ISO timestamp
    regime: str                                # "mr_stable" | "high_vol" | "shock"
    q_t: float                                 # Q_MR × Q_liq × Q_risk × Q_break
    q_break: float

    # Signaux des agents alpha (normalisés ∈ [-1,1])
    stat_arb_signals: Dict[str, float]
    ml_signals: Dict[str, float]
    sent_gate: float

    # Métriques agrégées 7 derniers jours (jamais tick-à-tick)
    metrics_7d: Dict[str, float]               # sharpe, dd, ic_mean, hit_rate, turnover

    # Feedback verbal du Reflect Agent (peut être vide)
    reflect_feedback: str = ""

    def to_prompt_json(self) -> str:
        """Sérialisation JSON propre pour un prompt LLM."""
        d = asdict(self)
        # Arrondir les floats pour compresser le prompt
        for key in ["q_t", "q_break", "sent_gate"]:
            d[key] = round(d[key], 3)
        d["stat_arb_signals"] = {k: round(v, 3) for k, v in d["stat_arb_signals"].items()}
        d["ml_signals"] = {k: round(v, 3) for k, v in d["ml_signals"].items()}
        d["metrics_7d"] = {k: round(v, 4) for k, v in d["metrics_7d"].items()}
        return json.dumps(d, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Orchestre les 3 agents alpha et produit le signal final combiné.

    Usage (dans backtest engine ou live runner) :
        orch = Orchestrator(OrchestratorConfig())

        # À chaque barre :
        final_signals = orch.combine(
            ts=ts,
            stat_arb_signals=s_stat,
            ml_signals=s_ml,
            regime_q=regime.q_t,
            q_break=regime.q_break,
            sent_gate=sentiment_agent.gate(ts),
            metrics_7d=metrics_window,
        )
        # final_signals : {symbol: weight ∈ [-1, 1]}
    """

    def __init__(self, cfg: Optional[OrchestratorConfig] = None):
        self.cfg = cfg or OrchestratorConfig()
        self._bar_count: int = 0
        self._reflect_memory: List[str] = []
        self._current_feedback: str = ""
        self._load_reflect_memory()

    def combine(
        self,
        ts: pd.Timestamp,
        stat_arb_signals: Dict[str, float],
        ml_signals: Dict[str, float],
        regime_q: float,
        q_break: float,
        sent_gate: float,
        metrics_7d: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Combine les signaux et retourne les poids finaux.
        Le résultat est passé directement à l'allocator existant.
        """
        self._bar_count += 1

        # 1. Sélection des poids selon régime
        w_stat, w_ml = self._regime_weights(regime_q, q_break)

        # Si choc total → flat
        if w_stat == 0.0 and w_ml == 0.0:
            logger.debug(f"[Orchestrator] Choc détecté à {ts} → flat total")
            return {sym: 0.0 for sym in stat_arb_signals}

        # 2. Gate sentiment (atténuateur uniquement, jamais amplificateur)
        sg = float(np.clip(sent_gate, self.cfg.sent_gate_min, self.cfg.sent_gate_max))

        # 3. Combinaison
        symbols = set(stat_arb_signals) | set(ml_signals)
        combined: Dict[str, float] = {}
        for sym in symbols:
            s_stat = float(stat_arb_signals.get(sym, 0.0))
            s_ml   = float(ml_signals.get(sym, 0.0))
            raw = w_stat * s_stat + w_ml * s_ml * sg
            combined[sym] = float(np.clip(raw, -1.0, 1.0))

        # 4. Déclencher le Reflect Agent si nécessaire
        if (
            self.cfg.enable_reflect
            and self._bar_count % self.cfg.reflect_every_bars == 0
            and metrics_7d is not None
        ):
            ctx = AgentContext(
                task_id=f"reflect_{ts.isoformat()}",
                ts=ts.isoformat(),
                regime=self._regime_label(regime_q, q_break),
                q_t=regime_q,
                q_break=q_break,
                stat_arb_signals=stat_arb_signals,
                ml_signals=ml_signals,
                sent_gate=sg,
                metrics_7d=metrics_7d or {},
                reflect_feedback=self._current_feedback,
            )
            self._run_reflect_agent(ctx)

        return combined

    def _regime_weights(self, q_t: float, q_break: float) -> Tuple[float, float]:
        """Pondération dynamique selon le régime."""
        cfg = self.cfg
        if q_break == 0.0 or q_t == 0.0:
            return cfg.w_stat_shock, cfg.w_ml_shock
        if q_t <= 0.5:
            return cfg.w_stat_high_vol, cfg.w_ml_high_vol
        return cfg.w_stat_default, cfg.w_ml_default

    @staticmethod
    def _regime_label(q_t: float, q_break: float) -> str:
        if q_break == 0.0 or q_t == 0.0:
            return "shock"
        if q_t <= 0.5:
            return "high_vol"
        return "mr_stable"

    # ──────────────────────────────────────────────────────────────────────
    # Reflect Agent (verbal feedback)
    # ──────────────────────────────────────────────────────────────────────

    def _run_reflect_agent(self, ctx: AgentContext) -> None:
        """
        Appelle le LLM pour générer un feedback verbal sur les performances.
        Le feedback est stocké en mémoire et injecté dans le prochain appel.

        Règle Li et al. 2512.02227 : jamais de P&L brut dans le contexte LLM.
        Règle Singhi 2510.08068   : le feedback ajuste les priorités, pas les paramètres.
        """
        try:
            feedback = self._call_reflect_llm(ctx)
            self._current_feedback = feedback
            self._reflect_memory.append(
                {"ts": ctx.ts, "regime": ctx.regime, "feedback": feedback,
                 "metrics": ctx.metrics_7d}
            )
            self._save_reflect_memory()
            logger.info(f"[ReflectAgent] Feedback généré à {ctx.ts}")
        except Exception as e:
            logger.warning(f"[ReflectAgent] Échec LLM : {e} — feedback vide")
            self._current_feedback = ""

    def _call_reflect_llm(self, ctx: AgentContext) -> str:
        """
        Appel LLM réel avec le contexte.
        Retourne le feedback en texte brut (max 512 tokens).
        """
        prompt = self._build_reflect_prompt(ctx)

        provider = self.cfg.llm_provider
        if provider == "anthropic":
            return self._call_anthropic(prompt)
        elif provider == "openai":
            return self._call_openai(prompt)
        elif provider == "local":
            return self._call_local_llm(prompt)
        else:
            raise ValueError(f"Provider LLM inconnu : {provider}")

    def _build_reflect_prompt(self, ctx: AgentContext) -> str:
        """
        Construit le prompt de réflexion hebdomadaire.
        Structure Singhi 2025 : analyse des métriques → critique → ajustements suggérés.
        """
        prev_feedback_section = (
            f"\n\nFeedback précédent (à améliorer si toujours d'actualité) :\n{ctx.reflect_feedback}"
            if ctx.reflect_feedback
            else ""
        )

        return f"""Tu es le Reflect Agent du système de trading statistique HyperStat.
Tu analyses les performances de la semaine passée et fournis des critiques constructives.

CONTEXTE (métriques agrégées, PAS de P&L brut) :
{ctx.to_prompt_json()}
{prev_feedback_section}

INSTRUCTIONS :
1. Identifie les 1-2 points forts de la semaine (ex : bonne gestion du régime, IC ML positif)
2. Identifie les 1-2 points faibles (ex : signal stat-arb peu actif en marché latéral)
3. Suggère 1-2 ajustements CONCRETS pour la semaine suivante :
   - Pondération des agents (ex : "augmenter w_ml si volatilité reste élevée")
   - Seuils de régime (ex : "abaisser le seuil Q_break si trop de flat périodes")
   - Gate sentiment (ex : "réduire sent_gate_min si Fear & Greed < 25")

CONTRAINTES :
- Maximum 200 mots
- Aucune référence à des prix ou P&L spécifiques
- Rester factuel et basé sur les métriques fournies
- Format : texte continu, pas de JSON

Analyse :"""

    def _call_anthropic(self, prompt: str) -> str:
        """Appel API Anthropic (Claude Haiku - low cost)."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.cfg.llm_model,
                max_tokens=self.cfg.llm_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except ImportError:
            raise RuntimeError("pip install anthropic requis pour le Reflect Agent")

    def _call_openai(self, prompt: str) -> str:
        """Appel API OpenAI (GPT-4o mini - low cost)."""
        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=self.cfg.llm_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            raise RuntimeError("pip install openai requis")

    def _call_local_llm(self, prompt: str) -> str:
        """Appel Ollama local (Llama 3.1 8B - gratuit)."""
        try:
            import requests
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.1:8b", "prompt": prompt, "stream": False},
                timeout=30,
            )
            return resp.json().get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama indisponible : {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Persist mémoire Reflect
    # ──────────────────────────────────────────────────────────────────────

    def _load_reflect_memory(self) -> None:
        path = Path(self.cfg.reflect_memory_file)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._reflect_memory = data.get("history", [])
                if self._reflect_memory:
                    self._current_feedback = self._reflect_memory[-1].get("feedback", "")
                logger.info(
                    f"[ReflectAgent] {len(self._reflect_memory)} feedbacks chargés"
                )
            except Exception as e:
                logger.warning(f"[ReflectAgent] Impossible de charger la mémoire : {e}")

    def _save_reflect_memory(self) -> None:
        path = Path(self.cfg.reflect_memory_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(
                    {"version": "v2", "history": self._reflect_memory[-50:]},  # garder les 50 derniers
                    f, indent=2
                )
        except Exception as e:
            logger.warning(f"[ReflectAgent] Impossible de sauvegarder : {e}")

    @property
    def reflect_memory(self) -> List[dict]:
        return self._reflect_memory.copy()

    @property
    def current_feedback(self) -> str:
        return self._current_feedback


# ─────────────────────────────────────────────────────────────────────────────
# Agent Sentiment (collecteur Fear & Greed + CryptoPanic)
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAgent:
    """
    Collecte le Fear & Greed Index et les scores de sentiment news.
    Retourne un gate ∈ [0, 1] qui atténue l'exposition globale.

    Sources gratuites :
      - Alternative.me Fear & Greed API (illimité, quotidien)
      - CryptoPanic RSS (100 req/jour gratuit)

    Règle : le sentiment RÉDUIT l'exposition, jamais ne l'amplifie.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[float, float]] = {}   # {date_str: (gate, ts)}
        self._last_fear_greed: float = 50.0                # neutre par défaut

    def fetch_fear_greed(self) -> float:
        """
        Récupère le Fear & Greed Index depuis Alternative.me.
        Cache quotidien pour respecter les limites d'API.
        Retourne un index ∈ [0, 100] (0=peur extrême, 100=avidité extrême).
        """
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        if today in self._cache:
            return self._cache[today][0]

        try:
            import urllib.request
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            value = float(data["data"][0]["value"])
            self._cache[today] = (value, time.time())
            self._last_fear_greed = value
            logger.debug(f"[SentimentAgent] Fear & Greed = {value:.0f}")
            return value
        except Exception as e:
            logger.warning(f"[SentimentAgent] Fear & Greed indisponible : {e}")
            return self._last_fear_greed

    def compute_gate(self, fear_greed: Optional[float] = None) -> float:
        """
        Calcule le gate ∈ [0, 1] depuis le Fear & Greed.

        Mapping :
          [0, 25]   → gate = 0.3  (peur extrême → réduire fortement)
          [25, 45]  → gate = 0.6  (peur → réduire modérément)
          [45, 55]  → gate = 1.0  (neutre → plein signal)
          [55, 75]  → gate = 0.8  (avidité → légère prudence)
          [75, 100] → gate = 0.5  (avidité extrême → réduire)

        Logique : les extrêmes de sentiment précèdent souvent des reversals.
        """
        fg = fear_greed if fear_greed is not None else self._last_fear_greed
        if fg <= 25:
            return 0.3
        elif fg <= 45:
            return 0.6
        elif fg <= 55:
            return 1.0
        elif fg <= 75:
            return 0.8
        else:
            return 0.5

    def gate(self, ts: Optional[pd.Timestamp] = None) -> float:
        """Point d'entrée principal : retourne le gate pour la barre courante."""
        fg = self.fetch_fear_greed()
        return self.compute_gate(fg)
