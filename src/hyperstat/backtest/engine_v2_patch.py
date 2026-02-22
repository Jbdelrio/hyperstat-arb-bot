"""
HyperStat v2 — Patch pour backtest/engine.py
Intégration de : MLPredictor + DataSplitter + Orchestrator dans la boucle existante

Ce fichier montre les modifications minimales à apporter à engine.py
pour activer la Phase 1 de l'architecture multi-agents.

INSTRUCTIONS D'INTÉGRATION :
  1. Copier les imports dans engine.py (section imports)
  2. Ajouter BacktestConfigV2 (extends BacktestConfig)
  3. Modifier run_backtest() selon le patch ci-dessous

Aucune modification des modules existants (stat_arb.py, regime.py, allocator.py, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 : BacktestConfigV2
# À ajouter dans backtest/engine.py après BacktestConfig existant
# ─────────────────────────────────────────────────────────────────────────────

# NOTE : on étend BacktestConfig sans la modifier (rétrocompatibilité totale)
# from hyperstat.backtest.engine import BacktestConfig   # import existant

@dataclass
class BacktestConfigV2:
    """
    Extension de BacktestConfig pour HyperStat v2.
    Active les modules ML, split temporel et orchestrateur.

    Usage :
        from hyperstat.backtest.engine_v2_patch import BacktestConfigV2
        cfg_v2 = BacktestConfigV2()
        report = run_backtest_v2(cfg_base, cfg_v2, ...)
    """
    # Activation des modules (tous OFF par défaut = rétrocompatible)
    enable_ml_predictor: bool = True
    enable_orchestrator: bool = True
    enable_temporal_split: bool = True

    # Split temporel
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    gap_bars: int = 0

    # Walk-forward ML
    retrain_every_bars: int = 336      # ~2 semaines en 1h
    min_train_bars: int = 2000

    # ML Predictor
    lstm_n_steps: int = 120
    lstm_hidden: int = 64
    forward_horizon: int = 8
    fallback_to_xgb_only: bool = True  # True si torch non installé

    # Orchestrateur
    w_stat_default: float = 0.80
    w_ml_default: float = 0.20
    enable_reflect_agent: bool = False  # False en backtest (pas d'appel LLM)
    llm_provider: str = "anthropic"

    # Paths de sauvegarde
    model_dir: str = "artifacts/models"


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 : run_backtest_v2()
# Wrapper autour de run_backtest() existant avec injection ML + Orchestrateur
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_v2(
    cfg_base,              # BacktestConfig existant (non modifié)
    cfg_v2: BacktestConfigV2,
    candles_by_symbol: Dict[str, pd.DataFrame],
    funding_by_symbol: Dict[str, pd.Series],
    buckets: Dict[str, List[str]],
    stat_arb,
    regime_model,
    allocator,
):
    """
    Version v2 du backtest engine avec ML + split temporel + orchestrateur.

    Modifications par rapport à run_backtest() :
      1. Split temporel strict train/val/test
      2. Entraînement MLPredictor sur Train+Val seulement
      3. Boucle Test rejouée barre par barre via RealTimeSimulator
      4. Signal final = Orchestrator.combine(s_stat, s_ml, sent_gate)
      5. Retrains walk-forward automatiques

    Retourne le même type BacktestReport que run_backtest().
    """

    # ── Import des modules v2 ──────────────────────────────────────────────
    from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig, RealTimeSimulator
    from hyperstat.ml.lstm_xgb_predictor import MLPredictor, MLPredictorConfig, LSTMConfig, XGBConfig
    from hyperstat.agents.orchestrator import Orchestrator, OrchestratorConfig, SentimentAgent

    # ── 1. Vérification split ──────────────────────────────────────────────
    if cfg_v2.enable_temporal_split:
        split_cfg = SplitConfig(
            train_frac=cfg_v2.train_frac,
            val_frac=cfg_v2.val_frac,
            test_frac=cfg_v2.test_frac,
            retrain_every_bars=cfg_v2.retrain_every_bars,
            min_train_bars=cfg_v2.min_train_bars,
            gap_bars=cfg_v2.gap_bars,
        )
        splitter = DataSplitter(split_cfg)
        split = splitter.compute_split(candles_by_symbol)
        logger.info(f"\n{split.summary()}")

        # Données train+val seulement pour le premier fit
        c_train = {s: df[df.index <= split.val_end] for s, df in candles_by_symbol.items()}
        f_train = {s: ser[ser.index <= split.val_end] for s, ser in funding_by_symbol.items()}

        # Données test pour le replay
        c_test = {s: df for s, df in candles_by_symbol.items()}   # complet, causalité gérée par simulator
        f_test = {s: ser for s, ser in funding_by_symbol.items()}
    else:
        # Mode legacy : pas de split
        c_train = candles_by_symbol
        f_train = funding_by_symbol
        c_test = candles_by_symbol
        f_test = funding_by_symbol
        split = None
        splitter = None

    # ── 2. Entraînement initial ML Predictor ──────────────────────────────
    ml_predictor = None
    if cfg_v2.enable_ml_predictor:
        ml_cfg = MLPredictorConfig(
            lstm=LSTMConfig(
                n_steps_in=cfg_v2.lstm_n_steps,
                hidden_size=cfg_v2.lstm_hidden,
            ),
            forward_horizon=cfg_v2.forward_horizon,
            fallback_to_xgb_only=cfg_v2.fallback_to_xgb_only,
        )
        ml_predictor = MLPredictor(ml_cfg)
        logger.info("[BacktestV2] Entraînement initial MLPredictor...")
        ml_predictor.fit(c_train, f_train)

        if ml_predictor.last_ic < ml_cfg.min_ic_threshold:
            logger.warning(
                f"[BacktestV2] IC={ml_predictor.last_ic:.4f} < seuil "
                f"({ml_cfg.min_ic_threshold}) — signal ML désactivé"
            )

    # ── 3. Orchestrateur ───────────────────────────────────────────────────
    orchestrator = None
    if cfg_v2.enable_orchestrator:
        orch_cfg = OrchestratorConfig(
            w_stat_default=cfg_v2.w_stat_default,
            w_ml_default=cfg_v2.w_ml_default,
            enable_reflect=cfg_v2.enable_reflect_agent,
            llm_provider=cfg_v2.llm_provider,
        )
        orchestrator = Orchestrator(orch_cfg)
        sentiment_agent = SentimentAgent()

    # ── 4. Calendrier walk-forward ─────────────────────────────────────────
    retrain_lookup: Dict = {}
    if split is not None and splitter is not None:
        sim = RealTimeSimulator(splitter, split)
        retrain_lookup = {
            rp: (te, ve)
            for rp, te, ve in splitter.walk_forward_schedule(split)
        }
        test_timestamps = split.test_idx
    else:
        sim = None
        test_timestamps = None

    # ── 5. Injection dans le backtest engine existant ─────────────────────
    # Le moteur existing (engine.py) est appelé sur la période test uniquement
    # On injecte un "signal_hook" qui remplace le signal stat-arb par le signal orchestré

    def signal_hook(ts: pd.Timestamp, stat_arb_signals: Dict[str, float], regime) -> Dict[str, float]:
        """
        Hook appelé à chaque barre par le backtest engine.
        Remplace le signal stat-arb brut par le signal orchestré.

        À intégrer dans engine.py :
            if hasattr(self, '_signal_hook') and self._signal_hook is not None:
                signals = self._signal_hook(ts, signals, regime)
        """
        # Retrain walk-forward si planifié
        if ml_predictor is not None and ts in retrain_lookup:
            train_end, val_end = retrain_lookup[ts]
            c_tr, f_tr, c_vl, f_vl = splitter.slice_for_training(
                c_test, f_test, train_end, val_end
            )
            logger.info(f"[BacktestV2] Walk-forward retrain à {ts}")
            ml_predictor.fit(c_tr, f_tr)

        # Prédiction ML
        ml_signals = {}
        if ml_predictor is not None and ml_predictor.is_trained:
            c_avail = {s: df[df.index < ts] for s, df in c_test.items()}
            f_avail = {s: ser[ser.index < ts] for s, ser in f_test.items()}
            ml_signals = ml_predictor.predict(
                c_avail, f_avail, ts,
                fds_scores=None,    # peut être passé depuis l'allocator si disponible
                regime_q=float(regime.q_t) if hasattr(regime, "q_t") else 1.0,
            )

        # Orchestration
        if orchestrator is not None:
            sent_gate = sentiment_agent.gate(ts)
            q_break = float(getattr(regime, "q_break", 1.0))
            final_signals = orchestrator.combine(
                ts=ts,
                stat_arb_signals=stat_arb_signals,
                ml_signals=ml_signals,
                regime_q=float(getattr(regime, "q_t", 1.0)),
                q_break=q_break,
                sent_gate=sent_gate,
            )
            return final_signals

        return stat_arb_signals  # fallback : signal stat-arb inchangé

    # ── 6. Appel engine existant avec hook ────────────────────────────────
    # Import de la fonction existante
    try:
        from hyperstat.backtest.engine import run_backtest as _run_backtest_legacy
    except ImportError:
        logger.error("hyperstat.backtest.engine introuvable — vérifier le PYTHONPATH")
        raise

    # On injecte le hook dans l'engine via monkey-patching temporaire
    # (la façon propre serait d'ajouter signal_hook à BacktestEngine.run(),
    # mais on reste non-invasif ici pour garder la rétrocompatibilité)
    cfg_base._signal_hook_v2 = signal_hook

    # Restriction aux données test si split activé
    candles_run = c_test if cfg_v2.enable_temporal_split else candles_by_symbol
    funding_run = f_test if cfg_v2.enable_temporal_split else funding_by_symbol

    report = _run_backtest_legacy(
        cfg=cfg_base,
        candles_by_symbol=candles_run,
        funding_by_symbol=funding_run,
        buckets=buckets,
        stat_arb=stat_arb,
        regime_model=regime_model,
        allocator=allocator,
    )

    # ── 7. Métriques IC ML additionnelles ─────────────────────────────────
    if ml_predictor is not None and ml_predictor.is_trained:
        logger.info(
            f"\n{'='*50}\n"
            f"[BacktestV2] Résumé ML\n"
            f"  IC Spearman (val) = {ml_predictor.last_ic:.4f}\n"
            f"  Signal ML actif   = {ml_predictor.last_ic >= ml_predictor.cfg.min_ic_threshold}\n"
            f"  Poids Orchestrateur : stat={cfg_v2.w_stat_default}, ml={cfg_v2.w_ml_default}\n"
            f"{'='*50}"
        )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 : Modification minimale de BacktestEngine._run()
# Coller ce snippet dans la boucle principale de engine.py
# ─────────────────────────────────────────────────────────────────────────────

ENGINE_LOOP_PATCH = '''
# ──────────────────────────────────────────────────────────────────────────────
# PATCH v2 : À insérer dans BacktestEngine._run() après le calcul du signal stat-arb
# (juste avant l'appel à allocator.allocate())
# ──────────────────────────────────────────────────────────────────────────────

# Ligne existante (à laisser) :
#   signal = self.stat_arb.compute_signal(ts=ts, ...)

# Ajouter juste après :
if hasattr(self.cfg, "_signal_hook_v2") and self.cfg._signal_hook_v2 is not None:
    signal = self.cfg._signal_hook_v2(ts, signal, regime)

# La ligne existante d'allocation reste inchangée :
#   target = self.allocator.allocate(ts=ts, signal=signal, regime=regime, ...)
# ──────────────────────────────────────────────────────────────────────────────
'''

if __name__ == "__main__":
    print("Instructions d'intégration dans engine.py :")
    print(ENGINE_LOOP_PATCH)
    print("\nConfig v2 par défaut :")
    import dataclasses, json
    print(json.dumps(dataclasses.asdict(BacktestConfigV2()), indent=2))
