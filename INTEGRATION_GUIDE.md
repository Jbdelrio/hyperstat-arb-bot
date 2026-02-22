# Guide d'intégration HyperStat v2
# Fichiers à copier et modifications à faire dans le repo existant

## ÉTAPE 1 — Copier les nouveaux fichiers dans le repo

```
cp src/hyperstat/ml/__init__.py                   → hyperstat-arb-bot/src/hyperstat/ml/__init__.py
cp src/hyperstat/ml/lstm_xgb_predictor.py         → hyperstat-arb-bot/src/hyperstat/ml/lstm_xgb_predictor.py
cp src/hyperstat/ml/walk_forward_split.py         → hyperstat-arb-bot/src/hyperstat/ml/walk_forward_split.py
cp src/hyperstat/agents/__init__.py               → hyperstat-arb-bot/src/hyperstat/agents/__init__.py
cp src/hyperstat/agents/orchestrator.py           → hyperstat-arb-bot/src/hyperstat/agents/orchestrator.py
cp src/hyperstat/backtest/engine_v2_patch.py      → hyperstat-arb-bot/src/hyperstat/backtest/engine_v2_patch.py
cp tests/test_v2_modules.py                       → hyperstat-arb-bot/tests/test_v2_modules.py
cp requirements-ml.txt                            → hyperstat-arb-bot/requirements-ml.txt
```

## ÉTAPE 2 — Installer les dépendances

```bash
pip install -r requirements-ml.txt
# Pour LSTM (optionnel, CPU) :
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## ÉTAPE 3 — Modification MINIMALE de backtest/engine.py

Ajouter UNE seule ligne dans la boucle principale de BacktestEngine._run()
(juste après le calcul du signal stat-arb, avant l'allocator) :

```python
# Existant (ne pas modifier) :
signal = self.stat_arb.compute_signal(ts=ts, ...)

# AJOUTER juste après :
if hasattr(self.cfg, "_signal_hook_v2") and self.cfg._signal_hook_v2 is not None:
    signal = self.cfg._signal_hook_v2(ts, signal, regime)

# Existant (ne pas modifier) :
target = self.allocator.allocate(ts=ts, signal=signal, ...)
```

## ÉTAPE 4 — Utilisation

### Backtest v2 avec split + ML + Orchestrateur

```python
from hyperstat.backtest.engine import BacktestConfig
from hyperstat.backtest.engine_v2_patch import BacktestConfigV2, run_backtest_v2

cfg_base = BacktestConfig(
    timeframe="1h",
    initial_equity=1500.0,
)

cfg_v2 = BacktestConfigV2(
    enable_ml_predictor=True,
    enable_orchestrator=True,
    enable_temporal_split=True,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    fallback_to_xgb_only=True,   # True si PyTorch non installé
    enable_reflect_agent=False,   # True en production (Phase 2)
)

report = run_backtest_v2(
    cfg_base=cfg_base,
    cfg_v2=cfg_v2,
    candles_by_symbol=candles,
    funding_by_symbol=funding,
    buckets=buckets,
    stat_arb=stat_arb,
    regime_model=regime_model,
    allocator=allocator,
)
```

### Backtest legacy inchangé (rétrocompatible)

```python
# L'API existante fonctionne exactement comme avant
from hyperstat.backtest.engine import run_backtest, BacktestConfig
report = run_backtest(cfg, candles, funding, buckets, stat_arb, regime_model, allocator)
```

## ÉTAPE 5 — Lancer les tests

```bash
pytest tests/test_v2_modules.py -v
# Résultat attendu :
# test_split_fractions         PASSED
# test_no_overlap              PASSED
# test_chronological_order     PASSED
# test_walk_forward_schedule   PASSED
# test_build_sequence_features PASSED
# test_no_lookahead            PASSED
# test_combine_basic           PASSED
# test_shock_regime_flat       PASSED
# test_full_pipeline_smoke     PASSED
```

## ÉTAPE 6 — Vérification IC sur données réelles Hyperliquid

Avant de passer en live, vérifier que l'IC est suffisant :

```python
from hyperstat.ml.lstm_xgb_predictor import MLPredictor, MLPredictorConfig

predictor = MLPredictor(MLPredictorConfig(fallback_to_xgb_only=True))
predictor.fit(candles_train, funding_train)

print(f"IC Spearman val = {predictor.last_ic:.4f}")
# Objectif : IC > 0.03 (t-stat > 2)
# Si IC < 0.02 → le signal ML est automatiquement éteint (scores = 0)
```

## Architecture résultante

```
HyperStat v2 (après intégration)
│
├── Layer 2 — Alpha Agents
│   ├── Agent StatArb    → stat_arb.py  (INCHANGÉ)
│   ├── Agent ML         → ml/lstm_xgb_predictor.py  (NOUVEAU)
│   └── Agent Sentiment  → agents/orchestrator.py SentimentAgent  (NOUVEAU)
│
├── Layer 3 — Orchestrateur
│   └── agents/orchestrator.py Orchestrator  (NOUVEAU)
│       ├── Combine signaux par régime
│       └── Reflect Agent (verbal feedback, Phase 2)
│
├── Walk-Forward Split
│   └── ml/walk_forward_split.py  (NOUVEAU)
│       ├── Train 70% | Val 15% | Test 15%
│       └── RealTimeSimulator (replay causal)
│
└── Backtest Engine (modifié minimalement)
    └── backtest/engine.py — UNE ligne ajoutée (_signal_hook_v2)
```
