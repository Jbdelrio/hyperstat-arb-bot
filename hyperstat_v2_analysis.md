# HyperStat v2 — Analyse & Plan d'évolution Multi-Agents

---

## 1. Analyse critique de l'état actuel

### Forces

**Signal alpha différenciant.** Le FDS (Funding Divergence Signal) avec ses 3 composantes (carry cross-sectionnel + désalignement prix/funding + vélocité) exploite une asymétrie structurelle propre à Hyperliquid — le biais long des traders on-chain. C'est un signal difficile à répliquer par des bots simples.

**Architecture modulaire bien posée.** La séparation data → strategy → backtest → live/execution est propre. Le pipeline vol-scaling → FDS gate → neutralisation → caps suit une logique institutionnelle solide.

**Réalisme des coûts.** Le modèle de slippage VWAP à 4 composantes (base + vol_impact + market_impact + vwap_dev_penalty) est conservateur et adapté aux altcoins à liquidité variable.

**Gating multi-dimensionnel.** $Q_t = Q_{MR} \times Q_{liq} \times Q_{risk} \times Q_{break}$ protège contre les faux signaux en régime de stress. C'est exactement ce qu'il faut pour éviter les drawdowns sur news macro.

### Faiblesses

**Signal purement quantitatif et statique.** Le modèle actuel n'intègre aucun signal textuel (news, sentiment, social). Sur crypto, des événements comme une approbation ETF ou un tweet de Musk peuvent créer des mouvements de 15% en 30 minutes qui invalident tout signal mean-reversion. Le modèle actuel subira ces chocs sans défense.

**Pas de prédiction directionnelle.** Le stat-arb est un pari de convergence relative (A/B), pas une prédiction du sens absolu du marché. Dans un rally ou un crash brutal, même un pari correctement neutralisé peut perdre si les corrélations intra-bucket explosent vers 1.

**Pas de train/test split explicite.** L'absence de split temporel rigoureux signifie que la calibration FDS et les seuils de régime ont été choisis "sur le même jeu de données qu'on teste", risquant un data snooping subtil même sans intention.

**Architecture monolithique pour les décisions.** L'allocator fait tout : vol-scaling, FDS, neutralisation, caps. Avec l'évolution vers le multi-agents, cette concentration devient un goulot d'étranglement.

---

## 2. Revue critique des papiers — extraction et priorisation

### Papier 1 — LSTM+XGBoost (Gautam, arXiv:2506.22055)
**Ratio innovation/faisabilité : ★★★★☆**

Le hybrid capture deux horizons : LSTM = dépendances temporelles longues (séquence), XGBoost = interactions non-linéaires sur features tabulaires (sentiment, indicateurs macro). La formule clé est le pipeline à deux étages : `z = LSTM(X) → hn` (vecteur 64-dim), puis `XGBoost(features ∥ hn)`.

**Application à HyperStat :** Ce modèle devient l'**Agent ML Predictor**. Le LSTM ingère les 120 barres précédentes de prix/volume/funding et produit un embedding. XGBoost y ajoute les features cross-sectionnelles (z-score bucket, FDS components, sentiment score) pour prédire la direction à H barres. Son output `directional_score ∈ [-1, 1]` est combiné avec le signal stat-arb existant par l'orchestrateur.

**Limite à surveiller :** MAPE/RMSE sont des métriques de régression, pas de trading. Un modèle avec MAPE=2% peut générer un Sharpe négatif s'il prédit mal les queues de distribution. Évaluer impérativement avec des métriques de trading (Sharpe, Information Coefficient, hit rate conditionnel sur les mouvements > 2σ).

---

### Papier 2 — Microstructure LOB (Wang, arXiv:2506.05764)
**Ratio innovation/faisabilité : ★★★☆☆ (Phase 2 seulement)**

La conclusion principale est contre-intuitive et très utile : **les gains de performance viennent du prétraitement des données, pas de l'ajout de couches**. Un XGBoost bien alimenté (Kalman + Savitzky-Golay sur le LOB) bat un DeepLOB complexe. Sur crypto, le LOB à 100ms de Bybit est disponible gratuitement.

**Application à HyperStat :** Dans la Phase 2, ajouter un signal d'imbalance du carnet d'ordres comme feature supplémentaire pour le ML Predictor. La formule OFI (Order Flow Imbalance) à intégrer dans `data/features.py` :

```python
OFI_t = Σ_k [ ΔV_bid(k,t) - ΔV_ask(k,t) ]  / total_volume
```

**Limite critique :** L'étude porte sur **un seul jour** (2025-01-30) et un seul pair (BTC/USDT). La robustesse sur altcoins illiquides et sur plusieurs régimes de marché n'est pas établie. À utiliser comme feature additionnelle uniquement, pas comme signal principal.

---

### Papier 3 — Adaptive Multi-Agent Bitcoin Trading (Singhi, arXiv:2510.08068)
**Ratio innovation/faisabilité : ★★★★★ — Priorité 1**

L'architecture à 4 agents est directement transposable :
- **Quants** (≈ notre Agent StatArb) : indicateurs techniques, VWAP, MACD
- **Signals** (≈ notre Agent Sentiment LLM) : score de sentiment, news flow
- **Decision** (≈ notre Orchestrateur) : synthèse pondérée
- **Reflect** : feedback verbal hebdomadaire — *la vraie innovation*

Le mécanisme de **verbal feedback** est ce qui rend ce papier exceptionnel : le Reflect agent produit des critiques en langage naturel des décisions passées (ex : "Quants a surpondéré les signaux baissiers sans tenir compte du RSI neutre"). Ces critiques sont injectées dans les futurs prompts, permettant au système d'**apprendre sans retraining**.

Résultats sur BTC Jul 2024–Apr 2025 :
- Agent Quants : +30% vs buy-and-hold en phase haussière
- Agent Signals : +100% vs buy-and-hold en marché latéral
- Verbal feedback hebdomadaire : +31% de performance totale

**Application à HyperStat :** Implémenter le Reflect Agent comme un LLM (GPT-4o mini gratuit jusqu'à ~50k tokens/jour) qui reçoit chaque semaine un résumé agrégé des décisions et des métriques, et produit des ajustements des poids de l'orchestrateur en JSON.

**Limite :** Le backtesting couvre uniquement Bitcoin et une courte période. Sur altcoins avec moins de liquidité et plus de bruit, le LLM peut produire des critiques basées sur du bruit. Mitigation : ne passer au LLM que des métriques agrégées (Sharpe, DD, hit rate par régime), jamais des P&L bruts tick par tick.

---

### Papier 4 — Orchestration Framework (Li et al., arXiv:2512.02227)
**Ratio innovation/faisabilité : ★★★★☆ — Priorité 2**

C'est le blueprint architectural le plus complet. Le système map chaque composante d'un système de trading algorithmique à un agent LLM :

```
Planner → Orchestrator → [Alpha, Risk, Portfolio, Execution, Backtest, Audit] → Memory
```

La règle clé de ce papier : **les LLM ne voient jamais les P&L bruts ni les returns tick par tick**. Ils reçoivent uniquement des métriques agrégées (Sharpe, DD, volatilité, turnover) via le Memory Agent. Cela prévient le data leakage et la sur-adaptation.

Pour BTC (données minute, juillet-août 2025) : return 8.39%, Sharpe 0.38, max DD -2.80% contre buy-and-hold +3.80%.

**Application à HyperStat :** Structure du contexte JSON à passer entre agents :
```json
{
  "task_id": "hyperstat_bar_20250222_14h",
  "agent_role": "orchestrator",
  "regime": "sideways",
  "q_t": 0.75,
  "agent_signals": {
    "stat_arb": {"ARB": -0.42, "OP": 0.31, "SOL": -0.18},
    "ml_predictor": {"ARB": -0.35, "OP": 0.28, "SOL": -0.12},
    "sentiment_gate": 0.82
  },
  "metrics_last_7d": {"sharpe": 1.24, "dd": -0.021, "ic_mean": 0.041}
}
```

**Limite :** Sur des données minute BTC, la fenêtre d'évaluation est très courte (18 jours). Les résultats sur des altcoins en perp sont moins bien établis. Le coût API LLM peut devenir non trivial si l'orchestrateur est appelé trop fréquemment — à limiter à 1 appel par barre (1h) maximum.

---

### Papiers 5-8 — Analyse rapide

**Integrating High-Dimensional Technical Indicators (Bitcoin, Ethereum, Ripple)** — Utile pour la liste des features à inclure dans l'Agent ML Predictor (RSI, MACD, Bollinger, ADX, OBV, etc.). Valider avec IC et non avec accuracy brute. Éviter le data mining sur 100+ indicators sans correction statistique.

**Machine Learning Framework for Algorithmic Trading (Nayak)** — Fournit un framework générique ML → trading. La contribution utile est la formalisation du pipeline : feature selection → model selection → risk-adjusted evaluation. Confirme l'importance de l'IC (Information Coefficient) comme métrique première.

**Explainable Patterns in Cryptocurrency** — Directement applicable pour l'Audit Agent : utiliser SHAP values pour expliquer quelles features du ML Predictor ont le plus contribué à chaque signal. Rend le système auditables et facilite la détection de drift de features.

**Solving a Million-Step LLM Task With Zero Errors** — Moins directement applicable au trading crypto. Le concept de décomposition de tâches longues en sous-tâches avec vérification à chaque étape est utile pour le Backtest Agent et l'Audit Agent, mais ne justifie pas à lui seul un effort d'implémentation important.

---

## 3. Architecture proposée : HyperStat v2

### Formule de combinaison des agents

L'orchestrateur combine les signaux selon :

```
signal_final_i = w_stat · s_stat_i + w_ml · s_ml_i · sent_gate
```

avec les contraintes :
- `w_stat + w_ml = 1.0`
- `w_stat ∈ [0.5, 0.9]` (le stat-arb reste dominant par défaut)
- `w_ml ∈ [0.1, 0.5]`
- `sent_gate ∈ [0, 1]` (le sentiment réduit l'exposition, ne l'amplifie jamais)

La pondération `(w_stat, w_ml)` est ajustée par le régime détecté :

| Régime | w_stat | w_ml | Logique |
|---|---|---|---|
| MR stable (Q_MR = 1.0) | 0.80 | 0.20 | Confiance stat-arb |
| Vol élevée (Q_risk = 0.5) | 0.60 | 0.40 | ML meilleur sur trending |
| Choc détecté (Q_break = 0) | 0 | 0 | Flat — aucun agent ne trade |
| Sentiment gate < 0.3 | 0.90 | 0.10 | Incertitude — stat-arb uniquement |

---

### Train / Val / Test split

```
Historique disponible (ex: 6 mois)
├── Train    70% = 4.2 mois   (calibration FDS, fit LSTM, XGB)
├── Val      15% = 0.9 mois   (sélection hyperparamètres)
└── Test     15% = 0.9 mois   (évaluation finale, jamais vu)

Walk-forward live :
  Fenêtre glissante de 90 jours → retrain LSTM+XGB toutes les 2 semaines
  FDS : calibration continue (walk-forward 30j)
```

**Simulation temps-réel :** rejouer le Test set barre par barre avec le modèle entraîné sur Train+Val uniquement = proxy valide de la performance out-of-sample. Utiliser `backtest/engine.py` en mode "replay" avec horodatage strict.

---

## 4. Plan d'action par phases

### Phase 1 — ML Predictor (2-3 semaines)
**Objectif :** Ajouter un deuxième signal alpha complémentaire.

Fichiers à créer/modifier :
- `src/hyperstat/ml/lstm_xgb_predictor.py` — modèle LSTM+XGB
- `src/hyperstat/ml/feature_store.py` — features techniques (RSI, MACD, ADX, OBV)
- `src/hyperstat/backtest/engine.py` — intégrer le split train/val/test
- `src/hyperstat/strategy/allocator.py` — intégrer `directional_score`

Librairies Python : `torch` (LSTM), `xgboost`, `scikit-learn` (pipeline), `ta-lib` ou `pandas_ta` (indicators), `shap` (explainability)

Données gratuites : Hyperliquid historique (déjà dans le repo), CryptoCompare free tier pour news.

Métriques de succès :
- IC (Information Coefficient) > 0.03 sur le test set
- Sharpe sur test set ≥ 1.0 (vs backtests actuels)
- Pas de dégradation du MaxDD vs stat-arb seul

---

### Phase 2 — LLM Sentiment + Orchestrateur (3-4 semaines)
**Objectif :** Ajouter le gating sentiment et la boucle de feedback verbal.

Fichiers à créer :
- `src/hyperstat/agents/sentiment_agent.py` — scraping CryptoPanic + Fear & Greed, appel LLM
- `src/hyperstat/agents/orchestrator.py` — combinaison des signaux par régime
- `src/hyperstat/agents/reflect_agent.py` — feedback verbal hebdomadaire
- `src/hyperstat/agents/memory_agent.py` — stockage contexte JSON

LLM : GPT-4o mini via API Anthropic (Claude Haiku) ou Ollama local (Llama 3.1 8B gratuit). Coût estimé : <$5/mois avec appels 1h.

Données gratuites : CryptoPanic API (gratuit, 100 req/jour), Alternative.me Fear & Greed (gratuit, illimité).

Métriques de succès :
- Sentiment gate réduit les positions en période de panique (DD réduit)
- Verbal feedback améliore le Sharpe de ≥ 10% après 4 semaines de boucle

---

### Phase 3 — LOB + On-chain (optionnel, 4-6 semaines)
**Objectif :** Signal de microstructure pour améliorer le timing d'entrée.

Fichiers à créer :
- `src/hyperstat/ml/lob_feature_extractor.py` — Kalman/SG débruitage + OFI
- `src/hyperstat/data/bybit_lob_loader.py` — téléchargement données historiques Bybit

Données gratuites : Bybit historical LOB data (disponible sur leur site, limité à quelques jours).

Métriques de succès :
- Réduction slippage VWAP de ≥ 15% avec signal LOB
- Latence prédiction < 100ms (sinon inutilisable en live)

---

## 5. Métriques de performance globales

| Métrique | Actuel (estimé) | Cible Phase 1 | Cible Phase 2 |
|---|---|---|---|
| Sharpe ratio | < 0 (Sharpe négatif) | ≥ 1.0 | ≥ 1.5 |
| Max Drawdown | inconnu | ≤ -8% | ≤ -5% |
| IC moyen FDS | non mesuré | > 0.03 | > 0.04 |
| IC moyen ML | — | > 0.03 | > 0.04 |
| Hit rate (direction) | — | > 55% | > 58% |
| Slippage moyen (bps) | ~20 bps | ~15 bps | ~12 bps |
| Turnover ratio | non mesuré | ≈ 1.0x | ≈ 1.0x |
| % beating VWAP | 0% | > 50% | > 55% |

---

## 6. Contraintes respectées

**Open-source / low-cost :**
- `torch` + `xgboost` + `pandas_ta` : 100% open-source
- LLM : Claude Haiku ($0.25/M tokens) ou Llama local gratuit
- Données : Hyperliquid (gratuit), CryptoPanic gratuit, Fear & Greed gratuit, Bybit LOB gratuit

**Limites API :**
- Hyperliquid : 1100 req/min (déjà géré par `rate_limiter.py`)
- CryptoPanic : 100 req/jour → 1 scrape/14min max → compatible avec 1h timeframe
- LLM : 1 appel/heure maximum → <50 appels/jour → coût négligeable

**Modularité :**
- Chaque agent est un module Python indépendant
- L'orchestrateur est le seul point d'intégration → swap facile d'un agent
- Les poids `(w_stat, w_ml, sent_gate)` sont configurables dans le YAML

**Pas de lookahead :**
- Split train/val/test strict avec horodatage
- Le Memory Agent ne passe jamais les P&L bruts aux LLM
- Walk-forward avec re-fit périodique (pas de fit global)
