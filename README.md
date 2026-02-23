# HyperStat — Stat-Arb Altcoins + Funding Divergence Signal + Multi-Agent AI Layer

> **v2.0 — Architecture Multi-Agents IA**  
> HyperStat est un framework **backtest + paper + live** pour du **statistical arbitrage cross-sectionnel** sur des altcoins (perps Hyperliquid), enrichi d'un **Funding Divergence Signal (FDS)**, d'une couche optionnelle de **funding carry**, et désormais d'une **couche d'orchestration multi-agents IA** pour l'adaptation dynamique des stratégies en temps réel.

---

## Sommaire

- [1. Stratégie](#1-stratégie)
- [2. Architecture du code](#2-architecture-du-code)
- [3. **NEW — Architecture Multi-Agents IA**](#3-new--architecture-multi-agents-ia)
- [4. **NEW — Revue des Papiers Scientifiques & Applicabilité**](#4-new--revue-des-papiers-scientifiques--applicabilité)
- [5. **NEW — Pipeline Données Gratuites & Simulation Temps Réel**](#5-new--pipeline-données-gratuites--simulation-temps-réel)
- [6. **NEW — Plan d'Action & Roadmap Priorisée**](#6-new--plan-daction--roadmap-priorisée)
- [7. Installation](#7-installation)
- [8. Données](#8-données)
- [9. Lancer un backtest](#9-lancer-un-backtest)
- [10. Lancer en paper / live](#10-lancer-en-paper--live)
- [11. Dashboard Streamlit](#11-dashboard-streamlit)
- [12. Calibration du FDS](#12-calibration-du-fds)
- [13. Notes Hyperliquid (API)](#13-notes-hyperliquid-api)
- [14. **NEW — Métriques de Performance & Évaluation**](#14-new--métriques-de-performance--évaluation)
- [15. Bugs connus / TODO](#15-bugs-connus--todo)

---

## 1. Stratégie

### 1.1 Univers & Buckets

**Filtrage univers** :

- Dollar volume proxy : $DV_{i,t} = P_{i,t} \cdot V_{i,t}$
- Illiquidité d'Amihud : 

    $$ILLIQ_{i} = \text{median}_t\left(\frac{|r_{i,t}|}{DV_{i,t}+\varepsilon}\right)$$
    
- Exclusion des coins à funding instable (toxiques pour le carry)

**Buckets** — regroupement des coins structurellement similaires pour capter la dispersion intra-groupe :

Distance corrélation sur résidus BTC-neutralisés :

$$r_{i,t} = \beta_i r_{BTC,t} + \varepsilon_{i,t}$$

$$d_{ij}=\sqrt{\tfrac{1}{2}(1-\rho_{ij}^{\varepsilon})}$$

Clustering hiérarchique → $K$ buckets (4–6 dans la config par défaut).

---

### 1.2 Signal Mean-Reversion Cross-Sectionnel

Log-return sur horizon $H$ (ex: 12 barres × 5m = 1h) :

$$R_{i,t}^{(H)}=\ln\!\left(\frac{P_{i,t}}{P_{i,t-H}}\right)$$

Z-score robuste intra-bucket $B$ via **median/MAD** :

$$z_{i,t}=\frac{R_{i,t}^{(H)}-\text{median}_B(R_t^{(H)})}{\text{MAD}_B(R_t^{(H)})+\varepsilon}$$

Signal contrarian avec hystérésis anti-churn :

$$s_{i,t}=-\text{clip}(z_{i,t},[-z_{max},z_{max}])$$

**Règles d'hystérésis** :
- Entrée : $|z| > z_{in} = 1.5$
- Sortie : $|z| < z_{out} = 0.5$
- Min hold 30 min / Max hold 24h

---

### 1.3 Régime / Gating ($Q_t$)

$$Q_t = Q^{MR}_t \cdot Q^{liq}_t \cdot Q^{risk}_t \in [0,1]$$

#### $Q^{MR}_t$ — Mean reversion quality

Spread bucket top/bottom quantiles → AR(1) rolling :

$$S_{B,t}=a+b \cdot S_{B,t-1}+u_t$$

$$t_{1/2}=\frac{\ln 2}{-\ln b}\cdot \Delta t$$

| Half-life | $Q^{MR}$ |
|-----------|----------|
| < 30 min | 0.5 (trop rapide = bruit) |
| 30 min – 6h | 1.0 ✅ |
| 6h – 24h | 0.5 |
| > 24h | 0.0 |

#### $Q^{liq}_t$ — Liquidité

Rangs cross-section : $\text{rank}_{pct}(DV)$ élevé et $1 - \text{rank}_{pct}(ILLIQ)$ élevé. Médiane cross-section. Si < 0.3 → $Q^{liq} = 0$.

#### $Q^{risk}_t$ — Volatilité (BTC proxy)

Percentiles rolling sur 60 jours :

| Vol BTC | $Q^{risk}$ |
|---------|------------|
| < p90 | 1.0 ✅ |
| p90 – p95 | 0.5 |
| > p95 | 0.0 |

---

### 1.4 Funding Divergence Signal (FDS) — Alpha Original

> **Insight clé :** sur Hyperliquid, les traders on-chain ont un biais structurel **long**, créant une asymétrie persistante dans le funding. L'alpha vient de combiner trois dimensions que la littérature ne croise pas.

Le FDS est appliqué comme un **gate de confiance multiplicatif** entre le vol-scaling et les caps :

$$\boxed{w_{i,t}^{after,FDS} = w_{i,t}^{stat} \cdot \left(1 + \alpha_{fds} \cdot \text{FDS}_{i,t}\right)}$$

avec $\alpha_{fds} = 0.6$ (configurable).

#### Composante 1 — Carry cross-sectionnel (poids 0.35)

EWMA lente du funding (span $\tau_s = 72$ barres) puis z-score cross-sectionnel robuste :

$$\tilde{f}_{i,t}^{slow} = \text{EWMA}_{\tau_s}(f_{i,t})$$

$$s^{carry}_{i,t} = -\text{clip}\!\left(\frac{\tilde{f}_{i,t}^{slow} - \text{median}_j(\tilde{f}_{j,t}^{slow})}{\text{MAD}_j + \varepsilon},\ [-z_{max}, z_{max}]\right)$$

#### Composante 2 — Désalignement funding/return (poids 0.40) ⭐

$$\rho_{i,t} = \text{Corr}_W\!\left(z_{i,t}^{return},\ \Delta\tilde{f}_{i,t}^{fast}\right)$$

$$\text{tension}_{i,t} = 1 - |\rho_{i,t}| \in [0, 1]$$

#### Composante 3 — Vélocité du funding (poids 0.25)

$$v_{i,t} = \frac{\text{EWMA}_{\tau_f}(f_{i,t}) - \text{EWMA}_{\tau_s}(f_{i,t})}{|\text{EWMA}_{\tau_s}(f_{i,t})| + \varepsilon}$$

#### Assemblage FDS

$$\text{FDS}_{i,t}^{raw} = 0.35 \cdot s^{carry}_{i,t} + 0.40 \cdot s^{div}_{i,t} + 0.25 \cdot s^{vel}_{i,t}$$

Implémentation : `src/hyperstat/strategy/funding_divergence_signal.py`

---

### 1.5 Allocation & Neutralisation

Pipeline complet dans `allocator.py` :

1. **Vol-scaling** : $w_{i,t}^{raw} = \frac{s_{i,t}}{\hat{\sigma}_{i,t}}$
2. **Regime scaling** : $w \leftarrow Q_t \cdot w$
3. **Neutralisation + normalisation gross** via projection nullspace
4. **FDS gate** + re-neutralisation
5. **Caps** : coin (12%) / bucket (35%)
6. **Emergency flatten** : si $|z_i| > 3.5$
7. **Funding overlay**
8. **Contrainte finale** : gross ≤ 1.40

---

### 1.6 Funding Carry Overlay

$$\Pi^{fund}_i \approx -N_i \cdot f_i$$

$$w^{final} = w^{stat} + \eta \cdot Q^{fund} \cdot w^{fund}$$

---

### 1.7 Coûts & Réalisme

$$\text{slip}_{bps} = 8 + 10 \cdot RV_{1h}(\%)$$

$$\text{Cost}_t = \sum_i |\Delta w_{i,t}| \cdot \text{Equity}_t \cdot \frac{\text{fee}_{bps} + \text{slip}_{bps}}{10^4}$$

Paramètres par défaut (mode taker) : fee = 6 bps, slip base = 8 bps.

---

### 1.8 Risk Management

- Kill-switch drawdown intraday : si DD > 3% → flat + cooldown 12h
- Emergency flatten par coin : si $|z| > 3.5$ → target = 0
- Caps d'expo : coin (12%) / bucket (35%) / gross total (140%)

---

## 2. Architecture du code

```
hyperstat-arb-bot/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── streamlit/
│   └── config.toml
├── configs/
│   ├── default.yaml
│   ├── strategy_stat_arb.yaml
│   ├── hyperliquid_testnet.yaml
│   └── hyperliquid_mainnet.yaml
├── apps/
│   ├── dashboard.py
│   └── analyse.py
├── src/
│   └── hyperstat/
│       ├── core/          (clock, logging, types, math, risk)
│       ├── data/          (storage, loaders, features, universe)
│       ├── exchange/      (sandbox, hyperliquid REST+WS)
│       ├── strategy/      (stat_arb, regime, FDS, funding_overlay, allocator)
│       ├── backtest/      (engine, costs, metrics, reports)
│       ├── live/          (runner, order_manager, health)
│       ├── monitoring/    (risk_metrics, sink)
│       └── cli/
├── scripts/
└── tests/
```

---

## 3. NEW — Architecture Multi-Agents IA

### 3.1 Vue d'ensemble & Justification

L'objectif est de passer d'un bot déterministe à un **système adaptatif** où des agents spécialisés collaborent pour améliorer les décisions de trading en temps réel. L'architecture actuelle fournit une base solide (signaux bien définis, risk management, modulaire) mais manque d'adaptation dynamique aux changements de régime de marché non anticipés et d'exploitation des données non-structurées (news, sentiment).

### 3.2 Schéma de la Nouvelle Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYPERSTAT v2 — MULTI-AGENT LAYER                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 SUPERVISOR AGENT (Orchestrateur)              │  │
│  │   • Agrège les signaux de tous les agents                    │  │
│  │   • Alloue le capital entre les stratégies                   │  │
│  │   • Déclenche kill-switch global si consensus négatif        │  │
│  │   • Adapte les paramètres de Q_t en temps réel               │  │
│  └──────────────┬────────────────────────────────┬──────────────┘  │
│                 │                                │                  │
│    ┌────────────▼──────────┐      ┌─────────────▼────────────┐     │
│    │  SIGNAL AGENTS        │      │   RISK AGENT              │     │
│    │                       │      │                           │     │
│    │ ┌───────────────────┐ │      │ • VaR/CVaR temps réel    │     │
│    │ │ TechnicalAgent    │ │      │ • Corrélation dynamique   │     │
│    │ │ (stat-arb + FDS)  │ │      │ • Stop-loss adaptatif     │     │
│    │ │ [EXISTANT v1]     │ │      │ • Drawdown monitoring     │     │
│    │ └───────────────────┘ │      │ • Sizing via Kelly        │     │
│    │ ┌───────────────────┐ │      └───────────────────────────┘     │
│    │ │ SentimentAgent    │ │                                        │
│    │ │ (news + on-chain) │ │      ┌─────────────────────────────┐  │
│    │ │ [NEW]             │ │      │   REGIME AGENT (LLM-based)  │  │
│    │ └───────────────────┘ │      │                             │  │
│    │ ┌───────────────────┐ │      │ • Classifie: trending /     │  │
│    │ │ PredictionAgent   │ │      │   mean-reverting / crisis   │  │
│    │ │ (LSTM+XGBoost)    │ │      │ • Lit les news macro        │  │
│    │ │ [NEW]             │ │      │ • Ajuste Q_t en conséquence │  │
│    │ └───────────────────┘ │      │ [NEW]                       │  │
│    └───────────────────────┘      └─────────────────────────────┘  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                    COUCHE DONNÉES (enrichie)                        │
│  Hyperliquid API │ CryptoCompare (free) │ RSS News │ Fear&Greed    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Description des Agents

#### SupervisorAgent — Orchestrateur Central

**Rôle** : agrège les scores de confiance de chaque agent spécialisé et décide de l'allocation finale du capital entre les stratégies.

**Localisation** : `src/hyperstat/agents/supervisor.py`

```python
# Interface cible
class SupervisorAgent:
    def orchestrate(
        self,
        technical_signal: PortfolioWeights,
        sentiment_score: float,           # [-1, 1]
        regime_label: str,                # "trending" | "mean_reverting" | "crisis"
        prediction_confidence: float,     # [0, 1]
        risk_state: RiskState,
    ) -> AllocationDecision:
        """
        Combine les signaux via weighted confidence voting.
        Réduit l'exposition si consensus faible.
        Déclenche kill-switch global si regime_label == "crisis".
        """
```

**Logique de combinaison** :

$$w^{final} = w^{stat} \cdot \underbrace{(1 + \alpha_{fds} \cdot \text{FDS})}_{\text{existant}} \cdot \underbrace{(1 + \alpha_{sent} \cdot \text{SENT})}_{\text{nouveau}} \cdot \underbrace{Q^{regime}_{llm}}_{\text{nouveau}}$$

---

#### TechnicalAgent — Signal Existant (v1)

**Rôle** : encapsule le pipeline existant (stat-arb + FDS + regime + allocator). Devient un agent parmi d'autres.

**Localisation** : `src/hyperstat/agents/technical_agent.py` (wrapper de `strategy/`)

Aucune modification du code v1 requise — injection de dépendance propre.

---

#### SentimentAgent — Analyse News & On-Chain

**Rôle** : agrège des signaux de sentiment depuis des sources gratuites pour filtrer les entrées en période d'incertitude macro.

**Localisation** : `src/hyperstat/agents/sentiment_agent.py`

**Sources de données gratuites** :

| Source | Données | API | Limite |
|--------|---------|-----|--------|
| CryptoCompare | News crypto | Gratuit (clé requise) | 100k req/mois |
| Alternative.me | Fear & Greed Index | Gratuit sans clé | ~unlimited |
| CoinGecko | Trending coins, market data | Gratuit | 30 req/min |
| Hyperliquid WS | Open Interest, liquidations | Gratuit (existant) | déjà intégré |
| RSS feeds | Coindesk, Cointelegraph | Gratuit | unlimited |

**Pipeline de traitement** :

```python
class SentimentAgent:
    def compute_sentiment_score(self, ts: datetime) -> float:
        """
        1. Fetch Fear & Greed Index (0-100) → normalize [-1, 1]
        2. Parse RSS news → keyword scoring (bearish/bullish keywords)
        3. On-chain: ratio liquidations long/short depuis HL WS (existant)
        4. Aggregate avec poids: 0.4 * F&G + 0.3 * news + 0.3 * onchain
        Returns: score in [-1, 1]
           > 0.3  → sentiment haussier → réduire positions short
           < -0.3 → sentiment baissier → renforcer positions short
        """
```

**Librairies** : `feedparser`, `requests`, `transformers` (optionnel pour NLP plus avancé)

---

#### PredictionAgent — LSTM + XGBoost (Inspiré papier #1)

**Rôle** : produit une probabilité directionnelle à horizon H pour chaque coin de l'univers, utilisée comme signal de confiance additionnel.

**Localisation** : `src/hyperstat/agents/prediction_agent.py`

**Implémentation inspirée de** : *CRYPTO PRICE PREDICTION USING LSTM+XGBOOST* (arxiv 2506.22055)

```python
class PredictionAgent:
    """
    Modèle hybride LSTM (séquences temporelles) + XGBoost (features tabular).
    Train/Test split : 80% train, 20% test, walk-forward validation.
    Features : OHLCV + indicateurs techniques (RSI, MACD, BB, vol)
    Target : direction binaire sur H barres (5m × 12 = 1h)
    Output : P(hausse) ∈ [0,1] par coin
    """
    
    def train(self, candles: pd.DataFrame, test_ratio: float = 0.2):
        # Walk-forward : entraîner sur t-N à t-k, valider sur t-k à t
        pass
    
    def predict(self, candles_window: pd.DataFrame) -> dict[str, float]:
        # Retourne {symbol: probability_up} pour l'univers courant
        pass
```

**Train/Test protocol** :
- Données historiques : 60 jours minimum (script `download_history.py` existant)
- Split temporel strict (pas de look-ahead biais)
- Walk-forward sur fenêtres glissantes de 30 jours
- Métriques de validation : accuracy, F1, Brier score

**Intégration avec SupervisorAgent** :

$$\text{confidence\_boost}_{i} = 2 \cdot (P_{up,i} - 0.5) \in [-1, 1]$$

Si `confidence_boost > 0` et signal stat-arb long → renforcer. Si contradictoire → réduire.

---

#### RegimeAgent — Classification LLM-Based

**Rôle** : classifie le régime de marché courant en utilisant un LLM léger (ou des règles enrichies) pour interpréter les données macro + micro.

**Localisation** : `src/hyperstat/agents/regime_agent.py`

**Régimes identifiés** :

| Régime | Description | Action sur $Q_t$ |
|--------|-------------|-----------------|
| `mean_reverting` | Vol modérée, spreads stables | $Q_t = 1.0$ (nominal) |
| `trending` | Momentum fort, spreads qui dérivent | $Q_t = 0.3$ (réduction) |
| `high_vol` | BTC > p95 vol | $Q_t = 0.0$ (flat) |
| `crisis` | Liquidations massives, spread OI/vol extrême | Kill-switch global |
| `carry_favorable` | Funding élevé et stable → opportunité carry | Boost overlay funding |

**Approche sans LLM externe (open-source, low-cost)** :

```python
class RegimeAgent:
    """
    Option A (règles enrichies) : extension de regime.py existant
      + indicateurs on-chain (OI, liquidations via HL WS)
      + Fear & Greed Index comme input supplémentaire
    
    Option B (LLM local léger) : mistral-7B ou phi-3-mini via llama.cpp
      Prompt : contexte marché → classification régime
      Coût : 0$ si local, ~$0.001/call si API (GPT-4o-mini ou Gemini Flash)
    
    Option C (hybride recommandée) :
      Règles déterministes pour les cas clairs (crisis, high_vol)
      LLM uniquement pour les cas ambigus (trending vs mean_reverting)
    """
```

---

#### RiskAgent — Gestion Dynamique du Risque

**Rôle** : surveille en temps réel les métriques de risque et ajuste les paramètres du RiskState pour l'ensemble du portefeuille.

**Localisation** : `src/hyperstat/agents/risk_agent.py`

**Améliorations vs v1** :

- **Kelly Criterion adaptatif** : sizing basé sur le win rate rolling de la stratégie
- **Corrélation dynamique** : réduit les positions sur les coins fortement corrélés soudainement
- **Stop-loss par régime** : DD toléré plus bas en régime `high_vol`
- **CVaR-constrained sizing** : limite la perte en queue de distribution

```python
@dataclass
class DynamicRiskParams:
    max_dd_intraday: float = 0.03        # existant
    dd_by_regime: dict = field(default_factory=lambda: {
        "mean_reverting": 0.03,
        "trending": 0.02,
        "high_vol": 0.01,
        "crisis": 0.0,
    })
    kelly_fraction: float = 0.25         # demi-Kelly conservateur
    max_corr_pair: float = 0.85          # réduction si corrélation > 85%
```

---

### 3.4 Flux de Données Multi-Agents

```
Hyperliquid WS/REST (existant)
        │
        ├──► TechnicalAgent ──► stat-arb signal + FDS gate
        │
        ├──► SentimentAgent ──► Fear&Greed + news RSS + liquidations HL
        │
        ├──► PredictionAgent ──► LSTM+XGBoost probability
        │
        └──► RegimeAgent ──────► classification régime + Q_t adaptatif
                │
                └──► Toutes sources ci-dessus
                         │
                         ▼
                 SupervisorAgent
                         │
                    ┌────▼────┐
                    │ RiskAgent│ ← VaR/CVaR/Kelly constraints
                    └────┬────┘
                         │
                    Allocator v2
                         │
                    ┌────▼────┐
                    │  Live   │ → Hyperliquid execution
                    │ Runner  │
                    └─────────┘
```

---

### 3.5 Nouveau dossier `agents/`

```
src/hyperstat/agents/
├── __init__.py
├── base_agent.py              # ABC : interface commune (observe, act, score)
├── supervisor.py              # Orchestrateur central
├── technical_agent.py         # Wrapper TechnicalAgent (v1 encapsulé)
├── sentiment_agent.py         # Fear&Greed + news + on-chain
├── prediction_agent.py        # LSTM + XGBoost (train/predict/backtest)
├── regime_agent.py            # Classification régime (règles + LLM optionnel)
├── risk_agent.py              # Gestion dynamique du risque
└── utils/
    ├── llm_client.py          # Client LLM (local llama.cpp ou API)
    ├── news_fetcher.py        # RSS + CryptoCompare fetch
    └── fear_greed.py          # Alternative.me API wrapper
```

---

## 4. NEW — Revue des Papiers Scientifiques & Applicabilité

### 4.1 CRYPTO PRICE PREDICTION USING LSTM+XGBOOST (arxiv 2506.22055v1)

**Concepts applicables** :
- Architecture hybride LSTM (capture des dépendances temporelles longues) + XGBoost (features tabular non-linéaires) → implémentation dans `PredictionAgent`
- Features : retours logarithmiques, volatilité réalisée, volume normalisé — cohérents avec `data/features.py` existant
- Normalisation MinMaxScaler par fenêtre glissante → à intégrer dans le preprocessing

**Limites / biais à éviter** :
- Fréquence daily dans le papier vs 5min dans HyperStat → recalibrer les hyperparamètres LSTM (séquence length, epochs)
- Risque de overfitting sur BTC uniquement → appliquer sur chaque bucket séparément avec modèles légers
- Pas de transaction costs dans les métriques → toujours évaluer net of fees dans notre backtest

**Ratio innovation/faisabilité** : ⭐⭐⭐⭐⭐ — **PRIORITÉ 1**. Librairies disponibles (PyTorch/TensorFlow + XGBoost), données déjà collectées.

---

### 4.2 Explainable Patterns in Cryptocurrency

**Concepts applicables** :
- SHAP values pour expliquer quelles features du FDS contribuent le plus aux décisions → `FDSDiagnostics` enrichi
- Identification de patterns temporels récurrents (heure du jour, jour de semaine) → features calendar dans PredictionAgent
- Anomaly detection via isolation forest → filtre pré-signal pour éviter les entrées sur données corrompues

**Limites** :
- Focus sur patterns passés → pas de garantie de stabilité dans le temps → walk-forward obligatoire
- Attention aux patterns exploités par de nombreux acteurs → alpha decay

**Ratio innovation/faisabilité** : ⭐⭐⭐⭐ — **PRIORITÉ 2**. SHAP déjà utilisable sur le FDS existant.

---

### 4.3 SOLVING A MILLION-STEP LLM TASK WITH ZERO ERRORS

**Concepts applicables** :
- Framework d'orchestration d'agents LLM avec vérification d'erreurs → architecture du SupervisorAgent
- Gestion de l'état long-terme pour les agents (mémoire des décisions passées) → `AgentMemory` dans base_agent.py
- Mécanismes de retry et fallback automatique → résilience du live runner

**Limites** :
- Coûts computationnels élevés pour des tasks millions de steps → adapter à notre contexte (décisions toutes les 5min)
- Latence LLM incompatible avec HFT → utiliser LLM uniquement pour les décisions lentes (régime, sizing global), pas tick-by-tick

**Ratio innovation/faisabilité** : ⭐⭐⭐ — **PRIORITÉ 3**. Utile pour la robustesse de l'orchestration mais complexité élevée.

---

### 4.4 Microstructural Dynamics in Cryptocurrency Limit Order Books

**Concepts applicables** :
- Features order book (bid-ask spread, depth imbalance, order flow imbalance) → enrichissement du signal de liquidité $Q^{liq}_t$
- "Better Inputs Matter More Than Stacking Another Hidden Layer" → message clé : améliorer les features avant la complexité du modèle
- Indicateur de toxicité du flow (VPIN-like) → meilleur proxy d'illiquidité vs Amihud seul

**Limites** :
- Données L2 order book nécessaires → vérifier disponibilité via Hyperliquid WS (endpoint `l2Book`)
- Coûts de stockage pour données L2 tick-by-tick → agréger en features résumées

**Ratio innovation/faisabilité** : ⭐⭐⭐⭐ — **PRIORITÉ 2**. Hyperliquid WS fournit l2Book gratuitement.

---

### 4.5 Integrating High-Dimensional Technical Indicators (Bitcoin, Ethereum, Ripple)

**Concepts applicables** :
- Feature selection via importance XGBoost → identifier les 10-15 indicateurs techniques les plus prédictifs parmi ~100
- PCA / Factor analysis pour réduire la dimensionnalité → cohérent avec roadmap "PCA eigenportfolios"
- Validation sur multiple crypto (pas seulement BTC) → déjà notre cas avec l'univers cross-sectionnel

**Limites** :
- Focus BTC/ETH/XRP → généralisation sur altcoins moins liquides à valider
- Indicateurs techniques standard (RSI, MACD) peu différenciants seuls → combiner avec FDS

**Ratio innovation/faisabilité** : ⭐⭐⭐ — **PRIORITÉ 3**. Enrichit le PredictionAgent.

---

### 4.6 An Adaptive Multi-Agent Bitcoin Trading System (Singhi)

**Concepts applicables** :
- Architecture multi-agents avec spécialisation par régime → directement applicable au SupervisorAgent
- Communication inter-agents via message bus → `AgentBus` dans notre architecture
- Mécanisme de vote pondéré par performance récente → supervisor alloue plus de poids aux agents qui ont raison dernièrement

**Limites** :
- Contexte BTC uniquement → adapter pour cross-sectional altcoins
- Pas de contraintes de coûts/latence explicites

**Ratio innovation/faisabilité** : ⭐⭐⭐⭐⭐ — **PRIORITÉ 1**. Blueprint le plus proche de notre architecture cible.

---

### 4.7 Orchestration Framework for Financial Agents (Li, Grover et al.)

**Concepts applicables** :
- Distinction algorithmic trading vs agentic trading → nous sommes en transition entre les deux
- Framework de communication standardisé entre agents (protocol A2A) → interface `BaseAgent` avec `observe()`, `act()`, `score()`
- Monitoring de la fiabilité de chaque agent → confidence tracking dans le SupervisorAgent

**Limites** :
- Framework orienté LLM heavy → adapter pour notre contexte majoritairement quantitatif
- Coûts d'inférence LLM à budgéter

**Ratio innovation/faisabilité** : ⭐⭐⭐⭐ — **PRIORITÉ 2**. Fournit le cadre architectural.

---

### 4.8 Machine Learning Framework for Algorithmic Trading (Nayak et al.)

**Concepts applicables** :
- Pipeline ML standardisé (feature engineering → model → evaluation → deployment) → structure de PredictionAgent
- Ensemble methods : Random Forest + Gradient Boosting → alternative ou complément à XGBoost seul
- Backtest avec transaction costs et slippage → déjà implémenté dans notre `costs.py`

**Limites** :
- Contexte actions/forex → adapter les features crypto-spécifiques (funding, on-chain)
- Pas de composante temps réel

**Ratio innovation/faisabilité** : ⭐⭐⭐ — **PRIORITÉ 3**. Références méthodologiques solides.

---

### 4.9 Synthèse des Priorités

| Priorité | Papier | Ce qu'on implémente | Agent cible |
|----------|--------|---------------------|-------------|
| P1 ⭐⭐⭐⭐⭐ | LSTM+XGBoost | Modèle prédictif directionnel | PredictionAgent |
| P1 ⭐⭐⭐⭐⭐ | Multi-Agent (Singhi) | Architecture supervisor + vote | SupervisorAgent |
| P2 ⭐⭐⭐⭐ | Order Book Micro | Features L2 pour $Q^{liq}$ | TechnicalAgent |
| P2 ⭐⭐⭐⭐ | Orchestration (Li) | Interface BaseAgent standardisée | Tous agents |
| P2 ⭐⭐⭐⭐ | Explainable Patterns | SHAP sur FDS | FDSDiagnostics |
| P3 ⭐⭐⭐ | High-Dim Indicators | Feature selection XGBoost | PredictionAgent |
| P3 ⭐⭐⭐ | ML Framework | Pipeline standardisé | PredictionAgent |
| P3 ⭐⭐⭐ | Million-Step LLM | Orchestration résiliente | SupervisorAgent |

---

## 5. NEW — Pipeline Données Gratuites & Simulation Temps Réel

### 5.1 Sources de Données Gratuites

```
données/
├── OHLCV 5min ──► Hyperliquid REST (existant, download_history.py)
├── Funding rates ──► Hyperliquid REST (existant)
├── Order Book L2 ──► Hyperliquid WS (endpoint: l2Book) [à activer]
├── Fear & Greed ──► https://api.alternative.me/fng/ [GRATUIT, sans clé]
├── News RSS ──────► Coindesk / Cointelegraph RSS feeds [GRATUIT]
├── Trending ──────► CoinGecko /trending [GRATUIT, 30 req/min]
└── Macro news ────► CryptoCompare Data API [clé gratuite, 100k/mois]
```

### 5.2 Split Train/Test & Walk-Forward

```
Timeline de données (exemple 90 jours) :

|←────── TRAIN (72j) ──────→|←── TEST (18j) ──→|
|                            |                  |
Day 0                      Day 72            Day 90

Walk-forward validation (fenêtres de 30j, step de 7j) :
├── Window 1 : train [0-60] test [60-90]
├── Window 2 : train [7-67] test [67-97]
├── Window 3 : train [14-74] test [74-104]
└── ...

Implémentation : scripts/walk_forward_validation.py [à créer]
```

### 5.3 Simulation Temps Réel (Paper Trading Enrichi)

Le mode `paper` existant est étendu avec une simulation temps réel plus réaliste :

```python
# apps/realtime_sim.py [à créer]
class RealtimeSimulator:
    """
    Rejoue les données historiques à vitesse accélérée (x100, x1000)
    pour simuler le comportement du bot en conditions réelles.
    
    Fonctionnalités :
    - Replay bar-by-bar avec latence simulée
    - Tous les agents actifs (sentiment, prediction, regime)
    - Métriques en temps réel dans le dashboard Streamlit
    - Comparaison vs baseline (v1 sans agents IA)
    """
    
    def run(
        self,
        data_path: str,
        speed_multiplier: int = 100,
        mode: Literal["replay", "live"] = "replay"
    ):
        pass
```

**Lancement** :
```bash
# Simulation replay (données historiques, vitesse x100)
python apps/realtime_sim.py --mode replay --speed 100 --days 30

# Simulation live (données Hyperliquid en temps réel, mode paper)
python apps/realtime_sim.py --mode live --config configs/hyperliquid_testnet.yaml
```

---

## 6. NEW — Plan d'Action & Roadmap Priorisée

### 6.1 Phase 0 — Corrections Bugs Critiques (Semaine 1)

**Objectif** : avoir un backtest valide avant d'ajouter de la complexité.

- [ ] **Bug 1** : Créer `src/hyperstat/data/features.py` avec `compute_returns`, `compute_ewma_vol`, `compute_rv_1h_pct`
- [ ] **Bug 2** : Passer `funding_rates` à l'allocator dans `engine.py` (ligne ~210)
- [ ] **Bug 3** : Mettre à jour `test_strategy_smoke.py` avec la bonne signature `BacktestConfig`
- [ ] **Bug 4** : Lire la section `funding_divergence_signal` dans `backtest_config_from_config()`
- [ ] **Validation** : lancer un backtest complet sur 60 jours et vérifier les métriques

### 6.2 Phase 1 — Foundation Agents (Semaines 2-3)

**Objectif** : créer le squelette multi-agents sans changer la logique de trading.

- [ ] Créer `src/hyperstat/agents/base_agent.py` (interface ABC)
- [ ] Créer `src/hyperstat/agents/technical_agent.py` (wrapper v1 existant)
- [ ] Créer `src/hyperstat/agents/supervisor.py` (orchestrateur simple, combine 1 agent au début)
- [ ] Créer `src/hyperstat/agents/risk_agent.py` (Kelly + corrélation dynamique)
- [ ] Tests unitaires : `tests/test_agents_smoke.py`

### 6.3 Phase 2 — SentimentAgent & Data (Semaines 4-5)

**Objectif** : enrichir les inputs avec données gratuites externes.

- [ ] Créer `src/hyperstat/agents/utils/fear_greed.py` (wrapper Alternative.me)
- [ ] Créer `src/hyperstat/agents/utils/news_fetcher.py` (RSS + CryptoCompare)
- [ ] Créer `src/hyperstat/agents/sentiment_agent.py`
- [ ] Activer l2Book dans `exchange/hyperliquid/ws_client.py` (order book features)
- [ ] Ajouter features order book dans `data/features.py`
- [ ] Backtest avec SentimentAgent : mesurer impact sur Sharpe

### 6.4 Phase 3 — PredictionAgent LSTM+XGBoost (Semaines 6-8)

**Objectif** : signal prédictionnel ML intégré.

- [ ] Créer `src/hyperstat/agents/prediction_agent.py`
- [ ] Implémenter LSTM (PyTorch Lightning) + XGBoost pipeline
- [ ] Créer `scripts/walk_forward_validation.py`
- [ ] Calibrer sur données Hyperliquid (objectif accuracy > 55% net of random)
- [ ] Intégrer dans SupervisorAgent comme signal additionnel
- [ ] Backtest comparatif : v1 vs v2 avec PredictionAgent

### 6.5 Phase 4 — RegimeAgent & Simulation Temps Réel (Semaines 9-11)

**Objectif** : adaptation dynamique aux régimes + simulation réaliste.

- [ ] Créer `src/hyperstat/agents/regime_agent.py` (règles enrichies d'abord)
- [ ] Intégrer Fear & Greed + OI/liquidations comme inputs du régime
- [ ] Créer `apps/realtime_sim.py` (replay accéléré)
- [ ] Étendre le dashboard Streamlit : scores des agents, régime courant, comparaison agents
- [ ] Tests en paper trading sur testnet Hyperliquid (2 semaines minimum)

### 6.6 Phase 5 — Optimisation & LLM Optionnel (Semaines 12+)

- [ ] Walk-forward calibration globale des paramètres
- [ ] LLM léger pour RegimeAgent (phi-3-mini local ou Gemini Flash API)
- [ ] PCA eigenportfolios (roadmap existante)
- [ ] Clustering dynamique des buckets
- [ ] Betas BTC rolling pour beta-neutralisation

---

## 7. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pyarrow plotly streamlit scipy

# Nouvelles dépendances pour la couche agents IA
pip install torch lightning xgboost shap feedparser
pip install scikit-learn ta-lib  # indicateurs techniques
```

Variables d'environnement :

```bash
export HL_ADDRESS=0x...
export HL_PRIVATE_KEY=0x...

# Optionnel (gratuit, requis pour news enrichies)
export CRYPTOCOMPARE_API_KEY=...  # https://min-api.cryptocompare.com (gratuit)
```

---

## 8. Données

```bash
# Télécharger l'historique altcoins Hyperliquid
python scripts/download_history.py --config configs/default.yaml --days 60

# Construire l'univers + buckets
python scripts/build_universe.py --config configs/default.yaml

# [NEW] Walk-forward split pour ML
python scripts/walk_forward_validation.py --config configs/default.yaml --window 30 --step 7
```

Les données sont stockées en parquet dans `./artifacts/data/`.

---

## 9. Lancer un backtest

```python
from hyperstat.backtest.engine import run_backtest, BacktestConfig

cfg = BacktestConfig(
    timeframe="5m",
    base_factor_symbol="BTC",
    initial_equity=1500.0,
    test_ratio=0.2,   # [NEW] 80% train, 20% test
)

report = run_backtest(
    cfg=cfg,
    candles_by_symbol=candles,
    funding_by_symbol=funding,
    buckets=buckets,
    stat_arb=stat_arb,
    regime_model=regime_model,
    allocator=allocator,
    # [NEW] agents optionnels
    supervisor=supervisor,
)

print(report.metrics)
report.equity_curve.plot()
```

---

## 10. Lancer en paper / live

```bash
# Paper trading
python -m hyperstat.main --config configs/default.yaml \
  --config configs/hyperliquid_testnet.yaml --mode paper

# [NEW] Paper avec agents IA
python -m hyperstat.main --config configs/default.yaml \
  --config configs/hyperliquid_testnet.yaml --mode paper --enable-agents

# Live
bash scripts/run_live.sh

# [NEW] Simulation replay temps réel
python apps/realtime_sim.py --mode replay --speed 100 --days 30
```

---

## 11. Dashboard Streamlit

```bash
streamlit run apps/dashboard.py
```

**v2 — Nouvelles sections du dashboard** :
- Scores des agents en temps réel (TechnicalAgent, SentimentAgent, PredictionAgent)
- Régime courant détecté (RegimeAgent)
- Contribution de chaque agent à la décision finale
- Comparaison equity curve : v1 (stat-arb seul) vs v2 (multi-agents)
- Fear & Greed Index historique
- Tableau SHAP : features les plus contributives au FDS

---

## 12. Calibration du FDS

```python
from hyperstat.strategy.funding_divergence_signal import (
    FundingDivergenceSignal, FDSConfig, FDSDiagnostics
)

fds  = FundingDivergenceSignal(FDSConfig())
diag = FDSDiagnostics(fds)

ic = diag.signal_ic(returns_df, funding_df, forward_horizon=8)
print(f"IC moyen : {ic.mean():.4f}   t-stat : {ic.mean()/ic.std()*len(ic)**0.5:.2f}")
```

**Ordre de calibration** :
1. `divergence_window` : 12–48 barres
2. `span_funding_fast` : 4–16 barres
3. `w_carry / w_divergence / w_velocity`
4. `gate_scale` (commencer à 0.4, augmenter si IC confirme)

---

## 13. Notes Hyperliquid (API)

- **Auth** : signature EIP-712 avec clé privée (`HL_PRIVATE_KEY`)
- **Rate limit** : 1200 req/min (on utilise 1100 avec marge) — déjà géré dans `rate_limiter.py`
- **WS** : max 10 connexions, max 1000 subscriptions
- **Funding** : toutes les 8h sur Hyperliquid
- **Perps disponibles** : ~150 altcoins en perpetual futures
- **Frais taker** : ~0.035% (3.5 bps) — config utilise 6 bps (conservateur)
- **[NEW] l2Book** : disponible via WS, activer pour features order book

---

## 14. NEW — Métriques de Performance & Évaluation

### 14.1 Métriques de la Stratégie (Backtest)

| Métrique | Cible v1 | Cible v2 (avec agents) | Méthode |
|----------|----------|------------------------|---------|
| Sharpe Ratio (net) | > 1.5 | > 2.0 | Annualisé sur période test |
| CAGR net of fees | > 15% | > 25% | Sur période test uniquement |
| Max Drawdown | < 15% | < 12% | Intraday + overnight |
| Turnover ratio | < 3x/jour | < 3x/jour | Gross turnover / equity |
| IC FDS moyen | > 0.03 | > 0.03 | Spearman, forward H barres |
| Win Rate | > 52% | > 55% | Trades net of fees |

### 14.2 Métriques ML (PredictionAgent)

| Métrique | Seuil minimal | Seuil cible |
|----------|--------------|-------------|
| Accuracy directionnelle | > 52% | > 56% |
| F1-score (walk-forward) | > 0.50 | > 0.55 |
| Brier Score | < 0.25 | < 0.22 |
| Stability (std Sharpe walk-forward) | < 0.5 | < 0.3 |

### 14.3 Métriques des Agents

| Agent | Métrique principale | Seuil cible |
|-------|---------------------|-------------|
| SentimentAgent | Corrélation score / rendement H+4 | > 0.10 |
| RegimeAgent | Accuracy classification | > 65% |
| PredictionAgent | IC Spearman directionnel | > 0.04 |
| SupervisorAgent | Sharpe incrémental vs baseline | > +0.3 |

### 14.4 Protocole d'Évaluation

```
1. Phase offline (backtest walk-forward) :
   - Toujours séparer train / test temporellement
   - Jamais de look-ahead bias (pas d'utilisation de données futures)
   - Répliquer sur au moins 3 fenêtres walk-forward
   
2. Phase paper trading (simulation temps réel) :
   - Minimum 2 semaines sur testnet Hyperliquid
   - Comparer v1 (stat-arb seul) vs v2 (multi-agents)
   - Logger toutes les décisions des agents pour analyse post-hoc

3. Phase live (si paper trading concluant) :
   - Commencer avec équity réduite (ex: 10% du capital cible)
   - Augmenter progressivement si Sharpe > 1.5 sur 30 jours
```

---

## 15. Bugs connus / TODO

### Bugs critiques (Phase 0 — à corriger avant tout backtest)

**Bug 1 — Module `hyperstat.data.features` manquant**

`backtest/engine.py` importe `compute_returns`, `compute_ewma_vol`, `compute_rv_1h_pct` depuis `hyperstat.data.features` qui n'existe pas. Créer ce module.

**Bug 2 — `funding_rates` non transmis à l'allocator dans engine.py**

```python
# Dans engine.py, ligne ~210, remplacer :
target = self.allocator.allocate(ts=..., signal=signal, regime=regime, ...)
# par :
funding_at_ts = self.funding_events.get(ts, {})
target = self.allocator.allocate(ts=..., signal=signal, regime=regime,
                                  funding_rates=funding_at_ts or None, ...)
```

**Bug 3 — Test smoke désynchronisé avec l'API réelle**

`test_strategy_smoke.py` appelle `BacktestConfig(run_name=..., out_dir=..., exec_mode=...)` — paramètres inexistants.

**Bug 4 — FDS non initialisé depuis le YAML**

`backtest_config_from_config()` ne lit pas la section `funding_divergence_signal` du YAML.

### TODO v2 (Agents IA)

- Implémenter `src/hyperstat/agents/` (Phase 1–4 ci-dessus)
- Activer l2Book Hyperliquid WS pour features microstructure
- `scripts/walk_forward_validation.py`
- `apps/realtime_sim.py` (simulation replay)
- Extension dashboard Streamlit avec métriques agents
- Tests d'intégration agents : `tests/test_supervisor_integration.py`

### Roadmap existante (inchangée)

- Walk-forward calibration FDS
- Clustering dynamique des buckets
- PCA eigenportfolios
- Illiquidité Amihud rolling dans le régime
- Betas BTC rolling pour beta-neutralisation
- Réconciliation positions dans `order_manager.py`

---

## About

Trading Bot crypto statistical arbitrage — framework backtest + paper + live sur Hyperliquid, avec couche multi-agents IA.

**Stack** : Python 3.11+ · PyTorch · XGBoost · Streamlit · Hyperliquid API (REST + WebSocket)

**Licence** : MIT
