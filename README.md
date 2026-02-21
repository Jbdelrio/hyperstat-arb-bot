# HyperStat — Stat-Arb Altcoins + Funding Divergence Signal (Hyperliquid)

HyperStat est un framework **backtest + paper + live** pour du **statistical arbitrage cross-sectionnel** sur des altcoins (perps Hyperliquid), enrichi d'un **Funding Divergence Signal (FDS)** comme gate de confiance multiplicatif, et d'une couche optionnelle de **funding carry**.

Le projet est conçu pour :
- **Rester réaliste** : coûts (fees + slippage proxy vol-dépendant), turnover, caps d'exposition, kill-switch drawdown.
- **Être itératif** : la même logique *data → signal → FDS gate → sizing → risk → execution* fonctionne en backtest puis en live.
- **Être monitorable** : dashboard Streamlit dark (PnL, positions, VaR/CVaR, corr, Sharpe, drawdown, FDS scores).

---

## Sommaire
- [1. Stratégie](#1-stratégie)
  - [1.1 Univers & Buckets](#11-univers--buckets)
  - [1.2 Signal Mean-Reversion Cross-Sectionnel](#12-signal-mean-reversion-cross-sectionnel)
  - [1.3 Régime / Gating (Q_t)](#13-régime--gating-q_t)
  - [1.4 Funding Divergence Signal (FDS) — Alpha Original](#14-funding-divergence-signal-fds--alpha-original)
  - [1.5 Allocation & Neutralisation](#15-allocation--neutralisation)
  - [1.6 Funding Carry Overlay](#16-funding-carry-overlay)
  - [1.7 Coûts & Réalisme](#17-coûts--réalisme)
  - [1.8 Risk Management](#18-risk-management)
- [2. Architecture du code](#2-architecture-du-code)
- [3. Installation](#3-installation)
- [4. Données](#4-données)
- [5. Lancer un backtest](#5-lancer-un-backtest)
- [6. Lancer en paper / live](#6-lancer-en-paper--live)
- [7. Dashboard Streamlit](#7-dashboard-streamlit)
- [8. Calibration du FDS](#8-calibration-du-fds)
- [9. Notes Hyperliquid (API)](#9-notes-hyperliquid-api)
- [10. Bugs connus / TODO](#10-bugs-connus--todo)
- [11. Roadmap](#11-roadmap)

---

## 1. Stratégie

### 1.1 Univers & Buckets

**Filtrage univers** :
- Dollar volume proxy : $DV_{i,t} = P_{i,t} \cdot V_{i,t}$
- Illiquidité d'Amihud : $ILLIQ_i = \text{median}_t\!\left(\frac{|r_{i,t}|}{DV_{i,t}+\varepsilon}\right)$
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
$$\boxed{w_{i,t}^{after\,FDS} = w_{i,t}^{stat} \cdot \left(1 + \alpha_{fds} \cdot \text{FDS}_{i,t}\right)}$$

avec $\alpha_{fds} = 0.6$ (configurable). Si $\text{FDS} = 0$ → identité. Si $\text{FDS} > 0$ → renforce le pari contrarian. Si $\text{FDS} < 0$ → atténue.

#### Composante 1 — Carry cross-sectionnel (poids 0.35)

EWMA lente du funding (span $\tau_s = 72$ barres) puis z-score cross-sectionnel robuste :
$$\tilde{f}_{i,t}^{slow} = \text{EWMA}_{\tau_s}(f_{i,t})$$
$$s^{carry}_{i,t} = -\text{clip}\!\left(\frac{\tilde{f}_{i,t}^{slow} - \text{median}_j(\tilde{f}_{j,t}^{slow})}{\text{MAD}_j + \varepsilon},\ [-z_{max}, z_{max}]\right)$$

Signe négatif : funding élevé $\Rightarrow$ longs overcrowdés $\Rightarrow$ signal short contrarian.

#### Composante 2 — Désalignement funding/return (poids 0.40) ⭐

C'est la composante originale. On mesure la **tension** entre direction des prix et direction du funding :

Vélocité du funding (EWMA rapide, span $\tau_f = 8$ barres) :
$$\Delta\tilde{f}_{i,t}^{fast} = \text{EWMA}_{\tau_f}(f_{i,t}) - \text{EWMA}_{\tau_f}(f_{i,t-1})$$

Corrélation rolling sur fenêtre $W = 24$ barres :
$$\rho_{i,t} = \text{Corr}_W\!\left(z_{i,t}^{return},\ \Delta\tilde{f}_{i,t}^{fast}\right)$$

Tension maximale quand $\rho \approx 0$ (signaux contradictoires) :
$$\text{tension}_{i,t} = 1 - |\rho_{i,t}| \in [0, 1]$$

Signe de la divergence :
$$\text{sign}_{i,t} = -\text{sgn}(z_{i,t}^{return}) \cdot \text{sgn}(\Delta\tilde{f}_{i,t}^{fast})$$

$$s^{div}_{i,t} = \text{clip}\!\left(\text{sign}_{i,t} \cdot \text{tension}_{i,t},\ [-1, 1]\right)$$

**Intuition :** si le prix monte ($r > 0$) mais le funding recule ($\Delta f < 0$), la hausse n'est pas crédible → signal contrarian renforcé. Si les deux sont alignés → tension faible, signal neutre.

#### Composante 3 — Vélocité du funding (poids 0.25)

Ratio fast/slow = mesure de l'accélération :
$$v_{i,t} = \frac{\text{EWMA}_{\tau_f}(f_{i,t}) - \text{EWMA}_{\tau_s}(f_{i,t})}{|\text{EWMA}_{\tau_s}(f_{i,t})| + \varepsilon}$$

$$s^{vel}_{i,t} = -\text{clip}\!\left(\frac{v_{i,t}}{v_{max}},\ [-1, 1]\right)$$

Funding qui accélère vers le haut → longs overcrowdés → signal short (contrarian).

#### Assemblage FDS

$$\text{FDS}_{i,t}^{raw} = 0.35 \cdot s^{carry}_{i,t} + 0.40 \cdot s^{div}_{i,t} + 0.25 \cdot s^{vel}_{i,t}$$

Normalisation cross-sectionnel finale :
$$\text{FDS}_{i,t} = \text{clip}\!\left(\frac{\text{FDS}_{i,t}^{raw} - \text{median}_j(\text{FDS}_{j,t}^{raw})}{2 \cdot \text{MAD}_j + \varepsilon},\ [-1, 1]\right)$$

Implémentation : `src/hyperstat/strategy/funding_divergence_signal.py`
- `FundingDivergenceSignal` : version pandas (backtest/calibration)
- `FundingDivergenceSignalLive` : version step-by-step (live/paper)
- `FDSDiagnostics` : IC, component breakdown, turnover impact

---

### 1.5 Allocation & Neutralisation

Pipeline complet dans `allocator.py` :

**1. Vol-scaling** :
$$w_{i,t}^{raw} = \frac{s_{i,t}}{\hat{\sigma}_{i,t}} \quad \text{(EWMA vol)}$$

**2. Regime scaling** : $w \leftarrow Q_t \cdot w$

**3. Neutralisation + normalisation gross** :

Projection dans le nullspace des contraintes $Aw = 0$ :
$$w \leftarrow \left(I - A^\top(AA^\top)^{-1}A\right)w$$

avec $A$ encodant dollar-neutral ($\sum w_i = 0$) et beta-neutral ($\sum w_i \beta_i = 0$).

**3.5. FDS gate** (entre vol-scaling et caps) :
$$w_i \leftarrow w_i \cdot (1 + \alpha_{fds} \cdot \text{FDS}_i)$$
Suivi d'une **re-neutralisation obligatoire** (le gate casse la neutralité).

**4. Caps** :
- Par coin : $|w_i| \leq 0.12$
- Par bucket : $\sum_{i \in b} |w_i| \leq 0.35$

**5. Emergency flatten** : si $|z_i| > 3.5$ → $w_i = 0$

**6. Funding overlay** : ajout de la couche carry (voir §1.6)

**7. Contrainte finale** : $\text{gross} \leq \text{gross}_{stat} + \text{gross}_{fund} = 1.40$

---

### 1.6 Funding Carry Overlay

Le funding est un carry périodique. Avec notional $N_i$ et funding $f_i$ :
$$\Pi^{fund}_i \approx -N_i \cdot f_i$$

EWMA du funding + MAD pour le bruit :
$$u_i = -\mu^f_i, \quad \text{SNR}^f_i = \frac{|\mu^f_i|}{\text{MAD}^f_i + \varepsilon}$$

Break-even avant inclusion :
$$|\mu^f_i| \cdot H_{fund} \cdot 10^4 > C_{bps} + \text{buffer}$$

Overlay projeté dollar/beta neutral, combiné :
$$w^{final} = w^{stat} + \eta \cdot Q^{fund} \cdot w^{fund}$$

---

### 1.7 Coûts & Réalisme

Slippage proxy vol-dépendant :
$$\text{slip}_{bps} = 8 + 10 \cdot RV_{1h}(\%)$$

Coût par rebalancement :
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
│   └── config.toml                    # thème dark
├── configs/
│   ├── default.yaml
│   ├── strategy_stat_arb.yaml         # paramètres FDS inclus
│   ├── hyperliquid_testnet.yaml
│   └── hyperliquid_mainnet.yaml
├── apps/
│   ├── dashboard.py                   # Streamlit live UI (PnL/positions/risques)
│   └── analyse.py                     # analyse offline / IC validation FDS
├── src/
│   └── hyperstat/
│       ├── __init__.py
│       ├── main.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── clock.py
│       │   ├── logging.py
│       │   ├── types.py               # Signal, RegimeScore, PortfolioWeights, ...
│       │   ├── math.py                # mad, zscore_mad, fit_ar1, neutralize_weights
│       │   └── risk.py                # KillSwitchConfig, RiskState, caps
│       ├── data/
│       │   ├── __init__.py
│       │   ├── storage.py             # DataStore : Parquet / DuckDB / SQLite
│       │   ├── loaders.py             # load_candles_csv_dir, load_funding_csv_dir, ...
│       │   ├── features.py            # compute_returns, ewma_vol, rv, amihud, beta, ...
│       │   └── universe.py            # select_universe, build_buckets (clustering)
│       ├── exchange/
│       │   ├── __init__.py
│       │   ├── sandbox.py
│       │   └── hyperliquid/
│       │       ├── __init__.py
│       │       ├── endpoints.py
│       │       ├── auth.py            # signature EIP-712
│       │       ├── rest_client.py
│       │       ├── ws_client.py
│       │       ├── rate_limiter.py
│       │       ├── market_data.py
│       │       ├── funding.py
│       │       └── execution.py
│       ├── strategy/
│       │   ├── __init__.py
│       │   ├── stat_arb.py            # signal MR cross-sectionnel + hystérésis
│       │   ├── regime.py              # Q_MR × Q_liq × Q_risk
│       │   ├── funding_divergence_signal.py  # FDS gate (batch + live)
│       │   ├── funding_overlay.py     # funding carry overlay
│       │   └── allocator.py           # vol-scaling → FDS → caps → overlay
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── engine.py              # boucle bar-par-bar close-to-close
│       │   ├── costs.py               # FeeModel + SlippageModel vol-dépendant
│       │   ├── metrics.py             # Sharpe, CAGR, max DD, turnover
│       │   └── reports.py             # BacktestReport + export CSV/HTML
│       ├── live/
│       │   ├── __init__.py
│       │   ├── runner.py              # boucle live asynchrone
│       │   ├── order_manager.py       # idempotence + réconciliation positions
│       │   └── health.py
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── risk_metrics.py        # VaR/CVaR, corrélations
│       │   └── sink.py                # telemetry
│       └── cli/
│           ├── __init__.py
│           └── commands.py
├── scripts/
│   ├── download_history.py
│   ├── build_universe.py
│   ├── fetch_candles.py
│   └── run_live.sh
└── tests/
    ├── test_strategy_smoke.py
    ├── test_cost_model.py
    └── test_projection_neutral.py
```

### Flux de données

```
Hyperliquid API
    │
    ▼
exchange/hyperliquid/          ← REST + WebSocket, rate limiter
    │
    ▼
data/
  storage.py                   ← persistence parquet/duckdb/sqlite
  loaders.py                   ← chargement candles + funding
  features.py                  ← returns, vol, RV, Amihud, beta, résidus
  universe.py                  ← filtrage univers + clustering buckets
    │
    ▼
strategy/
  stat_arb.py                  ← z-score cross-sectionnel + hystérésis
  regime.py                    ← Q_MR × Q_liq × Q_risk
  funding_divergence_signal.py ← FDS gate (confiance multiplicative)
  funding_overlay.py           ← carry funding
  allocator.py                 ← vol-scaling → FDS → neutralisation → caps
    │
    ▼
backtest/engine.py  ──OR──  live/runner.py
    │                              │
    ▼                              ▼
reports/ metrics/          exchange/execution.py
    │                              │
    ▼                              ▼
apps/dashboard.py          monitoring/sink.py
```

---

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pyarrow plotly streamlit scipy
```

Variables d'environnement requises pour le live :
```bash
export HL_ADDRESS=0x...
export HL_PRIVATE_KEY=0x...
```

---

## 4. Données

```bash
# Télécharger l'historique altcoins sur Hyperliquid
python scripts/download_history.py --config configs/default.yaml --days 60

# Construire l'univers + buckets
python scripts/build_universe.py --config configs/default.yaml
```

Les données sont stockées en parquet dans `./artifacts/data/`.

---

## 5. Lancer un backtest

```python
from hyperstat.backtest.engine import run_backtest, BacktestConfig

cfg = BacktestConfig(
    timeframe="5m",
    base_factor_symbol="BTC",
    initial_equity=1500.0,
)

report = run_backtest(
    cfg=cfg,
    candles_by_symbol=candles,      # {symbol: DataFrame avec cols ts/open/high/low/close/volume}
    funding_by_symbol=funding,      # {symbol: DataFrame avec cols ts/rate}
    buckets=buckets,                # {bucket_id: [symbols]}
    stat_arb=stat_arb,
    regime_model=regime_model,
    allocator=allocator,
)

print(report.metrics)
report.equity_curve.plot()
```

**Pour activer le FDS dans le backtest**, passer `funding_rates` à l'allocator — voir §10 (bug connu).

---

## 6. Lancer en paper / live

```bash
# Paper trading (paper = live sans exécution réelle)
python -m hyperstat.main --config configs/default.yaml --config configs/hyperliquid_testnet.yaml --mode paper

# Live
python -m hyperstat.main --config configs/default.yaml --config configs/hyperliquid_mainnet.yaml --mode live
# ou via le script shell :
bash scripts/run_live.sh
```

---

## 7. Dashboard Streamlit

```bash
streamlit run apps/dashboard.py
```

Affiche : courbe equity, positions courantes, VaR/CVaR, matrice de corrélation, Sharpe rolling, drawdown, scores FDS par coin.

---

## 8. Calibration du FDS

Le workflow recommandé avant de passer en live :

```python
from hyperstat.strategy.funding_divergence_signal import (
    FundingDivergenceSignal, FDSConfig, FDSDiagnostics
)

fds  = FundingDivergenceSignal(FDSConfig())
diag = FDSDiagnostics(fds)

# IC de Spearman : FDS_t vs returns_{t+H}
ic = diag.signal_ic(returns_df, funding_df, forward_horizon=8)
print(f"IC moyen : {ic.mean():.4f}   t-stat : {ic.mean()/ic.std()*len(ic)**0.5:.2f}")
# IC > 0.03 et t-stat > 2 → signal prédictif → augmenter gate_scale

# Décomposition des composantes
breakdown = diag.component_breakdown(returns_df, funding_df)

# Impact sur le turnover
to = diag.turnover_impact(w_stat, returns_df, funding_df)
print(f"Turnover ratio : {to['ratio']}x   (idéal ≈ 1.0)")
```

**Ordre de calibration (walk-forward, fixer les autres à leur valeur par défaut) :**
1. `divergence_window` : 12–48 barres
2. `span_funding_fast` : 4–16 barres
3. `w_carry / w_divergence / w_velocity`
4. `gate_scale` (commencer à 0.4, augmenter si IC confirme)

---

## 9. Notes Hyperliquid (API)

- **Auth** : signature EIP-712 avec clé privée (`HL_PRIVATE_KEY`)
- **Rate limit** : 1200 req/min (on utilise 1100 avec marge)
- **WS** : max 10 connexions, max 1000 subscriptions
- **Funding** : toutes les 8h sur Hyperliquid (horaire sur certains perps)
- **Perps disponibles** : environ 150 altcoins en perpetual futures
- **Frais taker** : ~0.035% (3.5 bps) en réalité — la config utilise 6 bps (conservateur)

---

## 10. Bugs connus / TODO

Ces points doivent être corrigés avant un backtest valide :

**Bug 1 — Module `hyperstat.data.features` manquant**
`backtest/engine.py` importe `compute_returns`, `compute_ewma_vol`, `compute_rv_1h_pct` depuis `hyperstat.data.features` qui n'existe pas dans le repo. Créer ce module ou déplacer ces fonctions dans `core/math.py`.

**Bug 2 — `funding_rates` non transmis à l'allocator dans engine.py**
```python
# Dans engine.py, ligne ~210, remplacer :
target = self.allocator.allocate(ts=..., signal=signal, regime=regime, ...)
# par :
funding_at_ts = self.funding_events.get(ts, {})
target = self.allocator.allocate(ts=..., signal=signal, regime=regime,
                                  funding_rates=funding_at_ts or None, ...)
```
Sans ce fix, le FDS est silencieusement désactivé dans tous les backtests.

**Bug 3 — Test smoke désynchronisé avec l'API réelle**
`test_strategy_smoke.py` appelle `BacktestConfig(run_name=..., out_dir=..., exec_mode=...)` mais ces paramètres n'existent pas dans la classe. Mettre à jour le test ou étendre `BacktestConfig`.

**Bug 4 — FDS non initialisé depuis le YAML**
`backtest_config_from_config()` ne lit pas la section `funding_divergence_signal` du YAML. Ajouter la lecture et l'instanciation du `FundingDivergenceSignalLive` depuis la config.

---

## 11. Roadmap

- [ ] **Fix bugs §10** (priorité avant tout backtest)
- [ ] **`hyperstat.data.features`** : implémenter `compute_returns`, `compute_ewma_vol`, `compute_rv_1h_pct`
- [ ] **Validation IC FDS** sur données Hyperliquid réelles (objectif : IC > 0.03, t-stat > 2)
- [ ] **Walk-forward calibration** des paramètres FDS
- [ ] **Clustering dynamique** via graph clustering (Korniejczuk 2024) — reconstruction des buckets à chaque période
- [ ] **PCA eigenportfolios** (Jung 2025) — signal sur résidus purés des facteurs communs
- [ ] **Illiquidité Amihud rolling** dans le régime (actuellement absent du backtest)
- [ ] **Betas BTC rolling** pour beta-neutralisation (actuellement `betas=None`)
- [ ] **Live : réconciliation positions** dans `order_manager.py`
- [ ] **Tests** : coverage backtest engine, FDS live vs batch cohérence
