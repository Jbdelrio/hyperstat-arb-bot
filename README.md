# HyperStat — Stat-Arb Altcoins + Funding Divergence Signal (Hyperliquid)

HyperStat est un framework **backtest + paper + live** pour du **statistical arbitrage cross-sectionnel** sur des altcoins (perps Hyperliquid), enrichi d'un **Funding Divergence Signal (FDS)** comme gate de confiance multiplicatif, d'une couche de **funding carry**, et d'une couche d'**exécution intelligente VWAP/TWAP**.

Le projet est conçu pour :

- **Rester réaliste** : coûts (fees + slippage VWAP-ajusté 4 composantes), turnover, caps d'exposition, kill-switch drawdown.
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
- [9. Exécution VWAP / TWAP](#9-exécution-vwap--twap)
- [10. Notes Hyperliquid (API)](#10-notes-hyperliquid-api)
- [11. Bugs connus / TODO](#11-bugs-connus--todo)
- [12. Roadmap](#12-roadmap)
- [13. Références](#13-références)

---

## 1. Stratégie

### 1.1 Univers & Buckets

**Filtrage univers** :

- Dollar volume proxy : $DV_{i,t} = P_{i,t} \cdot V_{i,t}$
- Illiquidité d'Amihud : $ILLIQ_{i} = \text{median}_t\!\left(\frac{|r_{i,t}|}{DV_{i,t}+\varepsilon}\right)$
- Exclusion des coins à funding instable (toxiques pour le carry)

**Buckets** — regroupement des coins structurellement similaires :

Distance corrélation sur résidus BTC-neutralisés :

$$r_{i,t} = \beta_i r_{BTC,t} + \varepsilon_{i,t}$$

$$d_{ij}=\sqrt{\tfrac{1}{2}(1-\rho_{ij}^{\varepsilon})}$$

Clustering hiérarchique → $K$ buckets (4–6 dans la config par défaut).

---

### 1.2 Signal Mean-Reversion Cross-Sectionnel

Log-return sur horizon $H$ :

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

$$Q_t = Q^{MR}_t \cdot Q^{liq}_t \cdot Q^{risk}_t \cdot Q^{break}_t \in [0,1]$$

> **Nouveau :** composante $Q^{break}_t$ — détecteur de rupture de volatilité (voir §1.3.4).

#### $Q^{MR}_t$ — Mean reversion quality

Spread bucket top/bottom quantiles → AR(1) rolling :

$$S_{B,t}=a+b \cdot S_{B,t-1}+u_t \qquad t_{1/2}=\frac{\ln 2}{-\ln b}\cdot \Delta t$$

| Half-life | $Q^{MR}$ |
|---|---|
| < 30 min | 0.5 (bruit) |
| 30 min – 6h | 1.0 ✅ |
| 6h – 24h | 0.5 |
| > 24h | 0.0 |

#### $Q^{liq}_t$ — Liquidité

Rangs cross-section : $\text{rank}_{pct}(DV)$ élevé et $1 - \text{rank}_{pct}(ILLIQ)$ élevé. Si médiane cross-section < 0.3 → $Q^{liq} = 0$.

#### $Q^{risk}_t$ — Volatilité (BTC proxy)

Percentiles rolling sur 60 jours :

| Vol BTC | $Q^{risk}$ |
|---|---|
| < p90 | 1.0 ✅ |
| p90 – p95 | 0.5 |
| > p95 | 0.0 |

#### $Q^{break}_t$ — Détecteur de rupture de volatilité *(nouveau)*

Ratio EWMA fast/slow sur les returns BTC :

$$\text{ratio}_t = \frac{\text{EWMA}_4(\sigma_t)}{\text{EWMA}_{48}(\sigma_t)}$$

| Ratio | $Q^{break}$ |
|---|---|
| < 2.0 | 1.0 ✅ (régime stable) |
| 2.0 – 2.5 | 0.5 (zone de transition) |
| > 2.5 | 0.0 (choc détecté — liquidations, news) |

Implémentation : `strategy/regime.py → detect_volatility_regime_break()`

---

### 1.4 Funding Divergence Signal (FDS) — Alpha Original

> **Insight clé :** sur Hyperliquid, les traders on-chain ont un biais structurel **long**, créant une asymétrie persistante dans le funding.

Le FDS est appliqué comme un **gate de confiance multiplicatif** :

$$\boxed{w_{i,t}^{after,FDS} = w_{i,t}^{stat} \cdot \left(1 + \alpha_{fds} \cdot \text{FDS}_{i,t} \cdot d_{i,t}^{stale}\right)}$$

avec $\alpha_{fds} = 0.6$ et $d_{i,t}^{stale}$ le **discount de staleness** *(nouveau)*.

#### Composante 1 — Carry cross-sectionnel (poids 0.35)

$$\tilde{f}_{i,t}^{slow} = \text{EWMA}_{\tau_s}(f_{i,t}), \quad \tau_s = 72 \text{ barres}$$

$$s^{carry}_{i,t} = -\text{clip}\!\left(\frac{\tilde{f}_{i,t}^{slow} - \text{median}_j(\tilde{f}_{j,t}^{slow})}{\text{MAD}_j + \varepsilon},\ [-z_{max}, z_{max}]\right)$$

#### Composante 2 — Désalignement funding/return (poids 0.40) ⭐

Vélocité du funding (EWMA rapide, span $\tau_f = 8$) :

$$\Delta\tilde{f}_{i,t}^{fast} = \text{EWMA}_{\tau_f}(f_{i,t}) - \text{EWMA}_{\tau_f}(f_{i,t-1})$$

$$\rho_{i,t} = \text{Corr}_{W=24}\!\left(z_{i,t}^{return},\ \Delta\tilde{f}_{i,t}^{fast}\right)$$

$$\text{tension}_{i,t} = 1 - |\rho_{i,t}|, \quad s^{div}_{i,t} = \text{clip}\!\left(-\text{sgn}(z) \cdot \text{sgn}(\Delta f) \cdot \text{tension},\ [-1,1]\right)$$

**Intuition :** prix monte ($r > 0$) mais funding recule ($\Delta f < 0$) → hausse non crédible → signal contrarian renforcé.

#### Composante 3 — Vélocité du funding (poids 0.25)

$$v_{i,t} = \frac{\text{EWMA}_{\tau_f}(f_{i,t}) - \text{EWMA}_{\tau_s}(f_{i,t})}{|\text{EWMA}_{\tau_s}(f_{i,t})| + \varepsilon}$$

$$s^{vel}_{i,t} = -\text{clip}\!\left(\frac{v_{i,t}}{v_{max}},\ [-1, 1]\right)$$

#### Assemblage FDS

$$\text{FDS}_{i,t}^{raw} = 0.35 \cdot s^{carry}_{i,t} + 0.40 \cdot s^{div}_{i,t} + 0.25 \cdot s^{vel}_{i,t}$$

$$\text{FDS}_{i,t} = \text{clip}\!\left(\frac{\text{FDS}_{i,t}^{raw} - \text{median}_j}{2 \cdot \text{MAD}_j + \varepsilon},\ [-1, 1]\right)$$

#### Discount de staleness *(nouveau)*

Sur Hyperliquid, le funding est payé toutes les 8h. Entre deux updates, l'EWMA rapide tourne sur un signal constant → bruit artificiel sur la composante vélocité. Le discount corrige ce biais :

$$d_{i,t}^{stale} = \max\!\left(0.5,\ 1 - 0.5 \cdot \text{clip}\!\left(\frac{h_{stale} - 9}{9},\ [0,1]\right)\right)$$

où $h_{stale}$ = heures écoulées depuis le dernier paiement de funding.

Implémentation : `strategy/funding_divergence_signal.py → funding_staleness_discount()`

---

### 1.5 Allocation & Neutralisation

Pipeline complet dans `allocator.py` :

1. **Vol-scaling** : $w_{i,t}^{raw} = s_{i,t} / \hat{\sigma}_{i,t}$
2. **Regime scaling** : $w \leftarrow Q_t \cdot w$
3. **Neutralisation + normalisation gross** (projection nullspace dollar/beta neutral)
4. **FDS gate** : $w_i \leftarrow w_i \cdot (1 + \alpha_{fds} \cdot \text{FDS}_i \cdot d_i^{stale})$ + re-neutralisation
5. **Execution cost adjustment** *(nouveau)* : $w_i \leftarrow w_i \cdot \text{scale}_{cost}$ si coût estimé > alpha proxy
6. **Caps** : coin (12%) / bucket (35%)
7. **Emergency flatten** : si $|z_i| > 3.5$ → $w_i = 0$
8. **Funding overlay** : $w^{final} = w^{stat} + \eta \cdot Q^{fund} \cdot w^{fund}$
9. **Contrainte finale** : gross ≤ 1.40

---

### 1.6 Funding Carry Overlay

$$\Pi^{fund}_i \approx -N_i \cdot f_i$$

Break-even avant inclusion :

$$|\mu^f_i| \cdot H_{fund} \cdot 10^4 > C_{bps} + \text{buffer}$$

$$w^{final} = w^{stat} + \eta \cdot Q^{fund} \cdot w^{fund}$$

---

### 1.7 Coûts & Réalisme

**Modèle de slippage étendu** *(mis à jour)* — 4 composantes :

$$\text{slip}_{bps} = \underbrace{8}_{\text{base}} + \underbrace{10 \cdot RV_{1h}(\%)}_{\text{vol impact}} + \underbrace{5\sqrt{\text{ADV\%}} \cdot 100}_{\text{market impact}} + \underbrace{0.2 \cdot \Delta_{VWAP}^{bps}}_{\text{VWAP dev}}$$

où $\Delta_{VWAP}^{bps} = |P_{close} - VWAP| / VWAP \times 10^4$.

Le modèle original (8 + 10×RV) est conservé comme cas de base quand le VWAP n'est pas disponible.

Implémentation : `backtest/costs.py → vwap_adjusted_slippage()`

Coût par rebalancement :

$$\text{Cost}_t = \sum_i |\Delta w_{i,t}| \cdot \text{Equity}_t \cdot \frac{\text{fee}_{bps} + \text{slip}_{bps}}{10^4}$$

Paramètres par défaut (mode taker) : fee = 6 bps.

---

### 1.8 Risk Management

- Kill-switch drawdown intraday : DD > 3% → flat + cooldown 12h
- Emergency flatten par coin : $|z| > 3.5$ → target = 0
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
│   ├── dashboard.py                   # Streamlit live UI
│   └── analyse.py                     # analyse offline / IC validation FDS
├── src/
│   └── hyperstat/
│       ├── __init__.py
│       ├── main.py
│       ├── core/
│       │   ├── clock.py
│       │   ├── logging.py
│       │   ├── types.py               # Signal, RegimeScore, PortfolioWeights, ...
│       │   ├── math.py                # mad, zscore_mad, fit_ar1, neutralize_weights
│       │   └── risk.py                # KillSwitchConfig, RiskState, caps
│       ├── data/
│       │   ├── storage.py             # DataStore : Parquet / DuckDB / SQLite
│       │   ├── loaders.py             # load_candles_csv_dir, load_funding_csv_dir
│       │   ├── features.py            # compute_returns, ewma_vol, rv, amihud, beta
│       │   └── universe.py            # select_universe, build_buckets (clustering)
│       ├── exchange/
│       │   ├── sandbox.py
│       │   └── hyperliquid/
│       │       ├── endpoints.py
│       │       ├── auth.py            # signature EIP-712
│       │       ├── rest_client.py
│       │       ├── ws_client.py
│       │       ├── rate_limiter.py
│       │       ├── market_data.py
│       │       ├── funding.py
│       │       └── execution.py
│       ├── execution/                 # ← NOUVEAU MODULE
│       │   ├── __init__.py
│       │   └── vwap_strategy.py       # VWAP/TWAP/hybrid order slicer
│       ├── strategy/
│       │   ├── stat_arb.py            # signal MR cross-sectionnel + hystérésis
│       │   ├── regime.py              # Q_MR × Q_liq × Q_risk × Q_break ← mis à jour
│       │   ├── funding_divergence_signal.py  # FDS gate + staleness discount ← mis à jour
│       │   ├── funding_overlay.py     # funding carry overlay
│       │   └── allocator.py           # vol-scaling → FDS → execution adj → caps
│       ├── backtest/
│       │   ├── engine.py              # boucle bar-par-bar (funding_rates corrigé)
│       │   ├── costs.py               # FeeModel + SlippageModel VWAP-ajusté ← mis à jour
│       │   ├── metrics.py             # Sharpe, CAGR, max DD, slippage_attribution ← mis à jour
│       │   └── reports.py             # BacktestReport + export CSV/HTML
│       ├── live/
│       │   ├── runner.py
│       │   ├── order_manager.py
│       │   └── health.py
│       ├── monitoring/
│       │   ├── risk_metrics.py        # VaR/CVaR, corrélations
│       │   └── sink.py
│       └── cli/
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
  regime.py                    ← Q_MR × Q_liq × Q_risk × Q_break
  funding_divergence_signal.py ← FDS gate + staleness discount
  funding_overlay.py           ← carry funding
  allocator.py                 ← vol-scaling → FDS → execution adj → caps
    │
    ▼
execution/
  vwap_strategy.py             ← OrderSlicer VWAP/TWAP/hybrid
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
    candles_by_symbol=candles,       # {symbol: DataFrame ts/open/high/low/close/volume}
    funding_by_symbol=funding,       # {symbol: DataFrame ts/rate}
    buckets=buckets,                 # {bucket_id: [symbols]}
    stat_arb=stat_arb,
    regime_model=regime_model,
    allocator=allocator,
)

print(report.metrics)
report.equity_curve.plot()
```

Le FDS est automatiquement activé — `funding_rates` est correctement transmis à l'allocator (bug corrigé dans `engine.py`, lignes 367-379).

**Métriques de slippage disponibles dans le report :**

```python
from hyperstat.backtest.metrics import slippage_attribution

attr = slippage_attribution(report.trades_df)
print(attr)
# {
#   'mean_vwap_slip_bps':   ...,
#   'median_vwap_slip_bps': ...,
#   'total_slip_cost_pct':  ...,
#   'pct_beating_vwap':     ...,
#   'worst_slip_bps':       ...,
# }
```

---

## 6. Lancer en paper / live

```bash
# Paper trading
python -m hyperstat.main \
    --config configs/default.yaml \
    --config configs/hyperliquid_testnet.yaml \
    --mode paper

# Live
python -m hyperstat.main \
    --config configs/default.yaml \
    --config configs/hyperliquid_mainnet.yaml \
    --mode live

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

**Workflow recommandé avant de passer en live :**

```python
from hyperstat.strategy.funding_divergence_signal import (
    FundingDivergenceSignal, FDSConfig, FDSDiagnostics
)

fds  = FundingDivergenceSignal(FDSConfig())
diag = FDSDiagnostics(fds)

# IC de Spearman : FDS_t vs returns_{t+H}
ic = diag.signal_ic(returns_df, funding_df, forward_horizon=8)
print(f"IC moyen : {ic.mean():.4f}   t-stat : {ic.mean()/ic.std()*len(ic)**0.5:.2f}")
# Cible : IC > 0.03 et t-stat > 2

# Décomposition des composantes
breakdown = diag.component_breakdown(returns_df, funding_df)

# Impact sur le turnover
to = diag.turnover_impact(w_stat, returns_df, funding_df)
print(f"Turnover ratio : {to['ratio']}x   (idéal ≈ 1.0)")
```

**Ordre de calibration (walk-forward, un paramètre à la fois) :**

1. `divergence_window` : 12–48 barres
2. `span_funding_fast` : 4–16 barres
3. `w_carry / w_divergence / w_velocity`
4. `gate_scale` (commencer à 0.4, augmenter si IC confirme)

---

## 9. Exécution VWAP / TWAP

Le module `execution/vwap_strategy.py` fragmente les ordres parents en sous-ordres pour réduire le slippage de marché.

### Modes disponibles

| Mode | Description | Quand l'utiliser |
|------|-------------|------------------|
| `vwap` | Tranches pondérées par le profil de volume historique | Altcoins à volume prévisible |
| `twap` | Tranches uniformes dans le temps | Altcoins à faible liquidité / volume imprévisible |
| `hybrid` | 60% VWAP + 40% TWAP (défaut) | Cas général |

### Usage de base

```python
from hyperstat.execution.vwap_strategy import OrderSlicer, ExecutionConfig

slicer = OrderSlicer(cfg=ExecutionConfig(
    n_slices=8,
    vwap_window=8,      # 8 barres = 8h pour timeframe 1h
    mode="hybrid",
    price_tolerance=0.003,   # 30 bps max au-dessus du benchmark
))

slices = slicer.slice_order(
    symbol="ARB",
    direction="BUY",
    total_notional=5000.0,
    df_recent=recent_candles,   # DataFrame avec high/low/close/volume
)

for limit_price, size in slices:
    exchange.place_limit_order("ARB", "BUY", size, limit_price)
```

### Ajustement du signal par les coûts

Avant l'allocation finale, les poids dont le coût d'exécution estimé dépasse l'alpha proxy sont automatiquement réduits :

```python
from hyperstat.execution.vwap_strategy import execution_cost_adjustment

# À appeler depuis allocator.py juste avant les caps
weights_adjusted = execution_cost_adjustment(
    signal_weights=weights,
    df_candles=latest_candles_by_symbol,
)
```

La réduction suit :

$$w_i^{adj} = w_i \cdot \max\!\left(0,\ 1 - \frac{\text{cost}_{bps} - \text{alpha}_{proxy}}{\text{alpha}_{proxy}}\right)$$

### Formules clés

**VWAP** (prix typique pondéré par le volume) :

$$VWAP_t = \frac{\sum_{i=t-w}^{t} V_i \cdot P_i^{typ}}{\sum_{i=t-w}^{t} V_i}, \quad P_i^{typ} = \frac{H_i + L_i + C_i}{3}$$

**TWAP** (moyenne simple des prix typiques) :

$$TWAP_t = \frac{1}{w}\sum_{i=t-w}^{t} P_i^{typ}$$

**Profil de volume** (pondérations des tranches) :

$$\phi_k = \frac{V_{t-w+k}}{\sum_{j=0}^{w-1} V_{t-w+j}}, \quad k = 0, \ldots, w-1$$

### Test rapide

```bash
python -c "
from hyperstat.execution.vwap_strategy import OrderSlicer
import pandas as pd, numpy as np

df = pd.DataFrame({
    'high':   np.random.uniform(100, 105, 50),
    'low':    np.random.uniform(95,  100, 50),
    'close':  np.random.uniform(98,  103, 50),
    'volume': np.random.uniform(1000, 5000, 50),
})
slicer = OrderSlicer()
slices = slicer.slice_order('BTC', 'BUY', 10000, df)
print(f'{len(slices)} tranches, total = \${sum(s for _, s in slices):.2f}')
"
```

---

## 10. Notes Hyperliquid (API)

- **Auth** : signature EIP-712 avec clé privée (`HL_PRIVATE_KEY`)
- **Rate limit** : 1200 req/min (on utilise 1100 avec marge)
- **WS** : max 10 connexions, max 1000 subscriptions
- **Funding** : toutes les 8h (attention au biais de staleness — voir §1.4)
- **Perps disponibles** : environ 150 altcoins en perpetual futures
- **Frais taker** : ~0.035% (3.5 bps) en réalité — la config utilise 6 bps (conservateur)

---

## 11. Bugs connus / TODO

> **Bugs 1 et 2 sont corrigés.**

~~**Bug 1 — Module `hyperstat.data.features` manquant**~~ ✅ *Corrigé — module complet.*

~~**Bug 2 — `funding_rates` non transmis à l'allocator dans engine.py**~~ ✅ *Corrigé — lignes 367-379 de `engine.py`.*

**Bug 3 — Test smoke désynchronisé avec l'API réelle**
`test_strategy_smoke.py` appelle `BacktestConfig(run_name=..., out_dir=..., exec_mode=...)` mais ces paramètres n'existent pas dans la classe. À mettre à jour.

**Bug 4 — FDS non initialisé depuis le YAML**
`backtest_config_from_config()` ne lit pas la section `funding_divergence_signal` du YAML. Ajouter la lecture et l'instanciation du `FundingDivergenceSignalLive` depuis la config.

---

## 12. Roadmap

- **Fix Bug 3 & 4** (tests + YAML init FDS)
- **Validation IC FDS** sur données Hyperliquid réelles (objectif : IC > 0.03, t-stat > 2)
- **Walk-forward calibration** des paramètres FDS
- **Intégration execution_cost_adjustment** dans `allocator.py` (appel déjà prêt)
- **Clustering dynamique** via graph clustering (Korniejczuk 2024) — reconstruction des buckets à chaque période
- **PCA eigenportfolios** (Jung 2025) — signal sur résidus purés des facteurs communs
- **Illiquidité Amihud rolling** dans le régime (actuellement absent du backtest)
- **Betas BTC rolling** pour beta-neutralisation (actuellement `betas=None`)
- **Live : réconciliation positions** dans `order_manager.py`
- **Tests** : coverage backtest engine, FDS live vs batch cohérence, VWAP slicer

---

## 13. Références

| Papier | Contribution dans HyperStat |
|---|---|
| Genet (2025). *Deep Learning for VWAP Execution in Crypto Markets.* [arXiv:2502.13722](https://arxiv.org/abs/2502.13722) | Fondement théorique du module `execution/vwap_strategy.py` — optimisation directe de l'objectif VWAP sans prédiction de la courbe de volume |
| He & Manela (2024). *Fundamentals of Perpetual Futures.* [arXiv:2212.06888](https://arxiv.org/abs/2212.06888) | Borne no-arbitrage futures-spot → condition de déclenchement pour `funding_divergence_signal.py` Composante 2 |
| Exploring Risk and Return Profiles of Funding Rate Arbitrage on CEX and DEX (2024). [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2096720925000818) | Validation empirique de la stratégie de carry cross-exchange — motiver l'extension future à un second exchange |
| Korniejczuk (2024). *Statistical arbitrage in multi-pair trading strategy based on graph clustering.* [arXiv:2406.10695](https://arxiv.org/abs/2406.10695) | Algorithme SPONGEsym pour le clustering dynamique des buckets (roadmap) |
