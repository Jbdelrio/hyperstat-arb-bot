# HyperStat — Stat-Arb Altcoins + Funding Overlay (Hyperliquid)

HyperStat est un framework **backtest + paper + live** pour faire du **statistical arbitrage** sur des altcoins (perps) avec une couche optionnelle de **funding carry**, déployable sur **Hyperliquid**.

Le projet est conçu pour :
- **Rester réaliste** : coûts (fees + slippage proxy), turnover, caps d’exposition, kill-switch drawdown.
- **Être itératif** : la même logique *data → signal → sizing → risk → execution* fonctionne en backtest puis en live.
- **Être monitorable** : dashboard Streamlit dark (PnL, positions, VaR/CVaR, corr, Sharpe, drawdown).

---

## Sommaire
- [1. Stratégie](#1-stratégie)
  - [1.1 Univers & Buckets](#11-univers--buckets)
  - [1.2 Signal Mean-Reversion Cross-Sectional](#12-signal-mean-reversion-cross-sectional)
  - [1.3 Régime / Gating (Q_t)](#13-régime--gating-q_t)
  - [1.4 Allocation & Neutralisation](#14-allocation--neutralisation)
  - [1.5 Funding Overlay](#15-funding-overlay)
  - [1.6 Coûts & Réalisme](#16-coûts--réalisme)
  - [1.7 Risk Management](#17-risk-management)
- [2. Architecture du code](#2-architecture-du-code)
- [3. Installation](#3-installation)
- [4. Données (download_history)](#4-données-download_history)
- [5. Lancer un backtest](#5-lancer-un-backtest)
- [6. Lancer en paper / live](#6-lancer-en-paper--live)
- [7. Dashboard Streamlit](#7-dashboard-streamlit)
- [8. Notes Hyperliquid (API)](#8-notes-hyperliquid-api)
- [9. Roadmap](#9-roadmap)

---

## 1. Stratégie

### 1.1 Univers & Buckets

Le stat-arb altcoins est dominé par :
- la **liquidité** (sinon slippage et adverse selection),
- la **corrélation** (si les coins d’un bucket ne partagent rien, le “spread” ne mean-revert pas).

**Filtrage univers** (idée) :
- Dollar volume proxy :  
  $$
  DV_{i,t} = P_{i,t}\cdot V_{i,t}
  $$
- Illiquidité d’Amihud (proxy impact) :
  $$
  ILLIQ_i = \text{median}_t\left(\frac{|r_{i,t}|}{DV_{i,t}+\varepsilon}\right)
  $$
- Funding instable (éviter les coins “toxiques” si on ne veut pas que tout le risque vienne du carry).

**Buckets** : on regroupe les coins “semblables” pour capter de la dispersion intra-groupe.
- Distance corrélation (classique) :
  $$
  d_{ij}=\sqrt{\tfrac{1}{2}(1-\rho_{ij})}
  $$
- Clustering hiérarchique → $K$ buckets.

> Extension robuste (souvent meilleure) : cluster sur des **résidus** (retirer le facteur BTC)  
> $$
> r_{i,t} = \beta_i r_{BTC,t} + \varepsilon_{i,t}
> $$
> puis cluster sur $\varepsilon$.

---

### 1.2 Signal Mean-Reversion Cross-Sectional

On cherche une opportunité de type “**dispersion mean-reverting**” au sein d’un bucket.

Retour log sur horizon $H$ (ex: 1h en 5m → $H=12$) :
$$
R_{i,t}^{(H)}=\ln\left(\frac{P_{i,t}}{P_{i,t-H}}\right)
$$

Score robuste dans le bucket $B$ via **median/MAD** :
$$
z_{i,t}=\frac{R_{i,t}^{(H)}-\text{median}_B(R_t^{(H)})}{\text{MAD}_B(R_t^{(H)})+\varepsilon}
$$

Signal contrarian :
$$
s_{i,t}=-\text{clip}(z_{i,t},[-z_{max},z_{max}])
$$

**Hystérésis anti-churn** :
- Entrée : $|z| > z_{in}$
- Sortie : $|z| < z_{out}$
- Min hold / Max hold

Objectif : limiter le **turnover** (sinon frais + slippage mangent l’alpha).

---

### 1.3 Régime / Gating (Q_t)

Même si un z-score existe, ça ne veut pas dire que la MR est exploitable **maintenant**.
On scale le risque via :
$$
Q_t = Q^{MR}_t \cdot Q^{liq}_t \cdot Q^{risk}_t \in [0,1]
$$

#### Mean reversion quality $Q^{MR}_t$
On construit un spread bucket simple (top vs bottom quantiles) :
$$
S_{B,t}=\text{mean}_{Top}(R^{(H)}) - \text{mean}_{Bottom}(R^{(H)})
$$

Fit AR(1) rolling :
$$
S_{B,t}=a+bS_{B,t-1}+u_t
$$

Half-life (si $0<b<1$) :
$$
t_{1/2}=\frac{\ln 2}{-\ln b}\cdot \Delta t
$$

Règle : si half-life trop longue → MR trop lente/instable → $Q^{MR}$ baisse.

#### Liquidité $Q^{liq}_t$
Proxy via rangs cross-section :
- DV élevé = bon
- ILLIQ faible = bon

#### Risk $Q^{risk}_t$
Proxy via vol (BTC et/ou univers) :
- si vol > p90 → réduire
- si vol > p95 → couper (flat)

---

### 1.4 Allocation & Neutralisation

On transforme le signal en **target weights** $w_{i,t}$ (dimensionless : notional / equity).

Vol-scaling :
$$
w^{raw}_{i,t}=\frac{s_{i,t}}{\hat{\sigma}_{i,t}}
$$
où $\hat{\sigma}$ est une vol EWMA.

Application du régime :
$$
w_{i,t} \leftarrow Q_t \cdot w_{i,t}
$$

**Neutralisations** :
- Dollar-neutral : $\sum_i w_i = 0$
- Beta-neutral (vs BTC) : $\sum_i w_i \beta_i = 0$

On applique une projection dans l’espace admissible $A w=0$ :
$$
w \leftarrow \left(I - A^\top(AA^\top)^{-1}A\right)u
$$

**Caps** :
- $|w_i| \le w_{max}$
- $\sum_{i\in bucket}|w_i| \le w^{bucket}_{max}$

---

### 1.5 Funding Overlay

Le funding est un **carry** (paiement périodique).
Avec un notional $N_i$ et un funding $f_i$ :
$$
\Pi^{fund}_i \approx -N_i \cdot f_i
$$
(en général, funding positif ⇒ **long pay short**, donc être short capture le carry).

On calcule un “préférentiel” :
$$
u_i = -\mu^f_i
$$
où $\mu^f_i$ est un EWMA du funding, et une mesure de bruit (EWMA MAD) :
$$
\text{SNR}^f_i = \frac{|\mu^f_i|}{\text{MAD}^f_i+\varepsilon}
$$

**Break-even** (indispensable avec capital faible) :
$$
|\mu^f_i|\cdot H_{fund} \cdot 10^4 \;>\; C_{bps} + \text{buffer}
$$
où $C_{bps}$ approxime fees+slip.

Ensuite on **projette** l’overlay pour rester dollar/beta neutral, et on combine :
$$
w^{final} = w^{stat} + \eta \cdot Q^{fund} \cdot w^{fund}
$$

---

### 1.6 Coûts & Réalisme

Backtest : modèle simple mais calibrable :
- Fees bps (taker/maker)
- Slippage bps proxy :
$$
\text{slip}_{bps} = s_0 + k \cdot RV_{1h}(\%) 
$$
Coût par rebal :
$$
\text{Cost}_t = \sum_i |\Delta w_{i,t}|\cdot Equity_t \cdot \frac{fee_{bps} + slip_{bps}}{10^4}
$$

---

### 1.7 Risk Management

- Kill-switch drawdown intraday :
  - si DD > seuil → flat + cooldown
- Emergency flatten coin :
  - si $|z| > z_{emergency}$ → target=0
- Caps d’expo (coin/bucket/gross)

---

## 2. Architecture du code

- `src/hyperstat/strategy/` : signal MR, regime, overlay funding, allocator
- `src/hyperstat/backtest/` : engine + costs + metrics + reports
- `src/hyperstat/live/` : runner live, order manager (idempotence), health
- `src/hyperstat/exchange/hyperliquid/` : REST/WS + signing + execution
- `src/hyperstat/monitoring/` : sink (telemetry) + risk metrics (VaR/corr)
- `apps/dashboard.py` : Streamlit UI dark

---

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# recommandé (parquet + charts)
pip install pyarrow plotly
# dashboard
pip install streamlit
