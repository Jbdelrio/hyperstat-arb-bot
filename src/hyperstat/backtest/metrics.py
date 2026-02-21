# src/hyperstat/backtest/metrics.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass de métriques
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Ensemble complet de métriques de performance pour un backtest.

    Champs market-dependent (beta, treynor, jensen_alpha, r_squared) sont NaN
    si market_returns n'est pas fourni à compute_performance_metrics().
    """
    start: pd.Timestamp
    end: pd.Timestamp
    n_steps: int

    # ── Rendement ─────────────────────────────────────────────────────────────
    total_return: float     # rendement total sur la période
    cagr: float             # taux de croissance annuel composé

    # ── Ratios performance/risque ─────────────────────────────────────────────
    sharpe: float           # rendement excédentaire annualisé / vol annualisée
    sortino: float          # rendement excédentaire / downside vol (pertes seulement)
    calmar: float           # CAGR / |max drawdown|

    # ── Risque ────────────────────────────────────────────────────────────────
    ann_vol: float          # volatilité annualisée
    max_drawdown: float     # drawdown maximum (négatif)
    avg_drawdown: float     # moyenne des drawdowns négatifs
    max_dd_duration_bars: int  # durée max consécutive en drawdown (barres)
    var_95: float           # Value at Risk 95%, perte maximale journalière attendue (positif)

    # ── Métriques de robustesse (trade-level) ─────────────────────────────────
    win_rate: float         # fraction des barres avec rendement positif
    loss_rate: float        # fraction des barres avec rendement négatif
    profit_factor: float    # gains totaux / pertes totales (>1 = profitable)
    avg_gain_loss_ratio: float  # gain moyen / perte moyenne
    kelly_fraction: float   # fraction optimale Kelly (informative — ne pas utiliser brute)

    # ── Exposition ────────────────────────────────────────────────────────────
    avg_gross: float        # exposition brute moyenne
    avg_net: float          # exposition nette moyenne
    avg_turnover: float     # turnover moyen par barre

    # ── Corrélation au marché (NaN si market_returns non fourni) ─────────────
    beta: float             # β = Cov(Rp, Rm) / Var(Rm)
    treynor: float          # (CAGR - Rf) / β
    jensen_alpha: float     # Rp - [Rf + β(Rm - Rf)] annualisé
    r_squared: float        # R² = corr(Rp, Rm)²

    # ── PnL décomposé ─────────────────────────────────────────────────────────
    pnl_gross: float        # PnL prix brut
    pnl_funding: float      # PnL funding carry
    pnl_fees: float         # frais de transaction
    pnl_slippage: float     # slippage
    pnl_net: float          # PnL net total


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_steps_per_year(index: pd.DatetimeIndex) -> float:
    """Estime le nombre de barres par an depuis le median des deltas temporels."""
    if len(index) < 3:
        return 0.0
    dt = index.to_series().diff().dropna()
    if dt.empty:
        return 0.0
    median_s = float(dt.median().total_seconds())
    if median_s <= 0:
        return 0.0
    return float((365.25 * 24 * 3600) / median_s)


def _max_drawdown(equity: pd.Series) -> float:
    """Drawdown maximum depuis un pic (valeur négative)."""
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _avg_drawdown(equity: pd.Series) -> float:
    """Moyenne des barres en drawdown négatif (valeur négative ou 0)."""
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = (equity - peak) / peak
    in_dd = dd[dd < 0]
    return float(in_dd.mean()) if not in_dd.empty else 0.0


def _max_dd_duration(equity: pd.Series) -> int:
    """Durée maximale consécutive sous le pic précédent (en barres)."""
    if equity.empty:
        return 0
    peak = equity.cummax()
    in_dd = equity < peak
    max_run = 0
    cur_run = 0
    for v in in_dd:
        if v:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run


def _sortino(returns: pd.Series, ann_factor: float, rf: float = 0.0) -> float:
    """
    Ratio de Sortino = (rendement annualisé - Rf) / downside vol annualisée.
    Downside vol = écart-type des rendements négatifs seulement.
    """
    if returns.empty or ann_factor <= 0:
        return float("nan")
    ann_ret = float(returns.mean() * ann_factor)
    neg = returns[returns < 0]
    if neg.empty:
        return float("inf")  # aucune perte
    down_vol = float(neg.std(ddof=1) * np.sqrt(ann_factor))
    return float((ann_ret - rf) / down_vol) if down_vol > 1e-12 else float("nan")


def _var_95(returns: pd.Series) -> float:
    """
    VaR paramétrique 95% en fraction du portefeuille (valeur positive = perte).
    VaR = −(μ + z_{0.05} · σ)  avec z_{0.05} = −1.645.
    """
    if returns.empty:
        return float("nan")
    mu = float(returns.mean())
    sig = float(returns.std(ddof=1))
    # norm.ppf(0.05) ≈ -1.6449
    return float(-(mu + float(norm.ppf(0.05)) * sig))


def _trade_stats(returns: pd.Series) -> tuple[float, float, float, float]:
    """
    Retourne (win_rate, loss_rate, profit_factor, avg_gain_loss_ratio).
    Opère sur des rendements barre par barre.
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    n = max(1, len(returns))

    win_rate = float(len(wins) / n)
    loss_rate = float(len(losses) / n)

    total_gain = float(wins.sum()) if not wins.empty else 0.0
    total_loss = float(losses.abs().sum()) if not losses.empty else 1e-12
    profit_factor = total_gain / total_loss if total_loss > 1e-12 else float("nan")

    avg_gain = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.abs().mean()) if not losses.empty else 1e-12
    avg_gl = avg_gain / avg_loss if avg_loss > 1e-12 else float("nan")

    return win_rate, loss_rate, profit_factor, avg_gl


def _kelly(win_rate: float, avg_gain_loss_ratio: float) -> float:
    """
    Fraction de Kelly : f* = p - (1-p) / b
    où p = win_rate, b = avg_gain / avg_loss.
    Valeur informative uniquement — appliquer une fraction (ex: 0.25 * f*).
    """
    if not np.isfinite(avg_gain_loss_ratio) or avg_gain_loss_ratio <= 0:
        return float("nan")
    p = win_rate
    b = avg_gain_loss_ratio
    return float(p - (1.0 - p) / b)


def _market_stats(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    ann_factor: float,
    cagr: float,
    rf: float,
) -> tuple[float, float, float, float]:
    """
    Retourne (beta, treynor, jensen_alpha, r_squared).
    Requiert des rendements de marché alignés sur le même index.
    """
    common = strategy_returns.index.intersection(market_returns.index)
    if len(common) < 20:
        nan = float("nan")
        return nan, nan, nan, nan

    s = strategy_returns[common].values
    m = market_returns[common].values

    # Bêta
    cov_mat = np.cov(s, m, ddof=1)
    var_m = cov_mat[1, 1]
    if var_m < 1e-14:
        nan = float("nan")
        return nan, nan, nan, nan
    beta = float(cov_mat[0, 1] / var_m)

    # Treynor
    treynor = float((cagr - rf) / beta) if abs(beta) > 1e-12 else float("nan")

    # Jensen's Alpha
    mkt_ann_ret = float(market_returns[common].mean() * ann_factor)
    jensen_alpha = float(cagr - (rf + beta * (mkt_ann_ret - rf)))

    # R²
    corr = float(np.corrcoef(s, m)[0, 1])
    r_squared = float(corr ** 2) if np.isfinite(corr) else float("nan")

    return beta, treynor, jensen_alpha, r_squared


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(
    equity_curve: pd.DataFrame,
    weights_curve: pd.DataFrame,
    turnover_curve: pd.Series,
    breakdown: Dict[str, float],
    market_returns: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> PerformanceMetrics:
    """
    Calcule l'ensemble complet des métriques de performance.

    Args:
        equity_curve:   DataFrame, colonne "equity", index DatetimeIndex.
        weights_curve:  DataFrame, colonnes = symboles, valeurs = poids.
        turnover_curve: Series, sum(|Δw|) à chaque barre.
        breakdown:      dict avec clés pnl_gross, pnl_funding, fees, slippage, pnl_net.
        market_returns: Series de rendements du marché (ex: BTC) alignée sur l'index.
                        Requis pour beta, Treynor, Jensen alpha, R².
        rf:             Taux sans risque annualisé (défaut 0.0 pour crypto).
    """
    _nan = float("nan")

    def _empty() -> PerformanceMetrics:
        n = int(eq.shape[0])
        return PerformanceMetrics(
            start=idx[0] if n else pd.Timestamp("1970-01-01", tz="UTC"),
            end=idx[-1] if n else pd.Timestamp("1970-01-01", tz="UTC"),
            n_steps=n,
            total_return=_nan, cagr=_nan, sharpe=_nan, sortino=_nan, calmar=_nan,
            ann_vol=_nan, max_drawdown=_nan, avg_drawdown=_nan,
            max_dd_duration_bars=0, var_95=_nan,
            win_rate=_nan, loss_rate=_nan, profit_factor=_nan,
            avg_gain_loss_ratio=_nan, kelly_fraction=_nan,
            avg_gross=_nan, avg_net=_nan, avg_turnover=_nan,
            beta=_nan, treynor=_nan, jensen_alpha=_nan, r_squared=_nan,
            pnl_gross=float(breakdown.get("pnl_gross", 0.0)),
            pnl_funding=float(breakdown.get("pnl_funding", 0.0)),
            pnl_fees=float(breakdown.get("fees", 0.0)),
            pnl_slippage=float(breakdown.get("slippage", 0.0)),
            pnl_net=float(breakdown.get("pnl_net", 0.0)),
        )

    eq = equity_curve["equity"].astype(float)
    idx = eq.index
    n = int(eq.shape[0])

    if n < 2:
        return _empty()

    # ── Rendement ──────────────────────────────────────────────────────────────
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    ann_factor = _infer_steps_per_year(idx)
    if ann_factor > 0:
        years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 24 * 3600)
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / max(years, 1e-12)) - 1.0)
    else:
        cagr = _nan

    # ── Rendements barre par barre ─────────────────────────────────────────────
    r = eq.pct_change().dropna()

    # ── Volatilité & Sharpe ────────────────────────────────────────────────────
    if ann_factor > 0 and not r.empty:
        ann_vol = float(r.std(ddof=1) * np.sqrt(ann_factor))
        ann_ret = float(r.mean() * ann_factor)
        sharpe = float((ann_ret - rf) / ann_vol) if ann_vol > 1e-12 else _nan
    else:
        ann_vol, sharpe = _nan, _nan
        ann_ret = _nan

    # ── Sortino ────────────────────────────────────────────────────────────────
    sortino = _sortino(r, ann_factor, rf) if ann_factor > 0 and not r.empty else _nan

    # ── Drawdown ───────────────────────────────────────────────────────────────
    mdd = _max_drawdown(eq)
    avg_dd = _avg_drawdown(eq)
    max_dd_dur = _max_dd_duration(eq)

    # ── Calmar ────────────────────────────────────────────────────────────────
    calmar = float(cagr / abs(mdd)) if np.isfinite(cagr) and abs(mdd) > 1e-12 else _nan

    # ── VaR 95% ───────────────────────────────────────────────────────────────
    var_95 = _var_95(r) if not r.empty else _nan

    # ── Trade stats ───────────────────────────────────────────────────────────
    if not r.empty:
        win_rate, loss_rate, profit_factor, avg_gl = _trade_stats(r)
        kelly = _kelly(win_rate, avg_gl)
    else:
        win_rate = loss_rate = profit_factor = avg_gl = kelly = _nan

    # ── Exposition ────────────────────────────────────────────────────────────
    if not weights_curve.empty:
        gross = weights_curve.abs().sum(axis=1)
        net_exp = weights_curve.sum(axis=1)
        avg_gross = float(gross.mean())
        avg_net = float(net_exp.mean())
    else:
        avg_gross = avg_net = _nan
    avg_turnover = float(turnover_curve.mean()) if not turnover_curve.empty else _nan

    # ── Métriques marché (optionnelles) ───────────────────────────────────────
    if market_returns is not None and ann_factor > 0 and np.isfinite(cagr):
        beta, treynor, jensen_alpha, r_squared = _market_stats(
            r, market_returns, ann_factor, cagr, rf
        )
    else:
        beta = treynor = jensen_alpha = r_squared = _nan

    return PerformanceMetrics(
        start=idx[0],
        end=idx[-1],
        n_steps=n,
        total_return=total_ret,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        ann_vol=ann_vol,
        max_drawdown=mdd,
        avg_drawdown=avg_dd,
        max_dd_duration_bars=max_dd_dur,
        var_95=var_95,
        win_rate=win_rate,
        loss_rate=loss_rate,
        profit_factor=profit_factor,
        avg_gain_loss_ratio=avg_gl,
        kelly_fraction=kelly,
        avg_gross=avg_gross,
        avg_net=avg_net,
        avg_turnover=avg_turnover,
        beta=beta,
        treynor=treynor,
        jensen_alpha=jensen_alpha,
        r_squared=r_squared,
        pnl_gross=float(breakdown.get("pnl_gross", 0.0)),
        pnl_funding=float(breakdown.get("pnl_funding", 0.0)),
        pnl_fees=float(breakdown.get("fees", 0.0)),
        pnl_slippage=float(breakdown.get("slippage", 0.0)),
        pnl_net=float(breakdown.get("pnl_net", 0.0)),
    )


def metrics_to_dict(m: PerformanceMetrics, pct_fields: Optional[list[str]] = None) -> Dict[str, str]:
    """
    Convertit PerformanceMetrics en dict formaté pour affichage.

    Les champs de rendement/risque sont multipliés par 100 et affichés en %.
    Les ratios (Sharpe, Sortino, etc.) sont affichés bruts avec 3 décimales.
    """
    if pct_fields is None:
        pct_fields = {
            "total_return", "cagr", "ann_vol", "max_drawdown", "avg_drawdown",
            "var_95", "win_rate", "loss_rate", "avg_gross", "avg_net", "avg_turnover",
            "pnl_gross", "pnl_funding", "pnl_fees", "pnl_slippage", "pnl_net",
        }
    result: Dict[str, str] = {}
    for k, v in m.__dict__.items():
        if isinstance(v, pd.Timestamp):
            result[k] = str(v)
        elif isinstance(v, int):
            result[k] = str(v)
        elif isinstance(v, float):
            if not np.isfinite(v):
                result[k] = "—"
            elif k in pct_fields:
                result[k] = f"{v * 100:.2f}%"
            else:
                result[k] = f"{v:.4f}"
        else:
            result[k] = str(v)
    return result
