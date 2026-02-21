# src/hyperstat/data/universe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .features import (
    compute_returns,
    compute_amihud_illiq,
    compute_beta_vs_factor,
    compute_residual_returns,
    compute_funding_ewma_stats,
)


@dataclass(frozen=True)
class UniverseStats:
    symbol: str
    missing_ratio: float
    dv_median: float
    illiq_amihud: float
    funding_vol: float  # proxy: EWMA MAD of funding
    keep: bool


def _missing_ratio(candles: pd.DataFrame, expected_bars: int) -> float:
    if expected_bars <= 0:
        return 1.0
    n = len(candles)
    miss = max(0, expected_bars - n)
    return float(miss / expected_bars)


def _expected_bars(days: int, timeframe_minutes: int) -> int:
    return int(days * 24 * 60 / timeframe_minutes)


def _timeframe_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {tf}")


def select_universe(
    candles_by_symbol: Dict[str, pd.DataFrame],
    funding_by_symbol: Dict[str, pd.DataFrame],
    timeframe: str,
    lookback_days: int,
    target_size: int,
    max_missing_ratio: float,
    amihud_exclude_pct: float = 0.80,
    funding_vol_exclude_pct: float = 0.80,
) -> Tuple[List[str], List[UniverseStats]]:
    """
    Returns:
      - selected symbols (size <= target_size)
      - stats per symbol (debuggable)

    Filters:
      1) missing ratio <= max_missing_ratio
      2) take top by DV median
      3) drop worst amihud (top percentiles)
      4) drop worst funding vol (top percentiles)
    """
    tfm = _timeframe_minutes(timeframe)
    exp = _expected_bars(lookback_days, tfm)

    stats: List[UniverseStats] = []
    for sym, cdf in candles_by_symbol.items():
        cdf = cdf.copy()
        cdf["ts"] = pd.to_datetime(cdf["ts"], utc=True, errors="coerce")
        cdf = cdf.dropna(subset=["ts"]).sort_values("ts")

        miss = _missing_ratio(cdf, exp)

        if cdf.empty or "close" not in cdf.columns or "volume" not in cdf.columns:
            dv_med = float("nan")
            illiq = float("nan")
        else:
            dv = cdf["close"].astype(float) * cdf["volume"].astype(float)
            dv_med = float(np.nanmedian(dv.values))
            illiq = compute_amihud_illiq(cdf)

        fdf = funding_by_symbol.get(sym, pd.DataFrame(columns=["ts", "rate"]))
        fstats = compute_funding_ewma_stats(fdf, lam=0.15)
        # funding volatility proxy = last available MAD
        fund_vol = float(fstats.mad.dropna().iloc[-1]) if not fstats.mad.dropna().empty else float("nan")

        stats.append(
            UniverseStats(
                symbol=sym,
                missing_ratio=miss,
                dv_median=dv_med,
                illiq_amihud=illiq,
                funding_vol=fund_vol,
                keep=True,
            )
        )

    df = pd.DataFrame([s.__dict__ for s in stats]).set_index("symbol")

    # filter missing first
    df["keep"] = df["missing_ratio"] <= max_missing_ratio
    df_keep = df[df["keep"]].copy()

    # if too few, return what we can
    if df_keep.empty:
        return [], [UniverseStats(**row) for row in df.reset_index().to_dict(orient="records")]

    # take top by dv_median
    df_keep = df_keep.sort_values("dv_median", ascending=False).head(target_size * 2)  # buffer before pruning

    # prune amihud worst
    if df_keep["illiq_amihud"].notna().sum() > 5:
        thr_illiq = df_keep["illiq_amihud"].quantile(amihud_exclude_pct)
        df_keep = df_keep[df_keep["illiq_amihud"] <= thr_illiq]

    # prune funding vol worst
    if df_keep["funding_vol"].notna().sum() > 5:
        thr_fv = df_keep["funding_vol"].quantile(funding_vol_exclude_pct)
        df_keep = df_keep[df_keep["funding_vol"] <= thr_fv]

    # final pick top by dv
    selected = list(df_keep.sort_values("dv_median", ascending=False).head(target_size).index)

    # update keep flags for returned stats
    sel_set = set(selected)
    out_stats = []
    for s in stats:
        out_stats.append(
            UniverseStats(
                symbol=s.symbol,
                missing_ratio=s.missing_ratio,
                dv_median=s.dv_median,
                illiq_amihud=s.illiq_amihud,
                funding_vol=s.funding_vol,
                keep=(s.symbol in sel_set),
            )
        )

    return selected, out_stats


def build_buckets(
    candles_by_symbol: Dict[str, pd.DataFrame],
    base_factor_symbol: str,
    k_min: int = 4,
    k_max: int = 6,
    min_bucket_size: int = 4,
    max_bucket_size: int = 10,
    beta_window: int = 12 * 24 * 7,
) -> Dict[str, List[str]]:
    """
    Builds buckets by clustering correlation distance of residual returns (beta-neutralized vs factor).

    Steps:
      1) returns for each symbol
      2) rolling beta vs factor (BTC)
      3) residual returns eps_i = r_i - beta_i * r_factor
      4) correlation matrix of eps over overlapping timestamps
      5) hierarchical clustering with correlation distance
    """
    if base_factor_symbol not in candles_by_symbol:
        raise KeyError(f"base_factor_symbol missing in candles_by_symbol: {base_factor_symbol}")

    # returns dict
    returns: Dict[str, pd.Series] = {}
    for sym, cdf in candles_by_symbol.items():
        if cdf.empty:
            returns[sym] = pd.Series(dtype=float)
            continue
        returns[sym] = compute_returns(cdf)

    rf = returns[base_factor_symbol].rename("rf")
    if rf.empty:
        raise ValueError("Factor returns are empty; cannot build buckets.")

    # compute betas and residuals
    betas = compute_beta_vs_factor(returns, rf, window=beta_window)
    eps = compute_residual_returns(returns, rf, betas)

    # assemble matrix for clustering (only symbols with enough overlap)
    symbols = [s for s in eps.keys() if s != base_factor_symbol]
    if len(symbols) < k_min:
        # fallback: single bucket
        return {"bucket_0": symbols}

    # align into a dataframe
    df_eps = pd.concat([eps[s].rename(s) for s in symbols], axis=1).dropna(how="all")
    # drop columns with too few obs
    min_obs = max(200, int(df_eps.shape[0] * 0.2))
    keep_cols = [c for c in df_eps.columns if df_eps[c].dropna().shape[0] >= min_obs]
    df_eps = df_eps[keep_cols].dropna()

    if df_eps.shape[1] < k_min:
        return {"bucket_0": list(df_eps.columns)}

    corr = df_eps.corr().fillna(0.0).clip(-1.0, 1.0)
    # distance matrix
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    # pick K between k_min and k_max that yields valid bucket sizes
    best_mapping: Optional[Dict[str, List[str]]] = None

    for K in range(k_min, k_max + 1):
        labels = fcluster(Z, t=K, criterion="maxclust")
        mapping: Dict[int, List[str]] = {}
        for sym, lab in zip(df_eps.columns, labels):
            mapping.setdefault(int(lab), []).append(sym)

        # enforce bucket size constraints by merging small buckets into nearest large bucket
        mapping = _merge_small_buckets(mapping, corr, min_bucket_size=min_bucket_size)

        # cap large buckets by splitting (simple: keep max_bucket_size, spill to new bucket ids)
        mapping2: Dict[str, List[str]] = {}
        bidx = 0
        for _, syms in mapping.items():
            if len(syms) <= max_bucket_size:
                mapping2[f"bucket_{bidx}"] = syms
                bidx += 1
            else:
                # chunk
                for j in range(0, len(syms), max_bucket_size):
                    mapping2[f"bucket_{bidx}"] = syms[j : j + max_bucket_size]
                    bidx += 1

        # accept if we have at least k_min buckets with decent sizes
        sizes = [len(v) for v in mapping2.values()]
        if len(mapping2) >= k_min and all(s >= min(2, min_bucket_size) for s in sizes):
            best_mapping = mapping2
            break

    if best_mapping is None:
        # fallback: single bucket
        best_mapping = {"bucket_0": list(df_eps.columns)}

    return best_mapping


def _merge_small_buckets(
    mapping: Dict[int, List[str]],
    corr: pd.DataFrame,
    min_bucket_size: int,
) -> Dict[int, List[str]]:
    """
    Merge buckets smaller than min_bucket_size into the most correlated bucket.
    """
    # separate small and big buckets
    small = {k: v for k, v in mapping.items() if len(v) < min_bucket_size}
    big = {k: v for k, v in mapping.items() if len(v) >= min_bucket_size}

    if not small or not big:
        return mapping

    for sk, syms in small.items():
        # find best target bucket based on average corr with its members
        best_k = None
        best_score = -1e9
        for bk, bsyms in big.items():
            # score = mean corr between all pairs (syms x bsyms)
            sub = corr.loc[syms, bsyms]
            score = float(np.nanmean(sub.values))
            if score > best_score:
                best_score = score
                best_k = bk
        if best_k is None:
            # arbitrary merge to first big
            best_k = next(iter(big.keys()))
        big[best_k] = big[best_k] + syms

    return big
