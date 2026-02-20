# src/hyperstat/backtest/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hyperstat.core.risk import KillSwitchConfig, RiskState
from hyperstat.core.types import PortfolioWeights, Signal
from hyperstat.data.features import compute_returns, compute_ewma_vol, compute_rv_1h_pct
from hyperstat.strategy.stat_arb import StatArbStrategy
from hyperstat.strategy.regime import RegimeModel
from hyperstat.strategy.funding_overlay import FundingOverlayModel
from hyperstat.strategy.allocator import PortfolioAllocator
from .costs import CostModel, TradeCostBreakdown, cost_model_from_config
from .metrics import compute_performance_metrics
from .reports import BacktestReport


def _to_utc_ts(s: pd.Series | pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(s, utc=True, errors="coerce")
    idx = pd.DatetimeIndex(idx).dropna()
    return idx


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing cols {missing}. Got={list(df.columns)}")


def _align_candles(candles_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s, df in candles_by_symbol.items():
        if df is None or df.empty:
            out[s] = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            continue
        df = df.copy()
        _require_cols(df, ["ts", "open", "high", "low", "close", "volume"], f"candles[{s}]")
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts")
        out[s] = df
    return out


def _align_funding(funding_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s, df in funding_by_symbol.items():
        if df is None or df.empty:
            out[s] = pd.DataFrame(columns=["ts", "rate"])
            continue
        df = df.copy()
        _require_cols(df, ["ts", "rate"], f"funding[{s}]")
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts")
        out[s] = df
    return out


def _timeline(candles_by_symbol: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """
    Use union timeline of all candle timestamps.
    For missing symbol bars we will ffill price/volume in series precomputation.
    """
    idxs = []
    for df in candles_by_symbol.values():
        if df is not None and not df.empty:
            idxs.append(pd.DatetimeIndex(df["ts"]))
    if not idxs:
        return pd.DatetimeIndex([], tz="UTC")
    idx = idxs[0].union_many(idxs[1:]).sort_values()
    idx = pd.DatetimeIndex(pd.to_datetime(idx, utc=True, errors="coerce")).dropna()
    return idx


def _precompute_series(
    candles_by_symbol: Dict[str, pd.DataFrame],
    timeframe_minutes: int,
) -> Tuple[
    Dict[str, pd.Series],  # close
    Dict[str, pd.Series],  # returns
    Dict[str, pd.Series],  # ewma vol
    Dict[str, pd.Series],  # rv_1h
    Dict[str, pd.Series],  # dollar volume dv (close*volume)
]:
    close: Dict[str, pd.Series] = {}
    rets: Dict[str, pd.Series] = {}
    vol: Dict[str, pd.Series] = {}
    rv1h: Dict[str, pd.Series] = {}
    dv: Dict[str, pd.Series] = {}

    for s, df in candles_by_symbol.items():
        if df.empty:
            close[s] = pd.Series(dtype=float)
            rets[s] = pd.Series(dtype=float)
            vol[s] = pd.Series(dtype=float)
            rv1h[s] = pd.Series(dtype=float)
            dv[s] = pd.Series(dtype=float)
            continue

        dfi = df.set_index("ts").sort_index()
        c = dfi["close"].astype(float)
        v = dfi["volume"].astype(float)
        close[s] = c
        dv[s] = (c * v).rename("dv")

        r = compute_returns(df).astype(float)
        rets[s] = r

        # EWMA vol of log returns (per step, not annualized)
        vol[s] = compute_ewma_vol(r, lam=0.94, min_periods=max(20, int(60 / timeframe_minutes))).rename("ewma_vol")

        # 1h realized vol
        rv1h[s] = compute_rv_1h_pct(df, timeframe_minutes=timeframe_minutes).rename("rv_1h")

    return close, rets, vol, rv1h, dv


def _funding_map(funding_by_symbol: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Map funding timestamp -> {symbol: rate}.
    If a symbol has multiple entries at same ts, last wins.
    """
    m: Dict[pd.Timestamp, Dict[str, float]] = {}
    for s, df in funding_by_symbol.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["ts"])
            ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
            rate = float(row["rate"])
            m.setdefault(ts, {})[s] = rate
    return m


def _tf_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {tf}")


@dataclass(frozen=True)
class BacktestConfig:
    """
    Minimal config needed by engine (extracted from YAML dict).
    """
    timeframe: str
    base_factor_symbol: str
    initial_equity: float
    fill_at: str = "close"  # close only in this version
    flat_when_no_trade: bool = False


class BacktestEngine:
    """
    Close-to-close backtest with instantaneous rebalance at each bar close.

    PnL components:
      - pnl_price: sum_{i} w_{t-1,i} * equity_{t-1} * (P_t/P_{t-1}-1)
      - pnl_funding: - sum_i (w_{t-1,i} * equity_{t-1}) * funding_rate_t  (when funding ts hits)
      - costs: fees+slippage on rebal notional traded at t

    Notes:
      - weights are dimensionless "notional / equity"
      - beta/dollar neutralization is handled in allocator
    """

    def __init__(
        self,
        cfg: BacktestConfig,
        candles_by_symbol: Dict[str, pd.DataFrame],
        funding_by_symbol: Dict[str, pd.DataFrame],
        buckets: Dict[str, List[str]],
        stat_arb: StatArbStrategy,
        regime_model: RegimeModel,
        allocator: PortfolioAllocator,
        funding_overlay: Optional[FundingOverlayModel] = None,
        cost_model: Optional[CostModel] = None,
        kill_switch_cfg: Optional[KillSwitchConfig] = None,
    ) -> None:
        self.cfg = cfg
        self.candles = _align_candles(candles_by_symbol)
        self.funding = _align_funding(funding_by_symbol)
        self.buckets = buckets

        self.stat_arb = stat_arb
        self.regime_model = regime_model
        self.allocator = allocator
        self.funding_overlay = funding_overlay

        self.cost_model = cost_model or cost_model_from_config({})

        ks = kill_switch_cfg or KillSwitchConfig(max_intraday_drawdown_pct=0.03, cooldown_minutes=720, z_emergency_flat=3.5)
        self.risk_state = RiskState(config=ks)

        self.tfm = _tf_minutes(cfg.timeframe)

        # Precompute time series
        self.timeline = _timeline(self.candles)
        self.close, self.rets, self.vol, self.rv1h, self.dv = _precompute_series(self.candles, timeframe_minutes=self.tfm)

        self.funding_events = _funding_map(self.funding)

        # cache last prices (for ffill in live loop)
        self._last_px: Dict[str, float] = {}
        self._last_dv: Dict[str, float] = {}

    def _px(self, sym: str, ts: pd.Timestamp) -> Optional[float]:
        s = self.close.get(sym)
        if s is None or s.empty:
            return None
        # exact match first
        if ts in s.index:
            v = s.loc[ts]
            if np.isfinite(v):
                self._last_px[sym] = float(v)
                return float(v)
        # ffill from last seen
        return self._last_px.get(sym)

    def _dv_at(self, sym: str, ts: pd.Timestamp) -> Optional[float]:
        s = self.dv.get(sym)
        if s is None or s.empty:
            return None
        if ts in s.index:
            v = s.loc[ts]
            if np.isfinite(v):
                self._last_dv[sym] = float(v)
                return float(v)
        return self._last_dv.get(sym)

    def _rv1h_at(self, sym: str, ts: pd.Timestamp) -> Optional[float]:
        s = self.rv1h.get(sym)
        if s is None or s.empty:
            return None
        # ffill is ok
        v = s.reindex([ts], method="ffill").iloc[0] if ts not in s.index else s.loc[ts]
        return float(v) if np.isfinite(v) else None

    def _vol_at(self, sym: str, ts: pd.Timestamp) -> Optional[float]:
        s = self.vol.get(sym)
        if s is None or s.empty:
            return None
        v = s.reindex([ts], method="ffill").iloc[0] if ts not in s.index else s.loc[ts]
        return float(v) if np.isfinite(v) else None

    def run(self) -> BacktestReport:
        if self.timeline.empty:
            raise ValueError("Empty timeline: no candle data.")

        symbols = sorted({s for b in self.buckets.values() for s in b})
        # ensure base factor present if needed for regime
        if self.cfg.base_factor_symbol not in symbols and self.cfg.base_factor_symbol in self.candles:
            symbols = [self.cfg.base_factor_symbol] + symbols

        # State
        equity = float(self.cfg.initial_equity)
        weights: Dict[str, float] = {s: 0.0 for s in symbols}

        # Curves
        eq_rows = []
        pnl_rows = []
        w_rows = []
        turnover_rows = []

        # Aggregated breakdown
        pnl_gross = 0.0
        pnl_funding = 0.0
        fees = 0.0
        slippage = 0.0

        prev_ts: Optional[pd.Timestamp] = None
        prev_prices: Dict[str, float] = {}

        for ts in self.timeline:
            ts = pd.Timestamp(ts).tz_convert("UTC") if ts.tzinfo else pd.Timestamp(ts, tz="UTC")

            # Build mids at ts
            mids: Dict[str, float] = {}
            for s in symbols:
                px = self._px(s, ts)
                if px is not None and np.isfinite(px) and px > 0:
                    mids[s] = float(px)

            # --- 1) price PnL from previous weights
            pnl_price_step = 0.0
            if prev_ts is not None:
                for s, w in weights.items():
                    if abs(w) < 1e-12:
                        continue
                    p0 = prev_prices.get(s)
                    p1 = mids.get(s)
                    if p0 is None or p1 is None or p0 <= 0:
                        continue
                    r = (p1 / p0) - 1.0
                    pnl_price_step += float(w * equity * r)

            # Apply price pnl
            equity_before_funding = equity + pnl_price_step

            # --- 2) funding PnL if funding event at ts
            pnl_funding_step = 0.0
            if ts in self.funding_events:
                # Update overlay stats BEFORE computing target (so it can tilt)
                if self.funding_overlay is not None:
                    self.funding_overlay.update(self.funding_events[ts])

                for s, rate in self.funding_events[ts].items():
                    if s not in weights:
                        continue
                    w = float(weights.get(s, 0.0))
                    # funding pnl = -notional * rate  (positive rate => longs pay)
                    pnl_funding_step += float(-(w * equity_before_funding) * float(rate))

            equity_before_rebal = equity_before_funding + pnl_funding_step

            # --- 3) risk kill-switch update
            self.risk_state.on_equity(ts.to_pydatetime(), equity_before_rebal)
            can_trade = self.risk_state.trading_allowed(ts.to_pydatetime())

            # --- 4) compute features dict at ts (minimal set used by allocator/regime)
            features: Dict[str, Dict[str, float]] = {}
            for s in symbols:
                d: Dict[str, float] = {}
                dv = self._dv_at(s, ts)
                if dv is not None:
                    d["dv"] = float(dv)

                # In this version, illiq is not rolling. If you want rolling illiq, we can add later.
                # We'll provide a constant fallback at 0.0 (regime treats missing gracefully).
                # d["illiq"] = ...

                vol = self._vol_at(s, ts)
                if vol is not None:
                    d["ewma_vol"] = float(vol)

                rv = self._rv1h_at(s, ts)
                if rv is not None:
                    d["rv_1h"] = float(rv)

                features[s] = d

            # --- 5) compute signal/regime/weights target
            signal: Signal = self.stat_arb.update(ts.to_pydatetime(), mids, self.buckets)
            regime = self.regime_model.update(ts.to_pydatetime(), signal, self.buckets, features)

            # Betas: optional; you can wire them later from data.features.compute_beta_vs_factor
            betas: Optional[Dict[str, float]] = None

            if not can_trade:
                # During cooldown: force flat
                target = PortfolioWeights(
                    ts=ts.to_pydatetime(),
                    weights={s: 0.0 for s in symbols},
                    gross=0.0,
                    net=0.0,
                    beta=0.0,
                    meta={"cooldown_minutes": str(self.risk_state.cooldown_remaining_minutes(ts.to_pydatetime()))},
                )
            else:
                target = self.allocator.allocate(
                    ts=ts.to_pydatetime(),
                    signal=signal,
                    regime=regime,
                    buckets=self.buckets,
                    features=features,
                    betas=betas,
                )

            # --- 6) compute turnover + trading costs for rebalance
            delta_w = {s: float(target.weights.get(s, 0.0) - weights.get(s, 0.0)) for s in symbols}
            turnover = float(sum(abs(v) for v in delta_w.values()))
            turnover_rows.append((ts, turnover))

            cost_bd = TradeCostBreakdown()
            # cost per symbol trade
            for s, dw in delta_w.items():
                if abs(dw) < 1e-12:
                    continue
                notional_trade = abs(dw) * equity_before_rebal
                rv = features.get(s, {}).get("rv_1h")
                total_cost, fee_cost, slip_cost = self.cost_model.trade_cost(notional_trade, rv_1h=rv)
                cost_bd.add(total_cost, fee_cost, slip_cost)

            equity_after = equity_before_rebal - cost_bd.total

            # --- 7) commit step
            pnl_net_step = (equity_after - equity)
            pnl_price_committed = pnl_price_step
            pnl_funding_committed = pnl_funding_step

            pnl_gross += pnl_price_committed
            pnl_funding += pnl_funding_committed
            fees += cost_bd.fees
            slippage += cost_bd.slippage

            equity = equity_after
            weights = {s: float(target.weights.get(s, 0.0)) for s in symbols}

            # update prev prices
            prev_prices = dict(mids)
            prev_ts = ts

            # curves
            eq_rows.append((ts, equity))
            pnl_rows.append(
                (
                    ts,
                    pnl_price_committed,
                    pnl_funding_committed,
                    -cost_bd.total,
                    pnl_net_step,
                    float(cost_bd.fees),
                    float(cost_bd.slippage),
                    float(target.gross),
                    float(target.net),
                )
            )
            w_rows.append((ts, {**weights}))

        # Build DataFrames
        equity_curve = pd.DataFrame(eq_rows, columns=["ts", "equity"]).set_index("ts")
        pnl_curve = pd.DataFrame(
            pnl_rows,
            columns=["ts", "pnl_price", "pnl_funding", "costs", "pnl_net_step", "fees", "slippage", "gross", "net"],
        ).set_index("ts")

        weights_df = pd.DataFrame([row[1] for row in w_rows], index=[row[0] for row in w_rows]).fillna(0.0)
        turnover_s = pd.Series([x[1] for x in turnover_rows], index=[x[0] for x in turnover_rows], name="turnover")

        breakdown = {
            "pnl_gross": float(pnl_gross),
            "pnl_funding": float(pnl_funding),
            "fees": float(fees),
            "slippage": float(slippage),
            "pnl_net": float(pnl_gross + pnl_funding - fees - slippage),
        }

        metrics = compute_performance_metrics(
            equity_curve=equity_curve,
            weights_curve=weights_df,
            turnover_curve=turnover_s,
            breakdown=breakdown,
        )

        meta = {
            "timeframe": self.cfg.timeframe,
            "base_factor_symbol": self.cfg.base_factor_symbol,
            "n_symbols": str(weights_df.shape[1]),
        }

        return BacktestReport(
            equity_curve=equity_curve,
            pnl_curve=pnl_curve,
            turnover=turnover_s,
            weights=weights_df,
            metrics=metrics,
            breakdown=breakdown,
            meta=meta,
        )


def backtest_config_from_config(cfg: dict) -> BacktestConfig:
    data = (cfg.get("data", {}) or {})
    portfolio = (cfg.get("portfolio", {}) or {})
    timeframe = str(data.get("timeframe", "5m"))
    base_factor = str(data.get("base_factor_symbol", "BTC"))
    initial_equity = float(portfolio.get("initial_equity_eur", 1500.0))  # treated as quote ccy in backtest

    fill_model = str((cfg.get("execution", {}) or {}).get("fill_model", "close")).lower()
    fill_at = "close" if "close" in fill_model else "close"

    return BacktestConfig(
        timeframe=timeframe,
        base_factor_symbol=base_factor,
        initial_equity=initial_equity,
        fill_at=fill_at,
    )


def run_backtest(
    cfg: dict,
    candles_by_symbol: Dict[str, pd.DataFrame],
    funding_by_symbol: Dict[str, pd.DataFrame],
    buckets: Dict[str, List[str]],
    stat_arb: StatArbStrategy,
    regime_model: RegimeModel,
    allocator: PortfolioAllocator,
    funding_overlay: Optional[FundingOverlayModel] = None,
) -> BacktestReport:
    bt_cfg = backtest_config_from_config(cfg)
    cost_model = cost_model_from_config(cfg)

    risk = (cfg.get("risk", {}) or {})
    ks = KillSwitchConfig(
        max_intraday_drawdown_pct=float(risk.get("max_intraday_drawdown_pct", 0.03)),
        cooldown_minutes=int(risk.get("cooldown_minutes", 720)),
        z_emergency_flat=float(risk.get("z_emergency_flat", 3.5)),
    )

    eng = BacktestEngine(
        cfg=bt_cfg,
        candles_by_symbol=candles_by_symbol,
        funding_by_symbol=funding_by_symbol,
        buckets=buckets,
        stat_arb=stat_arb,
        regime_model=regime_model,
        allocator=allocator,
        funding_overlay=funding_overlay,
        cost_model=cost_model,
        kill_switch_cfg=ks,
    )
    return eng.run()
