# src/hyperstat/strategy/regime.py
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

import numpy as np

from hyperstat.core.math import fit_ar1, half_life_minutes, clip
from hyperstat.core.types import RegimeScore, Signal


@dataclass(frozen=True)
class RegimeConfig:
    """
    Regime gating model Q_t = Q_MR * Q_LIQ * Q_RISK

    Inputs required each step (via features dicts):
      - dv: dollar volume proxy (close*volume)
      - illiq: Amihud illiq proxy (can be slow-moving)
      - rv_1h: realized vol 1h (or ewma_vol) for risk gating
      - base_factor_symbol rv_1h for BTC volatility
    """
    timeframe_minutes: int = 5

    # Mean reversion test on bucket spread AR(1)
    ar1_window_days: int = 7
    halflife_min_minutes: int = 30
    halflife_good_max_minutes: int = 360
    halflife_ok_max_minutes: int = 1440

    # Liquidity score: if missing, defaults to 1
    liq_gate_min: float = 0.30

    # Risk score: percentile thresholds computed from rolling history
    risk_history_days: int = 60
    high_vol_pctl: float = 0.90
    extreme_vol_pctl: float = 0.95


@dataclass
class RegimeModel:
    cfg: RegimeConfig
    base_factor_symbol: str = "BTC"

    # bucket_id -> spread history
    _bucket_spreads: Dict[str, Deque[float]] = field(default_factory=dict)

    # rolling histories for risk quantiles
    _btc_vol_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=60 * 24 * 12))  # 60d of 5m
    _uni_vol_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=60 * 24 * 12))

    def _bucket_maxlen(self) -> int:
        bars_per_day = int(24 * 60 / self.cfg.timeframe_minutes)
        return max(50, self.cfg.ar1_window_days * bars_per_day)

    def _risk_maxlen(self) -> int:
        bars_per_day = int(24 * 60 / self.cfg.timeframe_minutes)
        return max(200, self.cfg.risk_history_days * bars_per_day)

    def update(
        self,
        ts: datetime,
        signal: Signal,
        buckets: Dict[str, List[str]],
        features: Dict[str, Dict[str, float]],
    ) -> RegimeScore:
        """
        features: symbol -> {"dv":..., "illiq":..., "rv_1h":..., ...}
        """
        # ---- Q_MR: bucket spread AR(1) half-life gating
        q_mr_vals: List[float] = []
        maxlen = self._bucket_maxlen()

        for b_id in buckets.keys():
            key = f"spread:{b_id}"
            spread = signal.meta.get(key)
            if spread is None or not np.isfinite(spread):
                continue

            if b_id not in self._bucket_spreads:
                self._bucket_spreads[b_id] = deque(maxlen=maxlen)

            self._bucket_spreads[b_id].append(float(spread))

            hist = np.asarray(self._bucket_spreads[b_id], dtype=float)
            if hist.size < max(30, int(0.3 * maxlen)):
                continue

            _a, b = fit_ar1(hist)
            hl = half_life_minutes(b, dt_minutes=float(self.cfg.timeframe_minutes))

            if not (0.0 < b < 1.0):
                q = 0.0
            else:
                if hl < self.cfg.halflife_min_minutes:
                    q = 0.5  # too fast can also be noise; be conservative
                elif hl <= self.cfg.halflife_good_max_minutes:
                    q = 1.0
                elif hl <= self.cfg.halflife_ok_max_minutes:
                    q = 0.5
                else:
                    q = 0.0
            q_mr_vals.append(float(q))

        q_mr = float(np.nanmedian(q_mr_vals)) if q_mr_vals else 1.0

        # ---- Q_LIQ: cross-sectional liquidity proxy gating
        # dv high is good; illiq low is good. We map to percentile ranks at time t.
        dvs = []
        illiqs = []
        syms = list(features.keys())

        for s in syms:
            dv = features[s].get("dv")
            ill = features[s].get("illiq")
            if dv is not None and np.isfinite(dv):
                dvs.append(float(dv))
            if ill is not None and np.isfinite(ill):
                illiqs.append(float(ill))

        if len(dvs) < 5 or len(illiqs) < 5:
            q_liq = 1.0
        else:
            dv_arr = np.asarray(dvs, dtype=float)
            ill_arr = np.asarray(illiqs, dtype=float)

            # per-symbol q_liq_i = min(rank_pct(dv), 1-rank_pct(illiq))
            q_vals = []
            dv_sorted = np.sort(dv_arr)
            ill_sorted = np.sort(ill_arr)

            def _rank_pct(sorted_arr: np.ndarray, x: float) -> float:
                # pct in [0,1]
                return float(np.searchsorted(sorted_arr, x, side="right") / max(1, sorted_arr.size))

            for s in syms:
                dv = features[s].get("dv")
                ill = features[s].get("illiq")
                if dv is None or ill is None or (not np.isfinite(dv)) or (not np.isfinite(ill)):
                    continue
                q_dv = _rank_pct(dv_sorted, float(dv))
                q_ill = 1.0 - _rank_pct(ill_sorted, float(ill))
                q_vals.append(min(q_dv, q_ill))

            q_liq = float(np.nanmedian(q_vals)) if q_vals else 1.0
            if q_liq < self.cfg.liq_gate_min:
                q_liq = 0.0

        # ---- Q_RISK: volatility regime gating using rolling percentiles
        # Prefer BTC rv_1h as global risk. Fallback to universe median rv_1h.
        btc_rv = features.get(self.base_factor_symbol, {}).get("rv_1h")
        uni_rvs = [features[s].get("rv_1h") for s in syms if features[s].get("rv_1h") is not None]
        uni_rvs = [float(x) for x in uni_rvs if np.isfinite(float(x))]

        if btc_rv is not None and np.isfinite(btc_rv):
            self._btc_vol_hist = deque(self._btc_vol_hist, maxlen=self._risk_maxlen())
            self._btc_vol_hist.append(float(btc_rv))
        if uni_rvs:
            self._uni_vol_hist = deque(self._uni_vol_hist, maxlen=self._risk_maxlen())
            self._uni_vol_hist.append(float(np.nanmedian(uni_rvs)))

        q_risk = 1.0
        if len(self._btc_vol_hist) >= 200:
            hist = np.asarray(self._btc_vol_hist, dtype=float)
            hi = float(np.nanquantile(hist, self.cfg.high_vol_pctl))
            ex = float(np.nanquantile(hist, self.cfg.extreme_vol_pctl))
            cur = float(self._btc_vol_hist[-1])

            if cur >= ex:
                q_risk = 0.0
            elif cur >= hi:
                q_risk = 0.5
            else:
                q_risk = 1.0
        elif len(self._uni_vol_hist) >= 200:
            hist = np.asarray(self._uni_vol_hist, dtype=float)
            hi = float(np.nanquantile(hist, self.cfg.high_vol_pctl))
            ex = float(np.nanquantile(hist, self.cfg.extreme_vol_pctl))
            cur = float(self._uni_vol_hist[-1])
            if cur >= ex:
                q_risk = 0.0
            elif cur >= hi:
                q_risk = 0.5
            else:
                q_risk = 1.0

        return RegimeScore(ts=ts, q_mr=clip(q_mr, 0.0, 1.0), q_liq=clip(q_liq, 0.0, 1.0), q_risk=clip(q_risk, 0.0, 1.0))
