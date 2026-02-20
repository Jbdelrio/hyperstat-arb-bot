# src/hyperstat/strategy/stat_arb.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from hyperstat.core.math import mad, clip
from hyperstat.core.types import Signal


def _minutes_between(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 60.0


@dataclass(frozen=True)
class StatArbConfig:
    """
    Cross-sectional mean reversion inside each bucket.

    This module ONLY produces a raw contrarian signal (weights_raw).
    Sizing / vol-scaling / neutralization / caps are handled in allocator.
    """
    timeframe_minutes: int = 5
    horizon_bars: int = 12  # 12*5m = 1h

    z_in: float = 1.5
    z_out: float = 0.5
    z_max: float = 3.0

    min_hold_minutes: int = 30
    max_hold_minutes: int = 1440  # 24h

    # For bucket spread (used by regime MR model)
    spread_quantile: float = 0.20

    eps: float = 1e-12


@dataclass
class _LegState:
    """
    Per-symbol hysteresis state to reduce churn.
    """
    active: bool = False
    entered_at: Optional[datetime] = None
    side: int = 0  # +1 means long bias, -1 short bias (based on z when entered)


@dataclass
class StatArbState:
    """
    Stateful strategy memory:
    - price history for horizon returns
    - hysteresis activation per symbol
    """
    price_hist: Dict[str, Deque[Tuple[datetime, float]]] = field(default_factory=dict)
    leg_state: Dict[str, _LegState] = field(default_factory=dict)

    def _ensure_symbol(self, symbol: str, maxlen: int) -> None:
        if symbol not in self.price_hist:
            self.price_hist[symbol] = deque(maxlen=maxlen)
        if symbol not in self.leg_state:
            self.leg_state[symbol] = _LegState()


class StatArbStrategy:
    """
    Produces robust z-scores within each bucket, then applies hysteresis to decide
    whether each leg is active (to avoid constant micro-rebalances).

    Input each step:
      - ts: current timestamp (UTC)
      - mids: {symbol: mid/close}
      - buckets: {bucket_id: [symbols...]}

    Output:
      - Signal(weights_raw, zscores, meta including bucket spreads)
    """

    def __init__(self, cfg: StatArbConfig, state: Optional[StatArbState] = None) -> None:
        self.cfg = cfg
        self.state = state or StatArbState()

    def update(self, ts: datetime, mids: Dict[str, float], buckets: Dict[str, List[str]]) -> Signal:
        # update rolling price history
        maxlen = self.cfg.horizon_bars + 1
        for sym, px in mids.items():
            self.state._ensure_symbol(sym, maxlen=maxlen)
            self.state.price_hist[sym].append((ts, float(px)))

        # compute horizon returns (log) for symbols that have enough history
        hret: Dict[str, float] = {}
        for sym, hist in self.state.price_hist.items():
            if len(hist) < (self.cfg.horizon_bars + 1):
                continue
            # horizon return between now and H bars ago (assumes fixed bar spacing in backtest/live loop)
            p_now = hist[-1][1]
            p_old = hist[0][1]
            if p_now > 0 and p_old > 0:
                hret[sym] = float(np.log(p_now / p_old))

        zscores: Dict[str, float] = {}
        weights_raw: Dict[str, float] = {}
        meta: Dict[str, float] = {}

        # per-bucket z-score computation
        for b_id, syms in buckets.items():
            vals = [hret[s] for s in syms if s in hret and np.isfinite(hret[s])]
            if len(vals) < 4:
                # too few points => no signal
                meta[f"spread:{b_id}"] = float("nan")
                continue

            arr = np.asarray(vals, dtype=float)
            med = float(np.nanmedian(arr))
            s_mad = float(mad(arr))
            if not np.isfinite(s_mad) or s_mad < self.cfg.eps:
                s_mad = 0.0

            # compute bucket spread (top q - bottom q), used by regime MR
            q = self.cfg.spread_quantile
            lo = float(np.nanquantile(arr, q))
            hi = float(np.nanquantile(arr, 1.0 - q))
            meta[f"spread:{b_id}"] = hi - lo

            # assign z + hysteresis -> weights_raw
            for s in syms:
                if s not in hret:
                    continue
                x = float(hret[s])
                if not np.isfinite(x) or s_mad <= self.cfg.eps:
                    z = 0.0
                else:
                    z = (x - med) / (s_mad + self.cfg.eps)

                zscores[s] = float(z)

                # hysteresis
                st = self.state.leg_state.get(s)
                if st is None:
                    st = _LegState()
                    self.state.leg_state[s] = st

                # rules
                if not st.active:
                    # enter only if strong enough
                    if abs(z) >= self.cfg.z_in:
                        st.active = True
                        st.entered_at = ts
                        st.side = 1 if z < 0 else -1  # contrarian: if z positive => want short, side=-1
                else:
                    # still active: enforce min-hold unless max-hold reached
                    assert st.entered_at is not None
                    held_min = _minutes_between(st.entered_at, ts)

                    if held_min >= self.cfg.max_hold_minutes:
                        st.active = False
                        st.entered_at = None
                        st.side = 0
                    elif held_min >= self.cfg.min_hold_minutes:
                        # allow exit when signal mean-reverted
                        if abs(z) <= self.cfg.z_out:
                            st.active = False
                            st.entered_at = None
                            st.side = 0

                # raw contrarian weight only if active
                if st.active:
                    w = -clip(z, -self.cfg.z_max, self.cfg.z_max)
                else:
                    w = 0.0

                weights_raw[s] = float(w)

        return Signal(ts=ts, weights_raw=weights_raw, zscores=zscores, meta=meta)
