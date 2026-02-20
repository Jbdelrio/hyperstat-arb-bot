# src/hyperstat/core/risk.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from hyperstat.core.math import normalize_to_gross


@dataclass(frozen=True)
class KillSwitchConfig:
    max_intraday_drawdown_pct: float  # e.g. 0.03 => 3%
    cooldown_minutes: int             # disable trading after kill
    z_emergency_flat: float           # flatten leg if |z| exceeds


@dataclass
class RiskState:
    """
    Tracks intraday equity peak and kill-switch cooldown.
    Can be used both in backtest and live.

    Usage:
      risk_state.on_equity(ts, equity)
      if risk_state.trading_allowed(ts): ...
    """
    config: KillSwitchConfig
    day_start: Optional[datetime] = None
    day_peak_equity: Optional[float] = None
    in_cooldown_until: Optional[datetime] = None

    def _day_key(self, ts: datetime) -> Tuple[int, int, int]:
        return ts.year, ts.month, ts.day

    def on_equity(self, ts: datetime, equity: float) -> None:
        # reset at new day
        if self.day_start is None or self._day_key(ts) != self._day_key(self.day_start):
            self.day_start = ts
            self.day_peak_equity = equity

        if self.day_peak_equity is None:
            self.day_peak_equity = equity

        if equity > self.day_peak_equity:
            self.day_peak_equity = equity

        # evaluate drawdown
        peak = self.day_peak_equity
        if peak and peak > 0:
            dd = (peak - equity) / peak
            if dd >= self.config.max_intraday_drawdown_pct:
                self.in_cooldown_until = ts + timedelta(minutes=self.config.cooldown_minutes)

    def trading_allowed(self, ts: datetime) -> bool:
        if self.in_cooldown_until is None:
            return True
        return ts >= self.in_cooldown_until

    def cooldown_remaining_minutes(self, ts: datetime) -> int:
        if self.in_cooldown_until is None:
            return 0
        if ts >= self.in_cooldown_until:
            return 0
        delta = self.in_cooldown_until - ts
        return int(delta.total_seconds() // 60)


def apply_weight_caps(
    weights: Dict[str, float],
    max_weight_per_coin: float,
    gross_target: Optional[float] = None,
) -> Dict[str, float]:
    """
    Clip each weight to +/- max_weight_per_coin.
    Optionally renormalize to a target gross after clipping.
    """
    capped = {s: max(-max_weight_per_coin, min(max_weight_per_coin, float(w))) for s, w in weights.items()}
    if gross_target is not None:
        capped = normalize_to_gross(capped, gross_target=gross_target)
    return capped


def apply_bucket_caps(
    weights: Dict[str, float],
    buckets: Dict[str, list],
    max_weight_per_bucket: float,
) -> Dict[str, float]:
    """
    Ensure bucket gross exposure does not exceed max_weight_per_bucket.
    Bucket gross = sum(abs(w_i) for i in bucket)
    If exceeded, scale down all weights in that bucket proportionally.
    """
    out = dict(weights)
    for _, symbols in buckets.items():
        if not symbols:
            continue
        gross = sum(abs(out.get(s, 0.0)) for s in symbols)
        if gross <= max_weight_per_bucket or gross <= 0:
            continue
        scale = max_weight_per_bucket / gross
        for s in symbols:
            if s in out:
                out[s] = float(out[s] * scale)
    return out


def emergency_flatten_by_z(
    weights: Dict[str, float],
    zscores: Dict[str, float],
    z_emergency_flat: float,
) -> Dict[str, float]:
    """
    If |z| exceeds threshold, force target weight to 0 for that symbol.
    """
    out = dict(weights)
    for s, z in zscores.items():
        if abs(float(z)) >= z_emergency_flat:
            out[s] = 0.0
    return out
