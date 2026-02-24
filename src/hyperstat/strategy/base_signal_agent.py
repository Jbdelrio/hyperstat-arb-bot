# src/hyperstat/strategy/base_signal_agent.py
"""
Abstract base class for all HyperStat signal agents.

Each SignalAgent:
  - Maintains its own internal state (price history, ewma, etc.)
  - Produces a raw weight dict + zscores on every bar update
  - Can be reset() without re-fetching market data
  - Declares warmup_bars (bars needed before first signal)
  - Declares enabled_by_default (shown ON or OFF in GUI on first load)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Context passed to every agent on each bar
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """
    Market data snapshot available to every agent at each bar.
    All data is read-only — agents must not mutate these objects.
    """
    selected: List[str]                       # coins in the basket
    buckets: Dict[str, List[str]]             # bucket grouping {"live": [coins]}
    funding_rates: Dict[str, float]           # latest funding rate per coin (raw, e.g. 0.0001)
    ob_snapshots: Dict[str, Dict]             # order book snapshot per coin (for OB imbalance)
    trade_history: Dict[str, Deque]           # recent trades per coin (for liquidation proxy)


# ─────────────────────────────────────────────────────────────────────────────
# Output produced by every agent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentOutput:
    """
    Raw output from a signal agent — NOT yet normalized or cost-filtered.
    weights: raw target weights, will be cost-filtered then normalized by BackgroundEngine.
    zscores: per-coin scores for dashboard display (can be 0.0 for non-MR strategies).
    meta:    any diagnostic dict (spread, half-life, regime, turnover estimate, etc.)
    """
    weights: Dict[str, float] = field(default_factory=dict)
    zscores: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any]      = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseSignalAgent(ABC):
    """
    Abstract base for all SignalAgents.

    Subclasses MUST set class attributes:
        name: str                  — display name in GUI
        warmup_bars: int           — bars to collect before is_warmed_up == True
        enabled_by_default: bool   — shown as ON/OFF on first GUI load
    """

    name: str = "Base"
    warmup_bars: int = 12
    enabled_by_default: bool = True

    def __init__(self) -> None:
        self._bars_seen: int = 0

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def update(
        self,
        ts: Any,
        mids: Dict[str, float],
        context: AgentContext,
    ) -> AgentOutput:
        """
        Called once per bar. Must:
          1. Update internal state with new mids
          2. Increment _bars_seen
          3. Return AgentOutput (weights={} if not warmed up yet)
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal state to initial (price_history, ewma, etc.).
        Does NOT re-fetch market data — BackgroundEngine keeps streaming.
        Called when user clicks "Restart" in GUI.
        """

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_warmed_up(self) -> bool:
        return self._bars_seen >= self.warmup_bars

    @property
    def bars_seen(self) -> int:
        return self._bars_seen

    # ── Common helper: robust cross-sectional z-score via MAD ─────────────────

    @staticmethod
    def _zscore_cross(values: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
        """
        Compute cross-sectional robust z-score (MAD-based) for a dict of float values.
        Returns a dict with the same keys.
        """
        import numpy as np
        syms = list(values.keys())
        v = np.array([values[s] for s in syms], dtype=float)
        med = float(np.nanmedian(v))
        # MAD scaled to match std under normality
        m = float(np.nanmedian(np.abs(v - med))) * 1.4826
        if m < eps:
            return {s: 0.0 for s in syms}
        return {s: float((values[s] - med) / (m + eps)) for s in syms}

    @staticmethod
    def _ewma_update(prev: float, new_val: float, lam: float) -> float:
        """One-step EWMA: new = lam * prev + (1-lam) * new_val"""
        return lam * prev + (1.0 - lam) * new_val

    @staticmethod
    def _log_return(p_prev: float, p_now: float, eps: float = 1e-12) -> float:
        import math
        if p_prev <= 0 or p_now <= 0:
            return 0.0
        return math.log(p_now / (p_prev + eps))
