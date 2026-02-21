# src/hyperstat/strategy/__init__.py
from __future__ import annotations

from .stat_arb import StatArbConfig, StatArbState, StatArbStrategy
from .regime import RegimeConfig, RegimeModel
from .funding_overlay import FundingOverlayConfig, FundingOverlayModel
from .funding_divergence_signal import (
    FDSConfig,
    FundingDivergenceSignal,
    FundingDivergenceSignalLive,
    FDSDiagnostics,
)
from .allocator import AllocatorConfig, PortfolioAllocator

__all__ = [
    "StatArbConfig",
    "StatArbState",
    "StatArbStrategy",
    "RegimeConfig",
    "RegimeModel",
    "FundingOverlayConfig",
    "FundingOverlayModel",
    "FDSConfig",
    "FundingDivergenceSignal",
    "FundingDivergenceSignalLive",
    "FDSDiagnostics",
    "AllocatorConfig",
    "PortfolioAllocator",
]
