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

# Signal Agents (multi-strategy)
from .base_signal_agent import BaseSignalAgent, AgentContext, AgentOutput
from .momentum import CrossSectionalMomentumAgent, MomentumConfig
from .funding_carry_pure import FundingCarryPureAgent, FundingCarryConfig
from .liquidation_reversion import LiquidationReversionAgent, LiquidationReversionConfig
from .ob_imbalance import OrderFlowImbalanceAgent, OBImbalanceConfig
from .pca_residual_mr import PCAResiduaMRAgent, PCAResiduaMRConfig
from .quality_liquidity import QualityLiquidityAgent, QualityLiquidityConfig

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
    # Signal Agents
    "BaseSignalAgent",
    "AgentContext",
    "AgentOutput",
    "CrossSectionalMomentumAgent",
    "MomentumConfig",
    "FundingCarryPureAgent",
    "FundingCarryConfig",
    "LiquidationReversionAgent",
    "LiquidationReversionConfig",
    "OrderFlowImbalanceAgent",
    "OBImbalanceConfig",
    "PCAResiduaMRAgent",
    "PCAResiduaMRConfig",
    "QualityLiquidityAgent",
    "QualityLiquidityConfig",
]
