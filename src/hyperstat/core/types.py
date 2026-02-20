# src/hyperstat/core/types.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Market data
# ----------------------------

@dataclass(frozen=True)
class Candle:
    symbol: str
    ts: datetime  # UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FundingRate:
    symbol: str
    ts: datetime  # funding timestamp UTC
    rate: float   # e.g. 0.0001 = 1 bp


# ----------------------------
# Execution types
# ----------------------------

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    qty: float                 # base quantity
    order_type: OrderType
    limit_px: Optional[float] = None
    tif: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    client_order_id: Optional[str] = None


@dataclass(frozen=True)
class OrderAck:
    symbol: str
    client_order_id: Optional[str]
    exchange_order_id: Optional[str]
    accepted: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class Fill:
    symbol: str
    ts: datetime
    side: Side
    qty: float
    px: float
    fee_paid: float            # quote currency
    order_id: Optional[str] = None


@dataclass
class Position:
    symbol: str
    qty: float                 # signed: >0 long, <0 short
    avg_px: float
    unrealized_pnl: float = 0.0


@dataclass
class AccountState:
    ts: datetime
    equity: float              # quote currency (e.g. USDC)
    margin_used: float
    positions: Dict[str, Position]


# ----------------------------
# Strategy / portfolio objects
# ----------------------------

@dataclass(frozen=True)
class Universe:
    symbols: List[str]
    asof: datetime


@dataclass(frozen=True)
class Buckets:
    # bucket_id -> list of symbols
    mapping: Dict[str, List[str]]
    asof: datetime


@dataclass(frozen=True)
class FeaturesFrame:
    ts: datetime
    values: Dict[str, Dict[str, float]]  # symbol -> {feature: value}


@dataclass(frozen=True)
class Signal:
    ts: datetime
    weights_raw: Dict[str, float]        # symbol -> raw score
    zscores: Dict[str, float]            # symbol -> z
    meta: Dict[str, float]


@dataclass(frozen=True)
class RegimeScore:
    ts: datetime
    q_mr: float
    q_liq: float
    q_risk: float

    @property
    def q_total(self) -> float:
        q = self.q_mr * self.q_liq * self.q_risk
        return max(0.0, min(1.0, float(q)))


@dataclass(frozen=True)
class PortfolioWeights:
    ts: datetime
    weights: Dict[str, float]            # target weights
    gross: float                         # sum(abs(w))
    net: float                           # sum(w)
    beta: float                          # sum(w * beta), if available
    meta: Dict[str, float]


@dataclass
class BacktestResult:
    start: datetime
    end: datetime
    equity_curve: List[Tuple[datetime, float]]
    pnl_curve: List[Tuple[datetime, float]]
    turnover_curve: List[Tuple[datetime, float]]
    breakdown: Dict[str, float]
    meta: Dict[str, float]