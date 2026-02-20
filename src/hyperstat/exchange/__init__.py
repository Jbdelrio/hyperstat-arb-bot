from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable


Json = dict[str, Any]
WSCallback = Callable[[Json], None]


@runtime_checkable
class MarketDataAPI(Protocol):
    async def all_mids(self, dex: str = "") -> Json: ...
    async def l2_book(self, coin: str, n_sig_figs: int | None = None, mantissa: int | None = None) -> Json: ...
    async def candle_snapshot(
        self,
        coin: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> Json: ...

    # Streaming helpers
    async def ws_subscribe(self, subscription: Json, cb: WSCallback) -> str: ...
    async def ws_unsubscribe(self, subscription: Json) -> None: ...


@runtime_checkable
class FundingAPI(Protocol):
    async def funding_history(self, coin: str, start_time_ms: int, end_time_ms: int | None = None) -> Json: ...
    async def predicted_fundings(self) -> Json: ...
    async def user_funding_ledger(self, user: str, start_time_ms: int, end_time_ms: int | None = None) -> Json: ...


@runtime_checkable
class ExecutionAPI(Protocol):
    async def place_orders(
        self,
        orders: list[Json],
        grouping: str = "na",
        builder: Optional[Json] = None,
        expires_after_ms: Optional[int] = None,
    ) -> Json: ...

    async def cancel(self, cancels: list[Json], expires_after_ms: Optional[int] = None) -> Json: ...
    async def cancel_by_cloid(self, cancels: list[Json], expires_after_ms: Optional[int] = None) -> Json: ...

    async def update_leverage(self, asset: int, is_cross: bool, leverage: int, expires_after_ms: Optional[int] = None) -> Json: ...


@dataclass(frozen=True)
class ExchangeBundle:
    market_data: MarketDataAPI
    funding: FundingAPI
    execution: ExecutionAPI
