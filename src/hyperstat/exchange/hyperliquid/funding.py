from __future__ import annotations

from typing import Any, Callable, Optional

from .rest_client import HyperliquidRestClient
from .ws_client import HyperliquidWsClient


Json = dict[str, Any]
WSCallback = Callable[[Json], None]


class HyperliquidFunding:
    def __init__(self, rest: HyperliquidRestClient, ws: Optional[HyperliquidWsClient] = None) -> None:
        self.rest = rest
        self.ws = ws

    async def funding_history(self, coin: str, start_time_ms: int, end_time_ms: int | None = None) -> Json:
        body: Json = {"type": "fundingHistory", "coin": coin, "startTime": start_time_ms}
        if end_time_ms is not None:
            body["endTime"] = end_time_ms
        return await self.rest.info(body, weight=20)

    async def predicted_fundings(self) -> Json:
        body: Json = {"type": "predictedFundings"}
        return await self.rest.info(body, weight=20)

    async def user_funding_ledger(self, user: str, start_time_ms: int, end_time_ms: int | None = None) -> Json:
        # Perps page: type "userFunding" for funding ledger deltas (REST)
        body: Json = {"type": "userFunding", "user": user, "startTime": start_time_ms}
        if end_time_ms is not None:
            body["endTime"] = end_time_ms
        return await self.rest.info(body, weight=20)

    # WS: funding payments stream (snapshot then hourly updates)
    async def ws_subscribe(self, subscription: Json, cb: WSCallback) -> str:
        if self.ws is None:
            raise RuntimeError("WS client not configured")
        await self.ws.start()
        return await self.ws.subscribe(subscription, cb)

    async def stream_user_fundings(self, user: str, cb: WSCallback, dex: str = "") -> str:
        # websocket docs list: WsUserFundings feed (subscription type "userFundings")
        sub: Json = {"type": "userFundings", "user": user}
        if dex != "":
            sub["dex"] = dex
        return await self.ws_subscribe(sub, cb)

    async def stream_user_events(self, user: str, cb: WSCallback) -> str:
        # WS userEvents includes {"funding": WsUserFunding}
        sub: Json = {"type": "userEvents", "user": user}
        return await self.ws_subscribe(sub, cb)
