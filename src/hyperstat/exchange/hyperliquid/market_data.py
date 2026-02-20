from __future__ import annotations

from typing import Any, Callable, Optional

from .rest_client import HyperliquidRestClient
from .ws_client import HyperliquidWsClient


Json = dict[str, Any]
WSCallback = Callable[[Json], None]


class HyperliquidMarketData:
    def __init__(self, rest: HyperliquidRestClient, ws: Optional[HyperliquidWsClient] = None) -> None:
        self.rest = rest
        self.ws = ws

    # -------- REST snapshots --------

    async def all_mids(self, dex: str = "") -> Json:
        body: Json = {"type": "allMids"}
        if dex != "":
            body["dex"] = dex
        return await self.rest.info(body, weight=20)

    async def l2_book(self, coin: str, n_sig_figs: int | None = None, mantissa: int | None = None) -> Json:
        body: Json = {"type": "l2Book", "coin": coin}
        if n_sig_figs is not None:
            body["nSigFigs"] = n_sig_figs
        if mantissa is not None:
            body["mantissa"] = mantissa
        return await self.rest.info(body, weight=20)

    async def candle_snapshot(self, coin: str, interval: str, start_time_ms: int, end_time_ms: int) -> Json:
        # docs: {"type":"candleSnapshot","req":{"coin":..., "interval":..., "startTime":..., "endTime":...}}
        body: Json = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval, "startTime": start_time_ms, "endTime": end_time_ms},
        }
        return await self.rest.info(body, weight=20)

    async def meta(self, dex: str = "") -> Json:
        body: Json = {"type": "meta"}
        if dex != "":
            body["dex"] = dex
        return await self.rest.info(body, weight=20)

    async def meta_and_asset_ctxs(self, dex: str = "") -> Json:
        body: Json = {"type": "metaAndAssetCtxs"}
        if dex != "":
            body["dex"] = dex
        return await self.rest.info(body, weight=20)

    # -------- WebSocket streaming --------

    async def ws_subscribe(self, subscription: Json, cb: WSCallback) -> str:
        if self.ws is None:
            raise RuntimeError("WS client not configured")
        await self.ws.start()
        return await self.ws.subscribe(subscription, cb)

    async def ws_unsubscribe(self, subscription: Json) -> None:
        if self.ws is None:
            return
        await self.ws.unsubscribe(subscription)

    # Convenience wrappers

    async def stream_all_mids(self, cb: WSCallback, dex: str = "") -> str:
        sub: Json = {"type": "allMids"}
        if dex != "":
            sub["dex"] = dex
        return await self.ws_subscribe(sub, cb)

    async def stream_candles(self, coin: str, interval: str, cb: WSCallback) -> str:
        # websocket docs: {type:"candle", coin:"SOL", interval:"1m"}
        sub: Json = {"type": "candle", "coin": coin, "interval": interval}
        return await self.ws_subscribe(sub, cb)

    async def stream_l2_book(self, coin: str, n_sig_figs: int = 5, n_levels: int = 20, cb: WSCallback | None = None) -> str:
        if cb is None:
            raise ValueError("cb is required")
        sub: Json = {"type": "l2Book", "coin": coin, "nSigFigs": n_sig_figs, "nLevels": n_levels}
        return await self.ws_subscribe(sub, cb)

    async def stream_trades(self, coin: str, cb: WSCallback) -> str:
        sub: Json = {"type": "trades", "coin": coin}
        return await self.ws_subscribe(sub, cb)
