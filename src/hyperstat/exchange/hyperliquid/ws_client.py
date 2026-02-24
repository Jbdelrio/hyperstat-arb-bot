from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import websockets

from .endpoints import HyperliquidEndpoints


Json = dict[str, Any]
WSCallback = Callable[[Json], None]


@dataclass(frozen=True)
class WsClientConfig:
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    reconnect_base_s: float = 0.5
    reconnect_max_s: float = 10.0
    max_subscriptions: int = 900  # safety guard below common 1000 limit
    recv_queue_max: int = 4096


class HyperliquidWsClient:
    """
    Single WS connection multiplexing subscriptions.

    - Maintains subscription registry
    - Auto reconnect + resubscribe
    - Dispatch per-subscription callbacks
    """

    def __init__(self, endpoints: HyperliquidEndpoints, cfg: WsClientConfig = WsClientConfig()) -> None:
        self.endpoints = endpoints
        self.cfg = cfg
        self._log = logging.getLogger("hyperstat.ws")

        self._ws: Optional[Any] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

        # key: serialized subscription json
        self._subs: dict[str, tuple[Json, WSCallback]] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        self._task = None

    def subscription_count(self) -> int:
        return len(self._subs)

    async def subscribe(self, subscription: Json, cb: WSCallback) -> str:
        key = json.dumps(subscription, sort_keys=True)
        async with self._lock:
            if key in self._subs:
                self._subs[key] = (subscription, cb)
                return key
            if len(self._subs) >= self.cfg.max_subscriptions:
                raise RuntimeError(f"WS subscription cap reached: {len(self._subs)} >= {self.cfg.max_subscriptions}")
            self._subs[key] = (subscription, cb)

        await self._send({"method": "subscribe", "subscription": subscription})
        return key

    async def unsubscribe(self, subscription: Json) -> None:
        key = json.dumps(subscription, sort_keys=True)
        async with self._lock:
            self._subs.pop(key, None)
        await self._send({"method": "unsubscribe", "subscription": subscription})

    async def _connect(self) -> WebSocketClientProtocol:
        ws = await websockets.connect(
            self.endpoints.ws_url,
            ping_interval=self.cfg.ping_interval_s,
            ping_timeout=self.cfg.ping_timeout_s,
            max_queue=self.cfg.recv_queue_max,
        )
        return ws

    async def _send(self, msg: Json) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps(msg))
        except Exception as e:
            self._log.warning("WS send failed: %s", e)

    async def _resubscribe_all(self) -> None:
        async with self._lock:
            subs = list(self._subs.values())
        for subscription, _cb in subs:
            await self._send({"method": "subscribe", "subscription": subscription})

    async def _dispatch(self, message: Json) -> None:
        # Hyperliquid pushes:
        # {"channel":"subscriptionResponse","data":{...}}
        # {"channel":"allMids","data":{...}}
        ch = message.get("channel")
        if ch in (None, "subscriptionResponse"):
            return

        # Match by "subscription" echo? Not present for all feeds.
        # We'll use "channel" + some fields as best-effort fanout.
        # For deterministic mapping, the strategy should keep only needed feeds per client.
        payload = message.get("data", {})

        async with self._lock:
            items = list(self._subs.values())

        for subscription, cb in items:
            stype = subscription.get("type")
            if stype is None:
                continue

            # Heuristics per feed
            if ch == "allMids" and stype == "allMids":
                cb(message)
            elif ch == "candle" and stype == "candle":
                if payload.get("s") == subscription.get("coin") and payload.get("i") == subscription.get("interval"):
                    cb(message)
            elif ch == "l2Book" and stype == "l2Book":
                if payload.get("coin") == subscription.get("coin"):
                    cb(message)
            elif ch == "trades" and stype == "trades":
                if payload and isinstance(payload, list) and payload[0].get("coin") == subscription.get("coin"):
                    cb(message)
            elif ch == "openOrders" and stype == "openOrders":
                cb(message)
            elif ch == "clearinghouseState" and stype == "clearinghouseState":
                cb(message)
            elif ch == "userFundings" and stype == "userFundings":
                cb(message)
            elif ch == "userEvents" and stype == "userEvents":
                cb(message)

    async def _run(self) -> None:
        backoff = self.cfg.reconnect_base_s
        while not self._stop.is_set():
            try:
                self._log.info("WS connecting %s", self.endpoints.ws_url)
                self._ws = await self._connect()
                self._log.info("WS connected")
                backoff = self.cfg.reconnect_base_s

                await self._resubscribe_all()

                async for raw in self._ws:
                    if self._stop.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                        await self._dispatch(msg)
                    except Exception as e:
                        self._log.warning("WS parse/dispatch error: %s", e)

            except Exception as e:
                self._log.warning("WS error: %s", e)

            # close and reconnect
            try:
                if self._ws is not None:
                    await self._ws.close()
            except Exception:
                pass
            self._ws = None

            if self._stop.is_set():
                break

            sleep_s = min(self.cfg.reconnect_max_s, backoff)
            await asyncio.sleep(sleep_s)
            backoff = min(self.cfg.reconnect_max_s, backoff * 2)
