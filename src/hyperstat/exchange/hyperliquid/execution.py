from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from .auth import HyperliquidAuth
from .rest_client import HyperliquidRestClient


Json = dict[str, Any]


def float_to_wire(x: float) -> str:
    """
    Hyperliquid wire format: up to 8 decimals, normalized (no trailing zeros).
    """
    rounded = f"{x:.8f}"
    if abs(float(rounded) - x) >= 1e-12:
        raise ValueError(f"float_to_wire rounding error: {x} -> {rounded}")
    if rounded == "-0":
        rounded = "0"
    normalized = Decimal(rounded).normalize()
    return f"{normalized:f}"


def order_type_to_wire(order_type: Json) -> Json:
    """
    order_type:
      {"limit":{"tif":"Gtc"|"Ioc"|"Alo"}} OR
      {"trigger":{"triggerPx":float,"isMarket":bool,"tpsl":"tp"|"sl"}}
    """
    if "limit" in order_type:
        return {"limit": order_type["limit"]}
    if "trigger" in order_type:
        t = order_type["trigger"]
        return {"trigger": {"triggerPx": float_to_wire(float(t["triggerPx"])), "isMarket": bool(t["isMarket"]), "tpsl": t["tpsl"]}}
    raise ValueError(f"Invalid order_type: {order_type}")


@dataclass
class AssetResolver:
    """
    Maps coin name -> asset index using /info meta universe.
    """
    rest: HyperliquidRestClient
    dex: str = ""

    _coin_to_asset: Optional[dict[str, int]] = None

    async def refresh(self) -> None:
        body: Json = {"type": "meta"}
        if self.dex != "":
            body["dex"] = self.dex
        meta = await self.rest.info(body, weight=20)
        universe = meta.get("universe") or meta.get("universe".lower())
        if universe is None:
            # Some responses are nested; fall back best-effort
            universe = meta.get("universe", [])
        m: dict[str, int] = {}
        for idx, item in enumerate(universe):
            name = item.get("name")
            if name:
                m[str(name)] = idx
        self._coin_to_asset = m

    async def asset(self, coin: str) -> int:
        if self._coin_to_asset is None or coin not in self._coin_to_asset:
            await self.refresh()
        assert self._coin_to_asset is not None
        if coin not in self._coin_to_asset:
            raise KeyError(f"Unknown coin in meta universe: {coin}")
        return self._coin_to_asset[coin]


class HyperliquidExecution:
    def __init__(self, rest: HyperliquidRestClient, auth: HyperliquidAuth, dex: str = "") -> None:
        self.rest = rest
        self.auth = auth
        self.dex = dex
        self.resolver = AssetResolver(rest=rest, dex=dex)

    def _order_wire(self, order: Json, asset: int) -> Json:
        """
        Input order (user-friendly):
          {
            "coin": "ETH",
            "is_buy": True/False,
            "sz": float,
            "limit_px": float,
            "order_type": {"limit":{"tif":"Gtc"}},
            "reduce_only": bool,
            "cloid": Optional[str]
          }

        Output wire:
          {"a":asset,"b":isBuy,"p":str,"s":str,"r":bool,"t":{...},"c"?:cloid}
        """
        w: Json = {
            "a": asset,
            "b": bool(order["is_buy"]),
            "p": float_to_wire(float(order["limit_px"])),
            "s": float_to_wire(float(order["sz"])),
            "r": bool(order.get("reduce_only", False)),
            "t": order_type_to_wire(order["order_type"]),
        }
        cloid = order.get("cloid")
        if cloid:
            w["c"] = str(cloid)
        return w

    async def place_orders(
        self,
        orders: list[Json],
        grouping: str = "na",
        builder: Optional[Json] = None,
        expires_after_ms: Optional[int] = None,
    ) -> Json:
        wires: list[Json] = []
        for o in orders:
            coin = str(o["coin"])
            asset = await self.resolver.asset(coin)
            wires.append(self._order_wire(o, asset))

        action: Json = {"type": "order", "orders": wires, "grouping": grouping}
        if builder is not None:
            action["builder"] = builder

        nonce = self.auth.now_ms()
        sig = self.auth.sign_l1_action(action, nonce, expires_after=expires_after_ms)

        body: Json = {"action": action, "nonce": nonce, "signature": sig}
        if self.auth.vault_address is not None:
            body["vaultAddress"] = self.auth.vault_address
        if expires_after_ms is not None:
            body["expiresAfter"] = expires_after_ms

        return await self.rest.exchange(body, weight=20)

    async def cancel(self, cancels: list[Json], expires_after_ms: Optional[int] = None) -> Json:
        """
        cancels wire format list:
          [{"a": asset, "o": oid}, ...]
        """
        action: Json = {"type": "cancel", "cancels": cancels}
        nonce = self.auth.now_ms()
        sig = self.auth.sign_l1_action(action, nonce, expires_after=expires_after_ms)

        body: Json = {"action": action, "nonce": nonce, "signature": sig}
        if self.auth.vault_address is not None:
            body["vaultAddress"] = self.auth.vault_address
        if expires_after_ms is not None:
            body["expiresAfter"] = expires_after_ms

        return await self.rest.exchange(body, weight=20)

    async def cancel_by_cloid(self, cancels: list[Json], expires_after_ms: Optional[int] = None) -> Json:
        """
        cancels format:
          [{"asset": asset, "cloid": "0x.."}, ...]
        """
        action: Json = {"type": "cancelByCloid", "cancels": cancels}
        nonce = self.auth.now_ms()
        sig = self.auth.sign_l1_action(action, nonce, expires_after=expires_after_ms)

        body: Json = {"action": action, "nonce": nonce, "signature": sig}
        if self.auth.vault_address is not None:
            body["vaultAddress"] = self.auth.vault_address
        if expires_after_ms is not None:
            body["expiresAfter"] = expires_after_ms

        return await self.rest.exchange(body, weight=20)

    async def update_leverage(self, asset: int, is_cross: bool, leverage: int, expires_after_ms: Optional[int] = None) -> Json:
        action: Json = {"type": "updateLeverage", "asset": int(asset), "isCross": bool(is_cross), "leverage": int(leverage)}
        nonce = self.auth.now_ms()
        sig = self.auth.sign_l1_action(action, nonce, expires_after=expires_after_ms)

        body: Json = {"action": action, "nonce": nonce, "signature": sig}
        if self.auth.vault_address is not None:
            body["vaultAddress"] = self.auth.vault_address
        if expires_after_ms is not None:
            body["expiresAfter"] = expires_after_ms

        return await self.rest.exchange(body, weight=20)

    # Helpers (info endpoints useful for execution loop)
    async def open_orders(self, user: str, dex: str = "") -> Json:
        body: Json = {"type": "openOrders", "user": user}
        if dex != "":
            body["dex"] = dex
        return await self.rest.info(body, weight=20)

    async def clearinghouse_state(self, user: str, dex: str = "") -> Json:
        body: Json = {"type": "clearinghouseState", "user": user}
        if dex != "":
            body["dex"] = dex
        return await self.rest.info(body, weight=20)

    async def order_status(self, user: str, oid_or_cloid: Any) -> Json:
        body: Json = {"type": "orderStatus", "user": user, "oid": oid_or_cloid}
        return await self.rest.info(body, weight=20)
