from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional


Json = dict[str, Any]


@dataclass
class _SandboxState:
    cash_usdc: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)  # coin -> signed size
    avg_px: dict[str, float] = field(default_factory=dict)     # coin -> avg entry
    open_orders: dict[int, Json] = field(default_factory=dict) # oid -> order
    fills: list[Json] = field(default_factory=list)


class HyperliquidSandboxExchange:
    """
    Minimal “fake exchange” for unit tests / dry backtests:
    - Immediate fills at provided mid price (you inject it via set_mid)
    - Keeps positions + fills
    - Supports place_orders + cancel

    This is intentionally simple: it's for testing strategy plumbing & risk.
    """

    def __init__(self, starting_cash_usdc: float = 0.0) -> None:
        self.state = _SandboxState(cash_usdc=starting_cash_usdc)
        self._oid = itertools.count(1)
        self._mids: dict[str, float] = {}

    def set_mid(self, coin: str, mid: float) -> None:
        self._mids[coin] = float(mid)

    def _mid(self, coin: str) -> float:
        if coin not in self._mids:
            raise KeyError(f"Sandbox mid missing for {coin}. Call set_mid().")
        return self._mids[coin]

    async def place_orders(
        self,
        orders: list[Json],
        grouping: str = "na",
        builder: Optional[Json] = None,
        expires_after_ms: Optional[int] = None,
    ) -> Json:
        results = []
        for o in orders:
            oid = next(self._oid)
            coin = str(o["coin"])
            is_buy = bool(o["is_buy"])
            sz = float(o["sz"])
            px = self._mid(coin)  # instant fill at mid

            signed_sz = sz if is_buy else -sz
            prev = self.state.positions.get(coin, 0.0)
            new = prev + signed_sz
            self.state.positions[coin] = new

            # simplistic avg price tracking
            if prev == 0.0:
                self.state.avg_px[coin] = px
            else:
                # weighted by abs size
                prev_avg = self.state.avg_px.get(coin, px)
                w_prev = abs(prev)
                w_new = abs(signed_sz)
                if w_prev + w_new > 0:
                    self.state.avg_px[coin] = (prev_avg * w_prev + px * w_new) / (w_prev + w_new)

            fill = {"coin": coin, "px": px, "sz": sz, "side": "B" if is_buy else "S", "oid": oid}
            self.state.fills.append(fill)

            results.append({"oid": oid, "status": "filled", "fill": fill})

        return {"status": "ok", "results": results}

    async def cancel(self, cancels: list[Json], expires_after_ms: Optional[int] = None) -> Json:
        # In this sandbox, orders are immediately filled => nothing to cancel.
        return {"status": "ok", "canceled": []}

    async def cancel_by_cloid(self, cancels: list[Json], expires_after_ms: Optional[int] = None) -> Json:
        return {"status": "ok", "canceled": []}

    async def update_leverage(self, asset: int, is_cross: bool, leverage: int, expires_after_ms: Optional[int] = None) -> Json:
        return {"status": "ok", "note": "sandbox ignores leverage"}

    async def snapshot(self) -> Json:
        return {
            "cash_usdc": self.state.cash_usdc,
            "positions": dict(self.state.positions),
            "avg_px": dict(self.state.avg_px),
            "fills": list(self.state.fills),
        }
