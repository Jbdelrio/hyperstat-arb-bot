# src/hyperstat/live/order_manager.py
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hyperstat.core.logging import get_logger
from hyperstat.core.types import Side


Json = dict[str, Any]


@dataclass(frozen=True)
class OrderManagerConfig:
    """
    Execution policy for live/paper.

    - taker-like execution via aggressive IOC limit orders.
    - reconcile open orders at startup and periodically.
    """
    execution_enabled: bool = False        # paper=false, live=true
    quote_ccy: str = "USDC"

    # Do not trade if delta_notional is too small (avoid fee churn)
    min_trade_notional: float = 5.0

    # Convert “market order” to aggressive IOC limit:
    # buy limit = mid * (1 + aggress_bps/1e4), sell limit = mid * (1 - aggress_bps/1e4)
    aggress_bps: float = 15.0

    # Safety caps per rebalance step
    max_trade_notional_per_symbol: float = 250.0
    max_total_trade_notional: float = 800.0

    # Startup / reconciliation
    cancel_all_on_start: bool = True
    reconcile_every_s: float = 30.0

    # Idempotence
    session_salt: Optional[str] = None   # if None -> generated at runtime


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    qty: float
    limit_px: float
    reduce_only: bool
    cloid: str
    notional: float


@dataclass(frozen=True)
class PositionSnapshot:
    """
    Minimal normalized position snapshot for sizing.
    qty is signed (base units).
    """
    qty: float
    entry_px: float = 0.0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cloid(session_salt: str, ts_ms: int, symbol: str, side: str) -> str:
    """
    Deterministic client order id:
      0x + 32 bytes hex (64 chars)
    """
    payload = f"{session_salt}:{ts_ms}:{symbol}:{side}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return "0x" + h[:64]


class OrderManager:
    """
    Responsibilities:
      - Pull positions/equity from /info (clearinghouseState)
      - Pull open orders from /info (openOrders)
      - Cancel stale orders
      - Translate target weights into orders
      - Provide idempotent CLOIDs
    """

    def __init__(
        self,
        cfg: OrderManagerConfig,
        execution_api,   # HyperliquidExecution (or compatible)
    ) -> None:
        self.cfg = cfg
        self.exec = execution_api
        self.log = get_logger("hyperstat.order_manager")

        self.session_salt = cfg.session_salt or hashlib.sha256(str(_now_ms()).encode()).hexdigest()[:16]
        self._last_reconcile_ms: int = 0

        # cached state
        self.positions: Dict[str, PositionSnapshot] = {}
        self.equity: float = 0.0
        self.open_orders: List[Json] = []

    # -------------------------
    # Info parsing (best-effort)
    # -------------------------
    @staticmethod
    def _parse_equity(ch_state: Json) -> float:
        """
        Hyperliquid clearinghouseState is nested; we do best-effort extraction.
        Common field: marginSummary.accountValue (string)
        """
        try:
            ms = ch_state.get("marginSummary") or {}
            av = ms.get("accountValue")
            if av is not None:
                return float(av)
        except Exception:
            pass

        # fallback: search common places
        for k in ("accountValue", "equity", "totalEquity"):
            if k in ch_state:
                try:
                    return float(ch_state[k])
                except Exception:
                    continue
        return float("nan")

    @staticmethod
    def _parse_positions(ch_state: Json) -> Dict[str, PositionSnapshot]:
        """
        Common structure:
          ch_state["assetPositions"] = [{"position":{"coin":"ETH","szi":"0.5","entryPx":"3000"}} ...]
        """
        out: Dict[str, PositionSnapshot] = {}
        aps = ch_state.get("assetPositions") or []
        if not isinstance(aps, list):
            return out

        for item in aps:
            try:
                pos = item.get("position") or item.get("pos") or item
                coin = pos.get("coin") or pos.get("symbol")
                if not coin:
                    continue
                # size may be "szi" (string) or "sz"
                szi = pos.get("szi", pos.get("sz", pos.get("size", 0.0)))
                qty = float(szi)
                entry = float(pos.get("entryPx", pos.get("entry", 0.0)) or 0.0)
                out[str(coin)] = PositionSnapshot(qty=qty, entry_px=entry)
            except Exception:
                continue
        return out

    @staticmethod
    def _parse_open_orders(resp: Json) -> List[Json]:
        """
        openOrders response is typically a list.
        """
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict):
            # sometimes nested
            for k in ("orders", "openOrders", "data"):
                v = resp.get(k)
                if isinstance(v, list):
                    return v
        return []

    # -------------------------
    # Reconcile / refresh
    # -------------------------
    async def refresh_state(self, user: str) -> None:
        ch = await self.exec.clearinghouse_state(user=user)
        self.equity = float(self._parse_equity(ch))
        self.positions = self._parse_positions(ch)

        oo = await self.exec.open_orders(user=user)
        self.open_orders = self._parse_open_orders(oo)

    async def cancel_all_open_orders(self) -> None:
        """
        Cancel all open orders (best effort). Requires asset + oid.
        """
        if not self.open_orders:
            return

        cancels: List[Json] = []
        for o in self.open_orders:
            try:
                coin = o.get("coin") or o.get("symbol")
                oid = o.get("oid") or o.get("orderId") or o.get("id")
                if coin is None or oid is None:
                    continue
                asset = await self.exec.resolver.asset(str(coin))
                cancels.append({"a": int(asset), "o": int(oid)})
            except Exception:
                continue

        if not cancels:
            return

        if not self.cfg.execution_enabled:
            self.log.info("[paper] would cancel %d open orders", len(cancels))
            return

        res = await self.exec.cancel(cancels)
        self.log.info("Canceled open orders: %s", res)

    async def reconcile(self, user: str, force: bool = False) -> None:
        now = _now_ms()
        if not force and (now - self._last_reconcile_ms) < int(self.cfg.reconcile_every_s * 1000):
            return
        self._last_reconcile_ms = now

        await self.refresh_state(user=user)

    # -------------------------
    # Target -> intents -> orders
    # -------------------------
    def _target_notional(self, weight: float) -> float:
        return float(weight * self.equity)

    def _current_notional(self, symbol: str, mid: float) -> float:
        p = self.positions.get(symbol)
        if p is None:
            return 0.0
        return float(p.qty * mid)

    def build_intents(
        self,
        user: str,
        ts_ms: int,
        target_weights: Dict[str, float],
        mids: Dict[str, float],
    ) -> List[OrderIntent]:
        """
        Pure function: decides what to trade (no API calls).
        """
        if not (self.equity > 0.0):
            self.log.warning("Equity not available; skipping intents.")
            return []

        intents: List[OrderIntent] = []
        total_trade = 0.0

        for sym, w_tgt in target_weights.items():
            mid = float(mids.get(sym, 0.0))
            if mid <= 0.0:
                continue

            tgt = self._target_notional(float(w_tgt))
            cur = self._current_notional(sym, mid)
            delta = float(tgt - cur)

            if abs(delta) < self.cfg.min_trade_notional:
                continue

            # safety caps
            delta = max(-self.cfg.max_trade_notional_per_symbol, min(self.cfg.max_trade_notional_per_symbol, delta))
            if abs(delta) < self.cfg.min_trade_notional:
                continue

            remaining = self.cfg.max_total_trade_notional - total_trade
            if remaining <= 0:
                break
            if abs(delta) > remaining:
                delta = float(np.sign(delta) * remaining)

            side = Side.BUY if delta > 0 else Side.SELL
            notional = abs(delta)
            qty = float(notional / mid)

            # aggressive IOC limit
            bps = self.cfg.aggress_bps / 1e4
            if side == Side.BUY:
                limit_px = float(mid * (1.0 + bps))
            else:
                limit_px = float(mid * (1.0 - bps))

            reduce_only = False  # can be improved later with position-aware reduce-only logic
            cl = _cloid(self.session_salt, ts_ms, sym, side.value)

            intents.append(
                OrderIntent(
                    symbol=sym,
                    side=side,
                    qty=qty,
                    limit_px=limit_px,
                    reduce_only=reduce_only,
                    cloid=cl,
                    notional=notional,
                )
            )
            total_trade += notional

        return intents

    async def execute_intents(self, intents: List[OrderIntent]) -> Json:
        """
        Sends a batch order action to HyperliquidExecution.
        """
        if not intents:
            return {"status": "noop", "results": []}

        orders: List[Json] = []
        for it in intents:
            orders.append(
                {
                    "coin": it.symbol,
                    "is_buy": it.side == Side.BUY,
                    "sz": float(it.qty),
                    "limit_px": float(it.limit_px),
                    "order_type": {"limit": {"tif": "Ioc"}},  # taker-like IOC
                    "reduce_only": bool(it.reduce_only),
                    "cloid": it.cloid,
                }
            )

        if not self.cfg.execution_enabled:
            self.log.info("[paper] would place %d orders (total notional≈%.2f)", len(intents), sum(i.notional for i in intents))
            return {"status": "paper", "orders": orders}

        res = await self.exec.place_orders(orders=orders, grouping="na")
        self.log.info("place_orders: %s", res)
        return res
