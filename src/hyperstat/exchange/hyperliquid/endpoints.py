from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HyperliquidEndpoints:
    http_base: str
    ws_url: str
    info_path: str = "/info"
    exchange_path: str = "/exchange"

    @staticmethod
    def mainnet() -> "HyperliquidEndpoints":
        # Official constants (also used by the python SDK)
        return HyperliquidEndpoints(
            http_base="https://api.hyperliquid.xyz",
            ws_url="wss://api.hyperliquid.xyz/ws",
        )

    @staticmethod
    def testnet() -> "HyperliquidEndpoints":
        return HyperliquidEndpoints(
            http_base="https://api.hyperliquid-testnet.xyz",
            ws_url="wss://api.hyperliquid-testnet.xyz/ws",
        )
