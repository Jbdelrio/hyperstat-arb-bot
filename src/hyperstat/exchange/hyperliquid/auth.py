from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import msgpack
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak, to_hex


Json = dict[str, Any]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _address_to_bytes(address: str) -> bytes:
    a = address.lower()
    if a.startswith("0x"):
        a = a[2:]
    return bytes.fromhex(a)


def _construct_phantom_agent(hash_bytes32: bytes, is_mainnet: bool) -> Json:
    # Mirrors hyperliquid SDK behavior: "a" for mainnet, "b" for testnet.
    return {"source": "a" if is_mainnet else "b", "connectionId": hash_bytes32}


def _l1_payload(phantom_agent: Json) -> Json:
    # EIP712 payload for L1 actions
    return {
        "domain": {
            "chainId": 1337,
            "name": "Exchange",
            "verifyingContract": "0x0000000000000000000000000000000000000000",
            "version": "1",
        },
        "types": {
            "Agent": [
                {"name": "source", "type": "string"},
                {"name": "connectionId", "type": "bytes32"},
            ],
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
        },
        "primaryType": "Agent",
        "message": phantom_agent,
    }


def _action_hash(action: Any, vault_address: Optional[str], nonce: int, expires_after: Optional[int]) -> bytes:
    """
    Hashing scheme (msgpack(action) + nonce + vault flag + vault bytes + expires flag + expires).
    Matches hyperliquid SDK logic.
    """
    data = msgpack.packb(action)
    data += nonce.to_bytes(8, "big")

    if vault_address is None:
        data += b"\x00"
    else:
        data += b"\x01"
        data += _address_to_bytes(vault_address)

    if expires_after is not None:
        data += b"\x00"
        data += expires_after.to_bytes(8, "big")

    return keccak(data)


def _sign_inner(wallet: Account, typed_data: Json) -> Json:
    structured = encode_typed_data(full_message=typed_data)
    signed = wallet.sign_message(structured)
    return {"r": to_hex(signed["r"]), "s": to_hex(signed["s"]), "v": signed["v"]}


@dataclass
class HyperliquidAuth:
    private_key: str
    is_mainnet: bool = True
    account_address: Optional[str] = None
    vault_address: Optional[str] = None

    def __post_init__(self) -> None:
        self._acct = Account.from_key(self.private_key)
        if self.account_address is None:
            self.account_address = self._acct.address
        if self.vault_address is not None:
            self.vault_address = self.vault_address.lower()

    @property
    def wallet_address(self) -> str:
        assert self.account_address is not None
        return self.account_address.lower()

    def now_ms(self) -> int:
        return _now_ms()

    def sign_l1_action(self, action: Any, nonce: int, expires_after: Optional[int] = None) -> Json:
        """
        Signs an L1 action for /exchange using the agent EIP712 scheme.
        """
        h = _action_hash(action, self.vault_address, nonce, expires_after)
        phantom = _construct_phantom_agent(h, self.is_mainnet)
        payload = _l1_payload(phantom)
        return _sign_inner(self._acct, payload)
