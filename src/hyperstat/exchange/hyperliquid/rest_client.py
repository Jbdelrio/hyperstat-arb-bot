from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from .endpoints import HyperliquidEndpoints
from .rate_limiter import RateLimiter


Json = dict[str, Any]


@dataclass(frozen=True)
class RestClientConfig:
    timeout_s: float = 10.0
    max_retries: int = 5
    base_backoff_s: float = 0.25
    max_backoff_s: float = 3.0
    user_agent: str = "hyperstat/0.1"


class HyperliquidRestClient:
    def __init__(
        self,
        endpoints: HyperliquidEndpoints,
        rate_limiter: RateLimiter,
        cfg: RestClientConfig = RestClientConfig(),
    ) -> None:
        self.endpoints = endpoints
        self.rl = rate_limiter
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=self.endpoints.http_base,
            timeout=httpx.Timeout(cfg.timeout_s),
            headers={"Content-Type": "application/json", "User-Agent": cfg.user_agent},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def post(self, path: str, body: Json, weight: int = 20) -> Json:
        """
        Generic POST with:
        - async rate limit acquire
        - retry/backoff on 429 / 5xx / network errors
        """
        await self.rl.acquire(weight)

        attempt = 0
        backoff = self.cfg.base_backoff_s
        last_exc: Optional[Exception] = None

        while attempt <= self.cfg.max_retries:
            try:
                resp = await self._client.post(path, json=body)
                if resp.status_code == 200:
                    return resp.json()

                # Retryable statuses
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text}",
                        request=resp.request,
                        response=resp,
                    )

                # Non-retryable
                resp.raise_for_status()
                return resp.json()

            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_exc = e
                attempt += 1
                if attempt > self.cfg.max_retries:
                    break

                # jittered exponential backoff
                sleep_s = min(self.cfg.max_backoff_s, backoff) * (0.7 + 0.6 * random.random())
                await asyncio.sleep(sleep_s)
                backoff *= 2

        assert last_exc is not None
        raise last_exc

    async def info(self, body: Json, weight: int = 20) -> Json:
        return await self.post(self.endpoints.info_path, body, weight=weight)

    async def exchange(self, body: Json, weight: int = 20) -> Json:
        return await self.post(self.endpoints.exchange_path, body, weight=weight)
