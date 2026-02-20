from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimiterConfig:
    weight_budget_per_minute: int = 1200  # conservative default
    hard_floor_remaining: int = 0         # allow 0 => can fully spend


class RateLimiter:
    """
    Token bucket rate limiter in "weight units".

    - capacity = budget per minute
    - refill continuously
    - acquire(weight) blocks until enough tokens
    """

    def __init__(self, cfg: RateLimiterConfig = RateLimiterConfig()) -> None:
        self.cfg = cfg
        self.capacity = float(cfg.weight_budget_per_minute)
        self.tokens = float(cfg.weight_budget_per_minute)
        self.refill_per_sec = float(cfg.weight_budget_per_minute) / 60.0
        self.last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        dt = now - self.last
        self.last = now
        self.tokens = min(self.capacity, self.tokens + dt * self.refill_per_sec)

    async def acquire(self, weight: int) -> None:
        w = float(max(0, weight))
        while True:
            async with self._lock:
                self._refill()
                remaining_after = self.tokens - w
                if remaining_after >= self.cfg.hard_floor_remaining:
                    self.tokens = remaining_after
                    return

                # compute wait needed
                need = w - self.tokens
                wait_s = max(0.0, need / self.refill_per_sec)

            await asyncio.sleep(min(1.0, wait_s))
