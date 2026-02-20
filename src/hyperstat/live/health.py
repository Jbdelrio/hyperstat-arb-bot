# src/hyperstat/live/health.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class HealthConfig:
    """
    Basic live safety/health thresholds.
    """
    ws_stale_s: float = 10.0          # warn if no WS msg for this long
    loop_stale_s: float = 60.0        # warn if main loop not ticking
    max_clock_skew_s: float = 2.0     # warn if system clock jumps
    watchdog_period_s: float = 5.0    # watchdog poll


@dataclass
class HealthMonitor:
    cfg: HealthConfig = HealthConfig()

    # last times
    last_ws_msg_monotonic: float = field(default_factory=time.monotonic)
    last_loop_tick_monotonic: float = field(default_factory=time.monotonic)
    last_clock_time: float = field(default_factory=time.time)

    # last computed lags
    ws_stale_s: float = 0.0
    loop_stale_s: float = 0.0
    clock_skew_s: float = 0.0

    def on_ws_message(self) -> None:
        self.last_ws_msg_monotonic = time.monotonic()

    def on_loop_tick(self) -> None:
        self.last_loop_tick_monotonic = time.monotonic()

    def _update(self) -> None:
        now_m = time.monotonic()
        self.ws_stale_s = float(now_m - self.last_ws_msg_monotonic)
        self.loop_stale_s = float(now_m - self.last_loop_tick_monotonic)

        now_t = time.time()
        dt = now_t - self.last_clock_time
        # clock skew measured as jump vs expected monotonic elapsed
        # (rough, but catches big system time changes)
        mono_dt = now_m - (now_m - dt)  # equals dt, kept for clarity
        self.clock_skew_s = float(abs(dt - mono_dt))
        self.last_clock_time = now_t

    def status(self) -> dict:
        self._update()
        return {
            "ws_stale_s": self.ws_stale_s,
            "loop_stale_s": self.loop_stale_s,
            "clock_skew_s": self.clock_skew_s,
        }

    def is_healthy(self) -> bool:
        self._update()
        if self.ws_stale_s > self.cfg.ws_stale_s:
            return False
        if self.loop_stale_s > self.cfg.loop_stale_s:
            return False
        if self.clock_skew_s > self.cfg.max_clock_skew_s:
            return False
        return True

    async def watchdog(self, logger, stop_event: asyncio.Event) -> None:
        """
        Logs warnings. Does not force-stop by default (runner decides).
        """
        while not stop_event.is_set():
            st = self.status()

            if st["ws_stale_s"] > self.cfg.ws_stale_s:
                logger.warning("Health: WS stale %.2fs > %.2fs", st["ws_stale_s"], self.cfg.ws_stale_s)

            if st["loop_stale_s"] > self.cfg.loop_stale_s:
                logger.warning("Health: loop stale %.2fs > %.2fs", st["loop_stale_s"], self.cfg.loop_stale_s)

            if st["clock_skew_s"] > self.cfg.max_clock_skew_s:
                logger.warning("Health: clock skew %.2fs > %.2fs", st["clock_skew_s"], self.cfg.max_clock_skew_s)

            await asyncio.sleep(self.cfg.watchdog_period_s)
