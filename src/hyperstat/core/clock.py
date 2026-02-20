# src/hyperstat/core/clock.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional


class Clock:
    """Abstract clock used across live/backtest."""

    def now(self) -> datetime:
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        """Live clock may implement; backtest clock is a no-op."""
        return


@dataclass
class LiveClock(Clock):
    """UTC live clock."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass
class BacktestClock(Clock):
    """
    Backtest clock: iterates over a predefined timestamp sequence.
    You feed it the timeline from your dataset index.
    """

    timeline: Iterable[datetime]
    _current: Optional[datetime] = None

    def iter(self) -> Iterator[datetime]:
        for ts in self.timeline:
            self._current = ts
            yield ts

    def now(self) -> datetime:
        if self._current is None:
            raise RuntimeError("BacktestClock.now() called before iteration started.")
        return self._current
