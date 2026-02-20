# src/hyperstat/core/logging.py
from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logging with rich handler.
    Call once at program start (main CLI).
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)],
    )

    # Reduce noisy loggers by default
    for noisy in ("httpx", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(max(lvl, logging.WARNING))


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
