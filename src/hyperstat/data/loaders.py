# src/hyperstat/data/loaders.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .storage import DataStore, _to_utc_datetime_index


def _parse_timeframe(tf: str) -> str:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return tf[:-1] + "min"
    if tf.endswith("h"):
        return tf[:-1] + "h"
    return tf


def load_candles_csv_dir(
    root_dir: str,
    symbols: list[str],
    timeframe: str,
    file_pattern: str = "{symbol}_{timeframe}.csv",
) -> Dict[str, pd.DataFrame]:
    """
    Loads candles from CSV files like:
      <root_dir>/<symbol>_<timeframe>.csv
    Default timeframe string matches config (e.g. '5m').
    """
    root = Path(root_dir)
    out: Dict[str, pd.DataFrame] = {}

    for s in symbols:
        p = root / file_pattern.format(symbol=s, timeframe=timeframe)
        if not p.exists():
            out[s] = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
            continue
        df = pd.read_csv(p)
        df = _to_utc_datetime_index(df, "ts")
        out[s] = df[["ts", "open", "high", "low", "close", "volume"]].copy()

    return out


def load_funding_csv_dir(
    root_dir: str,
    symbols: list[str],
    file_pattern: str = "{symbol}_funding.csv",
) -> Dict[str, pd.DataFrame]:
    """
    Loads funding from CSV files like:
      <root_dir>/<symbol>_funding.csv
    """
    root = Path(root_dir)
    out: Dict[str, pd.DataFrame] = {}

    for s in symbols:
        p = root / file_pattern.format(symbol=s)
        if not p.exists():
            out[s] = pd.DataFrame(columns=["ts", "rate"])
            continue
        df = pd.read_csv(p)
        df = _to_utc_datetime_index(df, "ts")
        out[s] = df[["ts", "rate"]].copy()

    return out


def load_candles_from_store(
    store: DataStore,
    symbols: list[str],
    timeframe: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        out[s] = store.load_candles(s, timeframe, start=start, end=end)
    return out


def load_funding_from_store(
    store: DataStore,
    symbols: list[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        out[s] = store.load_funding(s, start=start, end=end)
    return out
