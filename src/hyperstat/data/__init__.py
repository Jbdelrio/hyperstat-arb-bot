# src/hyperstat/data/__init__.py
from __future__ import annotations

from .storage import DataStore, ParquetStore, DuckDBStore, SQLiteStore, store_from_config
from .loaders import (
    load_candles_csv_dir,
    load_funding_csv_dir,
    load_candles_from_store,
    load_funding_from_store,
)
from .features import (
    compute_returns,
    compute_ewma_vol,
    compute_rv,
    compute_rv_1h_pct,
    compute_amihud_illiq,
    compute_beta_vs_factor,
    compute_residual_returns,
    compute_funding_ewma_stats,
)
from .universe import (
    select_universe,
    build_buckets,
)

__all__ = [
    "DataStore",
    "ParquetStore",
    "DuckDBStore",
    "SQLiteStore",
    "store_from_config",
    "load_candles_csv_dir",
    "load_funding_csv_dir",
    "load_candles_from_store",
    "load_funding_from_store",
    "compute_returns",
    "compute_ewma_vol",
    "compute_rv",
    "compute_rv_1h_pct",
    "compute_amihud_illiq",
    "compute_beta_vs_factor",
    "compute_residual_returns",
    "compute_funding_ewma_stats",
    "select_universe",
    "build_buckets",
]
