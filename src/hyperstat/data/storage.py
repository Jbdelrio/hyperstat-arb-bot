# src/hyperstat/data/storage.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_utc_datetime_index(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """
    Ensures df[ts_col] is datetime64[ns, UTC] and sorted.
    """
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(ts_col)
    return out


def _filter_time(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp], ts_col: str = "ts"):
    if df.empty:
        return df
    if start is not None:
        df = df[df[ts_col] >= start]
    if end is not None:
        df = df[df[ts_col] <= end]
    return df


class DataStore(Protocol):
    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None: ...
    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame: ...

    def save_funding(self, symbol: str, df: pd.DataFrame) -> None: ...
    def load_funding(
        self,
        symbol: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ParquetStore:
    """
    Layout:
      root/
        candles/<timeframe>/<symbol>.parquet
        funding/<symbol>.parquet

    Candle schema expected:
      ts, open, high, low, close, volume
    Funding schema expected:
      ts, rate
    """
    root_dir: str

    def _candles_path(self, symbol: str, timeframe: str) -> Path:
        return Path(self.root_dir) / "candles" / timeframe / f"{symbol}.parquet"

    def _funding_path(self, symbol: str) -> Path:
        return Path(self.root_dir) / "funding" / f"{symbol}.parquet"

    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        p = self._candles_path(symbol, timeframe)
        _ensure_dir(p.parent)
        df = _to_utc_datetime_index(df, "ts")
        _require_cols(df, ["ts", "open", "high", "low", "close", "volume"])
        _write_parquet(df, p)

    def load_candles(
        self, symbol: str, timeframe: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        p = self._candles_path(symbol, timeframe)
        if not p.exists():
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        df = _read_parquet(p)
        df = _to_utc_datetime_index(df, "ts")
        return _filter_time(df, start, end, "ts")

    def save_funding(self, symbol: str, df: pd.DataFrame) -> None:
        p = self._funding_path(symbol)
        _ensure_dir(p.parent)
        df = _to_utc_datetime_index(df, "ts")
        _require_cols(df, ["ts", "rate"])
        _write_parquet(df, p)

    def load_funding(self, symbol: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        p = self._funding_path(symbol)
        if not p.exists():
            return pd.DataFrame(columns=["ts", "rate"])
        df = _read_parquet(p)
        df = _to_utc_datetime_index(df, "ts")
        return _filter_time(df, start, end, "ts")


@dataclass(frozen=True)
class DuckDBStore:
    """
    Optional store. Requires `duckdb` installed.
    Tables:
      candles(symbol, timeframe, ts, open, high, low, close, volume)
      funding(symbol, ts, rate)
    """
    db_path: str

    def __post_init__(self) -> None:
        _ensure_dir(Path(self.db_path).parent)

    def _connect(self):
        try:
            import duckdb  # type: ignore
        except Exception as e:
            raise RuntimeError("DuckDBStore requires `duckdb`. Install: pip install duckdb") from e
        return duckdb.connect(self.db_path)

    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        df = _to_utc_datetime_index(df, "ts")
        _require_cols(df, ["ts", "open", "high", "low", "close", "volume"])
        df = df.assign(symbol=symbol, timeframe=timeframe)
        con = self._connect()
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS candles(symbol VARCHAR, timeframe VARCHAR, ts TIMESTAMP, "
                "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE)"
            )
            con.register("df", df)
            con.execute("INSERT INTO candles SELECT symbol, timeframe, ts, open, high, low, close, volume FROM df")
        finally:
            con.close()

    def load_candles(self, symbol: str, timeframe: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        con = self._connect()
        try:
            q = "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=?"
            params: list[Any] = [symbol, timeframe]
            if start is not None:
                q += " AND ts >= ?"
                params.append(start.to_pydatetime())
            if end is not None:
                q += " AND ts <= ?"
                params.append(end.to_pydatetime())
            q += " ORDER BY ts"
            df = con.execute(q, params).fetchdf()
        finally:
            con.close()
        return _to_utc_datetime_index(df, "ts")

    def save_funding(self, symbol: str, df: pd.DataFrame) -> None:
        df = _to_utc_datetime_index(df, "ts")
        _require_cols(df, ["ts", "rate"])
        df = df.assign(symbol=symbol)
        con = self._connect()
        try:
            con.execute("CREATE TABLE IF NOT EXISTS funding(symbol VARCHAR, ts TIMESTAMP, rate DOUBLE)")
            con.register("df", df)
            con.execute("INSERT INTO funding SELECT symbol, ts, rate FROM df")
        finally:
            con.close()

    def load_funding(self, symbol: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        con = self._connect()
        try:
            q = "SELECT ts, rate FROM funding WHERE symbol=?"
            params: list[Any] = [symbol]
            if start is not None:
                q += " AND ts >= ?"
                params.append(start.to_pydatetime())
            if end is not None:
                q += " AND ts <= ?"
                params.append(end.to_pydatetime())
            q += " ORDER BY ts"
            df = con.execute(q, params).fetchdf()
        finally:
            con.close()
        return _to_utc_datetime_index(df, "ts")


@dataclass(frozen=True)
class SQLiteStore:
    """
    Optional store. Uses sqlite3. Slower but ubiquitous.
    """
    db_path: str

    def __post_init__(self) -> None:
        _ensure_dir(Path(self.db_path).parent)

    def _connect(self):
        import sqlite3
        con = sqlite3.connect(self.db_path)
        return con

    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        df = _to_utc_datetime_index(df, "ts")
        _require_cols(df, ["ts", "open", "high", "low", "close", "volume"])
        df = df.assign(symbol=symbol, timeframe=timeframe)
        con = self._connect()
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS candles("
                "symbol TEXT, timeframe TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)"
            )
            df2 = df.copy()
            df2["ts"] = df2["ts"].astype(str)
            df2.to_sql("candles", con, if_exists="append", index=False)
        finally:
            con.close()

    def load_candles(self, symbol: str, timeframe: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        con = self._connect()
        try:
            q = "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=?"
            params: list[Any] = [symbol, timeframe]
            if start is not None:
                q += " AND ts >= ?"
                params.append(str(start))
            if end is not None:
                q += " AND ts <= ?"
                params.append(str(end))
            q += " ORDER BY ts"
            df = pd.read_sql_query(q, con, params=params)
        finally:
            con.close()
        return _to_utc_datetime_index(df, "ts")

    def save_funding(self, symbol: str, df: pd.DataFrame) -> None:
        df = _to_utc_datetime_index(df, "ts")
        _require_cols(df, ["ts", "rate"])
        df = df.assign(symbol=symbol)
        con = self._connect()
        try:
            con.execute("CREATE TABLE IF NOT EXISTS funding(symbol TEXT, ts TEXT, rate REAL)")
            df2 = df.copy()
            df2["ts"] = df2["ts"].astype(str)
            df2.to_sql("funding", con, if_exists="append", index=False)
        finally:
            con.close()

    def load_funding(self, symbol: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        con = self._connect()
        try:
            q = "SELECT ts, rate FROM funding WHERE symbol=?"
            params: list[Any] = [symbol]
            if start is not None:
                q += " AND ts >= ?"
                params.append(str(start))
            if end is not None:
                q += " AND ts <= ?"
                params.append(str(end))
            q += " ORDER BY ts"
            df = pd.read_sql_query(q, con, params=params)
        finally:
            con.close()
        return _to_utc_datetime_index(df, "ts")


def store_from_config(cfg: Dict[str, Any]) -> DataStore:
    storage = (cfg.get("data", {}) or {}).get("storage", {}) or {}
    backend = str(storage.get("backend", "parquet")).lower()
    root = str(storage.get("root_dir", "./artifacts/data"))

    if backend == "parquet":
        return ParquetStore(root_dir=root)
    if backend == "duckdb":
        db_path = str(storage.get("db_path", os.path.join(root, "hyperstat.duckdb")))
        return DuckDBStore(db_path=db_path)
    if backend == "sqlite":
        db_path = str(storage.get("db_path", os.path.join(root, "hyperstat.sqlite")))
        return SQLiteStore(db_path=db_path)

    raise ValueError(f"Unknown storage backend: {backend}")


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got={list(df.columns)}")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        raise RuntimeError(
            "Writing parquet failed. Install a parquet engine:\n"
            "  pip install pyarrow\n"
            "or\n"
            "  pip install fastparquet\n"
        ) from e


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(
            "Reading parquet failed. Install a parquet engine:\n"
            "  pip install pyarrow\n"
            "or\n"
            "  pip install fastparquet\n"
        ) from e
