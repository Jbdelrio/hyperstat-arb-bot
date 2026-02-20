# src/hyperstat/monitoring/sink.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

Json = Dict[str, Any]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic write to avoid Streamlit reading partial files.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _utc_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class SnapshotSinkConfig:
    """
    run_dir:
      - ex: artifacts/live/my_run
    flush_each_write:
      - ensures data is visible immediately to Streamlit
    """
    run_dir: str = "artifacts/live/default"
    flush_each_write: bool = True

    # If you want to limit file growth, you can rotate externally (cron) or add rotation later.
    # rotation_bytes: Optional[int] = None


class SnapshotSink:
    """
    Append-only telemetry sink used by the live runner.

    Files written (CSV are append-friendly):
      - equity.csv: ts,equity,pnl_step,fees_est,slip_est,gross,net
      - mids.csv:   ts,symbol,mid
      - weights.csv: ts,symbol,weight
      - snapshot_latest.json: last known full snapshot (positions+mids+equity+target_weights)

    Notes:
      - Keep this module dependency-light and robust.
      - Don't store secrets.
    """

    def __init__(self, cfg: SnapshotSinkConfig) -> None:
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        _ensure_dir(self.run_dir)

        self.equity_path = self.run_dir / "equity.csv"
        self.mids_path = self.run_dir / "mids.csv"
        self.weights_path = self.run_dir / "weights.csv"
        self.latest_path = self.run_dir / "snapshot_latest.json"
        self.meta_path = self.run_dir / "run_meta.json"

        self._ensure_headers()

    def _ensure_headers(self) -> None:
        if not self.equity_path.exists():
            self.equity_path.write_text("ts,equity,pnl_step,fees_est,slip_est,gross,net\n", encoding="utf-8")
        if not self.mids_path.exists():
            self.mids_path.write_text("ts,symbol,mid\n", encoding="utf-8")
        if not self.weights_path.exists():
            self.weights_path.write_text("ts,symbol,weight\n", encoding="utf-8")

        if not self.meta_path.exists():
            meta = {
                "created_at": pd.Timestamp.utcnow().tz_localize("UTC").isoformat(),
                "pid": os.getpid(),
                "run_dir": str(self.run_dir),
            }
            _atomic_write_text(self.meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

    def append_equity(
        self,
        ts: pd.Timestamp,
        equity: float,
        pnl_step: Optional[float] = None,
        fees_est: Optional[float] = None,
        slip_est: Optional[float] = None,
        gross: Optional[float] = None,
        net: Optional[float] = None,
    ) -> None:
        ts = _utc_ts(ts)

        row = (
            f"{ts.isoformat()},"
            f"{float(equity)},"
            f"{'' if pnl_step is None else float(pnl_step)},"
            f"{'' if fees_est is None else float(fees_est)},"
            f"{'' if slip_est is None else float(slip_est)},"
            f"{'' if gross is None else float(gross)},"
            f"{'' if net is None else float(net)}\n"
        )

        with self.equity_path.open("a", encoding="utf-8") as f:
            f.write(row)
            if self.cfg.flush_each_write:
                f.flush()

    def append_mids(self, ts: pd.Timestamp, mids: Dict[str, float]) -> None:
        ts = _utc_ts(ts)
        if not mids:
            return

        with self.mids_path.open("a", encoding="utf-8") as f:
            for sym, mid in mids.items():
                try:
                    m = float(mid)
                except Exception:
                    continue
                if not (m > 0.0):
                    continue
                f.write(f"{ts.isoformat()},{sym},{m}\n")
            if self.cfg.flush_each_write:
                f.flush()

    def append_weights(self, ts: pd.Timestamp, weights: Dict[str, float]) -> None:
        ts = _utc_ts(ts)
        if not weights:
            return

        with self.weights_path.open("a", encoding="utf-8") as f:
            for sym, w in weights.items():
                try:
                    wf = float(w)
                except Exception:
                    continue
                f.write(f"{ts.isoformat()},{sym},{wf}\n")
            if self.cfg.flush_each_write:
                f.flush()

    def write_latest_snapshot(
        self,
        ts: pd.Timestamp,
        equity: float,
        positions: Dict[str, Dict[str, float]],
        mids: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None,
        extra: Optional[Json] = None,
    ) -> None:
        ts = _utc_ts(ts)

        payload: Json = {
            "ts": ts.isoformat(),
            "equity": float(equity),
            "positions": positions,  # {"ETH":{"qty":...,"entry_px":...},...}
            "mids": {k: float(v) for k, v in (mids or {}).items()},
        }
        if target_weights is not None:
            payload["target_weights"] = {k: float(v) for k, v in target_weights.items()}
        if extra is not None:
            payload["extra"] = extra

        _atomic_write_text(self.latest_path, json.dumps(payload, ensure_ascii=False))

    # --- Convenience: create "heartbeat" file (optional)
    def touch_heartbeat(self) -> None:
        hb = self.run_dir / "heartbeat.txt"
        _atomic_write_text(hb, str(time.time()))
