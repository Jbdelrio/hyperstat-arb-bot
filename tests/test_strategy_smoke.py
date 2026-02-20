# tests/test_strategy_smoke.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hyperstat.backtest.engine import run_backtest, BacktestConfig


def _fake_candles(start: str, n: int = 600, freq: str = "5min", seed: int = 7) -> pd.DataFrame:
    """
    Génère une série OHLCV "propre" (UTC, positive) pour un test smoke rapide.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")

    # random walk doucement
    steps = rng.normal(0.0, 0.05, size=n)  # petite volatilité
    px = 100.0 + np.cumsum(steps)
    px = np.maximum(px, 1.0)  # garde prix positifs

    df = pd.DataFrame(
        {
            "ts": idx,
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "volume": 1.0,
        }
    )
    return df


def test_backtest_runs_and_writes_artifacts(tmp_path: Path):
    """
    Smoke test:
      - run_backtest doit tourner sans data externe
      - renvoyer des DataFrames non vides
      - écrire equity/positions/weights/metrics dans out_dir
    """
    # 3 actifs (inclure BTC si possible, car le régime model peut prendre BTC comme factor)
    candles = {
        "BTC": _fake_candles("2025-01-01", seed=1),
        "ETH": _fake_candles("2025-01-01", seed=2),
        "SOL": _fake_candles("2025-01-01", seed=3),
    }

    funding = {}  # pas de funding dans ce smoke test
    buckets = {"bucket_0": ["BTC", "ETH", "SOL"]}

    cfg = BacktestConfig(
        timeframe="5m",
        initial_equity=1500.0,
        run_name="smoke_test",
        out_dir=str(tmp_path),
        exec_mode="taker",
    )

    equity_df, positions_df, weights_df, metrics_df = run_backtest(
        candles=candles,
        funding=funding,
        buckets=buckets,
        cfg=cfg,
    )

    # --- basic shape checks
    assert not equity_df.empty
    assert "equity" in equity_df.columns
    assert (equity_df["equity"].dropna() > 0).all()

    assert not positions_df.empty
    assert any(c.endswith("_qty") for c in positions_df.columns)
    assert any(c.endswith("_notional") for c in positions_df.columns)

    assert not weights_df.empty
    assert set(["BTC", "ETH", "SOL"]).issubset(set(weights_df.columns))

    assert not metrics_df.empty
    assert set(["ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover"]).issubset(metrics_df.columns)

    # --- artifacts written
    run_dir = tmp_path / "smoke_test"
    assert (run_dir / "equity.csv").exists()
    assert (run_dir / "positions.csv").exists()
    assert (run_dir / "weights.csv").exists()
    assert (run_dir / "metrics.csv").exists()
