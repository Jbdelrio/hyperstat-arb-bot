# tests/test_strategy_smoke.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hyperstat.backtest.engine import BacktestConfig, BacktestEngine, run_backtest
from hyperstat.backtest.costs import CostModel, FeeModel, SlippageModel
from hyperstat.backtest.reports import save_report_csv
from hyperstat.core.risk import KillSwitchConfig
from hyperstat.strategy.stat_arb import StatArbConfig, StatArbStrategy
from hyperstat.strategy.regime import RegimeConfig, RegimeModel
from hyperstat.strategy.allocator import AllocatorConfig, PortfolioAllocator
from hyperstat.strategy.funding_overlay import FundingOverlayConfig, FundingOverlayModel
from hyperstat.strategy.funding_divergence_signal import FundingDivergenceSignalLive, FDSConfig


def _fake_candles(
    start: str, n: int = 600, freq: str = "5min", seed: int = 7, base_px: float = 100.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    steps = rng.normal(0.0, 0.05, size=n)
    px = np.maximum(base_px + np.cumsum(steps), 1.0)
    return pd.DataFrame({
        "ts": idx,
        "open": px,
        "high": px * 1.001,
        "low": px * 0.999,
        "close": px,
        "volume": rng.uniform(1e4, 1e6, size=n),
    })


def _fake_funding(start: str, n: int = 75, freq: str = "8h", seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"ts": idx, "rate": rng.normal(loc=0.0001, scale=0.00005, size=n)})


def _build_components(fds_enabled: bool = True):
    stat_arb = StatArbStrategy(cfg=StatArbConfig(timeframe_minutes=5, horizon_bars=12))
    regime_model = RegimeModel(cfg=RegimeConfig(timeframe_minutes=5))
    overlay = FundingOverlayModel(cfg=FundingOverlayConfig(enabled=True, eta=0.10))
    fds = FundingDivergenceSignalLive(FDSConfig(gate_scale=0.6)) if fds_enabled else None
    allocator = PortfolioAllocator(
        cfg=AllocatorConfig(gross_target_stat=1.20, dollar_neutral=True, beta_neutral=False),
        funding_overlay=overlay,
        fds=fds,
    )
    return stat_arb, regime_model, overlay, allocator


def test_backtest_runs_and_returns_report():
    candles = {
        "BTC": _fake_candles("2025-01-01", seed=1, base_px=40000),
        "ETH": _fake_candles("2025-01-01", seed=2, base_px=2500),
        "SOL": _fake_candles("2025-01-01", seed=3, base_px=100),
    }
    funding = {"ETH": _fake_funding("2025-01-01"), "SOL": _fake_funding("2025-01-01", seed=11)}
    buckets = {"bucket_0": ["BTC", "ETH", "SOL"]}
    stat_arb, regime_model, overlay, allocator = _build_components(fds_enabled=True)

    cfg = BacktestConfig(timeframe="5m", base_factor_symbol="BTC", initial_equity=1500.0)
    engine = BacktestEngine(
        cfg=cfg,
        candles_by_symbol=candles,
        funding_by_symbol=funding,
        buckets=buckets,
        stat_arb=stat_arb,
        regime_model=regime_model,
        allocator=allocator,
        funding_overlay=overlay,
        kill_switch_cfg=KillSwitchConfig(max_intraday_drawdown_pct=0.03, cooldown_minutes=720, z_emergency_flat=3.5),
    )
    report = engine.run()

    assert not report.equity_curve.empty
    assert "equity" in report.equity_curve.columns
    assert (report.equity_curve["equity"].dropna() > 0).all()
    assert not report.weights.empty
    assert set(["BTC", "ETH", "SOL"]).issubset(set(report.weights.columns))
    assert "pnl_net_step" in report.pnl_curve.columns
    assert report.breakdown["fees"] >= 0.0
    assert report.metrics.max_drawdown <= 0.0


def test_fds_active_after_warmup():
    candles = {s: _fake_candles("2025-01-01", seed=i, base_px=100) for i, s in enumerate(["BTC","ETH","SOL","AVAX"], 1)}
    funding = {s: _fake_funding("2025-01-01", seed=i) for i, s in enumerate(candles, 20)}
    buckets = {"b0": ["BTC", "ETH"], "b1": ["SOL", "AVAX"]}
    stat_arb, regime_model, overlay, allocator = _build_components(fds_enabled=True)

    engine = BacktestEngine(
        cfg=BacktestConfig(timeframe="5m", base_factor_symbol="BTC", initial_equity=1500.0),
        candles_by_symbol=candles,
        funding_by_symbol=funding,
        buckets=buckets,
        stat_arb=stat_arb,
        regime_model=regime_model,
        allocator=allocator,
        funding_overlay=overlay,
    )
    report = engine.run()
    assert not report.equity_curve.empty


def test_save_report_csv(tmp_path: Path):
    candles = {
        "BTC": _fake_candles("2025-01-01", seed=1, base_px=40000),
        "ETH": _fake_candles("2025-01-01", seed=2, base_px=2500),
        "SOL": _fake_candles("2025-01-01", seed=3, base_px=100),
    }
    stat_arb, regime_model, overlay, allocator = _build_components(fds_enabled=False)
    engine = BacktestEngine(
        cfg=BacktestConfig(timeframe="5m", base_factor_symbol="BTC", initial_equity=1500.0),
        candles_by_symbol=candles,
        funding_by_symbol={},
        buckets={"bucket_0": ["BTC", "ETH", "SOL"]},
        stat_arb=stat_arb,
        regime_model=regime_model,
        allocator=allocator,
    )
    report = engine.run()
    save_report_csv(report, out_dir=str(tmp_path))

    for fname in ["equity_curve.csv", "pnl_curve.csv", "weights.csv", "metrics.csv", "breakdown.csv"]:
        assert (tmp_path / fname).exists(), f"Fichier manquant : {fname}"


def test_run_backtest_dict_helper():
    candles = {
        "BTC": _fake_candles("2025-01-01", seed=1, base_px=40000),
        "ETH": _fake_candles("2025-01-01", seed=2, base_px=2500),
        "SOL": _fake_candles("2025-01-01", seed=3, base_px=100),
    }
    stat_arb, regime_model, overlay, allocator = _build_components(fds_enabled=False)
    yaml_cfg = {
        "data": {"timeframe": "5m", "base_factor_symbol": "BTC"},
        "portfolio": {"initial_equity_eur": 1500.0},
        "execution": {
            "mode": "taker", "fill_model": "close",
            "fees": {"taker_bps": 6.0, "maker_bps": 2.0},
            "slippage": {"base_bps": 8.0, "k_bps_per_1pct_rv1h": 10.0, "cap_bps": 200.0},
        },
        "risk": {"max_intraday_drawdown_pct": 0.03, "cooldown_minutes": 720, "z_emergency_flat": 3.5},
    }
    report = run_backtest(
        cfg=yaml_cfg,
        candles_by_symbol=candles,
        funding_by_symbol={},
        buckets={"bucket_0": ["BTC", "ETH", "SOL"]},
        stat_arb=stat_arb,
        regime_model=regime_model,
        allocator=allocator,
    )
    assert not report.equity_curve.empty
    assert report.breakdown["fees"] >= 0.0
