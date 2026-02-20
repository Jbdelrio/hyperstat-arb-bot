# src/hyperstat/cli/commands.py
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from hyperstat.core.logging import get_logger, setup_logging

# These imports assume you already have these modules in your project:
from hyperstat.strategy.stat_arb import StatArbStrategy, StatArbConfig
from hyperstat.strategy.regime import RegimeModel, RegimeConfig
from hyperstat.strategy.funding_overlay import FundingOverlayModel, FundingOverlayConfig
from hyperstat.strategy.allocator import PortfolioAllocator, AllocatorConfig

from hyperstat.backtest.engine import run_backtest
from hyperstat.backtest.reports import save_report_csv, save_report_html
from hyperstat.monitoring.sink import SnapshotSink, SnapshotSinkConfig

# Optional: if you have scripts
# from scripts.download_history import download_history
# from scripts.build_universe import build_universe


Json = Dict[str, Any]


def _load_yaml(path: str) -> Json:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: PyYAML. Install with: pip install pyyaml") from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _deep_merge(a: Json, b: Json) -> Json:
    """
    Deep merge dict b into dict a (returns new dict).
    """
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(paths: List[str]) -> Json:
    cfg: Json = {}
    for p in paths:
        cfg = _deep_merge(cfg, _load_yaml(p))
    return cfg


def _parse_coins(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def _tf_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {tf}")


def _load_local_history(data_dir: str, coins: List[str], timeframe: str) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Minimal loader (no DuckDB required):
      - candles: data/candles/<COIN>_<TF>.parquet OR .csv
      - funding: data/funding/<COIN>.parquet OR .csv (or <COIN>_<TF>)
    Candle required cols: ts,open,high,low,close,volume
    Funding required cols: ts,rate
    """
    base = Path(data_dir)
    candles_dir = base / "candles"
    funding_dir = base / "funding"

    candles_by: Dict[str, pd.DataFrame] = {}
    funding_by: Dict[str, pd.DataFrame] = {}

    for c in coins:
        # candles
        cand_parq = candles_dir / f"{c}_{timeframe}.parquet"
        cand_csv = candles_dir / f"{c}_{timeframe}.csv"
        if cand_parq.exists():
            df = pd.read_parquet(cand_parq)
        elif cand_csv.exists():
            df = pd.read_csv(cand_csv)
        else:
            df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        candles_by[c] = df

        # funding
        fund_parq = funding_dir / f"{c}.parquet"
        fund_csv = funding_dir / f"{c}.csv"
        fund_parq2 = funding_dir / f"{c}_{timeframe}.parquet"
        fund_csv2 = funding_dir / f"{c}_{timeframe}.csv"
        if fund_parq2.exists():
            fdf = pd.read_parquet(fund_parq2)
        elif fund_csv2.exists():
            fdf = pd.read_csv(fund_csv2)
        elif fund_parq.exists():
            fdf = pd.read_parquet(fund_parq)
        elif fund_csv.exists():
            fdf = pd.read_csv(fund_csv)
        else:
            fdf = pd.DataFrame(columns=["ts", "rate"])
        funding_by[c] = fdf

    return candles_by, funding_by


def _build_buckets_simple(coins: List[str], bucket_size: int = 10) -> Dict[str, List[str]]:
    """
    Simple deterministic buckets if you don't load clustering output yet.
    Replace with data.universe clustering later.
    """
    coins = list(coins)
    buckets: Dict[str, List[str]] = {}
    for i in range(0, len(coins), bucket_size):
        buckets[f"bucket_{i//bucket_size}"] = coins[i : i + bucket_size]
    return buckets


def _make_strategy_stack(cfg: Json) -> tuple[StatArbStrategy, RegimeModel, FundingOverlayModel, PortfolioAllocator]:
    data = cfg.get("data", {}) or {}
    strat = cfg.get("strategy", {}) or {}
    portfolio = cfg.get("portfolio", {}) or {}
    risk = cfg.get("risk", {}) or {}

    tfm = _tf_minutes(str(data.get("timeframe", "5m")))
    base_factor = str(data.get("base_factor_symbol", "BTC"))

    # StatArb
    sig = (strat.get("signal", {}) or {})
    stat_arb = StatArbStrategy(
        StatArbConfig(
            timeframe_minutes=tfm,
            horizon_bars=int(sig.get("horizon_bars", 12)),
            z_in=float(sig.get("z_in", 1.5)),
            z_out=float(sig.get("z_out", 0.5)),
            z_max=float(sig.get("z_max", 3.0)),
            min_hold_minutes=int(sig.get("min_hold_minutes", 30)),
            max_hold_minutes=int(sig.get("max_hold_minutes", 1440)),
        )
    )

    # Regime
    reg = (strat.get("regime", {}) or {})
    mr = (reg.get("mr", {}) or {})
    rk = (reg.get("risk", {}) or {})
    regime = RegimeModel(
        RegimeConfig(
            timeframe_minutes=tfm,
            ar1_window_days=int(mr.get("ar1_window_days", 7)),
            halflife_min_minutes=int(mr.get("halflife_min_minutes", 30)),
            halflife_good_max_minutes=int(mr.get("halflife_good_max_minutes", 360)),
            halflife_ok_max_minutes=int(mr.get("halflife_ok_max_minutes", 1440)),
            risk_history_days=int(rk.get("vol_window_days", 60)),
            high_vol_pctl=float(rk.get("high_vol_pctl", 0.90)),
            extreme_vol_pctl=float(rk.get("extreme_vol_pctl", 0.95)),
        ),
        base_factor_symbol=base_factor,
    )

    # Funding overlay
    fcfg = (strat.get("funding_overlay", {}) or {})
    be = (fcfg.get("break_even", {}) or {})
    funding_overlay = FundingOverlayModel(
        FundingOverlayConfig(
            enabled=bool(fcfg.get("enabled", True)),
            ewma_lambda=float(fcfg.get("ewma_lambda", 0.15)),
            snr_gate_min=float(fcfg.get("snr_gate_min", 0.5)),
            eta=float(fcfg.get("eta", 0.10)),
            gross_target=float(fcfg.get("gross_target", 0.20)),
            horizon_funding_periods=int(be.get("horizon_funding_periods", 3)),
            fee_bps=float(be.get("fee_bps", 6.0)),
            slip_bps_base=float(be.get("slip_bps_base", 8.0)),
            buffer_bps=float(be.get("buffer_bps", 5.0)),
        )
    )

    # Allocator
    alloc = (strat.get("allocator", {}) or {})
    allocator = PortfolioAllocator(
        cfg=AllocatorConfig(
            gross_target_stat=float(portfolio.get("gross_target", 1.2)),
            gross_target_fund=float(fcfg.get("gross_target", 0.20)),
            max_weight_per_coin=float(portfolio.get("max_weight_per_coin", 0.12)),
            max_weight_per_bucket=float(portfolio.get("max_weight_per_bucket", 0.35)),
            dollar_neutral=bool(portfolio.get("dollar_neutral", True)),
            beta_neutral=bool(portfolio.get("beta_neutral", True)),
            z_emergency_flat=float(risk.get("z_emergency_flat", 3.5)),
        ),
        funding_overlay=funding_overlay,
    )

    return stat_arb, regime, funding_overlay, allocator


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    log = get_logger("hyperstat.cli")

    coins = _parse_coins(args.coins)
    timeframe = args.timeframe or str((cfg.get("data", {}) or {}).get("timeframe", "5m"))
    data_dir = args.data_dir or str((cfg.get("data", {}) or {}).get("data_dir", "data"))
    run_name = args.run_name or "backtest"

    # buckets: load from cfg if present, else simple buckets
    uni = (cfg.get("universe", {}) or {})
    buckets = uni.get("buckets") if isinstance(uni.get("buckets"), dict) else None
    if buckets is None:
        buckets = _build_buckets_simple(coins, bucket_size=int(uni.get("bucket_size", 10)))

    candles_by, funding_by = _load_local_history(data_dir=data_dir, coins=coins, timeframe=timeframe)
    stat_arb, regime, funding_overlay, allocator = _make_strategy_stack(cfg)

    report = run_backtest(
        cfg=cfg,
        candles_by_symbol=candles_by,
        funding_by_symbol=funding_by,
        buckets=buckets,
        stat_arb=stat_arb,
        regime_model=regime,
        allocator=allocator,
        funding_overlay=funding_overlay,
    )

    out_dir = Path("artifacts/backtests") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_report_csv(report, str(out_dir))
    save_report_html(report, str(out_dir), title=f"hyperstat backtest — {run_name}")

    log.info("Backtest done. Output: %s", out_dir)
    log.info("Metrics: total_return=%.3f sharpe=%.3f max_dd=%.3f",
             report.metrics.total_return, report.metrics.sharpe, report.metrics.max_drawdown)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """
    Live runner is in hyperstat.live.runner.LiveRunner.
    """
    cfg = load_config(args.config)
    run = cfg.get("run", {}) or {}
    run["mode"] = "live" if args.live else "paper"
    run["name"] = args.run_name or run.get("name", "default")
    cfg["run"] = run

    # Lazy import to avoid importing exchange deps for backtest-only usage
    from hyperstat.live.runner import LiveRunner

    r = LiveRunner(cfg)
    import asyncio
    asyncio.run(r.run())
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """
    Convenience command: launches Streamlit dashboard.
    Requires `streamlit` installed.
    """
    try:
        import streamlit  # noqa: F401
    except Exception as e:
        raise RuntimeError("streamlit not installed. Install with: pip install streamlit plotly") from e

    # We delegate to `streamlit run apps/dashboard.py`
    app = args.app or "apps/dashboard.py"
    os.execvp("streamlit", ["streamlit", "run", app])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hyperstat", description="HyperStat stat-arb bot CLI")
    p.add_argument("--config", action="append", default=[], help="YAML config file (can be repeated, merged in order)")

    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run local backtest from data dir")
    bt.add_argument("--coins", required=True, help="Comma-separated coins (ex: BTC,ETH,SOL)")
    bt.add_argument("--timeframe", default=None, help="Override timeframe (ex: 5m)")
    bt.add_argument("--data-dir", default=None, help="Data directory (default: cfg.data.data_dir or ./data)")
    bt.add_argument("--run-name", default="demo", help="Output name in artifacts/backtests/<run-name>/")
    bt.set_defaults(func=cmd_backtest)

    lv = sub.add_parser("live", help="Run paper/live using Hyperliquid")
    lv.add_argument("--run-name", default="default", help="artifacts/live/<run-name>/")
    lv.add_argument("--live", action="store_true", help="If set, sends real orders (otherwise paper).")
    lv.set_defaults(func=cmd_live)

    db = sub.add_parser("dashboard", help="Run Streamlit dashboard")
    db.add_argument("--app", default="apps/dashboard.py", help="Streamlit app path")
    db.set_defaults(func=cmd_dashboard)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    # If no --config provided, fallback to configs/default.yaml if exists
    if not args.config:
        default = Path("configs/default.yaml")
        if default.exists():
            args.config = [str(default)]

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
