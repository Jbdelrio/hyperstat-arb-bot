#!/usr/bin/env python3
# scripts/build_universe.py
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from hyperstat.core.logging import configure_logging, get_logger
from hyperstat.data.storage import Storage, StorageConfig
from hyperstat.data.universe import build_universe_by_liquidity, build_buckets_greedy_corr

from hyperstat.exchange.hyperliquid.endpoints import HyperliquidEndpoints
from hyperstat.exchange.hyperliquid.rate_limiter import RateLimiter, RateLimiterConfig
from hyperstat.exchange.hyperliquid.rest_client import HyperliquidRestClient, RestClientConfig
from hyperstat.exchange.hyperliquid.ws_client import HyperliquidWsClient, WsClientConfig
from hyperstat.exchange.hyperliquid.market_data import HyperliquidMarketData

log = get_logger("scripts.build_universe")
Json = Dict[str, Any]


def _deep_merge(a: Json, b: Json) -> Json:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_configs(paths: List[str]) -> Json:
    cfg: Json = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, d)
    return cfg


def _now_ms() -> int:
    return int(time.time() * 1000)


def _tf_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3_600_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def _parse_meta_and_asset_ctxs(res: Any) -> pd.DataFrame:
    """
    metaAndAssetCtxs is usually:
      {"universe":[{"name":"BTC",...},...], "assetCtxs":[{"dayNtlVlm":"...",...},...]}
    We align by index.
    """
    if not isinstance(res, dict):
        return pd.DataFrame()

    universe = res.get("universe") or (res.get("data") or {}).get("universe")
    ctxs = res.get("assetCtxs") or (res.get("data") or {}).get("assetCtxs")

    if not (isinstance(universe, list) and isinstance(ctxs, list) and len(universe) == len(ctxs)):
        return pd.DataFrame()

    rows = []
    for u, c in zip(universe, ctxs):
        if not isinstance(u, dict) or not isinstance(c, dict):
            continue
        coin = u.get("name") or u.get("coin")
        if not coin:
            continue
        day = c.get("dayNtlVlm") or c.get("dayNtlVol") or c.get("dayVolume") or c.get("volume")
        rows.append({"coin": str(coin), "dayNtlVlm": day})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dayNtlVlm"] = pd.to_numeric(df["dayNtlVlm"], errors="coerce")
    df = df.dropna(subset=["coin", "dayNtlVlm"]).sort_values("dayNtlVlm", ascending=False)
    return df


def _prices_wide_from_storage(storage: Storage, coins: List[str], timeframe: str, corr_window: int) -> pd.DataFrame:
    """
    Load local candles and build wide close price matrix.
    """
    prices: Dict[str, pd.Series] = {}
    for c in coins:
        df = storage.load(kind="candles", symbol=c, timeframe=timeframe)
        if df is None or df.empty:
            continue
        d = df.copy()
        if "ts" not in d.columns:
            continue
        d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
        d = d.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
        if "close" not in d.columns:
            continue
        px = pd.to_numeric(d["close"], errors="coerce").ffill()
        prices[c] = px

    if not prices:
        return pd.DataFrame()

    pxw = pd.DataFrame(prices).sort_index().ffill()
    # keep last corr_window bars (approx)
    if corr_window > 0 and len(pxw) > corr_window:
        pxw = pxw.tail(corr_window)
    return pxw


async def main_async(args: argparse.Namespace) -> int:
    cfg = load_configs(args.config) if args.config else {}
    exch = cfg.get("exchange", {}) or {}

    network = args.network or exch.get("network", "testnet")
    endpoints = HyperliquidEndpoints.mainnet() if network == "mainnet" else HyperliquidEndpoints.testnet()

    limits = exch.get("limits", {}) or {}
    rl = RateLimiter(RateLimiterConfig(weight_budget_per_minute=int(limits.get("rest_weight_per_min", 1100))))
    rest = HyperliquidRestClient(
        endpoints=endpoints,
        rate_limiter=rl,
        cfg=RestClientConfig(timeout_s=float(limits.get("rest_timeout_s", 10.0)), max_retries=int(limits.get("max_retries", 5))),
    )
    ws = HyperliquidWsClient(endpoints=endpoints, cfg=WsClientConfig(max_subscriptions=int(limits.get("max_ws_subscriptions", 800))))
    md = HyperliquidMarketData(rest=rest, ws=ws)

    # params
    timeframe = args.timeframe or str((cfg.get("data", {}) or {}).get("timeframe", "5m"))
    target_size = int(args.target_size or (cfg.get("universe", {}) or {}).get("target_size", 30))
    min_day_ntl_vlm = float(args.min_day_ntl_vlm)
    corr_threshold = float(args.corr_threshold)
    corr_window = int(args.corr_window)

    storage = Storage(StorageConfig(root=args.data_root, format=args.format))

    try:
        res = await md.meta_and_asset_ctxs()
        ctx_df = _parse_meta_and_asset_ctxs(res)
        if ctx_df.empty:
            raise RuntimeError("metaAndAssetCtxs returned no usable data")

        symbols = build_universe_by_liquidity(
            asset_ctxs=ctx_df,
            target_size=target_size,
            min_day_ntl_vlm=min_day_ntl_vlm,
        )
        if not symbols:
            raise RuntimeError("No symbols selected (liquidity filter too strict?)")

        # Build buckets from local candle data
        prices_wide = _prices_wide_from_storage(storage, symbols, timeframe=timeframe, corr_window=corr_window)
        if prices_wide.empty or prices_wide.shape[1] < 2:
            # fallback: single bucket if we don't have data yet
            buckets = {"bucket_0": symbols}
            log.warning("Not enough local candle data to build corr buckets -> fallback single bucket.")
        else:
            buckets = build_buckets_greedy_corr(prices_wide, corr_threshold=corr_threshold)

        out = {
            "universe": {
                "target_size": target_size,
                "symbols": symbols,
                # intended to be merged in config for live runner
                "buckets_live": buckets,
                "corr_threshold": corr_threshold,
                "corr_window": corr_window,
                "min_day_ntl_vlm": min_day_ntl_vlm,
            },
            "meta": {
                "generated_at": pd.Timestamp.utcnow().tz_localize("UTC").isoformat(),
                "network": network,
                "timeframe": timeframe,
            },
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
        log.info("Universe saved: %s (symbols=%d buckets=%d)", out_path, len(symbols), len(buckets))

    finally:
        await rest.aclose()
        await ws.stop()

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build universe + buckets (liquidity + corr buckets)")
    p.add_argument("--config", action="append", default=[], help="YAML config(s), merged in order")
    p.add_argument("--network", choices=["mainnet", "testnet"], default=None)

    p.add_argument("--timeframe", default=None, help="Use same tf as your live/backtest (default from config)")
    p.add_argument("--target-size", type=int, default=None, help="Number of coins to select")
    p.add_argument("--min-day-ntl-vlm", type=float, default=1e6, help="Liquidity filter (dayNtlVlm)")

    p.add_argument("--corr-threshold", type=float, default=0.75, help="Greedy corr bucket threshold")
    p.add_argument("--corr-window", type=int, default=2000, help="Bars used for corr buckets (local candles)")

    p.add_argument("--data-root", default="data", help="Local data root")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Local data format")

    p.add_argument("--out", default="artifacts/universe/universe_generated.yaml", help="Output YAML to merge in live config")
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(level=args.log_level)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
