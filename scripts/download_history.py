#!/usr/bin/env python3
# scripts/download_history.py
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from hyperstat.core.logging import configure_logging, get_logger
from hyperstat.data.storage import Storage, StorageConfig

from hyperstat.exchange.hyperliquid.endpoints import HyperliquidEndpoints
from hyperstat.exchange.hyperliquid.rate_limiter import RateLimiter, RateLimiterConfig
from hyperstat.exchange.hyperliquid.rest_client import HyperliquidRestClient, RestClientConfig
from hyperstat.exchange.hyperliquid.ws_client import HyperliquidWsClient, WsClientConfig
from hyperstat.exchange.hyperliquid.market_data import HyperliquidMarketData
from hyperstat.exchange.hyperliquid.funding import HyperliquidFunding

log = get_logger("scripts.download_history")
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


def _parse_list(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_candles(raw: Any) -> pd.DataFrame:
    """
    candleSnapshot returns array of candles. Common keys:
      t,o,h,l,c,v,n
    """
    arr = raw if isinstance(raw, list) else (raw.get("candles") or raw.get("data") or [])
    df = pd.DataFrame(arr)
    if df.empty:
        return df

    rename = {"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trades"}
    df = df.rename(columns=rename)

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [c for c in ["ts", "open", "high", "low", "close", "volume", "trades"] if c in df.columns]
    df = df[keep].dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"])
    return df


def _normalize_funding(raw: Any) -> pd.DataFrame:
    """
    fundingHistory returns array of items. Common keys:
      time, fundingRate (strings)
    """
    arr = raw if isinstance(raw, list) else (raw.get("fundings") or raw.get("data") or [])
    df = pd.DataFrame(arr)
    if df.empty:
        return df

    # ts
    if "time" in df.columns and "ts" not in df.columns:
        df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
    elif "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")

    # rate
    if "fundingRate" in df.columns and "rate" not in df.columns:
        df["rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    elif "rate" in df.columns:
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    keep = [c for c in ["ts", "rate"] if c in df.columns]
    df = df[keep].dropna(subset=["ts", "rate"]).sort_values("ts").drop_duplicates(subset=["ts"])
    return df


async def _paginate_candles(
    md: HyperliquidMarketData,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Hyperliquid /info candleSnapshot is time-ranged and returns a limited number of rows per call.
    Pagination strategy: use last returned candle timestamp as next startTime (+1ms).
    """
    cur = int(start_ms)
    out: List[pd.DataFrame] = []

    while cur < end_ms:
        res = await md.candle_snapshot(coin, interval, cur, end_ms)
        df = _normalize_candles(res)
        if df.empty:
            break

        out.append(df)

        last_ts = int(df["ts"].iloc[-1].value // 10**6)  # ms
        nxt = last_ts + 1
        if nxt <= cur:
            # safety break to avoid infinite loop
            break
        cur = nxt

        # if returned very small chunk, likely at end
        if len(df) < 10:
            break

    if not out:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    full = pd.concat(out, ignore_index=True)
    full = full.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"])
    return full


async def _paginate_funding(
    fund: HyperliquidFunding,
    coin: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Hyperliquid /info fundingHistory is time-ranged and returns a limited number of rows per call.
    Pagination strategy: use last returned timestamp as next startTime (+1ms).
    """
    cur = int(start_ms)
    out: List[pd.DataFrame] = []

    while cur < end_ms:
        res = await fund.funding_history(coin, cur, end_ms)
        df = _normalize_funding(res)
        if df.empty:
            break

        out.append(df)

        last_ts = int(df["ts"].iloc[-1].value // 10**6)  # ms
        nxt = last_ts + 1
        if nxt <= cur:
            break
        cur = nxt

        if len(df) < 10:
            break

    if not out:
        return pd.DataFrame(columns=["ts", "rate"])
    full = pd.concat(out, ignore_index=True)
    full = full.dropna(subset=["ts", "rate"]).sort_values("ts").drop_duplicates(subset=["ts"])
    return full


async def main_async(args: argparse.Namespace) -> int:
    cfg = load_configs(args.config) if args.config else {}
    exch = cfg.get("exchange", {}) or {}

    network = args.network or exch.get("network", "testnet")
    endpoints = HyperliquidEndpoints.mainnet() if network == "mainnet" else HyperliquidEndpoints.testnet()

    limits = exch.get("limits", {}) or {}
    rl = RateLimiter(
        RateLimiterConfig(weight_budget_per_minute=int(limits.get("rest_weight_per_min", 1100)))
    )
    rest = HyperliquidRestClient(
        endpoints=endpoints,
        rate_limiter=rl,
        cfg=RestClientConfig(
            timeout_s=float(limits.get("rest_timeout_s", 10.0)),
            max_retries=int(limits.get("max_retries", 5)),
        ),
    )
    ws = HyperliquidWsClient(
        endpoints=endpoints,
        cfg=WsClientConfig(max_subscriptions=int(limits.get("max_ws_subscriptions", 800))),
    )
    md = HyperliquidMarketData(rest=rest, ws=ws)
    fund = HyperliquidFunding(rest=rest, ws=ws)

    coins = _parse_list(args.coins)
    timeframe = args.timeframe
    start_ms = int(args.start_ms)
    end_ms = int(args.end_ms) if args.end_ms is not None else _now_ms()

    storage = Storage(StorageConfig(root=args.data_root, format=args.format))

    try:
        for coin in coins:
            if args.candles:
                cdf = await _paginate_candles(md, coin=coin, interval=timeframe, start_ms=start_ms, end_ms=end_ms)
                storage.save(cdf, kind="candles", symbol=coin, timeframe=timeframe)
                log.info("Saved candles %s tf=%s rows=%d", coin, timeframe, len(cdf))

            if args.funding:
                fdf = await _paginate_funding(fund, coin=coin, start_ms=start_ms, end_ms=end_ms)
                storage.save(fdf, kind="funding", symbol=coin, timeframe="8h")
                log.info("Saved funding %s tf=8h rows=%d", coin, len(fdf))

    finally:
        await rest.aclose()
        await ws.stop()

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Hyperliquid history (candles + funding) to local data/")
    p.add_argument("--config", action="append", default=[], help="YAML config(s), merged in order")
    p.add_argument("--network", choices=["mainnet", "testnet"], default=None, help="Override exchange.network")

    p.add_argument("--coins", required=True, help="Comma-separated coins, e.g. BTC,ETH,SOL")
    p.add_argument("--timeframe", default="5m", help="Candle interval: 1m/5m/15m/1h ...")
    p.add_argument("--start-ms", type=int, required=True, help="Start time (ms)")
    p.add_argument("--end-ms", type=int, default=None, help="End time (ms). Default: now")

    p.add_argument("--data-root", default="data", help="Storage root (default: data)")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Storage format")

    p.add_argument("--candles", action="store_true", help="Download candles")
    p.add_argument("--funding", action="store_true", help="Download fundingHistory")

    p.add_argument("--log-level", default="INFO")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(level=args.log_level)

    # default behavior if neither flag is set: do both
    if not args.candles and not args.funding:
        args.candles = True
        args.funding = True

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
