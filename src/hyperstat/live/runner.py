# src/hyperstat/live/runner.py
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from hyperstat.core.logging import get_logger
from hyperstat.core.risk import KillSwitchConfig, RiskState
from hyperstat.strategy.stat_arb import StatArbConfig, StatArbStrategy
from hyperstat.strategy.regime import RegimeConfig, RegimeModel
from hyperstat.strategy.funding_overlay import FundingOverlayConfig, FundingOverlayModel
from hyperstat.strategy.allocator import AllocatorConfig, PortfolioAllocator

from hyperstat.exchange.hyperliquid.endpoints import HyperliquidEndpoints
from hyperstat.exchange.hyperliquid.rate_limiter import RateLimiter, RateLimiterConfig
from hyperstat.exchange.hyperliquid.rest_client import HyperliquidRestClient, RestClientConfig
from hyperstat.exchange.hyperliquid.ws_client import HyperliquidWsClient, WsClientConfig
from hyperstat.exchange.hyperliquid.auth import HyperliquidAuth
from hyperstat.exchange.hyperliquid.market_data import HyperliquidMarketData
from hyperstat.exchange.hyperliquid.funding import HyperliquidFunding
from hyperstat.exchange.hyperliquid.execution import HyperliquidExecution

from .order_manager import OrderManager, OrderManagerConfig
from .health import HealthMonitor, HealthConfig


Json = dict[str, Any]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _tf_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {tf}")


def _ceil_time_to_tf(now: datetime, tf_minutes: int) -> datetime:
    """
    Next bar boundary in UTC.
    """
    ts = int(now.timestamp())
    step = tf_minutes * 60
    nxt = int(math.floor(ts / step) * step + step)
    return datetime.fromtimestamp(nxt, tz=timezone.utc)


@dataclass(frozen=True)
class LiveRunnerConfig:
    """
    Live/paper runner config (derived from YAML dict).
    """
    mode: str = "paper"  # paper | live
    timeframe: str = "5m"
    base_factor_symbol: str = "BTC"
    universe_symbols: List[str] = None  # if None -> must be provided externally

    # buckets must be provided (or we add a bucket builder later using live returns corr)
    buckets: Dict[str, List[str]] = None

    # data refresh
    poll_mids_every_s: float = 5.0
    poll_meta_ctx_every_s: float = 60.0
    poll_predicted_funding_every_s: float = 300.0

    # safety
    kill_switch: KillSwitchConfig = KillSwitchConfig(
        max_intraday_drawdown_pct=0.03,
        cooldown_minutes=720,
        z_emergency_flat=3.5,
    )
    health: HealthConfig = HealthConfig()


class LiveFeatureStore:
    """
    Live rolling feature computation from mids only.
    Provides:
      - ewma_vol (sqrt(ewma(r^2)))
      - rv_1h (sqrt(sum r^2 over 1h window))

    Liquidity features (dv/illiq) are left optional; can be injected from /metaAndAssetCtxs later.
    """

    def __init__(self, tf_minutes: int, ewma_lam: float = 0.94) -> None:
        self.tf_minutes = tf_minutes
        self.ewma_lam = ewma_lam
        self._last_px: Dict[str, float] = {}
        self._ewma_r2: Dict[str, float] = {}
        self._r2_window: Dict[str, List[float]] = {}
        self._win = max(1, int(60 / tf_minutes))  # 1 hour

        # optional injected features
        self._dv: Dict[str, float] = {}
        self._illiq: Dict[str, float] = {}

    def inject_dv(self, dv_by_symbol: Dict[str, float]) -> None:
        for s, dv in dv_by_symbol.items():
            if np.isfinite(dv):
                self._dv[s] = float(dv)

    def inject_illiq(self, illiq_by_symbol: Dict[str, float]) -> None:
        for s, ill in illiq_by_symbol.items():
            if np.isfinite(ill):
                self._illiq[s] = float(ill)

    def update_prices(self, mids: Dict[str, float]) -> None:
        for s, px in mids.items():
            px = float(px)
            if not np.isfinite(px) or px <= 0:
                continue

            prev = self._last_px.get(s)
            self._last_px[s] = px
            if prev is None or prev <= 0:
                continue

            r = math.log(px / prev)
            r2 = r * r

            # EWMA r^2
            prev_ewma = self._ewma_r2.get(s)
            if prev_ewma is None or not np.isfinite(prev_ewma):
                self._ewma_r2[s] = r2
            else:
                self._ewma_r2[s] = self.ewma_lam * prev_ewma + (1.0 - self.ewma_lam) * r2

            # rolling window for RV1h
            w = self._r2_window.setdefault(s, [])
            w.append(r2)
            if len(w) > self._win:
                w.pop(0)

    def features(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for s in symbols:
            d: Dict[str, float] = {}
            ew = self._ewma_r2.get(s)
            if ew is not None and np.isfinite(ew) and ew >= 0:
                d["ewma_vol"] = float(math.sqrt(ew))

            w = self._r2_window.get(s)
            if w:
                d["rv_1h"] = float(math.sqrt(sum(w)))

            if s in self._dv:
                d["dv"] = float(self._dv[s])
            if s in self._illiq:
                d["illiq"] = float(self._illiq[s])

            out[s] = d
        return out


class LiveRunner:
    """
    Live runner:
      WS/REST -> mids/funding -> strategy -> target weights -> order intents -> execution
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg_raw = cfg
        self.log = get_logger("hyperstat.live")

        run = cfg.get("run", {}) or {}
        data = cfg.get("data", {}) or {}
        ex = cfg.get("exchange", {}) or {}
        risk = cfg.get("risk", {}) or {}

        self.mode = str(run.get("mode", "paper")).lower()
        self.timeframe = str(data.get("timeframe", "5m"))
        self.tf_minutes = _tf_minutes(self.timeframe)
        self.base_factor_symbol = str(data.get("base_factor_symbol", "BTC"))

        # endpoints
        network = str(ex.get("network", "testnet")).lower()
        if network == "mainnet":
            endpoints = HyperliquidEndpoints.mainnet()
        else:
            endpoints = HyperliquidEndpoints.testnet()

        # rate limiter
        limits = ex.get("limits", {}) or {}
        rl = RateLimiter(
            RateLimiterConfig(
                weight_budget_per_minute=int(limits.get("rest_weight_per_min", 1100)),
                hard_floor_remaining=0,
            )
        )
        self.rate_limiter = rl

        # clients
        rest_cfg = RestClientConfig(
            timeout_s=float(limits.get("rest_timeout_s", 10)),
            max_retries=int(limits.get("max_retries", 5)),
        )
        self.rest = HyperliquidRestClient(endpoints=endpoints, rate_limiter=rl, cfg=rest_cfg)

        ws_cfg = WsClientConfig(
            max_subscriptions=int(limits.get("max_ws_subscriptions", 800)),
        )
        self.ws = HyperliquidWsClient(endpoints=endpoints, cfg=ws_cfg)

        # auth + execution (keys are read in main CLI; here we assume env is loaded)
        acc = ex.get("account", {}) or {}
        addr_env = str(acc.get("address_env", "HL_ADDRESS"))
        pk_env = str(acc.get("private_key_env", "HL_PRIVATE_KEY"))

        import os
        address = os.getenv(addr_env)
        priv = os.getenv(pk_env)
        if not address or not priv:
            raise RuntimeError(f"Missing env vars for exchange auth: {addr_env} / {pk_env}")

        self.user = address.lower()
        self.auth = HyperliquidAuth(private_key=priv, is_mainnet=(network == "mainnet"), account_address=address)

        self.market = HyperliquidMarketData(rest=self.rest, ws=self.ws)
        self.funding = HyperliquidFunding(rest=self.rest, ws=self.ws)
        self.exec = HyperliquidExecution(rest=self.rest, auth=self.auth)

        # strategy config from YAML (minimal extraction)
        strat_cfg = (cfg.get("strategy", {}) or {})
        sig_cfg = (strat_cfg.get("signal", {}) or {})
        reg_cfg = (strat_cfg.get("regime", {}) or {})
        fund_cfg = (strat_cfg.get("funding_overlay", {}) or {})

        self.stat_arb = StatArbStrategy(
            StatArbConfig(
                timeframe_minutes=self.tf_minutes,
                horizon_bars=int(sig_cfg.get("horizon_bars", 12)),
                z_in=float(sig_cfg.get("z_in", 1.5)),
                z_out=float(sig_cfg.get("z_out", 0.5)),
                z_max=float(sig_cfg.get("z_max", 3.0)),
                min_hold_minutes=int(sig_cfg.get("min_hold_minutes", 30)),
                max_hold_minutes=int(sig_cfg.get("max_hold_minutes", 1440)),
            )
        )
        self.regime = RegimeModel(
            RegimeConfig(
                timeframe_minutes=self.tf_minutes,
                ar1_window_days=int(reg_cfg.get("mr", {}).get("ar1_window_days", 7)),
                halflife_min_minutes=int(reg_cfg.get("mr", {}).get("halflife_min_minutes", 30)),
                halflife_good_max_minutes=int(reg_cfg.get("mr", {}).get("halflife_good_max_minutes", 360)),
                halflife_ok_max_minutes=int(reg_cfg.get("mr", {}).get("halflife_ok_max_minutes", 1440)),
                risk_history_days=int(reg_cfg.get("risk", {}).get("vol_window_days", 60)),
                high_vol_pctl=float(reg_cfg.get("risk", {}).get("high_vol_pctl", 0.90)),
                extreme_vol_pctl=float(reg_cfg.get("risk", {}).get("extreme_vol_pctl", 0.95)),
            ),
            base_factor_symbol=self.base_factor_symbol,
        )

        self.funding_overlay = FundingOverlayModel(
            FundingOverlayConfig(
                enabled=bool(fund_cfg.get("enabled", True)),
                ewma_lambda=float(fund_cfg.get("ewma_lambda", 0.15)),
                snr_gate_min=float(fund_cfg.get("snr_gate_min", 0.5)),
                eta=float(fund_cfg.get("eta", 0.10)),
                gross_target=float(fund_cfg.get("gross_target", 0.20)),
                horizon_funding_periods=int(fund_cfg.get("break_even", {}).get("horizon_funding_periods", 3)),
                fee_bps=float(fund_cfg.get("break_even", {}).get("buffer_bps", 5.0)),  # conservative fallback
            )
        )

        alloc_cfg = AllocatorConfig(
            gross_target_stat=float((cfg.get("portfolio", {}) or {}).get("gross_target", 1.2)),
            gross_target_fund=float(fund_cfg.get("gross_target", 0.20)),
            max_weight_per_coin=float((cfg.get("portfolio", {}) or {}).get("max_weight_per_coin", 0.12)),
            max_weight_per_bucket=float((cfg.get("portfolio", {}) or {}).get("max_weight_per_bucket", 0.35)),
            dollar_neutral=bool((cfg.get("portfolio", {}) or {}).get("dollar_neutral", True)),
            beta_neutral=bool((cfg.get("portfolio", {}) or {}).get("beta_neutral", True)),
            z_emergency_flat=float(risk.get("z_emergency_flat", 3.5)),
        )
        self.allocator = PortfolioAllocator(cfg=alloc_cfg, funding_overlay=self.funding_overlay)

        # order manager
        exe_cfg = (cfg.get("execution", {}) or {})
        om_cfg = OrderManagerConfig(
            execution_enabled=(self.mode == "live"),
            quote_ccy=str((cfg.get("data", {}) or {}).get("quote_ccy", "USDC")),
            min_trade_notional=float(exe_cfg.get("min_trade_notional", 5.0)) if "min_trade_notional" in exe_cfg else 5.0,
            aggress_bps=float(exe_cfg.get("aggress_bps", 15.0)) if "aggress_bps" in exe_cfg else 15.0,
            cancel_all_on_start=True,
        )
        self.orders = OrderManager(cfg=om_cfg, execution_api=self.exec)

        # risk + health
        self.risk_state = RiskState(
            config=KillSwitchConfig(
                max_intraday_drawdown_pct=float(risk.get("max_intraday_drawdown_pct", 0.03)),
                cooldown_minutes=int(risk.get("cooldown_minutes", 720)),
                z_emergency_flat=float(risk.get("z_emergency_flat", 3.5)),
            )
        )
        self.health = HealthMonitor(cfg=HealthConfig())

        # runtime state
        self._stop = asyncio.Event()
        self._latest_mids: Dict[str, float] = {}
        self._symbols: List[str] = []
        self._buckets: Dict[str, List[str]] = {}

        self.features_store = LiveFeatureStore(tf_minutes=self.tf_minutes, ewma_lam=0.94)

    # -------------------------
    # Bootstrap: symbols/buckets
    # -------------------------
    async def _load_universe_and_buckets(self) -> None:
        """
        For now: expects buckets pre-defined in config (strategy_stat_arb.yaml or computed offline).
        Later, we can compute live clustering after warmup.
        """
        uni = (self.cfg_raw.get("universe", {}) or {})
        target = int(uni.get("target_size", 30))

        # In a minimal live runner, we read symbols from buckets in cfg if present.
        buckets_cfg = (self.cfg_raw.get("universe", {}) or {}).get("buckets_live")
        strategy_buckets = None
        if buckets_cfg and isinstance(buckets_cfg, dict):
            strategy_buckets = buckets_cfg

        # Preferred: if you precomputed buckets and stored them in cfg under universe.buckets_live
        if strategy_buckets is not None:
            self._buckets = {str(k): list(v) for k, v in strategy_buckets.items()}
            self._symbols = sorted({s for v in self._buckets.values() for s in v})
            return

        # Fallback: single bucket from top mids snapshot (not ideal but runnable)
        mids = await self.market.all_mids()
        data = mids.get("mids") or mids.get("data") or mids
        if isinstance(data, dict):
            # keys are coins, values are strings
            coins = sorted(list(data.keys()))
        else:
            coins = []

        self._symbols = coins[:target]
        self._buckets = {"bucket_0": list(self._symbols)}

    # -------------------------
    # WS callbacks
    # -------------------------
    def _on_all_mids(self, msg: Json) -> None:
        """
        WS allMids:
          {"channel":"allMids","data":{"mids":{...}}}
        """
        self.health.on_ws_message()
        data = msg.get("data") or {}
        mids = data.get("mids") if isinstance(data, dict) else None
        if isinstance(mids, dict):
            for k, v in mids.items():
                try:
                    px = float(v)
                    if px > 0:
                        self._latest_mids[str(k)] = px
                except Exception:
                    continue

    def _on_user_events(self, msg: Json) -> None:
        """
        WS userEvents can include fills/funding/etc.
        For now: we only mark WS as alive; detailed fill tracking comes later.
        """
        self.health.on_ws_message()

    # -------------------------
    # Periodic REST refreshers
    # -------------------------
    async def _poll_all_mids(self) -> None:
        while not self._stop.is_set():
            try:
                res = await self.market.all_mids()
                d = res.get("mids") or (res.get("data") or {}).get("mids") or res.get("data") or res
                if isinstance(d, dict):
                    for k, v in d.items():
                        try:
                            px = float(v)
                            if px > 0:
                                self._latest_mids[str(k)] = px
                        except Exception:
                            continue
            except Exception as e:
                self.log.warning("poll all_mids failed: %s", e)

            await asyncio.sleep(float((self.cfg_raw.get("live", {}) or {}).get("poll_mids_every_s", 5.0)))

    async def _poll_meta_and_ctx(self) -> None:
        """
        Optional injection of liquidity proxies (dv) from metaAndAssetCtxs.
        """
        while not self._stop.is_set():
            try:
                res = await self.market.meta_and_asset_ctxs()
                # Expected: {"universe":[...], "assetCtxs":[...]} (or nested)
                universe = res.get("universe") or (res.get("data") or {}).get("universe")
                ctxs = res.get("assetCtxs") or (res.get("data") or {}).get("assetCtxs")
                if isinstance(universe, list) and isinstance(ctxs, list) and len(universe) == len(ctxs):
                    dv_by_symbol: Dict[str, float] = {}
                    for u, c in zip(universe, ctxs):
                        coin = u.get("name")
                        if not coin:
                            continue
                        # best-effort: dayNtlVlm is a common field (string)
                        day = c.get("dayNtlVlm") or c.get("dayNtlVol") or c.get("dayVolume")
                        if day is not None:
                            try:
                                dv_by_symbol[str(coin)] = float(day)
                            except Exception:
                                pass
                    if dv_by_symbol:
                        self.features_store.inject_dv(dv_by_symbol)
            except Exception as e:
                self.log.warning("poll metaAndAssetCtxs failed: %s", e)

            await asyncio.sleep(float((self.cfg_raw.get("live", {}) or {}).get("poll_meta_ctx_every_s", 60.0)))

    async def _poll_predicted_funding(self) -> None:
        """
        Funding overlay needs periodic observations. We use predictedFundings as a proxy signal.
        If you want *real* realized funding, we will add a handler for userFundings ledger later.
        """
        while not self._stop.is_set():
            try:
                res = await self.funding.predicted_fundings()
                # predictedFundings often returns list entries; we parse best-effort
                rates: Dict[str, float] = {}
                data = res.get("fundings") or res.get("data") or res
                if isinstance(data, list):
                    for item in data:
                        try:
                            coin = item.get("coin") or item.get("name")
                            fr = item.get("fundingRate") or item.get("funding") or item.get("rate")
                            if coin and fr is not None:
                                rates[str(coin)] = float(fr)
                        except Exception:
                            continue
                if rates:
                    self.funding_overlay.update(rates)
            except Exception as e:
                self.log.warning("poll predicted_fundings failed: %s", e)

            await asyncio.sleep(float((self.cfg_raw.get("live", {}) or {}).get("poll_predicted_funding_every_s", 300.0)))

    # -------------------------
    # Main event loop
    # -------------------------
    async def run(self) -> None:
        """
        Starts WS + periodic polls + main rebalance loop.
        """
        await self._load_universe_and_buckets()
        self.log.info("LiveRunner mode=%s timeframe=%s symbols=%d buckets=%d",
                      self.mode, self.timeframe, len(self._symbols), len(self._buckets))

        # start WS and subscribe
        await self.ws.start()
        await self.market.stream_all_mids(cb=self._on_all_mids)
        await self.market.ws_subscribe({"type": "userEvents", "user": self.user}, cb=self._on_user_events)

        # initial reconcile (positions/open orders)
        await self.orders.reconcile(user=self.user, force=True)
        if self.orders.cfg.cancel_all_on_start:
            await self.orders.cancel_all_open_orders()
            await self.orders.reconcile(user=self.user, force=True)

        # background tasks
        tasks = []
        tasks.append(asyncio.create_task(self.health.watchdog(self.log, self._stop)))
        tasks.append(asyncio.create_task(self._poll_all_mids()))
        tasks.append(asyncio.create_task(self._poll_meta_and_ctx()))
        tasks.append(asyncio.create_task(self._poll_predicted_funding()))

        # warm-up: wait for some mids
        t0 = time.monotonic()
        while not self._latest_mids and (time.monotonic() - t0) < 10.0:
            await asyncio.sleep(0.2)

        # main rebalance aligned to timeframe boundaries
        try:
            while not self._stop.is_set():
                self.health.on_loop_tick()

                now = _utcnow()
                next_bar = _ceil_time_to_tf(now, self.tf_minutes)
                sleep_s = max(0.0, (next_bar - now).total_seconds())
                await asyncio.sleep(sleep_s + 0.05)  # small buffer

                ts = _utcnow()
                ts_ms = int(ts.timestamp() * 1000)

                # refresh state periodically
                await self.orders.reconcile(user=self.user, force=False)

                # update live features from latest mids (one update per bar)
                mids = {s: self._latest_mids.get(s) for s in self._symbols if self._latest_mids.get(s) is not None}
                mids = {s: float(px) for s, px in mids.items() if px is not None and px > 0}

                if len(mids) < max(5, int(0.5 * len(self._symbols))):
                    self.log.warning("Not enough mids (%d/%d); skipping bar", len(mids), len(self._symbols))
                    continue

                self.features_store.update_prices(mids)
                features = self.features_store.features(self._symbols)

                # compute signal/regime/target weights
                signal = self.stat_arb.update(ts, mids, self._buckets)
                regime = self.regime.update(ts, signal, self._buckets, features)

                # Update kill-switch on equity
                if np.isfinite(self.orders.equity) and self.orders.equity > 0:
                    self.risk_state.on_equity(ts, float(self.orders.equity))

                can_trade = self.risk_state.trading_allowed(ts)
                if not can_trade:
                    self.log.warning("Kill-switch cooldown active (%d min left). Forcing flat.",
                                     self.risk_state.cooldown_remaining_minutes(ts))
                    target_weights = {s: 0.0 for s in self._symbols}
                else:
                    # betas not yet wired in live (can be added using rolling OLS over live returns)
                    target = self.allocator.allocate(
                        ts=ts,
                        signal=signal,
                        regime=regime,
                        buckets=self._buckets,
                        features=features,
                        betas=None,
                    )
                    target_weights = target.weights

                # build + execute intents
                intents = self.orders.build_intents(user=self.user, ts_ms=ts_ms, target_weights=target_weights, mids=mids)
                if intents:
                    self.log.info("Bar %s intents=%d gross≈%.2f equity≈%.2f q=%.2f",
                                  ts.isoformat(), len(intents),
                                  sum(abs(target_weights.get(s, 0.0)) for s in target_weights),
                                  self.orders.equity,
                                  float(regime.q_total))
                await self.orders.execute_intents(intents)

        finally:
            self._stop.set()
            for t in tasks:
                t.cancel()
            await self.ws.stop()
            await self.rest.aclose()

    def stop(self) -> None:
        self._stop.set()
