"""
apps/realtime_sim.py
======================
Simulation temps réel HyperStat v2.

Modes :
    replay  : rejoue les données historiques à vitesse accélérée (x1 à x10000)
    live    : données Hyperliquid en temps réel (paper trading)

Usage :
    python apps/realtime_sim.py --mode replay --speed 100 --days 30
    python apps/realtime_sim.py --mode live   --config configs/hyperliquid_testnet.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

class SimState:
    """État courant de la simulation."""

    def __init__(self, initial_equity: float = 10_000.0):
        self.equity           = initial_equity
        self.initial_equity   = initial_equity
        self.positions        : Dict[str, float] = {}   # {symbol: notional_usd}
        self.pnl_history      : List[dict]        = []
        self.trades_history   : List[dict]        = []
        self.current_ts       : Optional[datetime] = None
        self.bar_count        = 0
        self.start_ts         : Optional[datetime] = None

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_equity

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self.initial_equity * 100

    @property
    def max_drawdown(self) -> float:
        if not self.pnl_history:
            return 0.0
        equities = [p["equity"] for p in self.pnl_history]
        peak     = equities[0]
        max_dd   = 0.0
        for eq in equities:
            peak   = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak)
        return max_dd

    def record_bar(self, ts: datetime, prices: Dict[str, float]):
        self.current_ts = ts
        self.bar_count += 1
        if self.start_ts is None:
            self.start_ts = ts
        # Mark-to-market simple
        unrealized = sum(
            pos * (prices.get(sym, 1.0) / 1.0 - 1.0)  # simplifié
            for sym, pos in self.positions.items()
        )
        self.pnl_history.append({
            "ts"         : ts,
            "equity"     : self.equity + unrealized,
            "n_positions": len(self.positions),
        })
        # Garde 10k points
        if len(self.pnl_history) > 10_000:
            self.pnl_history.pop(0)

    def to_summary_dict(self) -> dict:
        elapsed = ""
        if self.start_ts and self.current_ts:
            delta = self.current_ts - self.start_ts
            elapsed = str(delta).split(".")[0]
        return {
            "ts"             : str(self.current_ts),
            "bars_processed" : self.bar_count,
            "elapsed_simtime": elapsed,
            "equity"         : round(self.equity, 2),
            "total_pnl"      : round(self.total_pnl, 2),
            "total_pnl_pct"  : round(self.total_pnl_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "n_positions"    : len(self.positions),
        }


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY MODE
# ─────────────────────────────────────────────────────────────────────────────

class ReplaySimulator:
    """
    Rejoue les données historiques bar-par-bar.
    Simule tous les agents avec les données passées.
    """

    def __init__(
        self,
        candles_by_symbol : Dict[str, pd.DataFrame],
        speed_multiplier  : int   = 100,
        initial_equity    : float = 10_000.0,
        print_every       : int   = 100,
    ):
        self.candles      = candles_by_symbol
        self.speed        = speed_multiplier
        self.state        = SimState(initial_equity)
        self.print_every  = print_every
        self._is_running  = False

        # Aligner tous les symboles sur le même index temporel
        symbols   = list(candles_by_symbol.keys())
        ref_idx   = candles_by_symbol[symbols[0]].index
        for sym in symbols[1:]:
            ref_idx = ref_idx.intersection(candles_by_symbol[sym].index)
        self.timestamps = sorted(ref_idx)
        logger.info(f"[ReplaySim] {len(self.timestamps)} barres alignées sur {len(symbols)} symboles")

    def run(
        self,
        days          : int  = 30,
        start_offset  : int  = 0,   # barres à sauter au début (warm-up)
    ):
        """Lance la simulation en mode synchrone."""
        self._is_running = True
        bars_total = min(days * 288, len(self.timestamps) - start_offset)
        start_idx  = max(0, len(self.timestamps) - days * 288 - start_offset)
        end_idx    = start_idx + bars_total

        ticks = self.timestamps[start_idx:end_idx]
        logger.info(
            f"[ReplaySim] Démarrage replay: {ticks[0]} → {ticks[-1]} "
            f"({len(ticks)} barres, vitesse ×{self.speed})"
        )

        bar_interval_real = (5 * 60) / self.speed  # secondes entre barres

        for i, ts in enumerate(ticks):
            if not self._is_running:
                break

            # Snapshot des prix courants
            prices = {
                sym: float(df.loc[ts, "close"])
                for sym, df in self.candles.items()
                if ts in df.index
            }

            # Enregistrement
            self.state.record_bar(ts, prices)

            # Affichage progression
            if i % self.print_every == 0 or i == len(ticks) - 1:
                self._print_progress(i, len(ticks), ts, prices)

            # Attente pour simuler la vitesse
            if self.speed < 1_000_000:
                time.sleep(max(0, bar_interval_real))

        self._is_running = False
        self._print_final_summary()

    def stop(self):
        self._is_running = False

    def _print_progress(self, i: int, total: int, ts: datetime, prices: dict):
        pct  = 100 * i / total
        summ = self.state.to_summary_dict()
        logger.info(
            f"[{pct:5.1f}%] {ts.strftime('%Y-%m-%d %H:%M')} | "
            f"Equity={summ['equity']:,.0f}$ | "
            f"PnL={summ['total_pnl']:+.1f}$ ({summ['total_pnl_pct']:+.2f}%) | "
            f"MaxDD={summ['max_drawdown_pct']:.2f}%"
        )

    def _print_final_summary(self):
        summ = self.state.to_summary_dict()
        logger.info("\n" + "="*60)
        logger.info("RÉSUMÉ SIMULATION")
        logger.info("="*60)
        logger.info(f"  Période simulée   : {self.state.start_ts} → {self.state.current_ts}")
        logger.info(f"  Barres traitées   : {summ['bars_processed']}")
        logger.info(f"  Equity finale     : {summ['equity']:,.2f}$")
        logger.info(f"  PnL total         : {summ['total_pnl']:+,.2f}$ ({summ['total_pnl_pct']:+.2f}%)")
        logger.info(f"  Max Drawdown      : {summ['max_drawdown_pct']:.2f}%")
        logger.info("="*60)

    def get_equity_curve(self) -> pd.DataFrame:
        """Retourne la courbe d'equity pour le dashboard."""
        return pd.DataFrame(self.state.pnl_history).set_index("ts")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def load_candles_from_dir(data_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Charge les candles Parquet depuis un dossier."""
    result = {}
    data_path = Path(data_dir)
    for sym in symbols:
        candidates = list(data_path.glob(f"{sym}*.parquet")) + list(data_path.glob(f"{sym}*.csv"))
        if not candidates:
            logger.warning(f"  Pas de données pour {sym} dans {data_dir}")
            continue
        fp = candidates[0]
        df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)
        for col in ["ts", "timestamp", "time", "open_time"]:
            if col in df.columns:
                df["ts"] = pd.to_datetime(df[col])
                break
        df = df.set_index("ts").sort_index()
        result[sym] = df
        logger.info(f"  ✅ {sym}: {len(df)} barres ({df.index[0]} → {df.index[-1]})")
    return result


def main():
    parser = argparse.ArgumentParser(description="HyperStat Realtime Simulator")
    parser.add_argument("--mode",    choices=["replay", "live"], default="replay")
    parser.add_argument("--speed",   type=int,   default=100,
                        help="Vitesse de replay (100 = 100× temps réel)")
    parser.add_argument("--days",    type=int,   default=30,
                        help="Nombre de jours à rejouer")
    parser.add_argument("--data",    default="./artifacts/data/candles",
                        help="Dossier des données Parquet")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTC", "ETH", "SOL", "AVAX", "LINK"],
                        help="Symboles à simuler")
    parser.add_argument("--equity",  type=float, default=10_000.0,
                        help="Capital initial en USD")
    parser.add_argument("--config",  default=None,
                        help="Fichier de config YAML (pour mode live)")
    parser.add_argument("--output",  default=None,
                        help="Sauvegarder les résultats en CSV")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("HyperStat v2 — Simulation Temps Réel")
    logger.info(f"Mode: {args.mode.upper()}  |  Vitesse: ×{args.speed}  |  Capital: {args.equity:,.0f}$")
    logger.info("="*60)

    if args.mode == "replay":
        logger.info(f"\nChargement des données depuis {args.data}...")
        candles = load_candles_from_dir(args.data, args.symbols)

        if not candles:
            logger.error("Aucune donnée chargée. Vérifiez --data et --symbols")
            return

        sim = ReplaySimulator(
            candles_by_symbol = candles,
            speed_multiplier  = args.speed,
            initial_equity    = args.equity,
        )
        sim.run(days=args.days)

        if args.output:
            df = sim.get_equity_curve()
            df.to_csv(args.output)
            logger.info(f"\n✅ Courbe equity sauvegardée: {args.output}")

    elif args.mode == "live":
        logger.info("Mode live non encore implémenté dans ce fichier.")
        logger.info("Utiliser : python -m hyperstat.main --mode paper")


if __name__ == "__main__":
    main()
