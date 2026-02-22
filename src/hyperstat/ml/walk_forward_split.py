"""
HyperStat v2 — Walk-Forward Split Engine
Gestion rigoureuse du split temporel train / val / test + simulation temps-réel

Principe :
  - Les données sont divisées en 3 fenêtres non-chevauchantes
  - Le modèle est entraîné UNIQUEMENT sur Train, validé sur Val, évalué sur Test
  - Aucune information du Test ne filtre vers Train/Val (pas de lookahead)
  - La "simulation temps-réel" rejoue le Test set barre par barre

Schéma temporel :
  |←── Train 70% ──→|←── Val 15% ──→|←── Test 15% ──→|
                                                       ↑
                                              simulation live ici

Walk-forward :
  Toutes les retrain_every_bars, la fenêtre glisse :
  |   Train+Val ──────────────────────────────→|← Test (live) →|
  |         [retrain]        [retrain]           ↑ now

Auteur : HyperStat v2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration split
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitConfig:
    """
    Paramètres du split temporel.

    Les fractions doivent sommer à 1.0.
    train_frac + val_frac + test_frac == 1.0
    """
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # Walk-forward : retrain toutes les N barres
    retrain_every_bars: int = 336   # ~2 semaines en 1h
    min_train_bars: int = 2000      # minimum pour entraîner LSTM

    # Buffer entre fenêtres pour éviter la contamination
    # (le modèle peut connaître des patterns des dernières barres train)
    gap_bars: int = 0

    def validate(self):
        total = self.train_frac + self.val_frac + self.test_frac
        assert abs(total - 1.0) < 1e-6, f"Fractions doivent sommer à 1.0 (actuel: {total})"
        assert self.train_frac > 0.3, "train_frac trop faible"
        assert self.val_frac > 0.05, "val_frac trop faible"
        assert self.test_frac > 0.05, "test_frac trop faible"


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale : DataSplitter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitResult:
    """Résultat d'un split sur l'index temporel."""
    all_timestamps: pd.DatetimeIndex

    train_start: pd.Timestamp
    train_end: pd.Timestamp

    val_start: pd.Timestamp
    val_end: pd.Timestamp

    test_start: pd.Timestamp
    test_end: pd.Timestamp

    n_train: int
    n_val: int
    n_test: int

    def summary(self) -> str:
        return (
            f"Split temporel :\n"
            f"  Train  [{self.train_start.date()} → {self.train_end.date()}]"
            f"  {self.n_train:,} barres  ({100*self.n_train/len(self.all_timestamps):.1f}%)\n"
            f"  Val    [{self.val_start.date()} → {self.val_end.date()}]"
            f"  {self.n_val:,} barres  ({100*self.n_val/len(self.all_timestamps):.1f}%)\n"
            f"  Test   [{self.test_start.date()} → {self.test_end.date()}]"
            f"  {self.n_test:,} barres  ({100*self.n_test/len(self.all_timestamps):.1f}%)\n"
        )

    @property
    def train_idx(self) -> pd.DatetimeIndex:
        return self.all_timestamps[
            (self.all_timestamps >= self.train_start) &
            (self.all_timestamps <= self.train_end)
        ]

    @property
    def val_idx(self) -> pd.DatetimeIndex:
        return self.all_timestamps[
            (self.all_timestamps >= self.val_start) &
            (self.all_timestamps <= self.val_end)
        ]

    @property
    def test_idx(self) -> pd.DatetimeIndex:
        return self.all_timestamps[
            (self.all_timestamps >= self.test_start) &
            (self.all_timestamps <= self.test_end)
        ]


class DataSplitter:
    """
    Calcule et gère les splits temporels train/val/test.

    Usage basique :
        splitter = DataSplitter(SplitConfig())
        split = splitter.compute_split(timestamps)
        print(split.summary())

        candles_train = candles[candles.index.isin(split.train_idx)]
        candles_val   = candles[candles.index.isin(split.val_idx)]
        candles_test  = candles[candles.index.isin(split.test_idx)]

    Usage walk-forward (simulation temps-réel) :
        for ts, train_mask, val_mask in splitter.walk_forward(timestamps):
            model.fit(data[train_mask], data[val_mask])
            signal = model.predict(data[ts])
    """

    def __init__(self, cfg: Optional[SplitConfig] = None):
        self.cfg = cfg or SplitConfig()
        self.cfg.validate()

    def compute_split(
        self,
        candles_by_symbol: Dict[str, pd.DataFrame],
    ) -> SplitResult:
        """
        Calcule le split sur l'union des timestamps de tous les symboles.
        Le split est identique pour tous les symboles (alignement temporel strict).
        """
        # Union des timestamps
        all_ts = pd.DatetimeIndex(
            sorted(set().union(*[df.index for df in candles_by_symbol.values()]))
        )
        return self._split_index(all_ts)

    def _split_index(self, idx: pd.DatetimeIndex) -> SplitResult:
        n = len(idx)
        cfg = self.cfg
        n_train = int(n * cfg.train_frac)
        n_val   = int(n * cfg.val_frac)
        n_test  = n - n_train - n_val

        assert n_train >= cfg.min_train_bars, (
            f"Pas assez de données pour l'entraînement : {n_train} < {cfg.min_train_bars}. "
            f"Télécharger plus d'historique."
        )

        train_end_i = n_train - 1
        val_start_i = train_end_i + 1 + cfg.gap_bars
        val_end_i   = val_start_i + n_val - 1
        test_start_i = val_end_i + 1 + cfg.gap_bars
        test_end_i   = n - 1

        result = SplitResult(
            all_timestamps=idx,
            train_start=idx[0],
            train_end=idx[train_end_i],
            val_start=idx[val_start_i],
            val_end=idx[val_end_i],
            test_start=idx[test_start_i],
            test_end=idx[test_end_i],
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
        logger.info(result.summary())
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Walk-forward
    # ──────────────────────────────────────────────────────────────────────

    def walk_forward_schedule(
        self,
        split: SplitResult,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Génère le calendrier de retraining walk-forward sur la période Test.

        Retourne : liste de (retrain_at, train_end, val_end)
          - retrain_at  : timestamp auquel on déclenche un retrain
          - train_end   : fin de la fenêtre train pour ce retrain
          - val_end     : fin de la fenêtre val pour ce retrain

        Utilisation dans le backtest engine :
            schedule = splitter.walk_forward_schedule(split)
            for ts in test_timestamps:
                if ts in retrain_timestamps:
                    model.fit(data[:train_end], data[train_end:val_end])
                signal = model.predict(data[:ts])
        """
        cfg = self.cfg
        test_idx = split.test_idx
        if len(test_idx) == 0:
            return []

        # Premier retrain : au début du Test, entraîné sur Train+Val
        schedule = []
        retrain_points = test_idx[::cfg.retrain_every_bars]

        for rp in retrain_points:
            # La fenêtre de train se termine juste avant le point de retrain
            # (pas de données futures dans l'entraînement)
            available = split.all_timestamps[split.all_timestamps < rp]
            if len(available) < cfg.min_train_bars + 100:
                continue

            n_avail = len(available)
            val_size = max(int(n_avail * cfg.val_frac), 100)
            train_end = available[-(val_size + 1)]
            val_end   = available[-1]
            schedule.append((rp, train_end, val_end))

        logger.info(
            f"[WalkForward] {len(schedule)} retrains planifiés "
            f"tous les {cfg.retrain_every_bars} barres"
        )
        return schedule

    def slice_for_training(
        self,
        candles_by_symbol: Dict[str, pd.DataFrame],
        funding_by_symbol: Dict[str, pd.Series],
        train_end: pd.Timestamp,
        val_end: pd.Timestamp,
    ) -> Tuple[
        Dict[str, pd.DataFrame],  # candles_train
        Dict[str, pd.Series],     # funding_train
        Dict[str, pd.DataFrame],  # candles_val
        Dict[str, pd.Series],     # funding_val
    ]:
        """Découpe les données pour un retrain walk-forward."""
        candles_tr, candles_vl, fund_tr, fund_vl = {}, {}, {}, {}

        for sym, df in candles_by_symbol.items():
            candles_tr[sym] = df[df.index <= train_end]
            candles_vl[sym] = df[(df.index > train_end) & (df.index <= val_end)]

        for sym, s in funding_by_symbol.items():
            fund_tr[sym] = s[s.index <= train_end]
            fund_vl[sym] = s[(s.index > train_end) & (s.index <= val_end)]

        return candles_tr, fund_tr, candles_vl, fund_vl


# ─────────────────────────────────────────────────────────────────────────────
# Simulation temps-réel (replay du Test set)
# ─────────────────────────────────────────────────────────────────────────────

class RealTimeSimulator:
    """
    Rejoue le Test set barre par barre, en respectant la causalité stricte.

    À chaque barre t :
      - Le modèle ne voit que les données jusqu'à t-1 (inclus)
      - Les retrain sont déclenchés selon le schedule walk-forward
      - Les métriques sont calculées sur les prédictions out-of-sample

    C'est l'équivalent d'une simulation de trading live, avec les données
    historiques mais avec les mêmes contraintes de timing qu'en production.

    Usage :
        sim = RealTimeSimulator(splitter, split, predictor)
        results = sim.run(candles_by_symbol, funding_by_symbol, ...)
    """

    def __init__(
        self,
        splitter: DataSplitter,
        split: SplitResult,
    ):
        self.splitter = splitter
        self.split = split
        self._retrain_schedule: Dict[pd.Timestamp, Tuple[pd.Timestamp, pd.Timestamp]] = {}

        # Construire le lookup rapide pour les retrains
        for rp, train_end, val_end in splitter.walk_forward_schedule(split):
            self._retrain_schedule[rp] = (train_end, val_end)

    def is_retrain_bar(self, ts: pd.Timestamp) -> bool:
        return ts in self._retrain_schedule

    def get_retrain_windows(
        self, ts: pd.Timestamp
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Retourne (train_end, val_end) si un retrain est planifié à ts."""
        return self._retrain_schedule.get(ts)

    def iter_test_bars(self) -> Iterator[pd.Timestamp]:
        """Itérateur sur les barres du Test set dans l'ordre chronologique."""
        for ts in self.split.test_idx:
            yield ts

    def get_available_data(
        self,
        ts: pd.Timestamp,
        candles_by_symbol: Dict[str, pd.DataFrame],
        funding_by_symbol: Dict[str, pd.Series],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Retourne uniquement les données disponibles jusqu'à ts-1 (strict).
        Simule ce qu'on aurait en production au moment de décider.
        """
        c_avail = {
            sym: df[df.index < ts]
            for sym, df in candles_by_symbol.items()
        }
        f_avail = {
            sym: s[s.index < ts]
            for sym, s in funding_by_symbol.items()
        }
        return c_avail, f_avail

    def compute_realtime_metrics(
        self,
        predictions: Dict[pd.Timestamp, Dict[str, float]],
        candles_by_symbol: Dict[str, pd.DataFrame],
        forward_horizon: int = 8,
    ) -> pd.DataFrame:
        """
        Calcule l'IC Spearman rolling entre prédictions et réalité.
        Uniquement sur la période Test (aucune contamination).

        Retourne un DataFrame avec :
          ts | ic_spearman | mean_pred | n_symbols
        """
        records = []
        for ts, scores in sorted(predictions.items()):
            future_ts_arr = []
            for sym, score in scores.items():
                candles = candles_by_symbol.get(sym)
                if candles is None:
                    continue
                # Retour réalisé dans forward_horizon barres
                future = candles[candles.index > ts].head(forward_horizon)
                if len(future) < forward_horizon:
                    continue
                fwd_ret = np.log(
                    float(future["close"].iloc[-1]) /
                    (float(candles[candles.index <= ts]["close"].iloc[-1]) + 1e-8)
                )
                future_ts_arr.append((sym, score, fwd_ret))

            if len(future_ts_arr) >= 3:
                preds_arr = np.array([x[1] for x in future_ts_arr])
                rets_arr  = np.array([x[2] for x in future_ts_arr])
                ic = float(
                    pd.Series(preds_arr).corr(pd.Series(rets_arr), method="spearman")
                ) if len(preds_arr) > 2 else 0.0
                records.append({
                    "ts": ts,
                    "ic_spearman": ic,
                    "mean_pred": float(preds_arr.mean()),
                    "n_symbols": len(future_ts_arr),
                })

        if not records:
            return pd.DataFrame(columns=["ts", "ic_spearman", "mean_pred", "n_symbols"])

        df = pd.DataFrame(records).set_index("ts")
        df["ic_rolling_30"] = df["ic_spearman"].rolling(30).mean()
        logger.info(
            f"[RealTimeSim] IC moyen = {df['ic_spearman'].mean():.4f}"
            f"  t-stat = {df['ic_spearman'].mean() / (df['ic_spearman'].std() / len(df)**0.5 + 1e-8):.2f}"
        )
        return df
