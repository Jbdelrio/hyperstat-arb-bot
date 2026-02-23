"""
PATCH — backtest/engine_funding_fix.py
========================================
Corrige le Bug #2 : funding_rates non transmis à l'allocator dans engine.py.

Comment l'appliquer :
    Dans src/hyperstat/backtest/engine.py, remplacer la ligne ~210 :
    
    AVANT (BUG) :
        target = self.allocator.allocate(
            ts=bar_ts, signal=signal, regime=regime,
        )
    
    APRÈS (FIX) :
        funding_at_ts = self.funding_events.get(bar_ts, {})
        target = self.allocator.allocate(
            ts=bar_ts,
            signal=signal,
            regime=regime,
            funding_rates=funding_at_ts or None,
        )

    Et dans la méthode `_build_funding_events`, s'assurer que
    funding_by_symbol est correctement indexé par timestamp.

Ce fichier contient également le mixin EngineV2Mixin qui intègre
le SupervisorAgent dans la boucle de backtest.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MIXIN : ajoute le SupervisorAgent à la boucle de backtest existante
# ─────────────────────────────────────────────────────────────────────────────

class EngineV2Mixin:
    """
    Mixin à ajouter à la classe BacktestEngine existante pour intégrer
    le SupervisorAgent sans modifier le code v1.

    Usage dans engine.py :
        from hyperstat.backtest.engine_funding_fix import EngineV2Mixin

        class BacktestEngine(EngineV2Mixin, _BacktestEngineBase):
            pass

    Le mixin surcharge `_process_bar` pour :
        1. Passer les funding_rates à l'allocator (Bug #2)
        2. Appliquer le scale_factor du SupervisorAgent
        3. Respecter le kill-switch
    """

    # Ces attributs sont définis sur la classe parente
    supervisor: Optional[Any] = None

    def _process_bar_v2(
        self,
        bar_ts    : datetime,
        signal    : Any,
        regime    : Any,
        bar_data  : Dict[str, Any],
    ) -> Any:
        """
        Version améliorée de _process_bar avec :
            - funding_rates passé à l'allocator (Bug #2 fix)
            - supervision multi-agents
        """
        # ── Bug #2 Fix : extraire funding_rates ──
        funding_at_ts = {}
        if hasattr(self, "funding_events") and self.funding_events:
            funding_at_ts = self.funding_events.get(bar_ts, {})

        # ── Appel allocator avec funding_rates ──
        target = self.allocator.allocate(
            ts            = bar_ts,
            signal        = signal,
            regime        = regime,
            funding_rates = funding_at_ts or None,
        )

        # ── SupervisorAgent (si activé) ──
        if self.supervisor is not None:
            try:
                # Observer les données du tick
                self.supervisor.observe(bar_ts, {
                    "btc_return"     : bar_data.get("btc_return", 0.0),
                    "liq_total_usd"  : bar_data.get("liq_total_usd", 0.0),
                    "avg_funding"    : _mean_funding(funding_at_ts),
                    "momentum_zscore": bar_data.get("momentum_zscore", 0.0),
                })
                # Décision du superviseur
                decision = self.supervisor.decide(bar_ts)

                if decision.kill_switch:
                    logger.warning(f"[Engine v2] Kill-switch superviseur à {bar_ts}")
                    target = _zero_weights(target)
                elif decision.scale_factor < 1.0:
                    target = _scale_weights(target, decision.scale_factor)

                # Qt override
                if decision.qt_override is not None and hasattr(regime, "qt"):
                    regime.qt = decision.qt_override

            except Exception as exc:
                logger.error(f"[Engine v2] Erreur superviseur: {exc}")

        return target


def _mean_funding(funding_dict: Dict[str, float]) -> float:
    if not funding_dict:
        return 0.0
    vals = list(funding_dict.values())
    return sum(vals) / len(vals)


def _zero_weights(weights: Any) -> Any:
    """Met tous les poids à 0 (flatten)."""
    if hasattr(weights, "weights"):
        weights.weights = {k: 0.0 for k in weights.weights}
    return weights


def _scale_weights(weights: Any, scale: float) -> Any:
    """Applique un facteur de réduction sur tous les poids."""
    if hasattr(weights, "weights"):
        weights.weights = {k: v * scale for k, v in weights.weights.items()}
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# HELPER : Construction du funding_events index
# ─────────────────────────────────────────────────────────────────────────────

def build_funding_events(
    funding_by_symbol: Dict[str, Any],
    candle_timestamps,
) -> Dict[datetime, Dict[str, float]]:
    """
    Construit un dictionnaire {timestamp: {symbol: funding_rate}}
    pour un accès O(1) dans la boucle de backtest.

    Corrige le Bug #2 : sans ce mapping, les funding_rates n'atteignent
    jamais l'allocator et le FDS est silencieusement désactivé.

    Parameters
    ----------
    funding_by_symbol : {symbol: DataFrame avec colonnes ts/rate}
    candle_timestamps : liste de datetime (index de la boucle backtest)

    Returns
    -------
    {datetime: {symbol: float}} — prêt pour engine.funding_events
    """
    import pandas as pd

    events: Dict[datetime, Dict[str, float]] = {}

    for sym, df in funding_by_symbol.items():
        if df is None or df.empty:
            continue
        # Normalise l'index
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ["ts", "timestamp", "time"]:
                if col in df.columns:
                    df = df.set_index(col)
                    break
        df.index = pd.to_datetime(df.index)

        for ts in df.index:
            if ts not in events:
                events[ts] = {}
            col = "rate" if "rate" in df.columns else df.columns[0]
            events[ts][sym] = float(df.loc[ts, col])

    logger.info(f"[build_funding_events] {len(events)} timestamps indexés pour {len(funding_by_symbol)} symboles")
    return events
