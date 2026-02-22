# src/hyperstat/backtest/costs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FeeModel:
    """
    Fees in basis points (bps) of notional traded.
    """
    taker_bps: float = 6.0
    maker_bps: float = 2.0
    mode: str = "taker"  # taker | maker

    def fee_bps(self) -> float:
        m = self.mode.lower()
        if m == "maker":
            return float(self.maker_bps)
        return float(self.taker_bps)


@dataclass(frozen=True)
class SlippageModel:
    """
    Slippage in bps:
      slip_bps = base_bps + k_bps_per_1pct_rv1h * (rv_1h * 100)

    where rv_1h is realized vol (fraction, e.g. 0.01 = 1%).
    """
    base_bps: float = 8.0
    k_bps_per_1pct_rv1h: float = 10.0
    cap_bps: float = 200.0  # avoid exploding during data issues

    def slippage_bps(self, rv_1h: float | None) -> float:
        if rv_1h is None or not np.isfinite(rv_1h):
            return float(self.base_bps)
        bps = float(self.base_bps + self.k_bps_per_1pct_rv1h * (float(rv_1h) * 100.0))
        return float(min(max(0.0, bps), self.cap_bps))


@dataclass(frozen=True)
class CostModel:
    fee: FeeModel
    slippage: SlippageModel

    def trade_cost(self, notional_abs: float, rv_1h: float | None = None) -> Tuple[float, float, float]:
        """
        Returns:
          total_cost, fee_cost, slip_cost
        """
        n = float(abs(notional_abs))
        fee_cost = n * (self.fee.fee_bps() / 1e4)
        slip_cost = n * (self.slippage.slippage_bps(rv_1h) / 1e4)
        return (fee_cost + slip_cost), fee_cost, slip_cost


@dataclass
class TradeCostBreakdown:
    """
    Per-step aggregated costs.
    """
    total: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0

    def add(self, total: float, fees: float, slippage: float) -> None:
        self.total += float(total)
        self.fees += float(fees)
        self.slippage += float(slippage)


def vwap_adjusted_slippage(
    rv_1h_pct: float,
    vwap_deviation_bps: float,
    order_size_pct_adv: float = 0.01,
) -> float:
    """
    Slippage ajusté VWAP — modèle plus conservateur que le proxy vol seul.

    Formule combinée :
        slip_bps = base_slip + vol_impact + market_impact + vwap_dev_penalty

    Composantes :
        base_slip        = 8 bps                                (spread bid-ask estimé)
        vol_impact       = 10 × rv_1h_pct                       (cohérent avec SlippageModel)
        market_impact    = 5 × sqrt(order_size_pct_adv) × 100   (modèle racine carrée)
        vwap_dev_penalty = 0.2 × vwap_deviation_bps             (exécution hors VWAP)

    Args:
        rv_1h_pct           : realized volatility 1h en % (ex : 1.2 pour 1.2%).
                              Note : SlippageModel.slippage_bps() prend rv_1h en fraction
                              et le multiplie par 100 en interne.
        vwap_deviation_bps  : |close - VWAP| / VWAP × 10 000.
        order_size_pct_adv  : fraction du volume journalier moyen traité.
                              Défaut 1% (ordres de taille modeste sur Hyperliquid).

    Returns:
        Slippage estimé en bps (float).

    Usage dans le backtest ::

        from hyperstat.backtest.costs import vwap_adjusted_slippage

        slip = vwap_adjusted_slippage(
            rv_1h_pct=rv * 100,            # convertir fraction → %
            vwap_deviation_bps=vwap_dev,
            order_size_pct_adv=notional / daily_volume,
        )
        total_cost = notional * (fee_bps + slip) / 1e4
    """
    base_slip = 8.0
    vol_impact = 10.0 * float(rv_1h_pct)
    market_impact = 5.0 * float(np.sqrt(max(0.0, order_size_pct_adv))) * 100.0
    vwap_dev_penalty = 0.2 * float(vwap_deviation_bps)
    return base_slip + vol_impact + market_impact + vwap_dev_penalty


def cost_model_from_config(cfg: dict) -> CostModel:
    exe = (cfg.get("execution", {}) or {})
    fees = (exe.get("fees", {}) or {})
    slip = (exe.get("slippage", {}) or {})

    fee_model = FeeModel(
        taker_bps=float(fees.get("taker_bps", 6.0)),
        maker_bps=float(fees.get("maker_bps", 2.0)),
        mode=str(exe.get("mode", "taker")),
    )
    slip_model = SlippageModel(
        base_bps=float(slip.get("base_bps", 8.0)),
        k_bps_per_1pct_rv1h=float(slip.get("k_bps_per_1pct_rv1h", 10.0)),
        cap_bps=float(slip.get("cap_bps", 200.0)),
    )
    return CostModel(fee=fee_model, slippage=slip_model)
