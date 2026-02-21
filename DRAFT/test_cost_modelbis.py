# tests/test_cost_model.py
from __future__ import annotations

import math

from hyperstat.backtest.costs import CostModel, CostConfig, FeeConfig, SlippageConfig


def test_costs_non_negative():
    """
    Le modèle de coûts ne doit jamais retourner des valeurs négatives.
    """
    cm = CostModel(
        CostConfig(
            fees=FeeConfig(taker_bps=6.0, maker_bps=2.0),
            slippage=SlippageConfig(base_bps=8.0, k_bps_per_1pct_rv1h=10.0, cap_bps=200.0),
            mode="taker",
        )
    )
    fee, slip, slip_bps = cm.estimate(traded_notional=1000.0, rv_1h=0.02)
    assert fee >= 0.0
    assert slip >= 0.0
    assert slip_bps >= 0.0


def test_fee_formula_taker_vs_maker():
    """
    fee = |notional| * fee_bps / 1e4
    et maker < taker si bps maker < bps taker.
    """
    n = 12_345.0

    cm_taker = CostModel(CostConfig(fees=FeeConfig(6.0, 2.0), slippage=SlippageConfig(), mode="taker"))
    fee_t, _, _ = cm_taker.estimate(traded_notional=n, rv_1h=0.01)
    assert abs(fee_t - (abs(n) * 6.0 / 1e4)) < 1e-12

    cm_maker = CostModel(CostConfig(fees=FeeConfig(6.0, 2.0), slippage=SlippageConfig(), mode="maker"))
    fee_m, _, _ = cm_maker.estimate(traded_notional=n, rv_1h=0.01)
    assert abs(fee_m - (abs(n) * 2.0 / 1e4)) < 1e-12
    assert fee_m < fee_t


def test_slippage_increases_with_rv_and_caps():
    """
    slip_bps = base_bps + k * (rv_1h / 0.01) capé à cap_bps.
    """
    cfg = CostConfig(
        fees=FeeConfig(6.0, 2.0),
        slippage=SlippageConfig(base_bps=8.0, k_bps_per_1pct_rv1h=10.0, cap_bps=50.0),
        mode="taker",
    )
    cm = CostModel(cfg)

    # rv_1h faible
    _, _, bps_low = cm.estimate(traded_notional=10_000.0, rv_1h=0.005)  # 0.5%
    expected_low = 8.0 + 10.0 * (0.005 / 0.01)  # 8 + 5 = 13
    assert abs(bps_low - expected_low) < 1e-12

    # rv_1h plus forte -> bps plus grand
    _, _, bps_high = cm.estimate(traded_notional=10_000.0, rv_1h=0.03)  # 3%
    expected_high = 8.0 + 10.0 * (0.03 / 0.01)  # 8 + 30 = 38
    assert abs(bps_high - expected_high) < 1e-12
    assert bps_high > bps_low

    # rv_1h énorme -> cap
    _, _, bps_cap = cm.estimate(traded_notional=10_000.0, rv_1h=0.50)  # 50%
    assert bps_cap == 50.0  # cap_bps
