# tests/test_cost_model.py
from __future__ import annotations

from hyperstat.backtest.costs import CostModel, FeeModel, SlippageModel


def test_costs_non_negative():
    """Le modèle de coûts ne doit jamais retourner des valeurs négatives."""
    cm = CostModel(
        fee=FeeModel(taker_bps=6.0, maker_bps=2.0, mode="taker"),
        slippage=SlippageModel(base_bps=8.0, k_bps_per_1pct_rv1h=10.0, cap_bps=200.0),
    )
    total, fee, slip = cm.trade_cost(notional_abs=1000.0, rv_1h=0.02)
    assert total >= 0.0
    assert fee >= 0.0
    assert slip >= 0.0


def test_fee_formula_taker_vs_maker():
    """fee = |notional| * fee_bps / 1e4  et  maker < taker."""
    n = 12_345.0

    cm_taker = CostModel(
        fee=FeeModel(taker_bps=6.0, maker_bps=2.0, mode="taker"),
        slippage=SlippageModel(),
    )
    _, fee_t, _ = cm_taker.trade_cost(notional_abs=n, rv_1h=0.01)
    assert abs(fee_t - (n * 6.0 / 1e4)) < 1e-9

    cm_maker = CostModel(
        fee=FeeModel(taker_bps=6.0, maker_bps=2.0, mode="maker"),
        slippage=SlippageModel(),
    )
    _, fee_m, _ = cm_maker.trade_cost(notional_abs=n, rv_1h=0.01)
    assert abs(fee_m - (n * 2.0 / 1e4)) < 1e-9
    assert fee_m < fee_t


def test_slippage_increases_with_rv_and_caps():
    """slip_bps = base + k * (rv_1h * 100), capé à cap_bps."""
    cm = CostModel(
        fee=FeeModel(taker_bps=6.0, maker_bps=2.0, mode="taker"),
        slippage=SlippageModel(base_bps=8.0, k_bps_per_1pct_rv1h=10.0, cap_bps=50.0),
    )

    # rv_1h = 0.5% → slip_bps = 8 + 10*0.5 = 13
    _, _, slip_low = cm.trade_cost(notional_abs=10_000.0, rv_1h=0.005)
    expected_low = 10_000.0 * (8.0 + 10.0 * 0.5) / 1e4
    assert abs(slip_low - expected_low) < 1e-9

    # rv_1h = 3% → slip_bps = 8 + 10*3 = 38
    _, _, slip_high = cm.trade_cost(notional_abs=10_000.0, rv_1h=0.03)
    expected_high = 10_000.0 * (8.0 + 10.0 * 3.0) / 1e4
    assert abs(slip_high - expected_high) < 1e-9
    assert slip_high > slip_low

    # rv_1h = 50% → cap à 50 bps
    _, _, slip_cap = cm.trade_cost(notional_abs=10_000.0, rv_1h=0.50)
    assert abs(slip_cap - 10_000.0 * 50.0 / 1e4) < 1e-9


def test_zero_notional():
    """Notional zéro → tout à zéro."""
    cm = CostModel(fee=FeeModel(), slippage=SlippageModel())
    total, fee, slip = cm.trade_cost(notional_abs=0.0, rv_1h=0.01)
    assert total == 0.0
    assert fee == 0.0
    assert slip == 0.0
