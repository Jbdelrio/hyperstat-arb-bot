"""
Tests unitaires HyperStat v2
Couvrent : MLPredictor, DataSplitter, RealTimeSimulator, Orchestrator, SentimentAgent

Usage :
    pytest tests/test_v2_modules.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures communes
# ─────────────────────────────────────────────────────────────────────────────

def make_candles(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Génère des OHLCV synthétiques réalistes."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="1h")
    log_ret = rng.normal(0, 0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    vol = np.abs(rng.normal(1e6, 2e5, size=n))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": vol,
    }, index=times)


def make_funding(n: int = 3000, seed: int = 0) -> pd.Series:
    """Génère un funding rate synthétique (style Hyperliquid, mise à jour 8h)."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="1h")
    # Funding change tous les 8 bars, bruit faible
    raw = rng.normal(0.0001, 0.0003, size=n // 8 + 1)
    funding_vals = np.repeat(raw, 8)[:n]
    return pd.Series(funding_vals, index=times, name="rate")


def make_universe(n_symbols: int = 6, n_bars: int = 3000) -> Dict[str, pd.DataFrame]:
    return {f"COIN{i}": make_candles(n_bars, seed=i) for i in range(n_symbols)}


def make_funding_universe(n_symbols: int = 6, n_bars: int = 3000) -> Dict[str, pd.Series]:
    return {f"COIN{i}": make_funding(n_bars, seed=i) for i in range(n_symbols)}


# ─────────────────────────────────────────────────────────────────────────────
# Tests DataSplitter
# ─────────────────────────────────────────────────────────────────────────────

class TestDataSplitter:

    def test_split_fractions(self):
        from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig
        candles = make_universe()
        splitter = DataSplitter(SplitConfig(train_frac=0.70, val_frac=0.15, test_frac=0.15))
        split = splitter.compute_split(candles)

        total = split.n_train + split.n_val + split.n_test
        assert total <= len(split.all_timestamps)
        assert abs(split.n_train / total - 0.70) < 0.02
        assert abs(split.n_val   / total - 0.15) < 0.02
        assert abs(split.n_test  / total - 0.15) < 0.02

    def test_no_overlap(self):
        from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig
        candles = make_universe()
        splitter = DataSplitter(SplitConfig())
        split = splitter.compute_split(candles)

        # Les index ne se chevauchent pas
        train_set = set(split.train_idx)
        val_set   = set(split.val_idx)
        test_set  = set(split.test_idx)
        assert len(train_set & val_set) == 0, "Overlap train/val détecté"
        assert len(train_set & test_set) == 0, "Overlap train/test détecté"
        assert len(val_set & test_set) == 0, "Overlap val/test détecté"

    def test_chronological_order(self):
        from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig
        candles = make_universe()
        splitter = DataSplitter(SplitConfig())
        split = splitter.compute_split(candles)

        # L'ordre train < val < test doit être respecté
        assert split.train_end < split.val_start
        assert split.val_end < split.test_start

    def test_walk_forward_schedule(self):
        from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig
        candles = make_universe()
        cfg = SplitConfig(retrain_every_bars=100, min_train_bars=200)
        splitter = DataSplitter(cfg)
        split = splitter.compute_split(candles)
        schedule = splitter.walk_forward_schedule(split)

        assert len(schedule) > 0, "Le schedule walk-forward est vide"
        # Vérifier que les timestamps de retrain sont dans le Test
        for rp, train_end, val_end in schedule:
            assert rp >= split.test_start
            assert train_end < rp
            assert val_end < rp

    def test_insufficient_data_raises(self):
        from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig
        candles = make_universe(n_bars=100)  # trop peu
        splitter = DataSplitter(SplitConfig(min_train_bars=2000))
        with pytest.raises(AssertionError, match="Pas assez de données"):
            splitter.compute_split(candles)


# ─────────────────────────────────────────────────────────────────────────────
# Tests MLPredictor
# ─────────────────────────────────────────────────────────────────────────────

class TestMLPredictor:

    def test_build_sequence_features_shape(self):
        from hyperstat.ml.lstm_xgb_predictor import build_sequence_features, MLPredictorConfig
        candles = make_candles(500)
        funding = make_funding(500)
        cfg = MLPredictorConfig()
        df = build_sequence_features(candles, funding, cfg)

        assert len(df) == len(candles)
        assert df.shape[1] == cfg.lstm.n_features
        assert not df.isnull().any().any(), "Features contiennent des NaN"

    def test_build_sequence_no_lookahead(self):
        """Vérifier que les features à t ne dépendent pas des données après t."""
        from hyperstat.ml.lstm_xgb_predictor import build_sequence_features, MLPredictorConfig
        candles = make_candles(500)
        funding = make_funding(500)
        cfg = MLPredictorConfig()

        # Feature à t avec données jusqu'à t
        df_full = build_sequence_features(candles, funding, cfg)
        # Feature à t avec données tronquées à t
        df_trunc = build_sequence_features(candles.iloc[:400], funding.iloc[:400], cfg)

        # Les valeurs à t=399 doivent être identiques
        t_idx = candles.index[399]
        if t_idx in df_trunc.index:
            np.testing.assert_allclose(
                df_full.loc[t_idx].values,
                df_trunc.loc[t_idx].values,
                atol=1e-5,
                err_msg="Lookahead détecté dans build_sequence_features",
            )

    def test_fit_predict_xgb_only(self):
        """Test du pipeline XGB-only (sans torch) sur données synthétiques."""
        from hyperstat.ml.lstm_xgb_predictor import MLPredictor, MLPredictorConfig

        try:
            import xgboost  # skip si non installé
        except ImportError:
            pytest.skip("xgboost non installé")

        candles = make_universe(n_symbols=4, n_bars=600)
        funding = make_funding_universe(n_symbols=4, n_bars=600)

        cfg = MLPredictorConfig(fallback_to_xgb_only=True, forward_horizon=4)
        predictor = MLPredictor(cfg)
        predictor.fit(candles, funding)

        assert predictor.is_trained

        ts = list(candles["COIN0"].index)[-1]
        scores = predictor.predict(candles, funding, ts)
        assert set(scores.keys()) == set(candles.keys())
        for sym, score in scores.items():
            assert -1.0 <= score <= 1.0, f"Score hors [-1,1] pour {sym}: {score}"

    def test_low_ic_returns_zeros(self):
        """Si IC < seuil, le predictor doit retourner des scores nuls."""
        from hyperstat.ml.lstm_xgb_predictor import MLPredictor, MLPredictorConfig

        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost non installé")

        candles = make_universe(n_symbols=3, n_bars=500)
        funding = make_funding_universe(n_symbols=3, n_bars=500)

        cfg = MLPredictorConfig(
            fallback_to_xgb_only=True,
            min_ic_threshold=0.99,  # seuil impossible → toujours vide
        )
        predictor = MLPredictor(cfg)
        predictor.fit(candles, funding)

        ts = list(candles["COIN0"].index)[-1]
        scores = predictor.predict(candles, funding, ts)
        assert all(v == 0.0 for v in scores.values()), \
            "Les scores devraient être 0 quand IC < seuil"

    def test_save_load_roundtrip(self, tmp_path):
        """Test persistence : save() puis load() doit restaurer l'état."""
        from hyperstat.ml.lstm_xgb_predictor import MLPredictor, MLPredictorConfig

        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost non installé")

        candles = make_universe(n_symbols=3, n_bars=500)
        funding = make_funding_universe(n_symbols=3, n_bars=500)

        cfg = MLPredictorConfig(fallback_to_xgb_only=True, min_ic_threshold=0.0)
        predictor = MLPredictor(cfg)
        predictor.fit(candles, funding)

        model_path = tmp_path / "ml_model"
        predictor.save(model_path)
        loaded = MLPredictor.load(model_path)

        assert loaded.is_trained == predictor.is_trained
        assert abs(loaded.last_ic - predictor.last_ic) < 1e-6

        ts = list(candles["COIN0"].index)[-1]
        s1 = predictor.predict(candles, funding, ts)
        s2 = loaded.predict(candles, funding, ts)
        for sym in s1:
            assert abs(s1[sym] - s2[sym]) < 1e-4, f"Prédictions divergent pour {sym}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests Orchestrateur
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestrator:

    def test_combine_basic(self):
        from hyperstat.agents.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(enable_reflect=False))
        stat_signals = {"BTC": 0.5, "ETH": -0.3, "SOL": 0.1}
        ml_signals   = {"BTC": 0.4, "ETH": -0.2, "SOL": 0.3}

        result = orch.combine(
            ts=pd.Timestamp("2025-01-01"),
            stat_arb_signals=stat_signals,
            ml_signals=ml_signals,
            regime_q=1.0,
            q_break=1.0,
            sent_gate=1.0,
        )
        assert set(result.keys()) == {"BTC", "ETH", "SOL"}
        for sym, score in result.items():
            assert -1.0 <= score <= 1.0, f"Score hors [-1,1] : {sym}={score}"

    def test_shock_regime_returns_flat(self):
        from hyperstat.agents.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(enable_reflect=False))
        stat_signals = {"BTC": 0.9, "ETH": -0.8}
        ml_signals   = {"BTC": 0.7, "ETH": -0.6}

        result = orch.combine(
            ts=pd.Timestamp("2025-01-01"),
            stat_arb_signals=stat_signals,
            ml_signals=ml_signals,
            regime_q=0.0,    # choc
            q_break=0.0,
            sent_gate=1.0,
        )
        assert all(v == 0.0 for v in result.values()), \
            "En régime choc, tous les signaux doivent être 0"

    def test_sentiment_gate_reduces_exposure(self):
        from hyperstat.agents.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig(enable_reflect=False))
        stat_signals = {"BTC": 1.0}
        ml_signals   = {"BTC": 1.0}

        # Plein gate
        r_full = orch.combine(
            ts=pd.Timestamp("2025-01-01"),
            stat_arb_signals=stat_signals, ml_signals=ml_signals,
            regime_q=1.0, q_break=1.0, sent_gate=1.0,
        )
        # Gate réduit
        r_reduced = orch.combine(
            ts=pd.Timestamp("2025-01-01"),
            stat_arb_signals=stat_signals, ml_signals=ml_signals,
            regime_q=1.0, q_break=1.0, sent_gate=0.3,
        )
        assert r_full["BTC"] > r_reduced["BTC"], \
            "Un gate réduit doit diminuer l'exposition"

    def test_high_vol_shifts_weights(self):
        from hyperstat.agents.orchestrator import Orchestrator, OrchestratorConfig

        cfg = OrchestratorConfig(
            w_stat_default=0.80, w_ml_default=0.20,
            w_stat_high_vol=0.60, w_ml_high_vol=0.40,
            enable_reflect=False,
        )
        orch = Orchestrator(cfg)

        # Signal ML positif fort, signal stat faible
        stat_s = {"BTC": 0.1}
        ml_s   = {"BTC": 0.9}

        r_stable = orch.combine(
            ts=pd.Timestamp("2025-01-01"),
            stat_arb_signals=stat_s, ml_signals=ml_s,
            regime_q=1.0, q_break=1.0, sent_gate=1.0,
        )
        r_high_vol = orch.combine(
            ts=pd.Timestamp("2025-01-01"),
            stat_arb_signals=stat_s, ml_signals=ml_s,
            regime_q=0.4, q_break=1.0, sent_gate=1.0,
        )
        # En haute vol, le poids ML est plus élevé → signal plus fort
        assert r_high_vol["BTC"] > r_stable["BTC"], \
            "En haute vol, le ML devrait avoir plus d'influence"


# ─────────────────────────────────────────────────────────────────────────────
# Test SentimentAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestSentimentAgent:

    def test_gate_mapping(self):
        from hyperstat.agents.orchestrator import SentimentAgent
        agent = SentimentAgent()

        # Peur extrême
        assert agent.compute_gate(10) == 0.3
        # Peur
        assert agent.compute_gate(35) == 0.6
        # Neutre
        assert agent.compute_gate(50) == 1.0
        # Avidité
        assert agent.compute_gate(65) == 0.8
        # Avidité extrême
        assert agent.compute_gate(90) == 0.5

    def test_gate_in_range(self):
        from hyperstat.agents.orchestrator import SentimentAgent
        agent = SentimentAgent()
        for fg in range(0, 101, 5):
            gate = agent.compute_gate(fg)
            assert 0.0 <= gate <= 1.0, f"Gate hors [0,1] pour F&G={fg}: {gate}"


# ─────────────────────────────────────────────────────────────────────────────
# Test RealTimeSimulator
# ─────────────────────────────────────────────────────────────────────────────

class TestRealTimeSimulator:

    def test_no_lookahead_in_available_data(self):
        """Vérifier que get_available_data() ne retourne jamais de données futures."""
        from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig, RealTimeSimulator
        candles = make_universe(n_bars=3000)
        splitter = DataSplitter(SplitConfig())
        split = splitter.compute_split(candles)
        sim = RealTimeSimulator(splitter, split)

        test_ts = list(split.test_idx)[10]
        c_avail, _ = sim.get_available_data(test_ts, candles, make_funding_universe())

        for sym, df in c_avail.items():
            assert df.index.max() < test_ts, \
                f"Lookahead : données après {test_ts} dans {sym}"

    def test_iter_test_bars_ordered(self):
        from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig, RealTimeSimulator
        candles = make_universe(n_bars=3000)
        splitter = DataSplitter(SplitConfig())
        split = splitter.compute_split(candles)
        sim = RealTimeSimulator(splitter, split)

        bars = list(sim.iter_test_bars())
        assert bars == sorted(bars), "Les barres test ne sont pas en ordre chronologique"


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test complet (intégration)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_smoke(self):
        """
        Test de fumée : split → fit ML → predict → orchestrate
        Sans backtest engine (ne dépend pas d'engine.py)
        """
        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost non installé")

        from hyperstat.ml.walk_forward_split import DataSplitter, SplitConfig
        from hyperstat.ml.lstm_xgb_predictor import MLPredictor, MLPredictorConfig
        from hyperstat.agents.orchestrator import Orchestrator, OrchestratorConfig

        n_bars = 2500
        n_sym = 4
        candles = make_universe(n_sym, n_bars)
        funding = make_funding_universe(n_sym, n_bars)

        # 1. Split
        splitter = DataSplitter(SplitConfig(min_train_bars=1000))
        split = splitter.compute_split(candles)
        assert split.n_test > 50

        # 2. Fit ML sur train+val
        c_tr = {s: df[df.index <= split.val_end] for s, df in candles.items()}
        f_tr = {s: ser[ser.index <= split.val_end] for s, ser in funding.items()}
        predictor = MLPredictor(MLPredictorConfig(fallback_to_xgb_only=True, min_ic_threshold=0.0))
        predictor.fit(c_tr, f_tr)
        assert predictor.is_trained

        # 3. Predict sur quelques barres test
        orch = Orchestrator(OrchestratorConfig(enable_reflect=False))
        test_bars = list(split.test_idx)[:5]

        for ts in test_bars:
            c_avail = {s: df[df.index < ts] for s, df in candles.items()}
            f_avail = {s: ser[ser.index < ts] for s, ser in funding.items()}

            ml_scores = predictor.predict(c_avail, f_avail, ts)
            stat_scores = {s: np.random.uniform(-0.5, 0.5) for s in candles}  # mock

            final = orch.combine(
                ts=ts,
                stat_arb_signals=stat_scores,
                ml_signals=ml_scores,
                regime_q=1.0,
                q_break=1.0,
                sent_gate=0.9,
            )
            assert set(final.keys()) == set(candles.keys())
            for score in final.values():
                assert -1.0 <= score <= 1.0

        print(f"\n✓ Pipeline complet fonctionnel sur {len(test_bars)} barres test")
        print(f"  IC ML = {predictor.last_ic:.4f}")
