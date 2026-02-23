"""
tests/test_agents_smoke.py
============================
Tests smoke pour la couche multi-agents HyperStat v2.
Vérifie que tout s'importe et s'initialise sans erreur.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajoute le src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def make_candles(n: int = 500, symbol: str = "BTC") -> pd.DataFrame:
    """Génère des candles synthétiques pour les tests."""
    ts    = pd.date_range(end=datetime.utcnow(), periods=n, freq="5min")
    price = 50000 * np.cumprod(1 + np.random.normal(0, 0.002, n))
    vol   = np.random.uniform(100, 1000, n)
    return pd.DataFrame({
        "open"  : price * 0.999,
        "high"  : price * 1.002,
        "low"   : price * 0.998,
        "close" : price,
        "volume": vol,
    }, index=ts)


@pytest.fixture
def sample_candles():
    return {"BTC": make_candles(500, "BTC"), "ETH": make_candles(500, "ETH")}


# ─────────────────────────────────────────────────────────────────────────────
# TESTS data/features.py
# ─────────────────────────────────────────────────────────────────────────────

class TestDataFeatures:
    def test_import(self):
        from hyperstat.data.features import (
            compute_returns, compute_ewma_vol, compute_rv_1h_pct
        )
        assert callable(compute_returns)
        assert callable(compute_ewma_vol)
        assert callable(compute_rv_1h_pct)

    def test_compute_returns(self):
        from hyperstat.data.features import compute_returns
        prices = pd.Series([100.0, 101.0, 100.5, 102.0], name="close")
        ret = compute_returns(prices)
        assert len(ret) == 4
        assert pd.isna(ret.iloc[0])
        assert abs(ret.iloc[1] - np.log(101/100)) < 1e-9

    def test_compute_ewma_vol(self):
        from hyperstat.data.features import compute_returns, compute_ewma_vol
        df  = make_candles(300)
        ret = compute_returns(df["close"])
        vol = compute_ewma_vol(ret, span=20)
        assert len(vol) == len(ret)
        assert vol.dropna().gt(0).all()

    def test_compute_rv_1h_pct(self):
        from hyperstat.data.features import compute_rv_1h_pct
        df = make_candles(300)
        rv = compute_rv_1h_pct(df)
        assert "rv_1h_pct" in rv.name
        assert rv.dropna().gt(0).all()

    def test_compute_all_features(self):
        from hyperstat.data.features import compute_all_features
        df   = make_candles(300)
        feat = compute_all_features(df)
        expected_cols = ["log_return", "ewma_vol_20", "rv_1h_pct", "dollar_volume"]
        for col in expected_cols:
            assert col in feat.columns, f"Colonne manquante: {col}"

    def test_cross_sectional(self, sample_candles):
        from hyperstat.data.features import compute_cross_sectional_features
        result = compute_cross_sectional_features(sample_candles)
        assert "BTC" in result
        assert "ETH" in result
        assert "beta_btc" in result["ETH"].columns


# ─────────────────────────────────────────────────────────────────────────────
# TESTS base_agent.py
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseAgent:
    def test_import(self):
        from hyperstat.agents.base_agent import (
            BaseAgent, AgentBus, AgentSignal, AgentStatus, SignalDirection
        )
        assert BaseAgent is not None

    def test_agent_bus(self):
        from hyperstat.agents.base_agent import AgentBus, AgentSignal, SignalDirection, AgentStatus
        bus = AgentBus()
        sig = AgentSignal(
            agent_name  = "TestAgent",
            ts          = datetime.utcnow(),
            direction   = SignalDirection.NEUTRAL,
            confidence  = 0.5,
            score       = 0.0,
        )
        bus.publish(sig)
        latest = bus.get_latest()
        assert "TestAgent" in latest
        assert latest["TestAgent"].score == 0.0

    def test_performance_tracker(self):
        from hyperstat.agents.base_agent import AgentPerformanceTracker
        tracker = AgentPerformanceTracker(agent_name="test")
        # IC doit être 0 avec peu de données
        assert tracker.ic == 0.0
        # Remplir avec des données corrélées
        for i in range(50):
            tracker.record(predicted_score=float(i/25 - 1), actual_return=float(i/25 - 1))
        assert tracker.ic > 0.5   # forte corrélation positive


# ─────────────────────────────────────────────────────────────────────────────
# TESTS SentimentAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestSentimentAgent:
    def test_import(self):
        from hyperstat.agents.sentiment_agent import SentimentAgent
        agent = SentimentAgent()
        assert agent.name == "SentimentAgent"

    def test_observe_and_act(self):
        from hyperstat.agents.sentiment_agent import SentimentAgent
        agent = SentimentAgent()
        ts    = datetime.utcnow()
        agent.observe(ts, {
            "liq_long_usd"  : 100_000,
            "liq_short_usd" : 200_000,
            "oi_total"      : 10_000_000,
        })
        signal = agent.act(ts)
        assert hasattr(signal, "score")
        assert -1.0 <= signal.score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_onchain_score_logic(self):
        """Short liquidations > long → bullish signal."""
        from hyperstat.agents.sentiment_agent import SentimentAgent
        agent = SentimentAgent()
        ts    = datetime.utcnow()
        agent.observe(ts, {
            "liq_long_usd"  : 100_000,
            "liq_short_usd" : 900_000,   # beaucoup de shorts liquidés → bullish
            "oi_total"      : 10_000_000,
        })
        score = agent._compute_onchain_score()
        assert score > 0, f"Score attendu > 0, obtenu {score}"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS PredictionAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionAgent:
    def test_import(self):
        from hyperstat.agents.prediction_agent import PredictionAgent, PredictionConfig
        cfg   = PredictionConfig(lstm_epochs=2, xgb_n_estimators=10)
        agent = PredictionAgent(symbols=["BTC"], cfg=cfg)
        assert agent.name == "PredictionAgent"

    def test_train_and_predict(self, sample_candles):
        from hyperstat.agents.prediction_agent import PredictionAgent, PredictionConfig
        cfg   = PredictionConfig(
            lstm_epochs=1, xgb_n_estimators=5, sequence_length=20,
            model_dir="/tmp/test_hyperstat_models"
        )
        agent = PredictionAgent(symbols=["BTC"], cfg=cfg)
        results = agent.train(sample_candles)
        assert "BTC" in results
        r = results["BTC"]
        assert "train_acc" in r or "error" in r

    def test_act_without_model_returns_neutral(self):
        from hyperstat.agents.prediction_agent import PredictionAgent
        agent  = PredictionAgent(symbols=["UNKNOWN"])
        ts     = datetime.utcnow()
        signal = agent.act(ts)
        assert -1.0 <= signal.score <= 1.0

    def test_walk_forward(self, sample_candles):
        from hyperstat.agents.prediction_agent import PredictionAgent, PredictionConfig
        cfg    = PredictionConfig(
            lstm_epochs=1, xgb_n_estimators=5,
            wf_window_days=1, wf_step_days=1, sequence_length=10,
            model_dir="/tmp/test_hyperstat_models_wf"
        )
        agent   = PredictionAgent(symbols=["BTC"], cfg=cfg)
        results = agent.evaluate_walk_forward(sample_candles)
        # Avec peu de données peut retourner {}
        if "BTC" in results:
            assert "mean_acc" in results["BTC"]


# ─────────────────────────────────────────────────────────────────────────────
# TESTS RegimeAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeAgent:
    def test_import(self):
        from hyperstat.agents.regime_agent import RegimeAgent, RegimeConfig
        agent = RegimeAgent()
        assert agent.name == "RegimeAgent"

    def test_crisis_detection(self):
        from hyperstat.agents.regime_agent import RegimeAgent
        agent = RegimeAgent()
        agent._set_active()
        ts    = datetime.utcnow()
        # Inject liquidations massives
        agent.observe(ts, {"liq_total_usd": 100_000_000, "btc_return": -0.05})
        signal = agent.act(ts)
        assert signal.regime_hint == "crisis"
        assert signal.score < 0

    def test_normal_regime(self):
        from hyperstat.agents.regime_agent import RegimeAgent
        agent = RegimeAgent()
        agent._set_active()
        ts    = datetime.utcnow()
        # Données normales
        for _ in range(100):
            agent.observe(ts, {"btc_return": 0.001, "liq_total_usd": 1000, "avg_funding": 0.0001})
        signal = agent.act(ts)
        assert signal.regime_hint in ["mean_reverting", "carry_favorable", "trending", "unknown"]


# ─────────────────────────────────────────────────────────────────────────────
# TESTS SupervisorAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestSupervisorAgent:
    def test_import(self):
        from hyperstat.agents.supervisor import SupervisorAgent, SupervisorConfig
        from hyperstat.agents.base_agent import AgentBus
        bus        = AgentBus()
        supervisor = SupervisorAgent(bus=bus)
        assert supervisor.name == "SupervisorAgent"

    def test_decide_empty_bus_returns_neutral(self):
        from hyperstat.agents.supervisor import SupervisorAgent
        from hyperstat.agents.base_agent import AgentBus
        bus      = AgentBus()
        sup      = SupervisorAgent(bus=bus)
        sup._set_active()
        decision = sup.decide(datetime.utcnow())
        # Pas de signaux → scale neutre ou réduit
        assert 0.0 <= decision.scale_factor <= 1.2

    def test_crisis_triggers_kill_switch(self):
        from hyperstat.agents.supervisor import SupervisorAgent
        from hyperstat.agents.base_agent import AgentBus, AgentSignal, SignalDirection
        bus = AgentBus()
        sup = SupervisorAgent(bus=bus)
        sup._set_active()
        # Publie un signal régime CRISIS
        crisis_signal = AgentSignal(
            agent_name  = "RegimeAgent",
            ts          = datetime.utcnow(),
            direction   = SignalDirection.SHORT,
            confidence  = 1.0,
            score       = -1.0,
            regime_hint = "crisis",
        )
        bus.publish(crisis_signal)
        decision = sup.decide(datetime.utcnow())
        assert decision.kill_switch is True
        assert decision.scale_factor == 0.0

    def test_bullish_signal_increases_scale(self):
        from hyperstat.agents.supervisor import SupervisorAgent, SupervisorConfig
        from hyperstat.agents.base_agent import AgentBus, AgentSignal, SignalDirection
        cfg = SupervisorConfig(use_ic_weighting=False)
        bus = AgentBus()
        sup = SupervisorAgent(bus=bus, cfg=cfg)
        sup._set_active()
        # Tous les agents bullish
        for name in ["TechnicalAgent", "SentimentAgent", "PredictionAgent", "RegimeAgent"]:
            bus.publish(AgentSignal(
                agent_name  = name,
                ts          = datetime.utcnow(),
                direction   = SignalDirection.LONG,
                confidence  = 0.9,
                score       = 0.8,
                regime_hint = "mean_reverting" if name == "RegimeAgent" else None,
            ))
        decision = sup.decide(datetime.utcnow())
        assert decision.scale_factor >= 1.0
        assert not decision.kill_switch


# ─────────────────────────────────────────────────────────────────────────────
# TESTS engine_funding_fix.py
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineFundingFix:
    def test_build_funding_events(self):
        from hyperstat.backtest.engine_funding_fix import build_funding_events
        # Créer des données de funding synthétiques
        ts = pd.date_range("2024-01-01", periods=100, freq="5min")
        funding = {
            "BTC": pd.DataFrame({"rate": np.random.normal(0.0001, 0.00005, 100)}, index=ts),
            "ETH": pd.DataFrame({"rate": np.random.normal(0.0002, 0.00008, 100)}, index=ts),
        }
        events = build_funding_events(funding, ts)
        assert len(events) > 0
        # Vérifier qu'un timestamp a bien les deux symboles
        first_ts = list(events.keys())[0]
        assert "BTC" in events[first_ts]
        assert "ETH" in events[first_ts]


# ─────────────────────────────────────────────────────────────────────────────
# TESTS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

class TestFearGreed:
    def test_normalize(self):
        from hyperstat.agents.utils.fear_greed import FearGreedClient
        client = FearGreedClient()
        assert client._normalize(0)   == -1.0
        assert client._normalize(50)  ==  0.0
        assert client._normalize(100) ==  1.0
        assert client._normalize(25)  == -0.5

    def test_regime_signal(self):
        from hyperstat.agents.utils.fear_greed import FearGreedClient
        client = FearGreedClient()
        # Simule le cache
        client._cache_val = {"value": 15, "value_classification": "Extreme Fear",
                             "timestamp": datetime.utcnow(), "normalized_score": -0.7}
        client._cache_ts  = float("inf")  # jamais périmé
        assert client.get_regime_signal() == "extreme_fear"


class TestNewsFetcher:
    def test_score_article_bullish(self):
        from hyperstat.agents.utils.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        article = {"title": "Bitcoin surges to new record high on institutional adoption",
                   "body": "Rally continues as ETF approval drives gains"}
        score = fetcher._score_article(article)
        assert score > 0

    def test_score_article_bearish(self):
        from hyperstat.agents.utils.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        article = {"title": "Crypto market crashes amid regulatory crackdown",
                   "body": "Bitcoin dumps as hack exploit revealed"}
        score = fetcher._score_article(article)
        assert score < 0


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])
