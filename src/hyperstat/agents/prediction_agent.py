"""
hyperstat.agents.prediction_agent
===================================
PredictionAgent — Modèle LSTM + XGBoost pour prédiction directionnelle.
Inspiré du papier : CRYPTO PRICE PREDICTION USING LSTM+XGBOOST (arxiv 2506.22055v1)

Architecture :
    LSTM   : capture les dépendances temporelles (séquences de 60 barres)
    XGBoost: features tabular (indicateurs techniques, microstructure)
    Fusion : moyenne pondérée des probabilités
    Output : P(hausse) ∈ [0,1] par symbole → score ∈ [-1, 1]

Train/Test split :
    - Walk-forward temporel strict (80% train / 20% test par fenêtre)
    - Jamais de look-ahead bias
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hyperstat.agents.base_agent import BaseAgent, AgentSignal, AgentStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionConfig:
    # Données
    sequence_length  : int   = 60       # barres LSTM (60 × 5min = 5h)
    forward_horizon  : int   = 12       # prédire dans 12 barres (1h)
    test_ratio       : float = 0.20     # 80% train / 20% test

    # LSTM
    lstm_hidden      : int   = 64
    lstm_layers      : int   = 2
    lstm_dropout     : float = 0.2
    lstm_epochs      : int   = 30
    lstm_batch_size  : int   = 64
    lstm_lr          : float = 1e-3

    # XGBoost
    xgb_n_estimators : int   = 200
    xgb_max_depth    : int   = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample    : float = 0.8

    # Fusion
    lstm_weight      : float = 0.5
    xgb_weight       : float = 0.5

    # Walk-forward
    wf_window_days   : int   = 30
    wf_step_days     : int   = 7

    # Persistence
    model_dir        : str   = "./artifacts/models"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING POUR LE ML
# ─────────────────────────────────────────────────────────────────────────────

def _build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features tabular pour XGBoost.
    Input : DataFrame avec colonnes open/high/low/close/volume + features existantes.
    Output : DataFrame de features normalisées.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"]
    ret   = np.log(close / close.shift(1))

    # Returns multi-horizons
    for h in [1, 3, 6, 12, 24, 48]:
        feat[f"ret_{h}b"] = np.log(close / close.shift(h))

    # Volatilité
    for span in [12, 48, 96]:
        feat[f"vol_ewm_{span}"] = ret.ewm(span=span).std()

    # RSI
    for w in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/w, min_periods=w).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/w, min_periods=w).mean()
        rs   = gain / (loss + 1e-9)
        feat[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    feat["macd_norm"]  = macd / (close + 1e-9)
    feat["macd_hist"]  = (macd - macd.ewm(span=9).mean()) / (close + 1e-9)

    # Bollinger
    roll20 = close.rolling(20)
    bb_std = roll20.std()
    bb_mid = roll20.mean()
    feat["bb_pct"] = (close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)
    feat["bb_width"] = (4*bb_std) / (bb_mid + 1e-9)

    # Volume
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(20).mean()
        feat["vol_ratio"] = df["volume"] / (vol_ma + 1e-9)
        feat["vol_log"]   = np.log1p(df["volume"])

    # Heure du jour (feature calendrier)
    if hasattr(df.index, "hour"):
        feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    # Funding si disponible
    if "funding_rate" in df.columns:
        feat["funding"]      = df["funding_rate"]
        feat["funding_ewm"]  = df["funding_rate"].ewm(span=24).mean()

    return feat.fillna(0.0)


def _build_target(close: pd.Series, horizon: int) -> pd.Series:
    """
    Target binaire : 1 si close(t+horizon) > close(t), else 0.
    """
    future_ret = np.log(close.shift(-horizon) / close)
    return (future_ret > 0).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMModel:
    """Wrapper LSTM PyTorch avec fallback si PyTorch absent."""

    def __init__(self, cfg: PredictionConfig, n_features: int):
        self.cfg        = cfg
        self.n_features = n_features
        self._model     = None
        self._scaler    = None
        self._has_torch = self._check_torch()

    @staticmethod
    def _check_torch() -> bool:
        try:
            import torch  # noqa
            return True
        except ImportError:
            logger.warning("[PredictionAgent] PyTorch absent — LSTM désactivé. pip install torch")
            return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        X shape : (n_samples, sequence_length, n_features)
        y shape : (n_samples,) binaire
        Retourne les métriques d'entraînement.
        """
        if not self._has_torch:
            return {"accuracy": 0.5, "status": "skipped_no_torch"}

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Normalisation
        from sklearn.preprocessing import StandardScaler
        n, seq, feat = X.shape
        X_2d   = X.reshape(-1, feat)
        scaler = StandardScaler()
        X_2d   = scaler.fit_transform(X_2d)
        X_norm = X_2d.reshape(n, seq, feat)
        self._scaler = scaler

        X_t = torch.FloatTensor(X_norm)
        y_t = torch.FloatTensor(y)

        # Architecture
        class LSTMNet(nn.Module):
            def __init__(self, inp, hidden, layers, drop):
                super().__init__()
                self.lstm = nn.LSTM(inp, hidden, layers,
                                    batch_first=True, dropout=drop)
                self.fc   = nn.Linear(hidden, 1)
                self.sig  = nn.Sigmoid()

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.sig(self.fc(out[:, -1, :])).squeeze()

        model = LSTMNet(
            inp    = feat,
            hidden = self.cfg.lstm_hidden,
            layers = self.cfg.lstm_layers,
            drop   = self.cfg.lstm_dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lstm_lr)
        criterion = nn.BCELoss()
        loader    = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.cfg.lstm_batch_size,
            shuffle=True,
        )

        model.train()
        for epoch in range(self.cfg.lstm_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Accuracy sur train
        model.eval()
        with torch.no_grad():
            preds = (model(X_t).numpy() > 0.5).astype(int)
        acc = (preds == y.astype(int)).mean()

        self._model = model
        return {"accuracy": float(acc), "status": "trained"}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne P(hausse) ∈ [0,1] pour chaque sample."""
        if not self._has_torch or self._model is None:
            return np.full(len(X), 0.5)
        import torch
        n, seq, feat = X.shape
        X_2d = X.reshape(-1, feat)
        if self._scaler:
            X_2d = self._scaler.transform(X_2d)
        X_norm = X_2d.reshape(n, seq, feat)
        self._model.eval()
        with torch.no_grad():
            proba = self._model(torch.FloatTensor(X_norm)).numpy()
        return proba.flatten()


class _XGBModel:
    """Wrapper XGBoost avec fallback si xgboost absent."""

    def __init__(self, cfg: PredictionConfig):
        self.cfg    = cfg
        self._model = None
        self._has_xgb = self._check_xgb()

    @staticmethod
    def _check_xgb() -> bool:
        try:
            import xgboost  # noqa
            return True
        except ImportError:
            logger.warning("[PredictionAgent] XGBoost absent — pip install xgboost")
            return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not self._has_xgb:
            return {"accuracy": 0.5, "status": "skipped_no_xgb"}
        import xgboost as xgb
        self._model = xgb.XGBClassifier(
            n_estimators  = self.cfg.xgb_n_estimators,
            max_depth      = self.cfg.xgb_max_depth,
            learning_rate  = self.cfg.xgb_learning_rate,
            subsample      = self.cfg.xgb_subsample,
            use_label_encoder=False,
            eval_metric    = "logloss",
            random_state   = 42,
            n_jobs         = -1,
        )
        self._model.fit(X, y)
        acc = (self._model.predict(X) == y).mean()
        # Feature importances
        importances = dict(zip(
            [f"f{i}" for i in range(X.shape[1])],
            self._model.feature_importances_.tolist()
        ))
        return {"accuracy": float(acc), "status": "trained", "importances": importances}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._has_xgb or self._model is None:
            return np.full(len(X), 0.5)
        return self._model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION AGENT
# ─────────────────────────────────────────────────────────────────────────────

class PredictionAgent(BaseAgent):
    """
    Agent ML directionnel par symbole.

    Usage
    -----
    agent = PredictionAgent(symbols=["BTC", "ETH", "SOL"])
    agent.train(candles_by_symbol)          # entraîne les modèles
    metrics = agent.evaluate_walk_forward() # valide sans look-ahead
    signal  = agent.act(ts)                 # signal agrégé pour l'univers
    """

    def __init__(
        self,
        symbols : List[str],
        cfg     : Optional[PredictionConfig] = None,
        **kwargs,
    ):
        super().__init__(name="PredictionAgent", **kwargs)
        self.symbols   = symbols
        self.cfg       = cfg or PredictionConfig()
        self._models   : Dict[str, Tuple[_LSTMModel, _XGBModel]] = {}
        self._last_proba : Dict[str, float] = {}       # {symbol: P(hausse)}
        self._candles_buffer : Dict[str, pd.DataFrame] = {}

    # ── Interface BaseAgent ───────────────────────────────────────────────

    def warm_up(self, candles_by_symbol: Optional[Dict[str, pd.DataFrame]] = None, **kwargs) -> bool:
        """Charge les modèles depuis le disque ou lance l'entraînement."""
        model_path = Path(self.cfg.model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        loaded = 0
        for sym in self.symbols:
            p = model_path / f"{sym}_prediction.pkl"
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        self._models[sym] = pickle.load(f)
                    loaded += 1
                except Exception as exc:
                    logger.warning(f"[PredictionAgent] Impossible de charger {sym}: {exc}")

        if loaded < len(self.symbols) and candles_by_symbol:
            logger.info("[PredictionAgent] Entraînement des modèles manquants...")
            self.train(candles_by_symbol)

        if self._models:
            self._set_active()
            return True

        self._set_degraded(reason="Aucun modèle disponible")
        return False

    def observe(self, ts: datetime, data: Dict[str, Any]) -> None:
        """Met à jour le buffer de candles en temps réel."""
        candles = data.get("candles_by_symbol", {})
        for sym, df in candles.items():
            self._candles_buffer[sym] = df

    def act(self, ts: datetime) -> AgentSignal:
        try:
            scores = {}
            for sym in self.symbols:
                if sym in self._candles_buffer and sym in self._models:
                    proba = self._predict_symbol(sym, self._candles_buffer[sym])
                    scores[sym] = proba
                    self._last_proba[sym] = proba

            if not scores:
                return self._make_signal(ts=ts, score=0.0, confidence=0.0)

            # Score agrégé : moyenne des (P - 0.5) × 2 convertis en [-1, 1]
            raw_scores = [2 * (p - 0.5) for p in scores.values()]
            agg_score  = float(np.mean(raw_scores))

            # Confidence = fraction de symboles avec signal fort (|score| > 0.2)
            strong = sum(1 for s in raw_scores if abs(s) > 0.2)
            confidence = min(1.0, 0.3 + 0.7 * strong / max(len(raw_scores), 1))

            return self._make_signal(
                ts          = ts,
                score       = agg_score,
                confidence  = confidence,
                n_symbols   = len(scores),
                probas      = {k: round(v, 4) for k, v in scores.items()},
            )

        except Exception as exc:
            return self._handle_error(exc)

    # ── Entraînement ──────────────────────────────────────────────────────

    def train(
        self,
        candles_by_symbol: Dict[str, pd.DataFrame],
        test_ratio        : Optional[float] = None,
    ) -> Dict[str, dict]:
        """
        Entraîne LSTM + XGBoost pour chaque symbole.

        Parameters
        ----------
        candles_by_symbol : {symbol: DataFrame OHLCV}
        test_ratio        : fraction de données pour le test (défaut: cfg.test_ratio)

        Returns
        -------
        {symbol: {"train_acc": float, "test_acc": float, "n_train": int, "n_test": int}}
        """
        tr       = test_ratio or self.cfg.test_ratio
        results  = {}
        model_dir = Path(self.cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        for sym, df in candles_by_symbol.items():
            if sym not in self.symbols:
                continue
            try:
                result = self._train_symbol(sym, df, tr)
                results[sym] = result
                # Sauvegarde
                with open(model_dir / f"{sym}_prediction.pkl", "wb") as f:
                    pickle.dump(self._models[sym], f)
                logger.info(f"[PredictionAgent] {sym} → train_acc={result['train_acc']:.3f} "
                            f"test_acc={result['test_acc']:.3f}")
            except Exception as exc:
                logger.error(f"[PredictionAgent] Erreur train {sym}: {exc}")
                results[sym] = {"error": str(exc)}

        if self._models:
            self._set_active()
        return results

    def _train_symbol(self, sym: str, df: pd.DataFrame, test_ratio: float) -> dict:
        """Entraîne un symbole avec split temporel strict."""
        features_df = _build_ml_features(df)
        target      = _build_target(df["close"], self.cfg.forward_horizon)

        # Aligner et nettoyer
        valid_idx   = features_df.dropna().index.intersection(target.dropna().index)
        valid_idx   = valid_idx[:-self.cfg.forward_horizon]  # enlever les derniers (pas de target)
        X_df        = features_df.loc[valid_idx]
        y_series    = target.loc[valid_idx]

        n           = len(X_df)
        split_idx   = int(n * (1 - test_ratio))

        X_train = X_df.iloc[:split_idx].values
        y_train = y_series.iloc[:split_idx].values
        X_test  = X_df.iloc[split_idx:].values
        y_test  = y_series.iloc[split_idx:].values

        # ── XGBoost ──
        xgb_model = _XGBModel(self.cfg)
        xgb_train_res = xgb_model.fit(X_train, y_train)
        xgb_test_acc  = (xgb_model.predict_proba(X_test) > 0.5).astype(int)
        xgb_test_acc  = float((xgb_test_acc == y_test).mean())

        # ── LSTM (séquences) ──
        seq     = self.cfg.sequence_length
        lstm_model = _LSTMModel(self.cfg, n_features=X_df.shape[1])
        if len(X_train) > seq + 10:
            Xs, ys = self._make_sequences(X_train, y_train, seq)
            lstm_train_res = lstm_model.fit(Xs, ys)
            Xs_t, ys_t     = self._make_sequences(X_test, y_test, seq)
            lstm_test_proba = lstm_model.predict_proba(Xs_t)
            lstm_test_acc   = float(((lstm_test_proba > 0.5).astype(int) == ys_t).mean())
        else:
            lstm_train_res = {"accuracy": 0.5, "status": "skipped_insufficient_data"}
            lstm_test_acc  = 0.5

        self._models[sym] = (lstm_model, xgb_model)

        return {
            "train_acc" : float(xgb_train_res["accuracy"]),
            "test_acc"  : float((xgb_test_acc + lstm_test_acc) / 2),
            "xgb_test"  : xgb_test_acc,
            "lstm_test" : lstm_test_acc,
            "n_train"   : split_idx,
            "n_test"    : n - split_idx,
        }

    def evaluate_walk_forward(
        self,
        candles_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[str, dict]:
        """
        Walk-forward validation : entraîne sur fenêtre glissante, évalue sur suivante.
        Retourne les métriques de stabilité pour chaque symbole.
        """
        window_bars = self.cfg.wf_window_days * 288  # 288 barres 5min par jour
        step_bars   = self.cfg.wf_step_days   * 288
        results     = {}

        for sym, df in candles_by_symbol.items():
            if sym not in self.symbols:
                continue

            accs = []
            n    = len(df)
            start = 0
            while start + window_bars + step_bars < n:
                train_df = df.iloc[start : start + window_bars]
                test_df  = df.iloc[start + window_bars : start + window_bars + step_bars]

                try:
                    self._train_symbol(sym, train_df, test_ratio=0.0)  # tout en train
                    feats = _build_ml_features(test_df)
                    tgt   = _build_target(test_df["close"], self.cfg.forward_horizon)
                    valid = feats.dropna().index.intersection(tgt.dropna().index)
                    if len(valid) > 10:
                        _, xgb = self._models[sym]
                        preds  = (xgb.predict_proba(feats.loc[valid].values) > 0.5).astype(int)
                        acc    = float((preds == tgt.loc[valid].values).mean())
                        accs.append(acc)
                except Exception as exc:
                    logger.debug(f"[WF] {sym} window erreur: {exc}")

                start += step_bars

            if accs:
                results[sym] = {
                    "mean_acc" : round(float(np.mean(accs)), 4),
                    "std_acc"  : round(float(np.std(accs)), 4),
                    "n_windows": len(accs),
                    "min_acc"  : round(float(np.min(accs)), 4),
                    "max_acc"  : round(float(np.max(accs)), 4),
                }

        return results

    # ── Prédiction live ───────────────────────────────────────────────────

    def _predict_symbol(self, sym: str, df: pd.DataFrame) -> float:
        """Retourne P(hausse) ∈ [0,1] pour le dernier point du symbole."""
        if sym not in self._models:
            return 0.5
        lstm_model, xgb_model = self._models[sym]

        feats = _build_ml_features(df).fillna(0.0)
        if len(feats) < self.cfg.sequence_length + 1:
            return 0.5

        X_last = feats.values[-1:, :]   # dernière ligne pour XGBoost
        xgb_p  = float(xgb_model.predict_proba(X_last)[0])

        seq = self.cfg.sequence_length
        if len(feats) >= seq:
            X_seq   = feats.values[-seq:, :].reshape(1, seq, -1)
            lstm_p  = float(lstm_model.predict_proba(X_seq)[0])
        else:
            lstm_p = 0.5

        return self.cfg.lstm_weight * lstm_p + self.cfg.xgb_weight * xgb_p

    @staticmethod
    def _make_sequences(
        X: np.ndarray, y: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforme un array 2D en séquences 3D pour le LSTM."""
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i-seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def get_per_symbol_probas(self) -> Dict[str, float]:
        """Retourne le dernier P(hausse) par symbole pour le dashboard."""
        return dict(self._last_proba)
