"""
HyperStat v2 — Agent ML Predictor
LSTM + XGBoost hybrid (Gautam 2025, arXiv:2506.22055)

Pipeline :
  1. LSTM extrait un embedding temporel z = hn ∈ R^64 depuis les 120 barres précédentes
  2. XGBoost combine cet embedding avec des features cross-sectionnelles
     (z-score FDS, funding, OFI) pour prédire directional_score ∈ [-1, 1]

Intégration dans HyperStat :
  signal_final = w_stat * s_stat + w_ml * directional_score * sent_gate

Auteur   : HyperStat v2
Réf.     : Gautam (2025) arXiv:2506.22055
           Wang   (2025) arXiv:2506.05764  (feature engineering / débruitage)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Imports optionnels (installés via requirements-ml.txt)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch non disponible. Installer : pip install torch")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost non disponible. Installer : pip install xgboost")

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LSTMConfig:
    """Paramètres du LSTM feature extractor."""
    n_steps_in: int = 120          # fenêtre temporelle (barres)
    n_features: int = 8            # features par barre (voir _build_sequence_features)
    hidden_size: int = 64          # dimension embedding hn
    num_layers: int = 2
    dropout: float = 0.20
    batch_size: int = 128
    max_epochs: int = 50
    lr: float = 1e-3
    patience: int = 7              # early stopping
    device: str = "cpu"


@dataclass
class XGBConfig:
    """Paramètres du XGBoost régresseur."""
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 20
    eval_metric: str = "rmse"


@dataclass
class MLPredictorConfig:
    """Config globale du pipeline ML."""
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)

    # Horizon de prédiction (barres forward)
    forward_horizon: int = 8

    # Débruitage des features (Wang 2025)
    denoise_method: str = "savgol"   # "savgol" | "kalman" | "none"
    savgol_window: int = 11
    savgol_poly: int = 3

    # Walk-forward
    train_window_bars: int = 4320    # ~90 jours en 1h
    val_window_bars: int = 720       # ~15 jours
    retrain_every_bars: int = 336    # ~2 semaines

    # Modèle de repli si torch absent
    fallback_to_xgb_only: bool = True

    # Seuil de confiance minimum pour émettre un signal
    min_ic_threshold: float = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMExtractor(nn.Module if TORCH_AVAILABLE else object):
    """
    LSTM → hidden state hn ∈ R^hidden_size
    Architecture : LSTM(n_layers) + Linear(hidden_size)
    """

    def __init__(self, cfg: LSTMConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch requis pour LSTMExtractor")
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, 1),  # regression : retour forward normalisé
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, n_features)
        _, (hn, _) = self.lstm(x)
        z = hn[-1]   # dernière couche : (batch, hidden_size)
        return self.head(z).squeeze(-1)   # (batch,)

    def extract_embedding(self, x: "torch.Tensor") -> "torch.Tensor":
        """Retourne l'embedding z = hn[-1] sans la tête de régression."""
        with torch.no_grad():
            _, (hn, _) = self.lstm(x)
        return hn[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering (séquences LSTM + features tabulaires XGB)
# ─────────────────────────────────────────────────────────────────────────────

def _denoise(series: pd.Series, method: str, window: int, poly: int) -> pd.Series:
    """Débruitage d'une série (Wang 2025 : Savitzky-Golay ou Kalman)."""
    if method == "none" or len(series) < window:
        return series
    if method == "savgol" and SCIPY_AVAILABLE:
        w = min(window, len(series))
        if w % 2 == 0:
            w -= 1
        smoothed = savgol_filter(series.values, window_length=w, polyorder=poly)
        return pd.Series(smoothed, index=series.index)
    # Fallback : EWM léger
    return series.ewm(span=5, min_periods=1).mean()


def build_sequence_features(
    candles: pd.DataFrame,
    funding: pd.Series,
    cfg: MLPredictorConfig,
) -> pd.DataFrame:
    """
    Construit les features par barre pour le LSTM.

    Features par barre (cfg.lstm.n_features = 8) :
      0. log_return normalisé
      1. EWMA_vol_12 (vol courte)
      2. EWMA_vol_48 (vol longue)
      3. vol_ratio = EWMA_vol_12 / EWMA_vol_48   (régime vol)
      4. funding_rate (ou 0 si absent)
      5. funding_EWMA_fast (span=8)
      6. funding_EWMA_slow (span=72)
      7. volume_normalized (log DV normalisé)

    Tous les features sont z-scorés rolling sur 240 barres.
    """
    assert "close" in candles.columns, "candles doit avoir une colonne 'close'"
    assert "volume" in candles.columns, "candles doit avoir une colonne 'volume'"

    df = pd.DataFrame(index=candles.index)

    # --- Returns ---
    close = candles["close"].astype(float)
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    log_ret = _denoise(log_ret, cfg.denoise_method, cfg.savgol_window, cfg.savgol_poly)
    df["log_return"] = log_ret

    # --- Volatilité ---
    df["ewma_vol_12"] = log_ret.ewm(span=12, min_periods=3).std().fillna(0.0)
    df["ewma_vol_48"] = log_ret.ewm(span=48, min_periods=12).std().fillna(0.0)
    eps = 1e-8
    df["vol_ratio"] = df["ewma_vol_12"] / (df["ewma_vol_48"] + eps)

    # --- Funding ---
    if funding is not None and len(funding) > 0:
        f = funding.reindex(candles.index, method="ffill").fillna(0.0).astype(float)
        f = _denoise(f, cfg.denoise_method, cfg.savgol_window, cfg.savgol_poly)
    else:
        f = pd.Series(0.0, index=candles.index)
    df["funding"] = f
    df["funding_fast"] = f.ewm(span=8, min_periods=2).mean().fillna(0.0)
    df["funding_slow"] = f.ewm(span=72, min_periods=12).mean().fillna(0.0)

    # --- Volume ---
    vol = candles["volume"].astype(float).replace(0, np.nan).fillna(method="ffill")
    dv = np.log(vol * close + 1.0)
    df["log_dv"] = dv

    # --- Z-score rolling (240 barres) pour normalisation ---
    roll_w = 240
    for col in df.columns:
        mu = df[col].rolling(roll_w, min_periods=30).mean()
        sigma = df[col].rolling(roll_w, min_periods=30).std().replace(0, np.nan)
        df[col] = ((df[col] - mu) / sigma).fillna(0.0).clip(-5, 5)

    return df.astype(np.float32)


def build_tabular_features(
    symbol: str,
    candles_all: Dict[str, pd.DataFrame],
    funding_all: Dict[str, pd.Series],
    fds_scores: Optional[pd.Series],
    regime_q: Optional[pd.Series],
    ts: pd.Timestamp,
    lookback: int = 120,
) -> np.ndarray:
    """
    Features tabulaires pour XGBoost à un instant ts.
    Combinées avec l'embedding LSTM → input final XGBoost.

    Features cross-sectionnelles (6) :
      0. z_score_return_12         (signal MR actuel)
      1. fds_score                 (FDS gate)
      2. funding_spread_vs_median  (carry component)
      3. vol_rank_pct              (illiquidité relative)
      4. regime_q                  (Q_t qualité régime)
      5. funding_velocity          (accélération funding)
    """
    feats = np.zeros(6, dtype=np.float32)
    candles = candles_all.get(symbol)
    if candles is None or ts not in candles.index:
        return feats

    # Slice jusqu'à ts
    mask = candles.index <= ts
    c = candles.loc[mask].tail(lookback)
    if len(c) < 12:
        return feats

    close = c["close"].astype(float)
    log_ret_12 = np.log(close.iloc[-1] / close.iloc[-12]) if len(c) >= 12 else 0.0

    # z-score vs autres symboles (median/MAD cross-sectionnel)
    all_ret12 = []
    for sym, cv in candles_all.items():
        mask_s = cv.index <= ts
        cs = cv.loc[mask_s].tail(13)
        if len(cs) >= 12:
            all_ret12.append(np.log(cs["close"].iloc[-1] / cs["close"].iloc[-12]))
    if len(all_ret12) > 2:
        med = np.median(all_ret12)
        mad = np.median(np.abs(np.array(all_ret12) - med)) + 1e-8
        feats[0] = float(np.clip((log_ret_12 - med) / mad, -5, 5))
    else:
        feats[0] = 0.0

    feats[1] = float(fds_scores.get(symbol, 0.0)) if fds_scores is not None else 0.0

    # Funding spread
    f = funding_all.get(symbol)
    if f is not None and ts in f.index:
        f_val = float(f.loc[:ts].iloc[-1]) if len(f.loc[:ts]) > 0 else 0.0
        all_f = [float(fv.loc[:ts].iloc[-1]) for fv in funding_all.values()
                 if len(fv.loc[:ts]) > 0]
        feats[2] = float(np.clip(f_val - np.median(all_f), -0.01, 0.01) / 0.01) \
                   if all_f else 0.0
    else:
        feats[2] = 0.0

    # Vol rank
    vol_12 = c["close"].astype(float).pct_change().tail(12).std()
    all_vols = []
    for sym, cv in candles_all.items():
        mask_s = cv.index <= ts
        cs = cv.loc[mask_s]
        if len(cs) >= 12:
            all_vols.append(cs["close"].astype(float).pct_change().tail(12).std())
    feats[3] = float(np.searchsorted(np.sort(all_vols), vol_12) / (len(all_vols) + 1)) \
               if all_vols else 0.5

    feats[4] = float(regime_q) if regime_q is not None else 1.0

    # Funding velocity
    if f is not None:
        f_slice = f.loc[:ts].tail(20).astype(float)
        if len(f_slice) >= 8:
            fast = f_slice.ewm(span=8).mean().iloc[-1]
            slow = f_slice.ewm(span=16).mean().iloc[-1]
            feats[5] = float(np.clip((fast - slow) / (abs(slow) + 1e-8), -2, 2))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale : MLPredictor
# ─────────────────────────────────────────────────────────────────────────────

class MLPredictor:
    """
    Agent ML Predictor LSTM+XGBoost pour HyperStat v2.

    Usage (backtest walk-forward) :
        predictor = MLPredictor(MLPredictorConfig())
        predictor.fit(candles_train, funding_train, buckets)

        # À chaque barre :
        scores = predictor.predict(candles_window, funding_window, ts,
                                   fds_scores, regime_q)
        # scores : {symbol: directional_score ∈ [-1, 1]}
    """

    def __init__(self, cfg: Optional[MLPredictorConfig] = None):
        self.cfg = cfg or MLPredictorConfig()
        self._lstm: Optional["_LSTMExtractor"] = None
        self._xgb: Optional["xgb.XGBRegressor"] = None
        self._trained = False
        self._last_ic: float = 0.0
        self._symbols: List[str] = []

    # ──────────────────────────────────────────────────────────────────────
    # ENTRAÎNEMENT
    # ──────────────────────────────────────────────────────────────────────

    def fit(
        self,
        candles_by_symbol: Dict[str, pd.DataFrame],
        funding_by_symbol: Dict[str, pd.Series],
        val_fraction: float = 0.15,
    ) -> "MLPredictor":
        """
        Entraîne LSTM puis XGBoost sur les données fournies.
        Gère automatiquement le split train/val interne.
        """
        if not XGB_AVAILABLE:
            logger.error("XGBoost requis pour l'entraînement.")
            return self

        self._symbols = list(candles_by_symbol.keys())
        logger.info(f"[MLPredictor] fit sur {len(self._symbols)} symboles")

        X_seq, X_tab, y = self._build_training_data(
            candles_by_symbol, funding_by_symbol
        )

        if len(y) < 100:
            logger.warning(f"[MLPredictor] Seulement {len(y)} exemples — skip fit")
            return self

        # Split val
        n_val = max(int(len(y) * val_fraction), 20)
        X_seq_tr, X_seq_val = X_seq[:-n_val], X_seq[-n_val:]
        X_tab_tr, X_tab_val = X_tab[:-n_val], X_tab[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        # 1. Entraîner LSTM (si disponible)
        if TORCH_AVAILABLE and not self.cfg.fallback_to_xgb_only:
            self._fit_lstm(X_seq_tr, y_tr, X_seq_val, y_val)
            emb_tr = self._get_embeddings(X_seq_tr)
            emb_val = self._get_embeddings(X_seq_val)
        else:
            # Sans LSTM : résumé statistique de la séquence comme features
            emb_tr = self._seq_summary(X_seq_tr)
            emb_val = self._seq_summary(X_seq_val)
            logger.info("[MLPredictor] Mode XGB-only (LSTM désactivé)")

        # 2. Entraîner XGBoost sur embedding + features tabulaires
        X_xgb_tr = np.hstack([emb_tr, X_tab_tr])
        X_xgb_val = np.hstack([emb_val, X_tab_val])
        self._fit_xgb(X_xgb_tr, y_tr, X_xgb_val, y_val)

        # 3. Calculer IC sur validation
        if self._xgb is not None:
            preds = self._xgb.predict(X_xgb_val)
            self._last_ic = float(
                pd.Series(preds).corr(pd.Series(y_val), method="spearman")
            )
            logger.info(
                f"[MLPredictor] IC Spearman val = {self._last_ic:.4f}"
                f"  ({'OK ✓' if self._last_ic > self.cfg.min_ic_threshold else 'FAIBLE ⚠'})"
            )

        self._trained = True
        return self

    def _build_training_data(
        self,
        candles_by_symbol: Dict[str, pd.DataFrame],
        funding_by_symbol: Dict[str, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construit (X_seq, X_tab, y) pour tous les symboles et timestamps."""
        cfg = self.cfg
        H = cfg.forward_horizon
        n = cfg.lstm.n_steps_in
        all_seq, all_tab, all_y = [], [], []

        for symbol, candles in candles_by_symbol.items():
            if len(candles) < n + H + 10:
                continue
            funding = funding_by_symbol.get(symbol, pd.Series(dtype=float))
            seq_df = build_sequence_features(candles, funding, cfg)
            arr = seq_df.values  # (T, n_features)
            close = candles["close"].astype(float).values

            for t in range(n, len(arr) - H):
                x_seq = arr[t - n: t]           # (n_steps, n_features)
                # Target : log-return forward normalisé par vol
                fwd_ret = np.log(close[t + H] / (close[t] + 1e-8))
                local_vol = arr[t - min(12, n): t, 0].std() + 1e-8
                y_norm = float(np.clip(fwd_ret / local_vol, -4, 4))

                # Features tabulaires simplifiées (au temps t)
                x_tab = np.array([
                    float(arr[t - 1, 0]),   # log_return
                    float(arr[t - 1, 3]),   # vol_ratio
                    float(arr[t - 1, 4]),   # funding
                    float(arr[t - 1, 5]),   # funding_fast
                    float(arr[t - 1, 6]),   # funding_slow
                    float(arr[t - 1, 7]),   # log_dv
                ], dtype=np.float32)

                all_seq.append(x_seq)
                all_tab.append(x_tab)
                all_y.append(y_norm)

        if not all_seq:
            return np.empty((0, n, cfg.lstm.n_features)), np.empty((0, 6)), np.empty(0)

        return (
            np.array(all_seq, dtype=np.float32),
            np.array(all_tab, dtype=np.float32),
            np.array(all_y, dtype=np.float32),
        )

    def _fit_lstm(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ):
        cfg = self.cfg.lstm
        device = torch.device(cfg.device)
        model = _LSTMExtractor(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.HuberLoss(delta=1.0)

        ds_tr = TensorDataset(
            torch.tensor(X_tr).to(device),
            torch.tensor(y_tr).to(device),
        )
        dl = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_count = 0
        best_state = None

        X_val_t = torch.tensor(X_val).to(device)
        y_val_t = torch.tensor(y_val).to(device)

        for epoch in range(cfg.max_epochs):
            model.train()
            for xb, yb in dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    logger.info(f"[LSTM] Early stop epoch {epoch + 1}")
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        self._lstm = model
        logger.info(f"[LSTM] Entraîné — best val loss = {best_val_loss:.4f}")

    def _fit_xgb(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ):
        cfg = self.cfg.xgb
        self._xgb = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            early_stopping_rounds=cfg.early_stopping_rounds,
            eval_metric=cfg.eval_metric,
            verbosity=0,
            n_jobs=-1,
        )
        self._xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info(f"[XGB] best_iteration = {self._xgb.best_iteration}")

    # ──────────────────────────────────────────────────────────────────────
    # INFÉRENCE
    # ──────────────────────────────────────────────────────────────────────

    def predict(
        self,
        candles_by_symbol: Dict[str, pd.DataFrame],
        funding_by_symbol: Dict[str, pd.Series],
        ts: pd.Timestamp,
        fds_scores: Optional[Dict[str, float]] = None,
        regime_q: float = 1.0,
    ) -> Dict[str, float]:
        """
        Retourne directional_score ∈ [-1, 1] pour chaque symbole à ts.
        Si IC < min_ic_threshold ou modèle non entraîné → scores = 0 (signal éteint).
        """
        if not self._trained or self._xgb is None:
            return {sym: 0.0 for sym in candles_by_symbol}

        if self._last_ic < self.cfg.min_ic_threshold:
            logger.debug("[MLPredictor] IC faible — signal ML éteint")
            return {sym: 0.0 for sym in candles_by_symbol}

        cfg = self.cfg
        n = cfg.lstm.n_steps_in
        scores: Dict[str, float] = {}

        for symbol, candles in candles_by_symbol.items():
            mask = candles.index <= ts
            c_slice = candles.loc[mask].tail(n + 5)
            if len(c_slice) < n:
                scores[symbol] = 0.0
                continue

            funding = funding_by_symbol.get(symbol, pd.Series(dtype=float))
            seq_df = build_sequence_features(c_slice, funding, cfg)
            x_seq = seq_df.values[-n:].astype(np.float32)  # (n, n_features)

            # Embedding
            if TORCH_AVAILABLE and self._lstm is not None:
                xt = torch.tensor(x_seq[np.newaxis]).to(self.cfg.lstm.device)
                emb = self._lstm.extract_embedding(xt).numpy()  # (1, 64)
            else:
                emb = self._seq_summary(x_seq[np.newaxis])      # (1, summary_size)

            # Features tabulaires
            fds_s = fds_scores if fds_scores else {}
            x_tab = np.array([
                float(x_seq[-1, 0]),  # log_return
                float(x_seq[-1, 3]),  # vol_ratio
                float(x_seq[-1, 4]),  # funding
                float(x_seq[-1, 5]),  # funding_fast
                float(fds_s.get(symbol, 0.0)),
                float(regime_q),
            ], dtype=np.float32)[np.newaxis]

            X_input = np.hstack([emb, x_tab])
            raw = float(self._xgb.predict(X_input)[0])
            # Normaliser vers [-1, 1] via tanh
            scores[symbol] = float(np.tanh(raw / 2.0))

        return scores

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Batch embedding LSTM → (N, hidden_size)."""
        if self._lstm is None:
            return self._seq_summary(X)
        device = torch.device(self.cfg.lstm.device)
        xt = torch.tensor(X).to(device)
        with torch.no_grad():
            emb = self._lstm.extract_embedding(xt).cpu().numpy()
        return emb

    @staticmethod
    def _seq_summary(X: np.ndarray) -> np.ndarray:
        """
        Résumé statistique d'une séquence : mean, std, min, max, last par feature.
        Utilisé comme fallback si LSTM non disponible.
        Shape in  : (N, T, F)
        Shape out : (N, 5*F)
        """
        return np.hstack([
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            X[:, -1, :],  # dernier pas de temps
        ])

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def last_ic(self) -> float:
        return self._last_ic

    # ──────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._xgb is not None:
            self._xgb.save_model(str(path / "xgb.json"))
        if TORCH_AVAILABLE and self._lstm is not None:
            torch.save(self._lstm.state_dict(), str(path / "lstm.pt"))
        meta = {
            "trained": self._trained,
            "last_ic": self._last_ic,
            "symbols": self._symbols,
            "cfg": self.cfg,
        }
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"[MLPredictor] Sauvegardé dans {path}")

    @classmethod
    def load(cls, path: Path) -> "MLPredictor":
        path = Path(path)
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(meta["cfg"])
        obj._trained = meta["trained"]
        obj._last_ic = meta["last_ic"]
        obj._symbols = meta["symbols"]
        if XGB_AVAILABLE and (path / "xgb.json").exists():
            obj._xgb = xgb.XGBRegressor()
            obj._xgb.load_model(str(path / "xgb.json"))
        if TORCH_AVAILABLE and (path / "lstm.pt").exists():
            obj._lstm = _LSTMExtractor(meta["cfg"].lstm)
            obj._lstm.load_state_dict(
                torch.load(str(path / "lstm.pt"), map_location="cpu")
            )
            obj._lstm.eval()
        logger.info(f"[MLPredictor] Chargé depuis {path} — IC={obj._last_ic:.4f}")
        return obj
