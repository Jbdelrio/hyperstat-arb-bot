"""
scripts/walk_forward_validation.py
====================================
Script de walk-forward validation pour les modèles ML du PredictionAgent.

Usage :
    python scripts/walk_forward_validation.py \
        --data ./artifacts/data/candles \
        --symbols BTC ETH SOL \
        --window 30 \
        --step 7 \
        --output ./artifacts/wf_results.json

Train/Test split :
    Fenêtre glissante de `window` jours, avançant de `step` jours à chaque itération.
    Jamais de look-ahead biais : test toujours sur données futures par rapport au train.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_candles(data_dir: str, symbol: str) -> pd.DataFrame:
    """Charge les candles Parquet pour un symbole."""
    p = Path(data_dir)
    candidates = list(p.glob(f"{symbol}*.parquet")) + list(p.glob(f"{symbol}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Pas de données pour {symbol} dans {data_dir}")
    fp = candidates[0]
    if fp.suffix == ".parquet":
        df = pd.read_parquet(fp)
    else:
        df = pd.read_csv(fp)
    # Normalise le nom de la colonne timestamp
    for col in ["ts", "timestamp", "time", "open_time"]:
        if col in df.columns:
            df["ts"] = pd.to_datetime(df[col])
            break
    df = df.set_index("ts").sort_index()
    return df


def run_walk_forward(
    df       : pd.DataFrame,
    symbol   : str,
    window_d : int = 30,
    step_d   : int = 7,
    fwd_h    : int = 12,
    test_r   : float = 0.20,
) -> dict:
    """
    Exécute la walk-forward validation pour un symbole.

    Parameters
    ----------
    df       : DataFrame OHLCV
    symbol   : nom du symbole
    window_d : taille fenêtre train en jours
    step_d   : pas d'avancement en jours
    fwd_h    : horizon de prédiction en barres
    test_r   : fraction test dans chaque fenêtre

    Returns
    -------
    dict avec métriques agrégées et détail par fenêtre
    """
    from hyperstat.agents.prediction_agent import (
        PredictionAgent, PredictionConfig, _build_ml_features, _build_target
    )

    bars_per_day  = 288      # 288 barres × 5min = 24h
    window_bars   = window_d * bars_per_day
    step_bars     = step_d   * bars_per_day

    cfg    = PredictionConfig(forward_horizon=fwd_h, test_ratio=test_r)
    agent  = PredictionAgent(symbols=[symbol], cfg=cfg)

    windows     = []
    accuracies  = []
    n           = len(df)
    win_idx     = 0
    start       = 0

    while start + window_bars + step_bars < n:
        win_idx += 1
        train_df  = df.iloc[start : start + window_bars]
        test_df   = df.iloc[start + window_bars : start + window_bars + step_bars]

        train_start = train_df.index[0].strftime("%Y-%m-%d")
        train_end   = train_df.index[-1].strftime("%Y-%m-%d")
        test_end    = test_df.index[-1].strftime("%Y-%m-%d")

        try:
            # Entraînement sur la fenêtre train
            res = agent._train_symbol(symbol, train_df, test_ratio=0.0)

            # Évaluation sur la fenêtre test
            feats   = _build_ml_features(test_df).fillna(0.0)
            target  = _build_target(test_df["close"], fwd_h)
            valid   = feats.dropna().index.intersection(target.dropna().index)

            if len(valid) < 20:
                logger.warning(f"  Fenêtre {win_idx}: trop peu de données valides ({len(valid)})")
                start += step_bars
                continue

            _, xgb_model = agent._models[symbol]
            X_test   = feats.loc[valid].values
            y_test   = target.loc[valid].values
            preds    = (xgb_model.predict_proba(X_test) > 0.5).astype(int)
            acc      = float((preds == y_test).mean())

            # Calcul Brier Score
            proba    = xgb_model.predict_proba(X_test)
            brier    = float(np.mean((proba - y_test) ** 2))

            win_result = {
                "window"      : win_idx,
                "train_start" : train_start,
                "train_end"   : train_end,
                "test_end"    : test_end,
                "n_train"     : len(train_df),
                "n_test"      : len(valid),
                "accuracy"    : round(acc, 4),
                "brier_score" : round(brier, 4),
            }
            windows.append(win_result)
            accuracies.append(acc)

            logger.info(
                f"  Fenêtre {win_idx:2d} | train [{train_start}→{train_end}] "
                f"test [{train_end}→{test_end}] | acc={acc:.3f} brier={brier:.3f}"
            )

        except Exception as exc:
            logger.error(f"  Fenêtre {win_idx} erreur: {exc}")

        start += step_bars

    if not accuracies:
        return {"symbol": symbol, "status": "no_valid_windows", "windows": []}

    return {
        "symbol"          : symbol,
        "status"          : "ok",
        "mean_accuracy"   : round(float(np.mean(accuracies)), 4),
        "std_accuracy"    : round(float(np.std(accuracies)), 4),
        "min_accuracy"    : round(float(np.min(accuracies)), 4),
        "max_accuracy"    : round(float(np.max(accuracies)), 4),
        "n_windows"       : len(accuracies),
        "above_55pct"     : sum(1 for a in accuracies if a > 0.55),
        "target_met"      : float(np.mean(accuracies)) > 0.52,
        "windows"         : windows,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation HyperStat")
    parser.add_argument("--data",    default="./artifacts/data/candles", help="Dossier données Parquet")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"], help="Symboles à valider")
    parser.add_argument("--window",  type=int, default=30,  help="Taille fenêtre train (jours)")
    parser.add_argument("--step",    type=int, default=7,   help="Pas d'avancement (jours)")
    parser.add_argument("--fwd",     type=int, default=12,  help="Horizon prédiction (barres)")
    parser.add_argument("--output",  default="./artifacts/wf_results.json", help="Fichier résultats JSON")
    args = parser.parse_args()

    results = {}
    for sym in args.symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-forward validation : {sym}")
        logger.info(f"  Window={args.window}j  Step={args.step}j  FwdHorizon={args.fwd}barres")
        logger.info(f"{'='*60}")
        try:
            df     = load_candles(args.data, sym)
            logger.info(f"  Données chargées : {len(df)} barres ({df.index[0]} → {df.index[-1]})")
            result = run_walk_forward(df, sym, args.window, args.step, args.fwd)
            results[sym] = result

            # Résumé
            logger.info(f"\n  ── Résumé {sym} ──")
            logger.info(f"  Fenêtres validées : {result.get('n_windows', 0)}")
            logger.info(f"  Accuracy moyenne  : {result.get('mean_accuracy', 0):.3f} "
                        f"± {result.get('std_accuracy', 0):.3f}")
            logger.info(f"  Objectif > 52%    : {'✅ ATTEINT' if result.get('target_met') else '❌ NON ATTEINT'}")

        except Exception as exc:
            logger.error(f"  Erreur {sym}: {exc}")
            results[sym] = {"symbol": sym, "status": "error", "error": str(exc)}

    # Sauvegarde JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results["_meta"] = {
        "generated_at" : datetime.utcnow().isoformat(),
        "window_days"  : args.window,
        "step_days"    : args.step,
        "fwd_horizon"  : args.fwd,
        "symbols"      : args.symbols,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Résultats sauvegardés dans {out_path}")

    # Résumé global
    logger.info("\n" + "="*60)
    logger.info("RÉSUMÉ GLOBAL")
    logger.info("="*60)
    for sym, res in results.items():
        if sym == "_meta":
            continue
        status = "✅" if res.get("target_met") else "❌"
        acc    = res.get("mean_accuracy", 0)
        n      = res.get("n_windows", 0)
        logger.info(f"  {status} {sym:10s} acc={acc:.3f}  ({n} fenêtres)")


if __name__ == "__main__":
    main()
