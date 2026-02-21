#!/usr/bin/env python3
# scripts/fetch_candles.py
"""
Récupération de données historiques (candles + funding) depuis l'API Hyperliquid.
Script autonome — pas besoin de clé API, l'endpoint /info est public.

Usage rapide :
  python scripts/fetch_candles.py --coins ETH,SOL,ARB,OP,AVAX --tf 5m --days 30
  python scripts/fetch_candles.py --coins BTC,ETH --tf 1m --days 7 --network mainnet

Résultat : fichiers parquet dans data/candles/<COIN>/5m.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ajoute src/ au path pour les imports hyperstat
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import httpx

# ─── Config ───────────────────────────────────────────────────────────────────
MAINNET_URL  = "https://api.hyperliquid.xyz/info"
TESTNET_URL  = "https://api.hyperliquid-testnet.xyz/info"
MAX_CANDLES_PER_CALL = 500          # limite empirique Hyperliquid
WEIGHT_PER_CALL      = 20           # poids d'un appel /info
WEIGHT_BUDGET_MIN    = 1100         # budget conservateur (max 1200/min)
MIN_SLEEP_S          = 60.0 / (WEIGHT_BUDGET_MIN / WEIGHT_PER_CALL)  # ~1.1s entre appels


# ─── HTTP brut (sans dépendance hyperstat) ────────────────────────────────────
async def _post(client: httpx.AsyncClient, url: str, body: dict, retries: int = 5) -> dict:
    backoff = 0.5
    last_exc = None
    for attempt in range(retries):
        try:
            r = await client.post(url, json=body, timeout=15.0)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = backoff * (2 ** attempt)
                print(f"  ⚠️  HTTP {r.status_code} — retry dans {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
            last_exc = e
            wait = backoff * (2 ** attempt)
            print(f"  ⚠️  Erreur réseau ({e}) — retry dans {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Échec après {retries} tentatives : {last_exc}")


# ─── Normalisation des candles ────────────────────────────────────────────────
def _normalize_candles(raw) -> pd.DataFrame:
    """
    Hyperliquid renvoie une liste de dicts avec les clés :
      t (timestamp ms), o, h, l, c (OHLC strings), v (volume), n (nb trades)
    """
    arr = raw if isinstance(raw, list) else []
    if not arr:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume","trades"])

    df = pd.DataFrame(arr)
    df = df.rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume","n":"trades"})
    df["ts"]     = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    for col in ("open","high","low","close","volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce")

    keep = [c for c in ["ts","open","high","low","close","volume","trades"] if c in df.columns]
    df = df[keep].dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"])
    return df


# ─── Normalisation du funding ─────────────────────────────────────────────────
def _normalize_funding(raw) -> pd.DataFrame:
    """
    Hyperliquid renvoie une liste de dicts :
      time (ms), coin, fundingRate (string)
    """
    arr = raw if isinstance(raw, list) else []
    if not arr:
        return pd.DataFrame(columns=["ts","rate"])

    df = pd.DataFrame(arr)
    if "time" in df.columns:
        df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
    if "fundingRate" in df.columns:
        df["rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

    keep = [c for c in ["ts","rate"] if c in df.columns]
    df = df[keep].dropna(subset=["ts","rate"]).sort_values("ts").drop_duplicates(subset=["ts"])
    return df


# ─── Pagination candles ───────────────────────────────────────────────────────
async def fetch_candles(
    client: httpx.AsyncClient,
    url: str,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pagine les appels candleSnapshot jusqu'à couvrir [start_ms, end_ms].
    Chaque appel retourne au max ~500 bougies.
    """
    cur = int(start_ms)
    frames: list[pd.DataFrame] = []
    call_count = 0

    while cur < end_ms:
        body = {
            "type": "candleSnapshot",
            "req": {
                "coin":      coin,
                "interval":  interval,
                "startTime": cur,
                "endTime":   end_ms,
            }
        }

        raw = await _post(client, url, body)
        df  = _normalize_candles(raw)
        call_count += 1

        if df.empty:
            break

        frames.append(df)
        n_rows = len(df)

        # Dernière ts reçue → avancer le curseur
        last_ms = int(df["ts"].iloc[-1].value // 10**6)
        nxt = last_ms + 1
        if nxt <= cur:
            break
        cur = nxt

        if verbose:
            pct = min(100, (last_ms - start_ms) / max(1, end_ms - start_ms) * 100)
            print(f"    {coin} [{interval}] — appel #{call_count:3d} | "
                  f"+{n_rows} bougies | dernier: {df['ts'].iloc[-1].strftime('%Y-%m-%d %H:%M')} "
                  f"| {pct:.0f}%")

        # Respecter le rate limit : ~1.1s entre appels
        await asyncio.sleep(MIN_SLEEP_S)

        # Si on a reçu très peu de bougies : probablement à la fin
        if n_rows < 10:
            break

    if not frames:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
    return full


# ─── Pagination funding ───────────────────────────────────────────────────────
async def fetch_funding(
    client: httpx.AsyncClient,
    url: str,
    coin: str,
    start_ms: int,
    end_ms: int,
    verbose: bool = True,
) -> pd.DataFrame:
    cur = int(start_ms)
    frames = []

    while cur < end_ms:
        body = {
            "type":      "fundingHistory",
            "coin":      coin,
            "startTime": cur,
            "endTime":   end_ms,
        }

        raw = await _post(client, url, body)
        df  = _normalize_funding(raw)

        if df.empty:
            break

        frames.append(df)
        last_ms = int(df["ts"].iloc[-1].value // 10**6)
        nxt = last_ms + 1
        if nxt <= cur:
            break
        cur = nxt

        await asyncio.sleep(MIN_SLEEP_S)

        if len(df) < 5:
            break

    if not frames:
        return pd.DataFrame(columns=["ts","rate"])

    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["ts","rate"]).sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)

    if verbose:
        print(f"    {coin} [funding] — {len(full)} ticks")

    return full


# ─── Sauvegarde ───────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True, errors="coerce")
            df = pd.concat([existing, df], ignore_index=True)
            df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
        except Exception:
            pass
    df.to_parquet(path, index=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = pd.read_csv(path)
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True, errors="coerce")
            df = pd.concat([existing, df], ignore_index=True)
            df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
        except Exception:
            pass
    df.to_csv(path, index=False)


# ─── Main ─────────────────────────────────────────────────────────────────────
async def main_async(args: argparse.Namespace) -> int:
    url     = MAINNET_URL if args.network == "mainnet" else TESTNET_URL
    coins   = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    now_ms  = int(time.time() * 1000)
    start_ms = now_ms - int(args.days * 24 * 3600 * 1000)
    end_ms   = now_ms
    fmt      = args.format.lower()

    print(f"\n🚀 HyperStat — fetch_candles.py")
    print(f"   Network  : {args.network} ({url})")
    print(f"   Coins    : {coins}")
    print(f"   Interval : {args.tf}")
    print(f"   Période  : {args.days} jours ({pd.Timestamp(start_ms, unit='ms', tz='UTC').strftime('%Y-%m-%d')} → aujourd'hui)")
    print(f"   Format   : {fmt}")
    print(f"   Data dir : {args.data_dir}\n")

    headers = {
        "Content-Type": "application/json",
        "User-Agent":   "hyperstat-fetch/0.1",
    }

    async with httpx.AsyncClient(headers=headers) as client:

        # ── Vérification rapide : liste des coins disponibles ──────────────
        print("📋 Vérification des coins disponibles sur Hyperliquid...")
        try:
            meta_raw = await _post(client, url, {"type": "meta"})
            available = {u["name"].upper() for u in meta_raw.get("universe", [])}
            print(f"   {len(available)} coins trouvés sur {args.network}")

            invalid = [c for c in coins if c not in available]
            if invalid:
                print(f"   ⚠️  Coins non trouvés sur {args.network} : {invalid}")
                print(f"   Coins disponibles (extrait) : {sorted(list(available))[:30]}")
                coins = [c for c in coins if c in available]
                if not coins:
                    print("   ❌ Aucun coin valide. Arrêt.")
                    return 1
        except Exception as e:
            print(f"   ⚠️  Impossible de vérifier les coins : {e} — on continue quand même")

        # ── Boucle par coin ────────────────────────────────────────────────
        results = {}

        for coin in coins:
            print(f"\n{'='*60}")
            print(f"⬇️  {coin}")

            # Candles
            print(f"  Candles [{args.tf}]...")
            try:
                df_candles = await fetch_candles(
                    client, url, coin, args.tf, start_ms, end_ms, verbose=True
                )
                results[coin] = df_candles

                if not df_candles.empty:
                    out_path = Path(args.data_dir) / "candles" / coin / f"{args.tf}.{fmt}"
                    if fmt == "parquet":
                        save_parquet(df_candles, out_path)
                    else:
                        save_csv(df_candles, out_path)

                    span = (df_candles["ts"].iloc[-1] - df_candles["ts"].iloc[0])
                    print(f"  ✅ {len(df_candles):,} bougies | "
                          f"{df_candles['ts'].iloc[0].strftime('%Y-%m-%d')} → "
                          f"{df_candles['ts'].iloc[-1].strftime('%Y-%m-%d')} "
                          f"({span.days}j) → {out_path}")
                else:
                    print(f"  ⚠️  Aucune donnée reçue pour {coin}")

            except Exception as e:
                print(f"  ❌ Erreur candles {coin} : {e}")

            # Funding (optionnel)
            if args.funding:
                print(f"  Funding [8h]...")
                try:
                    df_fund = await fetch_funding(client, url, coin, start_ms, end_ms, verbose=True)
                    if not df_fund.empty:
                        out_path = Path(args.data_dir) / "funding" / coin / f"8h.{fmt}"
                        if fmt == "parquet":
                            save_parquet(df_fund, out_path)
                        else:
                            save_csv(df_fund, out_path)
                        print(f"  ✅ {len(df_fund)} ticks funding → {out_path}")
                except Exception as e:
                    print(f"  ❌ Erreur funding {coin} : {e}")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📦 Résumé :")
    total_rows = 0
    for coin, df in results.items():
        n = len(df) if df is not None and not df.empty else 0
        total_rows += n
        status = "✅" if n > 0 else "❌"
        print(f"  {status} {coin:8s} : {n:6,} bougies")
    print(f"\n  Total : {total_rows:,} bougies pour {len(coins)} coins")
    print(f"  Dossier : {Path(args.data_dir).resolve()}")
    print(f"\n🎉 Terminé !\n")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Télécharge les données historiques Hyperliquid (candles + funding)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # 30 jours de bougies 5m pour 6 altcoins
  python scripts/fetch_candles.py --coins ETH,SOL,ARB,OP,AVAX,BTC --tf 5m --days 30

  # 7 jours de bougies 1m + funding pour ETH et SOL
  python scripts/fetch_candles.py --coins ETH,SOL --tf 1m --days 7 --funding

  # Sur le testnet
  python scripts/fetch_candles.py --coins ETH,BTC --tf 5m --days 7 --network testnet

  # Format CSV au lieu de parquet
  python scripts/fetch_candles.py --coins ETH --tf 15m --days 14 --format csv
        """,
    )
    p.add_argument("--coins",    required=True, help="Coins séparés par virgule, ex: ETH,SOL,ARB")
    p.add_argument("--tf",       default="5m",  help="Timeframe : 1m, 3m, 5m, 15m, 1h, ... (défaut: 5m)")
    p.add_argument("--days",     type=float, default=30, help="Nombre de jours à télécharger (défaut: 30)")
    p.add_argument("--network",  choices=["mainnet","testnet"], default="mainnet", help="mainnet ou testnet")
    p.add_argument("--funding",  action="store_true", help="Télécharger aussi le funding 8h")
    p.add_argument("--data-dir", default="data", help="Dossier de sortie (défaut: data/)")
    p.add_argument("--format",   choices=["parquet","csv"], default="parquet", help="Format de sortie")
    args = p.parse_args()

    # Vérification rapide du timeframe
    valid_tfs = {"1m","3m","5m","15m","30m","1h","2h","4h","8h","12h","1d"}
    if args.tf not in valid_tfs:
        print(f"❌ Timeframe '{args.tf}' non supporté. Valides : {sorted(valid_tfs)}")
        return 1

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
