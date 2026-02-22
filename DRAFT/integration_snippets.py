"""
SNIPPETS D'INTÉGRATION
======================
Ces extraits montrent où et comment intégrer credentials.py et universe_filter.py
dans les fichiers existants du repo.

NE PAS copier ce fichier entier dans le repo — ce sont des extraits ciblés.
"""

# ═══════════════════════════════════════════════════════════════════════
# 1. apps/dashboard.py — ajouter en haut du fichier
# ═══════════════════════════════════════════════════════════════════════

DASHBOARD_SNIPPET = '''
# apps/dashboard.py  — modifications à apporter

import streamlit as st
from hyperstat.core.credentials import render_credentials_sidebar, resolve_credentials

st.set_page_config(page_title="HyperStat Dashboard", layout="wide")

# ── Connexion ────────────────────────────────────────────────────────
connected = render_credentials_sidebar()
if not connected:
    st.info("👈 Entrez vos credentials Hyperliquid dans la sidebar pour démarrer.")
    st.stop()

# Résolution (session_state prioritaire sur env vars)
creds = resolve_credentials(streamlit_session=st.session_state)

# Injecter dans le client exchange
from hyperstat.exchange.hyperliquid.rest_client import HyperliquidRestClient
client = HyperliquidRestClient(
    address=creds.address,
    private_key=creds.private_key,
    testnet=creds.testnet,
)

# ... reste du dashboard inchangé
'''

# ═══════════════════════════════════════════════════════════════════════
# 2. hyperstat/cli/commands.py — ajouter sur le sous-parser 'live' et 'paper'
# ═══════════════════════════════════════════════════════════════════════

CLI_SNIPPET = '''
# src/hyperstat/cli/commands.py  — modifications à apporter

import argparse
from hyperstat.core.credentials import add_credential_args, resolve_credentials

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperstat")
    sub = parser.add_subparsers(dest="command")

    # Sous-commande 'live'
    live_parser = sub.add_parser("live", help="Trading live sur Hyperliquid")
    add_credential_args(live_parser)     # ← ajoute --hl-address, --hl-private-key, --testnet
    live_parser.add_argument("--config", action="append", default=[])

    # Sous-commande 'paper'
    paper_parser = sub.add_parser("paper", help="Paper trading (sans exécution réelle)")
    add_credential_args(paper_parser)
    paper_parser.add_argument("--config", action="append", default=[])

    return parser


def run_live(args: argparse.Namespace) -> None:
    creds = resolve_credentials(cli_args=args)
    print(f"[live] Connecté : {creds.masked()}")
    # ... reste de la logique live inchangée


# Exemple d\'appel CLI :
# python -m hyperstat.main live \\
#     --hl-address 0xYourAddress \\
#     --hl-private-key 0xYourPrivateKey \\
#     --config configs/default.yaml \\
#     --config configs/hyperliquid_mainnet.yaml
'''

# ═══════════════════════════════════════════════════════════════════════
# 3. scripts/build_universe.py — intégrer le filtre dynamique
# ═══════════════════════════════════════════════════════════════════════

BUILD_UNIVERSE_SNIPPET = '''
# scripts/build_universe.py  — modifications à apporter

from hyperstat.data.universe_filter import (
    UniverseFilter, UniverseFilterConfig,
    build_meta_from_hyperliquid, enrich_market_cap_coingecko,
)

def main(config_path: str) -> None:
    # ... chargement config existant ...

    # 1. Récupérer les métadonnées depuis Hyperliquid
    raw_meta  = hl_client.get("/info", {"type": "meta"})["universe"]
    meta_df   = build_meta_from_hyperliquid(raw_meta)

    # 2. Enrichir avec les market caps CoinGecko (1 appel HTTP)
    meta_df   = enrich_market_cap_coingecko(meta_df)

    # 3. Filtrer l\'univers
    uf_cfg = UniverseFilterConfig(
        funding_frequency="hourly",     # ← perps à funding horaire uniquement
        min_market_cap_usd=50_000_000,  # $50M
        min_adv_usd=500_000,            # $500K ADV
        max_amihud=1e-6,
        min_funding_std=3e-5,
        min_history_bars=500,
    )
    uf = UniverseFilter(uf_cfg)
    selected_symbols, report = uf.filter(candles_by_symbol, funding_by_symbol, meta_df)

    print(report.summary())
    # Sauvegarder le rapport pour audit
    report.as_dataframe().to_csv("artifacts/universe_filter_report.csv", index=False)

    # 4. Passer les symboles filtrés au clustering de buckets
    filtered_candles  = {s: candles_by_symbol[s]  for s in selected_symbols}
    filtered_funding  = {s: funding_by_symbol[s]  for s in selected_symbols}
    buckets = build_buckets(filtered_candles, config)
    # ... suite inchangée
'''

# ═══════════════════════════════════════════════════════════════════════
# 4. Ajout dans configs/default.yaml — section universe_filter
# ═══════════════════════════════════════════════════════════════════════

YAML_SNIPPET = '''
# Ajouter dans configs/default.yaml

universe_filter:
  funding_frequency: "hourly"       # "hourly" | "8h" | "any"
  min_market_cap_usd: 50_000_000    # $50M
  min_adv_usd: 500_000              # $500K
  max_amihud: 1.0e-6
  min_funding_std: 3.0e-5
  min_funding_abs_mean: 1.0e-5
  min_history_bars: 500
  exclude_symbols:
    - USDC
    - USDT
    - BUSD
    - DAI
    - TUSD
  verbose: true
'''

if __name__ == "__main__":
    print(DASHBOARD_SNIPPET)
    print(CLI_SNIPPET)
    print(BUILD_UNIVERSE_SNIPPET)
    print(YAML_SNIPPET)
