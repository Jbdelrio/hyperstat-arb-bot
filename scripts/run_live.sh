#!/usr/bin/env bash
# scripts/run_live.sh
set -euo pipefail

# Usage:
#   ./scripts/run_live.sh paper testnet my_run
#   ./scripts/run_live.sh live  mainnet my_run
#
# Requires:
#   - .env populated (HL_ADDRESS, HL_PRIVATE_KEY)
#   - pip install -e .

MODE="${1:-paper}"        # paper | live
NET="${2:-testnet}"       # testnet | mainnet
RUN_NAME="${3:-default}"  # artifacts/live/<RUN_NAME>/

if [[ "$MODE" != "paper" && "$MODE" != "live" ]]; then
  echo "MODE must be paper|live"
  exit 1
fi

if [[ "$NET" != "testnet" && "$NET" != "mainnet" ]]; then
  echo "NET must be testnet|mainnet"
  exit 1
fi

CFG_NET="configs/hyperliquid_${NET}.yaml"

echo "Running HyperStat: mode=$MODE net=$NET run=$RUN_NAME"
echo "Using configs: configs/default.yaml + $CFG_NET + configs/strategy_stat_arb.yaml"

if [[ "$MODE" == "live" ]]; then
  hyperstat --config configs/default.yaml --config "$CFG_NET" --config configs/strategy_stat_arb.yaml \
    live --run-name "$RUN_NAME" --live
else
  hyperstat --config configs/default.yaml --config "$CFG_NET" --config configs/strategy_stat_arb.yaml \
    live --run-name "$RUN_NAME"
fi
