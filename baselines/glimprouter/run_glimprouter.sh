#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash baselines/glimprouter/run_glimprouter.sh

SARR_CONFIG="${SARR_CONFIG:-configs/sarr_code_aggressive.json}"
ROUTER_CONFIG="${ROUTER_CONFIG:-baselines/glimprouter/config.example.json}"
DATASET="${DATASET:-aime25}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MODEL_PAIR="${MODEL_PAIR:-}"
RESUME="${RESUME:-1}"

cmd=(python baselines/glimprouter/run_glimprouter.py
  --sarr-config "${SARR_CONFIG}"
  --router-config "${ROUTER_CONFIG}"
  --dataset "${DATASET}")

if [[ -n "${MAX_PROBLEMS}" ]]; then
  cmd+=(--max-problems "${MAX_PROBLEMS}")
fi

if [[ -n "${MODEL_PAIR}" ]]; then
  cmd+=(--model-pair "${MODEL_PAIR}")
fi

if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi

echo "${cmd[@]}"
"${cmd[@]}"
