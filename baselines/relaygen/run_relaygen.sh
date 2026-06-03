#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash baselines/relaygen/run_relaygen.sh

SARR_CONFIG="${SARR_CONFIG:-configs/sarr_code_aggressive.json}"
RELAYGEN_CONFIG="${RELAYGEN_CONFIG:-baselines/relaygen/config.example.json}"
DATASET="${DATASET:-aime25}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MODEL_PAIR="${MODEL_PAIR:-}"
RESUME="${RESUME:-1}"

cmd=(python baselines/relaygen/run_relaygen.py
  --sarr-config "${SARR_CONFIG}"
  --relaygen-config "${RELAYGEN_CONFIG}"
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
