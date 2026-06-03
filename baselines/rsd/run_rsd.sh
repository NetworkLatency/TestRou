#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash baselines/rsd/run_rsd.sh

SARR_CONFIG="${SARR_CONFIG:-configs/sarr_code_aggressive.json}"
RSD_CONFIG="${RSD_CONFIG:-baselines/rsd/config.example.json}"
DATASET="${DATASET:-aime24}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MODEL_PAIR="${MODEL_PAIR:-}"
RESUME="${RESUME:-1}"

cmd=(python baselines/rsd/run_rsd.py
  --sarr-config "${SARR_CONFIG}"
  --rsd-config "${RSD_CONFIG}"
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
