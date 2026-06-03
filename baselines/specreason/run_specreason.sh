#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash baselines/specreason/run_specreason.sh

SARR_CONFIG="${SARR_CONFIG:-configs/sarr_code_aggressive.json}"
SPECREASON_CONFIG="${SPECREASON_CONFIG:-baselines/specreason/config.example.json}"
DATASET="${DATASET:-aime24}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MODEL_PAIR="${MODEL_PAIR:-}"
RESUME="${RESUME:-1}"

cmd=(python baselines/specreason/run_specreason.py
  --sarr-config "${SARR_CONFIG}"
  --specreason-config "${SPECREASON_CONFIG}"
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
