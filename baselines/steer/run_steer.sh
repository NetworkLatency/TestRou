#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash baselines/steer/run_steer.sh

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SARR_CONFIG="${SARR_CONFIG:-configs/sarr_code_aggressive.json}"
STEER_CONFIG="${STEER_CONFIG:-baselines/steer/config.example.json}"
DATASET="${DATASET:-math500}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MODEL_PAIR="${MODEL_PAIR:-}"
RESUME="${RESUME:-1}"

cmd=(python baselines/steer/run_steer.py
  --sarr-config "${SARR_CONFIG}"
  --steer-config "${STEER_CONFIG}"
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
