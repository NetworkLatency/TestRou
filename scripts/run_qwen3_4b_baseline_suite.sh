#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash scripts/run_qwen3_4b_baseline_suite.sh

CONFIG="${CONFIG:-configs/sarr_code_aggressive.json}"
DATASETS="${DATASETS:-aime25 aime24 gpqa humaneval}"
MODEL_ROLE="${MODEL_ROLE:-llm}"
VARIANT_PREFIX="${VARIANT_PREFIX:-qwen3_4b_single_baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-}"
RESUME="${RESUME:-1}"

if [[ -n "${HUMANEVAL_PATH}" ]]; then
  TMP_CONFIG="$(mktemp /tmp/qwen3_4b_baseline_config.XXXXXX.json)"
  python -c "import json; p='${CONFIG}'; h='${HUMANEVAL_PATH}'; o='${TMP_CONFIG}'; d=json.load(open(p, encoding='utf-8')); paths=dict(d.get('dataset_paths') or {}); paths['humaneval']=h; d['dataset_paths']=paths; json.dump(d, open(o, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)"
  CONFIG="${TMP_CONFIG}"
  trap 'rm -f "${TMP_CONFIG}"' EXIT
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="$(python -c "import json; print(json.load(open('${CONFIG}', encoding='utf-8')).get('output_dir', 'sarr_results'))")"
fi

for dataset in ${DATASETS}; do
  variant="${VARIANT_PREFIX}_${dataset}"
  cmd=(python scripts/run_single_model_baseline.py
    --config "${CONFIG}"
    --dataset "${dataset}"
    --model-role "${MODEL_ROLE}"
    --variant "${variant}"
    --output-root "${OUTPUT_ROOT}")

  if [[ -n "${MAX_PROBLEMS}" ]]; then
    cmd+=(--max-problems "${MAX_PROBLEMS}")
  fi

  if [[ -n "${MAX_NEW_TOKENS}" ]]; then
    cmd+=(--max-new-tokens "${MAX_NEW_TOKENS}")
  fi

  if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
  fi

  echo
  echo "[qwen3-4b-baseline] ${cmd[*]}"
  "${cmd[@]}"
done

python -c '
import csv
import json
from pathlib import Path

datasets = "'"${DATASETS}"'".split()
output_root = Path("'"${OUTPUT_ROOT}"'")
variant_prefix = "'"${VARIANT_PREFIX}"'"
rows = []
for dataset in datasets:
    variant = f"{variant_prefix}_{dataset}"
    metrics_path = output_root / dataset / variant / "summary_metrics.json"
    if not metrics_path.exists():
        rows.append({
            "dataset": dataset,
            "variant": variant,
            "metrics_path": str(metrics_path),
            "error": "missing summary_metrics.json",
        })
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    llm_decode = metrics.get("total_llm_decode_tokens") or 0
    llm_prefill = metrics.get("total_llm_prefill_tokens") or 0
    slm_decode = metrics.get("total_slm_decode_tokens") or 0
    slm_prefill = metrics.get("total_slm_prefill_tokens") or 0
    rows.append({
        "dataset": dataset,
        "variant": variant,
        "accuracy": metrics.get("accuracy"),
        "num_evaluated": metrics.get("num_evaluated"),
        "num_correct": metrics.get("num_correct"),
        "dataset_wall_time": metrics.get("dataset_wall_time"),
        "avg_problem_wall_time": metrics.get("avg_problem_wall_time"),
        "total_decode_tokens": llm_decode + slm_decode,
        "total_prefill_tokens": llm_prefill + slm_prefill,
        "total_tokens": llm_decode + slm_decode + llm_prefill + slm_prefill,
        "llm_decode_tokens": llm_decode,
        "llm_prefill_tokens": llm_prefill,
        "slm_decode_tokens": slm_decode,
        "slm_prefill_tokens": slm_prefill,
        "metrics_path": str(metrics_path),
        "error": "",
    })

summary_dir = output_root / "baseline_suites"
summary_dir.mkdir(parents=True, exist_ok=True)
summary_path = summary_dir / f"{variant_prefix}_metrics.csv"
fieldnames = [
    "dataset", "variant", "accuracy", "num_evaluated", "num_correct",
    "dataset_wall_time", "avg_problem_wall_time",
    "total_decode_tokens", "total_prefill_tokens", "total_tokens",
    "llm_decode_tokens", "llm_prefill_tokens", "slm_decode_tokens", "slm_prefill_tokens",
    "metrics_path", "error",
]
with summary_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"[qwen3-4b-baseline] aggregate metrics: {summary_path}")
for row in rows:
    print(
        f"{row['dataset']}: accuracy={row.get('accuracy')} "
        f"time={row.get('dataset_wall_time')}s "
        f"tokens={row.get('total_tokens')} "
        f"metrics={row.get('metrics_path')}"
    )
'
