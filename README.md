# GlimpRouter / DASR Validation

This repository now focuses on a compact validation path:

- strict GlimpRouter `H_init` routing sweep
- DASR sequence-level confidence `s_k` sweep
- same-format `SLM-only` and `LLM-only` baselines when needed

Old sampling-disagreement, evidence-consensus, boundary-continuation, and termination-signal diagnostics have been removed from the active code path.

## Setup

Install dependencies on the GPU host:

```bash
pip install -r requirements.txt
```

Edit `configs/bpa_default.json` for local model and dataset paths:

```json
{
  "slm_model_path": "/path/to/DeepSeek-R1-Distill-Qwen-1.5B",
  "llm_model_path": "/path/to/Qwen-or-DeepSeek-target-model",
  "slm_backend": "vllm",
  "llm_backend": "vllm",
  "dataset_paths": {
    "aime25": "data/aime25.parquet"
  }
}
```

OpenAI-compatible vLLM endpoints are supported through the `*_api_base_url`, `*_api_key`, and `*_api_model` fields. Keep a local tokenizer path configured when the remote endpoint does not expose tokenizer files.

Supported local dataset formats: `.jsonl`, `.json`, `.csv`, `.tsv`, `.parquet`.

For MATH/AIME rows, provide `problem` and one of `answer`, `solution`, or `target`.

## Run The Validation

Run the strict AIME25 validation:

```bash
python scripts/run_dasr_validation.py \
  --config configs/bpa_default.json \
  --dataset aime25 \
  --max-problems 30 \
  --stage both \
  --output-root bpa_results/dasr_validation_strict \
  --resume
```

Defaults are source-faithful for the GlimpRouter-style reproduction:

- `--step-token-budget 512`
- `--think-token-budget 8192`
- `--answer-token-budget 2048`
- `--baselines auto`
- `--baseline-protocol strict`

If you need to reuse externally generated baseline metrics, pass:

```bash
--slm-only-metrics path/to/slm_only/summary_metrics.json
--llm-only-metrics path/to/llm_only/summary_metrics.json
```

If those files are missing required fields, `--baselines auto` reruns the missing baseline in the same output format.

## Evaluate Results

Primary output:

```text
bpa_results/dasr_validation_strict/aime25/dasr_validation_gate_report.json
```

Quick view:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("bpa_results/dasr_validation_strict/aime25")
report = json.loads((root / "dasr_validation_gate_report.json").read_text())

print("Stage 1 GO:", report["stage1_gate"]["go"])
for row in report["stage1_gate"]["candidates"]:
    print("GLIMP", row)

print("\nStage 2 GO:", report["stage2_gate"]["go"])
for row in report["stage2_gate"]["pairs"]:
    if row["pass"]:
        print("PASS", row)
PY
```

Interpretation:

- Stage 1 `true`: at least one GlimpRouter threshold enters the Pareto improvement region.
- Stage 1 `false`: this model pair is weak for GlimpRouter-style routing.
- Stage 2 `true`: DASR `s_k` matches or beats GlimpRouter accuracy at comparable LLM call rate.
- Stage 2 `false`: sequence-level confidence alone did not beat `H_init` in this slice.

## Tests

Run static compilation and unit tests:

```bash
python -m compileall bpa scripts tests
python -m unittest tests.test_bpa_core
```

## Notes

Engineering ideas retained from removed diagnostics are summarized in `docs/engineering_notes.md`.
