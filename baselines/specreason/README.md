# SpecReason Baseline Adapter

This folder contains the project-local SpecReason comparison adapter.

Upstream repository: https://github.com/ruipeterpan/specreason

## Current Experiment Version

All experiments use only Qwen3 and DeepSeek-R1-Distill-Qwen model families. Generation and scoring use remote vLLM `chat.completions.create(messages=..., extra_body=...)` continuation calls, matching the GlimpRouter-style chat-template controls.

## Model Pairs

Use `--model-pair` to select one of:

- `qwen3_1p7b_qwen3_32b`
- `qwen3_1p7b_deepseek_qwen_32b`
- `deepseek_qwen_1p5b_qwen3_32b`
- `deepseek_qwen_1p5b_deepseek_qwen_32b`

Edit `config.example.json` to match your remote vLLM ports.

## Routing Logic

For each reasoning step:

1. Generate a candidate step with the small model.
2. Ask the base model to score the last reasoning step from 0 to 9.
3. Accept the small step if `score >= score_threshold`.
4. Otherwise generate the step with the base model from the accepted prefix.
5. Continue until an answer marker appears, repeated steps are detected, or the token budget is exhausted.

## Run

```bash
python baselines/specreason/run_specreason.py \
  --sarr-config configs/sarr_code_aggressive.json \
  --specreason-config baselines/specreason/config.example.json \
  --dataset aime24 \
  --model-pair qwen3_1p7b_qwen3_32b \
  --resume
```

Outputs are written under `specreason_results/<dataset>/<variant>/` with `summary.csv`, `summary_metrics.json`, and per-problem metadata.
