# GlimpRouter Baseline Adapter

This folder contains the project-local GlimpRouter comparison adapter.

Upstream repository: https://github.com/Zengwh02/GlimpRouter

## Current Experiment Version

All experiments use only Qwen3 and DeepSeek-R1-Distill-Qwen model families. GlimpRouter already uses remote vLLM `chat.completions.create(messages=..., extra_body=...)`; this adapter exposes the same four model pairs used by the other comparison methods.

## Model Pairs

Use `--model-pair` to select one of:

- `qwen3_1p7b_qwen3_32b`
- `qwen3_1p7b_deepseek_qwen_32b`
- `deepseek_qwen_1p5b_qwen3_32b`
- `deepseek_qwen_1p5b_deepseek_qwen_32b`

Edit `config.example.json` to match your remote vLLM ports.

## Budget

GlimpRouter keeps the explicit split discussed in the paper-style setup:

- reasoning/think budget: `token_budget=14336`
- answer budget: `answer_max_tokens=2048`

## Routing Logic

For each reasoning step:

1. Score the small model's next-token uncertainty using first-token entropy.
2. Route the next step to the base model if entropy is above `score_threshold`.
3. Otherwise route the step to the small model.
4. Stop reasoning on answer markers or budget, then generate the final answer with the base model.

## Run

```bash
python baselines/glimprouter/run_glimprouter.py \
  --sarr-config configs/sarr_code_aggressive.json \
  --router-config baselines/glimprouter/config.example.json \
  --dataset aime25 \
  --model-pair qwen3_1p7b_qwen3_32b \
  --resume
```

Wrapper example:

```bash
DATASET=gpqa MODEL_PAIR=deepseek_qwen_1p5b_deepseek_qwen_32b bash baselines/glimprouter/run_glimprouter.sh
```

Outputs are written under `glimprouter_results/<dataset>/<variant>/` with `summary.csv`, `summary_metrics.json`, and per-problem metadata.
