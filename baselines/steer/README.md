# STEER Baseline Adapter

This folder contains the project-local STEER comparison adapter for the current Qwen3 / DeepSeek-R1-Distill-Qwen experiments.

Upstream repository: https://github.com/helmsman12/STEER

## Current Experiment Version

The original STEER implementation needs local vLLM internals that expose raw top logits. The current experiments require remote OpenAI-compatible vLLM chat services, so this adapter uses `chat.completions.create(messages=..., extra_body=...)` and computes a routing confidence from returned `logprobs/top_logprobs`.

This preserves the STEER stepwise GMM routing structure while making it runnable under the same remote chat API as GlimpRouter.

## Model Pairs

Use `--model-pair` to select one of:

- `qwen3_1p7b_qwen3_32b`
- `qwen3_1p7b_deepseek_qwen_32b`
- `deepseek_qwen_1p5b_qwen3_32b`
- `deepseek_qwen_1p5b_deepseek_qwen_32b`

Edit `config.example.json` to match your remote vLLM ports.

## Routing Logic

For each active prompt:

1. Generate one continuation step with the routed model through chat completions.
2. Read token `top_logprobs` and compute a confidence value.
3. Aggregate token confidence with `reliability_mode`.
4. Route the next step using the original two-component GMM logic.
5. Stop on answer markers, budget, `max_steps`, or patience.

## Run

```bash
python baselines/steer/run_steer.py \
  --sarr-config configs/sarr_code_aggressive.json \
  --steer-config baselines/steer/config.example.json \
  --dataset math500 \
  --model-pair qwen3_1p7b_deepseek_qwen_32b \
  --resume
```

Wrapper example:

```bash
DATASET=aime25 MODEL_PAIR=deepseek_qwen_1p5b_deepseek_qwen_32b bash baselines/steer/run_steer.sh
```

Outputs are written under `steer_results/<dataset>/<variant>/` with `summary.csv`, `summary_metrics.json`, and per-problem step metadata.
