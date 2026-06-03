# RSD Baseline Adapter

This folder contains the project-local RSD comparison adapter for the current Qwen3 / DeepSeek-R1-Distill-Qwen experiments.

Upstream repository: https://github.com/BaohaoLiao/RSD

## Current Experiment Version

The original RSD uses a separate PRM endpoint. For this project, all comparison experiments are restricted to Qwen3 and DeepSeek-R1-Distill-Qwen models, so this adapter removes the third PRM model and uses the target/base model as the step judge through `chat.completions.create(messages=..., extra_body=...)`.

Each step uses chat-template continuation controls:

- first call: `extra_body={"add_generation_prompt": true, ...}`
- continuation call: `extra_body={"add_generation_prompt": false, "continue_final_message": true, ...}`

## Model Pairs

Use `--model-pair` to select one of:

- `qwen3_1p7b_qwen3_32b`
- `qwen3_1p7b_deepseek_qwen_32b`
- `deepseek_qwen_1p5b_qwen3_32b`
- `deepseek_qwen_1p5b_deepseek_qwen_32b`

Edit `config.example.json` to match your remote vLLM ports.

## Routing Logic

For each reasoning step:

1. Generate a candidate step with the draft/small model.
2. Ask the target/base model to score the last step with one chat token.
3. Accept the draft step if the normalized score is at least `prm_threshold`.
4. Otherwise discard it and generate the step with the target/base model.
5. Continue until generation finishes, the completion budget is reached, `max_steps` is reached, or patience is exceeded.

`prm_threshold` is kept as the RSD-facing interface. In this adapted version, `0.7` means roughly score `>= 7/9`.

The target/base judge is an adaptation fallback, not part of the standard RSD cost. Its chat token usage is recorded only in `excluded_judge_*` diagnostic fields and is not counted in `slm_*` / `llm_*` token-cost summaries.

## Run

```bash
python baselines/rsd/run_rsd.py \
  --sarr-config configs/sarr_code_aggressive.json \
  --rsd-config baselines/rsd/config.example.json \
  --dataset aime24 \
  --model-pair qwen3_1p7b_qwen3_32b \
  --resume
```

Wrapper example:

```bash
DATASET=aime25 MODEL_PAIR=deepseek_qwen_1p5b_qwen3_32b bash baselines/rsd/run_rsd.sh
```

Outputs are written under `rsd_results/<dataset>/<variant>/` with `summary.csv`, `summary_metrics.json`, and per-problem metadata.
