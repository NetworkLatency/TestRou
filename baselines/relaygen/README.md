# RelayGen Baseline Adapter

This folder contains the project-local RelayGen comparison adapter.

Upstream repository: https://github.com/jiwonsong-dev/RelayGen

## Current Experiment Version

All experiments use only Qwen3 and DeepSeek-R1-Distill-Qwen model families. Generation uses remote vLLM `chat.completions.create(messages=..., extra_body=...)` continuation calls.

## Model Pairs

Use `--model-pair` to select one of:

- `qwen3_1p7b_qwen3_32b`
- `qwen3_1p7b_deepseek_qwen_32b`
- `deepseek_qwen_1p5b_qwen3_32b`
- `deepseek_qwen_1p5b_deepseek_qwen_32b`

`cue_family` defaults to the base model family when not provided: `qwen3` for Qwen3 base models and `r1` for DeepSeek-R1-Distill-Qwen base models.

Edit `config.example.json` to match your remote vLLM ports.

## Budget

RelayGen receives `budget=16384`, matching the unified completion-token budget for methods without an explicit final-answer split.

## Routing Logic

RelayGen starts in large-model mode during the thinking phase. It switches between base and small continuations on family-specific cues and sentence boundaries, then uses `answer_model` for the answer phase.

`answer_model` is configurable as `small` or `base`.

## Run

```bash
python baselines/relaygen/run_relaygen.py \
  --sarr-config configs/sarr_code_aggressive.json \
  --relaygen-config baselines/relaygen/config.example.json \
  --dataset aime25 \
  --model-pair deepseek_qwen_1p5b_qwen3_32b \
  --resume
```

Wrapper example:

```bash
DATASET=aime24 MODEL_PAIR=qwen3_1p7b_deepseek_qwen_32b bash baselines/relaygen/run_relaygen.sh
```

Outputs are written under `relaygen_results/<dataset>/<variant>/` with `summary.csv`, `summary_metrics.json`, and per-problem metadata.
