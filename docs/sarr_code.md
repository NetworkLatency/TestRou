# SARR-CoDE Aggressive Prefix Experiment

This path implements **Aggressive Prefix-Centric SARR-CoDE**. The main method only uses SLM next-token distributions after generated prefixes. It does not use Hinit routing, answer probes, LLM judging, boxed-answer parsing, or KV-cache rollback.

## 1. Local Assets

The server is expected to be offline. Configure all model and dataset paths as local filesystem paths in:

```bash
configs/sarr_code_aggressive.json
```

Important fields:

```json
{
  "slm": {
    "model_path": "/local/path/to/slm",
    "chat_template_path": "server/template/deepseekr1.jinja",
    "device": "cuda:0",
    "backend": "transformers",
    "local_files_only": true
  },
  "llm": {
    "model_path": "/local/path/to/llm-tokenizer",
    "chat_template_path": "server/template/qwen3.jinja",
    "backend": "openai",
    "api_base_url": "http://127.0.0.1:8000/v1",
    "api_model": "Qwen-32B",
    "local_files_only": true
  }
}
```

The SLM is loaded by transformers in the experiment process so later diagnostics can access full logits. The LLM is expected to run as a vLLM OpenAI-compatible server.

## 2. Start vLLM for the LLM

Example:

```bash
MODEL=/home/lhyang/Documents/code/reasoning_boundary/models/qwen3-4B \
SERVED_MODEL_NAME=qwen3-4B \
CUDA_DEVICE=0 \
PORT=8000 \
CHAT_TEMPLATE=server/template/qwen3.jinja \
TRUST_REMOTE_CODE=1 \
ENABLE_PREFIX_CACHING=1 \
GPU_MEMORY_UTILIZATION=0.6 \
bash server/serve.sh
```

Use `server/template/deepseekr1.jinja` for DeepSeek-R1-style models and `server/template/qwen3.jinja` for Qwen3-style models. The experiment script also loads the configured template locally to render completion prompts consistently.

## 3. Build Percentile Calibration

Run SLM-only calibration first:

```bash
python scripts/run_sarr_code.py \
  --config configs/sarr_code_aggressive.json \
  --mode calibrate \
  --dataset aime25 \
  --max-problems 30 \
  --calibration-output sarr_results/calibration/aime25_slm_cdf.json
```

This writes:

```text
sarr_results/calibration/aime25_slm_cdf.json
sarr_results/calibration/aime25_slm_cdf.traces.jsonl
```

The formal run requires `confidence.calibration_path` unless `confidence.allow_identity_normalizer=true` is explicitly enabled for debugging.

## 4. Run SARR-CoDE

```bash
python scripts/run_sarr_code.py \
  --config configs/sarr_code_aggressive.json \
  --mode run \
  --dataset aime25 \
  --max-problems 30 \
  --output-root sarr_results \
  --resume
```

Outputs are written under:

```text
sarr_results/<dataset>/sarr_code_aggressive_prefix/
```

Per problem:

```text
<id>.problem.json
<id>.steps.jsonl
<id>.rollback_events.jsonl
<id>.transitions.jsonl
<id>.trace.json
```

Summary files:

```text
summary.csv
summary_metrics.json
```

## 5. Run The D1-D8 Sweep

After calibration is ready and `confidence.calibration_path` points to it, run:

```bash
python scripts/run_sarr_sweep.py \
  --base-config configs/sarr_code_aggressive.json \
  --dataset aime25 \
  --max-problems 30 \
  --output-root sarr_results \
  --resume
```

The sweep script materializes one config per variant under:

```text
sarr_results/<dataset>/sarr_sweep/configs/
```

and writes:

```text
sarr_results/<dataset>/sarr_sweep/variant_manifest.json
sarr_results/<dataset>/sarr_sweep/sweep_summary.csv
sarr_results/<dataset>/sarr_sweep/sweep_summary.json
```

To run a subset:

```bash
python scripts/run_sarr_sweep.py \
  --base-config configs/sarr_code_aggressive.json \
  --dataset aime25 \
  --only D1_balanced_055,D5_conservative \
  --resume
```

Use `--dry-run` to print the commands without launching model runs.

## 6. Method Boundary

Only the `<think>...</think>` portion uses SARR-CoDE collaboration. There is no step-count limit; thinking stops by natural `</think>`/EOS, context exhaustion, or the configurable thinking token budget. If the token budget is hit, `generation.force_close_think_on_budget` controls whether the script appends the configured `</think>` bridge. The final answer is then generated with `generation.final_answer_generator` and does not feed back into routing or rollback decisions.

## 7. Rollback Convergence Guards

The thinking token budget is measured on the active retained prefix. Rollback can delete suffix steps, so repeated rollback at the same prefix position can prevent the active prefix from monotonically reaching the budget. To avoid anchor-level cycles without restoring a global `max_steps` limit, rollback records include:

```json
{
  "requested_anchor_step": 38,
  "anchor_step": 37,
  "anchor_repeat_count_before": 1,
  "anchor_backoff_steps": 1
}
```

Configured behavior:

```json
{
  "rollback": {
    "long_span_policy": "fallback_once_then_rollback",
    "max_long_span_fallbacks_per_anchor": 1,
    "long_span_recovery_steps": 1,
    "anchor_repeat_backoff_after": 1,
    "anchor_repeat_backoff_steps": 1,
    "max_root_rollbacks": 2,
    "root_rollback_action": "force_close_think"
  }
}
```

The first repeated rollback request at the same anchor backs off the effective rollback anchor by one step; further repeats back off farther. If the requested anchor is already root and repeats beyond `max_root_rollbacks`, the run closes `<think>` and proceeds to final-answer generation.

## 8. Summary Metrics

`summary_metrics.json` records:

```text
rollback_rate
startup_rollback_rate
post_stable_rollback_rate
avg_rollback_span
avg_recovery_steps
recovery_ready_rate
recovery_exhausted_rate
forced_close_think_rate
force_slm_after_recovery_fail_rate
llm_token_ratio
```

The rate denominators are explicit in the raw per-problem fields in `summary.csv`. Rollback type rates are problem-level trigger rates. Recovery rates and average spans are aggregated over rollback events. `force_slm_after_recovery_fail_rate` is the fraction of forced SLM handoffs that roll back again before reaching a stable anchor.
