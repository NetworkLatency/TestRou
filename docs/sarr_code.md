# SARR-CoDE v2 Raw HCS Confirmed Rollback

This path implements **SARR-CoDE with raw-readiness High-Confidence Stagnation confirmed rollback**. The main flow remains:

```text
SLM first -> online confidence-dynamics monitoring -> rollback/recovery -> bounded LLM continuation -> return to SLM
```

The method no longer builds or reads calibration CDFs. Strategy decisions use raw SLM continuation confidence only:

```text
R_k = c_raw_k
R_smooth_k = mean(recent raw c_raw values)
```

HCS is only an auxiliary failure mode. It does not turn the system into a repetition detector plus LLM bridge, and it does not use Hinit routing, LLM judging, answer probes, boxed-answer parsing, embeddings, hidden states, or diverse sampling.

## 1. Calibration Is Disabled

Formal runs should not set a calibration path and should not run calibration mode.

Required config:

```json
{
  "method": "sarr_code_v2_raw_hcs_confirmed_rollback",
  "calibration": {
    "enabled": false,
    "build_cdf": false,
    "load_cdf": false,
    "use_percentile": false
  },
  "confidence": {
    "percentile_normalization": false,
    "calibration_path": null
  }
}
```

`c_norm` and `c_smooth` may still appear as backward-compatible log fields, but they are not used for rollback, anchor refresh, HCS, startup, or low-confidence collapse. The run script does not load a normalizer during `--mode run`.

If `confidence.calibration_path` is set, or any `calibration.*` switch is enabled, the run fails fast.

## 2. Local Assets

Configure model and dataset paths in:

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
    "api_model": "qwen3-4B",
    "local_files_only": true
  }
}
```

The SLM is loaded locally through transformers so the experiment can read next-token logits for raw continuation confidence. The LLM is expected to run as an OpenAI-compatible vLLM server.

## 3. Start vLLM for the LLM

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

Use `server/template/deepseekr1.jinja` for DeepSeek-R1-style models and `server/template/qwen3.jinja` for Qwen3-style models.

## 4. Run SARR-CoDE Raw HCS

Do not run `--mode calibrate`. Run the experiment directly:

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
sarr_results/<dataset>/sarr_code_v2_raw_hcs_confirmed_rollback/
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

## 5. Raw Readiness

Recommended defaults:

```json
{
  "readiness": {
    "signal": "continuation_confidence",
    "normalization": "raw",
    "use_calibration": false,
    "value_field": "c_raw",
    "smooth_window": 3,
    "high_threshold": 0.75,
    "low_threshold": 0.35
  }
}
```

Each step records:

```json
{
  "calibration_enabled": false,
  "readiness_source": "raw",
  "c_raw": 0.83,
  "readiness_raw": 0.83,
  "readiness_raw_smooth": 0.79,
  "readiness_high": true,
  "readiness_low": false
}
```

All strategy decisions read readiness through the raw readiness path. Do not use `c_norm` or old percentile-smoothed `c_smooth` for interpretation of current decisions.

## 6. HCS Configuration

Recommended defaults:

```json
{
  "stagnation": {
    "enabled": true,
    "unit": "step_or_small_block",
    "block_min_tokens": 32,
    "block_max_steps": 2,
    "metric": "word_3gram_jaccard",
    "repeat_window": 10,
    "high_threshold": 0.85
  },
  "anchor": {
    "type": "clean_autonomy_anchor",
    "refresh_condition": "raw_readiness_high_and_not_hcs_suspect",
    "freeze_on_hcs_suspect": true,
    "fallback": "startup_anchor_or_zero"
  },
  "hcs": {
    "enabled": true,
    "enable_after_clean_anchor": true,
    "suspect_condition": "raw_readiness_high_and_stagnation_high",
    "suspect_patience": 3,
    "action": "rollback_to_clean_anchor",
    "max_hcs_rollbacks_per_problem": 2
  }
}
```

HCS suspect condition:

```text
readiness_raw_smooth >= readiness.high_threshold
and
stagnation_score >= stagnation.high_threshold
```

A suspect step is logged immediately and cannot refresh `clean_autonomy_anchor`. HCS confirms after three consecutive suspect SLM steps/blocks. Rollback target is always `clean_autonomy_anchor`, never the stagnation onset.

## 7. HCS Recovery

Recommended defaults:

```json
{
  "hcs_recovery": {
    "generator": "llm",
    "prompt_type": "normal_continuation",
    "mention_stagnation": false,
    "mention_repetition": false,
    "max_llm_steps": 2,
    "max_tokens_per_step": 128,
    "return_to_slm_after_recovery": true
  }
}
```

After HCS rollback, the LLM sees only the clean prefix and uses the normal continuation prompt path. The prompt must not mention stagnation, repetition, stuckness, loops, strategy changes, or repair instructions.

HCS rollback events include:

```json
{
  "event": "hcs_rollback",
  "reason": "HCS_CONFIRMED_RAW_READINESS",
  "trigger_step": 135,
  "clean_anchor_step": 118,
  "rollback_span": 17,
  "hcs_rollback_count": 1,
  "readiness_source": "raw",
  "calibration_enabled": false,
  "llm_recovery_prompt_type": "normal_continuation",
  "mention_stagnation": false,
  "return_to_slm": true
}
```

## 8. Low-Confidence Path

Low-confidence collapse remains independent from HCS and also uses raw readiness:

```json
{
  "low_confidence": {
    "useful_exploration_grace_blocks": 2,
    "collapse_patience_blocks": 3,
    "action_after_patience": "existing_rollback_recovery"
  }
}
```

If raw readiness is low or raw-readiness degeneration appears during the grace window, the step is logged as `USEFUL_EXPLORATION` and SLM continues. If it persists through the patience window, the existing rollback/recovery path is used. This path does not require stagnation.

## 9. Startup

HCS does not handle startup bad prefixes:

```json
{
  "startup_guard": {
    "hcs_enabled": false,
    "enable_hcs_after_clean_anchor": true
  }
}
```

Startup continues to use the existing startup rollback / confidence-degeneration logic, but the confidence signal is raw readiness. Jaccard stagnation detection does not interfere with startup.

## 10. Problem 6 Checks

For Problem 6, inspect `steps.jsonl` and `rollback_events.jsonl`:

```text
calibration_enabled=false
readiness_source=raw
readiness_raw_smooth is present
hcs_suspect appears on high-confidence repeated tail steps
anchor_refresh_blocked_reason=HCS_SUSPECT during suspect steps
third consecutive suspect triggers event=hcs_rollback
reason=HCS_CONFIRMED_RAW_READINESS
clean_anchor_step is the rollback target
hcs_rollback_count <= 2
```

The expected behavior is that the high-confidence repeated tail is deleted back to the clean autonomy anchor, LLM performs one or two ordinary continuation steps, and control returns to SLM without another long high-confidence loop.

## 11. Sweep

The existing sweep script can still materialize variants:

```bash
python scripts/run_sarr_sweep.py \
  --base-config configs/sarr_code_aggressive.json \
  --dataset aime25 \
  --max-problems 30 \
  --output-root sarr_results \
  --resume
```

The sweep should inherit raw-readiness settings from the base config. Do not add calibration paths to generated variants.

## 12. Summary Metrics

`summary_metrics.json` records:

```text
rollback_rate
startup_rollback_rate
post_stable_rollback_rate
hcs_rollback_rate
avg_rollback_span
avg_recovery_steps
recovery_ready_rate
recovery_exhausted_rate
forced_close_think_rate
force_slm_after_recovery_fail_rate
llm_token_ratio
```

Raw per-problem fields in `summary.csv` include:

```text
hcs_rollback_count
has_hcs_rollback
hcs_suspect_count
hcs_confirmed_count
```
