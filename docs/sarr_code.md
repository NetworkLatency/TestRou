# SARR-CoDE v3 State-Aware Routing

This path implements **SARR-CoDE with state-aware step routing and confirmed-stagnation rollback**. The main flow is still SLM-first:

```text
SLM default generation
-> raw continuation-confidence monitoring
-> optional short LLM lease without rollback
-> confirmed-stagnation rollback to clean autonomy anchor
-> short LLM lease
-> return to SLM
```

The method does not use calibration CDFs, percentile confidence, LLM judges, answer probes, final-answer parsing, embeddings, hidden states, or diverse sampling.

## 1. Calibration Is Disabled

Do not run `--mode calibrate`. Formal runs should use:

```json
{
  "method": "sarr_code_v3_state_aware_routing_rollback",
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

`c_norm` and `c_smooth` may remain in logs for compatibility, but strategy decisions use raw readiness only.

## 2. Run

Start the OpenAI-compatible vLLM server for the LLM, then run:

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
sarr_results/<dataset>/sarr_code_v3_state_aware_routing_rollback/
```

Per-problem logs include `steps.jsonl`, `rollback_events.jsonl`, `transitions.jsonl`, and `trace.json`.

## 3. Thinking Stop

SLM step generation still uses `\n\n` as the normal boundary. After a boundary, the local SLM can look ahead for `</think>`:

```json
{
  "generation": {
    "step_delimiters": ["\n\n"],
    "close_tag_lookahead_tokens": 16
  }
}
```

If `</think>`, EOS, or an empty step is observed, thinking stops immediately and the final step skips continuation-confidence forward.

## 4. Readiness

The strategy uses raw confidence with optional raw smoothing:

```text
readiness_value = readiness_raw_smooth if available else c_raw
```

Recommended config:

```json
{
  "readiness": {
    "signal": "continuation_confidence",
    "normalization": "raw",
    "use_calibration": false,
    "value_field": "readiness_smooth_or_raw",
    "smooth_window": 3,
    "high_threshold": 0.70,
    "low_threshold": 0.35
  }
}
```

Each monitored SLM step records `c_raw`, `readiness_raw`, `readiness_raw_smooth`, `readiness_value`, `readiness_high`, `readiness_mid`, and `readiness_low`.

## 5. States

The routing state machine uses:

```text
STARTUP
SLM_ACTIVE
LLM_LEASE
POST_LEASE_OBSERVE
ROLLBACK_RECOVERY
UNRECOVERABLE
```

`STARTUP` is only for the beginning of a problem. After the system has left STARTUP, recovery or LLM lease cannot send it back there. Lease and recovery both return through `POST_LEASE_OBSERVE`, which suppresses startup rollback and immediate rollback for a short observation window.

## 6. Confirmed Stagnation

Surface stagnation uses word 3-gram Jaccard over recent active SLM steps or small blocks:

```json
{
  "stagnation": {
    "enabled": true,
    "metric": "word_3gram_jaccard",
    "repeat_window": 10,
    "high_threshold": 0.85,
    "patience": 3,
    "block_min_tokens": 32,
    "block_max_steps": 2,
    "include_mid_readiness": true
  }
}
```

`hcs_suspect` is still logged for `readiness_high and stagnation_high`, but rollback now triggers on confirmed stagnation, including mid-confidence repeated tails.

Autonomy states include:

```text
HIGH_CONF_STAGNATION
MID_CONF_STAGNATION
LOW_CONF_STAGNATION_COLLAPSE
```

Confirmed stagnation rolls back to `clean_autonomy_anchor`, not to the repetition onset.

## 7. Clean Anchor

The clean autonomy anchor refreshes only when:

```text
generator == "slm"
and readiness_high
and not stagnation_suspect
and state == SLM_ACTIVE
and current step is active
```

LLM steps, recovery steps, POST_LEASE_OBSERVE steps, and any stagnation-suspect step do not refresh the anchor.

## 8. LLM Lease

LLM can now appear without rollback when SLM has persistent low readiness and no confirmed stagnation:

```json
{
  "low_readiness": {
    "useful_exploration_grace_steps": 2,
    "persistent_low_after_grace_action": "llm_lease_no_rollback"
  },
  "llm_lease": {
    "enabled": true,
    "prompt_type": "normal_continuation",
    "mention_uncertainty": false,
    "mention_stagnation": false,
    "mention_repetition": false,
    "mention_error": false,
    "persistent_uncertainty_steps": 2,
    "confirmed_stagnation_steps": 3,
    "low_conf_rollback_steps": 2,
    "max_tokens_per_step": 128,
    "max_events_per_problem": 4,
    "max_total_tokens_per_problem": 1024,
    "return_to_slm": true
  }
}
```

The lease prompt remains normal continuation. It must not mention uncertainty, stagnation, repetition, stuckness, strategy changes, or repair.

Lease events are logged as:

```json
{
  "event": "llm_lease",
  "reason": "PERSISTENT_UNCERTAINTY",
  "rollback_before_lease": false,
  "lease_steps": 2,
  "prompt_type": "normal_continuation",
  "mention_uncertainty": false,
  "mention_stagnation": false,
  "mention_repetition": false,
  "return_to_slm": true,
  "state_after": "POST_LEASE_OBSERVE"
}
```

For confirmed stagnation, the same event has `rollback_before_lease=true`, `reason=CONFIRMED_STAGNATION`, `rollback_anchor=<clean anchor>`, and `removed_steps=[...]`.

## 9. Budgets

Problem-level budgets prevent LLM takeover:

```json
{
  "budget": {
    "max_llm_lease_events_per_problem": 4,
    "max_llm_lease_tokens_per_problem": 1024,
    "max_rollbacks_per_problem": 4,
    "max_stagnation_rollbacks_per_problem": 2
  }
}
```

If lease budget is exhausted, the system records `routing_budget_exceeded` and falls back to the configured continue/close/unrecoverable policy instead of repeatedly calling LLM.

## 10. Checks

For Problem 6 and Problem 0, inspect:

```text
calibration_enabled=false
readiness_source=raw
readiness_value uses readiness_raw_smooth when available
LLM_LEASE can appear without rollback for persistent low readiness
STAGNATION_ROLLBACK appears for confirmed stagnation
MID_CONF_STAGNATION appears for mid-confidence repeated tails
anchor_refresh_blocked_reason=STAGNATION_SUSPECT
LLM steps do not refresh clean_autonomy_anchor
state transitions include LLM_LEASE -> POST_LEASE_OBSERVE
no mature prefix re-enters STARTUP after lease or recovery
llm lease counts and tokens stay within budget
```

The expected behavior is bounded SLM-first collaboration: the LLM gets short continuation leases, confirmed polluted tails are removed back to the clean anchor, and control returns to SLM after the post-lease observation window.
