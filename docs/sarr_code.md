# SARR-CoDE v3 State-Aware Routing

This implementation is an SLM-first continuation system with bounded LLM continuation leases, rollback to a clean autonomy anchor, and CI-OD shadow logging.

The LLM role is intentionally narrow:

```text
LLM = continuation only
LLM != verifier
LLM != reset controller
LLM != semantic parser
LLM != judge
```

The method does not use calibration CDFs, percentile confidence, answer probes, embeddings, hidden states, LLM judges, or diverse sampling for routing.

## 1. Current Flow

```text
SLM default generation
-> raw continuation-confidence readiness
-> optional short LLM continuation lease without rollback
-> confirmed rollback when needed
-> optional short LLM continuation after rollback
-> return control to SLM
```

Recovery and lease are now separated:

- Non-lease rollback recovery ends by returning directly to `SLM_ACTIVE`.
- LLM leases return through `POST_LEASE_OBSERVE`.
- `ROLLBACK_RECOVERY` is valid only while a real `recovery_context` exists.
- If `ROLLBACK_RECOVERY` appears without `recovery_context`, the invariant guard logs it and restores `SLM_ACTIVE`.

## 2. Run Commands

Start the OpenAI-compatible LLM endpoint first. Its URL and served model name must match `llm.api_base_url` and `llm.api_model` in `configs/sarr_code_aggressive.json`.

Smoke run:

```powershell
python scripts/run_sarr_code.py --config configs/sarr_code_aggressive.json --dataset aime25 --max-problems 1 --output-root sarr_results --variant sarr_code_v3_state_aware_routing_rollback
```

Resume a partial run:

```powershell
python scripts/run_sarr_code.py --config configs/sarr_code_aggressive.json --dataset aime25 --output-root sarr_results --variant sarr_code_v3_state_aware_routing_rollback --resume
```

Run the predefined D1-D8 sweep:

```powershell
python scripts/run_sarr_sweep.py --base-config configs/sarr_code_aggressive.json --dataset aime25 --output-root sarr_results --resume
```

Run tests:

```powershell
pytest tests/test_sarr_code.py
pytest
```

Outputs are written under:

```text
sarr_results/<dataset>/<variant>/
```

Each problem directory contains:

```text
<problem_id>.problem.json
<problem_id>.steps.jsonl
<problem_id>.rollback_events.jsonl
<problem_id>.transitions.jsonl
<problem_id>.trace.json
```

## 3. Calibration

Calibration is disabled for this method. Formal runs should keep:

```json
{
  "calibration": {
    "enabled": false,
    "build_cdf": false,
    "load_cdf": false,
    "use_percentile": false
  },
  "confidence": {
    "percentile_normalization": false,
    "calibration_path": null
  },
  "readiness": {
    "normalization": "raw",
    "use_calibration": false
  }
}
```

The SARR runner validates this at startup. There is no calibration mode in the active entrypoint.

## 4. Readiness

Readiness is raw continuation confidence with optional raw smoothing:

```text
readiness_value = readiness_raw_smooth if configured and available else c_raw
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

Each scored SLM step records `c_raw`, `readiness_raw`, `readiness_raw_smooth`, `readiness_value`, `readiness_high`, `readiness_mid`, and `readiness_low`.

## 5. CI-OD Shadow Logging

CI-OD is shadow-only by default. It is recorded in traces and summaries, and it changes routing only when `enable_ciod_active_routing=true`.
The original consecutive high-run hazard is retained as `ciod_risk_v1` for comparison; the current CI-OD signal is `ciod_risk_v2`.
`ciod_shadow_trigger_v2` is a sustained risk-state flag. `ciod_event_shadow` is the sparse event controller output intended for routing experiments.

Per scored SLM step, `extra.confidence_process` records:

```text
raw_low
smooth_low
masked_uncertainty
raw_low_count
smooth_low_count
masked_uncertainty_count
masked_uncertainty_gap
high_run_length
high_run_start_step
masked_memory_at_high_run_start
ciod_risk
ciod_shadow_trigger
ciod_risk_v1
ciod_shadow_trigger_v1
masked_memory
last_masked_step
steps_since_last_masked
post_masked_exposure
post_masked_high_count
post_masked_mid_high_count
ciod_risk_v2
ciod_shadow_trigger_v2
ciod_grid_risks
ciod_grid_triggers
ciod_event_shadow
ciod_episode_active
ciod_episode_id
ciod_cooldown_until_step
new_masked_mass_since_last_ciod_event
new_exposure_since_last_ciod_event
```

Definitions:

```text
raw_low = c_raw <= 0.35
smooth_low = readiness_value <= 0.35
masked_uncertainty = raw_low and not smooth_low
masked_uncertainty_gap = raw_low_count - smooth_low_count
high_run_length = consecutive readiness_value >= 0.70
masked_memory = decayed masked_uncertainty memory
post_masked_exposure = decayed confidence exposure after masked uncertainty
```

The v2 risk uses a post-masked confidence exposure conditional hazard, not linear weighting:

```text
if masked_memory < min_masked_memory:
    risk = 0
else:
    exposure_excess = max(0, post_masked_exposure - exposure_e0)
    cumulative_hazard =
      lambda0
      * (1 + masked_memory)^alpha
      * exposure_excess^power
    risk = 1 - exp(-cumulative_hazard)
```

Exposure increments by `1.0` when `readiness_value >= high_threshold`, by `0.5` when `readiness_value >= mid_high_threshold`, and by `0.0` otherwise. `ciod_shadow_trigger_v2 = ciod_risk_v2 >= risk_threshold`.

The event controller converts the sustained risk flag into sparse events:

```text
on:  ciod_risk_v2 >= ciod_event_on_threshold
off: ciod_risk_v2 <= ciod_event_off_threshold
cooldown: no retrigger before ciod_cooldown_until_step
retrigger: new masked mass >= 2.0 or new exposure >= 4.0
```

Default active routing is off. If `enable_ciod_active_routing=true`, only `ciod_event_shadow=True` may create `LLM_LEASE_BY_CIOD_EVENT`, capped by `max_ciod_active_leases_per_problem`. `ciod_shadow_trigger_v2` never directly routes, and readiness-low routing keeps priority on readiness-low steps.

Defaults:

```json
{
  "confidence_process": {
    "lambda0": 0.003,
    "alpha": 1.0,
    "r0": 20,
    "power": 2.0,
    "high_threshold": 0.70,
    "mid_high_threshold": 0.60,
    "raw_low_threshold": 0.35,
    "smooth_low_threshold": 0.35,
    "masked_decay": 0.995,
    "exposure_decay": 0.98,
    "min_masked_memory": 3.0,
    "exposure_e0": 4.0,
    "risk_threshold": 0.10,
    "ciod_event_on_threshold": 0.10,
    "ciod_event_off_threshold": 0.03,
    "ciod_event_cooldown_steps": 32,
    "min_new_masked_mass_for_retrigger": 2.0,
    "min_new_exposure_for_retrigger": 4.0,
    "max_ciod_active_leases_per_problem": 1,
    "enable_ciod_active_routing": false,
    "v1_lambda0": 0.002,
    "ciod_grid": [
      {"exposure_e0": 3.0, "lambda0": 0.003, "risk_threshold": 0.10},
      {"exposure_e0": 4.0, "lambda0": 0.003, "risk_threshold": 0.10},
      {"exposure_e0": 5.0, "lambda0": 0.003, "risk_threshold": 0.10},
      {"exposure_e0": 5.0, "lambda0": 0.005, "risk_threshold": 0.10},
      {"exposure_e0": 8.0, "lambda0": 0.005, "risk_threshold": 0.10}
    ]
  }
}
```

Problem summaries include:

```text
raw_low_count
smooth_low_count
masked_uncertainty_count
masked_uncertainty_gap
max_high_run_length
ciod_risk
ciod_shadow_trigger
max_ciod_risk
ciod_shadow_trigger_count
first_ciod_shadow_trigger_step
ciod_risk_v1
ciod_shadow_trigger_v1
max_ciod_risk_v1
ciod_shadow_trigger_count_v1
first_ciod_shadow_trigger_step_v1
max_post_masked_exposure
ciod_risk_v2
ciod_shadow_trigger_v2
max_ciod_risk_v2
ciod_shadow_trigger_count_v2
first_ciod_shadow_trigger_step_v2
ciod_grid_summary
ciod_event_count
first_ciod_event_step
last_ciod_event_step
ciod_event_before_first_readiness_low
ciod_event_steps
ciod_active_lease_count
```

## 6. States

The state machine uses:

```text
STARTUP
SLM_ACTIVE
LLM_LEASE
POST_LEASE_OBSERVE
ROLLBACK_RECOVERY
UNRECOVERABLE
```

State rules:

- `STARTUP` is only for the beginning of a problem.
- Once startup is left, attempted startup re-entry is blocked and mapped to `SLM_ACTIVE`.
- `ROLLBACK_RECOVERY` is entered only for real rollback recovery with `recovery_context`.
- Recovery stops such as `SLM_READY`, `EXHAUSTED_FORCE_SLM`, `RECOVERY_BUDGET_EXCEEDED`, and `ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM` clear recovery context and return to `SLM_ACTIVE`.
- `ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM` must have `state_after=SLM_ACTIVE`.
- `POST_LEASE_OBSERVE` is only the post-lease observation window.

Invariant logs on every step include:

```text
state_duration
invalid_rollback_recovery_state
anchor_refresh_blocked_reason
```

If `invalid_rollback_recovery_state` is true, the system records:

```text
action = STATE_RECOVERED_TO_SLM_ACTIVE
```

and restores `SLM_ACTIVE`.

## 7. Clean Anchor

The clean autonomy anchor refreshes only when:

```text
generator == "slm"
and readiness_high
and not stagnation_suspect
and state == SLM_ACTIVE
and current step is active
```

LLM steps, lease observation steps, recovery steps, and stagnation-suspect steps do not refresh the anchor.

`STATE_ROLLBACK_RECOVERY` is used as an anchor-refresh blocked reason only when a real `recovery_context` exists. A stale `ROLLBACK_RECOVERY` state is repaired to `SLM_ACTIVE` instead of blocking anchor refresh.

## 8. LLM Lease

LLM lease is a short continuation-only handoff. It can happen without rollback for persistent low readiness, or after rollback for confirmed stagnation / confirmed low-confidence degeneration.

Config:

```json
{
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

The prompt remains normal continuation. It must not mention uncertainty, stagnation, repetition, stuckness, verification, reset, or repair.

Lease event example:

```json
{
  "event": "llm_lease",
  "reason": "LLM_LEASE_BY_READINESS_LOW",
  "rollback_before_lease": false,
  "lease_steps": 2,
  "prompt_type": "normal_continuation",
  "mention_uncertainty": false,
  "mention_stagnation": false,
  "mention_repetition": false,
  "mention_error": false,
  "return_to_slm": true,
  "state_after": "POST_LEASE_OBSERVE"
}
```

## 9. Rollback And Recovery

Confirmed stagnation uses word 3-gram Jaccard over recent active SLM steps or small blocks:

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

Autonomy states include:

```text
HIGH_CONF_STAGNATION
MID_CONF_STAGNATION
LOW_CONF_STAGNATION_COLLAPSE
```

Confirmed stagnation rolls back to `clean_autonomy_anchor`, not to the repetition onset.

Recovery records normal continuation from the LLM and then returns to SLM. For non-lease recovery, `state_after` is `SLM_ACTIVE`.

## 10. Budgets

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

If lease budget is exhausted, the system records:

```text
action = ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM
extra.routing_budget_exceeded = true
state_after = SLM_ACTIVE
```

No extra LLM route is created for this condition.

## 11. Checks

Useful checks in logs:

```text
calibration_enabled=false
readiness_source=raw
confidence_process exists in every step extra
ciod_shadow_trigger_v2 is a sustained risk flag, not a direct routing trigger
ciod_event_shadow is sparse; active CI-OD routing is off unless explicitly enabled
LLM_LEASE can appear without rollback for persistent low readiness
STAGNATION_ROLLBACK appears for confirmed stagnation
MID_CONF_STAGNATION appears for mid-confidence repeated tails
anchor_refresh_blocked_reason=STAGNATION_SUSPECT for stagnation suspects
STATE_ROLLBACK_RECOVERY appears only with real recovery_context
ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM has state_after=SLM_ACTIVE
no mature prefix re-enters STARTUP after lease or recovery
llm lease counts and tokens stay within budget
```
