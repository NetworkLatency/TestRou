# SARR-CoDE CI-OD Event Routing

This implementation is an SLM-first continuation system. The LLM role is intentionally narrow:

```text
LLM = continuation only
LLM != verifier
LLM != reset controller
LLM != semantic parser
LLM != judge
```

The current main method uses the CI-OD Event Controller as the only active LLM lease source. Readiness and stagnation remain in the logs as diagnostics only.

## Run Commands

Start the OpenAI-compatible LLM endpoint first. Its URL and served model name must match `llm.api_base_url` and `llm.api_model` in `configs/sarr_code_aggressive.json`.

Smoke run:

```bash
python scripts/run_sarr_code.py \
  --config configs/sarr_code_aggressive.json \
  --mode run \
  --dataset aime25 \
  --max-problems 1 \
  --output-root sarr_results
```

Run 30 AIME25 problems:

```bash
python scripts/run_sarr_code.py \
  --config configs/sarr_code_aggressive.json \
  --mode run \
  --dataset aime25 \
  --max-problems 30 \
  --output-root sarr_results
```

Resume a partial run:

```bash
python scripts/run_sarr_code.py \
  --config configs/sarr_code_aggressive.json \
  --mode run \
  --dataset aime25 \
  --output-root sarr_results \
  --resume
```

Run tests:

```bash
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

## Active Routing

Only sparse CI-OD events can lease the LLM:

```text
ciod_event_shadow == true
and confidence_process.enable_ciod_active_routing == true
and routing budget is available
-> LLM_LEASE_BY_CIOD_EVENT
```

Readiness-low and confirmed stagnation never trigger active routing:

```text
READINESS_LOW_DIAGNOSTIC_ONLY
STAGNATION_DIAGNOSTIC_ONLY
```

`ciod_shadow_trigger_v2` is a sustained risk flag. It never routes directly. Routing can only use `ciod_event_shadow`.

Current generation settings keep the thinking-step budget wide enough for SLM/LLM collaboration and avoid fixed-LLM final-answer masking:

```json
{
  "generation": {
    "max_new_tokens_per_step": 256,
    "think_token_budget": 8192,
    "final_answer_generator": "active"
  }
}
```

## CI-OD Risk

The current CI-OD risk is the post-masked confidence exposure hazard:

```text
raw_low = c_raw <= raw_low_threshold
smooth_low = readiness_value <= smooth_low_threshold
masked_uncertainty = raw_low and not smooth_low
```

Masked uncertainty accumulates into decayed memory:

```text
if masked_uncertainty:
    masked_memory = masked_decay * masked_memory + 1
else:
    masked_memory = masked_decay * masked_memory
```

Exposure grows after masked uncertainty when confidence returns to mid/high values:

```text
readiness_value >= high_threshold      -> +1.0 exposure
readiness_value >= mid_high_threshold  -> +0.5 exposure
otherwise                              -> +0.0 exposure
```

The hazard is:

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

Default CI-OD event settings:

```json
{
  "confidence_process": {
    "ciod_event_on_threshold": 0.10,
    "ciod_event_off_threshold": 0.03,
    "ciod_event_cooldown_steps": 32,
    "min_new_masked_mass_for_retrigger": 2.0,
    "min_new_exposure_for_retrigger": 4.0,
    "max_ciod_active_leases_per_problem": 1,
    "enable_ciod_active_routing": true
  }
}
```

## Lease And Budget

LLM leases use normal continuation prompts and do not mention uncertainty, stagnation, repetition, verification, reset, or repair.

Current lease settings:

```json
{
  "llm_lease": {
    "enabled": true,
    "prompt_type": "normal_continuation",
    "mention_uncertainty": false,
    "mention_stagnation": false,
    "mention_repetition": false,
    "mention_error": false,
    "max_tokens_per_step": 256,
    "max_steps_per_event": 2,
    "return_to_slm": false
  }
}
```

Unified routing budget:

```json
{
  "routing_budget": {
    "max_total_llm_events_per_problem": 8,
    "max_total_llm_tokens_per_problem": 4096,
    "max_ciod_events_per_problem": 1,
    "max_readiness_events_per_problem": 0,
    "max_stagnation_events_per_problem": 0
  }
}
```

If the active CI-OD route is blocked by budget, the step records:

```text
action = ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM
extra.routing_source = ROUTING_BUDGET_EXCEEDED
extra.routing_budget_exceeded = true
state_after = SLM_ACTIVE
```

## Diagnostics

Every scored SLM step records readiness fields:

```text
c_raw
readiness_raw
readiness_raw_smooth
readiness_value
readiness_high
readiness_mid
readiness_low
```

Readiness thresholds and smoothing are diagnostics only. They do not create LLM leases, do not mark recovery ready, and do not control anchor refresh.

Stagnation diagnostics include:

```text
stagnation_score
stagnation_high
stagnation_suspect
stagnation_confirmed
```

Confirmed stagnation does not roll back and does not lease the LLM.

Every scored SLM step also records `extra.confidence_process`, including:

```text
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

Problem summaries include:

```text
raw_low_count
smooth_low_count
masked_uncertainty_count
masked_uncertainty_gap
max_high_run_length
max_post_masked_exposure
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

## State Machine Invariants

`ROLLBACK_RECOVERY` is valid only while a real `recovery_context` exists. If stale `ROLLBACK_RECOVERY` appears without recovery context, the invariant guard records:

```text
invalid_rollback_recovery_state = true
action = STATE_RECOVERED_TO_SLM_ACTIVE
```

and restores `SLM_ACTIVE`.

Each step logs:

```text
state_duration
invalid_rollback_recovery_state
anchor_refresh_blocked_reason
```

Anchor refresh is blocked by `STATE_ROLLBACK_RECOVERY` only when a real `recovery_context` exists. Otherwise stale recovery state is repaired instead of blocking anchor refresh.
