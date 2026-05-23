# SARR-CoDE Ownership Controller

This implementation is an SLM-led collaborative reasoning system. The LLM role is intentionally narrow:

```text
LLM = temporary continuation owner
LLM != verifier
LLM != reset controller
LLM != semantic parser
LLM != answer judge
```

The active controller is an ownership state machine:

```text
SLM_ACTIVE
LLM_FORWARD_OWNERSHIP
LLM_REPAIR_OWNERSHIP
HANDOFF_PROBE
CLOSE_OR_FINALIZE
```

The controller switches by observable continuation signals, not by LLM token share, fixed LLM step counts, or decayed re-entry risk.

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

Resume a partial run:

```bash
python scripts/run_sarr_code.py \
  --config configs/sarr_code_aggressive.json \
  --mode run \
  --dataset aime25 \
  --output-root sarr_results \
  --resume
```

Outputs are written under:

```text
sarr_results/<dataset>/<variant>/
```

Each problem directory contains:

```text
<problem_id>.problem.json
<problem_id>.steps.jsonl
<problem_id>.controller_events.jsonl
<problem_id>.transitions.jsonl
<problem_id>.trace.json
```

## Routing Logic

`SLM_ACTIVE` accepts normal SLM steps and updates observable signals, answer stability, and stable-step memory. It can transfer ownership to the LLM on local difficulty, prefix contamination, or a degenerative loop without a stable candidate answer.

`LLM_FORWARD_OWNERSHIP` lets the LLM continue from the current prefix for one coarse step, then asks for an SLM handoff probe.

`LLM_REPAIR_OWNERSHIP` is entered after a prefix-contamination rollback. The rollback interval is sealed, and the LLM must generate at least `repair_horizon` replacement steps before a handoff probe is allowed.

`HANDOFF_PROBE` generates an SLM probe step without committing it. The probe is accepted only when it looks stable relative to local stable memory, does not repeat sealed content, and does not immediately degenerate. Failed probes are recorded as `probe_discarded` and do not affect the active prefix.

`CLOSE_OR_FINALIZE` closes thinking and generates the final answer. If no close marker appeared naturally, the controller appends a uniform `</think>` marker before the final-answer call.

## Signals

Each step records:

```text
raw_next_token_confidence
entropy
margin
repeated_ngram_ratio
repeated_sentence_count
repeated_phrase_count
repeated_verification_pattern_count
repeated_answer_mention_count
low_new_information_score
reflection_pattern_count
has_new_equation
has_new_case_split
has_new_constraint
has_candidate_answer
candidate_answer_value
repeats_existing_candidate_answer
degeneration_score
has_progress
```

Confidence fields come from logits already captured during SLM generation. The main loop does not do an additional long-prefix confidence forward on every step.

## Summary Schema

Problem summaries include:

```text
problem_id
finish_reason
final_answer
final_answer_generator
driver_state
driver_switch_count
llm_ownership_episodes
llm_forward_episodes
llm_repair_episodes
handoff_probe_count
handoff_success_count
handoff_failure_count
local_difficulty_count
prefix_contamination_count
degenerative_loop_count
rollback_count
sealed_interval_count
repeated_rollback_blocked_count
slm_thinking_tokens
llm_thinking_tokens
total_thinking_tokens
slm_step_count
llm_step_count
confidence_forward_count
handoff_probe_forward_count
lookahead_count
slm_prefill_count
llm_prefill_count
total_wall_time
slm_wall_time
llm_wall_time
```

`lookahead_count` is expected to be `0`.
