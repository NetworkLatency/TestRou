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

The controller switches by observable continuation signals and SLM handoff probes, not by LLM token share, LLM self-judgment, or answer correctness.

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

`SLM_ACTIVE` accepts normal SLM steps and updates observable signals and stable-step memory. It can transfer ownership to the LLM on local difficulty, prefix contamination, or a degenerative loop.

`LLM_FORWARD_OWNERSHIP` lets the LLM continue from the current prefix. After each LLM step, the controller schedules an SLM handoff probe according to `risk.handoff_probe_strategy`:

```text
eager    = probe after every eligible LLM step
periodic = probe every risk.handoff_probe_interval LLM steps
hybrid   = wait risk.handoff_probe_warmup_steps, then probe every interval
```

`LLM_REPAIR_OWNERSHIP` is entered after a prefix-contamination rollback. The rollback interval is sealed, and `repair_horizon` is logged as the removed-step replacement target, but it no longer blocks SLM handoff probes.

`HANDOFF_PROBE` generates an SLM probe step without committing it. If the probe emits `</think>`, the controller treats that as an explicit SLM termination intent, commits the probe, and closes thinking. Otherwise, the probe is accepted only when it looks closer to local stable memory or the LLM continuation than to failure memory or rejected probe memory, does not repeat sealed content, and does not immediately return to self-check/repetition. Failed probes are recorded as `probe_discarded` and do not affect the active prefix.

## Online Regime Logic

SARR-CoDE now uses a fully online regime comparison for handoff acceptance. During a problem it maintains four local memories:

```text
stable SLM steps
failure-triggering SLM steps
current LLM ownership episode
rejected SLM handoff probes
```

Distances are computed only against signals observed inside the same problem. A handoff probe is requested by the configured probe schedule, then accepted when the SLM continuation is closer to stable/LLM-continuation memory than to failure/rejected-probe memory. LLM step quality is logged with a recent-window diagnostic, but it no longer gates whether the SLM is allowed to try taking ownership back.

`CLOSE_OR_FINALIZE` closes thinking and generates the final answer with `generation.final_answer_generator`, which defaults to `slm`. If no close marker appeared naturally, the controller appends a uniform `</think>` marker before the final-answer call. The final-answer call stops at any newly generated `</think>` marker so post-answer text is not treated as part of the answer.

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
has_candidate_answer
candidate_answer_value
repeats_existing_candidate_answer
degeneration_score
```

Confidence fields come from logits already captured during SLM generation. The main loop does not do an additional long-prefix confidence forward on every step.

For offline probability-periodicity analysis, SLM and LLM step records persist generated-token logprobs in `extra.generated_token_logprobs` and step-level aggregates in `extra.token_probability` when the backend exposes them. These fields are logged from generation scores and are not used by the online controller.

Run token/step/chunk probability periodicity analysis:

```bash
python scripts/analyze_probability_periodicity.py \
  --input sarr_results/aime25/sarr_code_v5_ownership_controller \
  --output sarr_results/aime25/probability_periodicity
```

The analysis defaults to `--source ALL --statuses active,sealed,probe_discarded` so SLM and LLM generations are reviewed symmetrically. Use `--source SLM` or `--source LLM` only for source-specific diagnostics.

## Summary Schema

Problem summaries include:

```text
problem_id
finish_reason
final_answer
final_answer_generator
handoff_probe_strategy
handoff_probe_interval
handoff_probe_warmup_steps
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
handoff_probe_skipped_count
slm_thinking_tokens
llm_thinking_tokens
total_thinking_tokens
probe_discarded_tokens
slm_probe_discarded_tokens
llm_probe_discarded_tokens
probe_discarded_step_count
slm_generated_thinking_tokens
llm_generated_thinking_tokens
total_generated_thinking_tokens
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
