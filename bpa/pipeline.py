from __future__ import annotations

import time

from .arbitration import llm_arbitrate
from .cascade.l0 import l0_filter
from .cascade.l1 import l1_shadow_rollout
from .cascade.l2 import l2_compute
from .config import BPAConfig
from .engines import finish_reason, generated_text, generated_token_ids
from .phase_machine import CLOSE_THINK_TAG, check_and_transition_phase
from .render import render_for_continuation
from .safety import ensure_step_terminator, extract_answer, update_repetition
from .state import CascadeResult, Decision, GenerationState, Phase, RejectedBranch, RepetitionState, TraceEvent
from .trace import BPAResult


def run_cascade(state: GenerationState, slm, llm, config: BPAConfig, disabled: bool = False) -> CascadeResult:
    l0 = l0_filter(state, slm, config)
    state.trace.append(TraceEvent(state.step_count, "l0", l0.to_dict()))

    if config.cascade_mode == "hinit":
        decision = Decision.LLM_FULL if (not disabled and l0.passed) else Decision.SLM_DIRECT
        return CascadeResult(decision=decision, l0=l0)

    if disabled or not l0.passed:
        return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0)

    try:
        branch1, branch2 = l1_shadow_rollout(state, slm, config, l0)
    except ValueError as exc:
        state.trace.append(TraceEvent(state.step_count, "l1_skipped", {"reason": str(exc)}))
        return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0)

    state.trace.append(
        TraceEvent(
            state.step_count,
            "l1",
            {
                "b1_text": branch1.raw_rollout_text[:200],
                "b2_text": branch2.raw_rollout_text[:200],
                "b1_truncated": branch1.step_branch_was_truncated,
                "b2_truncated": branch2.step_branch_was_truncated,
            },
        )
    )

    l2 = l2_compute(branch1, branch2, config)
    state.trace.append(TraceEvent(state.step_count, "l2", l2.to_dict()))

    arbitration = None
    if l2.triggered_arbitration:
        arbitration = llm_arbitrate(state, llm, branch1, branch2, config)
        state.trace.append(TraceEvent(state.step_count, "arbitration", arbitration.to_dict()))

    if config.collect_branch_logs:
        state.branch_logs.append(
            {
                "step_idx": state.step_count,
                "problem_text": state.problem_text,
                "assistant_prefix_text": state.assistant_prefix_text,
                "l0": l0.to_dict(),
                "branch1": branch1.to_dict(),
                "branch2": branch2.to_dict(),
                "l2": l2.to_dict(),
                "arbitration": arbitration.to_dict() if arbitration is not None else None,
            }
        )

    if not l2.triggered_arbitration:
        return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0, l1=(branch1, branch2), l2=l2)

    if arbitration is None or arbitration.is_invalid:
        if config.invalid_fallback == "llm_full":
            return CascadeResult(
                decision=Decision.LLM_FULL,
                l0=l0,
                l1=(branch1, branch2),
                l2=l2,
                arbitration=arbitration,
            )
        return CascadeResult(
            decision=Decision.SLM_DIRECT,
            l0=l0,
            l1=(branch1, branch2),
            l2=l2,
            arbitration=arbitration,
        )

    winner = branch1 if arbitration.winner_idx == 0 else branch2
    loser = branch2 if arbitration.winner_idx == 0 else branch1
    state.rejected_branches_log.append(
        RejectedBranch(
            step_idx=state.step_count,
            loser_text=loser.step_branch_text,
            winner_text=winner.step_branch_text,
            l2=l2,
        )
    )
    return CascadeResult(
        decision=Decision.LLM_ARBITRATE,
        l0=l0,
        l1=(branch1, branch2),
        l2=l2,
        arbitration=arbitration,
        winner_branch=winner,
    )


def _generate_step_with_engine(state: GenerationState, engine, config: BPAConfig, account: str, prefix_extension: str = "") -> tuple[str, str]:
    rendered = render_for_continuation(
        state.problem_text,
        state.assistant_prefix_text + prefix_extension,
        engine.ensure_tokenizer(),
    )
    sampling = engine.sampling_params(
        max_tokens=config.max_step_tokens,
        temperature=0.0,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    out = engine.generate(rendered, sampling)[0]
    token_count = len(generated_token_ids(out))
    if account == "slm":
        state.slm_decode_tokens += token_count
        state.slm_prefill_tokens += len(engine.encode(rendered))
    else:
        state.llm_decode_tokens += token_count
        state.llm_prefill_tokens += len(engine.encode(rendered))
        state.llm_full_calls += 1
    return generated_text(out), finish_reason(out)


def _slm_generate_step(state: GenerationState, slm, config: BPAConfig) -> tuple[str, str]:
    return _generate_step_with_engine(state, slm, config, account="slm")


def _slm_continue_step(state: GenerationState, slm, config: BPAConfig, prefix_extension: str) -> tuple[str, str]:
    return _generate_step_with_engine(state, slm, config, account="slm", prefix_extension=prefix_extension)


def _llm_generate_step(state: GenerationState, llm, config: BPAConfig) -> tuple[str, str]:
    return _generate_step_with_engine(state, llm, config, account="llm")


def generate_one_step(state: GenerationState, slm, llm, config: BPAConfig, cascade: CascadeResult) -> tuple[str, str]:
    if cascade.decision == Decision.SLM_DIRECT:
        return _slm_generate_step(state, slm, config)

    if cascade.decision == Decision.LLM_FULL:
        return _llm_generate_step(state, llm, config)

    if cascade.decision == Decision.LLM_ARBITRATE:
        if not config.apply_arbitration:
            return _slm_generate_step(state, slm, config)
        if cascade.winner_branch is None:
            return _slm_generate_step(state, slm, config)
        winner = cascade.winner_branch
        if winner.step_branch_was_truncated:
            return winner.step_branch_text + "\n\n", "stop_in_branch"
        suffix_text, finish = _slm_continue_step(state, slm, config, prefix_extension=winner.step_branch_text)
        return winner.step_branch_text + suffix_text, finish

    raise ValueError(f"Unknown cascade decision: {cascade.decision}")


def _ends_with_chunk_stop(text: str, stops: list[str]) -> bool:
    return any(text.endswith(stop) for stop in stops)


def llm_generate_final_with_rep_guard(state: GenerationState, llm, config: BPAConfig) -> tuple[str, str]:
    rep = RepetitionState()
    accumulated_text = ""
    accumulated_tokens = 0
    chunk_stops = [".\n", "!\n", "?\n", "\n\n"]

    while accumulated_tokens < config.final_answer_max_tokens:
        rendered = render_for_continuation(
            state.problem_text,
            state.assistant_prefix_text + accumulated_text,
            llm.ensure_tokenizer(),
        )
        remaining = config.final_answer_max_tokens - accumulated_tokens
        sampling = llm.sampling_params(
            max_tokens=min(remaining, config.final_answer_chunk_tokens),
            temperature=0.0,
            stop=chunk_stops,
            include_stop_str_in_output=True,
        )
        out = llm.generate(rendered, sampling)[0]
        chunk_text = generated_text(out)
        chunk_tokens = len(generated_token_ids(out))
        chunk_finish = finish_reason(out)

        state.llm_decode_tokens += chunk_tokens
        state.llm_prefill_tokens += len(llm.encode(rendered))
        state.llm_full_calls += 1

        accumulated_text += chunk_text
        accumulated_tokens += chunk_tokens

        if chunk_finish == "eos" or (chunk_finish == "stop" and not _ends_with_chunk_stop(chunk_text, chunk_stops)):
            return accumulated_text, "eos"

        trigger = update_repetition(
            rep,
            chunk_text,
            config.repetition_ngram_size,
            config.repetition_ngram_threshold,
        )
        if trigger is not None:
            return accumulated_text, "repetition"

        if not chunk_text or chunk_tokens == 0:
            return accumulated_text, "empty_chunk"

    return accumulated_text, "max_tokens"


def _normalized_step_for_duplicate(step_text: str) -> str:
    return step_text.rstrip()


def bpa_solve(problem_text: str, slm, llm, config: BPAConfig) -> BPAResult:
    state = GenerationState(problem_text=problem_text)
    rep = RepetitionState()
    previous_final_step: str | None = None
    start_time = time.time()
    step_logs: list[dict] = []

    while state.phase != Phase.DONE:
        if state.slm_decode_tokens + state.llm_decode_tokens >= config.max_total_tokens:
            state.trace.append(TraceEvent(state.step_count, "budget_exhausted_total", {}))
            state.stop_reason = "budget"
            break

        phase_before_step = state.phase

        if state.phase == Phase.FINAL_ANSWER and config.final_answer_mode == "llm_chunked":
            final_text, final_stop_reason = llm_generate_final_with_rep_guard(state, llm, config)
            state.assistant_prefix_text += final_text
            state.phase = Phase.DONE
            state.stop_reason = f"final_{final_stop_reason}"
            if final_stop_reason == "repetition":
                state.trace.append(TraceEvent(state.step_count, "final_answer_stopped_by_repetition", {}))
            break

        intervention_disabled = state.llm_scoring_calls >= config.max_llm_interventions * 2
        cascade = run_cascade(state, slm, llm, config, disabled=intervention_disabled)
        step_text, finish = generate_one_step(state, slm, llm, config, cascade)
        step_text_normalized = ensure_step_terminator(step_text, finish)
        state.assistant_prefix_text += step_text_normalized
        state.step_count += 1

        check_and_transition_phase(state, step_text_normalized)

        if phase_before_step == Phase.FINAL_ANSWER:
            if finish == "eos":
                state.phase = Phase.DONE
                state.stop_reason = "final_eos"
            else:
                normalized_final_step = _normalized_step_for_duplicate(step_text_normalized)
                if previous_final_step is not None and normalized_final_step == previous_final_step:
                    state.phase = Phase.DONE
                    state.stop_reason = "final_duplicate_step"
                    state.trace.append(
                        TraceEvent(
                            state.step_count,
                            "final_answer_duplicate_step",
                            {"step_text": normalized_final_step[:200]},
                        )
                    )
                else:
                    previous_final_step = normalized_final_step

        step_logs.append(
            {
                "step_idx": state.step_count - 1,
                "decision": cascade.decision.value,
                "finish_reason": finish,
                "step_text": step_text_normalized,
                "phase_after": state.phase.value,
            }
        )

        if state.phase == Phase.THINKING:
            trigger = update_repetition(
                rep,
                step_text_normalized,
                config.repetition_ngram_size,
                config.repetition_ngram_threshold,
            )
            if trigger is not None:
                state.assistant_prefix_text += CLOSE_THINK_TAG + "\n\n"
                state.has_seen_close_think = True
                state.phase = Phase.FINAL_ANSWER
                state.trace.append(
                    TraceEvent(
                        state.step_count,
                        "thinking_repetition_force_close",
                        {"trigger_reason": trigger},
                    )
                )
                rep = RepetitionState()

        if finish == "eos" and phase_before_step != Phase.FINAL_ANSWER:
            state.phase = Phase.DONE
            state.stop_reason = "eos"

    state.trace.append(TraceEvent(state.step_count, "step_logs", {"steps": step_logs}))
    return BPAResult(
        answer=extract_answer(state.assistant_prefix_text),
        state=state,
        total_wall_time=time.time() - start_time,
    )


def solve_engine_only(problem_text: str, engine, config: BPAConfig, account: str) -> BPAResult:
    state = GenerationState(problem_text=problem_text)
    start_time = time.time()
    rendered = render_for_continuation(problem_text, "", engine.ensure_tokenizer())
    sampling = engine.sampling_params(max_tokens=config.max_total_tokens, temperature=0.0)
    out = engine.generate(rendered, sampling)[0]
    text = generated_text(out)
    token_count = len(generated_token_ids(out))
    if account == "slm":
        state.slm_decode_tokens += token_count
        state.slm_prefill_tokens += len(engine.encode(rendered))
    else:
        state.llm_decode_tokens += token_count
        state.llm_prefill_tokens += len(engine.encode(rendered))
        state.llm_full_calls += 1
    state.assistant_prefix_text = text
    state.phase = Phase.DONE
    state.stop_reason = finish_reason(out) or "done"
    return BPAResult(answer=extract_answer(text), state=state, total_wall_time=time.time() - start_time)
