from __future__ import annotations

import time

from .cascade.l0 import l0_filter
from .config import BPAConfig
from .context_budget import ContextBudgetExceeded, generation_budget_for_rendered
from .engines import finish_reason, generated_text, generated_token_ids
from .render import render_for_continuation
from .safety import ensure_step_terminator, extract_answer, update_strict_step_repetition
from .state import CascadeResult, Decision, GenerationState, Phase, RepetitionState, TraceEvent
from .trace import BPAResult


EOS_FINISH_REASONS = {"eos"}


def run_cascade(state: GenerationState, slm, llm, config: BPAConfig) -> CascadeResult:
    l0 = l0_filter(state, slm, config)
    state.trace.append(TraceEvent(state.step_count, "l0", l0.to_dict()))
    decision = Decision.LLM_FULL if l0.passed else Decision.SLM_DIRECT
    return CascadeResult(decision=decision, l0=l0)


def _generate_step_with_engine(state: GenerationState, engine, config: BPAConfig, account: str, prefix_extension: str = "") -> tuple[str, str]:
    rendered = render_for_continuation(
        state.problem_text,
        state.assistant_prefix_text + prefix_extension,
        engine.ensure_tokenizer(),
    )
    remaining_total_tokens = max(config.max_total_tokens - state.slm_decode_tokens - state.llm_decode_tokens, 1)
    requested_step_tokens = min(config.max_step_tokens, remaining_total_tokens)
    max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, engine, config, requested_step_tokens)
    sampling = engine.sampling_params(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    generate_start = time.time()
    out = engine.generate(rendered, sampling)[0]
    generate_wall_time = time.time() - generate_start
    token_count = len(generated_token_ids(out))
    if account == "slm":
        state.slm_generate_calls += 1
        state.slm_wall_time += generate_wall_time
        state.slm_decode_tokens += token_count
        state.slm_prefill_tokens += prompt_tokens
    else:
        state.llm_generation_wall_time += generate_wall_time
        state.llm_decode_tokens += token_count
        state.llm_prefill_tokens += prompt_tokens
        state.llm_full_calls += 1
    return generated_text(out), finish_reason(out)


def _slm_generate_step(state: GenerationState, slm, config: BPAConfig) -> tuple[str, str]:
    return _generate_step_with_engine(state, slm, config, account="slm")


def _llm_generate_step(state: GenerationState, llm, config: BPAConfig) -> tuple[str, str]:
    return _generate_step_with_engine(state, llm, config, account="llm")


def generate_one_step(state: GenerationState, slm, llm, config: BPAConfig, cascade: CascadeResult) -> tuple[str, str]:
    if cascade.decision == Decision.SLM_DIRECT:
        return _slm_generate_step(state, slm, config)

    if cascade.decision == Decision.LLM_FULL:
        return _llm_generate_step(state, llm, config)

    raise ValueError(f"Unknown cascade decision: {cascade.decision}")


def _is_eos_finish(finish: str) -> bool:
    return finish in EOS_FINISH_REASONS


def _generation_source(cascade: CascadeResult) -> str:
    if cascade.decision == Decision.LLM_FULL:
        return "llm"
    return "slm"


def bpa_solve(problem_text: str, slm, llm, config: BPAConfig) -> BPAResult:
    state = GenerationState(problem_text=problem_text)
    rep = RepetitionState()
    start_time = time.time()
    step_logs: list[dict] = []

    while state.phase != Phase.DONE:
        if state.slm_decode_tokens + state.llm_decode_tokens >= config.max_total_tokens:
            state.trace.append(TraceEvent(state.step_count, "total_token_budget_exhausted", {}))
            state.stop_reason = "total_token_budget"
            break

        try:
            cascade = run_cascade(state, slm, llm, config)
            if state.slm_decode_tokens + state.llm_decode_tokens >= config.max_total_tokens:
                state.trace.append(TraceEvent(state.step_count, "total_token_budget_exhausted", {}))
                state.stop_reason = "total_token_budget"
                break
            decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
            step_text, finish = generate_one_step(state, slm, llm, config, cascade)
        except ContextBudgetExceeded as exc:
            state.phase = Phase.DONE
            state.stop_reason = "context_budget"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
            break
        step_text_normalized = ensure_step_terminator(step_text, finish)
        generated_step_tokens = state.slm_decode_tokens + state.llm_decode_tokens - decode_tokens_before
        state.assistant_prefix_text += step_text_normalized
        state.step_count += 1

        log_row = {
            "step_idx": state.step_count - 1,
            "decision": cascade.decision.value,
            "generation_source": _generation_source(cascade),
            "finish_reason": finish,
            "step_text": step_text_normalized,
            "generated_step_tokens": generated_step_tokens,
        }
        step_logs.append(log_row)

        if generated_step_tokens <= 0 and not step_text_normalized.strip():
            state.phase = Phase.DONE
            state.stop_reason = "empty_step"
            state.trace.append(TraceEvent(state.step_count, "empty_step", {}))
            break

        trigger = update_strict_step_repetition(rep, step_text_normalized)
        if trigger is not None:
            state.phase = Phase.DONE
            state.stop_reason = trigger
            state.trace.append(TraceEvent(state.step_count, "step_repetition_stop", {"trigger_reason": trigger}))
            break

        if _is_eos_finish(finish):
            state.phase = Phase.DONE
            state.stop_reason = "eos"

    state.trace.append(TraceEvent(state.step_count, "step_logs", {"steps": step_logs}))
    return BPAResult(
        answer=extract_answer(state.assistant_prefix_text),
        state=state,
        total_wall_time=time.time() - start_time,
    )


def solve_engine_only(problem_text: str, engine, config: BPAConfig, account: str) -> BPAResult:
    state = GenerationState(problem_text=problem_text, generation_protocol="oneshot")
    start_time = time.time()
    rendered = render_for_continuation(problem_text, "", engine.ensure_tokenizer())
    try:
        max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, engine, config, config.max_total_tokens)
    except ContextBudgetExceeded as exc:
        state.phase = Phase.DONE
        state.stop_reason = "context_budget"
        state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
        return BPAResult(answer=None, state=state, total_wall_time=time.time() - start_time)
    sampling = engine.sampling_params(max_tokens=max_tokens, temperature=0.0)
    generate_start = time.time()
    out = engine.generate(rendered, sampling)[0]
    generate_wall_time = time.time() - generate_start
    text = generated_text(out)
    token_count = len(generated_token_ids(out))
    if account == "slm":
        state.slm_generate_calls += 1
        state.slm_wall_time += generate_wall_time
        state.slm_decode_tokens += token_count
        state.slm_prefill_tokens += prompt_tokens
    else:
        state.llm_generation_wall_time += generate_wall_time
        state.llm_decode_tokens += token_count
        state.llm_prefill_tokens += prompt_tokens
        state.llm_full_calls += 1
    state.assistant_prefix_text = text
    state.phase = Phase.DONE
    state.stop_reason = finish_reason(out) or "done"
    return BPAResult(answer=extract_answer(text), state=state, total_wall_time=time.time() - start_time)
