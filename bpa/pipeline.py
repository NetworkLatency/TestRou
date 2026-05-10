from __future__ import annotations

import time

from .cascade.l0 import l0_filter
from .config import BPAConfig
from .context_budget import ContextBudgetExceeded, generation_budget_for_rendered
from .engines import completion, finish_reason, generated_text, generated_token_ids
from .render import render_for_continuation
from .safety import (
    captured_close_think_prefix,
    ensure_step_terminator,
    extract_answer_from_final_step,
    extract_answer_from_steps,
    has_close_think_tag,
    update_strict_step_repetition,
)
from .state import CascadeResult, Decision, GenerationState, Phase, RepetitionState, TraceEvent
from .trace import BPAResult


EOS_FINISH_REASONS = {"eos"}


def run_cascade(state: GenerationState, slm, llm, config: BPAConfig) -> CascadeResult:
    l0 = l0_filter(state, slm, config)
    state.trace.append(TraceEvent(state.step_count, "l0", l0.to_dict()))
    decision = Decision.LLM_FULL if l0.passed else Decision.SLM_DIRECT
    return CascadeResult(decision=decision, l0=l0)


def _account_generation_cost(
    state: GenerationState,
    account: str,
    *,
    wall_time: float,
    token_count: int,
    prompt_tokens: int,
) -> None:
    if account == "slm":
        state.slm_generate_calls += 1
        state.slm_wall_time += wall_time
        state.slm_decode_tokens += token_count
        state.slm_prefill_tokens += prompt_tokens
    else:
        state.llm_generation_wall_time += wall_time
        state.llm_decode_tokens += token_count
        state.llm_prefill_tokens += prompt_tokens
        state.llm_full_calls += 1


def _tokenizer_eos_token_ids(engine) -> set[int]:
    tokenizer = engine.ensure_tokenizer()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    try:
        return {int(token_id) for token_id in eos_token_id if token_id is not None}
    except TypeError:
        return set()


def _output_hit_eos(output, engine) -> bool:
    if finish_reason(output) in EOS_FINISH_REASONS:
        return True

    eos_token_ids = _tokenizer_eos_token_ids(engine)
    if not eos_token_ids:
        return False

    stop_reason = getattr(completion(output), "stop_reason", None)
    if isinstance(stop_reason, int) and stop_reason in eos_token_ids:
        return True
    if isinstance(stop_reason, str):
        try:
            if int(stop_reason) in eos_token_ids:
                return True
        except ValueError:
            pass

    token_ids = generated_token_ids(output)
    return bool(token_ids and token_ids[-1] in eos_token_ids)


def _post_stop_lookahead(
    state: GenerationState,
    engine,
    config: BPAConfig,
    *,
    account: str,
    step_text: str,
    prefix_extension: str = "",
    total_token_offset: int = 0,
) -> tuple[str, str]:
    lookahead_tokens = int(config.post_stop_lookahead_tokens or 0)
    if lookahead_tokens <= 0:
        return "", "stop"

    remaining_total_tokens = config.max_total_tokens - state.slm_decode_tokens - state.llm_decode_tokens - total_token_offset
    if remaining_total_tokens <= 0:
        return "", "stop"

    lookahead_prefix = prefix_extension + ensure_step_terminator(step_text, "stop")
    rendered = render_for_continuation(
        state.problem_text,
        state.assistant_prefix_text + lookahead_prefix,
        engine.ensure_tokenizer(),
    )
    max_tokens, prompt_tokens = generation_budget_for_rendered(
        rendered,
        engine,
        config,
        min(lookahead_tokens, remaining_total_tokens),
    )
    sampling = engine.sampling_params(
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=1,
    )
    generate_start = time.time()
    out = engine.generate(rendered, sampling)[0]
    generate_wall_time = time.time() - generate_start
    token_count = len(generated_token_ids(out))
    _account_generation_cost(
        state,
        account,
        wall_time=generate_wall_time,
        token_count=token_count,
        prompt_tokens=prompt_tokens,
    )

    text = generated_text(out)
    close_think_prefix = captured_close_think_prefix(text)
    if close_think_prefix:
        return close_think_prefix, "stop"
    if finish_reason(out) == "stop" or _output_hit_eos(out, engine):
        return text, "eos"
    return "", "stop"


def _generate_step_with_engine(
    state: GenerationState,
    engine,
    config: BPAConfig,
    account: str,
    prefix_extension: str = "",
    step_token_budget: int | None = None,
    total_token_offset: int = 0,
    stop_on_step_boundary: bool = True,
) -> tuple[str, str]:
    rendered = render_for_continuation(
        state.problem_text,
        state.assistant_prefix_text + prefix_extension,
        engine.ensure_tokenizer(),
    )
    remaining_total_tokens = max(
        config.max_total_tokens - state.slm_decode_tokens - state.llm_decode_tokens - total_token_offset,
        1,
    )
    default_step_budget = config.max_step_tokens if stop_on_step_boundary else remaining_total_tokens
    requested_step_tokens = min(default_step_budget if step_token_budget is None else step_token_budget, remaining_total_tokens)
    max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, engine, config, requested_step_tokens)
    sampling_kwargs = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": 1,
    }
    if stop_on_step_boundary:
        sampling_kwargs["stop"] = ["\n\n"]
        sampling_kwargs["include_stop_str_in_output"] = True
    sampling = engine.sampling_params(**sampling_kwargs)
    generate_start = time.time()
    out = engine.generate(rendered, sampling)[0]
    generate_wall_time = time.time() - generate_start
    token_count = len(generated_token_ids(out))
    _account_generation_cost(
        state,
        account,
        wall_time=generate_wall_time,
        token_count=token_count,
        prompt_tokens=prompt_tokens,
    )

    text = generated_text(out)
    finish = finish_reason(out)
    if stop_on_step_boundary and finish == "stop":
        lookahead_text, lookahead_finish = _post_stop_lookahead(
            state,
            engine,
            config,
            account=account,
            step_text=text,
            prefix_extension=prefix_extension,
            total_token_offset=total_token_offset,
        )
        if lookahead_text or lookahead_finish == "eos":
            text += lookahead_text
            finish = lookahead_finish
    return text, finish


def _slm_generate_step(state: GenerationState, slm, config: BPAConfig) -> tuple[str, str]:
    return _generate_step_with_engine(state, slm, config, account="slm")


def _llm_generate_step(state: GenerationState, llm, config: BPAConfig) -> tuple[str, str]:
    return _generate_step_with_engine(state, llm, config, account="llm")


def _slm_generate_final_answer(state: GenerationState, slm, config: BPAConfig) -> tuple[str, str]:
    remaining_total_tokens = config.max_total_tokens - state.slm_decode_tokens - state.llm_decode_tokens
    return _generate_step_with_engine(
        state,
        slm,
        config,
        account="slm",
        step_token_budget=remaining_total_tokens,
        stop_on_step_boundary=False,
    )


def _in_final_answer_phase(state: GenerationState, config: BPAConfig) -> bool:
    return has_close_think_tag(state.assistant_prefix_text)


def generate_one_step(state: GenerationState, slm, llm, config: BPAConfig, cascade: CascadeResult) -> tuple[str, str]:
    if cascade.decision == Decision.SLM_DIRECT:
        return _slm_generate_step(state, slm, config)

    if cascade.decision == Decision.LLM_FULL:
        return _llm_generate_step(state, llm, config)

    raise ValueError(f"Unknown cascade decision: {cascade.decision}")


def _is_eos_finish(finish: str) -> bool:
    return finish in EOS_FINISH_REASONS


def _final_answer_stop_reason(finish: str) -> str:
    if _is_eos_finish(finish):
        return "eos"
    if finish:
        return f"final_answer_{finish}"
    return "final_answer_done"


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
            decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
            if _in_final_answer_phase(state, config):
                cascade = None
                step_text, finish = _slm_generate_final_answer(state, slm, config)
            else:
                cascade = run_cascade(state, slm, llm, config)
                if state.slm_decode_tokens + state.llm_decode_tokens >= config.max_total_tokens:
                    state.trace.append(TraceEvent(state.step_count, "total_token_budget_exhausted", {}))
                    state.stop_reason = "total_token_budget"
                    break
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
            "decision": "slm_final_answer" if cascade is None else cascade.decision.value,
            "generation_source": "slm" if cascade is None else _generation_source(cascade),
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

        if cascade is None:
            state.phase = Phase.DONE
            state.stop_reason = _final_answer_stop_reason(finish)
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
        answer=extract_answer_from_steps(step_logs, state.assistant_prefix_text),
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
    return BPAResult(answer=extract_answer_from_final_step(text), state=state, total_wall_time=time.time() - start_time)
