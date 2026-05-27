from __future__ import annotations

import time
from typing import Any

from bpa.context_budget import ContextBudgetExceeded
from bpa.safety import CLOSE_THINK_TAG, extract_answer_from_final_step, has_close_think_tag
from bpa.state import GenerationState, Phase, TraceEvent
from bpa.trace import BPAResult

from .config import SARRConfig
from .controller import (
    MODE_COLD_START,
    MODE_FINALIZE,
    MODE_LLM_FINALIZE,
    MODE_LLM_REPAIR,
    MODE_SLM_NORMAL,
    MODE_SLM_PROBATION,
    MODE_SLM_REENTRY,
    MODE_SLM_TRANSITION,
    OWNER_LLM,
    OWNER_SLM,
    PDIController,
    STEP_FINAL_ANSWER,
    Step,
    has_answer_intent,
)
from .records import StepOutput


ANSWER_INTENT_TERMINAL_PEEK_TOKENS = 8


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
        return
    state.llm_generation_wall_time += wall_time
    state.llm_decode_tokens += token_count
    state.llm_prefill_tokens += prompt_tokens
    state.llm_full_calls += 1


def _actual_token_count(output: StepOutput) -> int:
    return int(output.extra.get("actual_token_count") or output.token_count)


def _stop_strings_for_slm(cfg: SARRConfig) -> list[str]:
    return [stop for stop in cfg.generation.step_delimiters if stop]


def _stop_strings_for_llm(cfg: SARRConfig) -> list[str]:
    stops: list[str] = []
    for stop in [*cfg.generation.step_delimiters, CLOSE_THINK_TAG]:
        if stop and stop not in stops:
            stops.append(stop)
    return stops


def _llm_repair_problem_text(problem_text: str, cfg: SARRConfig) -> str:
    instruction = (cfg.generation.llm_repair_instruction or "").strip()
    if not instruction:
        return problem_text
    return f"{problem_text}\n\nLLM repair instruction:\n{instruction}"


def _strict_thinking_stop_reason(generation: StepOutput, step_text: str) -> str | None:
    if generation.token_count <= 0 and not step_text.strip():
        return "empty_step"
    if CLOSE_THINK_TAG in step_text:
        return "finished"
    if generation.finish_reason == "eos":
        return "eos"
    if generation.finish_reason == "length":
        return "length"
    return None


def _final_answer_prefix(thinking_text: str) -> str:
    thinking = thinking_text.split(CLOSE_THINK_TAG, 1)[0] if CLOSE_THINK_TAG in thinking_text else thinking_text
    return f"{thinking.rstrip()}\n{CLOSE_THINK_TAG}\n\n"


def _generate_slm_step(
    slm,
    state: GenerationState,
    cfg: SARRConfig,
    *,
    remaining_think_tokens: int,
) -> StepOutput:
    max_new_tokens = max(1, remaining_think_tokens)
    output = slm.generate_step(
        state.problem_text,
        state.assistant_prefix_text,
        max_new_tokens=max_new_tokens,
        stop_delimiters=_stop_strings_for_slm(cfg),
        capture_token_entropy=False,
        capture_token_logprobs=True,
        topk_entropy=cfg.confidence.top_k,
        immediate_stop_strings=[CLOSE_THINK_TAG],
    )
    _account_generation_cost(
        state,
        "slm",
        wall_time=output.wall_time,
        token_count=_actual_token_count(output),
        prompt_tokens=output.prompt_tokens,
    )
    return output


def _generate_llm_step(
    llm,
    state: GenerationState,
    cfg: SARRConfig,
    *,
    remaining_think_tokens: int,
) -> StepOutput:
    max_new_tokens = max(1, remaining_think_tokens)
    output = llm.generate_step(
        _llm_repair_problem_text(state.problem_text, cfg),
        state.assistant_prefix_text,
        max_new_tokens=max_new_tokens,
        stop_delimiters=_stop_strings_for_llm(cfg),
        capture_token_logprobs=False,
        topk_logprobs=1,
    )
    _account_generation_cost(
        state,
        "llm",
        wall_time=output.wall_time,
        token_count=_actual_token_count(output),
        prompt_tokens=output.prompt_tokens,
    )
    return output


def _generate_self_reentry_candidates(
    slm,
    state: GenerationState,
    cfg: SARRConfig,
    controller: PDIController,
) -> list[StepOutput]:
    outputs: list[StepOutput] = []
    tentative_prefix = controller.active_text()
    required_scored_tokens = max(
        cfg.controller.t_min,
        cfg.controller.self_reentry_min_scored_tokens or cfg.controller.t_min,
    )
    scored_tokens = 0

    for attempt_idx in range(1, cfg.controller.self_reentry_max_attempt_steps + 1):
        tentative_tokens = sum(_actual_token_count(output) for output in outputs)
        remaining = cfg.generation.think_token_budget - controller.visible_token_count() - tentative_tokens
        if remaining <= 0:
            break

        output = slm.generate_step(
            state.problem_text,
            tentative_prefix,
            max_new_tokens=remaining,
            stop_delimiters=_stop_strings_for_slm(cfg),
            capture_token_entropy=False,
            capture_token_logprobs=True,
            topk_entropy=cfg.confidence.top_k,
            immediate_stop_strings=[CLOSE_THINK_TAG],
        )
        output.extra["self_reentry_candidate_attempt"] = attempt_idx
        _account_generation_cost(
            state,
            "slm",
            wall_time=output.wall_time,
            token_count=_actual_token_count(output),
            prompt_tokens=output.prompt_tokens,
        )
        outputs.append(output)

        generated_logprobs = output.extra.get("generated_token_logprobs")
        logprob_count = len(generated_logprobs) if isinstance(generated_logprobs, list) else 0
        scored_tokens += min(output.token_count, logprob_count)
        tentative_prefix += output.text

        if CLOSE_THINK_TAG in output.text:
            break
        if scored_tokens >= required_scored_tokens:
            break
        if output.token_count <= 0 or output.finish_reason in {"eos", "length"}:
            break

    return outputs


def _sync_prefix(state: GenerationState, controller: PDIController) -> None:
    state.assistant_prefix_text = controller.active_text()
    state.step_count = len(controller.active_steps())


def _force_close_if_needed(state: GenerationState, cfg: SARRConfig, reason: str) -> str:
    if has_close_think_tag(state.assistant_prefix_text):
        return reason
    state.assistant_prefix_text = state.assistant_prefix_text.rstrip() + cfg.generation.force_close_think_text
    state.trace.append(
        TraceEvent(
            state.step_count,
            "forced_close_think",
            {"reason": reason, "text": cfg.generation.force_close_think_text},
        )
    )
    return reason


def _answer_intent_terminal_peek(
    *,
    state: GenerationState,
    cfg: SARRConfig,
    controller: PDIController,
    step: Step,
    engine,
    account: str,
) -> bool:
    if step.finish_reason != "stop" or not has_answer_intent(step.text):
        return False

    remaining = cfg.generation.think_token_budget - controller.visible_token_count()
    max_new_tokens = min(ANSWER_INTENT_TERMINAL_PEEK_TOKENS, max(0, remaining))
    if max_new_tokens <= 0:
        return False

    output = engine.generate_text(
        state.problem_text,
        state.assistant_prefix_text,
        max_new_tokens=max_new_tokens,
        stop_delimiters=[CLOSE_THINK_TAG],
        include_stop_str_in_output=True,
    )
    _account_generation_cost(
        state,
        account,
        wall_time=output.wall_time,
        token_count=_actual_token_count(output),
        prompt_tokens=output.prompt_tokens,
    )

    found_close = CLOSE_THINK_TAG in output.text
    payload = {
        "step_id": step.step_id,
        "source": account.upper(),
        "peek_token_budget": max_new_tokens,
        "peek_token_count": output.token_count,
        "peek_finish_reason": output.finish_reason,
        "found_close_think": found_close,
    }
    step.extra["answer_intent_terminal_peek"] = payload
    state.trace.append(TraceEvent(state.step_count, "answer_intent_terminal_peek", payload))
    if not found_close:
        return False

    controller.mark_finished(step, reason="finished")
    return True


def _append_final_answer(
    *,
    problem_id: str,
    state: GenerationState,
    slm,
    llm,
    cfg: SARRConfig,
    controller: PDIController,
    generator: str,
) -> tuple[str | None, str, dict[str, Any]]:
    account = generator
    engine = llm if account == "llm" else slm
    prefix = _final_answer_prefix(state.assistant_prefix_text)
    output = engine.generate_text(
        state.problem_text,
        prefix,
        max_new_tokens=cfg.generation.answer_token_budget,
        stop_delimiters=[CLOSE_THINK_TAG],
        include_stop_str_in_output=False,
    )
    generated_token_count = _actual_token_count(output)
    if CLOSE_THINK_TAG in output.text:
        output.extra["final_answer_truncated_at_close_think"] = True
        output.text = output.text.split(CLOSE_THINK_TAG, 1)[0].rstrip()
        output.token_ids = engine.encode(output.text)
    _account_generation_cost(
        state,
        account,
        wall_time=output.wall_time,
        token_count=generated_token_count,
        prompt_tokens=output.prompt_tokens,
    )
    row: dict[str, Any] = {
        "problem_id": problem_id,
        "step_id": controller.next_step_id,
        "source": account.upper(),
        "generator": account,
        "status": STEP_FINAL_ANSWER,
        "active": True,
        "text": output.text,
        "token_ids": output.token_ids,
        "token_count": output.token_count,
        "finish_reason": output.finish_reason,
        "action": "FINAL_ANSWER",
        "is_final_answer": True,
    }
    state.assistant_prefix_text = prefix + output.text
    state.step_count = controller.next_step_id
    state.trace.append(TraceEvent(state.step_count, "final_answer", row))
    answer = extract_answer_from_final_step(output.text)
    return answer, account, row


def _step_transition_rows(problem_id: str, steps: list[Step]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    active_steps = [step for step in steps if step.active]
    for prev, curr in zip(active_steps, active_steps[1:]):
        rows.append(
            {
                "problem_id": problem_id,
                "prev_step_id": prev.step_id,
                "curr_step_id": curr.step_id,
                "prev_generator": prev.owner.lower(),
                "curr_generator": curr.owner.lower(),
                "transition_type": f"{prev.owner}->{curr.owner}",
            }
        )
    return rows


def run_sarr_code(
    *,
    problem_id: str,
    problem_text: str,
    slm,
    llm,
    cfg: SARRConfig,
) -> tuple[BPAResult, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    state = GenerationState(problem_text=problem_text, generation_protocol=cfg.method)
    start_time = time.time()
    controller = PDIController(problem_id, cfg.controller)
    attempt_id = 1
    stop_reason: str | None = None
    final_answer_row: dict[str, Any] | None = None
    slm_scoring_wall_time = 0.0
    slm_scoring_count = 0

    try:
        while state.phase != Phase.DONE:
            controller.validate_state()
            visible_tokens = controller.visible_token_count()
            if visible_tokens >= cfg.generation.think_token_budget:
                stop_reason = "think_token_budget"
                controller.force_finalize(reason=stop_reason)
                break

            remaining = cfg.generation.think_token_budget - visible_tokens
            mode = controller.state.mode

            if mode in {MODE_FINALIZE, MODE_LLM_FINALIZE}:
                stop_reason = "llm_finalize" if mode == MODE_LLM_FINALIZE else (stop_reason or "finalize")
                break

            if mode in {MODE_COLD_START, MODE_SLM_NORMAL}:
                output = _generate_slm_step(slm, state, cfg, remaining_think_tokens=remaining)
                step = controller.append_step(output, owner=OWNER_SLM, action="SLM_CONTINUE", attempt_id=attempt_id)
                attempt_id += 1
                _sync_prefix(state, controller)

                thinking_stop = _strict_thinking_stop_reason(output, output.text)
                if thinking_stop is not None:
                    controller.mark_finished(step, reason=thinking_stop)
                    stop_reason = thinking_stop
                    break

                if _answer_intent_terminal_peek(
                    state=state,
                    cfg=cfg,
                    controller=controller,
                    step=step,
                    engine=slm,
                    account="slm",
                ):
                    stop_reason = "finished"
                    break

                decision = controller.process_slm_window(step)
                if step.active:
                    step.action = decision.action
                _sync_prefix(state, controller)
                if controller.state.mode == MODE_FINALIZE:
                    stop_reason = "early_stop"
                    break
                continue

            if mode == MODE_SLM_TRANSITION:
                output = _generate_slm_step(slm, state, cfg, remaining_think_tokens=remaining)
                step = controller.append_step(output, owner=OWNER_SLM, action="SLM_TRANSITION_CONTINUE", attempt_id=attempt_id)
                attempt_id += 1
                _sync_prefix(state, controller)

                thinking_stop = _strict_thinking_stop_reason(output, output.text)
                if thinking_stop is not None:
                    controller.mark_finished(step, reason=thinking_stop)
                    stop_reason = thinking_stop
                    break

                if _answer_intent_terminal_peek(
                    state=state,
                    cfg=cfg,
                    controller=controller,
                    step=step,
                    engine=slm,
                    account="slm",
                ):
                    stop_reason = "finished"
                    break

                decision = controller.process_transition_window(step)
                if step.active:
                    step.action = decision.action
                _sync_prefix(state, controller)
                continue

            if mode == MODE_LLM_REPAIR:
                output = _generate_llm_step(llm, state, cfg, remaining_think_tokens=remaining)
                step = controller.append_step(output, owner=OWNER_LLM, action="LLM_REPAIR_CONTINUE", attempt_id=attempt_id)
                attempt_id += 1
                _sync_prefix(state, controller)

                thinking_stop = _strict_thinking_stop_reason(output, output.text)
                if thinking_stop is not None:
                    controller.mark_finished(step, reason=thinking_stop)
                    stop_reason = thinking_stop
                    break

                if _answer_intent_terminal_peek(
                    state=state,
                    cfg=cfg,
                    controller=controller,
                    step=step,
                    engine=llm,
                    account="llm",
                ):
                    stop_reason = "finished"
                    break

                handoff_payload = controller.repair_step_for_handoff(step)
                if handoff_payload is not None:
                    prefix_text, suffix_text, _suffix_steps = handoff_payload
                    handoff_strategy = cfg.controller.handoff_strategy

                    # Compute the old suffix score once: the new strategy logs it as a shadow signal,
                    # while the old strategy still uses it for the handoff decision.
                    old_slm_side_pdi: float | None = None
                    old_slm_side_q: float | None = None
                    if int(step.token_count) > 0:
                        score = slm.score_suffix_pdi(state.problem_text, prefix_text, suffix_text)
                        slm_scoring_wall_time += float(score.get("wall_time") or 0.0)
                        slm_scoring_count += 1
                        step.extra["slm_side_handoff_score"] = {
                            "strategy": "latest_llm_step_only",
                            "pdi": score["pdi"],
                            "token_count": score["token_count"],
                            "logprobs": score.get("logprobs", []),
                            "token_ids": score.get("token_ids", []),
                            "tokens": score.get("tokens", []),
                            "token_offsets": score.get("token_offsets", []),
                            "prompt_tokens": score["prompt_tokens"],
                            "wall_time": score["wall_time"],
                            "covered_step_ids": [s.step_id for s in _suffix_steps],
                        }
                        if int(score.get("token_count") or 0) > 0 and math_is_finite(score.get("pdi")):
                            old_slm_side_pdi = float(score["pdi"])

                    if handoff_strategy == "self_reentry_certification":
                        candidate_outputs = _generate_self_reentry_candidates(slm, state, cfg, controller)
                        decision = controller.process_self_reentry_candidate(
                            llm_repair_step=step,
                            candidate_outputs=candidate_outputs,
                            old_slm_side_pdi=old_slm_side_pdi,
                            old_slm_side_q=None,
                        )
                        step.action = decision.action
                        if controller.state.mode == MODE_SLM_REENTRY:
                            _sync_prefix(state, controller)
                            continue

                    else:  # repair_landing_index (old method)
                        if old_slm_side_pdi is not None:
                            decision = controller.process_handoff_score(step=step, slm_side_pdi=old_slm_side_pdi)
                            step.action = decision.action
                            if controller.state.mode == MODE_SLM_REENTRY:
                                _sync_prefix(state, controller)
                                continue
                        else:
                            controller.reset_handoff_candidate_buffer(step=step, reason="invalid_slm_side_score")

                decision = controller.note_llm_repair_step(step)
                step.action = decision.action
                if controller.state.mode == MODE_LLM_FINALIZE:
                    stop_reason = "llm_finalize"
                    break
                continue

            if mode in {MODE_SLM_REENTRY, MODE_SLM_PROBATION}:
                output = _generate_slm_step(slm, state, cfg, remaining_think_tokens=remaining)
                step = controller.append_step(output, owner=OWNER_SLM, action="SLM_REENTRY_CONTINUE", attempt_id=attempt_id)
                attempt_id += 1
                _sync_prefix(state, controller)

                thinking_stop = _strict_thinking_stop_reason(output, output.text)
                if thinking_stop is not None:
                    controller.mark_finished(step, reason=thinking_stop)
                    stop_reason = thinking_stop
                    break

                if _answer_intent_terminal_peek(
                    state=state,
                    cfg=cfg,
                    controller=controller,
                    step=step,
                    engine=slm,
                    account="slm",
                ):
                    stop_reason = "finished"
                    break

                decision = controller.process_reentry_window(step)
                if step.active:
                    step.action = decision.action
                _sync_prefix(state, controller)
                continue

            raise RuntimeError(f"Unknown PDI controller mode: {mode}")

    except ContextBudgetExceeded as exc:
        state.phase = Phase.DONE
        stop_reason = "context_budget"
        controller.force_finalize(reason=stop_reason)
        state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    if stop_reason is None:
        stop_reason = "done"

    if stop_reason != "context_budget":
        should_close = (
            stop_reason in {"think_token_budget", "done", "empty_step", "length", "early_stop", "llm_finalize"}
            or not has_close_think_tag(state.assistant_prefix_text)
        )
        if should_close:
            stop_reason = _force_close_if_needed(state, cfg, stop_reason)

    state.stop_reason = stop_reason

    answer = None
    final_answer_generator = None
    if stop_reason != "context_budget":
        final_generator = "llm" if controller.state.mode == MODE_LLM_FINALIZE else cfg.generation.final_answer_generator
        try:
            answer, final_answer_generator, final_answer_row = _append_final_answer(
                problem_id=problem_id,
                state=state,
                slm=slm,
                llm=llm,
                cfg=cfg,
                controller=controller,
                generator=final_generator,
            )
        except ContextBudgetExceeded as exc:
            state.stop_reason = "context_budget_final_answer"
            stop_reason = state.stop_reason
            controller.force_finalize(reason=stop_reason)
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    state.phase = Phase.DONE
    total_wall_time = time.time() - start_time

    llm_wall_time = state.llm_generation_wall_time + state.llm_scoring_wall_time
    summary = controller.summary(
        finish_reason=state.stop_reason or stop_reason,
        final_answer=answer,
        final_answer_generator=final_answer_generator,
        total_wall_time=total_wall_time,
        slm_wall_time=state.slm_wall_time,
        llm_wall_time=llm_wall_time,
        slm_scoring_wall_time=slm_scoring_wall_time,
        slm_scoring_count=slm_scoring_count,
        slm_prefill_count=state.slm_generate_calls,
        llm_prefill_count=state.llm_full_calls,
    )
    state.trace.append(TraceEvent(state.step_count, "sarr_summary", summary))
    state.trace.append(TraceEvent(state.step_count, "pdi_windows", {"windows": controller.serialize_windows()}))
    state.trace.append(TraceEvent(state.step_count, "controller_events", {"events": controller.events}))

    result = BPAResult(answer=answer, state=state, total_wall_time=total_wall_time)
    step_rows = controller.serialize_steps()
    if final_answer_row is not None:
        step_rows.append(final_answer_row)
    return result, step_rows, controller.events, _step_transition_rows(problem_id, controller.steps)


def math_is_finite(value: Any) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
