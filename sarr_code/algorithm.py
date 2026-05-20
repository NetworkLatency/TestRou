from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from bpa.context_budget import ContextBudgetExceeded
from bpa.safety import CLOSE_THINK_TAG, extract_answer_from_steps, has_close_think_tag
from bpa.state import GenerationState, Phase, TraceEvent
from bpa.trace import BPAResult

from .calibration import code_style_degeneration_event, smooth_confidence
from .config import SARRConfig
from .records import RollbackEvent, StepOutput, StepRecord


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


def _ensure_step_terminator(step_text: str, finish_reason: str, delimiters: list[str]) -> str:
    if finish_reason == "eos":
        return step_text
    primary = delimiters[0] if delimiters else "\n\n"
    if not step_text.endswith(primary):
        return step_text + primary
    return step_text


def _strict_thinking_stop_reason(generation: StepOutput, step_text: str) -> str | None:
    if generation.token_count <= 0 and not step_text.strip():
        return "empty_step"
    if CLOSE_THINK_TAG in step_text:
        return "finished"
    if generation.finish_reason == "eos":
        return "eos"
    return None


def _final_answer_prefix(thinking_text: str) -> str:
    thinking = thinking_text.split(CLOSE_THINK_TAG, 1)[0] if CLOSE_THINK_TAG in thinking_text else thinking_text
    return f"{thinking.rstrip()}\n{CLOSE_THINK_TAG}\n\n"


def _serialize_step(record: StepRecord) -> dict[str, Any]:
    row = asdict(record)
    row["token_count"] = record.token_count
    return row


def _serialize_removed(records: list[StepRecord]) -> list[dict[str, Any]]:
    return [
        {
            "step_id": r.step_id,
            "attempt_id": r.attempt_id,
            "generator": r.generator,
            "text": r.text,
            "token_count": r.token_count,
            "c_raw": r.c_raw,
            "c_norm": r.c_norm,
            "c_smooth": r.c_smooth,
        }
        for r in records
    ]


def rollback_to_anchor(
    problem_text: str,
    step_records: list[StepRecord],
    anchor_step_id: int,
    current_step_id: int,
) -> tuple[str, list[StepRecord], list[StepRecord], int]:
    kept = [r for r in step_records if r.step_id <= anchor_step_id]
    removed = [r for r in step_records if anchor_step_id < r.step_id <= current_step_id]
    context = "".join(r.text for r in kept)
    return context, kept, removed, current_step_id - anchor_step_id


def choose_best_prefix_anchor(step_records: list[StepRecord], *, allow_zero: bool = True) -> int:
    candidates: list[tuple[int, float]] = []
    if allow_zero:
        candidates.append((0, 0.0))
    for record in step_records:
        if record.c_smooth is not None:
            candidates.append((int(record.step_id), float(record.c_smooth)))
    if not candidates:
        return 0
    return max(candidates, key=lambda item: (item[1], -item[0]))[0]


def _visible_think_tokens(records: list[StepRecord]) -> int:
    return sum(r.token_count for r in records)


def _mark_removed(removed: list[StepRecord]) -> None:
    for record in removed:
        record.active = False
        record.removed_by_rollback = True


def _apply_transition_metadata(records: list[StepRecord]) -> dict[str, Any] | None:
    if len(records) < 2:
        return None
    prev = records[-2]
    curr = records[-1]
    transition_type = f"{prev.generator}->{curr.generator}"
    delta = None
    if prev.c_norm is not None and curr.c_norm is not None:
        delta = float(curr.c_norm - prev.c_norm)
    curr.transition_type = transition_type
    curr.delta_c_norm = delta
    return {
        "prev_step_id": prev.step_id,
        "curr_step_id": curr.step_id,
        "prev_generator": prev.generator,
        "curr_generator": curr.generator,
        "transition_type": transition_type,
        "delta_c_norm": delta,
    }


def _make_record(
    *,
    problem_id: str,
    attempt_id: int,
    step_id: int,
    generator: str,
    output: StepOutput,
    c_raw: float | None,
    c_norm: float | None,
    c_smooth: float | None,
    state_before: str,
    D_start: int,
    D_post: int,
    stable_anchor: int | None,
    action: str,
    c_info: dict[str, Any] | None = None,
    is_recovery: bool = False,
) -> StepRecord:
    extra = dict(output.extra)
    if c_info:
        extra["confidence"] = c_info
    return StepRecord(
        problem_id=problem_id,
        step_id=step_id,
        generator=generator,
        text=output.text,
        token_ids=output.token_ids,
        c_raw=c_raw,
        c_norm=c_norm,
        c_smooth=c_smooth,
        state_before=state_before,
        D_start=D_start,
        D_post=D_post,
        stable_anchor=stable_anchor,
        action=action,
        finish_reason=output.finish_reason,
        prompt_tokens=output.prompt_tokens,
        wall_time=output.wall_time,
        attempt_id=attempt_id,
        is_recovery=is_recovery,
        extra=extra,
    )


def _generate_slm_step(
    slm,
    state: GenerationState,
    cfg: SARRConfig,
    *,
    remaining_think_tokens: int,
) -> StepOutput:
    max_new_tokens = max(1, min(cfg.generation.max_new_tokens_per_step, remaining_think_tokens))
    output = slm.generate_step(
        state.problem_text,
        state.assistant_prefix_text,
        max_new_tokens=max_new_tokens,
        stop_delimiters=cfg.generation.step_delimiters,
        capture_token_entropy=cfg.confidence.capture_slm_token_entropy,
        topk_entropy=cfg.confidence.topk_entropy,
    )
    output.text = _ensure_step_terminator(output.text, output.finish_reason, cfg.generation.step_delimiters)
    _account_generation_cost(
        state,
        "slm",
        wall_time=output.wall_time,
        token_count=output.token_count,
        prompt_tokens=output.prompt_tokens,
    )
    return output


def _generate_llm_step(
    llm,
    state: GenerationState,
    cfg: SARRConfig,
    *,
    assistant_prefix_text: str,
    remaining_think_tokens: int,
) -> StepOutput:
    max_new_tokens = max(1, min(cfg.generation.max_new_tokens_per_step, remaining_think_tokens))
    output = llm.generate_step(
        state.problem_text,
        assistant_prefix_text,
        max_new_tokens=max_new_tokens,
        stop_delimiters=cfg.generation.step_delimiters,
    )
    output.text = _ensure_step_terminator(output.text, output.finish_reason, cfg.generation.step_delimiters)
    _account_generation_cost(
        state,
        "llm",
        wall_time=output.wall_time,
        token_count=output.token_count,
        prompt_tokens=output.prompt_tokens,
    )
    return output


def _confidence_for_prefix(slm, normalizer, cfg: SARRConfig, problem_text: str, assistant_prefix_text: str):
    c_raw, c_info = slm.continuation_confidence(
        problem_text,
        assistant_prefix_text,
        topk=cfg.confidence.topk_entropy,
    )
    c_norm = normalizer.transform(c_raw)
    return c_raw, c_norm, c_info


def confidence_gated_recovery(
    *,
    problem_id: str,
    attempt_id_start: int,
    start_step_id: int,
    state: GenerationState,
    context: str,
    llm,
    slm,
    normalizer,
    cfg: SARRConfig,
    max_recovery_steps: int,
    remaining_think_tokens: int,
) -> tuple[str, list[StepRecord], str, int]:
    records: list[StepRecord] = []
    attempt_id = attempt_id_start
    local_context = context
    for recovery_idx in range(1, max_recovery_steps + 1):
        remaining = max(1, remaining_think_tokens - _visible_think_tokens(records))
        output = _generate_llm_step(
            llm,
            state,
            cfg,
            assistant_prefix_text=local_context,
            remaining_think_tokens=remaining,
        )
        local_context += output.text
        c_raw, c_norm, c_info = _confidence_for_prefix(
            slm,
            normalizer,
            cfg,
            state.problem_text,
            local_context,
        )
        record = _make_record(
            problem_id=problem_id,
            attempt_id=attempt_id,
            step_id=start_step_id + recovery_idx - 1,
            generator="llm",
            output=output,
            c_raw=c_raw,
            c_norm=c_norm,
            c_smooth=None,
            state_before="RECOVERY",
            D_start=0,
            D_post=0,
            stable_anchor=None,
            action="RECOVERY_STEP",
            c_info=c_info,
            is_recovery=True,
        )
        record.state_after = "RECOVERY"
        record.extra["recovery_step"] = recovery_idx
        record.extra["ready_for_slm"] = bool(c_norm >= cfg.stable.theta_s)
        records.append(record)
        attempt_id += 1
        if c_norm >= cfg.stable.theta_s:
            return local_context, records, "SLM_READY", attempt_id
    return local_context, records, "EXHAUSTED_FORCE_SLM", attempt_id


def _append_final_answer(
    *,
    problem_id: str,
    state: GenerationState,
    step_logs: list[dict[str, Any]],
    slm,
    llm,
    cfg: SARRConfig,
) -> str | None:
    engine = llm if cfg.generation.final_answer_generator == "llm" else slm
    account = cfg.generation.final_answer_generator
    prefix = _final_answer_prefix(state.assistant_prefix_text)
    output = engine.generate_text(
        state.problem_text,
        prefix,
        max_new_tokens=cfg.generation.answer_token_budget,
        stop_delimiters=None,
        include_stop_str_in_output=False,
    )
    _account_generation_cost(
        state,
        account,
        wall_time=output.wall_time,
        token_count=output.token_count,
        prompt_tokens=output.prompt_tokens,
    )
    row = {
        "problem_id": problem_id,
        "step_id": state.step_count + 1,
        "generator": account,
        "text": output.text,
        "token_count": output.token_count,
        "finish_reason": output.finish_reason,
        "action": "FINAL_ANSWER",
        "is_final_answer": True,
    }
    step_logs.append(row)
    state.assistant_prefix_text = prefix + output.text
    state.step_count += 1
    state.trace.append(TraceEvent(state.step_count, "final_answer", row))
    return extract_answer_from_steps([{"step_text": output.text}], state.assistant_prefix_text)


def run_sarr_code(
    *,
    problem_id: str,
    problem_text: str,
    slm,
    llm,
    normalizer,
    cfg: SARRConfig,
) -> tuple[BPAResult, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    state = GenerationState(problem_text=problem_text, generation_protocol=cfg.method)
    start_time = time.time()

    active_records: list[StepRecord] = []
    all_records: list[StepRecord] = []
    rollback_events: list[RollbackEvent] = []
    transition_events: list[dict[str, Any]] = []
    step_logs: list[dict[str, Any]] = []

    monitor_state = "STARTUP"
    D_start = 0
    D_post = 0
    stable_anchor: int | None = None
    c_norm_history: list[float] = []
    c_smooth_history: list[float] = []
    force_next_step_slm = False
    pending_force_recovery_event_idx: int | None = None
    long_span_fallback_counts: dict[int, int] = {}
    startup_monitor_steps = 0
    attempt_id = 1
    stop_reason: str | None = None

    try:
        while state.phase != Phase.DONE:
            visible_tokens = _visible_think_tokens(active_records)
            if visible_tokens >= cfg.generation.think_token_budget:
                stop_reason = "think_token_budget"
                break

            remaining_think_tokens = cfg.generation.think_token_budget - visible_tokens
            state_before = monitor_state
            forced_after_recovery = force_next_step_slm
            force_next_step_slm = False
            output = _generate_slm_step(
                slm,
                state,
                cfg,
                remaining_think_tokens=remaining_think_tokens,
            )
            state.assistant_prefix_text += output.text
            step_id = len(active_records) + 1
            c_raw, c_norm, c_info = _confidence_for_prefix(
                slm,
                normalizer,
                cfg,
                state.problem_text,
                state.assistant_prefix_text,
            )
            c_norm_history.append(c_norm)
            c_smooth = smooth_confidence(c_norm_history, cfg.confidence.smooth_window)
            if c_smooth is not None:
                c_smooth_history.append(c_smooth)

            record = _make_record(
                problem_id=problem_id,
                attempt_id=attempt_id,
                step_id=step_id,
                generator="slm",
                output=output,
                c_raw=c_raw,
                c_norm=c_norm,
                c_smooth=c_smooth,
                state_before=state_before,
                D_start=D_start,
                D_post=D_post,
                stable_anchor=stable_anchor,
                action="TRUST",
                c_info=c_info,
            )
            if forced_after_recovery:
                record.extra["forced_after_recovery"] = True
            attempt_id += 1
            active_records.append(record)
            all_records.append(record)
            transition = _apply_transition_metadata(active_records)
            if transition is not None:
                transition_events.append({"problem_id": problem_id, **transition})

            startup_monitor_steps += 1
            state.step_count = len(active_records)

            thinking_stop = _strict_thinking_stop_reason(output, output.text)
            if c_smooth is None:
                record.state_after = monitor_state
                step_logs.append(_serialize_step(record))
                if thinking_stop is not None:
                    stop_reason = thinking_stop
                    break
                continue

            if c_smooth >= cfg.stable.theta_s:
                monitor_state = "STABLE"
                stable_anchor = step_id
                D_start = 0
                D_post = 0
                if pending_force_recovery_event_idx is not None:
                    rollback_events[pending_force_recovery_event_idx].force_slm_after_recovery_failed = False
                    pending_force_recovery_event_idx = None
                record.action = "REFRESH_STABLE_ANCHOR"
                record.state_after = monitor_state
                record.D_start = D_start
                record.D_post = D_post
                record.stable_anchor = stable_anchor
                step_logs.append(_serialize_step(record))
                if thinking_stop is not None:
                    stop_reason = thinking_stop
                    break
                continue

            v = 0
            if len(c_smooth_history) >= 2:
                v = code_style_degeneration_event(
                    c_smooth_history[-2],
                    c_smooth_history[-1],
                    cfg.confidence.delta,
                )
            record.degeneration_event = v

            should_rollback = False
            rollback_type = ""
            rollback_reason = ""
            anchor = 0

            if monitor_state == "STARTUP":
                D_start += v
                record.D_start = D_start
                if startup_monitor_steps >= cfg.startup.B_min and D_start >= cfg.startup.tau_start:
                    should_rollback = True
                    rollback_type = "STARTUP_ROLLBACK"
                    rollback_reason = "STARTUP_DEGENERATION"
                if startup_monitor_steps >= cfg.startup.B_max:
                    should_rollback = True
                    rollback_type = "STARTUP_ROLLBACK"
                    rollback_reason = "STARTUP_NOT_STABLE_WITHIN_BUDGET"
                if should_rollback:
                    if pending_force_recovery_event_idx is not None:
                        rollback_events[pending_force_recovery_event_idx].force_slm_after_recovery_failed = True
                        pending_force_recovery_event_idx = None
                    anchor = choose_best_prefix_anchor(active_records, allow_zero=True)
                    record.action = rollback_reason
            elif monitor_state == "STABLE":
                D_post += v
                record.D_post = D_post
                if D_post >= cfg.stable.tau_D:
                    monitor_state = "DEGENERATED"
                    should_rollback = True
                    rollback_type = "POST_STABLE_ROLLBACK"
                    rollback_reason = "POST_STABLE_DEGENERATION"
                    anchor = stable_anchor if stable_anchor is not None else 0
                    record.action = "POST_STABLE_ROLLBACK"

            record.state_after = monitor_state
            step_logs.append(_serialize_step(record))

            if thinking_stop is not None:
                stop_reason = thinking_stop
                break

            if not should_rollback:
                continue

            current_step_id = len(active_records)
            rollback_context, kept, removed, span = rollback_to_anchor(
                state.problem_text,
                active_records,
                anchor,
                current_step_id,
            )
            long_span = span > cfg.rollback.M_max
            long_span_fallback_count_before = long_span_fallback_counts.get(anchor, 0)
            allow_long_span_delete = False
            if long_span:
                if cfg.rollback.long_span_policy == "rollback_to_anchor":
                    allow_long_span_delete = True
                elif cfg.rollback.long_span_policy == "fallback_once_then_rollback":
                    allow_long_span_delete = (
                        long_span_fallback_count_before >= cfg.rollback.max_long_span_fallbacks_per_anchor
                    )
            fallback_no_delete = span <= 0 or (long_span and not allow_long_span_delete)
            if fallback_no_delete:
                if long_span:
                    long_span_fallback_counts[anchor] = long_span_fallback_count_before + 1
                rollback_context = state.assistant_prefix_text
                kept = list(active_records)
                removed = []
                max_recovery = 1
                recovery_start_step_id = len(kept) + 1
            else:
                _mark_removed(removed)
                if long_span:
                    max_recovery = min(span + 1, cfg.rollback.long_span_recovery_steps)
                else:
                    max_recovery = span + 1
                recovery_start_step_id = anchor + 1

            rec_context, rec_records, rec_stop_reason, attempt_id = confidence_gated_recovery(
                problem_id=problem_id,
                attempt_id_start=attempt_id,
                start_step_id=recovery_start_step_id,
                state=state,
                context=rollback_context,
                llm=llm,
                slm=slm,
                normalizer=normalizer,
                cfg=cfg,
                max_recovery_steps=max_recovery,
                remaining_think_tokens=max(1, cfg.generation.think_token_budget - _visible_think_tokens(kept)),
            )

            new_active_records = kept + rec_records
            for rec in rec_records:
                all_records.append(rec)
            for idx in range(max(1, len(kept)), len(new_active_records)):
                transition = _apply_transition_metadata(new_active_records[: idx + 1])
                if transition is not None:
                    transition_events.append({"problem_id": problem_id, **transition})

            event = RollbackEvent(
                problem_id=problem_id,
                type=rollback_type,
                reason=rollback_reason,
                trigger_step=current_step_id,
                anchor_step=anchor,
                rollback_span=span,
                removed_steps=_serialize_removed(removed),
                recovery_steps=[_serialize_step(r) for r in rec_records],
                stop_reason=rec_stop_reason,
                recovery_max_steps=max_recovery,
                recovery_actual_steps=len(rec_records),
                recovery_c_norm=[float(r.c_norm) for r in rec_records if r.c_norm is not None],
                fallback_no_delete=fallback_no_delete,
                long_span=long_span,
                long_span_policy=cfg.rollback.long_span_policy if long_span else None,
                long_span_fallback_count_before=long_span_fallback_count_before,
                long_span_recovery_limited=bool(long_span and not fallback_no_delete and max_recovery < span + 1),
                force_next_step_slm=cfg.rollback.force_slm_after_recovery,
            )
            rollback_events.append(event)
            if cfg.rollback.force_slm_after_recovery:
                pending_force_recovery_event_idx = len(rollback_events) - 1
            state.trace.append(TraceEvent(current_step_id, "rollback_event", asdict(event)))

            active_records = new_active_records
            state.assistant_prefix_text = rec_context
            state.step_count = len(active_records)

            for rec in rec_records:
                step_logs.append(_serialize_step(rec))

            monitor_state = "STARTUP"
            D_start = 0
            D_post = 0
            stable_anchor = None
            c_norm_history = []
            c_smooth_history = []
            startup_monitor_steps = 0
            force_next_step_slm = cfg.rollback.force_slm_after_recovery
            if force_next_step_slm:
                state.trace.append(
                    TraceEvent(state.step_count, "force_next_step_slm", {"enabled": True})
                )

            if has_close_think_tag(state.assistant_prefix_text):
                if pending_force_recovery_event_idx is not None:
                    rollback_events[pending_force_recovery_event_idx].force_slm_after_recovery_failed = False
                    pending_force_recovery_event_idx = None
                stop_reason = "finished"
                break

    except ContextBudgetExceeded as exc:
        state.phase = Phase.DONE
        stop_reason = "context_budget"
        state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    if stop_reason is None:
        stop_reason = "done"

    if pending_force_recovery_event_idx is not None:
        rollback_events[pending_force_recovery_event_idx].force_slm_after_recovery_failed = False
        pending_force_recovery_event_idx = None

    if stop_reason == "think_token_budget":
        should_force = cfg.generation.force_close_think_on_budget
        if should_force and not has_close_think_tag(state.assistant_prefix_text):
            state.assistant_prefix_text += cfg.generation.force_close_think_text
            state.trace.append(
                TraceEvent(
                    state.step_count,
                    "forced_close_think",
                    {
                        "reason": stop_reason,
                        "text": cfg.generation.force_close_think_text,
                    },
                )
            )
            stop_reason = f"{stop_reason}_forced_close_think"

    state.stop_reason = stop_reason

    answer = None
    if stop_reason != "context_budget":
        try:
            answer = _append_final_answer(
                problem_id=problem_id,
                state=state,
                step_logs=step_logs,
                slm=slm,
                llm=llm,
                cfg=cfg,
            )
        except ContextBudgetExceeded as exc:
            state.stop_reason = "context_budget_final_answer"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    state.phase = Phase.DONE
    state.trace.append(
        TraceEvent(
            state.step_count,
            "sarr_summary",
            {
                "num_active_steps": len(active_records),
                "num_generated_thinking_attempts": len(all_records),
                "num_rollbacks": len(rollback_events),
                "num_transition_events": len(transition_events),
                "stop_reason": state.stop_reason,
            },
        )
    )
    state.trace.append(TraceEvent(state.step_count, "step_logs", {"steps": step_logs}))

    result = BPAResult(
        answer=answer,
        state=state,
        total_wall_time=time.time() - start_time,
    )
    return (
        result,
        [_serialize_step(record) for record in all_records],
        [asdict(event) for event in rollback_events],
        transition_events,
    )
