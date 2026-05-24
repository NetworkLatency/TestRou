from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from bpa.context_budget import ContextBudgetExceeded
from bpa.safety import CLOSE_THINK_TAG, extract_answer_from_final_step, has_close_think_tag
from bpa.state import GenerationState, Phase, TraceEvent
from bpa.trace import BPAResult

from .config import SARRConfig
from .controller import (
    ACTIVE,
    CLOSE_OR_FINALIZE,
    HANDOFF_PROBE,
    LLM_FORWARD_OWNERSHIP,
    LLM_REPAIR_OWNERSHIP,
    SLM_ACTIVE,
    SOURCE_LLM,
    SOURCE_SLM,
    HandoffReadiness,
    ObservableSignals,
    OwnershipController,
    RiskDetector,
    SealedIntervalLock,
    StableStepMemory,
    StepSignals,
    TrajectoryState,
)
from .records import StepOutput, StepRecord


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


def _ensure_step_terminator(step_text: str, finish_reason: str, delimiters: list[str]) -> str:
    if finish_reason == "eos" or CLOSE_THINK_TAG in step_text or not delimiters:
        return step_text
    primary = delimiters[0]
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


def _text_step_count(text: str, delimiters: list[str]) -> int:
    if not str(text or "").strip():
        return 0
    primary = next((delimiter for delimiter in delimiters if delimiter), "\n\n")
    count = str(text).count(primary)
    if count > 0:
        return count
    return 1


def _output_step_count(output: StepOutput, cfg: SARRConfig) -> int:
    return _text_step_count(output.text, cfg.generation.step_delimiters)


def _removed_text_step_count(records: list[StepRecord], cfg: SARRConfig) -> int:
    return max(1, sum(_text_step_count(record.text, cfg.generation.step_delimiters) for record in records))


def _final_answer_prefix(thinking_text: str) -> str:
    thinking = thinking_text.split(CLOSE_THINK_TAG, 1)[0] if CLOSE_THINK_TAG in thinking_text else thinking_text
    return f"{thinking.rstrip()}\n{CLOSE_THINK_TAG}\n\n"


def _resolve_final_answer_generator(
    *,
    cfg: SARRConfig,
    trajectory: TrajectoryState,
    controller: OwnershipController,
) -> str:
    account = cfg.generation.final_answer_generator
    if account != "active":
        return account
    last_active = next((record for record in reversed(trajectory.records) if record.status == ACTIVE), None)
    if last_active is not None and last_active.source == SOURCE_LLM:
        return "llm"
    if controller.driver_state in {LLM_FORWARD_OWNERSHIP, LLM_REPAIR_OWNERSHIP}:
        return "llm"
    return "slm"


def _append_final_answer(
    *,
    problem_id: str,
    state: GenerationState,
    step_logs: list[dict[str, Any]],
    slm,
    llm,
    cfg: SARRConfig,
    trajectory: TrajectoryState,
    controller: OwnershipController,
    fallback_answer: str | None,
) -> tuple[str | None, str]:
    account = _resolve_final_answer_generator(cfg=cfg, trajectory=trajectory, controller=controller)
    engine = llm if account == "llm" else slm
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
    row: dict[str, Any] = {
        "problem_id": problem_id,
        "step_id": trajectory.next_step_id,
        "source": account.upper(),
        "generator": account,
        "status": "final_answer",
        "text": output.text,
        "token_count": output.token_count,
        "finish_reason": output.finish_reason,
        "action": "FINAL_ANSWER",
        "is_final_answer": True,
    }
    step_logs.append(row)
    state.assistant_prefix_text = prefix + output.text
    state.step_count = trajectory.next_step_id
    state.trace.append(TraceEvent(state.step_count, "final_answer", row))
    answer = extract_answer_from_final_step(output.text) or fallback_answer
    return answer, account


def _serialize_step(record: StepRecord) -> dict[str, Any]:
    row = asdict(record)
    row["token_count"] = record.token_count
    row["generator"] = record.source.lower()
    row["active"] = record.status == ACTIVE
    return row


def _apply_transition_metadata(records: list[StepRecord]) -> dict[str, Any] | None:
    active_records = [record for record in records if record.status == ACTIVE]
    if len(active_records) < 2:
        return None
    prev = active_records[-2]
    curr = active_records[-1]
    transition_type = f"{prev.source}->{curr.source}"
    curr.transition_type = transition_type
    return {
        "prev_step_id": prev.step_id,
        "curr_step_id": curr.step_id,
        "prev_generator": prev.source.lower(),
        "curr_generator": curr.source.lower(),
        "transition_type": transition_type,
    }


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
        stop_delimiters=_stop_strings_for_slm(cfg),
        capture_token_entropy=True,
        topk_entropy=cfg.confidence.top_k,
        immediate_stop_strings=[CLOSE_THINK_TAG],
        min_stop_tokens=min(cfg.generation.min_new_tokens_per_step, max_new_tokens),
    )
    output.text = _ensure_step_terminator(output.text, output.finish_reason, cfg.generation.step_delimiters)
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
    max_new_tokens = max(1, min(cfg.generation.max_new_tokens_per_step, remaining_think_tokens))
    output = llm.generate_step(
        state.problem_text,
        state.assistant_prefix_text,
        max_new_tokens=max_new_tokens,
        stop_delimiters=_stop_strings_for_llm(cfg),
    )
    output.text = _ensure_step_terminator(output.text, output.finish_reason, cfg.generation.step_delimiters)
    _account_generation_cost(
        state,
        "llm",
        wall_time=output.wall_time,
        token_count=_actual_token_count(output),
        prompt_tokens=output.prompt_tokens,
    )
    return output


def _sync_prefix(state: GenerationState, trajectory: TrajectoryState) -> None:
    state.assistant_prefix_text = trajectory.get_active_prefix()
    state.step_count = len(trajectory.active_steps())


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


def _record_step(
    *,
    trajectory: TrajectoryState,
    output: StepOutput,
    source: str,
    controller: OwnershipController,
    cfg: SARRConfig,
    signals: StepSignals,
    action: str,
    attempt_id: int,
    step_logs: list[dict[str, Any]],
    transition_events: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> StepRecord:
    payload = dict(extra or {})
    payload.setdefault("text_step_count", _output_step_count(output, cfg))
    record = trajectory.append_active_step(
        output=output,
        source=source,
        driver_state=controller.driver_state,
        observed_signals=signals,
        action=action,
        attempt_id=attempt_id,
        extra=payload,
    )
    trans = _apply_transition_metadata(trajectory.records)
    if trans is not None:
        transition_events.append({"problem_id": trajectory.problem_id, **trans})
    step_logs.append(_serialize_step(record))
    return record


def _replace_step_log(step_logs: list[dict[str, Any]], record: StepRecord) -> None:
    serialized = _serialize_step(record)
    for idx in range(len(step_logs) - 1, -1, -1):
        if step_logs[idx].get("step_id") == record.step_id:
            step_logs[idx] = serialized
            return
    step_logs.append(serialized)


def _append_probe_discard(
    *,
    trajectory: TrajectoryState,
    output: StepOutput,
    controller: OwnershipController,
    cfg: SARRConfig,
    signals: StepSignals,
    attempt_id: int,
    reason: str,
    step_logs: list[dict[str, Any]],
) -> StepRecord:
    record = trajectory.append_probe_discarded(
        output=output,
        source=SOURCE_SLM,
        driver_state=HANDOFF_PROBE,
        observed_signals=signals,
        action="HANDOFF_PROBE_DISCARDED",
        attempt_id=attempt_id,
        reason=reason,
        extra={"text_step_count": _output_step_count(output, cfg)},
    )
    step_logs.append(_serialize_step(record))
    return record


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

    trajectory = TrajectoryState(problem_id)
    signals_engine = ObservableSignals(cfg.risk)
    stable_memory = StableStepMemory(cfg.risk)
    risk_detector = RiskDetector(cfg.risk)
    handoff_readiness = HandoffReadiness(cfg.risk)
    sealed_lock = SealedIntervalLock()
    controller = OwnershipController(problem_id, cfg.risk, cfg.controller.initial_driver)

    step_logs: list[dict[str, Any]] = []
    transition_events: list[dict[str, Any]] = []
    attempt_id = 1
    stop_reason: str | None = None

    try:
        while state.phase != Phase.DONE:
            visible_tokens = trajectory.visible_token_count()
            if visible_tokens >= cfg.generation.think_token_budget:
                stop_reason = "think_token_budget"
                controller.switch(CLOSE_OR_FINALIZE, step_id=None, reason=stop_reason)
                break

            remaining = cfg.generation.think_token_budget - visible_tokens

            if controller.driver_state == SLM_ACTIVE:
                output = _generate_slm_step(slm, state, cfg, remaining_think_tokens=remaining)
                signals = signals_engine.compute(
                    text=output.text,
                    source=SOURCE_SLM,
                    output_extra=output.extra,
                    trajectory=trajectory,
                    known_candidates=[],
                )
                record = _record_step(
                    trajectory=trajectory,
                    output=output,
                    source=SOURCE_SLM,
                    controller=controller,
                    cfg=cfg,
                    signals=signals,
                    action="SLM_CONTINUE",
                    attempt_id=attempt_id,
                    step_logs=step_logs,
                    transition_events=transition_events,
                )
                attempt_id += 1
                _sync_prefix(state, trajectory)

                thinking_stop = _strict_thinking_stop_reason(output, output.text)

                if thinking_stop is not None:
                    record.action = "FINISHED" if thinking_stop == "finished" else f"STOP_{thinking_stop.upper()}"
                    _replace_step_log(step_logs, record)
                    stop_reason = thinking_stop
                    controller.switch(CLOSE_OR_FINALIZE, step_id=record.step_id, reason=stop_reason)
                    break

                loop = risk_detector.degenerative_loop(trajectory)
                if loop.triggered:
                    controller.degenerative_loop_count += 1
                    controller.note_failure_step(signals, step_id=record.step_id, reason=loop.reason)
                    controller.note_event(
                        "degenerative_loop_detected",
                        step_id=record.step_id,
                        reason=loop.reason,
                        data=loop.to_dict(),
                    )
                    controller.switch(
                        LLM_FORWARD_OWNERSHIP,
                        step_id=record.step_id,
                        reason="degenerative_loop",
                        data=loop.to_dict(),
                    )
                    continue

                contamination = risk_detector.prefix_contamination(trajectory, stable_memory)
                if contamination.triggered:
                    controller.prefix_contamination_count += 1
                    controller.note_failure_step(signals, step_id=record.step_id, reason=contamination.reason)
                    onset = contamination.onset_step_id or record.step_id
                    if sealed_lock.blocks(onset, record.step_id):
                        controller.repeated_rollback_blocked_count += 1
                        controller.note_event(
                            "repeated_rollback_blocked",
                            step_id=record.step_id,
                            reason=contamination.reason,
                            data=contamination.to_dict(),
                        )
                        controller.switch(
                            LLM_FORWARD_OWNERSHIP,
                            step_id=record.step_id,
                            reason="sealed_interval_blocks_repeated_rollback",
                            data=contamination.to_dict(),
                        )
                        continue

                    anchor = stable_memory.latest_anchor_before(onset)
                    if anchor is not None:
                        removed = trajectory.rollback_to_step(anchor)
                        if removed:
                            repair_horizon = _removed_text_step_count(removed, cfg)
                            controller.rollback_count += 1
                            stable_memory.remove_after(anchor)
                            removed_start = min(item.step_id for item in removed)
                            removed_end = max(item.step_id for item in removed)
                            interval = trajectory.seal_interval(
                                anchor_step_id=anchor,
                                removed_start_step_id=removed_start,
                                removed_end_step_id=removed_end,
                                reason=contamination.reason,
                                repair_horizon=repair_horizon,
                                removed_steps=removed,
                            )
                            sealed_lock.add(interval)
                            for removed_record in removed:
                                _replace_step_log(step_logs, removed_record)
                            _sync_prefix(state, trajectory)
                            controller.start_repair(repair_horizon=repair_horizon)
                            controller.note_event(
                                "prefix_contamination_rollback",
                                step_id=record.step_id,
                                reason=contamination.reason,
                                data={
                                    **contamination.to_dict(),
                                    "anchor_step_id": anchor,
                                    "removed_start_step_id": removed_start,
                                    "removed_end_step_id": removed_end,
                                    "repair_horizon": repair_horizon,
                                    "removed_record_count": len(removed),
                                },
                            )
                            controller.switch(
                                LLM_REPAIR_OWNERSHIP,
                                step_id=record.step_id,
                                reason="prefix_contamination_repair",
                                data=contamination.to_dict(),
                            )
                            continue

                    controller.note_event(
                        "prefix_contamination_no_anchor",
                        step_id=record.step_id,
                        reason=contamination.reason,
                        data=contamination.to_dict(),
                    )
                    controller.switch(
                        LLM_FORWARD_OWNERSHIP,
                        step_id=record.step_id,
                        reason="prefix_contamination_without_stable_anchor",
                        data=contamination.to_dict(),
                    )
                    continue

                if cfg.risk.enable_local_difficulty_routing:
                    local = risk_detector.local_difficulty(record, signals, stable_memory)
                    if local.triggered:
                        controller.local_difficulty_count += 1
                        controller.note_failure_step(signals, step_id=record.step_id, reason=local.reason)
                        controller.note_event(
                            "local_difficulty_detected",
                            step_id=record.step_id,
                            reason=local.reason,
                            data=local.to_dict(),
                        )
                        controller.switch(
                            LLM_FORWARD_OWNERSHIP,
                            step_id=record.step_id,
                            reason="local_difficulty",
                            data=local.to_dict(),
                        )
                        continue

                stable_memory.add_if_stable(record, signals)
                continue

            if controller.driver_state in {LLM_FORWARD_OWNERSHIP, LLM_REPAIR_OWNERSHIP}:
                ownership_state = controller.driver_state
                output = _generate_llm_step(llm, state, cfg, remaining_think_tokens=remaining)
                signals = signals_engine.compute(
                    text=output.text,
                    source=SOURCE_LLM,
                    output_extra=output.extra,
                    trajectory=trajectory,
                    known_candidates=[],
                )
                record = _record_step(
                    trajectory=trajectory,
                    output=output,
                    source=SOURCE_LLM,
                    controller=controller,
                    cfg=cfg,
                    signals=signals,
                    action="LLM_FORWARD_CONTINUE" if ownership_state == LLM_FORWARD_OWNERSHIP else "LLM_REPAIR_CONTINUE",
                    attempt_id=attempt_id,
                    step_logs=step_logs,
                    transition_events=transition_events,
                )
                attempt_id += 1
                _sync_prefix(state, trajectory)

                thinking_stop = _strict_thinking_stop_reason(output, output.text)

                if thinking_stop is not None:
                    record.action = "FINISHED" if thinking_stop == "finished" else f"STOP_{thinking_stop.upper()}"
                    _replace_step_log(step_logs, record)
                    stop_reason = thinking_stop
                    controller.switch(CLOSE_OR_FINALIZE, step_id=record.step_id, reason=stop_reason)
                    break

                output_step_count = _output_step_count(output, cfg)
                llm_quality = controller.note_llm_ownership_step(
                    signals,
                    step_count=output_step_count,
                    stable_memory=stable_memory,
                )

                if ownership_state == LLM_REPAIR_OWNERSHIP:
                    controller.note_repair_step(output_step_count)
                    if not controller.repair_horizon_satisfied():
                        continue

                handoff_eligibility = controller.llm_ready_for_handoff_probe(ownership_state, stable_memory)
                if not handoff_eligibility.triggered:
                    controller.note_handoff_deferred()
                    controller.note_event(
                        "llm_handoff_probe_deferred",
                        step_id=record.step_id,
                        reason=handoff_eligibility.reason,
                        data={
                            **handoff_eligibility.to_dict(),
                            "ownership_state": ownership_state,
                            "repair_horizon": controller.repair_horizon,
                            "repair_generated_steps": controller.repair_generated_steps,
                            "llm_quality": llm_quality,
                        },
                    )
                    continue

                controller.previous_ownership_state = ownership_state
                controller.switch(
                    HANDOFF_PROBE,
                    step_id=record.step_id,
                    reason="llm_continuation_ready_for_slm_probe",
                    data={
                        **handoff_eligibility.to_dict(),
                        "ownership_state": ownership_state,
                        "repair_horizon": controller.repair_horizon,
                        "repair_generated_steps": controller.repair_generated_steps,
                        "llm_quality": llm_quality,
                    },
                )
                continue

            if controller.driver_state == HANDOFF_PROBE:
                controller.handoff_probe_count += 1
                controller.handoff_probe_forward_count += 1
                output = _generate_slm_step(slm, state, cfg, remaining_think_tokens=remaining)
                signals = signals_engine.compute(
                    text=output.text,
                    source=SOURCE_SLM,
                    output_extra=output.extra,
                    trajectory=trajectory,
                    known_candidates=[],
                )
                readiness = handoff_readiness.evaluate(
                    probe_text=output.text,
                    probe_signals=signals,
                    stable_memory=stable_memory,
                    sealed_lock=sealed_lock,
                    failure_signals=controller.failure_signals(),
                    llm_episode_signals=controller.llm_episode_signals(),
                    rejected_probe_signals=controller.rejected_probe_signals(),
                )
                if readiness.triggered:
                    controller.handoff_success_count += 1
                    record = _record_step(
                        trajectory=trajectory,
                        output=output,
                        source=SOURCE_SLM,
                        controller=controller,
                        cfg=cfg,
                        signals=signals,
                        action="HANDOFF_PROBE_ACCEPTED",
                        attempt_id=attempt_id,
                        step_logs=step_logs,
                        transition_events=transition_events,
                        extra={"handoff_readiness": readiness.to_dict()},
                    )
                    attempt_id += 1
                    _sync_prefix(state, trajectory)
                    thinking_stop = _strict_thinking_stop_reason(output, output.text)
                    if thinking_stop is not None:
                        stop_reason = thinking_stop
                        controller.switch(CLOSE_OR_FINALIZE, step_id=record.step_id, reason=stop_reason)
                        break
                    stable_memory.add_if_stable(record, signals)
                    controller.switch(SLM_ACTIVE, step_id=record.step_id, reason="handoff_probe_passed")
                    continue

                controller.handoff_failure_count += 1
                controller.note_rejected_probe(signals)
                _append_probe_discard(
                    trajectory=trajectory,
                    output=output,
                    controller=controller,
                    cfg=cfg,
                    signals=signals,
                    attempt_id=attempt_id,
                    reason=readiness.reason,
                    step_logs=step_logs,
                )
                attempt_id += 1
                controller.note_event(
                    "handoff_probe_failed",
                    step_id=trajectory.next_step_id - 1,
                    reason=readiness.reason,
                    data=readiness.to_dict(),
                )
                fallback_state = controller.previous_ownership_state or LLM_FORWARD_OWNERSHIP
                controller.switch(fallback_state, step_id=trajectory.next_step_id - 1, reason="handoff_probe_failed")
                continue

            if controller.driver_state == CLOSE_OR_FINALIZE:
                stop_reason = stop_reason or "done"
                break

            raise RuntimeError(f"Unknown controller state: {controller.driver_state}")

    except ContextBudgetExceeded as exc:
        state.phase = Phase.DONE
        stop_reason = "context_budget"
        controller.switch(CLOSE_OR_FINALIZE, step_id=state.step_count, reason=stop_reason)
        state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    if stop_reason is None:
        stop_reason = "done"

    if stop_reason != "context_budget":
        should_close = (
            stop_reason in {
                "think_token_budget",
                "done",
                "empty_step",
            }
            or not has_close_think_tag(state.assistant_prefix_text)
        )
        if should_close:
            stop_reason = _force_close_if_needed(state, cfg, stop_reason)

    state.stop_reason = stop_reason

    answer = None
    final_answer_generator = None
    if stop_reason != "context_budget":
        try:
            answer, final_answer_generator = _append_final_answer(
                problem_id=problem_id,
                state=state,
                step_logs=step_logs,
                slm=slm,
                llm=llm,
                cfg=cfg,
                trajectory=trajectory,
                controller=controller,
                fallback_answer=None,
            )
        except ContextBudgetExceeded as exc:
            state.stop_reason = "context_budget_final_answer"
            stop_reason = state.stop_reason
            controller.switch(CLOSE_OR_FINALIZE, step_id=state.step_count, reason=stop_reason)
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    state.phase = Phase.DONE
    total_wall_time = time.time() - start_time

    summary = controller.summary(
        problem_id=problem_id,
        finish_reason=state.stop_reason or stop_reason,
        final_answer=answer,
        final_answer_generator=final_answer_generator,
        trajectory=trajectory,
        total_wall_time=total_wall_time,
        slm_wall_time=state.slm_wall_time,
        llm_wall_time=state.llm_generation_wall_time + state.llm_scoring_wall_time,
        slm_prefill_count=state.slm_generate_calls,
        llm_prefill_count=state.llm_full_calls,
    )
    state.trace.append(TraceEvent(state.step_count, "sarr_summary", summary))
    state.trace.append(TraceEvent(state.step_count, "step_logs", {"steps": step_logs}))
    state.trace.append(TraceEvent(state.step_count, "controller_events", {"events": controller.events}))

    result = BPAResult(
        answer=answer,
        state=state,
        total_wall_time=total_wall_time,
    )
    return (
        result,
        [_serialize_step(record) for record in trajectory.records],
        controller.events,
        transition_events + controller.transition_events,
    )
