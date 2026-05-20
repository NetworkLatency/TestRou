from __future__ import annotations

import time
import re
from dataclasses import asdict
from typing import Any

from bpa.context_budget import ContextBudgetExceeded
from bpa.safety import CLOSE_THINK_TAG, extract_answer_from_steps, has_close_think_tag
from bpa.state import GenerationState, Phase, TraceEvent
from bpa.trace import BPAResult

from .calibration import code_style_degeneration_event
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
            "readiness_raw": r.readiness_raw,
            "readiness_raw_smooth": r.readiness_raw_smooth,
            "readiness_source": r.readiness_source,
            "calibration_enabled": r.calibration_enabled,
            "readiness": r.readiness,
            "stagnation_score": r.stagnation_score,
            "hcs_suspect": r.hcs_suspect,
        }
        for r in records
    ]


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = _word_tokens(text)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def word_ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    grams_a = _ngram_set(text_a, n)
    grams_b = _ngram_set(text_b, n)
    if not grams_a or not grams_b:
        return 0.0
    return float(len(grams_a & grams_b) / len(grams_a | grams_b))


def _block_ending_at(records: list[StepRecord], end_idx: int, *, min_tokens: int, max_steps: int) -> list[StepRecord]:
    block: list[StepRecord] = []
    token_count = 0
    idx = end_idx
    while idx >= 0 and len(block) < max_steps:
        record = records[idx]
        if record.generator != "slm" or record.is_recovery:
            break
        block.insert(0, record)
        token_count += len(_word_tokens(record.text))
        if token_count >= min_tokens:
            break
        idx -= 1
    return block


def _block_text(records: list[StepRecord]) -> str:
    return "".join(record.text for record in records)


def surface_stagnation_score(active_records: list[StepRecord], cfg: SARRConfig) -> float:
    if not cfg.stagnation.enabled or not active_records:
        return 0.0
    current = active_records[-1]
    if current.generator != "slm" or current.is_recovery:
        return 0.0

    current_block = _block_ending_at(
        active_records,
        len(active_records) - 1,
        min_tokens=cfg.stagnation.block_min_tokens,
        max_steps=cfg.stagnation.block_max_steps,
    )
    if not current_block:
        return 0.0
    current_text = _block_text(current_block)
    current_first_step = current_block[0].step_id
    min_step = current.step_id - cfg.stagnation.repeat_window
    best = 0.0
    for idx in range(len(active_records) - 2, -1, -1):
        candidate_end = active_records[idx]
        if candidate_end.step_id < min_step:
            break
        if candidate_end.step_id >= current_first_step:
            continue
        if candidate_end.generator != "slm" or candidate_end.is_recovery:
            continue
        candidate_block = _block_ending_at(
            active_records,
            idx,
            min_tokens=cfg.stagnation.block_min_tokens,
            max_steps=cfg.stagnation.block_max_steps,
        )
        if not candidate_block:
            continue
        best = max(best, word_ngram_jaccard(current_text, _block_text(candidate_block), cfg.stagnation.ngram_n))
    return best


def update_raw_readiness(step: StepRecord, recent_steps: list[StepRecord], cfg: SARRConfig) -> None:
    vals = [
        float(record.c_raw)
        for record in recent_steps
        if record.generator == "slm" and record.active and record.c_raw is not None
    ]
    vals = vals[-cfg.readiness.smooth_window :]
    raw = float(step.c_raw) if step.c_raw is not None else None
    step.readiness_raw = raw
    step.readiness_raw_smooth = (sum(vals) / len(vals)) if vals else raw
    step.readiness_source = "raw"
    step.calibration_enabled = False


def get_readiness(step: StepRecord, cfg: SARRConfig) -> float | None:
    if cfg.readiness.normalization != "raw" or cfg.readiness.use_calibration:
        raise ValueError("This experiment disables calibration; only raw readiness is allowed.")
    if cfg.readiness.smooth_window > 1:
        return step.readiness_raw_smooth
    return step.readiness_raw


def _latest_clean_autonomy_anchor(records: list[StepRecord]) -> int | None:
    for record in reversed(records):
        if (
            record.active
            and record.generator == "slm"
            and not record.is_recovery
            and record.readiness_high
            and not record.hcs_suspect
            and not record.removed_by_rollback
        ):
            return record.step_id
    return None


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
        value = record.readiness_raw_smooth
        if value is None:
            value = record.readiness
        if value is None:
            value = record.c_raw
        if value is not None:
            candidates.append((int(record.step_id), float(value)))
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
    max_new_tokens_override: int | None = None,
) -> StepOutput:
    configured_limit = cfg.generation.max_new_tokens_per_step
    if max_new_tokens_override is not None:
        configured_limit = min(configured_limit, max_new_tokens_override)
    max_new_tokens = max(1, min(configured_limit, remaining_think_tokens))
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
    if cfg.calibration.enabled or cfg.calibration.use_percentile or cfg.readiness.use_calibration:
        raise ValueError("This experiment disables calibration; only raw readiness is allowed.")
    return c_raw, None, c_info


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
    max_tokens_per_step: int | None = None,
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
            max_new_tokens_override=max_tokens_per_step,
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
        record.readiness_raw = c_raw
        record.readiness_raw_smooth = c_raw
        record.readiness_source = "raw"
        record.calibration_enabled = False
        record.readiness = get_readiness(record, cfg)
        record.readiness_high = bool(
            record.readiness is not None and record.readiness >= cfg.readiness.high_threshold
        )
        record.readiness_low = bool(
            record.readiness is not None and record.readiness <= cfg.readiness.low_threshold
        )
        record.state_after = "RECOVERY"
        record.extra["recovery_step"] = recovery_idx
        record.extra["ready_for_slm"] = record.readiness_high
        records.append(record)
        attempt_id += 1
        if record.readiness_high:
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
    D_suspect = 0
    stable_anchor: int | None = None
    suspect_anchor: int | None = None
    suspect_start_step: int | None = None
    suspect_steps = 0
    suspect_max_readiness: float | None = None
    clean_autonomy_anchor: int | None = None
    hcs_suspect_run = 0
    hcs_rollback_count = 0
    low_confidence_run = 0
    readiness_history: list[float] = []
    force_next_step_slm = False
    pending_force_recovery_event_idx: int | None = None
    long_span_fallback_counts: dict[int, int] = {}
    rollback_anchor_counts: dict[int, int] = {}
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
            record = _make_record(
                problem_id=problem_id,
                attempt_id=attempt_id,
                step_id=step_id,
                generator="slm",
                output=output,
                c_raw=c_raw,
                c_norm=c_norm,
                c_smooth=None,
                state_before=state_before,
                D_start=D_start,
                D_post=D_post,
                stable_anchor=stable_anchor,
                action="TRUST",
                c_info=c_info,
            )
            update_raw_readiness(record, active_records + [record], cfg)
            readiness = get_readiness(record, cfg)
            record.readiness = readiness
            if readiness is not None:
                readiness_history.append(readiness)
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
            should_rollback = False
            rollback_type = ""
            rollback_reason = ""
            anchor = 0
            is_hcs_rollback = False

            record.readiness_high = bool(readiness is not None and readiness >= cfg.readiness.high_threshold)
            record.readiness_low = bool(readiness is not None and readiness <= cfg.readiness.low_threshold)
            record.stagnation_score = surface_stagnation_score(active_records, cfg)
            record.stagnation_high = bool(
                cfg.stagnation.enabled and record.stagnation_score >= cfg.stagnation.high_threshold
            )
            hcs_detection_enabled = bool(
                cfg.hcs.enabled
                and cfg.stagnation.enabled
                and (cfg.startup_guard.hcs_enabled or monitor_state != "STARTUP")
            )
            record.hcs_suspect = bool(hcs_detection_enabled and record.readiness_high and record.stagnation_high)
            hcs_suspect_run = hcs_suspect_run + 1 if record.hcs_suspect else 0
            record.hcs_suspect_run = hcs_suspect_run
            hcs_can_confirm = hcs_suspect_run >= cfg.hcs.suspect_patience
            if cfg.hcs.enable_after_clean_anchor and clean_autonomy_anchor is None:
                hcs_can_confirm = False
            if cfg.startup_guard.enable_hcs_after_clean_anchor and clean_autonomy_anchor is None:
                hcs_can_confirm = False
            record.hcs_confirmed = bool(record.hcs_suspect and hcs_can_confirm)

            refresh_has_stable_confidence = record.readiness_high
            record.anchor_refresh_allowed = bool(
                record.readiness_high
                and not record.hcs_suspect
                and refresh_has_stable_confidence
                and not record.is_recovery
            )
            if record.anchor_refresh_allowed:
                clean_autonomy_anchor = step_id
                record.anchor_refresh_blocked_reason = None
            elif record.hcs_suspect:
                record.anchor_refresh_blocked_reason = "HCS_SUSPECT"
            elif not record.readiness_high:
                record.anchor_refresh_blocked_reason = "READINESS_NOT_HIGH"
            elif not refresh_has_stable_confidence:
                record.anchor_refresh_blocked_reason = "NOT_STABLE_PREFIX"
            else:
                record.anchor_refresh_blocked_reason = "IN_RECOVERY"
            record.clean_autonomy_anchor = clean_autonomy_anchor
            if record.hcs_confirmed:
                record.autonomy_state = "HCS_CONFIRMED"
            elif record.hcs_suspect:
                record.autonomy_state = "HCS_SUSPECT"
            else:
                record.autonomy_state = "NORMAL"

            if record.hcs_confirmed:
                if hcs_rollback_count >= cfg.hcs.max_hcs_rollbacks_per_problem:
                    monitor_state = "UNRECOVERABLE_HCS"
                    record.action = "UNRECOVERABLE_HCS"
                    record.autonomy_state = "UNRECOVERABLE_HCS"
                    record.state_after = monitor_state
                    record.extra["stop_reason"] = "HCS_ROLLBACK_LIMIT"
                    record.extra["hcs_rollback_count"] = hcs_rollback_count
                    record.extra["max_hcs_rollbacks_per_problem"] = cfg.hcs.max_hcs_rollbacks_per_problem
                    step_logs.append(_serialize_step(record))
                    state.trace.append(
                        TraceEvent(
                            step_id,
                            "unrecoverable_hcs",
                            {
                                "trigger_step": step_id,
                                "clean_anchor_step": clean_autonomy_anchor,
                                "hcs_rollback_count": hcs_rollback_count,
                                "max_hcs_rollbacks_per_problem": cfg.hcs.max_hcs_rollbacks_per_problem,
                            },
                        )
                    )
                    stop_reason = "HCS_ROLLBACK_LIMIT"
                    break
                if clean_autonomy_anchor is not None:
                    should_rollback = True
                    is_hcs_rollback = True
                    rollback_type = "HCS_ROLLBACK"
                    rollback_reason = "HCS_CONFIRMED_RAW_READINESS"
                    anchor = clean_autonomy_anchor
                    monitor_state = "HCS_CONFIRMED"
                    record.action = "HCS_CONFIRMED_ROLLBACK"
                    record.state_after = monitor_state
                    record.extra["clean_anchor_step"] = clean_autonomy_anchor
                    record.extra["llm_recovery_prompt_type"] = cfg.hcs_recovery.prompt_type
                    record.extra["mention_stagnation"] = cfg.hcs_recovery.mention_stagnation

            if not should_rollback and record.readiness_high:
                monitor_state = "STABLE"
                if not record.hcs_suspect:
                    stable_anchor = step_id
                D_start = 0
                D_post = 0
                D_suspect = 0
                low_confidence_run = 0
                suspect_anchor = None
                suspect_start_step = None
                suspect_steps = 0
                suspect_max_readiness = None
                if pending_force_recovery_event_idx is not None and not record.hcs_suspect:
                    rollback_events[pending_force_recovery_event_idx].force_slm_after_recovery_failed = False
                    pending_force_recovery_event_idx = None
                if record.hcs_suspect:
                    record.action = "HCS_SUSPECT"
                else:
                    record.action = "SUSPECT_RECOVERED" if state_before == "SUSPECT" else "REFRESH_STABLE_ANCHOR"
                record.state_after = monitor_state
                record.D_start = D_start
                record.D_post = D_post
                record.stable_anchor = stable_anchor
                if state_before == "SUSPECT" and not record.hcs_suspect:
                    record.extra["suspect_recovered"] = True
                step_logs.append(_serialize_step(record))
                if thinking_stop is not None:
                    stop_reason = thinking_stop
                    break
                continue

            v = 0
            if len(readiness_history) >= 2:
                v = code_style_degeneration_event(
                    readiness_history[-2],
                    readiness_history[-1],
                    cfg.confidence.delta,
                )
            record.degeneration_event = v

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
                low_confidence_signal = bool(record.readiness_low or v)
                if low_confidence_signal:
                    low_confidence_run += 1
                else:
                    low_confidence_run = 0
                record.extra["low_confidence_run"] = low_confidence_run
                record.extra["useful_exploration_grace_blocks"] = cfg.low_confidence.useful_exploration_grace_blocks
                record.extra["collapse_patience_blocks"] = cfg.low_confidence.collapse_patience_blocks
                if (
                    low_confidence_signal
                    and low_confidence_run <= cfg.low_confidence.useful_exploration_grace_blocks
                ):
                    record.action = "USEFUL_EXPLORATION"
                    record.autonomy_state = "USEFUL_EXPLORATION"
                elif low_confidence_signal and low_confidence_run < cfg.low_confidence.collapse_patience_blocks:
                    record.action = "LOW_CONFIDENCE_OBSERVE"
                    record.autonomy_state = "LOW_CONFIDENCE_OBSERVE"
                elif low_confidence_signal and D_post >= cfg.stable.tau_D:
                    if cfg.rollback.post_stable_intervention_policy == "suspect_confirmed_rollback":
                        monitor_state = "SUSPECT"
                        suspect_anchor = stable_anchor if stable_anchor is not None else 0
                        suspect_start_step = step_id
                        suspect_steps = 0
                        D_suspect = 0
                        suspect_max_readiness = readiness
                        record.action = "ENTER_SUSPECT"
                        record.extra["suspect_anchor"] = suspect_anchor
                        record.extra["suspect_start_step"] = suspect_start_step
                    else:
                        monitor_state = "DEGENERATED"
                        should_rollback = True
                        rollback_type = "POST_STABLE_ROLLBACK"
                        rollback_reason = "POST_STABLE_DEGENERATION"
                        anchor = stable_anchor if stable_anchor is not None else 0
                        record.action = "POST_STABLE_ROLLBACK"
            elif monitor_state == "SUSPECT":
                suspect_steps += 1
                D_suspect += v
                suspect_max_readiness = (
                    readiness
                    if suspect_max_readiness is None
                    else max(float(suspect_max_readiness), float(readiness or 0.0))
                )
                record.extra["suspect_anchor"] = suspect_anchor
                record.extra["suspect_start_step"] = suspect_start_step
                record.extra["suspect_steps"] = suspect_steps
                record.extra["D_suspect"] = D_suspect
                confirmed_by_trend = (
                    suspect_steps >= cfg.rollback.suspect_confirm_steps
                    and D_suspect >= cfg.rollback.tau_confirm
                )
                confirmed_by_timeout = (
                    suspect_steps >= cfg.rollback.suspect_max_steps
                    and (
                        suspect_max_readiness is None
                        or suspect_max_readiness < cfg.readiness.high_threshold
                    )
                )
                if confirmed_by_trend or confirmed_by_timeout:
                    monitor_state = "DEGENERATED"
                    should_rollback = True
                    rollback_type = "POST_STABLE_ROLLBACK"
                    rollback_reason = (
                        "POST_STABLE_CONFIRMED_DEGENERATION"
                        if confirmed_by_trend
                        else "POST_STABLE_SUSPECT_TIMEOUT"
                    )
                    anchor = suspect_anchor if suspect_anchor is not None else 0
                    record.action = rollback_reason
                    record.extra["confirmed_by_trend"] = confirmed_by_trend
                    record.extra["confirmed_by_timeout"] = confirmed_by_timeout
                else:
                    record.action = "SUSPECT_OBSERVE"

            record.state_after = monitor_state
            step_logs.append(_serialize_step(record))

            if thinking_stop is not None:
                stop_reason = thinking_stop
                break

            if not should_rollback:
                continue

            current_step_id = len(active_records)
            requested_anchor = anchor
            anchor_repeat_count_before = rollback_anchor_counts.get(requested_anchor, 0)
            if pending_force_recovery_event_idx is not None:
                rollback_events[pending_force_recovery_event_idx].force_slm_after_recovery_failed = True
                pending_force_recovery_event_idx = None

            if (
                not is_hcs_rollback
                and requested_anchor == 0
                and cfg.rollback.root_rollback_action == "force_close_think"
                and anchor_repeat_count_before >= cfg.rollback.max_root_rollbacks
            ):
                record.action = "ROOT_ROLLBACK_FORCE_CLOSE_THINK"
                record.extra["root_rollback_count_before"] = anchor_repeat_count_before
                if step_logs:
                    step_logs[-1] = _serialize_step(record)
                state.trace.append(
                    TraceEvent(
                        current_step_id,
                        "root_rollback_limit",
                        {
                            "requested_anchor_step": requested_anchor,
                            "root_rollback_count_before": anchor_repeat_count_before,
                            "max_root_rollbacks": cfg.rollback.max_root_rollbacks,
                        },
                    )
                )
                stop_reason = "root_rollback_limit"
                break

            anchor_backoff_steps = 0
            if (
                not is_hcs_rollback
                and anchor_repeat_count_before > 0
                and cfg.rollback.anchor_repeat_policy == "suppress"
            ):
                rollback_anchor_counts[requested_anchor] = anchor_repeat_count_before + 1
                record.action = f"SUPPRESS_REPEATED_{rollback_type}"
                record.extra["suppressed_rollback"] = True
                record.extra["requested_anchor_step"] = requested_anchor
                record.extra["anchor_repeat_count_before"] = anchor_repeat_count_before
                monitor_state = "STABLE"
                stable_anchor = current_step_id
                D_start = 0
                D_post = 0
                D_suspect = 0
                suspect_anchor = None
                suspect_start_step = None
                suspect_steps = 0
                suspect_max_readiness = None
                readiness_history = []
                startup_monitor_steps = 0
                record.state_after = monitor_state
                record.stable_anchor = stable_anchor
                record.D_start = D_start
                record.D_post = D_post
                if step_logs:
                    step_logs[-1] = _serialize_step(record)
                state.trace.append(
                    TraceEvent(
                        current_step_id,
                        "suppressed_repeated_rollback",
                        {
                            "type": rollback_type,
                            "requested_anchor_step": requested_anchor,
                            "anchor_repeat_count_before": anchor_repeat_count_before,
                        },
                    )
                )
                continue

            if (
                not is_hcs_rollback
                and cfg.rollback.anchor_repeat_policy == "backoff"
                and anchor_repeat_count_before >= cfg.rollback.anchor_repeat_backoff_after
            ):
                repeats_after_threshold = anchor_repeat_count_before - cfg.rollback.anchor_repeat_backoff_after + 1
                anchor_backoff_steps = repeats_after_threshold * cfg.rollback.anchor_repeat_backoff_steps
                anchor = max(0, requested_anchor - anchor_backoff_steps)

            long_span_fallback_count_before = long_span_fallback_counts.get(requested_anchor, 0)
            rollback_context, kept, removed, span = rollback_to_anchor(
                state.problem_text,
                active_records,
                anchor,
                current_step_id,
            )
            long_span = span > cfg.rollback.M_max
            allow_long_span_delete = False
            if is_hcs_rollback:
                allow_long_span_delete = True
            elif long_span:
                if cfg.rollback.long_span_policy == "rollback_to_anchor":
                    allow_long_span_delete = True
                elif cfg.rollback.long_span_policy == "fallback_once_then_rollback":
                    allow_long_span_delete = (
                        long_span_fallback_count_before >= cfg.rollback.max_long_span_fallbacks_per_anchor
                    )
            fallback_no_delete = span <= 0 or (long_span and not allow_long_span_delete)
            if fallback_no_delete:
                if long_span:
                    long_span_fallback_counts[requested_anchor] = long_span_fallback_count_before + 1
                rollback_context = state.assistant_prefix_text
                kept = list(active_records)
                removed = []
                max_recovery = 1
                recovery_start_step_id = len(kept) + 1
            else:
                _mark_removed(removed)
                if is_hcs_rollback:
                    max_recovery = cfg.hcs_recovery.max_llm_steps
                elif long_span:
                    max_recovery = min(span + 1, cfg.rollback.long_span_recovery_steps)
                else:
                    max_recovery = span + 1
                recovery_start_step_id = anchor + 1

            if is_hcs_rollback:
                hcs_rollback_count += 1

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
                max_tokens_per_step=cfg.hcs_recovery.max_tokens_per_step if is_hcs_rollback else None,
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
                requested_anchor_step=requested_anchor,
                anchor_repeat_count_before=anchor_repeat_count_before,
                anchor_backoff_steps=anchor_backoff_steps,
                suspect_start_step=suspect_start_step,
                suspect_steps=suspect_steps,
                D_suspect=D_suspect,
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
                force_next_step_slm=(
                    cfg.hcs_recovery.return_to_slm_after_recovery
                    if is_hcs_rollback
                    else cfg.rollback.force_slm_after_recovery
                ),
                event="hcs_rollback" if is_hcs_rollback else None,
                clean_anchor_step=anchor if is_hcs_rollback else None,
                hcs_rollback_count=hcs_rollback_count if is_hcs_rollback else 0,
                readiness_source="raw" if is_hcs_rollback else None,
                calibration_enabled=False if is_hcs_rollback else None,
                llm_recovery_prompt_type=cfg.hcs_recovery.prompt_type if is_hcs_rollback else None,
                mention_stagnation=cfg.hcs_recovery.mention_stagnation if is_hcs_rollback else None,
                return_to_slm=cfg.hcs_recovery.return_to_slm_after_recovery if is_hcs_rollback else None,
            )
            rollback_events.append(event)
            if not is_hcs_rollback:
                rollback_anchor_counts[requested_anchor] = anchor_repeat_count_before + 1
            return_to_slm_after_recovery = (
                cfg.hcs_recovery.return_to_slm_after_recovery
                if is_hcs_rollback
                else cfg.rollback.force_slm_after_recovery
            )
            if return_to_slm_after_recovery:
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
            D_suspect = 0
            stable_anchor = None
            suspect_anchor = None
            suspect_start_step = None
            suspect_steps = 0
            suspect_max_readiness = None
            readiness_history = []
            startup_monitor_steps = 0
            if rec_stop_reason == "SLM_READY" and rec_records:
                monitor_state = "STABLE"
                stable_anchor = rec_records[-1].step_id
                ready_readiness = rec_records[-1].readiness
                readiness_history = [float(ready_readiness)] if ready_readiness is not None else []
            clean_autonomy_anchor = _latest_clean_autonomy_anchor(active_records)
            hcs_suspect_run = 0
            low_confidence_run = 0
            force_next_step_slm = return_to_slm_after_recovery
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

    if stop_reason in {"think_token_budget", "root_rollback_limit", "HCS_ROLLBACK_LIMIT"}:
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
