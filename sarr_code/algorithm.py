from __future__ import annotations

import math
import re
import time
from dataclasses import asdict
from typing import Any

from bpa.context_budget import ContextBudgetExceeded
from bpa.safety import CLOSE_THINK_TAG, extract_answer_from_steps, has_close_think_tag
from bpa.state import GenerationState, Phase, TraceEvent
from bpa.trace import BPAResult

from .config import SARRConfig
from .records import StepOutput, StepRecord


# ---------------------------------------------------------------------------
# Segment-level CI-OD tracker
# ---------------------------------------------------------------------------

class SegmentCIODTracker:
    """Per-SLM-segment CI-OD v2 tracker.

    Receives pre-smoothed (c_raw, c_smooth) per step; does not manage
    its own smoothing buffer. Resets on open_new_segment().

    State equations:
        masked_uncertainty = (c_raw <= masked_low_threshold) AND (c_smooth > masked_low_threshold)
        masked_memory_t  = masked_decay * masked_memory_{t-1} + 1[masked_uncertainty]
        exp_inc = 1 if (masked_memory > 0 AND c_smooth >= exposure_threshold) else 0
        post_masked_exposure_t = exposure_decay * post_masked_exposure_{t-1} + exp_inc

    Risk:
        hazard = hazard_scale * (1 + masked_memory) * max(0, exposure - exposure_e0)^2
        ciod_risk = 1 - exp(-hazard)
    """

    def __init__(self, cfg: SARRConfig) -> None:
        self.cfg = cfg.ciod

        self.segment_id = 1
        self.masked_memory = 0.0
        self.post_masked_exposure = 0.0
        self.ciod_risk = 0.0
        self.segment_step_count = 0

        # Global summaries across all segments
        self.total_masked_uncertainty_count = 0
        self.max_segment_masked_memory = 0.0
        self.max_segment_post_masked_exposure = 0.0
        self.max_segment_ciod_risk = 0.0
        self.switch_to_llm_count = 0
        self.switch_to_slm_count = 0

    def _compute_risk(self) -> float:
        if self.masked_memory < self.cfg.min_masked_memory:
            return 0.0
        excess = max(0.0, self.post_masked_exposure - self.cfg.exposure_e0)
        if excess <= 0.0:
            return 0.0
        h = self.cfg.hazard_scale * (1.0 + self.masked_memory) * (excess ** 2.0)
        return float(1.0 - math.exp(-h))

    def update_slm_step(self, c_raw: float, c_smooth: float) -> None:
        """Update segment state with one SLM step's confidence values."""
        masked_uncertainty = (
            c_raw <= self.cfg.masked_low_threshold
            and c_smooth > self.cfg.masked_low_threshold
        )

        if masked_uncertainty:
            self.masked_memory = self.cfg.masked_decay * self.masked_memory + 1.0
            self.total_masked_uncertainty_count += 1
        else:
            self.masked_memory = self.cfg.masked_decay * self.masked_memory

        if self.masked_memory > 0.0:
            exp_inc = 1.0 if c_smooth >= self.cfg.exposure_threshold else 0.0
            self.post_masked_exposure = self.cfg.exposure_decay * self.post_masked_exposure + exp_inc
        else:
            self.post_masked_exposure = 0.0

        self.ciod_risk = self._compute_risk()
        self.segment_step_count += 1

        self.max_segment_masked_memory = max(self.max_segment_masked_memory, self.masked_memory)
        self.max_segment_post_masked_exposure = max(
            self.max_segment_post_masked_exposure, self.post_masked_exposure
        )
        self.max_segment_ciod_risk = max(self.max_segment_ciod_risk, self.ciod_risk)

    def close_segment(self) -> tuple[float, float]:
        """Close current SLM segment. Returns (masked_memory, post_masked_exposure) for probe init."""
        self.switch_to_llm_count += 1
        return self.masked_memory, self.post_masked_exposure

    def open_new_segment(self) -> None:
        """Start a new SLM segment with fresh per-segment state."""
        self.segment_id += 1
        self.masked_memory = 0.0
        self.post_masked_exposure = 0.0
        self.ciod_risk = 0.0
        self.segment_step_count = 0
        self.switch_to_slm_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "segment_step_count": self.segment_step_count,
            "masked_memory": self.masked_memory,
            "post_masked_exposure": self.post_masked_exposure,
            "ciod_risk": self.ciod_risk,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "total_masked_uncertainty_count": self.total_masked_uncertainty_count,
            "max_segment_masked_memory": self.max_segment_masked_memory,
            "max_segment_post_masked_exposure": self.max_segment_post_masked_exposure,
            "max_segment_ciod_risk": self.max_segment_ciod_risk,
            "switch_to_llm_count": self.switch_to_llm_count,
            "switch_to_slm_count": self.switch_to_slm_count,
        }


# ---------------------------------------------------------------------------
# SLM re-entry probe
# ---------------------------------------------------------------------------

class ReentryProbeState:
    """Carries CI-OD probe state during LLM_ACTIVE, initialised from the closed SLM segment.

    No smoothing history is available so masked_uncertainty cannot fire (single probe
    sample → smooth ≈ raw → masked_uncertainty = raw_low AND NOT raw_low = False).
    masked_memory therefore only decays during LLM steps, naturally converging toward zero.

    The half-life of masked_memory under decay d is log(0.5) / log(d) ≈ 34 steps for d=0.98.
    """

    def __init__(
        self,
        cfg: SARRConfig,
        *,
        saved_masked_memory: float,
        saved_post_masked_exposure: float,
    ) -> None:
        self.cfg = cfg.ciod
        self.masked_memory = saved_masked_memory
        self.post_masked_exposure = saved_post_masked_exposure

    def _compute_risk(self) -> float:
        if self.masked_memory < self.cfg.min_masked_memory:
            return 0.0
        excess = max(0.0, self.post_masked_exposure - self.cfg.exposure_e0)
        if excess <= 0.0:
            return 0.0
        h = self.cfg.hazard_scale * (1.0 + self.masked_memory) * (excess ** 2.0)
        return float(1.0 - math.exp(-h))

    def update(self, probe_c_raw: float) -> float:
        """Simulate one probe step. Returns slm_reentry_risk."""
        self.masked_memory = self.cfg.masked_decay * self.masked_memory

        if self.masked_memory > 0.0:
            exp_inc = 1.0 if probe_c_raw >= self.cfg.exposure_threshold else 0.0
            self.post_masked_exposure = self.cfg.exposure_decay * self.post_masked_exposure + exp_inc
        else:
            self.post_masked_exposure = 0.0

        return self._compute_risk()


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

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


def _visible_think_tokens(records: list[StepRecord]) -> int:
    return sum(r.token_count for r in records)


def _compute_c_smooth(
    c_raw: float,
    active_records: list[StepRecord],
    smooth_window: int,
) -> float:
    """Exponential moving average over last smooth_window SLM c_raw values (including current)."""
    vals = [
        float(r.c_raw)
        for r in active_records
        if r.generator == "slm" and r.active and r.c_raw is not None
    ]
    vals = vals[-(smooth_window - 1):] + [c_raw]
    return sum(vals) / len(vals)


def choose_best_prefix_anchor(step_records: list[StepRecord], *, allow_zero: bool = True) -> int:
    """Return the step_id with the highest c_smooth (or c_raw) as the best anchor."""
    candidates: list[tuple[int, float]] = []
    if allow_zero:
        candidates.append((0, 0.0))
    for record in step_records:
        value = record.c_smooth
        if value is None:
            value = record.c_raw
        if value is not None:
            candidates.append((int(record.step_id), float(value)))
    if not candidates:
        return 0
    return max(candidates, key=lambda item: (item[1], -item[0]))[0]


def _apply_transition_metadata(records: list[StepRecord]) -> dict[str, Any] | None:
    if len(records) < 2:
        return None
    prev = records[-2]
    curr = records[-1]
    transition_type = f"{prev.generator}->{curr.generator}"
    curr.transition_type = transition_type
    delta: float | None = None
    if prev.c_smooth is not None and curr.c_smooth is not None:
        delta = float(curr.c_smooth - prev.c_smooth)
    curr.delta_c_smooth = delta
    return {
        "prev_step_id": prev.step_id,
        "curr_step_id": curr.step_id,
        "prev_generator": prev.generator,
        "curr_generator": curr.generator,
        "transition_type": transition_type,
        "delta_c_smooth": delta,
    }


def _make_record(
    *,
    problem_id: str,
    attempt_id: int,
    step_id: int,
    generator: str,
    output: StepOutput,
    c_raw: float | None,
    c_smooth: float | None,
    driver_state: str,
    action: str,
    c_info: dict[str, Any] | None = None,
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
        c_smooth=c_smooth,
        driver_state=driver_state,
        action=action,
        finish_reason=output.finish_reason,
        prompt_tokens=output.prompt_tokens,
        wall_time=output.wall_time,
        attempt_id=attempt_id,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Step generation
# ---------------------------------------------------------------------------

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
        capture_token_entropy=False,
        topk_entropy=cfg.confidence.top_k,
        close_tag_lookahead=CLOSE_THINK_TAG,
        close_tag_lookahead_tokens=cfg.generation.close_tag_lookahead_tokens,
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
        token_count=_actual_token_count(output),
        prompt_tokens=output.prompt_tokens,
    )
    return output


def _confidence_for_prefix(
    slm,
    cfg: SARRConfig,
    problem_text: str,
    assistant_prefix_text: str,
) -> tuple[float, dict[str, Any]]:
    c_raw, c_info = slm.continuation_confidence(
        problem_text,
        assistant_prefix_text,
        topk=cfg.confidence.top_k,
    )
    return float(c_raw), c_info


def _append_final_answer(
    *,
    problem_id: str,
    state: GenerationState,
    step_logs: list[dict[str, Any]],
    slm,
    llm,
    cfg: SARRConfig,
    ciod_snapshot: dict[str, Any] | None = None,
) -> str | None:
    account = cfg.generation.final_answer_generator
    if account == "active":
        last_generator = next(
            (
                str(row.get("generator"))
                for row in reversed(step_logs)
                if row.get("generator") in {"slm", "llm"} and not row.get("is_final_answer")
            ),
            "slm",
        )
        account = last_generator
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
        "step_id": state.step_count + 1,
        "generator": account,
        "text": output.text,
        "token_count": output.token_count,
        "finish_reason": output.finish_reason,
        "action": "FINAL_ANSWER",
        "is_final_answer": True,
        "extra": {"ciod": ciod_snapshot or {}},
    }
    step_logs.append(row)
    state.assistant_prefix_text = prefix + output.text
    state.step_count += 1
    state.trace.append(TraceEvent(state.step_count, "final_answer", row))
    return extract_answer_from_steps([{"step_text": output.text}], state.assistant_prefix_text)


# ---------------------------------------------------------------------------
# Stagnation utilities (kept for offline analysis; not called by main loop)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

    active_records: list[StepRecord] = []
    all_records: list[StepRecord] = []
    driver_switch_events: list[dict[str, Any]] = []
    transition_events: list[dict[str, Any]] = []
    step_logs: list[dict[str, Any]] = []

    driver_state = cfg.controller.initial_driver.upper() + "_ACTIVE"  # "SLM_ACTIVE"
    segment_tracker = SegmentCIODTracker(cfg)
    reentry_probe: ReentryProbeState | None = None

    clean_autonomy_anchor: int | None = None
    attempt_id = 1
    stop_reason: str | None = None

    try:
        while state.phase != Phase.DONE:
            visible_tokens = _visible_think_tokens(active_records)
            if visible_tokens >= cfg.generation.think_token_budget:
                stop_reason = "think_token_budget"
                break

            remaining = cfg.generation.think_token_budget - visible_tokens
            driver_state_before = driver_state
            step_id = len(active_records) + 1

            # ------------------------------------------------------------------
            # SLM_ACTIVE
            # ------------------------------------------------------------------
            if driver_state == "SLM_ACTIVE":
                output = _generate_slm_step(slm, state, cfg, remaining_think_tokens=remaining)
                state.assistant_prefix_text += output.text

                thinking_stop = _strict_thinking_stop_reason(output, output.text)
                if thinking_stop is not None:
                    record = _make_record(
                        problem_id=problem_id,
                        attempt_id=attempt_id,
                        step_id=step_id,
                        generator="slm",
                        output=output,
                        c_raw=None,
                        c_smooth=None,
                        driver_state=driver_state,
                        action="FINISHED" if thinking_stop == "finished" else f"STOP_{thinking_stop.upper()}",
                    )
                    record.extra.update(
                        {
                            "driver_state_before": driver_state_before,
                            "confidence_skipped": True,
                            "confidence_skipped_reason": thinking_stop,
                            **segment_tracker.snapshot(),
                        }
                    )
                    attempt_id += 1
                    active_records.append(record)
                    all_records.append(record)
                    _apply_transition_metadata(active_records)
                    state.step_count = len(active_records)
                    step_logs.append(_serialize_step(record))
                    stop_reason = thinking_stop
                    break

                # Confidence computation
                c_raw, c_info = _confidence_for_prefix(slm, cfg, state.problem_text, state.assistant_prefix_text)
                c_smooth = _compute_c_smooth(c_raw, active_records, cfg.confidence.smooth_window)

                # Segment CI-OD update
                segment_tracker.update_slm_step(c_raw, c_smooth)

                # Anchor refresh
                anchor_refresh = driver_state == "SLM_ACTIVE"
                if anchor_refresh:
                    clean_autonomy_anchor = step_id

                # Driver switch check
                action = "REFRESH_ANCHOR" if anchor_refresh else "CONTINUE"
                switch_reason: str | None = None

                if segment_tracker.ciod_risk >= cfg.ciod.on_threshold:
                    saved_mm, saved_pe = segment_tracker.close_segment()
                    reentry_probe = ReentryProbeState(
                        cfg,
                        saved_masked_memory=saved_mm,
                        saved_post_masked_exposure=saved_pe,
                    )
                    driver_state = "LLM_ACTIVE"
                    action = "SWITCH_TO_LLM_BY_CIOD"
                    switch_reason = "CIOD_RISK_THRESHOLD"
                    switch_event: dict[str, Any] = {
                        "event": "driver_switch",
                        "from": "SLM_ACTIVE",
                        "to": "LLM_ACTIVE",
                        "reason": switch_reason,
                        "step_id": step_id,
                        "ciod_risk": segment_tracker.ciod_risk,
                        "segment_id": segment_tracker.segment_id,
                        "masked_memory": saved_mm,
                        "post_masked_exposure": saved_pe,
                    }
                    driver_switch_events.append({"problem_id": problem_id, **switch_event})
                    state.trace.append(TraceEvent(step_id, "driver_switch", switch_event))

                record = _make_record(
                    problem_id=problem_id,
                    attempt_id=attempt_id,
                    step_id=step_id,
                    generator="slm",
                    output=output,
                    c_raw=c_raw,
                    c_smooth=c_smooth,
                    driver_state=driver_state,
                    action=action,
                    c_info=c_info,
                )
                record.extra.update(
                    {
                        "driver_state_before": driver_state_before,
                        "switch_reason": switch_reason,
                        "clean_autonomy_anchor": clean_autonomy_anchor,
                        "slm_reentry_risk": None,
                        **segment_tracker.snapshot(),
                    }
                )

                attempt_id += 1
                active_records.append(record)
                all_records.append(record)
                trans = _apply_transition_metadata(active_records)
                if trans is not None:
                    transition_events.append({"problem_id": problem_id, **trans})
                state.step_count = len(active_records)
                step_logs.append(_serialize_step(record))

            # ------------------------------------------------------------------
            # LLM_ACTIVE
            # ------------------------------------------------------------------
            elif driver_state == "LLM_ACTIVE":
                assert reentry_probe is not None

                output = _generate_llm_step(
                    llm,
                    state,
                    cfg,
                    assistant_prefix_text=state.assistant_prefix_text,
                    remaining_think_tokens=remaining,
                )
                state.assistant_prefix_text += output.text

                thinking_stop = _strict_thinking_stop_reason(output, output.text)

                # SLM re-entry probe
                probe_c_raw, _ = _confidence_for_prefix(slm, cfg, state.problem_text, state.assistant_prefix_text)
                slm_reentry_risk = reentry_probe.update(probe_c_raw)

                if thinking_stop is not None:
                    action = "FINISHED" if thinking_stop == "finished" else f"STOP_{thinking_stop.upper()}"
                    switch_reason = None
                elif slm_reentry_risk <= cfg.ciod.off_threshold:
                    driver_state = "SLM_ACTIVE"
                    segment_tracker.open_new_segment()
                    reentry_probe = None
                    action = "SWITCH_TO_SLM_BY_REENTRY_RISK"
                    switch_reason = "REENTRY_RISK_BELOW_THRESHOLD"
                    back_event: dict[str, Any] = {
                        "event": "driver_switch",
                        "from": "LLM_ACTIVE",
                        "to": "SLM_ACTIVE",
                        "reason": switch_reason,
                        "step_id": step_id,
                        "slm_reentry_risk": slm_reentry_risk,
                        "new_segment_id": segment_tracker.segment_id,
                    }
                    driver_switch_events.append({"problem_id": problem_id, **back_event})
                    state.trace.append(TraceEvent(step_id, "driver_switch", back_event))
                else:
                    action = "KEEP_LLM_BY_REENTRY_RISK"
                    switch_reason = "REENTRY_RISK_ABOVE_THRESHOLD"

                record = _make_record(
                    problem_id=problem_id,
                    attempt_id=attempt_id,
                    step_id=step_id,
                    generator="llm",
                    output=output,
                    c_raw=None,
                    c_smooth=None,
                    driver_state=driver_state,
                    action=action,
                )
                record.extra.update(
                    {
                        "driver_state_before": driver_state_before,
                        "switch_reason": switch_reason,
                        "probe_c_raw": probe_c_raw,
                        "slm_reentry_risk": slm_reentry_risk,
                        "probe_masked_memory": reentry_probe.masked_memory if reentry_probe else None,
                        "probe_post_masked_exposure": reentry_probe.post_masked_exposure if reentry_probe else None,
                        **segment_tracker.snapshot(),
                    }
                )

                attempt_id += 1
                active_records.append(record)
                all_records.append(record)
                trans = _apply_transition_metadata(active_records)
                if trans is not None:
                    transition_events.append({"problem_id": problem_id, **trans})
                state.step_count = len(active_records)
                step_logs.append(_serialize_step(record))

                if thinking_stop is not None:
                    stop_reason = thinking_stop
                    break

    except ContextBudgetExceeded as exc:
        state.phase = Phase.DONE
        stop_reason = "context_budget"
        state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    if stop_reason is None:
        stop_reason = "done"

    if stop_reason == "think_token_budget":
        if cfg.generation.force_close_think_on_budget and not has_close_think_tag(state.assistant_prefix_text):
            state.assistant_prefix_text += cfg.generation.force_close_think_text
            state.trace.append(
                TraceEvent(
                    state.step_count,
                    "forced_close_think",
                    {"reason": stop_reason, "text": cfg.generation.force_close_think_text},
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
                ciod_snapshot=segment_tracker.snapshot(),
            )
        except ContextBudgetExceeded as exc:
            state.stop_reason = "context_budget_final_answer"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    state.phase = Phase.DONE

    slm_decode = state.slm_decode_tokens
    llm_decode = state.llm_decode_tokens
    slm_prefill = state.slm_prefill_tokens
    llm_prefill = state.llm_prefill_tokens
    total_decode = slm_decode + llm_decode
    total_tokens = total_decode + slm_prefill + llm_prefill

    seg_summary = segment_tracker.summary()
    state.trace.append(
        TraceEvent(
            state.step_count,
            "sarr_summary",
            {
                "num_active_steps": len(active_records),
                "num_generated_attempts": len(all_records),
                "num_driver_switch_events": len(driver_switch_events),
                "stop_reason": state.stop_reason,
                # Per-generator token counts
                "slm_decode_tokens": slm_decode,
                "llm_decode_tokens": llm_decode,
                "slm_prefill_tokens": slm_prefill,
                "llm_prefill_tokens": llm_prefill,
                "slm_generate_calls": state.slm_generate_calls,
                "llm_generate_calls": state.llm_full_calls,
                # Driver switching
                "switch_to_llm_count": seg_summary["switch_to_llm_count"],
                "switch_to_slm_count": seg_summary["switch_to_slm_count"],
                # Token share
                "llm_decode_share": llm_decode / max(1, total_decode),
                "llm_total_token_share": (llm_decode + llm_prefill) / max(1, total_tokens),
                # Segment CI-OD
                "max_segment_ciod_risk": seg_summary["max_segment_ciod_risk"],
                "max_segment_masked_memory": seg_summary["max_segment_masked_memory"],
                "max_segment_post_masked_exposure": seg_summary["max_segment_post_masked_exposure"],
                "total_masked_uncertainty_count": seg_summary["total_masked_uncertainty_count"],
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
        driver_switch_events,
        transition_events,
    )
