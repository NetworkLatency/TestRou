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

from .calibration import code_style_degeneration_event
from .config import SARRConfig
from .records import RollbackEvent, StepOutput, StepRecord


RECOVERY_COMPLETE_TO_SLM = {
    "SLM_READY",
    "EXHAUSTED_FORCE_SLM",
    "RECOVERY_BUDGET_EXCEEDED",
    "ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM",
}


class ConfidenceProcessTracker:
    def __init__(self, cfg: SARRConfig) -> None:
        self.cfg = cfg.confidence_process
        self.raw_low_count = 0
        self.smooth_low_count = 0
        self.masked_uncertainty_count = 0

        # Legacy CI-OD v1: consecutive high-readiness run after masked uncertainty.
        self.high_run_length = 0
        self.high_run_start_step: int | None = None
        self.masked_memory_at_high_run_start = 0
        self.max_high_run_length = 0
        self.max_ciod_risk_v1 = 0.0
        self.ciod_shadow_trigger_count_v1 = 0
        self.first_ciod_shadow_trigger_step_v1: int | None = None
        self._last_ciod_risk_v1 = 0.0
        self._last_ciod_shadow_trigger_v1 = False

        # CI-OD v2: post-masked confidence exposure hazard.
        self.masked_memory = 0.0
        self.last_masked_step: int | None = None
        self.steps_since_last_masked: int | None = None
        self.post_masked_exposure = 0.0
        self.post_masked_high_count = 0
        self.post_masked_mid_high_count = 0
        self.max_post_masked_exposure = 0.0
        self.max_ciod_risk_v2 = 0.0
        self.ciod_shadow_trigger_count_v2 = 0
        self.first_ciod_shadow_trigger_step_v2: int | None = None
        self._last_ciod_risk_v2 = 0.0
        self._last_ciod_shadow_trigger_v2 = False
        self.ciod_episode_active = False
        self.ciod_episode_id = 0
        self.ciod_event_count = 0
        self.first_ciod_event_step: int | None = None
        self.last_ciod_event_step: int | None = None
        self.ciod_cooldown_until_step: int | None = None
        self.last_ciod_event_masked_memory = 0.0
        self.last_ciod_event_exposure = 0.0
        self.ciod_active_lease_count = 0
        self.ciod_event_steps: list[int] = []
        self.first_readiness_low_step: int | None = None
        self._last_ciod_event_shadow = False
        self._last_new_masked_mass_since_event = 0.0
        self._last_new_exposure_since_event = 0.0

        self._grid_configs = [
            {
                "key": self._grid_key(i, grid),
                "exposure_e0": float(grid["exposure_e0"]),
                "lambda0": float(grid["lambda0"]),
                "risk_threshold": float(grid["risk_threshold"]),
            }
            for i, grid in enumerate(self.cfg.ciod_grid)
        ]
        self._last_ciod_grid_risks = {grid["key"]: 0.0 for grid in self._grid_configs}
        self._last_ciod_grid_triggers = {grid["key"]: False for grid in self._grid_configs}
        self._ciod_grid_stats = {
            grid["key"]: {
                "exposure_e0": grid["exposure_e0"],
                "lambda0": grid["lambda0"],
                "risk_threshold": grid["risk_threshold"],
                "max_risk": 0.0,
                "trigger_count": 0,
                "first_trigger_step": None,
            }
            for grid in self._grid_configs
        }

    @property
    def masked_uncertainty_gap(self) -> int:
        return self.raw_low_count - self.smooth_low_count

    @staticmethod
    def _compact_float(value: float) -> str:
        return f"{value:g}".replace(".", "p").replace("-", "m")

    def _grid_key(self, index: int, grid: dict[str, float]) -> str:
        e0 = float(grid["exposure_e0"])
        lambda0 = float(grid["lambda0"])
        threshold = float(grid["risk_threshold"])
        return (
            f"g{index}_e0_{self._compact_float(e0)}"
            f"_lambda0_{self._compact_float(lambda0)}"
            f"_threshold_{self._compact_float(threshold)}"
        )

    def _ciod_risk_v1(self) -> float:
        dwell = max(0.0, float(self.high_run_length - self.cfg.r0))
        if dwell <= 0.0:
            return 0.0
        hazard = (
            self.cfg.v1_lambda0
            * ((1.0 + float(self.masked_memory_at_high_run_start)) ** self.cfg.alpha)
            * (dwell ** self.cfg.power)
        )
        return float(1.0 - math.exp(-hazard))

    def _ciod_risk_v2(self, *, exposure_e0: float, lambda0: float) -> float:
        if self.masked_memory < self.cfg.min_masked_memory:
            return 0.0
        exposure_excess = max(0.0, self.post_masked_exposure - exposure_e0)
        if exposure_excess <= 0.0:
            return 0.0
        cumulative_hazard = (
            lambda0
            * ((1.0 + self.masked_memory) ** self.cfg.alpha)
            * (exposure_excess ** self.cfg.power)
        )
        return float(1.0 - math.exp(-cumulative_hazard))

    def _confidence_process_snapshot(
        self,
        *,
        scored: bool,
        raw_low: bool,
        smooth_low: bool,
        masked_uncertainty: bool,
    ) -> dict[str, Any]:
        return {
            "scored": scored,
            "raw_low": raw_low,
            "smooth_low": smooth_low,
            "masked_uncertainty": masked_uncertainty,
            "raw_low_count": self.raw_low_count,
            "smooth_low_count": self.smooth_low_count,
            "masked_uncertainty_count": self.masked_uncertainty_count,
            "masked_uncertainty_gap": self.masked_uncertainty_gap,
            "high_run_length": self.high_run_length,
            "high_run_start_step": self.high_run_start_step,
            "masked_memory_at_high_run_start": self.masked_memory_at_high_run_start,
            "ciod_risk": self._last_ciod_risk_v1,
            "ciod_shadow_trigger": self._last_ciod_shadow_trigger_v1,
            "ciod_risk_v1": self._last_ciod_risk_v1,
            "ciod_shadow_trigger_v1": self._last_ciod_shadow_trigger_v1,
            "masked_memory": self.masked_memory,
            "last_masked_step": self.last_masked_step,
            "steps_since_last_masked": self.steps_since_last_masked,
            "post_masked_exposure": self.post_masked_exposure,
            "post_masked_high_count": self.post_masked_high_count,
            "post_masked_mid_high_count": self.post_masked_mid_high_count,
            "ciod_risk_v2": self._last_ciod_risk_v2,
            "ciod_shadow_trigger_v2": self._last_ciod_shadow_trigger_v2,
            "ciod_grid_risks": dict(self._last_ciod_grid_risks),
            "ciod_grid_triggers": dict(self._last_ciod_grid_triggers),
            "ciod_event_shadow": self._last_ciod_event_shadow,
            "ciod_episode_active": self.ciod_episode_active,
            "ciod_episode_id": self.ciod_episode_id,
            "ciod_event_count": self.ciod_event_count,
            "first_ciod_event_step": self.first_ciod_event_step,
            "last_ciod_event_step": self.last_ciod_event_step,
            "ciod_cooldown_until_step": self.ciod_cooldown_until_step,
            "new_masked_mass_since_last_ciod_event": self._last_new_masked_mass_since_event,
            "new_exposure_since_last_ciod_event": self._last_new_exposure_since_event,
            "last_ciod_event_masked_memory": self.last_ciod_event_masked_memory,
            "last_ciod_event_exposure": self.last_ciod_event_exposure,
            "ciod_active_lease_count": self.ciod_active_lease_count,
        }

    def _update_grid(self, step_id: int) -> None:
        risks: dict[str, float] = {}
        triggers: dict[str, bool] = {}
        for grid in self._grid_configs:
            key = str(grid["key"])
            risk = self._ciod_risk_v2(
                exposure_e0=float(grid["exposure_e0"]),
                lambda0=float(grid["lambda0"]),
            )
            trigger = risk >= float(grid["risk_threshold"])
            risks[key] = risk
            triggers[key] = trigger

            stats = self._ciod_grid_stats[key]
            stats["max_risk"] = max(float(stats["max_risk"]), risk)
            if trigger:
                stats["trigger_count"] = int(stats["trigger_count"]) + 1
                if stats["first_trigger_step"] is None:
                    stats["first_trigger_step"] = step_id

        self._last_ciod_grid_risks = risks
        self._last_ciod_grid_triggers = triggers

    def _grid_summary(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._ciod_grid_stats.items()}

    def _record_ciod_event(self, step_id: int) -> None:
        self._last_ciod_event_shadow = True
        self.ciod_event_count += 1
        self.last_ciod_event_step = step_id
        if self.first_ciod_event_step is None:
            self.first_ciod_event_step = step_id
        self.ciod_event_steps.append(step_id)
        self.last_ciod_event_masked_memory = self.masked_memory
        self.last_ciod_event_exposure = self.post_masked_exposure
        self.ciod_cooldown_until_step = step_id + self.cfg.ciod_event_cooldown_steps

    def _update_ciod_event_controller(self, record: StepRecord) -> None:
        self._last_ciod_event_shadow = False
        self._last_new_masked_mass_since_event = self.masked_memory - self.last_ciod_event_masked_memory
        self._last_new_exposure_since_event = self.post_masked_exposure - self.last_ciod_event_exposure

        if self.first_readiness_low_step is None and record.readiness_low:
            self.first_readiness_low_step = record.step_id

        if not self.ciod_episode_active:
            if self._last_ciod_risk_v2 >= self.cfg.ciod_event_on_threshold:
                self.ciod_episode_active = True
                self.ciod_episode_id += 1
                self._record_ciod_event(record.step_id)
            return

        if self._last_ciod_risk_v2 <= self.cfg.ciod_event_off_threshold:
            self.ciod_episode_active = False
            return

        cooldown_ready = (
            self.ciod_cooldown_until_step is None
            or record.step_id >= self.ciod_cooldown_until_step
        )
        enough_new_mass = (
            self._last_new_masked_mass_since_event >= self.cfg.min_new_masked_mass_for_retrigger
        )
        enough_new_exposure = (
            self._last_new_exposure_since_event >= self.cfg.min_new_exposure_for_retrigger
        )
        if (
            cooldown_ready
            and self._last_ciod_risk_v2 >= self.cfg.ciod_event_on_threshold
            and (enough_new_mass or enough_new_exposure)
        ):
            self._record_ciod_event(record.step_id)

    def ciod_active_route_available(self) -> bool:
        return self.ciod_active_lease_count < self.cfg.max_ciod_active_leases_per_problem

    def mark_ciod_active_lease(self) -> int:
        self.ciod_active_lease_count += 1
        return self.ciod_active_lease_count

    def observe(self, record: StepRecord) -> dict[str, Any]:
        scored = (
            record.generator == "slm"
            and record.active
            and record.c_raw is not None
            and record.readiness_value is not None
        )
        raw_low = False
        smooth_low = False
        masked_uncertainty = False
        if not scored:
            self._last_ciod_event_shadow = False
        if scored:
            c_raw = float(record.c_raw)
            readiness_value = float(record.readiness_value)
            raw_low = c_raw <= self.cfg.raw_low_threshold
            smooth_low = readiness_value <= self.cfg.smooth_low_threshold
            masked_uncertainty = raw_low and not smooth_low
            masked_before_step = self.masked_uncertainty_count

            if readiness_value >= self.cfg.high_threshold:
                if self.high_run_length == 0:
                    self.high_run_start_step = record.step_id
                    self.masked_memory_at_high_run_start = masked_before_step
                self.high_run_length += 1
            else:
                self.high_run_length = 0
                self.high_run_start_step = None
                self.masked_memory_at_high_run_start = 0

            if raw_low:
                self.raw_low_count += 1
            if smooth_low:
                self.smooth_low_count += 1
            if masked_uncertainty:
                self.masked_uncertainty_count += 1
                self.masked_memory = self.cfg.masked_decay * self.masked_memory + 1.0
                self.last_masked_step = record.step_id
            else:
                self.masked_memory = self.cfg.masked_decay * self.masked_memory

            if self.last_masked_step is None:
                self.steps_since_last_masked = None
            else:
                self.steps_since_last_masked = record.step_id - self.last_masked_step

            if self.masked_memory > 0.0:
                exposure_increment = 0.0
                if readiness_value >= self.cfg.high_threshold:
                    exposure_increment = 1.0
                    self.post_masked_high_count += 1
                elif readiness_value >= self.cfg.mid_high_threshold:
                    exposure_increment = 0.5
                    self.post_masked_mid_high_count += 1
                self.post_masked_exposure = self.cfg.exposure_decay * self.post_masked_exposure + exposure_increment
            else:
                self.post_masked_exposure = 0.0

            self.max_high_run_length = max(self.max_high_run_length, self.high_run_length)
            self.max_post_masked_exposure = max(self.max_post_masked_exposure, self.post_masked_exposure)

            self._last_ciod_risk_v1 = self._ciod_risk_v1()
            self.max_ciod_risk_v1 = max(self.max_ciod_risk_v1, self._last_ciod_risk_v1)
            self._last_ciod_shadow_trigger_v1 = self._last_ciod_risk_v1 > 0.0
            if self._last_ciod_shadow_trigger_v1:
                self.ciod_shadow_trigger_count_v1 += 1
                if self.first_ciod_shadow_trigger_step_v1 is None:
                    self.first_ciod_shadow_trigger_step_v1 = record.step_id

            self._last_ciod_risk_v2 = self._ciod_risk_v2(
                exposure_e0=self.cfg.exposure_e0,
                lambda0=self.cfg.lambda0,
            )
            self.max_ciod_risk_v2 = max(self.max_ciod_risk_v2, self._last_ciod_risk_v2)
            self._last_ciod_shadow_trigger_v2 = self._last_ciod_risk_v2 >= self.cfg.risk_threshold
            if self._last_ciod_shadow_trigger_v2:
                self.ciod_shadow_trigger_count_v2 += 1
                if self.first_ciod_shadow_trigger_step_v2 is None:
                    self.first_ciod_shadow_trigger_step_v2 = record.step_id

            self._update_grid(record.step_id)
            self._update_ciod_event_controller(record)

        return self._confidence_process_snapshot(
            scored=scored,
            raw_low=raw_low,
            smooth_low=smooth_low,
            masked_uncertainty=masked_uncertainty,
        )

    def current_snapshot(self) -> dict[str, Any]:
        snapshot = self._confidence_process_snapshot(
            scored=False,
            raw_low=False,
            smooth_low=False,
            masked_uncertainty=False,
        )
        snapshot["ciod_event_shadow"] = False
        return snapshot

    def summary(self) -> dict[str, Any]:
        return {
            "raw_low_count": self.raw_low_count,
            "smooth_low_count": self.smooth_low_count,
            "masked_uncertainty_count": self.masked_uncertainty_count,
            "masked_uncertainty_gap": self.masked_uncertainty_gap,
            "max_high_run_length": self.max_high_run_length,
            "ciod_risk": self._last_ciod_risk_v1,
            "ciod_shadow_trigger": self._last_ciod_shadow_trigger_v1,
            "max_ciod_risk": self.max_ciod_risk_v1,
            "ciod_shadow_trigger_count": self.ciod_shadow_trigger_count_v1,
            "first_ciod_shadow_trigger_step": self.first_ciod_shadow_trigger_step_v1,
            "ciod_risk_v1": self._last_ciod_risk_v1,
            "ciod_shadow_trigger_v1": self._last_ciod_shadow_trigger_v1,
            "max_ciod_risk_v1": self.max_ciod_risk_v1,
            "ciod_shadow_trigger_count_v1": self.ciod_shadow_trigger_count_v1,
            "first_ciod_shadow_trigger_step_v1": self.first_ciod_shadow_trigger_step_v1,
            "masked_memory": self.masked_memory,
            "last_masked_step": self.last_masked_step,
            "steps_since_last_masked": self.steps_since_last_masked,
            "post_masked_exposure": self.post_masked_exposure,
            "post_masked_high_count": self.post_masked_high_count,
            "post_masked_mid_high_count": self.post_masked_mid_high_count,
            "max_post_masked_exposure": self.max_post_masked_exposure,
            "ciod_risk_v2": self._last_ciod_risk_v2,
            "ciod_shadow_trigger_v2": self._last_ciod_shadow_trigger_v2,
            "max_ciod_risk_v2": self.max_ciod_risk_v2,
            "ciod_shadow_trigger_count_v2": self.ciod_shadow_trigger_count_v2,
            "first_ciod_shadow_trigger_step_v2": self.first_ciod_shadow_trigger_step_v2,
            "ciod_grid_summary": self._grid_summary(),
            "ciod_event_count": self.ciod_event_count,
            "first_ciod_event_step": self.first_ciod_event_step,
            "last_ciod_event_step": self.last_ciod_event_step,
            "ciod_event_before_first_readiness_low": bool(
                self.first_ciod_event_step is not None
                and (
                    self.first_readiness_low_step is None
                    or self.first_ciod_event_step < self.first_readiness_low_step
                )
            ),
            "ciod_event_steps": list(self.ciod_event_steps),
            "ciod_active_lease_count": self.ciod_active_lease_count,
        }


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
            "readiness_value": r.readiness_value,
            "stagnation_score": r.stagnation_score,
            "stagnation_suspect": r.stagnation_suspect,
            "stagnation_confirmed": r.stagnation_confirmed,
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
    if cfg.readiness.value_field == "c_raw":
        return step.readiness_raw
    if cfg.readiness.smooth_window > 1:
        return step.readiness_raw_smooth
    return step.readiness_raw


def _latest_clean_autonomy_anchor(records: list[StepRecord]) -> int | None:
    for record in reversed(records):
        if (
            record.active
            and record.generator == "slm"
            and not record.is_recovery
            and not record.removed_by_rollback
        ):
            return record.step_id
    return None


def _rollback_anchor_for_stagnation(
    active_records: list[StepRecord],
    clean_autonomy_anchor: int | None,
    cfg: SARRConfig,
) -> int:
    if clean_autonomy_anchor is not None:
        return clean_autonomy_anchor
    if cfg.rollback.fallback_if_no_clean_anchor == "zero" or cfg.anchor.fallback == "zero":
        return 0
    return choose_best_prefix_anchor(active_records, allow_zero=True)


def _readiness_label(record: StepRecord) -> str:
    if record.readiness_high:
        return "HIGH_CONF"
    if record.readiness_mid:
        return "MID_CONF"
    if record.readiness_low:
        return "LOW_CONF"
    return "UNKNOWN_CONF"


def _lease_budget_exceeded(
    *,
    cfg: SARRConfig,
    llm_lease_event_count: int,
    llm_lease_token_count: int,
    requested_steps: int,
    route_source: str,
    ciod_event_count: int = 0,
) -> bool:
    if not cfg.routing.enabled or not cfg.llm_lease.enabled:
        return True
    if requested_steps <= 0:
        return True
    if llm_lease_event_count >= cfg.routing_budget.max_total_llm_events_per_problem:
        return True
    if llm_lease_token_count >= cfg.routing_budget.max_total_llm_tokens_per_problem:
        return True
    if route_source == "ciod_event" and ciod_event_count >= cfg.routing_budget.max_ciod_events_per_problem:
        return True
    if route_source == "readiness" and cfg.routing_budget.max_readiness_events_per_problem <= 0:
        return True
    if route_source == "stagnation" and cfg.routing_budget.max_stagnation_events_per_problem <= 0:
        return True
    return False


def _lease_steps_within_budget(
    *,
    cfg: SARRConfig,
    llm_lease_token_count: int,
    requested_steps: int,
) -> int:
    remaining_tokens = cfg.routing_budget.max_total_llm_tokens_per_problem - llm_lease_token_count
    if remaining_tokens <= 0:
        return 0
    token_bounded_steps = remaining_tokens // cfg.llm_lease.max_tokens_per_step
    return max(0, min(requested_steps, token_bounded_steps))


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
        token_count=_actual_token_count(output),
        prompt_tokens=output.prompt_tokens,
    )
    return output


def _confidence_for_prefix(slm, cfg: SARRConfig, problem_text: str, assistant_prefix_text: str):
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
        record.extra["readiness_high_diagnostic_only"] = record.readiness_high
        record.extra["ready_for_slm"] = False
        records.append(record)
        attempt_id += 1
    return local_context, records, "EXHAUSTED_FORCE_SLM", attempt_id


def run_llm_lease(
    *,
    problem_id: str,
    attempt_id_start: int,
    start_step_id: int,
    state: GenerationState,
    context: str,
    llm,
    cfg: SARRConfig,
    lease_steps: int,
    remaining_think_tokens: int,
    reason: str,
) -> tuple[str, list[StepRecord], str, int]:
    records: list[StepRecord] = []
    attempt_id = attempt_id_start
    local_context = context
    for lease_idx in range(1, lease_steps + 1):
        remaining = max(1, remaining_think_tokens - _visible_think_tokens(records))
        output = _generate_llm_step(
            llm,
            state,
            cfg,
            assistant_prefix_text=local_context,
            remaining_think_tokens=remaining,
            max_new_tokens_override=cfg.llm_lease.max_tokens_per_step,
        )
        local_context += output.text
        record = _make_record(
            problem_id=problem_id,
            attempt_id=attempt_id,
            step_id=start_step_id + lease_idx - 1,
            generator="llm",
            output=output,
            c_raw=None,
            c_norm=None,
            c_smooth=None,
            state_before="LLM_LEASE",
            D_start=0,
            D_post=0,
            stable_anchor=None,
            action="LLM_LEASE_STEP",
            c_info=None,
            is_recovery=True,
        )
        record.readiness_source = "raw"
        record.calibration_enabled = False
        record.state_after = "LLM_LEASE"
        record.autonomy_state = "LLM_LEASE"
        record.extra["lease_step"] = lease_idx
        record.extra["lease_reason"] = reason
        record.extra["prompt_type"] = cfg.llm_lease.prompt_type
        record.extra["mention_uncertainty"] = cfg.llm_lease.mention_uncertainty
        record.extra["mention_stagnation"] = cfg.llm_lease.mention_stagnation
        record.extra["mention_repetition"] = cfg.llm_lease.mention_repetition
        record.extra["mention_error"] = cfg.llm_lease.mention_error
        records.append(record)
        attempt_id += 1
        if _strict_thinking_stop_reason(output, output.text) is not None:
            return local_context, records, "LEASE_FINISHED_THINKING_STOP", attempt_id
    return local_context, records, "LEASE_FINISHED", attempt_id


def _append_final_answer(
    *,
    problem_id: str,
    state: GenerationState,
    step_logs: list[dict[str, Any]],
    slm,
    llm,
    cfg: SARRConfig,
    confidence_process: dict[str, Any] | None = None,
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
    row = {
        "problem_id": problem_id,
        "step_id": state.step_count + 1,
        "generator": account,
        "text": output.text,
        "token_count": output.token_count,
        "finish_reason": output.finish_reason,
        "action": "FINAL_ANSWER",
        "is_final_answer": True,
        "extra": {"confidence_process": confidence_process or {}},
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
    ever_left_startup = False
    post_lease_observe_remaining = 0
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
    stagnation_suspect_run = 0
    hcs_rollback_count = 0
    stagnation_rollback_count = 0
    total_rollback_count = 0
    llm_lease_event_count = 0
    llm_lease_token_count = 0
    low_confidence_run = 0
    readiness_low_run = 0
    readiness_history: list[float] = []
    force_next_step_slm = False
    pending_force_recovery_event_idx: int | None = None
    long_span_fallback_counts: dict[int, int] = {}
    rollback_anchor_counts: dict[int, int] = {}
    startup_monitor_steps = 0
    attempt_id = 1
    stop_reason: str | None = None
    state_enter_step = 0
    recovery_context: str | None = None
    pending_state_machine_action: str | None = None
    pending_invalid_rollback_recovery_state = False
    confidence_process = ConfidenceProcessTracker(cfg)

    def state_duration(step_id: int) -> int:
        return max(1, step_id - state_enter_step)

    def recover_invalid_rollback_recovery_state(step_id: int) -> None:
        nonlocal pending_invalid_rollback_recovery_state, pending_state_machine_action
        if monitor_state != "ROLLBACK_RECOVERY" or recovery_context is not None:
            return
        pending_invalid_rollback_recovery_state = True
        pending_state_machine_action = "STATE_RECOVERED_TO_SLM_ACTIVE"
        event = {
            "event": "state_machine_invariant",
            "state": monitor_state,
            "state_duration": state_duration(step_id),
            "invalid_rollback_recovery_state": True,
            "recovery_context_present": False,
            "action": "STATE_RECOVERED_TO_SLM_ACTIVE",
        }
        transition_events.append({"problem_id": problem_id, "step_id": step_id, **event})
        state.trace.append(TraceEvent(step_id, "state_machine_invariant", event))
        transition_state(step_id, "SLM_ACTIVE", "STATE_RECOVERED_TO_SLM_ACTIVE")

    def finalize_record(record: StepRecord) -> StepRecord:
        nonlocal pending_invalid_rollback_recovery_state, pending_state_machine_action
        if pending_state_machine_action is not None:
            if record.action != pending_state_machine_action:
                record.extra["action_before_state_machine_recovery"] = record.action
            record.action = pending_state_machine_action
            pending_state_machine_action = None
        invalid = bool(
            pending_invalid_rollback_recovery_state
            or (monitor_state == "ROLLBACK_RECOVERY" and recovery_context is None)
        )
        record.invalid_rollback_recovery_state = invalid
        record.state_duration = state_duration(record.step_id)
        record.extra["state_duration"] = record.state_duration
        record.extra["invalid_rollback_recovery_state"] = invalid
        record.extra["anchor_refresh_blocked_reason"] = record.anchor_refresh_blocked_reason
        record.extra["recovery_context_present"] = recovery_context is not None
        if invalid:
            record.extra["state_machine_action"] = "STATE_RECOVERED_TO_SLM_ACTIVE"
        if "confidence_process" not in record.extra:
            record.extra["confidence_process"] = confidence_process.observe(record)
        confidence_snapshot = record.extra.get("confidence_process")
        if isinstance(confidence_snapshot, dict) and confidence_snapshot.get("ciod_event_shadow"):
            if record.action == "LLM_LEASE_BY_CIOD_EVENT":
                record.extra["ciod_event_reason"] = "LLM_LEASE_BY_CIOD_EVENT"
            else:
                record.extra["ciod_event_reason"] = "CIOD_EVENT_SHADOW_ONLY"
                passive_actions = {
                    "TRUST",
                    "REFRESH_STABLE_ANCHOR",
                    "SUSPECT_RECOVERED",
                    "USEFUL_EXPLORATION",
                    "LOW_CONFIDENCE_OBSERVE",
                    "SUSPECT_OBSERVE",
                    "READINESS_LOW_DIAGNOSTIC_ONLY",
                    "STAGNATION_DIAGNOSTIC_ONLY",
                }
                if record.action in passive_actions:
                    record.extra.setdefault("action_without_ciod_event", record.action)
                    record.action = "CIOD_EVENT_SHADOW_ONLY"
        pending_invalid_rollback_recovery_state = False
        return record

    def append_record(record: StepRecord) -> None:
        step_logs.append(_serialize_step(finalize_record(record)))

    def replace_last_record(record: StepRecord) -> None:
        if step_logs:
            step_logs[-1] = _serialize_step(finalize_record(record))

    def transition_state(step_id: int, new_state: str, reason: str) -> None:
        nonlocal monitor_state, ever_left_startup, state_enter_step
        old_state = monitor_state
        if old_state == new_state:
            return
        if old_state == "STARTUP":
            ever_left_startup = True
        if new_state == "STARTUP" and ever_left_startup:
            event = {
                "event": "startup_reentry_blocked",
                "reason": "NEVER_REENTER_AFTER_RECOVERY_OR_LEASE",
                "active_step_count": len(active_records),
                "requested_state": new_state,
            }
            state.trace.append(TraceEvent(step_id, "startup_reentry_blocked", event))
            transition_events.append({"problem_id": problem_id, **event})
            new_state = "SLM_ACTIVE"
        event = {
            "event": "state_transition",
            "from": old_state,
            "to": new_state,
            "reason": reason,
        }
        transition_events.append({"problem_id": problem_id, "step_id": step_id, **event})
        state.trace.append(TraceEvent(step_id, "state_transition", event))
        monitor_state = new_state
        state_enter_step = step_id

    try:
        while state.phase != Phase.DONE:
            if monitor_state == "STARTUP" and ever_left_startup:
                transition_state(state.step_count, "SLM_ACTIVE", "STARTUP_REENTRY_BLOCKED")
            recover_invalid_rollback_recovery_state(state.step_count)
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
            thinking_stop = _strict_thinking_stop_reason(output, output.text)
            if thinking_stop is not None:
                record = _make_record(
                    problem_id=problem_id,
                    attempt_id=attempt_id,
                    step_id=step_id,
                    generator="slm",
                    output=output,
                    c_raw=None,
                    c_norm=None,
                    c_smooth=None,
                    state_before=state_before,
                    D_start=D_start,
                    D_post=D_post,
                    stable_anchor=stable_anchor,
                    action="FINISHED" if thinking_stop == "finished" else f"STOP_{thinking_stop.upper()}",
                    c_info=None,
                )
                if forced_after_recovery:
                    record.extra["forced_after_recovery"] = True
                record.extra["confidence_skipped"] = True
                record.extra["confidence_skipped_reason"] = thinking_stop
                attempt_id += 1
                active_records.append(record)
                all_records.append(record)
                transition = _apply_transition_metadata(active_records)
                if transition is not None:
                    transition_events.append({"problem_id": problem_id, **transition})
                startup_monitor_steps += 1
                state.step_count = len(active_records)
                record.state_after = monitor_state
                append_record(record)
                stop_reason = thinking_stop
                break

            c_raw, c_norm, c_info = _confidence_for_prefix(
                slm,
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

            should_rollback = False
            rollback_type = ""
            rollback_reason = ""
            anchor = 0
            is_hcs_rollback = False
            is_stagnation_rollback = False
            lease_reason: str | None = None
            lease_steps = 0
            lease_rollback_before = False
            lease_source: str | None = None

            record.readiness_value = readiness
            record.readiness = readiness
            record.readiness_high = bool(readiness is not None and readiness >= cfg.readiness.high_threshold)
            record.readiness_low = bool(readiness is not None and readiness <= cfg.readiness.low_threshold)
            record.readiness_mid = bool(
                readiness is not None and not record.readiness_high and not record.readiness_low
            )
            record.extra["readiness_value"] = readiness
            record.extra["readiness_field"] = cfg.readiness.value_field
            record.extra["confidence_process"] = confidence_process.observe(record)
            record.stagnation_score = surface_stagnation_score(active_records, cfg)
            record.stagnation_high = bool(
                cfg.stagnation.enabled and record.stagnation_score >= cfg.stagnation.high_threshold
            )
            record.stagnation_suspect = bool(cfg.stagnation.enabled and record.stagnation_high)
            stagnation_suspect_run = stagnation_suspect_run + 1 if record.stagnation_suspect else 0
            record.stagnation_suspect_run = stagnation_suspect_run
            record.stagnation_confirmed = bool(
                record.stagnation_suspect and stagnation_suspect_run >= cfg.stagnation.patience
            )
            hcs_detection_enabled = bool(cfg.hcs.enabled and cfg.stagnation.enabled)
            record.hcs_suspect = bool(hcs_detection_enabled and record.readiness_high and record.stagnation_high)
            hcs_suspect_run = hcs_suspect_run + 1 if record.hcs_suspect else 0
            record.hcs_suspect_run = hcs_suspect_run
            record.hcs_confirmed = bool(
                record.hcs_suspect and hcs_suspect_run >= cfg.hcs.suspect_patience
            )

            active_slm_steps = sum(1 for r in active_records if r.active and r.generator == "slm")
            if monitor_state == "STARTUP":
                if startup_monitor_steps >= cfg.startup.B_min:
                    transition_state(step_id, "SLM_ACTIVE", "STARTUP_DIAGNOSTIC_WINDOW_COMPLETE")
                elif active_slm_steps > int(cfg.startup.max_steps or cfg.startup.B_max):
                    transition_state(step_id, "SLM_ACTIVE", "STARTUP_MAX_STEPS_EXCEEDED")

            record.anchor_refresh_allowed = bool(
                record.generator == "slm"
                and monitor_state == "SLM_ACTIVE"
                and record.active
            )
            if record.anchor_refresh_allowed:
                clean_autonomy_anchor = step_id
                stable_anchor = step_id
                record.stable_anchor = stable_anchor
                record.anchor_refresh_blocked_reason = None
                if cfg.startup.never_reenter_after_clean_anchor and not ever_left_startup:
                    ever_left_startup = True
            elif record.generator != "slm":
                record.anchor_refresh_blocked_reason = "NOT_SLM_STEP"
            elif monitor_state == "ROLLBACK_RECOVERY" and recovery_context is not None:
                record.anchor_refresh_blocked_reason = "STATE_ROLLBACK_RECOVERY"
            elif monitor_state == "ROLLBACK_RECOVERY":
                record.anchor_refresh_blocked_reason = "STATE_RECOVERED_TO_SLM_ACTIVE"
            elif monitor_state != "SLM_ACTIVE":
                record.anchor_refresh_blocked_reason = f"STATE_{monitor_state}"
            else:
                record.anchor_refresh_blocked_reason = "INACTIVE_STEP"
            record.clean_autonomy_anchor = clean_autonomy_anchor

            if record.stagnation_confirmed:
                if record.readiness_high:
                    record.autonomy_state = "HIGH_CONF_STAGNATION"
                elif record.readiness_mid:
                    record.autonomy_state = "MID_CONF_STAGNATION"
                elif record.readiness_low:
                    record.autonomy_state = "LOW_CONF_STAGNATION_COLLAPSE"
                else:
                    record.autonomy_state = "UNKNOWN_CONF_STAGNATION"
            elif record.stagnation_suspect:
                record.autonomy_state = f"{_readiness_label(record)}_STAGNATION_SUSPECT"
            elif record.readiness_low:
                record.autonomy_state = "LOW_READINESS"
            else:
                record.autonomy_state = "NORMAL"

            if record.stagnation_confirmed:
                record.extra["stagnation_routing_source"] = "STAGNATION_DIAGNOSTIC_ONLY"
                record.extra["stagnation_diagnostic_only"] = True
                record.extra["routing_source"] = "STAGNATION_DIAGNOSTIC_ONLY"
                if record.action == "TRUST":
                    record.action = "STAGNATION_DIAGNOSTIC_ONLY"

            confidence_snapshot = record.extra.get("confidence_process", {})
            ciod_event_shadow = bool(
                isinstance(confidence_snapshot, dict)
                and confidence_snapshot.get("ciod_event_shadow")
            )
            if (
                not should_rollback
                and lease_reason is None
                and ciod_event_shadow
                and monitor_state == "SLM_ACTIVE"
            ):
                record.extra["ciod_event_active_routing_enabled"] = cfg.confidence_process.enable_ciod_active_routing
                if not cfg.confidence_process.enable_ciod_active_routing:
                    record.extra["ciod_event_routing_decision"] = "CIOD_EVENT_SHADOW_ONLY"
                elif not confidence_process.ciod_active_route_available():
                    record.extra["ciod_event_routing_decision"] = "CIOD_EVENT_ACTIVE_LEASE_LIMIT"
                elif not (cfg.routing.enabled and cfg.llm_lease.enabled):
                    record.extra["ciod_event_routing_decision"] = "CIOD_EVENT_ROUTING_DISABLED"
                else:
                    lease_reason = "LLM_LEASE_BY_CIOD_EVENT"
                    lease_steps = cfg.llm_lease.max_steps_per_event
                    lease_rollback_before = False
                    lease_source = "ciod_event"
                    record.action = "LLM_LEASE_BY_CIOD_EVENT"
                    record.autonomy_state = "CIOD_EVENT"
                    record.extra["ciod_event_routing_decision"] = "LLM_LEASE_BY_CIOD_EVENT"

            if not should_rollback and lease_reason is None and record.anchor_refresh_allowed:
                if record.action == "TRUST":
                    record.action = "REFRESH_STABLE_ANCHOR"

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
                record.extra["startup_degradation_diagnostic_only"] = bool(v)
            elif monitor_state == "SLM_ACTIVE" and lease_reason is None:
                D_post += v
                record.D_post = D_post
                low_confidence_signal = bool(record.readiness_low or v)
                if low_confidence_signal:
                    low_confidence_run += 1
                else:
                    low_confidence_run = 0
                if record.readiness_low:
                    readiness_low_run += 1
                else:
                    readiness_low_run = 0
                record.extra["low_confidence_run"] = low_confidence_run
                record.extra["readiness_low_run"] = readiness_low_run
                if record.readiness_low:
                    record.extra["readiness_routing_source"] = "READINESS_LOW_DIAGNOSTIC_ONLY"
                    record.extra["readiness_diagnostic_only"] = True
                    if record.action in {"TRUST", "REFRESH_STABLE_ANCHOR"}:
                        record.action = "READINESS_LOW_DIAGNOSTIC_ONLY"
                    record.extra.setdefault("routing_source", "READINESS_LOW_DIAGNOSTIC_ONLY")
                if record.stagnation_confirmed:
                    record.extra["stagnation_routing_source"] = "STAGNATION_DIAGNOSTIC_ONLY"
                    record.extra["stagnation_diagnostic_only"] = True
                    record.extra["routing_source"] = "STAGNATION_DIAGNOSTIC_ONLY"
                    if record.action in {"TRUST", "REFRESH_STABLE_ANCHOR", "READINESS_LOW_DIAGNOSTIC_ONLY"}:
                        record.action = "STAGNATION_DIAGNOSTIC_ONLY"
                if low_confidence_signal:
                    record.extra["low_confidence_diagnostic_only"] = True
            elif monitor_state == "SUSPECT":
                record.extra["suspect_diagnostic_only"] = True
                if record.action == "TRUST":
                    record.action = "SUSPECT_DIAGNOSTIC_ONLY"

            record.state_after = monitor_state
            append_record(record)

            if not should_rollback and lease_reason is None:
                continue

            current_step_id = len(active_records)
            if lease_reason is not None and not should_rollback:
                record.extra["lease_source"] = lease_source
                record.extra["routing_source"] = lease_reason
                lease_steps = _lease_steps_within_budget(
                    cfg=cfg,
                    llm_lease_token_count=llm_lease_token_count,
                    requested_steps=lease_steps,
                )
                if _lease_budget_exceeded(
                    cfg=cfg,
                    llm_lease_event_count=llm_lease_event_count,
                    llm_lease_token_count=llm_lease_token_count,
                    requested_steps=lease_steps,
                    route_source=lease_source or "",
                    ciod_event_count=confidence_process.ciod_active_lease_count,
                ):
                    if lease_source == "ciod_event":
                        record.extra["ciod_event_routing_decision"] = "CIOD_EVENT_ROUTING_BUDGET_EXCEEDED"
                    record.action = "ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM"
                    record.extra["routing_budget_exceeded"] = True
                    record.extra["routing_source"] = "ROUTING_BUDGET_EXCEEDED"
                    recovery_context = None
                    transition_state(current_step_id, "SLM_ACTIVE", "ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM")
                    record.state_after = monitor_state
                    replace_last_record(record)
                    continue
                if lease_source == "ciod_event":
                    ciod_active_lease_count = confidence_process.mark_ciod_active_lease()
                    record.extra["ciod_active_lease_count"] = ciod_active_lease_count
                    if isinstance(record.extra.get("confidence_process"), dict):
                        record.extra["confidence_process"]["ciod_active_lease_count"] = ciod_active_lease_count
                    replace_last_record(record)
                transition_state(current_step_id, "LLM_LEASE", lease_reason)
                lease_context, lease_records, lease_stop_reason, attempt_id = run_llm_lease(
                    problem_id=problem_id,
                    attempt_id_start=attempt_id,
                    start_step_id=len(active_records) + 1,
                    state=state,
                    context=state.assistant_prefix_text,
                    llm=llm,
                    cfg=cfg,
                    lease_steps=lease_steps,
                    remaining_think_tokens=max(1, cfg.generation.think_token_budget - _visible_think_tokens(active_records)),
                    reason=lease_reason,
                )
                for lease_record in lease_records:
                    all_records.append(lease_record)
                new_active_records = active_records + lease_records
                for idx in range(max(1, len(active_records)), len(new_active_records)):
                    transition = _apply_transition_metadata(new_active_records[: idx + 1])
                    if transition is not None:
                        transition_events.append({"problem_id": problem_id, **transition})
                active_records = new_active_records
                state.assistant_prefix_text = lease_context
                state.step_count = len(active_records)
                for lease_record in lease_records:
                    append_record(lease_record)
                event_tokens = sum(r.token_count for r in lease_records)
                llm_lease_event_count += 1
                llm_lease_token_count += event_tokens
                event = RollbackEvent(
                    problem_id=problem_id,
                    type="LLM_LEASE",
                    reason=lease_reason,
                    trigger_step=current_step_id,
                    anchor_step=-1,
                    rollback_span=0,
                    removed_steps=[],
                    recovery_steps=[_serialize_step(r) for r in lease_records],
                    stop_reason=lease_stop_reason,
                    recovery_max_steps=lease_steps,
                    recovery_actual_steps=len(lease_records),
                    recovery_c_norm=[],
                    force_next_step_slm=False,
                    event="llm_lease",
                    rollback_before_lease=False,
                    rollback_anchor=None,
                    lease_steps=lease_steps,
                    max_tokens_per_step=cfg.llm_lease.max_tokens_per_step,
                    prompt_type=cfg.llm_lease.prompt_type,
                    mention_uncertainty=cfg.llm_lease.mention_uncertainty,
                    mention_stagnation=cfg.llm_lease.mention_stagnation,
                    mention_repetition=cfg.llm_lease.mention_repetition,
                    mention_error=cfg.llm_lease.mention_error,
                    return_to_slm=False,
                    state_after="SLM_ACTIVE",
                )
                rollback_events.append(event)
                state.trace.append(TraceEvent(current_step_id, "llm_lease", asdict(event)))
                transition_state(state.step_count, "SLM_ACTIVE", "LEASE_FINISHED")
                post_lease_observe_remaining = 0
                force_next_step_slm = False
                ever_left_startup = True
                hcs_suspect_run = 0
                stagnation_suspect_run = 0
                low_confidence_run = 0
                readiness_low_run = 0
                if has_close_think_tag(state.assistant_prefix_text):
                    stop_reason = "finished"
                    break
                continue

            requested_anchor = anchor
            anchor_repeat_count_before = rollback_anchor_counts.get(requested_anchor, 0)
            if lease_reason is not None:
                lease_steps = _lease_steps_within_budget(
                    cfg=cfg,
                    llm_lease_token_count=llm_lease_token_count,
                    requested_steps=lease_steps,
                )
            if lease_reason is not None and _lease_budget_exceeded(
                cfg=cfg,
                llm_lease_event_count=llm_lease_event_count,
                llm_lease_token_count=llm_lease_token_count,
                requested_steps=lease_steps,
                route_source=lease_source or "",
                ciod_event_count=confidence_process.ciod_active_lease_count,
            ):
                record.action = "ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM"
                record.extra["routing_budget_exceeded"] = True
                record.extra["routing_source"] = "ROUTING_BUDGET_EXCEEDED"
                recovery_context = None
                transition_state(current_step_id, "SLM_ACTIVE", "ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM")
                D_suspect = 0
                suspect_anchor = None
                suspect_start_step = None
                suspect_steps = 0
                suspect_max_readiness = None
                low_confidence_run = 0
                readiness_low_run = 0
                record.state_after = monitor_state
                replace_last_record(record)
                continue
            if pending_force_recovery_event_idx is not None:
                rollback_events[pending_force_recovery_event_idx].force_slm_after_recovery_failed = True
                pending_force_recovery_event_idx = None

            if (
                not is_stagnation_rollback
                and requested_anchor == 0
                and cfg.rollback.root_rollback_action == "force_close_think"
                and anchor_repeat_count_before >= cfg.rollback.max_root_rollbacks
            ):
                record.action = "ROOT_ROLLBACK_FORCE_CLOSE_THINK"
                record.extra["root_rollback_count_before"] = anchor_repeat_count_before
                replace_last_record(record)
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
                not is_stagnation_rollback
                and anchor_repeat_count_before > 0
                and cfg.rollback.anchor_repeat_policy == "suppress"
            ):
                rollback_anchor_counts[requested_anchor] = anchor_repeat_count_before + 1
                record.action = f"SUPPRESS_REPEATED_{rollback_type}"
                record.extra["suppressed_rollback"] = True
                record.extra["requested_anchor_step"] = requested_anchor
                record.extra["anchor_repeat_count_before"] = anchor_repeat_count_before
                transition_state(current_step_id, "SLM_ACTIVE", "SUPPRESSED_REPEATED_ROLLBACK")
                stable_anchor = current_step_id
                D_start = 0
                D_post = 0
                D_suspect = 0
                suspect_anchor = None
                suspect_start_step = None
                suspect_steps = 0
                suspect_max_readiness = None
                readiness_history = []
                low_confidence_run = 0
                readiness_low_run = 0
                startup_monitor_steps = 0
                record.state_after = monitor_state
                record.stable_anchor = stable_anchor
                record.D_start = D_start
                record.D_post = D_post
                replace_last_record(record)
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
                not is_stagnation_rollback
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
            if is_stagnation_rollback:
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
                if lease_reason is not None:
                    max_recovery = lease_steps
                elif is_hcs_rollback:
                    max_recovery = cfg.hcs_recovery.max_llm_steps
                elif long_span:
                    max_recovery = min(span + 1, cfg.rollback.long_span_recovery_steps)
                else:
                    max_recovery = span + 1
                recovery_start_step_id = anchor + 1

            if is_hcs_rollback:
                hcs_rollback_count += 1
            if is_stagnation_rollback:
                stagnation_rollback_count += 1
            total_rollback_count += 1

            recovery_context = None if lease_reason is not None else rollback_context
            transition_state(current_step_id, "LLM_LEASE" if lease_reason is not None else "ROLLBACK_RECOVERY", rollback_reason)
            if lease_reason is not None:
                rec_context, rec_records, rec_stop_reason, attempt_id = run_llm_lease(
                    problem_id=problem_id,
                    attempt_id_start=attempt_id,
                    start_step_id=recovery_start_step_id,
                    state=state,
                    context=rollback_context,
                    llm=llm,
                    cfg=cfg,
                    lease_steps=max_recovery,
                    remaining_think_tokens=max(1, cfg.generation.think_token_budget - _visible_think_tokens(kept)),
                    reason=lease_reason,
                )
            else:
                rec_context, rec_records, rec_stop_reason, attempt_id = confidence_gated_recovery(
                    problem_id=problem_id,
                    attempt_id_start=attempt_id,
                    start_step_id=recovery_start_step_id,
                    state=state,
                    context=rollback_context,
                    llm=llm,
                    slm=slm,
                    cfg=cfg,
                    max_recovery_steps=max_recovery,
                    remaining_think_tokens=max(1, cfg.generation.think_token_budget - _visible_think_tokens(kept)),
                    max_tokens_per_step=cfg.hcs_recovery.max_tokens_per_step if is_hcs_rollback else None,
                )

            new_active_records = kept + rec_records
            recovery_state_after = "SLM_ACTIVE"
            if rec_records and (lease_reason is not None or rec_stop_reason in RECOVERY_COMPLETE_TO_SLM):
                rec_records[-1].state_after = recovery_state_after
                rec_records[-1].extra["recovery_terminal_state_after"] = recovery_state_after
            for rec in rec_records:
                finalize_record(rec)
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
                    cfg.llm_lease.return_to_slm
                    if lease_reason is not None
                    else cfg.hcs_recovery.return_to_slm_after_recovery
                    if is_hcs_rollback
                    else cfg.rollback.force_slm_after_recovery
                ),
                event="llm_lease" if lease_reason is not None else "hcs_rollback" if is_hcs_rollback else None,
                rollback_before_lease=lease_rollback_before if lease_reason is not None else None,
                rollback_anchor=anchor if lease_reason is not None and lease_rollback_before else None,
                lease_steps=max_recovery if lease_reason is not None else None,
                max_tokens_per_step=cfg.llm_lease.max_tokens_per_step if lease_reason is not None else None,
                prompt_type=cfg.llm_lease.prompt_type if lease_reason is not None else None,
                mention_uncertainty=cfg.llm_lease.mention_uncertainty if lease_reason is not None else None,
                mention_repetition=cfg.llm_lease.mention_repetition if lease_reason is not None else None,
                mention_error=cfg.llm_lease.mention_error if lease_reason is not None else None,
                state_after=recovery_state_after,
                clean_anchor_step=anchor if is_stagnation_rollback else None,
                hcs_rollback_count=hcs_rollback_count if is_hcs_rollback else 0,
                stagnation_rollback_count=stagnation_rollback_count if is_stagnation_rollback else 0,
                readiness_source="raw" if is_stagnation_rollback or is_hcs_rollback else None,
                calibration_enabled=False if is_stagnation_rollback or is_hcs_rollback else None,
                llm_recovery_prompt_type=(
                    cfg.llm_lease.prompt_type if lease_reason is not None else cfg.hcs_recovery.prompt_type if is_hcs_rollback else None
                ),
                mention_stagnation=(
                    cfg.llm_lease.mention_stagnation if lease_reason is not None else cfg.hcs_recovery.mention_stagnation if is_hcs_rollback else None
                ),
                return_to_slm=(
                    cfg.llm_lease.return_to_slm
                    if lease_reason is not None
                    else cfg.hcs_recovery.return_to_slm_after_recovery
                    if is_hcs_rollback
                    else None
                ),
            )
            rollback_events.append(event)
            if lease_reason is not None:
                event_tokens = sum(r.token_count for r in rec_records)
                llm_lease_event_count += 1
                llm_lease_token_count += event_tokens
            if not is_stagnation_rollback:
                rollback_anchor_counts[requested_anchor] = anchor_repeat_count_before + 1
            return_to_slm_after_recovery = (
                cfg.llm_lease.return_to_slm
                if lease_reason is not None
                else cfg.hcs_recovery.return_to_slm_after_recovery
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

            transition_state(state.step_count, recovery_state_after, "LEASE_FINISHED" if lease_reason is not None else rec_stop_reason)
            recovery_context = None
            post_lease_observe_remaining = 0
            ever_left_startup = True
            D_start = 0
            D_post = 0
            D_suspect = 0
            stable_anchor = clean_autonomy_anchor
            suspect_anchor = None
            suspect_start_step = None
            suspect_steps = 0
            suspect_max_readiness = None
            readiness_history = []
            clean_autonomy_anchor = _latest_clean_autonomy_anchor(active_records)
            hcs_suspect_run = 0
            stagnation_suspect_run = 0
            low_confidence_run = 0
            readiness_low_run = 0
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

    if stop_reason in {
        "think_token_budget",
        "root_rollback_limit",
        "HCS_ROLLBACK_LIMIT",
        "STAGNATION_ROLLBACK_LIMIT",
        "ROLLBACK_LIMIT",
    }:
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
                confidence_process=confidence_process.current_snapshot(),
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
                "num_rollbacks": sum(1 for event in rollback_events if event.type != "LLM_LEASE"),
                "num_llm_lease_events": sum(1 for event in rollback_events if event.event == "llm_lease"),
                "num_intervention_events": len(rollback_events),
                "num_transition_events": len(transition_events),
                "stop_reason": state.stop_reason,
                **confidence_process.summary(),
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
