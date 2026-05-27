from __future__ import annotations

import math
import re
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import ControllerConfig
from .records import StepOutput


OWNER_SLM = "SLM"
OWNER_LLM = "LLM"

MODE_COLD_START = "COLD_START"
MODE_SLM_NORMAL = "SLM_NORMAL"
MODE_SLM_TRANSITION = "SLM_TRANSITION"
MODE_LLM_REPAIR = "LLM_REPAIR"
MODE_SLM_PROBATION = "SLM_PROBATION"
MODE_SLM_REENTRY = "SLM_REENTRY"
MODE_FINALIZE = "FINALIZE"
MODE_LLM_FINALIZE = "LLM_FINALIZE"

WINDOW_TRUSTED = "trusted"
WINDOW_SUSPECT = "suspect"
WINDOW_FAILURE = "failure"
WINDOW_INVALID = "invalid"

STEP_ACTIVE = "active"
STEP_REMOVED = "removed"
STEP_FINAL_ANSWER = "final_answer"

REPAIR_LANDING_ANCHOR_Q = 0.95
REPAIR_LANDING_HIGH_Q = 0.75
REPAIR_LANDING_RLI_THRESHOLD = 0.60
REPAIR_LANDING_EPS = 1e-8


_ANSWER_INTENT_PATTERNS = [
    re.compile(r"final\s+answer", re.I),
    re.compile(r"the\s+answer\s+is", re.I),
    re.compile(r"therefore\s+the\s+answer", re.I),
    re.compile(r"answer\s*:", re.I),
    re.compile(r"\\boxed\s*\{", re.I),
]


class InvalidControllerState(RuntimeError):
    pass


@dataclass
class Step:
    step_id: int
    owner: str
    mode: str
    text: str
    token_ids: list[int]
    logprobs: list[float]
    start_token_idx: int
    end_token_idx: int
    episode_id: int
    active: bool = True
    action: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    wall_time: float = 0.0
    attempt_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def scored_token_count(self) -> int:
        return min(len(self.token_ids), len(self.logprobs))

    @property
    def status(self) -> str:
        return STEP_ACTIVE if self.active else STEP_REMOVED

    def to_dict(self, problem_id: str) -> dict[str, Any]:
        row = asdict(self)
        row.update(
            {
                "problem_id": problem_id,
                "source": self.owner,
                "generator": self.owner.lower(),
                "status": self.status,
                "token_count": self.token_count,
                "scored_token_count": self.scored_token_count,
            }
        )
        return row


@dataclass
class PDIWindow:
    window_id: int
    owner: str
    covered_step_ids: list[int]
    start_token_idx: int
    end_token_idx: int
    pdi: float
    status: str
    episode_id: int
    token_count: int
    q_percentile: float | None = None
    upper_excess: float | None = None
    upper_evidence: float | None = None

    def overlaps(self, start_token_idx: int, end_token_idx: int) -> bool:
        return self.start_token_idx <= end_token_idx and self.end_token_idx >= start_token_idx


@dataclass
class UpperEvidencePoint:
    window: PDIWindow
    q_percentile: float
    upper_excess: float
    upper_evidence: float


@dataclass
class EpisodeState:
    owner: str = OWNER_SLM
    mode: str = MODE_COLD_START
    trusted_buffer: list[PDIWindow] = field(default_factory=list)
    failure_buffer: list[PDIWindow] = field(default_factory=list)
    pre_suspect_snapshot: list[float] = field(default_factory=list)
    upper_evidence_history: list[UpperEvidencePoint] = field(default_factory=list)
    lower_tail_history: list[bool] = field(default_factory=list)
    handoff_history: list[float] = field(default_factory=list)
    handoff_candidate_step_ids: list[int] = field(default_factory=list)
    handoff_candidate_scores: list[float] = field(default_factory=list)
    repair_slm_side_scores: list[float] = field(default_factory=list)
    repair_slm_side_step_ids: list[int] = field(default_factory=list)
    handoff_point_token_idx: int | None = None
    probation_stable_count: int = 0
    transition_start_window: PDIWindow | None = None
    transition_windows: list[PDIWindow] = field(default_factory=list)
    reentry_cached_windows: list[PDIWindow] = field(default_factory=list)
    reentry_stable_count: int = 0
    reentry_transition_windows: list[PDIWindow] = field(default_factory=list)


@dataclass
class ControllerDecision:
    action: str
    step: Step | None = None
    window: PDIWindow | None = None
    q_percentile: float | None = None
    upper_excess: float | None = None
    upper_evidence: float | None = None
    rollback_start_token_idx: int | None = None
    handoff_q_slm_side: float | None = None
    repair_landing_index: float | None = None
    probation_status: str | None = None


class EmpiricalCDF:
    def __init__(self, values: list[float]) -> None:
        self.values = sorted(float(v) for v in values if math.isfinite(float(v)))

    def __call__(self, value: float) -> float:
        if not self.values:
            return 0.5
        return bisect_right(self.values, float(value)) / len(self.values)

    def smoothed(self, value: float) -> float:
        if not self.values:
            return 0.5
        count = bisect_right(self.values, float(value))
        return (count + 0.5) / (len(self.values) + 1.0)

    def to_list(self) -> list[float]:
        return list(self.values)


class EffectiveCDF:
    def __init__(self, *, prior: list[float], trusted: list[float], lambda0: float) -> None:
        self.prior = EmpiricalCDF(prior)
        self.trusted = EmpiricalCDF(trusted)
        self.lambda0 = max(0.0, float(lambda0)) if self.prior.values else 0.0
        self.n_trusted = len(self.trusted.values)

    @property
    def denominator(self) -> float:
        return self.lambda0 + self.n_trusted

    @property
    def prior_weight(self) -> float:
        denom = self.denominator
        return self.lambda0 / denom if denom > 0 else 0.0

    def __call__(self, value: float) -> float:
        denom = self.denominator
        if denom <= 0:
            return 0.5
        prior_part = self.lambda0 * self.prior(value)
        trusted_part = self.n_trusted * self.trusted(value) if self.n_trusted else 0.0
        return float((prior_part + trusted_part) / denom)

    def smoothed(self, value: float) -> float:
        denom = self.denominator
        if denom <= 0:
            return 0.5
        prior_part = self.lambda0 * self.prior.smoothed(value)
        trusted_part = self.n_trusted * self.trusted.smoothed(value) if self.n_trusted else 0.0
        return float((prior_part + trusted_part) / denom)


def _load_distribution_from_config(*, path_value: str | None, inline_values: list[Any] | None, label: str) -> list[float]:
    if path_value:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"controller.{label} not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            import json

            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("raw_values") or data.get("values") or data.get("calibration_values") or []
            return _clean_floats(list(data))
        values: list[float] = []
        for line in text.splitlines():
            for piece in re.split(r"[,\s]+", line.strip()):
                if piece:
                    values.append(float(piece))
        return _clean_floats(values)
    return _clean_floats(list(inline_values or []))


def load_self_prior_distribution(cfg: ControllerConfig) -> list[float]:
    path_value = cfg.self_prior_distribution_path or cfg.prior_distribution_path
    inline_values = cfg.self_prior_distribution if cfg.self_prior_distribution is not None else cfg.prior_distribution
    return _load_distribution_from_config(
        path_value=path_value,
        inline_values=inline_values,
        label="self_prior_distribution_path",
    )


def _clean_floats(values: list[Any]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            cleaned.append(parsed)
    return cleaned


def _quantile(values: list[float], q: float) -> float | None:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    position = min(max(float(q), 0.0), 1.0) * (len(cleaned) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return cleaned[lower]
    weight = position - lower
    return cleaned[lower] * (1.0 - weight) + cleaned[upper] * weight


def _step_logprobs(output: StepOutput) -> list[float]:
    values = output.extra.get("generated_token_logprobs")
    if not isinstance(values, list):
        return []
    return _clean_floats(values)


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    if end_a < start_a or end_b < start_b:
        return False
    return start_a <= end_b and end_a >= start_b


def has_answer_intent(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _ANSWER_INTENT_PATTERNS)


class PDIController:
    def __init__(self, problem_id: str, cfg: ControllerConfig) -> None:
        self.problem_id = problem_id
        self.cfg = cfg
        self.self_prior_distribution = load_self_prior_distribution(cfg)
        self.lambda0_self = float(cfg.lambda0_self if cfg.lambda0_self is not None else cfg.lambda0)
        self.state = EpisodeState()
        self.steps: list[Step] = []
        self.windows: list[PDIWindow] = []
        self.events: list[dict[str, Any]] = []

        self._next_step_id = 1
        self._next_window_id = 1
        self._episode_id = 1
        self._repair_step_count = 0
        self._probation_windows: list[PDIWindow] = []
        self._last_no_valid_pdi_window: dict[str, Any] = {}

        self.driver_switch_count = 0
        self.llm_repair_episodes = 0
        self.rollback_count = 0
        self.handoff_attempt_count = 0
        self.handoff_success_count = 0
        self.handoff_failure_count = 0
        self.probation_failure_count = 0
        self.reentry_failure_count = 0
        self.self_reentry_attempt_count = 0
        self.self_reentry_accept_count = 0
        self.self_reentry_reject_count = 0
        self._self_reentry_pdi_sum = 0.0
        self._self_reentry_q_sum = 0.0
        self._self_reentry_reject_reasons: dict[str, int] = {}
        self._shadow_old_handoff_ready_count = 0
        self.early_stop_trigger_count = 0
        self.pdi_decision_count = 0
        self.no_valid_pdi_window_count = 0
        self.no_valid_pdi_window_reasons: dict[str, int] = {}
        self.answer_intent_seen = False

    @property
    def next_step_id(self) -> int:
        return self._next_step_id

    def validate_state(self) -> None:
        mode = self.state.mode
        owner = self.state.owner
        if mode in {MODE_COLD_START, MODE_SLM_NORMAL, MODE_SLM_TRANSITION, MODE_SLM_REENTRY, MODE_SLM_PROBATION} and owner != OWNER_SLM:
            raise InvalidControllerState(f"Invalid owner/mode pair: owner={owner}, mode={mode}")
        if mode in {MODE_LLM_REPAIR, MODE_LLM_FINALIZE} and owner != OWNER_LLM:
            raise InvalidControllerState(f"Invalid owner/mode pair: owner={owner}, mode={mode}")
        if mode == MODE_FINALIZE and owner not in {OWNER_SLM, OWNER_LLM}:
            raise InvalidControllerState(f"Invalid owner/mode pair: owner={owner}, mode={mode}")

    def append_step(self, output: StepOutput, *, owner: str, action: str, attempt_id: int | None = None) -> Step:
        start = self.visible_token_count()
        token_ids = list(output.token_ids)
        end = start + len(token_ids) - 1
        step = Step(
            step_id=self._next_step_id,
            owner=owner,
            mode=self.state.mode,
            text=output.text,
            token_ids=token_ids,
            logprobs=_step_logprobs(output),
            start_token_idx=start,
            end_token_idx=end,
            episode_id=self._episode_id,
            action=action,
            finish_reason=output.finish_reason,
            prompt_tokens=output.prompt_tokens,
            wall_time=output.wall_time,
            attempt_id=attempt_id,
            extra=dict(output.extra),
        )
        self.steps.append(step)
        self._next_step_id += 1
        if has_answer_intent(step.text):
            self.answer_intent_seen = True
        return step

    def active_steps(self) -> list[Step]:
        return [step for step in self.steps if step.active]

    def active_text(self) -> str:
        return "".join(step.text for step in self.active_steps())

    def visible_token_count(self) -> int:
        return sum(step.token_count for step in self.steps if step.active)

    def source_token_count(self, owner: str) -> int:
        return sum(step.token_count for step in self.steps if step.active and step.owner == owner)

    def source_step_count(self, owner: str) -> int:
        return sum(1 for step in self.steps if step.active and step.owner == owner)

    def build_step_window(self, active_steps: list[Step] | None = None) -> PDIWindow | None:
        self._last_no_valid_pdi_window = {}
        source_steps = active_steps if active_steps is not None else self.active_steps()
        candidates = [
            step
            for step in source_steps
            if step.active and step.owner == self.state.owner and step.episode_id == self._episode_id
        ]
        if not candidates:
            self._set_no_valid_pdi_window(
                reason="no_episode_steps",
                candidates=[],
                scored_count=0,
            )
            return None

        missing_logprob_steps = [step for step in candidates if step.scored_token_count <= 0]
        if missing_logprob_steps:
            self._set_no_valid_pdi_window(
                reason="missing_logprobs",
                candidates=candidates,
                scored_count=sum(step.scored_token_count for step in candidates),
                missing_step_ids=[step.step_id for step in missing_logprob_steps],
            )
            return None

        if self._has_current_episode_window():
            selected_steps: list[Step] = []
            scored_count = 0
            for step in reversed(candidates):
                selected_steps.append(step)
                scored_count += step.scored_token_count
                if scored_count >= self.cfg.t_min:
                    break
            selected_steps.reverse()
        else:
            selected_steps = list(candidates)
            scored_count = sum(step.scored_token_count for step in selected_steps)

        if scored_count < self.cfg.t_min:
            self._set_no_valid_pdi_window(
                reason="insufficient_scored_tokens",
                candidates=candidates,
                scored_count=scored_count,
            )
            return None

        owners = {step.owner for step in selected_steps}
        if len(owners) != 1:
            raise InvalidControllerState("PDI window crossed an ownership boundary.")

        logprob_sum = 0.0
        for step in selected_steps:
            logprob_sum += sum(step.logprobs[: step.scored_token_count])
        pdi = -logprob_sum / scored_count
        window = PDIWindow(
            window_id=self._next_window_id,
            owner=selected_steps[0].owner,
            covered_step_ids=[step.step_id for step in selected_steps],
            start_token_idx=selected_steps[0].start_token_idx,
            end_token_idx=selected_steps[-1].end_token_idx,
            pdi=float(pdi),
            status=WINDOW_SUSPECT,
            episode_id=self._episode_id,
            token_count=scored_count,
        )
        self._next_window_id += 1
        self.windows.append(window)
        return window

    def _has_current_episode_window(self) -> bool:
        return any(
            window.owner == self.state.owner
            and window.episode_id == self._episode_id
            and window.status not in {WINDOW_FAILURE, WINDOW_INVALID}
            for window in self.windows
        )

    def _set_no_valid_pdi_window(
        self,
        *,
        reason: str,
        candidates: list[Step],
        scored_count: int,
        missing_step_ids: list[int] | None = None,
    ) -> None:
        self._last_no_valid_pdi_window = {
            "reason": reason,
            "owner": self.state.owner,
            "mode": self.state.mode,
            "episode_id": self._episode_id,
            "candidate_step_ids": [step.step_id for step in candidates],
            "candidate_step_count": len(candidates),
            "scored_token_count": int(scored_count),
            "required_token_count": self.cfg.t_min,
            "missing_step_ids": list(missing_step_ids or []),
        }

    def effective_cdf(self, trusted_buffer: list[PDIWindow] | None = None) -> EffectiveCDF:
        windows = self.state.trusted_buffer if trusted_buffer is None else trusted_buffer
        return EffectiveCDF(
            prior=self.self_prior_distribution,
            trusted=[window.pdi for window in windows if window.status == WINDOW_TRUSTED],
            lambda0=self.lambda0_self,
        )

    def effective_cdf_from_values(self, trusted_values: list[float]) -> EffectiveCDF:
        return EffectiveCDF(
            prior=self.self_prior_distribution,
            trusted=list(trusted_values),
            lambda0=self.lambda0_self,
        )

    def percentile_rank(self, value: float, cdf: EffectiveCDF) -> float:
        return cdf(value)

    def upper_excess(self, q_percentile: float) -> float:
        q_high = float(self.cfg.q_high)
        return max(0.0, float(q_percentile) - q_high) / max(1e-12, 1.0 - q_high)

    def update_upper_evidence(self, window: PDIWindow, q_percentile: float) -> tuple[float, float]:
        upper_excess = self.upper_excess(q_percentile)
        recent_before = self.state.upper_evidence_history[-(self.cfg.r_upper - 1) :] if self.cfg.r_upper > 1 else []
        had_active_upper_region = any(point.upper_excess > 0 for point in recent_before)
        if upper_excess > 0 and not had_active_upper_region:
            self.state.pre_suspect_snapshot = [w.pdi for w in self.state.trusted_buffer]

        point = UpperEvidencePoint(
            window=window,
            q_percentile=q_percentile,
            upper_excess=upper_excess,
            upper_evidence=0.0,
        )
        self.state.upper_evidence_history.append(point)
        self.state.upper_evidence_history = self.state.upper_evidence_history[-self.cfg.r_upper :]
        upper_evidence = sum(item.upper_excess for item in self.state.upper_evidence_history) / self.cfg.r_upper
        point.upper_evidence = upper_evidence
        window.q_percentile = q_percentile
        window.upper_excess = upper_excess
        window.upper_evidence = upper_evidence
        return upper_excess, upper_evidence

    def alarm_start_window(self) -> PDIWindow | None:
        positives = [point.window for point in self.state.upper_evidence_history if point.upper_excess > 0]
        return positives[0] if positives else None

    def _annotate_window(self, window: PDIWindow, q_percentile: float) -> float:
        upper_excess = self.upper_excess(q_percentile)
        window.q_percentile = q_percentile
        window.upper_excess = upper_excess
        window.upper_evidence = None
        return upper_excess

    def _mode_after_trusted_update(self) -> str:
        if len(self.state.trusted_buffer) >= self.cfg.n_min:
            return MODE_SLM_NORMAL
        return MODE_COLD_START

    def _add_trusted_window(self, window: PDIWindow) -> None:
        window.status = WINDOW_TRUSTED
        if window not in self.state.trusted_buffer:
            self.state.trusted_buffer.append(window)

    def _clear_transition_watch(self) -> None:
        self.state.transition_start_window = None
        self.state.transition_windows = []

    def _clear_reentry_watch(self) -> None:
        self.state.reentry_transition_windows = []

    def _overlaps_any_window(self, window: PDIWindow, others: list[PDIWindow]) -> bool:
        return any(window.overlaps(other.start_token_idx, other.end_token_idx) for other in others)

    def _enter_transition_watch(self, window: PDIWindow, *, step_id: int) -> None:
        window.status = WINDOW_SUSPECT
        self.state.pre_suspect_snapshot = [w.pdi for w in self.state.trusted_buffer]
        self.state.lower_tail_history = []
        self.state.transition_start_window = window
        self.state.transition_windows = [window]
        self._switch(MODE_SLM_TRANSITION, OWNER_SLM, step_id=step_id, reason="slm_upper_tail_transition")

    def _transition_failed(self) -> bool:
        windows = self.state.transition_windows
        if len(windows) < self.cfg.transition_grace_windows:
            return False
        return all((window.q_percentile or 0.0) > self.cfg.q_recover for window in windows)

    def process_slm_window(self, step: Step) -> ControllerDecision:
        mode_at_decision = self.state.mode
        window = self.build_step_window()
        if window is None:
            self._log_no_valid_pdi_window(step=step, mode=mode_at_decision)
            return ControllerDecision(action="NO_VALID_PDI_WINDOW", step=step)

        cdf = self.effective_cdf()
        q_percentile = self.percentile_rank(window.pdi, cdf)
        upper_excess = self._annotate_window(window, q_percentile)
        self.pdi_decision_count += 1

        if q_percentile > self.cfg.q_high:
            self._enter_transition_watch(window, step_id=step.step_id)
            action = "ENTER_SLM_TRANSITION"
            self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
            return ControllerDecision(
                action=action,
                step=step,
                window=window,
                q_percentile=q_percentile,
                upper_excess=upper_excess,
            )

        self._add_trusted_window(window)
        action = "TRUST_PDI_WINDOW"

        next_mode = self._mode_after_trusted_update()
        if self.state.mode != next_mode:
            self._switch(next_mode, OWNER_SLM, step_id=step.step_id, reason="trusted_buffer_update")

        early_stop = False
        if self.state.mode != MODE_COLD_START:
            lower_hit = q_percentile <= self.cfg.q_low
            self.state.lower_tail_history.append(lower_hit)
            self.state.lower_tail_history = self.state.lower_tail_history[-self.cfg.r_low :]
            early_stop = (
                self.answer_intent_seen
                and len(self.state.trusted_buffer) >= self.cfg.n_min
                and len(self.state.lower_tail_history) >= self.cfg.r_low
                and all(self.state.lower_tail_history[-self.cfg.r_low :])
            )

        if early_stop:
            self.early_stop_trigger_count += 1
            action = "EARLY_STOP_LOWER_TAIL_AFTER_ANSWER_INTENT"
            self._switch(MODE_FINALIZE, self.state.owner, step_id=step.step_id, reason=action)

        self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
        return ControllerDecision(
            action=action,
            step=step,
            window=window,
            q_percentile=q_percentile,
            upper_excess=upper_excess,
        )

    def process_transition_window(self, step: Step) -> ControllerDecision:
        mode_at_decision = self.state.mode
        window = self.build_step_window()
        if window is None:
            self._log_no_valid_pdi_window(step=step, mode=mode_at_decision)
            return ControllerDecision(action="NO_VALID_PDI_WINDOW", step=step)

        cdf = self.effective_cdf()
        q_percentile = self.percentile_rank(window.pdi, cdf)
        upper_excess = self._annotate_window(window, q_percentile)
        self.pdi_decision_count += 1
        window.status = WINDOW_SUSPECT
        self.state.transition_windows.append(window)

        if q_percentile <= self.cfg.q_recover:
            transition_windows = list(self.state.transition_windows)
            if not self._overlaps_any_window(window, transition_windows[:-1]):
                self._add_trusted_window(window)
            self._clear_transition_watch()
            next_mode = self._mode_after_trusted_update()
            self._switch(next_mode, OWNER_SLM, step_id=step.step_id, reason="slm_transition_recovered")
            action = "SLM_TRANSITION_RECOVERED"
            self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
            return ControllerDecision(
                action=action,
                step=step,
                window=window,
                q_percentile=q_percentile,
                upper_excess=upper_excess,
            )

        if self._transition_failed():
            alarm = self.state.transition_start_window
            if alarm is None:
                raise InvalidControllerState("SLM transition failed without a transition start window.")
            rollback_start = alarm.start_token_idx
            action = "SLM_TRANSITION_FAILED_ROLLBACK_TO_LLM_REPAIR"
            self._log_pdi_decision(
                step=step,
                window=window,
                mode=mode_at_decision,
                action=action,
                cdf=cdf,
                rollback_start_token_idx=rollback_start,
            )
            self.rollback_to_token(rollback_start, reason="slm_transition_failed")
            return ControllerDecision(
                action=action,
                step=step,
                window=window,
                q_percentile=q_percentile,
                upper_excess=upper_excess,
                rollback_start_token_idx=rollback_start,
            )

        action = "SLM_TRANSITION_CONTINUE"
        self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
        return ControllerDecision(
            action=action,
            step=step,
            window=window,
            q_percentile=q_percentile,
            upper_excess=upper_excess,
        )

    def process_reentry_window(self, step: Step) -> ControllerDecision:
        mode_at_decision = self.state.mode
        window = self.build_step_window()
        if window is None:
            self._log_no_valid_pdi_window(step=step, mode=mode_at_decision)
            return ControllerDecision(action="NO_VALID_PDI_WINDOW", step=step)

        cdf = self.effective_cdf()
        q_percentile = self.percentile_rank(window.pdi, cdf)
        upper_excess = self._annotate_window(window, q_percentile)
        self.pdi_decision_count += 1
        window.status = WINDOW_SUSPECT

        if q_percentile > self.cfg.q_high:
            if not self.state.reentry_transition_windows:
                self.state.reentry_cached_windows = []
                self.state.reentry_stable_count = 0
            self.state.reentry_transition_windows.append(window)
            action = "SLM_REENTRY_TRANSITION_CONTINUE"
            if len(self.state.reentry_transition_windows) >= self.cfg.reentry_transition_grace:
                self.reentry_failure_count += 1
                self.probation_failure_count += 1
                rollback_start = self.state.handoff_point_token_idx
                action = "SLM_REENTRY_FAILED_ROLLBACK_TO_LLM_REPAIR"
                self._log_pdi_decision(
                    step=step,
                    window=window,
                    mode=mode_at_decision,
                    action=action,
                    cdf=cdf,
                    rollback_start_token_idx=rollback_start,
                    probation_status="failed",
                )
                if rollback_start is None:
                    raise InvalidControllerState("SLM re-entry failed without a handoff point.")
                self.rollback_to_token(rollback_start, reason="reentry_transition_failure")
                self.handoff_failure_count += 1
                return ControllerDecision(
                    action=action,
                    step=step,
                    window=window,
                    q_percentile=q_percentile,
                    upper_excess=upper_excess,
                    rollback_start_token_idx=rollback_start,
                    probation_status="failed",
                )

            self._log_pdi_decision(
                step=step,
                window=window,
                mode=mode_at_decision,
                action=action,
                cdf=cdf,
                probation_status="transition",
            )
            return ControllerDecision(
                action=action,
                step=step,
                window=window,
                q_percentile=q_percentile,
                upper_excess=upper_excess,
                probation_status="transition",
            )

        transition_windows = list(self.state.reentry_transition_windows)
        self._clear_reentry_watch()
        if self._overlaps_any_window(window, transition_windows):
            action = "SLM_REENTRY_RECOVERED"
            probation_status = "recovered"
        else:
            self.state.reentry_cached_windows.append(window)
            self.state.reentry_stable_count += 1
            action = "SLM_REENTRY_CONTINUE"
            probation_status = "stable"

        if self.state.reentry_stable_count >= self.cfg.m_reentry:
            for reentry_window in self.state.reentry_cached_windows:
                self._add_trusted_window(reentry_window)
            self.state.reentry_cached_windows = []
            self.state.reentry_stable_count = 0
            self._clear_reentry_watch()
            self._reset_evidence_histories()
            self._switch(MODE_SLM_NORMAL, OWNER_SLM, step_id=step.step_id, reason="slm_reentry_stable")
            action = "SLM_REENTRY_STABLE"
            probation_status = "passed"

        self._log_pdi_decision(
            step=step,
            window=window,
            mode=mode_at_decision,
            action=action,
            cdf=cdf,
            probation_status=probation_status,
        )
        return ControllerDecision(
            action=action,
            step=step,
            window=window,
            q_percentile=q_percentile,
            upper_excess=upper_excess,
            probation_status=probation_status,
        )

    def process_probation_window(self, step: Step) -> ControllerDecision:
        return self.process_reentry_window(step)

    def repair_step_for_handoff(self, step: Step) -> tuple[str, str, list[Step]] | None:
        if not step.active or step.owner != OWNER_LLM or step.episode_id != self._episode_id:
            return None
        prefix_text = "".join(active.text for active in self.active_steps() if active.step_id < step.step_id)
        return prefix_text, step.text, [step]

    def _repair_landing_anchor(self, reference_values: list[float], prior_values: list[float]) -> float | None:
        values = reference_values if reference_values else prior_values
        return _quantile(values, REPAIR_LANDING_ANCHOR_Q)

    def _repair_high_reference(self, previous_scores: list[float]) -> float | None:
        return _quantile(previous_scores, REPAIR_LANDING_HIGH_Q)

    def process_handoff_score(self, *, step: Step, slm_side_pdi: float) -> ControllerDecision:
        self.handoff_attempt_count += 1
        old_cdf = self.effective_cdf_from_values(self.state.pre_suspect_snapshot)
        q_slm_side = self.percentile_rank(slm_side_pdi, old_cdf)
        q_slm_side_smoothed = old_cdf.smoothed(slm_side_pdi)
        reference_values = old_cdf.trusted.to_list()
        prior_values = old_cdf.prior.to_list()
        previous_scores = list(self.state.repair_slm_side_scores)
        repair_anchor = self._repair_landing_anchor(reference_values, prior_values)
        repair_high = self._repair_high_reference(previous_scores)
        repair_best_before = min(previous_scores) if previous_scores else None
        direct_anchor_hit = repair_anchor is not None and slm_side_pdi <= repair_anchor
        episode_best = repair_best_before is None or slm_side_pdi <= repair_best_before
        rli_raw: float | None = None
        rli: float | None = None
        if repair_high is not None and repair_anchor is not None:
            denominator = repair_high - repair_anchor
            if denominator > REPAIR_LANDING_EPS:
                rli_raw = (repair_high - slm_side_pdi) / denominator
                rli = max(0.0, min(1.0, rli_raw))
        acceptable = direct_anchor_hit or (
            episode_best and rli is not None and rli >= REPAIR_LANDING_RLI_THRESHOLD
        )
        self.state.repair_slm_side_scores.append(slm_side_pdi)
        self.state.repair_slm_side_step_ids.append(step.step_id)
        if acceptable:
            self.state.handoff_history.append(rli if rli is not None else 1.0)
            self.state.handoff_candidate_scores.append(slm_side_pdi)
            self.state.handoff_candidate_step_ids.append(step.step_id)
        else:
            self.state.handoff_history = []
            self.state.handoff_candidate_scores = []
            self.state.handoff_candidate_step_ids = []
        ready = acceptable and len(self.state.handoff_candidate_step_ids) >= self.cfg.r_handoff
        event = {
            "problem_id": self.problem_id,
            "event": "handoff_readiness",
            "step_id": step.step_id,
            "owner": self.state.owner,
            "mode": self.state.mode,
            "handoff_strategy": "repair_landing_index",
            "slm_side_pdi": slm_side_pdi,
            "slm_side_pdi_raw": slm_side_pdi,
            "handoff_q_slm_side": q_slm_side,
            "slm_side_q": q_slm_side,
            "slm_side_q_smoothed": q_slm_side_smoothed,
            "repair_landing_index": rli,
            "repair_landing_index_raw": rli_raw,
            "repair_landing_threshold": REPAIR_LANDING_RLI_THRESHOLD,
            "repair_landing_anchor_q": REPAIR_LANDING_ANCHOR_Q,
            "repair_landing_high_q": REPAIR_LANDING_HIGH_Q,
            "repair_landing_anchor_pdi": repair_anchor,
            "repair_landing_high_ref": repair_high,
            "repair_landing_best_pdi_before": repair_best_before,
            "repair_landing_direct_anchor_hit": direct_anchor_hit,
            "repair_landing_episode_best": episode_best,
            "repair_score_count": len(self.state.repair_slm_side_scores),
            "repair_previous_score_count": len(previous_scores),
            "repair_previous_p25": _quantile(previous_scores, 0.25),
            "repair_previous_p50": _quantile(previous_scores, 0.50),
            "repair_previous_p75": _quantile(previous_scores, 0.75),
            "repair_previous_min": min(previous_scores) if previous_scores else None,
            "repair_previous_max": max(previous_scores) if previous_scores else None,
            "r_handoff": self.cfg.r_handoff,
            "handoff_acceptable": acceptable,
            "handoff_candidate_step_ids": list(self.state.handoff_candidate_step_ids),
            "handoff_candidate_scores": list(self.state.handoff_candidate_scores),
            "handoff_candidate_count": len(self.state.handoff_candidate_step_ids),
            "ready": ready,
            "trusted_buffer_size": len(self.state.trusted_buffer),
            "slm_side_reference_count": len(reference_values),
            "slm_side_reference_p90": _quantile(reference_values, 0.90),
            "slm_side_reference_p95": _quantile(reference_values, 0.95),
            "slm_side_reference_max": max(reference_values) if reference_values else None,
            "slm_side_prior_count": len(prior_values),
            "slm_side_prior_p90": _quantile(prior_values, 0.90),
            "slm_side_prior_p95": _quantile(prior_values, 0.95),
            "slm_side_prior_max": max(prior_values) if prior_values else None,
            "prior_weight": old_cdf.prior_weight,
        }
        self.events.append(event)

        if ready:
            self.handoff_success_count += 1
            self.state.handoff_point_token_idx = self.visible_token_count()
            self._reset_evidence_histories()
            self.state.probation_stable_count = 0
            self.state.reentry_stable_count = 0
            self.state.reentry_cached_windows = []
            self.state.reentry_transition_windows = []
            self._probation_windows = []
            self._new_episode()
            self._switch(MODE_SLM_REENTRY, OWNER_SLM, step_id=step.step_id, reason="repair_landing_ready")
            return ControllerDecision(
                action="HANDOFF_TO_SLM_REENTRY",
                step=step,
                handoff_q_slm_side=q_slm_side,
                repair_landing_index=rli,
            )

        return ControllerDecision(
            action="LLM_REPAIR_CONTINUE",
            step=step,
            handoff_q_slm_side=q_slm_side,
            repair_landing_index=rli,
        )

    def process_self_reentry_candidate(
        self,
        *,
        llm_repair_step: Step,
        candidate_outputs: list[StepOutput],
        old_slm_side_pdi: float | None = None,
        old_slm_side_q: float | None = None,
    ) -> ControllerDecision:
        self.self_reentry_attempt_count += 1

        required_scored_tokens = max(
            self.cfg.t_min,
            self.cfg.self_reentry_min_scored_tokens or self.cfg.t_min,
        )
        candidate_token_counts = [int(output.token_count) for output in candidate_outputs]
        candidate_scored_counts = [
            min(len(output.token_ids), len(_step_logprobs(output))) for output in candidate_outputs
        ]
        scored_count = sum(candidate_scored_counts)
        saw_close_think = any("</think>" in (output.text or "") for output in candidate_outputs)

        reject_reason: str | None = None
        if not candidate_outputs:
            reject_reason = "empty_self_reentry_candidate"
        elif saw_close_think:
            reject_reason = "close_think_in_self_reentry_candidate"
        elif scored_count < required_scored_tokens:
            reject_reason = "insufficient_self_reentry_tokens"

        micro_pdi: float | None = None
        q_self_reentry: float | None = None

        if reject_reason is None:
            logprob_sum = 0.0
            for output in candidate_outputs:
                output_logprobs = _step_logprobs(output)
                output_scored_count = min(len(output.token_ids), len(output_logprobs))
                logprob_sum += sum(output_logprobs[:output_scored_count])
            micro_pdi = -logprob_sum / scored_count
            ref_values = self.state.pre_suspect_snapshot or [w.pdi for w in self.state.trusted_buffer]
            cdf = self.effective_cdf_from_values(ref_values)
            q_self_reentry = cdf(micro_pdi)
            if q_self_reentry > self.cfg.self_reentry_q_threshold:
                reject_reason = "q_above_threshold"

        accepted = reject_reason is None

        # shadow old handoff readiness for comparison
        old_handoff_ready: bool | None = None
        if old_slm_side_pdi is not None:
            old_cdf = self.effective_cdf_from_values(
                self.state.pre_suspect_snapshot or [w.pdi for w in self.state.trusted_buffer]
            )
            old_q = old_slm_side_q if old_slm_side_q is not None else old_cdf(old_slm_side_pdi)
            old_slm_side_q = old_q
            reference_values = old_cdf.trusted.to_list()
            prior_values = old_cdf.prior.to_list()
            previous_scores = list(self.state.repair_slm_side_scores)
            repair_anchor = self._repair_landing_anchor(reference_values, prior_values)
            repair_high = self._repair_high_reference(previous_scores)
            repair_best_before = min(previous_scores) if previous_scores else None
            direct_anchor_hit = repair_anchor is not None and old_slm_side_pdi <= repair_anchor
            episode_best = repair_best_before is None or old_slm_side_pdi <= repair_best_before
            rli: float | None = None
            if repair_high is not None and repair_anchor is not None:
                denominator = repair_high - repair_anchor
                if denominator > REPAIR_LANDING_EPS:
                    rli_raw = (repair_high - old_slm_side_pdi) / denominator
                    rli = max(0.0, min(1.0, rli_raw))
            acceptable = direct_anchor_hit or (
                episode_best and rli is not None and rli >= REPAIR_LANDING_RLI_THRESHOLD
            )
            self.state.repair_slm_side_scores.append(old_slm_side_pdi)
            self.state.repair_slm_side_step_ids.append(llm_repair_step.step_id)
            if acceptable:
                self.state.handoff_candidate_scores.append(old_slm_side_pdi)
                self.state.handoff_candidate_step_ids.append(llm_repair_step.step_id)
            else:
                self.state.handoff_candidate_scores = []
                self.state.handoff_candidate_step_ids = []
            old_handoff_ready = acceptable and len(self.state.handoff_candidate_step_ids) >= self.cfg.r_handoff
            if old_handoff_ready:
                self._shadow_old_handoff_ready_count += 1

        candidate_step_ids: list[int] = []
        if accepted and self.cfg.commit_self_reentry_step:
            handoff_point_token_idx = self.visible_token_count()
            self.state.handoff_point_token_idx = handoff_point_token_idx
            self._reset_evidence_histories()
            self.state.probation_stable_count = 0
            self.state.reentry_stable_count = 0
            self.state.reentry_cached_windows = []
            self.state.reentry_transition_windows = []
            self._probation_windows = []
            self._new_episode()
            self._switch(
                MODE_SLM_REENTRY,
                OWNER_SLM,
                step_id=llm_repair_step.step_id,
                reason="self_reentry_certification_accepted",
            )
            for idx, output in enumerate(candidate_outputs, start=1):
                token_ids = list(output.token_ids)
                logprobs = _step_logprobs(output)
                start = self.visible_token_count()
                end = start + len(token_ids) - 1
                micro_step = Step(
                    step_id=self._next_step_id,
                    owner=OWNER_SLM,
                    mode=self.state.mode,
                    text=output.text,
                    token_ids=token_ids,
                    logprobs=logprobs,
                    start_token_idx=start,
                    end_token_idx=end,
                    episode_id=self._episode_id,
                    action="SELF_REENTRY_ACCEPTED",
                    finish_reason=output.finish_reason,
                    prompt_tokens=output.prompt_tokens,
                    wall_time=output.wall_time,
                    extra={
                        **dict(output.extra),
                        "self_reentry_certification": True,
                        "self_reentry_candidate_index": idx,
                    },
                )
                self.steps.append(micro_step)
                self._next_step_id += 1
                candidate_step_ids.append(micro_step.step_id)
                if has_answer_intent(micro_step.text):
                    self.answer_intent_seen = True

        if accepted:
            self.self_reentry_accept_count += 1
            if micro_pdi is not None:
                self._self_reentry_pdi_sum += micro_pdi
            if q_self_reentry is not None:
                self._self_reentry_q_sum += q_self_reentry
            self.handoff_success_count += 1
            if not self.cfg.commit_self_reentry_step:
                self.state.handoff_point_token_idx = self.visible_token_count()
                self._reset_evidence_histories()
                self.state.probation_stable_count = 0
                self.state.reentry_stable_count = 0
                self.state.reentry_cached_windows = []
                self.state.reentry_transition_windows = []
                self._probation_windows = []
                self._new_episode()
                self._switch(
                    MODE_SLM_REENTRY,
                    OWNER_SLM,
                    step_id=llm_repair_step.step_id,
                    reason="self_reentry_certification_accepted",
                )
        else:
            self.self_reentry_reject_count += 1
            self._self_reentry_reject_reasons[reject_reason] = self._self_reentry_reject_reasons.get(reject_reason, 0) + 1

        event: dict[str, Any] = {
            "problem_id": self.problem_id,
            "event": "self_reentry_certification",
            "step_id": llm_repair_step.step_id,
            "candidate_step_id": candidate_step_ids[0] if candidate_step_ids else None,
            "candidate_step_ids": list(candidate_step_ids),
            "candidate_attempt_count": len(candidate_outputs),
            "candidate_token_counts": candidate_token_counts,
            "candidate_scored_token_counts": candidate_scored_counts,
            "micro_token_count": scored_count,
            "required_scored_tokens": required_scored_tokens,
            "self_reentry_pdi": micro_pdi,
            "self_reentry_q": q_self_reentry,
            "self_reentry_threshold": self.cfg.self_reentry_q_threshold,
            "accepted": accepted,
            "reject_reason": reject_reason,
            "commit": accepted and self.cfg.commit_self_reentry_step,
            "old_llm_suffix_pdi": old_slm_side_pdi,
            "old_llm_suffix_q": old_slm_side_q,
            "old_handoff_ready_shadow": old_handoff_ready,
        }
        self.events.append(event)
        self.handoff_attempt_count += 1

        if accepted:
            return ControllerDecision(action="HANDOFF_TO_SLM_REENTRY", step=llm_repair_step)
        return ControllerDecision(action="LLM_REPAIR_CONTINUE", step=llm_repair_step)

    def reset_handoff_candidate_buffer(self, *, step: Step, reason: str) -> None:
        self.state.handoff_history = []
        self.state.handoff_candidate_step_ids = []
        self.state.handoff_candidate_scores = []
        self.events.append(
            {
                "problem_id": self.problem_id,
                "event": "handoff_candidate_reset",
                "step_id": step.step_id,
                "owner": self.state.owner,
                "mode": self.state.mode,
                "reason": reason,
                "ready": False,
            }
        )

    def note_llm_repair_step(self, step: Step) -> ControllerDecision:
        self._repair_step_count += 1
        if self._repair_step_count >= self.cfg.max_llm_repair_steps:
            self.handoff_failure_count += 1
            self._switch(MODE_LLM_FINALIZE, OWNER_LLM, step_id=step.step_id, reason="max_llm_repair_steps")
            return ControllerDecision(action="LLM_FINALIZE", step=step)
        return ControllerDecision(action="LLM_REPAIR_CONTINUE", step=step)

    def rollback_to_window(self, alarm_start_window: PDIWindow) -> None:
        self.rollback_to_token(alarm_start_window.start_token_idx, reason="upper_tail_alarm")

    def rollback_to_token(self, rollback_start_token_idx: int, *, reason: str) -> None:
        end = self.visible_token_count() - 1
        if end < rollback_start_token_idx:
            end = rollback_start_token_idx
        self.rollback_count += 1

        for step in self.steps:
            if step.active and spans_overlap(step.start_token_idx, step.end_token_idx, rollback_start_token_idx, end):
                step.active = False
                step.action = f"REMOVED_BY_{reason.upper()}"

        kept_trusted: list[PDIWindow] = []
        for window in self.state.trusted_buffer:
            if window.overlaps(rollback_start_token_idx, end):
                window.status = WINDOW_INVALID
                self.state.failure_buffer.append(window)
            else:
                kept_trusted.append(window)
        self.state.trusted_buffer = kept_trusted

        for window in self.windows:
            if window.status == WINDOW_SUSPECT and window.overlaps(rollback_start_token_idx, end):
                window.status = WINDOW_FAILURE
                if window not in self.state.failure_buffer:
                    self.state.failure_buffer.append(window)

        self._reset_evidence_histories()
        self._clear_transition_watch()
        self._clear_reentry_watch()
        self._probation_windows = []
        self.state.probation_stable_count = 0
        self.state.reentry_cached_windows = []
        self.state.reentry_stable_count = 0
        self.state.handoff_point_token_idx = None
        self._new_episode()
        self.llm_repair_episodes += 1
        self._repair_step_count = 0
        self._switch(MODE_LLM_REPAIR, OWNER_LLM, step_id=None, reason=reason)
        self.events.append(
            {
                "problem_id": self.problem_id,
                "event": "rollback",
                "reason": reason,
                "rollback_start_token_idx": rollback_start_token_idx,
                "rollback_end_token_idx": end,
                "trusted_buffer_size": len(self.state.trusted_buffer),
                "failure_buffer_size": len(self.state.failure_buffer),
            }
        )

    def mark_finished(self, step: Step, *, reason: str) -> None:
        step.action = "FINISHED" if reason == "finished" else f"STOP_{reason.upper()}"
        self._switch(MODE_FINALIZE, self.state.owner, step_id=step.step_id, reason=reason)

    def force_finalize(self, *, owner: str | None = None, reason: str) -> None:
        self._switch(MODE_FINALIZE, owner or self.state.owner, step_id=None, reason=reason)

    def serialize_steps(self) -> list[dict[str, Any]]:
        return [step.to_dict(self.problem_id) for step in self.steps]

    def serialize_windows(self) -> list[dict[str, Any]]:
        return [asdict(window) for window in self.windows]

    def summary(
        self,
        *,
        finish_reason: str,
        final_answer: str | None,
        final_answer_generator: str | None,
        total_wall_time: float,
        slm_wall_time: float,
        llm_wall_time: float,
        slm_scoring_wall_time: float,
        slm_scoring_count: int,
        slm_prefill_count: int,
        llm_prefill_count: int,
    ) -> dict[str, Any]:
        slm_tokens = self.source_token_count(OWNER_SLM)
        llm_tokens = self.source_token_count(OWNER_LLM)
        total_tokens = slm_tokens + llm_tokens
        handoff_rate = (
            self.handoff_success_count / self.handoff_attempt_count if self.handoff_attempt_count else 0.0
        )
        return {
            "problem_id": self.problem_id,
            "finish_reason": finish_reason,
            "final_answer": final_answer,
            "final_answer_generator": final_answer_generator,
            "controller_mode": "pdi_step_window",
            "owner": self.state.owner,
            "mode": self.state.mode,
            "driver_switch_count": self.driver_switch_count,
            "llm_ownership_episodes": self.llm_repair_episodes,
            "llm_repair_episodes": self.llm_repair_episodes,
            "rollback_count": self.rollback_count,
            "handoff_attempt_count": self.handoff_attempt_count,
            "handoff_success_count": self.handoff_success_count,
            "handoff_failure_count": self.handoff_failure_count,
            "handoff_success_rate": handoff_rate,
            "probation_failure_count": self.probation_failure_count,
            "probation_failure_rate": (
                self.probation_failure_count / self.handoff_success_count if self.handoff_success_count else 0.0
            ),
            "reentry_failure_count": self.reentry_failure_count,
            "reentry_failure_rate": (
                self.reentry_failure_count / self.handoff_success_count if self.handoff_success_count else 0.0
            ),
            "early_stop_trigger_count": self.early_stop_trigger_count,
            "pdi_decision_count": self.pdi_decision_count,
            "no_valid_pdi_window_count": self.no_valid_pdi_window_count,
            "no_valid_pdi_window_reasons": dict(self.no_valid_pdi_window_reasons),
            "pdi_window_count": len(self.windows),
            "trusted_buffer_size": len(self.state.trusted_buffer),
            "failure_buffer_size": len(self.state.failure_buffer),
            "prior_size": len(self.self_prior_distribution),
            "self_prior_size": len(self.self_prior_distribution),
            "prior_weight": self.effective_cdf().prior_weight,
            "slm_thinking_tokens": slm_tokens,
            "llm_thinking_tokens": llm_tokens,
            "total_thinking_tokens": total_tokens,
            "llm_participation_rate": (llm_tokens / total_tokens) if total_tokens else 0.0,
            "slm_step_count": self.source_step_count(OWNER_SLM),
            "llm_step_count": self.source_step_count(OWNER_LLM),
            "slm_prefill_count": slm_prefill_count,
            "llm_prefill_count": llm_prefill_count,
            "total_wall_time": total_wall_time,
            "slm_wall_time": slm_wall_time,
            "llm_wall_time": llm_wall_time,
            "slm_scoring_overhead": slm_scoring_wall_time,
            "slm_scoring_count": slm_scoring_count,
            "self_reentry_attempt_count": self.self_reentry_attempt_count,
            "self_reentry_accept_count": self.self_reentry_accept_count,
            "self_reentry_reject_count": self.self_reentry_reject_count,
            "self_reentry_accept_rate": (
                self.self_reentry_accept_count / self.self_reentry_attempt_count
                if self.self_reentry_attempt_count else 0.0
            ),
            "avg_self_reentry_pdi": (
                self._self_reentry_pdi_sum / self.self_reentry_accept_count
                if self.self_reentry_accept_count else None
            ),
            "avg_self_reentry_q": (
                self._self_reentry_q_sum / self.self_reentry_accept_count
                if self.self_reentry_accept_count else None
            ),
            "self_reentry_reject_reasons": dict(self._self_reentry_reject_reasons),
            "shadow_old_handoff_ready_count": self._shadow_old_handoff_ready_count,
            "config": {
                "t_min": self.cfg.t_min,
                "lambda0": self.cfg.lambda0,
                "lambda0_self": self.lambda0_self,
                "n_min": self.cfg.n_min,
                "q_high": self.cfg.q_high,
                "q_recover": self.cfg.q_recover,
                "transition_grace_windows": self.cfg.transition_grace_windows,
                "r_upper": self.cfg.r_upper,
                "eta_upper": self.cfg.eta_upper,
                "handoff_strategy": self.cfg.handoff_strategy,
                "repair_landing_anchor_q": REPAIR_LANDING_ANCHOR_Q,
                "repair_landing_high_q": REPAIR_LANDING_HIGH_Q,
                "repair_landing_rli_threshold": REPAIR_LANDING_RLI_THRESHOLD,
                "r_handoff": self.cfg.r_handoff,
                "m_probation": self.cfg.m_probation,
                "m_reentry": self.cfg.m_reentry,
                "reentry_transition_grace": self.cfg.reentry_transition_grace,
                "q_low": self.cfg.q_low,
                "r_low": self.cfg.r_low,
                "max_llm_repair_steps": self.cfg.max_llm_repair_steps,
                "self_reentry_min_scored_tokens": max(
                    self.cfg.t_min,
                    self.cfg.self_reentry_min_scored_tokens or self.cfg.t_min,
                ),
                "self_reentry_max_attempt_steps": self.cfg.self_reentry_max_attempt_steps,
                "self_reentry_q_threshold": self.cfg.self_reentry_q_threshold,
                "commit_self_reentry_step": self.cfg.commit_self_reentry_step,
            },
        }

    def _switch(self, mode: str, owner: str, *, step_id: int | None, reason: str) -> None:
        old_mode = self.state.mode
        old_owner = self.state.owner
        if old_mode == mode and old_owner == owner:
            return
        self.state.mode = mode
        self.state.owner = owner
        self.driver_switch_count += 1
        self.events.append(
            {
                "problem_id": self.problem_id,
                "event": "driver_switch",
                "step_id": step_id,
                "from_mode": old_mode,
                "to_mode": mode,
                "from_owner": old_owner,
                "to_owner": owner,
                "reason": reason,
            }
        )

    def _new_episode(self) -> None:
        self._episode_id += 1

    def _reset_evidence_histories(self) -> None:
        self.state.upper_evidence_history = []
        self.state.lower_tail_history = []
        self.state.handoff_history = []
        self.state.handoff_candidate_step_ids = []
        self.state.handoff_candidate_scores = []
        self.state.repair_slm_side_scores = []
        self.state.repair_slm_side_step_ids = []

    def _log_pdi_decision(
        self,
        *,
        step: Step,
        window: PDIWindow,
        mode: str,
        action: str,
        cdf: EffectiveCDF,
        rollback_start_token_idx: int | None = None,
        handoff_q_slm_side: float | None = None,
        probation_status: str | None = None,
    ) -> None:
        self.events.append(
            {
                "problem_id": self.problem_id,
                "event": "pdi_decision",
                "step_id": step.step_id,
                "window_id": window.window_id,
                "owner": window.owner,
                "mode": mode,
                "start_token_idx": window.start_token_idx,
                "end_token_idx": window.end_token_idx,
                "pdi": window.pdi,
                "q_percentile": window.q_percentile,
                "upper_excess": window.upper_excess,
                "upper_evidence": window.upper_evidence,
                "lower_tail_count": sum(1 for item in self.state.lower_tail_history[-self.cfg.r_low :] if item),
                "answer_intent_seen": self.answer_intent_seen,
                "action": action,
                "trusted_buffer_size": len(self.state.trusted_buffer),
                "prior_weight": cdf.prior_weight,
                "rollback_start_token_idx": rollback_start_token_idx,
                "handoff_q_slm_side": handoff_q_slm_side,
                "probation_status": probation_status,
            }
        )

    def _log_no_valid_pdi_window(self, *, step: Step, mode: str) -> None:
        detail = dict(self._last_no_valid_pdi_window)
        reason = str(detail.get("reason") or "unknown")
        self.no_valid_pdi_window_count += 1
        self.no_valid_pdi_window_reasons[reason] = self.no_valid_pdi_window_reasons.get(reason, 0) + 1
        self.events.append(
            {
                "problem_id": self.problem_id,
                "event": "no_valid_pdi_window",
                "step_id": step.step_id,
                "owner": self.state.owner,
                "mode": mode,
                "reason": reason,
                "action": "NO_VALID_PDI_WINDOW",
                "trusted_buffer_size": len(self.state.trusted_buffer),
                **detail,
            }
        )
