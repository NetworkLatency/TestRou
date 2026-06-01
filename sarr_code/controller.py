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
MODE_SLM_REENTRY = "SLM_REENTRY"
MODE_FINALIZE = "FINALIZE"
MODE_LLM_FINALIZE = "LLM_FINALIZE"

MSM_STABLE = "stable"
MSM_TRANSITION_RISK = "transition-risk"
MSM_LLM_CONFIRMED = "llm-confirmed"
MSM_REENTRY_READY = "reentry-ready"
MSM_STATES = (
    MSM_STABLE,
    MSM_TRANSITION_RISK,
    MSM_LLM_CONFIRMED,
    MSM_REENTRY_READY,
)
MSM_INITIAL_POSTERIOR = {
    MSM_STABLE: 0.85,
    MSM_TRANSITION_RISK: 0.10,
    MSM_LLM_CONFIRMED: 0.03,
    MSM_REENTRY_READY: 0.02,
}

WINDOW_TRUSTED = "trusted"
WINDOW_SUSPECT = "suspect"
WINDOW_FAILURE = "failure"
WINDOW_INVALID = "invalid"

STEP_ACTIVE = "active"
STEP_REMOVED = "removed"
STEP_FINAL_ANSWER = "final_answer"


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

    def overlaps(self, start_token_idx: int, end_token_idx: int) -> bool:
        return self.start_token_idx <= end_token_idx and self.end_token_idx >= start_token_idx


@dataclass
class EpisodeState:
    owner: str = OWNER_SLM
    mode: str = MODE_COLD_START
    trusted_buffer: list[PDIWindow] = field(default_factory=list)
    failure_buffer: list[PDIWindow] = field(default_factory=list)
    pre_suspect_snapshot: list[float] = field(default_factory=list)
    handoff_point_token_idx: int | None = None
    transition_start_window: PDIWindow | None = None
    transition_windows: list[PDIWindow] = field(default_factory=list)
    reentry_stable_count: int = 0
    reentry_confirm_count: int = 0
    recent_trusted_pdi: list[float] = field(default_factory=list)
    recent_q_percentile: list[float] = field(default_factory=list)

    diagnostic_history: list[dict] = field(default_factory=list)
    last_diagnostic: dict = field(default_factory=dict)
    msm_posterior: dict[str, float] = field(default_factory=lambda: dict(MSM_INITIAL_POSTERIOR))
    msm_history: list[dict] = field(default_factory=list)
    last_msm_update: dict = field(default_factory=dict)


@dataclass
class ControllerDecision:
    action: str
    step: Step | None = None
    window: PDIWindow | None = None
    q_percentile: float | None = None
    upper_excess: float | None = None
    rollback_start_token_idx: int | None = None
    reentry_status: str | None = None


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


def _repeat_key(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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
        self._repair_q_history: list[float] = []
        self._repair_q_baseline: float | None = None
        self._transition_llm_repair_count = 0
        self._last_no_valid_pdi_window: dict[str, Any] = {}
        self.state.msm_posterior = self._normalize_msm(self.cfg.msm_initial_posterior)

        self.driver_switch_count = 0
        self.llm_repair_episodes = 0
        self.rollback_count = 0
        self.handoff_success_count = 0
        self.handoff_failure_count = 0
        self.reentry_failure_count = 0
        self.step_text_repeat_count = 0
        self.msm_update_count = 0
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
        if mode in {MODE_COLD_START, MODE_SLM_NORMAL, MODE_SLM_TRANSITION, MODE_SLM_REENTRY} and owner != OWNER_SLM:
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
        q_high = float(self.cfg.slm_high_q)
        return max(0.0, float(q_percentile) - q_high) / max(1e-12, 1.0 - q_high)

    def _annotate_window(self, window: PDIWindow, q_percentile: float) -> float:
        upper_excess = self.upper_excess(q_percentile)
        window.q_percentile = q_percentile
        window.upper_excess = upper_excess
        return upper_excess

    def _mode_after_trusted_update(self) -> str:
        if len(self.state.trusted_buffer) >= self.cfg.n_min:
            return MODE_SLM_NORMAL
        return MODE_COLD_START

    def _add_trusted_window(self, window: PDIWindow) -> None:
        window.status = WINDOW_TRUSTED
        if window not in self.state.trusted_buffer:
            self.state.trusted_buffer.append(window)
            self.state.recent_trusted_pdi.append(window.pdi)
            self.state.recent_trusted_pdi = self.state.recent_trusted_pdi[-(2 * self.cfg.pdi_repeat_window):]

    def _clear_transition_watch(self) -> None:
        self.state.transition_start_window = None
        self.state.transition_windows = []
        self._transition_llm_repair_count = 0

    def _overlaps_any_window(self, window: PDIWindow, others: list[PDIWindow]) -> bool:
        return any(window.overlaps(other.start_token_idx, other.end_token_idx) for other in others)

    def _enter_transition_watch(self, window: PDIWindow, *, step_id: int) -> None:
        window.status = WINDOW_SUSPECT
        self.state.pre_suspect_snapshot = [w.pdi for w in self.state.trusted_buffer]
        self.state.transition_start_window = window
        self.state.transition_windows = [window]
        self._switch(MODE_SLM_TRANSITION, OWNER_SLM, step_id=step_id, reason="slm_upper_tail_transition")

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
        q_eff = self._compute_q_eff(q_percentile)
        pi_after = self._msm_update(step=step, window=window, q_percentile=q_eff)
        msm_action = self._msm_suggest_action(pi_after, diagnostic_used=False)

        if msm_action in {"llm-repair", "llm-diagnose", "watch"}:
            self._enter_transition_watch(window, step_id=step.step_id)
            action = "ENTER_SLM_TRANSITION"
            self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
            return ControllerDecision(action=action, step=step, window=window, q_percentile=q_percentile, upper_excess=upper_excess)

        if msm_action == "finalize":
            self._add_trusted_window(window)
            action = "EARLY_STOP_MSM_FINALIZE"
            self.early_stop_trigger_count += 1
            self._switch(MODE_FINALIZE, self.state.owner, step_id=step.step_id, reason=action)
            self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
            return ControllerDecision(action=action, step=step, window=window, q_percentile=q_percentile, upper_excess=upper_excess)

        self._add_trusted_window(window)
        action = "TRUST_PDI_WINDOW"

        next_mode = self._mode_after_trusted_update()
        if self.state.mode != next_mode:
            self._switch(next_mode, OWNER_SLM, step_id=step.step_id, reason="trusted_buffer_update")

        if self.state.mode != MODE_COLD_START:
            if (
                self.cfg.repeat_finalize_enabled
                and not self.answer_intent_seen
                and self._check_step_text_repeat(step)
            ):
                action = "STEP_TEXT_REPEAT_FINALIZE"

        self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
        return ControllerDecision(action=action, step=step, window=window, q_percentile=q_percentile, upper_excess=upper_excess)

    def process_transition_window(self, step: Step, *, d_llm: float | None = None) -> ControllerDecision:
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

        if d_llm is not None:
            self.record_diagnostic(step, window.pdi, q_percentile, d_llm)

        q_eff = self._compute_q_eff(q_percentile)
        pi_after = self._msm_update(step=step, window=window, q_percentile=q_eff, d_llm=d_llm)
        msm_action = self._msm_suggest_action(pi_after, diagnostic_used=d_llm is not None)

        if msm_action == "slm-continue" or msm_action == "handoff-back":
            self._transition_llm_repair_count = 0
            if not self._overlaps_any_window(window, self.state.transition_windows[:-1]):
                self._add_trusted_window(window)
            self._clear_transition_watch()
            next_mode = self._mode_after_trusted_update()
            self._switch(next_mode, OWNER_SLM, step_id=step.step_id, reason="slm_transition_recovered")
            action = "SLM_TRANSITION_RECOVERED"
            self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
            return ControllerDecision(action=action, step=step, window=window, q_percentile=q_percentile, upper_excess=upper_excess)

        if msm_action == "llm-repair":
            self._transition_llm_repair_count = 0
            alarm = self.state.transition_start_window
            if alarm is None:
                raise InvalidControllerState("SLM transition failed without a transition start window.")
            rollback_start = alarm.start_token_idx
            action = "SLM_TRANSITION_FAILED_ROLLBACK_TO_LLM_REPAIR"
            self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf, rollback_start_token_idx=rollback_start)
            self.rollback_to_token(rollback_start, reason="slm_transition_failed")
            return ControllerDecision(action=action, step=step, window=window, q_percentile=q_percentile, upper_excess=upper_excess, rollback_start_token_idx=rollback_start)

        action = "SLM_TRANSITION_CONTINUE"
        self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf)
        return ControllerDecision(action=action, step=step, window=window, q_percentile=q_percentile, upper_excess=upper_excess)

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

        pi_after = self._msm_update(step=step, window=window, q_percentile=q_percentile)
        msm_action = self._msm_suggest_action(pi_after, diagnostic_used=False)

        if msm_action in {"llm-repair", "llm-diagnose"}:
            self.reentry_failure_count += 1
            rollback_start = self.state.handoff_point_token_idx
            action = "SLM_REENTRY_FAILED_ROLLBACK_TO_LLM_REPAIR"
            self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf, rollback_start_token_idx=rollback_start, reentry_status="failed")
            if rollback_start is None:
                raise InvalidControllerState("SLM re-entry failed without a handoff point.")
            self.rollback_to_token(rollback_start, reason="reentry_transition_failure")
            self.handoff_failure_count += 1
            return ControllerDecision(action=action, step=step, window=window, q_percentile=q_percentile, upper_excess=upper_excess, rollback_start_token_idx=rollback_start, reentry_status="failed")

        self.state.reentry_stable_count += 1
        self._add_trusted_window(window)
        action = "SLM_REENTRY_CONTINUE"
        reentry_status = "stable"

        if self.state.reentry_stable_count >= self.cfg.m_reentry:
            self.state.reentry_stable_count = 0
            self._reset_evidence_histories()
            self._switch(MODE_SLM_NORMAL, OWNER_SLM, step_id=step.step_id, reason="slm_reentry_stable")
            action = "SLM_REENTRY_STABLE"
            reentry_status = "passed"

        self._log_pdi_decision(step=step, window=window, mode=mode_at_decision, action=action, cdf=cdf, reentry_status=reentry_status)
        return ControllerDecision(action=action, step=step, window=window, q_percentile=q_percentile, upper_excess=upper_excess, reentry_status=reentry_status)

    def _normalize_msm(self, values: dict[str, float]) -> dict[str, float]:
        cleaned = {
            state: max(0.0, float(values.get(state, 0.0) or 0.0))
            for state in MSM_STATES
        }
        total = sum(cleaned.values())
        if total <= 0:
            return dict(MSM_INITIAL_POSTERIOR)
        return {state: cleaned[state] / total for state in MSM_STATES}

    def _msm_predict(self, posterior: dict[str, float]) -> dict[str, float]:
        predicted = {state: 0.0 for state in MSM_STATES}
        normalized = self._normalize_msm(posterior)
        for source, source_prob in normalized.items():
            row = self._normalize_msm(self.cfg.msm_transition_matrix.get(source, {}))
            for target, transition_prob in row.items():
                predicted[target] += source_prob * transition_prob
        return self._normalize_msm(predicted)

    def _compute_q_eff(self, q_percentile: float) -> float:
        k = 4
        hist = self.state.recent_q_percentile
        hist.append(q_percentile)
        self.state.recent_q_percentile = hist[-k:]
        if len(self.state.recent_q_percentile) < k:
            return q_percentile
        vals = self.state.recent_q_percentile
        mean_q = sum(vals) / k
        trend_q = (vals[-1] - vals[0]) / (k - 1)
        return min(1.0, max(0.0, mean_q + float(self.cfg.msm_trend_alpha) * max(0.0, trend_q)))

    def _msm_emission_likelihood(
        self,
        *,
        q_percentile: float,
        d_slm: float,
        d_llm: float | None = None,
    ) -> dict[str, float]:
        q = min(1.0, max(0.0, float(q_percentile)))
        floor = float(self.cfg.msm_emission_floor)
        likelihood = {
            MSM_STABLE: max(floor, 1.0 - q),
            MSM_TRANSITION_RISK: max(floor, q),
            MSM_LLM_CONFIRMED: floor,
            MSM_REENTRY_READY: max(floor, 1.0 - q),
        }
        if d_llm is None:
            return likelihood
        try:
            d_llm_value = float(d_llm)
            d_slm_value = float(d_slm)
        except (TypeError, ValueError):
            return likelihood
        if not (math.isfinite(d_llm_value) and math.isfinite(d_slm_value)):
            return likelihood

        delta = d_slm_value - d_llm_value
        slm_high = q > self.cfg.slm_high_q
        if slm_high:
            likelihood[MSM_LLM_CONFIRMED] = max(floor, q) * self.cfg.msm_llm_beneficial_boost
            likelihood[MSM_TRANSITION_RISK] *= 1.50
            likelihood[MSM_STABLE] *= 0.20
            likelihood[MSM_REENTRY_READY] *= 0.20
        elif delta < self.cfg.delta_reentry_threshold:
            likelihood[MSM_REENTRY_READY] *= self.cfg.msm_reentry_ready_boost
            likelihood[MSM_STABLE] *= 1.50
            likelihood[MSM_TRANSITION_RISK] *= 0.40
        return likelihood

    def _msm_suggest_action(self, posterior: dict[str, float], *, diagnostic_used: bool) -> str:
        thresholds = self.cfg.msm_action_thresholds
        if posterior[MSM_LLM_CONFIRMED] >= thresholds["llm_repair"]:
            return "llm-repair"
        if posterior[MSM_TRANSITION_RISK] >= thresholds["transition_watch"]:
            return "watch"
        if posterior[MSM_REENTRY_READY] >= thresholds["handoff_back"]:
            return "handoff-back"
        if posterior[MSM_STABLE] >= thresholds["slm_continue"]:
            return "slm-continue"
        return "watch"

    def _msm_update(self, *, step: Step, window: PDIWindow, q_percentile: float, d_llm: float | None = None) -> dict[str, float]:
        pi_before = self._normalize_msm(self.state.msm_posterior)
        pi_pred = self._msm_predict(pi_before)
        likelihood = self._msm_emission_likelihood(q_percentile=q_percentile, d_slm=window.pdi, d_llm=d_llm)
        pi_after = self._normalize_msm({state: pi_pred[state] * likelihood[state] for state in MSM_STATES})
        self.state.msm_posterior = pi_after
        self.msm_update_count += 1
        record = {
            "step_id": step.step_id,
            "window_id": window.window_id,
            "mode": self.state.mode,
            "d_slm": window.pdi,
            "q_percentile": q_percentile,
            "d_llm": d_llm,
            "diagnostic_used": d_llm is not None,
            "pi_before": pi_before,
            "pi_pred": pi_pred,
            "emission_likelihood": likelihood,
            "pi_after": pi_after,
            "msm_suggested_action": self._msm_suggest_action(pi_after, diagnostic_used=d_llm is not None),
        }
        self.state.last_msm_update = record
        self.state.msm_history.append(record)
        self.state.msm_history = self.state.msm_history[-128:]
        self.events.append({"problem_id": self.problem_id, "event": "msm_update", **record})
        return pi_after

    def classify_risk_zone(self, *, d_slm: float, q_slm: float, d_llm: float) -> str:
        delta = d_slm - d_llm
        slm_high = q_slm > self.cfg.slm_high_q
        slm_recovered = q_slm <= self.cfg.slm_recover_q
        if slm_high and delta > self.cfg.delta_llm_beneficial_threshold:
            return "llm_confirmed"
        if (slm_recovered or not slm_high) and delta < self.cfg.delta_reentry_threshold:
            return "reentry_ready"
        if slm_high:
            return "transition_risk"
        return "stable"

    def record_diagnostic(self, step: Step, d_slm: float, q_slm: float, d_llm: float) -> str:
        delta = d_slm - d_llm
        zone = self.classify_risk_zone(d_slm=d_slm, q_slm=q_slm, d_llm=d_llm)
        record = {
            "step_id": step.step_id,
            "d_slm": d_slm,
            "q_slm": q_slm,
            "d_llm": d_llm,
            "delta": delta,
            "zone": zone,
            "delta_llm_beneficial_threshold": self.cfg.delta_llm_beneficial_threshold,
            "delta_reentry_threshold": self.cfg.delta_reentry_threshold,
        }
        self.state.diagnostic_history.append(record)
        self.state.last_diagnostic = record
        self.events.append({"problem_id": self.problem_id, "event": "dual_model_diagnostic", **record})
        return zone

    def _check_pdi_repeat_candidate(self, step: Step) -> bool:
        K = self.cfg.pdi_repeat_window
        hist = self.state.recent_trusted_pdi
        if len(hist) < 2 * K:
            return False
        first = hist[-2 * K:-K]
        second = hist[-K:]
        std = (sum((v - sum(hist[-2*K:]) / (2*K)) ** 2 for v in hist[-2*K:]) / (2*K)) ** 0.5
        eps = max(1e-6, std * 0.5)
        if all(abs(a - b) < eps for a, b in zip(first, second)):
            self.events.append({
                "problem_id": self.problem_id,
                "event": "pdi_repeat_candidate",
                "step_id": step.step_id,
                "recent_pdi": list(hist[-2 * K:]),
            })
            return True
        return False

    def _check_step_text_repeat(self, step: Step) -> bool:
        current_key = _repeat_key(step.text)
        if not current_key:
            return False
        active = self.active_steps()
        previous = [
            (item.step_id, _repeat_key(item.text))
            for item in active[-8:]
            if item.step_id != step.step_id and _repeat_key(item.text)
        ]
        repeated_step_ids = [step_id for step_id, text_key in previous if text_key == current_key]
        if not repeated_step_ids:
            return False
        occurrence_count = len(repeated_step_ids) + 1
        if occurrence_count < self.cfg.step_text_repeat_min_occurrences:
            self.events.append({
                "problem_id": self.problem_id,
                "event": "step_text_repeat_observed",
                "step_id": step.step_id,
                "recent_pdi": list(self.state.recent_trusted_pdi),
                "repeat_key": current_key,
                "repeated_step_ids": repeated_step_ids,
                "occurrence_count": occurrence_count,
                "min_occurrences": self.cfg.step_text_repeat_min_occurrences,
            })
            return False
        self.step_text_repeat_count += 1
        self.events.append({
            "problem_id": self.problem_id,
            "event": "step_text_repeat_detected",
            "step_id": step.step_id,
            "recent_pdi": list(self.state.recent_trusted_pdi),
            "repeat_key": current_key,
            "repeated_step_ids": repeated_step_ids,
            "occurrence_count": occurrence_count,
            "min_occurrences": self.cfg.step_text_repeat_min_occurrences,
        })
        self._switch(MODE_FINALIZE, self.state.owner, step_id=step.step_id, reason="step_text_repeat")
        return True

    def note_llm_repair_step(self, step: Step, *, d_llm: float | None = None) -> ControllerDecision:
        self._repair_step_count += 1

        if d_llm is not None and self._repair_step_count >= self.cfg.msm_repair_min_steps_before_reentry:
            ref_cdf = self.effective_cdf()
            q_llm_as_slm = ref_cdf(d_llm)
            self._repair_q_history.append(q_llm_as_slm)
            self._repair_q_history = self._repair_q_history[-2:]
            if self._repair_q_baseline is None and len(self._repair_q_history) >= self.cfg.msm_repair_min_steps_before_reentry:
                self._repair_q_baseline = sum(self._repair_q_history) / len(self._repair_q_history)
            floor = float(self.cfg.msm_emission_floor)
            likelihood = {
                MSM_STABLE: max(floor, 1.0 - q_llm_as_slm) * self.cfg.msm_repair_stable_boost,
                MSM_TRANSITION_RISK: max(floor, q_llm_as_slm) * 0.5,
                MSM_LLM_CONFIRMED: max(floor, q_llm_as_slm) * 0.5,
                MSM_REENTRY_READY: max(floor, 1.0 - q_llm_as_slm) * self.cfg.msm_repair_reentry_boost,
            }
            pi_before = self._normalize_msm(self.state.msm_posterior)
            pi_pred = self._msm_predict(pi_before)
            pi_after = self._normalize_msm({s: pi_pred[s] * likelihood[s] for s in MSM_STATES})
            self.state.msm_posterior = pi_after
            sorted_q = sorted(self._repair_q_history)
            n = len(sorted_q)
            median_q_llm = (sorted_q[n // 2] + sorted_q[(n - 1) // 2]) / 2
            handoff_threshold = (
                self._repair_q_baseline * self.cfg.msm_repair_handoff_decay_factor
                if self._repair_q_baseline is not None
                else self.cfg.msm_repair_handoff_q_threshold
            )
            self.events.append({
                "problem_id": self.problem_id,
                "event": "msm_llm_repair_update",
                "step_id": step.step_id,
                "d_llm": d_llm,
                "q_llm_as_slm": q_llm_as_slm,
                "median_q_llm": median_q_llm,
                "repair_q_baseline": self._repair_q_baseline,
                "handoff_threshold": handoff_threshold,
                "pi_after": pi_after,
            })
            if median_q_llm <= handoff_threshold:
                self.state.reentry_confirm_count += 1
            else:
                self.state.reentry_confirm_count = 0
            if self.state.reentry_confirm_count >= 1:
                self.state.reentry_confirm_count = 0
                self.handoff_success_count += 1
                self.state.handoff_point_token_idx = self.visible_token_count()
                self._reset_evidence_histories()
                self.state.reentry_stable_count = 0
                self._new_episode()
                self._switch(MODE_SLM_REENTRY, OWNER_SLM, step_id=step.step_id, reason="msm_reentry_ready")
                return ControllerDecision(action="HANDOFF_TO_SLM_REENTRY", step=step)

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
        self.state.reentry_stable_count = 0
        self.state.handoff_point_token_idx = None
        self._new_episode()
        self.llm_repair_episodes += 1
        self._repair_step_count = 0
        self._repair_q_history = []
        self._repair_q_baseline = None
        self.state.recent_trusted_pdi = []
        self.state.recent_q_percentile = []
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
        llm_diagnostic_wall_time: float,
        llm_diagnostic_count: int,
        slm_prefill_count: int,
        llm_prefill_count: int,
    ) -> dict[str, Any]:
        slm_tokens = self.source_token_count(OWNER_SLM)
        llm_tokens = self.source_token_count(OWNER_LLM)
        total_tokens = slm_tokens + llm_tokens
        handoff_rate = (
            self.handoff_success_count / (self.handoff_success_count + self.handoff_failure_count)
            if (self.handoff_success_count + self.handoff_failure_count) else 0.0
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
            "handoff_success_count": self.handoff_success_count,
            "handoff_failure_count": self.handoff_failure_count,
            "handoff_success_rate": handoff_rate,
            "reentry_failure_count": self.reentry_failure_count,
            "reentry_failure_rate": (
                self.reentry_failure_count / self.handoff_success_count if self.handoff_success_count else 0.0
            ),
            "early_stop_trigger_count": self.early_stop_trigger_count,
            "step_text_repeat_count": self.step_text_repeat_count,
            "diagnostic_count": sum(1 for e in self.events if e.get("event") == "dual_model_diagnostic"),
            "msm_update_count": self.msm_update_count,
            "msm_final_posterior": dict(self.state.msm_posterior),
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
            "llm_diagnostic_wall_time": llm_diagnostic_wall_time,
            "llm_diagnostic_count": llm_diagnostic_count,
            "config": {
                "t_min": self.cfg.t_min,
                "lambda0": self.cfg.lambda0,
                "lambda0_self": self.lambda0_self,
                "n_min": self.cfg.n_min,
                "slm_high_q": self.cfg.slm_high_q,
                "slm_recover_q": self.cfg.slm_recover_q,
                "m_reentry": self.cfg.m_reentry,
                "max_llm_repair_steps": self.cfg.max_llm_repair_steps,
                "msm_initial_posterior": dict(self.cfg.msm_initial_posterior),
                "msm_transition_matrix": self.cfg.msm_transition_matrix,
                "msm_action_thresholds": self.cfg.msm_action_thresholds,
                "msm_emission_floor": self.cfg.msm_emission_floor,
                "msm_llm_beneficial_boost": self.cfg.msm_llm_beneficial_boost,
                "msm_reentry_ready_boost": self.cfg.msm_reentry_ready_boost,
                "msm_stable_boost": self.cfg.msm_stable_boost,
                "msm_repair_min_steps_before_reentry": self.cfg.msm_repair_min_steps_before_reentry,
                "msm_repair_reentry_boost": self.cfg.msm_repair_reentry_boost,
                "msm_repair_stable_boost": self.cfg.msm_repair_stable_boost,
                "delta_llm_beneficial_threshold": self.cfg.delta_llm_beneficial_threshold,
                "delta_reentry_threshold": self.cfg.delta_reentry_threshold,
                "llm_diagnostic_enabled": self.cfg.llm_diagnostic_enabled,
                "repeat_finalize_enabled": self.cfg.repeat_finalize_enabled,
                "step_text_repeat_min_occurrences": self.cfg.step_text_repeat_min_occurrences,
                "pdi_repeat_window": self.cfg.pdi_repeat_window,
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
        return

    def _log_pdi_decision(
        self,
        *,
        step: Step,
        window: PDIWindow,
        mode: str,
        action: str,
        cdf: EffectiveCDF,
        rollback_start_token_idx: int | None = None,
        reentry_status: str | None = None,
    ) -> None:
        msm = (
            dict(self.state.last_msm_update)
            if self.state.last_msm_update.get("window_id") == window.window_id
            else {}
        )
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
                "answer_intent_seen": self.answer_intent_seen,
                "action": action,
                "trusted_buffer_size": len(self.state.trusted_buffer),
                "prior_weight": cdf.prior_weight,
                "rollback_start_token_idx": rollback_start_token_idx,
                "reentry_status": reentry_status,
                "msm_pi_before": msm.get("pi_before"),
                "msm_pi_pred": msm.get("pi_pred"),
                "msm_emission_likelihood": msm.get("emission_likelihood"),
                "msm_pi_after": msm.get("pi_after"),
                "msm_suggested_action": msm.get("msm_suggested_action"),
                "msm_diagnostic_used": msm.get("diagnostic_used"),
                "msm_d_llm": msm.get("d_llm"),
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
