from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepOutput:
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int = 0
    wall_time: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass
class StepRecord:
    problem_id: str
    step_id: int
    generator: str
    text: str
    token_ids: list[int]

    c_raw: float | None = None
    c_norm: float | None = None
    c_smooth: float | None = None
    readiness_raw: float | None = None
    readiness_raw_smooth: float | None = None
    readiness_source: str = "raw"
    calibration_enabled: bool = False
    readiness: float | None = None
    readiness_value: float | None = None
    readiness_high: bool = False
    readiness_mid: bool = False
    readiness_low: bool = False
    stagnation_score: float = 0.0
    stagnation_high: bool = False
    stagnation_suspect: bool = False
    stagnation_suspect_run: int = 0
    stagnation_confirmed: bool = False
    hcs_suspect: bool = False
    hcs_suspect_run: int = 0
    hcs_confirmed: bool = False
    clean_autonomy_anchor: int | None = None
    anchor_refresh_allowed: bool = False
    anchor_refresh_blocked_reason: str | None = None
    autonomy_state: str | None = None

    state_before: str | None = None
    state_after: str | None = None

    degeneration_event: int = 0
    D_start: int = 0
    D_post: int = 0
    stable_anchor: int | None = None

    action: str = "TRUST"
    finish_reason: str | None = None
    prompt_tokens: int = 0
    wall_time: float = 0.0
    attempt_id: int | None = None
    active: bool = True
    removed_by_rollback: bool = False
    is_recovery: bool = False
    transition_type: str | None = None
    delta_c_norm: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass
class RollbackEvent:
    problem_id: str
    type: str
    reason: str
    trigger_step: int
    anchor_step: int
    rollback_span: int

    removed_steps: list[dict]
    recovery_steps: list[dict]
    stop_reason: str

    recovery_max_steps: int
    recovery_actual_steps: int
    recovery_c_norm: list[float]
    requested_anchor_step: int | None = None
    anchor_repeat_count_before: int = 0
    anchor_backoff_steps: int = 0
    suspect_start_step: int | None = None
    suspect_steps: int = 0
    D_suspect: int = 0
    fallback_no_delete: bool = False
    long_span: bool = False
    long_span_policy: str | None = None
    long_span_fallback_count_before: int = 0
    long_span_recovery_limited: bool = False
    force_next_step_slm: bool = True
    force_slm_after_recovery_failed: bool | None = None
    event: str | None = None
    rollback_before_lease: bool | None = None
    rollback_anchor: int | None = None
    lease_steps: int | None = None
    max_tokens_per_step: int | None = None
    prompt_type: str | None = None
    mention_uncertainty: bool | None = None
    mention_repetition: bool | None = None
    mention_error: bool | None = None
    state_after: str | None = None
    clean_anchor_step: int | None = None
    hcs_rollback_count: int = 0
    stagnation_rollback_count: int = 0
    readiness_source: str | None = None
    calibration_enabled: bool | None = None
    llm_recovery_prompt_type: str | None = None
    mention_stagnation: bool | None = None
    return_to_slm: bool | None = None
