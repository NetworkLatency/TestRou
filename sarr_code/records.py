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
