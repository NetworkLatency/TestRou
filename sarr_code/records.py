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
    text: str
    token_ids: list[int]
    source: str                    # "SLM" | "LLM" | "SYSTEM"
    status: str                    # active | removed | sealed | probe_discarded
    driver_state_when_generated: str
    observed_signals: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    action: str = "TRUST"

    finish_reason: str | None = None
    prompt_tokens: int = 0
    wall_time: float = 0.0
    attempt_id: int | None = None

    transition_type: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass
class ControllerEvent:
    problem_id: str
    event: str
    step_id: int | None = None
    from_state: str | None = None
    to_state: str | None = None
    reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
