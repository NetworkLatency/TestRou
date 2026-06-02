from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(Enum):
    RUNNING = "running"
    DONE = "done"


@dataclass
class TraceEvent:
    step_idx: int
    event: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationState:
    problem_text: str
    assistant_prefix_text: str = ""
    generation_protocol: str = "sarr_stepwise"
    phase: Phase = Phase.RUNNING
    step_count: int = 0
    slm_decode_tokens: int = 0
    slm_prefill_tokens: int = 0
    llm_decode_tokens: int = 0
    llm_prefill_tokens: int = 0
    slm_generate_calls: int = 0
    llm_scoring_calls: int = 0
    llm_full_calls: int = 0
    slm_wall_time: float = 0.0
    llm_generation_wall_time: float = 0.0
    llm_scoring_wall_time: float = 0.0
    trace: list[TraceEvent] = field(default_factory=list)
    stop_reason: str | None = None
