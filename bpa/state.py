from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(Enum):
    THINKING = "thinking"
    FINAL_ANSWER = "final_answer"
    DONE = "done"


class Decision(Enum):
    SLM_DIRECT = "slm_direct"
    LLM_FULL = "llm_full"


@dataclass
class TraceEvent:
    step_idx: int
    event: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationState:
    problem_text: str
    assistant_prefix_text: str = ""
    generation_protocol: str = "routed_stepwise"
    phase: Phase = Phase.THINKING
    has_seen_close_think: bool = False
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


@dataclass
class RepetitionState:
    recent_steps: deque[str] = field(default_factory=lambda: deque(maxlen=4))
    ngram_counter: Counter[str] = field(default_factory=Counter)
    triggered: bool = False
    trigger_reason: str | None = None


@dataclass
class L0Result:
    passed: bool
    h_init: float
    margin: float
    top_logprobs: dict[int, float]
    top_token_strs: list[str]
    first_char_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "h_init": self.h_init,
            "margin": self.margin,
            "top_logprobs": self.top_logprobs,
            "top_token_strs": self.top_token_strs,
            "first_char_class": self.first_char_class,
        }


@dataclass
class CascadeResult:
    decision: Decision
    l0: L0Result
