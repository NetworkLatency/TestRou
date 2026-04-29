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
    LLM_ARBITRATE = "llm_arbitrate"
    LLM_FULL = "llm_full"


@dataclass
class TraceEvent:
    step_idx: int
    event: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RejectedBranch:
    step_idx: int
    loser_text: str
    winner_text: str
    l2: "L2Result"


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
    rejected_branches_log: list[RejectedBranch] = field(default_factory=list)
    branch_logs: list[dict[str, Any]] = field(default_factory=list)
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
class BranchCandidate:
    first_token_id: int
    first_token_str: str
    raw_rollout_text: str
    raw_rollout_token_ids: list[int]
    step_branch_text: str
    step_branch_was_truncated: bool
    rollout_logprobs: list[float]
    first_token_logprob: float
    sum_logprob_raw: float
    sum_logprob_step: float
    cutoff_tok_count: int | None = None
    ended_by_eos: bool = False
    finish_reason: str = "length"

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_token_id": self.first_token_id,
            "first_token_str": self.first_token_str,
            "raw_rollout_text": self.raw_rollout_text,
            "raw_rollout_token_ids": self.raw_rollout_token_ids,
            "step_branch_text": self.step_branch_text,
            "step_branch_was_truncated": self.step_branch_was_truncated,
            "rollout_logprobs": self.rollout_logprobs,
            "first_token_logprob": self.first_token_logprob,
            "sum_logprob_raw": self.sum_logprob_raw,
            "sum_logprob_step": self.sum_logprob_step,
            "cutoff_tok_count": self.cutoff_tok_count,
            "ended_by_eos": self.ended_by_eos,
            "finish_reason": self.finish_reason,
        }


@dataclass
class L2Result:
    avg_lp_raw_1: float
    avg_lp_raw_2: float
    delta_avg_lp_raw: float
    avg_lp_step_1: float
    avg_lp_step_2: float
    delta_avg_lp_step: float
    text_jaccard_3gram_raw: float
    text_jaccard_3gram_step: float
    branches_diverged_at_token: int
    triggered_arbitration: bool
    trigger_reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SpanLocateResult:
    token_ids: list[int]
    branch_start_token: int
    branch_end_token: int
    span_method: str
    has_boundary_crossing_token: bool
    char_start: int
    char_end: int
    is_invalid: bool
    invalid_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class BranchScore:
    mean_logprob: float | None
    branch_token_count: int
    span_locate: SpanLocateResult
    is_invalid: bool
    invalid_reason: str | None
    prefill_tokens: int
    missing_count: int = 0
    missing_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["span_locate"] = self.span_locate.to_dict()
        return data


@dataclass
class ArbitrationResult:
    score1: BranchScore
    score2: BranchScore
    winner_idx: int
    is_invalid: bool
    invalid_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score1": self.score1.to_dict(),
            "score2": self.score2.to_dict(),
            "winner_idx": self.winner_idx,
            "is_invalid": self.is_invalid,
            "invalid_reason": self.invalid_reason,
        }


@dataclass
class CascadeResult:
    decision: Decision
    l0: L0Result
    l1: tuple[BranchCandidate, BranchCandidate] | None = None
    l2: L2Result | None = None
    arbitration: ArbitrationResult | None = None
    winner_branch: BranchCandidate | None = None
