from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .state import GenerationState


@dataclass
class BPAResult:
    answer: str | None
    state: GenerationState
    total_wall_time: float
    correct: bool | None = None

    @property
    def slm_decode_tokens(self) -> int:
        return self.state.slm_decode_tokens

    @property
    def slm_prefill_tokens(self) -> int:
        return self.state.slm_prefill_tokens

    @property
    def llm_decode_tokens(self) -> int:
        return self.state.llm_decode_tokens

    @property
    def llm_prefill_tokens(self) -> int:
        return self.state.llm_prefill_tokens

    @property
    def llm_scoring_calls(self) -> int:
        return self.state.llm_scoring_calls

    @property
    def llm_full_calls(self) -> int:
        return self.state.llm_full_calls

    def equivalent_llm_tokens(self, slm_to_llm_flop_ratio: float) -> float:
        slm_total = self.slm_decode_tokens + self.slm_prefill_tokens
        llm_total = self.llm_decode_tokens + self.llm_prefill_tokens
        return slm_total * slm_to_llm_flop_ratio + llm_total


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def result_summary(result: BPAResult, slm_to_llm_flop_ratio: float) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "correct": result.correct,
        "total_wall_time": result.total_wall_time,
        "slm_decode_tokens": result.slm_decode_tokens,
        "slm_prefill_tokens": result.slm_prefill_tokens,
        "llm_decode_tokens": result.llm_decode_tokens,
        "llm_prefill_tokens": result.llm_prefill_tokens,
        "llm_scoring_calls": result.llm_scoring_calls,
        "llm_full_calls": result.llm_full_calls,
        "equivalent_llm_tokens": result.equivalent_llm_tokens(slm_to_llm_flop_ratio),
        "stop_reason": result.state.stop_reason,
    }
