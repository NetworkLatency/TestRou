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

    @property
    def slm_generate_calls(self) -> int:
        return self.state.slm_generate_calls

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


def result_summary(result: BPAResult) -> dict[str, Any]:
    slm_total_tokens = result.slm_decode_tokens + result.slm_prefill_tokens
    llm_total_tokens = result.llm_decode_tokens + result.llm_prefill_tokens
    total_model_tokens = slm_total_tokens + llm_total_tokens
    total_decode_tokens = result.slm_decode_tokens + result.llm_decode_tokens
    llm_wall_time = result.state.llm_generation_wall_time + result.state.llm_scoring_wall_time
    model_wall_time = result.state.slm_wall_time + llm_wall_time
    return {
        "answer": result.answer,
        "correct": result.correct,
        "generation_protocol": result.state.generation_protocol,
        "step_count": result.state.step_count,
        "total_wall_time": result.total_wall_time,
        "slm_decode_tokens": result.slm_decode_tokens,
        "slm_prefill_tokens": result.slm_prefill_tokens,
        "llm_decode_tokens": result.llm_decode_tokens,
        "llm_prefill_tokens": result.llm_prefill_tokens,
        "slm_total_tokens": slm_total_tokens,
        "llm_total_tokens": llm_total_tokens,
        "total_model_tokens": total_model_tokens,
        "llm_token_share": (llm_total_tokens / total_model_tokens) if total_model_tokens else 0.0,
        "llm_decode_share": (result.llm_decode_tokens / total_decode_tokens) if total_decode_tokens else 0.0,
        "slm_generate_calls": result.slm_generate_calls,
        "llm_generate_calls": result.llm_full_calls,
        "llm_scoring_calls": result.llm_scoring_calls,
        "llm_full_calls": result.llm_full_calls,
        "slm_wall_time": result.state.slm_wall_time,
        "llm_generation_wall_time": result.state.llm_generation_wall_time,
        "llm_scoring_wall_time": result.state.llm_scoring_wall_time,
        "llm_wall_time": llm_wall_time,
        "model_wall_time": model_wall_time,
        "llm_wall_time_share": (llm_wall_time / model_wall_time) if model_wall_time else 0.0,
        "stop_reason": result.state.stop_reason,
    }
