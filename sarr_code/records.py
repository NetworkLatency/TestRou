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
