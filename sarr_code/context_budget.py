from __future__ import annotations

from typing import Any


CONTEXT_LENGTH_SAFETY_MARGIN = 8


class ContextBudgetExceeded(RuntimeError):
    def __init__(self, *, prompt_tokens: int, max_model_len: int, safety_margin: int = CONTEXT_LENGTH_SAFETY_MARGIN):
        super().__init__(
            f"Prompt uses {prompt_tokens} tokens, leaving no safe generation room under max_model_len={max_model_len}."
        )
        self.prompt_tokens = prompt_tokens
        self.max_model_len = max_model_len
        self.safety_margin = safety_margin

    def to_trace_data(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "max_model_len": self.max_model_len,
            "safety_margin": self.safety_margin,
        }


def generation_budget_for_rendered(
    rendered_prompt: str,
    engine: Any,
    runtime: Any,
    requested_max_tokens: int,
) -> tuple[int, int]:
    prompt_tokens = len(engine.encode(rendered_prompt))
    max_model_len = int(runtime.max_model_len)
    available_tokens = max_model_len - prompt_tokens - CONTEXT_LENGTH_SAFETY_MARGIN
    if available_tokens <= 0:
        raise ContextBudgetExceeded(prompt_tokens=prompt_tokens, max_model_len=max_model_len)
    return min(int(requested_max_tokens), available_tokens), prompt_tokens
