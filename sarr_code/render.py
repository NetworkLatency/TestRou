from __future__ import annotations

from typing import Any

from .safety import OPEN_THINK_TAG


def render_for_continuation(problem_text: str, assistant_prefix_text: str, tokenizer: Any) -> str:
    generation_prompt = _render_generation_prompt(problem_text, tokenizer)
    return generation_prompt + OPEN_THINK_TAG + assistant_prefix_text


def _render_generation_prompt(problem_text: str, tokenizer: Any) -> str:
    messages = [{"role": "user", "content": problem_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=False,
        add_generation_prompt=True,
    )
