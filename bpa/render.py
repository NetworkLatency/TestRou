from __future__ import annotations

import hashlib
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


def chat_template_hash(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", "") or ""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def rendered_initial_assistant_marker(problem_text: str, tokenizer: Any, width: int = 200) -> str:
    rendered = render_for_continuation(problem_text, "", tokenizer)
    user_tail = problem_text[-80:]
    idx = rendered.rfind(user_tail)
    if idx >= 0:
        return rendered[idx + len(user_tail) : idx + len(user_tail) + width]
    return rendered[-width:]
