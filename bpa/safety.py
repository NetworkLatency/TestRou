from __future__ import annotations

import re
from typing import Optional

from .phase_machine import CLOSE_THINK_TAG
from .state import RepetitionState


def ensure_step_terminator(step_text: str, finish_reason: str) -> str:
    if finish_reason == "eos":
        return step_text
    if not step_text.endswith("\n\n"):
        return step_text + "\n\n"
    return step_text


def update_repetition(
    rep: RepetitionState,
    new_step_text: str,
    ngram_size: int = 8,
    ngram_threshold: int = 4,
) -> str | None:
    normalized = new_step_text.rstrip("\n").rstrip()
    if len(normalized) < 10:
        rep.recent_steps.append(normalized)
        return None

    if rep.recent_steps and rep.recent_steps[-1] == normalized:
        rep.triggered = True
        rep.trigger_reason = "duplicate_step"
        return "duplicate_step"

    if len(rep.recent_steps) >= 2 and rep.recent_steps[-2] == normalized:
        rep.triggered = True
        rep.trigger_reason = "alternating_step"
        return "alternating_step"

    rep.recent_steps.append(normalized)

    if len(normalized) >= ngram_size:
        for i in range(len(normalized) - ngram_size + 1):
            ng = normalized[i : i + ngram_size]
            rep.ngram_counter[ng] += 1
            if rep.ngram_counter[ng] >= ngram_threshold:
                rep.triggered = True
                rep.trigger_reason = "ngram_repeat"
                return "ngram_repeat"
    return None


def extract_last_boxed(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    positions = []
    start = 0
    while True:
        idx = text.find(r"\boxed", start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(r"\boxed")
    if not positions:
        return None

    idx = positions[-1] + len(r"\boxed")
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        return None
    if text[idx] == "{":
        depth = 0
        content_start = idx + 1
        for j in range(idx, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:j]
        return None
    j = idx
    while j < len(text) and not text[j].isspace():
        j += 1
    token = text[idx:j].strip()
    return token or None


def extract_choice_letter(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    boxed = extract_last_boxed(text)
    candidates = [boxed, text]
    for candidate in candidates:
        if not candidate:
            continue
        match = re.search(r"\b([ABCD])\b", candidate.upper())
        if match:
            return match.group(1)
    return None


def extract_answer(assistant_text: str) -> str | None:
    boxed = extract_last_boxed(assistant_text)
    if boxed is not None:
        return boxed
    if CLOSE_THINK_TAG in assistant_text:
        return assistant_text.split(CLOSE_THINK_TAG, 1)[1].strip() or None
    return assistant_text.strip() or None
