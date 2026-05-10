from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Optional

from .state import RepetitionState


CLOSE_THINK_TAG = "</think>"
OPEN_THINK_TAG = "<think>"


def captured_close_think_prefix(text: str) -> str:
    idx = text.find(CLOSE_THINK_TAG)
    if idx < 0:
        return ""
    return text[: idx + len(CLOSE_THINK_TAG)]


def has_close_think_tag(text: str) -> bool:
    return CLOSE_THINK_TAG in text


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


def update_strict_step_repetition(rep: RepetitionState, new_step_text: str, min_chars: int = 10) -> str | None:
    normalized = new_step_text.rstrip("\n").rstrip()
    if len(normalized) < min_chars:
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


def _strip_answer_candidate(candidate: Optional[str]) -> Optional[str]:
    if candidate is None:
        return None
    s = str(candidate).strip()
    if not s:
        return None
    s = s.strip(" \t\r\n\"'`")
    while len(s) >= 2 and s[0] in "([{":
        closing = {"(": ")", "[": "]", "{": "}"}[s[0]]
        if s.endswith(closing):
            s = s[1:-1].strip()
            continue
        break
    while s and s[-1] in ",;:\uff0c\uff1b\uff1a\u3002":
        s = s[:-1].strip()
    while s.endswith("."):
        without_dot = s[:-1].rstrip()
        if re.search(r"\d\.\d+$", without_dot):
            break
        s = without_dot
    return clean_latex_answer(s)


def _extract_incomplete_boxed(text: str) -> Optional[str]:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(matches):
        start = match.end()
        tail = text[start:].strip()
        if not tail:
            continue
        candidate = re.split(r"[\r\n]", tail, maxsplit=1)[0]
        candidate = candidate[:80].strip()
        if "}" in candidate:
            candidate = candidate.split("}", 1)[0]
        candidate = _strip_answer_candidate(candidate)
        if candidate:
            return candidate
    return None


_ANSWER_LABEL_RE = re.compile(
    r"(?is)\b(?:final\s+answer|final\s+result|answer|the\s+answer|result)\b\s*(?:is|=|:)?\s*"
)


def _extract_delimited_math(text: str) -> Optional[str]:
    patterns = (
        r"^\$\$(.+?)\$\$",
        r"^\$(.+?)\$",
        r"^\\\((.+?)\\\)",
        r"^\\\[(.+?)\\\]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.S)
        if match:
            return _strip_answer_candidate(match.group(1))
    return None


def _extract_short_math_candidate(text: str) -> Optional[str]:
    s = text.strip()
    if not s:
        return None
    s = re.sub(r"^(?:is|=|:)\s*", "", s, flags=re.I).strip()
    if not s:
        return None

    boxed = extract_last_boxed(s)
    if boxed is not None:
        return clean_latex_answer(boxed)
    incomplete_boxed = _extract_incomplete_boxed(s)
    if incomplete_boxed is not None:
        return incomplete_boxed
    delimited = _extract_delimited_math(s)
    if delimited is not None:
        return delimited

    latex_match = re.match(
        r"(\\(?:frac|sqrt)\s*\{[^{}\r\n]{1,80}\}(?:\s*\{[^{}\r\n]{1,80}\})?)",
        s,
    )
    if latex_match:
        return _strip_answer_candidate(latex_match.group(1))

    number_match = re.match(r"([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?)\b", s)
    if number_match:
        return _strip_answer_candidate(number_match.group(1))

    choice_match = re.match(r"\(?([ABCD])\)?\b", s, flags=re.I)
    if choice_match:
        return choice_match.group(1).upper()

    line = re.split(r"[\r\n\u3002\uff1b;]", s, maxsplit=1)[0].strip()
    sentence = re.split(r"(?<!\d)\.(?!\d)", line, maxsplit=1)[0].strip()
    candidate = sentence[:80].strip()
    if re.search(r"\d|\\(?:frac|sqrt)|[=+\-*/^]", candidate):
        return _strip_answer_candidate(candidate)
    return None


def _extract_labeled_final_answer(text: str) -> Optional[str]:
    matches = list(_ANSWER_LABEL_RE.finditer(text))
    for match in reversed(matches):
        candidate = _extract_short_math_candidate(text[match.end() :])
        if candidate is not None:
            return candidate
    return None


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


def clean_latex_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    s = str(answer).strip()
    if not s:
        return None

    while True:
        if len(s) >= 4 and s.startswith("$$") and s.endswith("$$"):
            s = s[2:-2].strip()
            continue
        if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
            s = s[1:-1].strip()
            continue
        break

    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9])(?![A-Za-z0-9])", r"\\sqrt{\1}", s)
    s = s.strip()
    return s or None


def _after_last_close_think_tag(text: str) -> str | None:
    idx = text.rfind(CLOSE_THINK_TAG)
    if idx < 0:
        return None
    return text[idx + len(CLOSE_THINK_TAG) :]


def extract_answer_from_final_step(final_step_text: Optional[str]) -> str | None:
    if not isinstance(final_step_text, str) or not final_step_text.strip():
        return None
    post_think = _after_last_close_think_tag(final_step_text)
    if post_think is not None and post_think.strip():
        boxed = extract_last_boxed(post_think)
        if boxed is not None:
            return clean_latex_answer(boxed)
        labeled = _extract_labeled_final_answer(post_think)
        if labeled is not None:
            return labeled
        return clean_latex_answer(post_think)
    boxed = extract_last_boxed(final_step_text)
    if boxed is not None:
        return clean_latex_answer(boxed)
    labeled = _extract_labeled_final_answer(final_step_text)
    if labeled is not None:
        return labeled
    return clean_latex_answer(final_step_text)


def extract_answer_from_steps(
    step_logs: Iterable[dict[str, Any]],
    assistant_text: Optional[str] = None,
) -> str | None:
    steps = list(step_logs)
    if steps:
        return extract_answer_from_final_step(steps[-1].get("step_text"))
    return extract_answer_from_final_step(assistant_text)


def extract_answer(assistant_text: str) -> str | None:
    boxed = extract_last_boxed(assistant_text)
    if boxed is not None:
        return clean_latex_answer(boxed)
    if CLOSE_THINK_TAG in assistant_text:
        return clean_latex_answer(assistant_text.split(CLOSE_THINK_TAG, 1)[1])
    return clean_latex_answer(assistant_text)
