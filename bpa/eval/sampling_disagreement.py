from __future__ import annotations

import re
from typing import Any

from bpa.safety import clean_latex_answer, extract_last_boxed


EVIDENCE_CHANNEL_PRIORITY = (
    "boxed_answer",
    "rhs_novel_number",
    "equation_claim",
    "novel_number_set",
    "operation_intent",
)

ROUTING_EVIDENCE_CHANNEL_PRIORITY = (
    "boxed_answer",
    "rhs_novel_number",
    "equation_claim",
    "novel_number_set",
)

CONTENT_ANCHOR_CHANNEL = "content_anchor"

_GENERIC_ANCHOR_WORDS = {
    "about",
    "above",
    "after",
    "again",
    "answer",
    "because",
    "before",
    "being",
    "calculate",
    "calculation",
    "check",
    "clearly",
    "compute",
    "consider",
    "continue",
    "could",
    "does",
    "done",
    "equation",
    "expression",
    "final",
    "find",
    "first",
    "from",
    "given",
    "have",
    "hence",
    "into",
    "just",
    "know",
    "maybe",
    "must",
    "need",
    "next",
    "note",
    "now",
    "obvious",
    "only",
    "problem",
    "recheck",
    "right",
    "same",
    "show",
    "simplify",
    "since",
    "solve",
    "step",
    "still",
    "suppose",
    "that",
    "then",
    "therefore",
    "this",
    "thus",
    "using",
    "wait",
    "want",
    "where",
    "which",
    "with",
    "wrong",
}
_GENERIC_ANCHOR_PHRASES = (
    "let us",
    "let's",
    "we need",
    "we have",
    "we can",
    "we get",
    "the answer is not obvious",
)
_CODE_KEYWORDS = {
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "elif",
    "else",
    "except",
    "for",
    "if",
    "import",
    "lambda",
    "raise",
    "return",
    "try",
    "while",
    "yield",
}
_IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CHOICE_PATTERN = re.compile(r"(?i)(?:\b(?:option|choice|answer)\s*[:\-]?\s*([A-E])\b|\(([A-E])\))")
_SYMBOLIC_FRAGMENT_PATTERN = re.compile(
    r"(?<!\w)(?:[A-Za-z_]\w*|\d+)\s*(?:[+\-*/^]|\\(?:mid|mod|equiv|leq|geq)|[<>|])\s*(?:[A-Za-z_]\w*|\d+)(?!\w)"
)

INTENT_KEYWORDS = {
    "backtrack": ("wait", "mistake", "wrong", "not right", "recheck"),
    "finalization": ("therefore", "hence", "answer", "boxed", "final"),
    "case_split": ("case", "suppose", "assume", "consider", "if "),
    "substitution": ("substitute", "plug", "replace"),
    "solve_equation": ("solve", "quadratic", "factor", "equation", "root"),
    "counting": ("count", "choose", "combination", "permutation", "arrangement", "number of"),
    "geometry": ("angle", "arc", "circle", "triangle", "radius", "area", "perimeter", "parallel"),
    "calculation": ("compute", "calculate", "simplify", "expand", "evaluate"),
    "verification": ("check", "verify"),
}

_LATEX_NUMBER_PATTERNS = (
    re.compile(r"\\(?:d?frac)\s*\{[^{}]+\}\s*\{[^{}]+\}"),
    re.compile(r"\\sqrt\s*\{?[-+]?\d+(?:\.\d+)?\}?"),
)
_PLAIN_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?%?)(?![A-Za-z])")
_MATH_BOUNDARY_CHARS = set(" \t\r\n.,;:")
_MATH_SYMBOL_CHARS = set("+-*/^_()[]{}\\|<>")


def _normalize_signature_value(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n$.,;:")


def _balanced_braces(text: str) -> bool:
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def _math_left_boundary(text: str, eq_idx: int) -> int:
    idx = eq_idx - 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    while idx >= 0:
        ch = text[idx]
        if ch.isalnum() or ch in _MATH_SYMBOL_CHARS:
            idx -= 1
            continue
        if ch in _MATH_BOUNDARY_CHARS or ch == "$":
            break
        break
    return idx + 1


def _math_right_boundary(text: str, eq_idx: int) -> int:
    idx = eq_idx + 1
    while idx < len(text) and text[idx].isspace():
        idx += 1
    while idx < len(text):
        ch = text[idx]
        if ch.isalnum() or ch in _MATH_SYMBOL_CHARS:
            idx += 1
            continue
        if ch in _MATH_BOUNDARY_CHARS or ch == "$":
            break
        break
    return idx


def _looks_like_math_side(side: str) -> bool:
    compact = re.sub(r"\s+", "", side.strip("$ "))
    if not compact:
        return False
    if re.search(r"[A-Za-z0-9\\]", compact) is None:
        return False
    allowed_latex = ("frac", "dfrac", "tfrac", "sqrt", "left", "right")
    for command in allowed_latex:
        compact = compact.replace(command, "")
    alpha_words = re.findall(r"[A-Za-z]{2,}", compact)
    return not alpha_words


def _extract_equation(text: str) -> str | None:
    for line in text.splitlines():
        if "=" not in line:
            continue
        line = line.strip()
        if not line:
            continue
        for match in re.finditer("=", line):
            left = _math_left_boundary(line, match.start())
            right = _math_right_boundary(line, match.start())
            candidate = _normalize_signature_value(line[left:right])
            if not candidate or candidate.count("=") != 1 or len(candidate) > 120:
                continue
            lhs, rhs = candidate.split("=", 1)
            if not (_looks_like_math_side(lhs) and _looks_like_math_side(rhs)):
                continue
            if not _balanced_braces(candidate):
                continue
            return candidate
    return None


def _normalize_equation_claim(equation: str) -> str:
    normalized = clean_latex_answer(_normalize_signature_value(equation))
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return re.sub(r"\s+", "", normalized)


def _normalize_number_value(value: str) -> str:
    normalized = clean_latex_answer(_normalize_signature_value(value))
    normalized = normalized.replace(",", "")
    return re.sub(r"\s+", "", normalized)


def _number_matches(text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for pattern in _LATEX_NUMBER_PATTERNS:
        matches.extend((match.start(), _normalize_signature_value(match.group(0))) for match in pattern.finditer(text))
    matches.extend((match.start(), _normalize_signature_value(match.group(0))) for match in _PLAIN_NUMBER_PATTERN.finditer(text))
    return sorted(matches, key=lambda item: item[0])


def _context_number_values(context_text: str) -> set[str]:
    return {_normalize_number_value(value) for _, value in _number_matches(context_text)}


def _extract_novel_numbers(text: str, context_text: str = "", limit: int = 3) -> list[str]:
    context_numbers = _context_number_values(context_text)
    novel: list[str] = []
    seen: set[str] = set()
    for _, value in _number_matches(text):
        normalized = _normalize_number_value(value)
        if normalized in context_numbers or normalized in seen:
            continue
        seen.add(normalized)
        novel.append(normalized)
        if len(novel) >= limit:
            break
    return novel


def _extract_first_novel_number(text: str, context_text: str = "") -> str | None:
    context_numbers = _context_number_values(context_text)
    for _, value in _number_matches(text):
        if _normalize_number_value(value) not in context_numbers:
            return value
    return None


def _extract_first_rhs_number(text: str, context_text: str = "") -> str | None:
    context_numbers = _context_number_values(context_text)
    for line in text.splitlines():
        if "=" not in line:
            continue
        rhs = line.split("=", 1)[1]
        for _, value in _number_matches(rhs):
            if _normalize_number_value(value) not in context_numbers:
                return value
    return _extract_first_novel_number(text, context_text)


def _extract_operation_intent(text: str) -> str | None:
    lowered = f" {text.lower()} "
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.startswith(" ") or keyword.endswith(" "):
                if keyword in lowered:
                    return intent
            elif re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
                return intent
    return None


def _strip_generic_anchor_phrases(text: str) -> str:
    stripped = text
    for phrase in _GENERIC_ANCHOR_PHRASES:
        stripped = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", stripped, flags=re.IGNORECASE)
    return stripped


def _normalize_anchor(value: str) -> str:
    value = clean_latex_answer(_normalize_signature_value(value)).lower()
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = re.sub(r"\s+", "", value) if re.search(r"[+\-*/^<>=|\\]", value) else re.sub(r"\s+", " ", value)
    return value.strip(" .,:;$")


def extract_content_anchors(text: str, *, limit: int = 24) -> list[str]:
    """Extract task-agnostic content anchors for conservative text consensus."""
    normalized_text = _strip_generic_anchor_phrases(text)
    anchors: list[str] = []
    seen: set[str] = set()

    def add(anchor: str) -> None:
        anchor = _normalize_anchor(anchor)
        if not anchor or anchor in seen:
            return
        if len(anchor) < 3 and not anchor.startswith(("choice:", "kw:", "expr:")):
            return
        seen.add(anchor)
        anchors.append(anchor)

    for match in _CHOICE_PATTERN.finditer(normalized_text):
        choice = match.group(1) or match.group(2)
        add(f"choice:{choice.upper()}")

    for match in _SYMBOLIC_FRAGMENT_PATTERN.finditer(normalized_text):
        add(f"expr:{match.group(0)}")

    for match in _IDENTIFIER_PATTERN.finditer(normalized_text):
        word = match.group(0)
        lowered = word.lower()
        if lowered in _CODE_KEYWORDS:
            add(f"kw:{lowered}")
            continue
        if lowered in _GENERIC_ANCHOR_WORDS:
            continue
        if len(lowered) < 3:
            continue
        add(lowered)

    return anchors[:limit]


def extract_step_evidence(text: str, context_text: str = "") -> dict[str, Any]:
    boxed = extract_last_boxed(text)
    equation = _extract_equation(text)
    rhs_number = _extract_first_rhs_number(text, context_text)
    novel_numbers = _extract_novel_numbers(text, context_text)
    operation_intent = _extract_operation_intent(text)

    boxed_answer = None
    if boxed is not None:
        cleaned = clean_latex_answer(boxed)
        if cleaned:
            boxed_answer = f"boxed_answer:{_normalize_number_value(cleaned)}"

    rhs_novel_number = None
    if rhs_number is not None:
        rhs_novel_number = f"rhs_novel_number:{_normalize_number_value(rhs_number)}"

    equation_claim = None
    if equation is not None:
        normalized_equation = _normalize_equation_claim(equation)
        if normalized_equation:
            equation_claim = f"equation_claim:{normalized_equation}"

    novel_number_set = None
    if novel_numbers:
        novel_number_set = "novel_number_set:" + "|".join(novel_numbers)

    evidence = {
        "boxed_answer": boxed_answer,
        "rhs_novel_number": rhs_novel_number,
        "equation_claim": equation_claim,
        "novel_number_set": novel_number_set,
        "operation_intent": f"operation_intent:{operation_intent}" if operation_intent else None,
        "content_anchors": extract_content_anchors(text),
    }
    evidence["evidence_channels"] = [
        channel for channel in EVIDENCE_CHANNEL_PRIORITY if evidence.get(channel) is not None
    ]
    return evidence
