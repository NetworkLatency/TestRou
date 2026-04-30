from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from typing import Any

from bpa.safety import clean_latex_answer, extract_last_boxed


OPERATOR_KEYWORDS = (
    "let",
    "substitute",
    "solve",
    "equate",
    "simplify",
    "calculate",
    "compute",
    "pythagorean",
    "factor",
    "expand",
    "differentiate",
    "integrate",
)

_LATEX_NUMBER_PATTERNS = (
    re.compile(r"\\(?:d?frac)\s*\{[^{}]+\}\s*\{[^{}]+\}"),
    re.compile(r"\\sqrt\s*\{?[-+]?\d+(?:\.\d+)?\}?"),
)
_PLAIN_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?%?)(?![A-Za-z])")
_OPERATOR_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in OPERATOR_KEYWORDS) + r")\b", re.IGNORECASE)
_BLEU_TOKEN_PATTERN = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|[-+]?\d+(?:\.\d+)?|[^\s]")


def _normalize_signature_value(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\r\n$.,;:")
    return value


def _signature(signature_type: str, value: str) -> dict[str, str]:
    normalized = _normalize_signature_value(value)
    if signature_type == "boxed":
        normalized = clean_latex_answer(normalized)
    return {
        "signature_type": signature_type,
        "signature_value": normalized,
        "signature": f"{signature_type}:{normalized}",
    }


def _extract_equation(text: str) -> str | None:
    for line in text.splitlines():
        if "=" not in line:
            continue
        line = line.strip()
        if not line:
            continue
        match = re.search(r"([^.;\n]{0,80}=[^.;\n]{0,80})", line)
        if match:
            return _normalize_signature_value(match.group(1))
        return _normalize_signature_value(line[:160])
    return None


def _extract_first_number(text: str) -> str | None:
    matches: list[re.Match[str]] = []
    for pattern in _LATEX_NUMBER_PATTERNS:
        matches.extend(pattern.finditer(text))
    matches.extend(_PLAIN_NUMBER_PATTERN.finditer(text))
    if not matches:
        return None
    first = min(matches, key=lambda match: match.start())
    return _normalize_signature_value(first.group(0))


def _extract_first_operator(text: str) -> str | None:
    operator = _OPERATOR_PATTERN.search(text)
    return operator.group(1).lower() if operator is not None else None


def extract_structured_signature(text: str) -> dict[str, str]:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return _signature("boxed", boxed)

    equation = _extract_equation(text)
    if equation is not None:
        return _signature("equation", equation)

    number = _extract_first_number(text)
    if number is not None:
        return _signature("number", number)

    operator = _extract_first_operator(text)
    if operator is not None:
        return _signature("operator", operator)

    return _signature("none", "")


def extract_number_signature(text: str) -> dict[str, str]:
    number = _extract_first_number(text)
    if number is None:
        return _signature("none", "")
    return _signature("number", number)


def extract_operation_signature(text: str) -> dict[str, str]:
    operator = _extract_first_operator(text)
    if operator is None:
        return _signature("none", "")
    return _signature("operator", operator)


def _signature_key(value: str | dict[str, Any]) -> str:
    if isinstance(value, dict):
        signature = value.get("signature")
        if signature is None:
            signature_type = value.get("signature_type", "none")
            signature_value = value.get("signature_value", "")
            return f"{signature_type}:{signature_value}"
        return str(signature)
    return str(value)


def compute_vote_disagreement(signatures: list[str] | list[dict[str, Any]]) -> dict[str, Any]:
    keys = [_signature_key(signature) for signature in signatures]
    if not keys:
        return {
            "signature_counts": {},
            "majority_signature": None,
            "vote_fraction": None,
            "structured_disagreement": None,
        }
    counts = Counter(keys)
    majority_signature, majority_count = counts.most_common(1)[0]
    vote_fraction = majority_count / len(keys)
    return {
        "signature_counts": dict(counts),
        "majority_signature": majority_signature,
        "vote_fraction": vote_fraction,
        "structured_disagreement": 1.0 - vote_fraction,
    }


def _renamed_vote_result(result: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_signature_counts": result["signature_counts"],
        f"{prefix}_majority_signature": result["majority_signature"],
        f"{prefix}_vote_fraction": result["vote_fraction"],
        f"{prefix}_vote_disagreement": result["structured_disagreement"],
    }


def operation_vote_disagreement(texts: list[str]) -> dict[str, Any]:
    signatures = [extract_operation_signature(text)["signature"] for text in texts]
    return _renamed_vote_result(compute_vote_disagreement(signatures), "operation")


def number_vote_disagreement(texts: list[str]) -> dict[str, Any]:
    signatures = [extract_number_signature(text)["signature"] for text in texts]
    return _renamed_vote_result(compute_vote_disagreement(signatures), "number")


def _char_units(text: str) -> set[str]:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return set()
    if len(compact) < 3:
        return set(compact)
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def char_jaccard_disagreement(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None

    distances: list[float] = []
    for left, right in combinations(texts, 2):
        left_units = _char_units(left)
        right_units = _char_units(right)
        union = left_units | right_units
        if not union:
            distances.append(0.0)
            continue
        distances.append(1.0 - (len(left_units & right_units) / len(union)))
    return sum(distances) / len(distances) if distances else None


def _bleu_tokens(text: str) -> list[str]:
    return [token.lower() for token in _BLEU_TOKEN_PATTERN.findall(text)]


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _closest_ref_len(hyp_len: int, refs: list[list[str]]) -> int:
    return min((len(ref) for ref in refs), key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))


def _sentence_bleu(hypothesis: list[str], references: list[list[str]], max_n: int = 4) -> float:
    if not hypothesis or not references:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hypothesis, n)
        if not hyp_counts:
            precisions.append(1.0)
            continue
        max_ref_counts: Counter[tuple[str, ...]] = Counter()
        for ref in references:
            ref_counts = _ngram_counts(ref, n)
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(max_ref_counts[gram], count)
        overlap = sum(min(count, max_ref_counts[gram]) for gram, count in hyp_counts.items())
        # Add-one smoothing keeps short mathematical rollouts from collapsing to zero
        # when they differ in one symbol.
        precisions.append((overlap + 1.0) / (sum(hyp_counts.values()) + 1.0))

    hyp_len = len(hypothesis)
    ref_len = _closest_ref_len(hyp_len, references)
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_n)


def self_bleu_disagreement(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None
    tokenized = [_bleu_tokens(text) for text in texts]
    scores = []
    for idx, hypothesis in enumerate(tokenized):
        references = [tokens for ref_idx, tokens in enumerate(tokenized) if ref_idx != idx]
        scores.append(_sentence_bleu(hypothesis, references))
    self_bleu = sum(scores) / len(scores) if scores else 0.0
    return 1.0 - self_bleu


def score_variance(scores: list[float | None]) -> float | None:
    values = []
    for score in scores:
        if score is None:
            continue
        value = float(score)
        if math.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def rollout_disagreement_metrics(texts: list[str], scores: list[float | None]) -> dict[str, Any]:
    structured = compute_vote_disagreement([extract_structured_signature(text)["signature"] for text in texts])
    return {
        "signature_counts": structured["signature_counts"],
        "majority_signature": structured["majority_signature"],
        "vote_fraction": structured["vote_fraction"],
        "structured_disagreement": structured["structured_disagreement"],
        **operation_vote_disagreement(texts),
        **number_vote_disagreement(texts),
        "self_bleu_disagreement": self_bleu_disagreement(texts),
        "char_jaccard_disagreement": char_jaccard_disagreement(texts),
        "score_variance": score_variance(scores),
    }
