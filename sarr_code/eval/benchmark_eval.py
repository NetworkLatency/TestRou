from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from typing import Optional

from sarr_code.safety import clean_latex_answer, extract_choice_letter, extract_last_boxed


def strip_outer_wrappers(s: str) -> str:
    pairs = {"(": ")", "{": "}", "[": "]"}

    def is_single_wrapped(x: str, left: str, right: str) -> bool:
        if not (x.startswith(left) and x.endswith(right)):
            return False
        depth = 0
        for i, ch in enumerate(x):
            if ch == left:
                depth += 1
            elif ch == right:
                depth -= 1
                if depth == 0 and i != len(x) - 1:
                    return False
        return depth == 0

    changed = True
    while changed and s:
        changed = False
        for left, right in pairs.items():
            if is_single_wrapped(s, left, right):
                s = s[1:-1].strip()
                changed = True
    return s


def normalize_math_expr(expr: Optional[str]) -> Optional[str]:
    if expr is None:
        return None
    s = clean_latex_answer(expr)
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = strip_outer_wrappers(s)
    s = re.sub(r"\^\{?\\circ\}?$", "", s)
    s = re.sub(r"\^\{?circ\}?$", "", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = strip_outer_wrappers(s)
    if re.fullmatch(r"[+-]?\d+", s):
        return str(int(s))
    frac_patterns = [
        re.compile(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}$"),
        re.compile(r"([+-]?\d+)/([+-]?\d+)$"),
    ]
    for pattern in frac_patterns:
        match = pattern.fullmatch(s)
        if match:
            den = int(match.group(2))
            if den == 0:
                return s
            frac = Fraction(int(match.group(1)), den)
            return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
    return s


def _math_verify_match(pred: str, gold: str) -> bool | None:
    try:
        from math_verify import parse, verify
    except Exception:
        return None
    try:
        pred_parsed = parse(pred)
        gold_parsed = parse(gold)
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return None


def _sympy_match(pred: str, gold: str) -> bool | None:
    try:
        import sympy as sp
    except Exception:
        return None
    try:
        p = sp.sympify(pred.replace("^", "**"))
        g = sp.sympify(gold.replace("^", "**"))
        return bool(sp.simplify(p - g) == 0)
    except Exception:
        return None


def math_match(predicted_text: str | None, ground_truth: str | None) -> bool:
    if predicted_text is None or ground_truth is None:
        return False
    boxed = extract_last_boxed(predicted_text) or predicted_text
    pred_norm = normalize_math_expr(boxed)
    gold_norm = normalize_math_expr(ground_truth)
    if pred_norm is None or gold_norm is None:
        return False
    verified = _math_verify_match(pred_norm, gold_norm)
    if verified is not None:
        return verified
    sympy_verified = _sympy_match(pred_norm, gold_norm)
    if sympy_verified is not None:
        return sympy_verified
    return pred_norm == gold_norm


def gpqa_match(predicted_text: str | None, ground_truth: str | None) -> bool:
    if predicted_text is None or ground_truth is None:
        return False
    pred = extract_choice_letter(predicted_text)
    gold = extract_choice_letter(str(ground_truth)) or str(ground_truth).strip().upper()[:1]
    return pred is not None and pred == gold


def _extract_python_code(text: str, entry_point: str) -> str:
    fence = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S | re.I)
    if fence:
        return fence.group(1).strip()
    if entry_point:
        pattern = re.compile(rf"(^|\n)(def\s+{re.escape(entry_point)}\s*\()", flags=re.S)
        match = pattern.search(text)
        if match:
            return text[match.start(2) :].strip()
    generic = re.search(r"(^|\n)(def\s+\w+\s*\()", text, flags=re.S)
    if generic:
        return text[generic.start(2) :].strip()
    return text.strip()


def _strip_humaneval_tail(code: str) -> str:
    markers = [
        "\nif __name__",
        "\n# Test",
        "\n# Example",
        "\nprint(",
        "\nassert ",
    ]
    cleaned = code.strip().strip("`").strip()
    for marker in markers:
        idx = cleaned.find(marker)
        if idx > 0:
            cleaned = cleaned[:idx].rstrip()
    return cleaned


def humaneval_match(predicted_text: str | None, ground_truth: str | None, timeout_s: float = 8.0) -> bool:
    if predicted_text is None or ground_truth is None:
        return False
    try:
        payload = json.loads(ground_truth)
    except Exception:
        return False
    prompt = str(payload.get("prompt") or "")
    tests = str(payload.get("test") or "")
    entry_point = str(payload.get("entry_point") or "")
    if not prompt or not tests or not entry_point:
        return False

    completion = _strip_humaneval_tail(_extract_python_code(predicted_text, entry_point))
    if re.search(rf"(^|\n)def\s+{re.escape(entry_point)}\s*\(", completion):
        program = completion
    else:
        separator = "" if prompt.endswith((" ", "\t", "\n")) else "\n"
        program = prompt + separator + completion.rstrip() + "\n"
    program = program.rstrip() + "\n\n" + tests.rstrip() + f"\n\ncheck({entry_point})\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "candidate.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            completed = subprocess.run(
                [sys.executable, script_path],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False
    return completed.returncode == 0


def benchmark_eval_match(predicted: str | None, ground_truth: str | None, dataset: str) -> bool:
    if dataset in {"math500", "aime24", "aime25"}:
        return math_match(predicted, ground_truth)
    if dataset in {"gpqa", "gpqa_diamond"}:
        return gpqa_match(predicted, ground_truth)
    if dataset == "humaneval":
        return humaneval_match(predicted, ground_truth)
    raise ValueError(f"Evaluator is not implemented for dataset {dataset!r}.")
