from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompts import build_problem_text


@dataclass
class EvalProblem:
    problem_id: int
    problem_text: str
    raw: dict[str, Any]
    gold_answer: str | None
    question_id: str | None = None


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for benchmark loading.") from exc
    return load_dataset


def _gold(row: dict[str, Any], dataset_name: str) -> str | None:
    if dataset_name in {"aime24", "aime25", "math500"}:
        value = row.get("answer") or row.get("solution") or row.get("target")
        return str(value).strip() if value is not None else None
    if dataset_name in {"gpqa", "gpqa_diamond"}:
        for key in ["answer", "correct_answer", "Correct Answer", "label", "target"]:
            if row.get(key) is not None:
                if key == "Correct Answer":
                    return "A"
                return str(row[key]).strip()
    return None


def load_eval_dataset(dataset_name: str, config, max_problems: int | None = None) -> list[EvalProblem]:
    load_dataset = _require_datasets()
    if dataset_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset_name == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name == "aime25":
        data_file = config.dataset_paths.get("aime25", "data/aime25.parquet")
        ds = load_dataset("parquet", data_files=data_file, split="train")
    elif dataset_name in {"gpqa", "gpqa_diamond"}:
        data_file = config.dataset_paths.get("gpqa") or config.dataset_paths.get("gpqa_diamond")
        if not data_file:
            raise ValueError("Set dataset_paths.gpqa or dataset_paths.gpqa_diamond in the config.")
        ds = load_dataset("json", data_files=data_file, split="train")
    elif dataset_name == "humaneval":
        data_file = config.dataset_paths.get("humaneval", "data/HumanEval.jsonl")
        ds = load_dataset("json", data_files=data_file, split="train")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    problems: list[EvalProblem] = []
    limit = len(ds) if max_problems is None else min(len(ds), max_problems)
    for idx in range(limit):
        raw = dict(ds[idx])
        problems.append(
            EvalProblem(
                problem_id=int(raw.get("id", idx)) if str(raw.get("id", idx)).isdigit() else idx,
                problem_text=build_problem_text(raw, dataset_name),
                raw=raw,
                gold_answer=_gold(raw, dataset_name),
                question_id=str(raw.get("question_id")) if raw.get("question_id") is not None else None,
            )
        )
    return problems
