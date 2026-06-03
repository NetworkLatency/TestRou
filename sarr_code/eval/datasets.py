from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import build_problem_text


@dataclass
class EvalProblem:
    problem_id: int
    problem_text: str
    raw: dict[str, Any]
    gold_answer: str | None
    question_id: str | None = None


DEFAULT_DATASET_PATHS = {
    "math500": "data/math500.jsonl",
    "aime24": "data/aime24.jsonl",
    "aime25": "data/aime25.parquet",
    "gpqa": "data/gpqa_diamond.jsonl",
    "gpqa_diamond": "data/gpqa_diamond.jsonl",
    "humaneval": "data/HumanEval.jsonl",
}


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
    if dataset_name == "humaneval":
        payload = {
            "prompt": row.get("prompt") or "",
            "test": row.get("test") or "",
            "entry_point": row.get("entry_point") or "",
            "task_id": row.get("task_id") or row.get("id"),
        }
        return json.dumps(payload, ensure_ascii=False)
    return None


def _dataset_path(dataset_name: str, config) -> Path:
    if dataset_name not in DEFAULT_DATASET_PATHS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    configured = config.dataset_paths.get(dataset_name)
    if configured is None and dataset_name == "gpqa_diamond":
        configured = config.dataset_paths.get("gpqa")
    if configured is None and dataset_name == "gpqa":
        configured = config.dataset_paths.get("gpqa_diamond")
    path = Path(configured or DEFAULT_DATASET_PATHS[dataset_name])
    if not path.exists():
        raise FileNotFoundError(
            f"Local dataset file not found for {dataset_name!r}: {path}. "
            f"Set dataset_paths.{dataset_name} in the config to a local file."
        )
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object per line.")
            rows.append(value)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        for key in ["data", "train", "test", "examples", "rows"]:
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        else:
            rows = [value]
    else:
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSON dataset rows must be objects: {path}")
    return list(rows)


def _read_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required to load local parquet datasets.") from exc
    return pd.read_parquet(path).to_dict(orient="records")


def load_local_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(source)
    if suffix == ".json":
        return _read_json(source)
    if suffix == ".csv":
        return _read_delimited(source, ",")
    if suffix == ".tsv":
        return _read_delimited(source, "\t")
    if suffix == ".parquet":
        return _read_parquet(source)
    raise ValueError(f"Unsupported local dataset format {suffix!r}: {source}")


def load_eval_dataset(dataset_name: str, config, max_problems: int | None = None) -> list[EvalProblem]:
    path = _dataset_path(dataset_name, config)
    ds = load_local_rows(path)

    problems: list[EvalProblem] = []
    limit = len(ds) if max_problems is None else min(len(ds), max_problems)
    for idx in range(limit):
        raw = dict(ds[idx])
        raw_id = raw.get("id", raw.get("task_id", idx))
        if isinstance(raw_id, str) and raw_id.startswith("HumanEval/"):
            raw_id = raw_id.rsplit("/", 1)[-1]
        problems.append(
            EvalProblem(
                problem_id=int(raw_id) if str(raw_id).isdigit() else idx,
                problem_text=build_problem_text(raw, dataset_name),
                raw=raw,
                gold_answer=_gold(raw, dataset_name),
                question_id=str(raw.get("question_id")) if raw.get("question_id") is not None else None,
            )
        )
    return problems
