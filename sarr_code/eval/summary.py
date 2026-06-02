from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def _is_evaluated(value) -> bool:
    return value is not None and str(value) != ""


def _is_correct(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _average(rows, key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return None
    return sum(values) / len(values)


def _sum_values(rows, key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return None
    return sum(values)


def build_summary_metrics(dataset: str, variant: str, rows: list[dict], dataset_wall_time: float) -> dict:
    evaluated = [row for row in rows if _is_evaluated(row.get("correct"))]
    num_correct = sum(1 for row in evaluated if _is_correct(row.get("correct")))
    return {
        "dataset": dataset,
        "variant": variant,
        "num_problems": len(rows),
        "num_evaluated": len(evaluated),
        "num_correct": num_correct,
        "accuracy": (num_correct / len(evaluated)) if evaluated else None,
        "avg_total_wall_time": _average(rows, "total_wall_time"),
        "avg_problem_wall_time": _average(rows, "problem_wall_time"),
        "dataset_wall_time": dataset_wall_time,
        "avg_step_count": _average(rows, "step_count"),
        "avg_slm_generate_calls": _average(rows, "slm_generate_calls"),
        "avg_llm_generate_calls": _average(rows, "llm_generate_calls"),
        "avg_llm_scoring_calls": _average(rows, "llm_scoring_calls"),
        "avg_llm_token_share": _average(rows, "llm_token_share"),
        "avg_llm_decode_share": _average(rows, "llm_decode_share"),
        "avg_llm_wall_time_share": _average(rows, "llm_wall_time_share"),
        "avg_slm_wall_time": _average(rows, "slm_wall_time"),
        "avg_llm_generation_wall_time": _average(rows, "llm_generation_wall_time"),
        "avg_llm_scoring_wall_time": _average(rows, "llm_scoring_wall_time"),
        "total_slm_wall_time": _sum_values(rows, "slm_wall_time"),
        "total_llm_generation_wall_time": _sum_values(rows, "llm_generation_wall_time"),
        "total_llm_scoring_wall_time": _sum_values(rows, "llm_scoring_wall_time"),
        "total_slm_decode_tokens": _sum_values(rows, "slm_decode_tokens"),
        "total_slm_prefill_tokens": _sum_values(rows, "slm_prefill_tokens"),
        "total_llm_decode_tokens": _sum_values(rows, "llm_decode_tokens"),
        "total_llm_prefill_tokens": _sum_values(rows, "llm_prefill_tokens"),
    }


def load_summary_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row["problem_id"]): row for row in csv.DictReader(f) if row.get("problem_id") not in (None, "")}


def write_summary_files(summary_path: Path, rows: list[dict], metrics: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    tmp_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with tmp_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_summary, summary_path)

    metrics_path = summary_path.parent / "summary_metrics.json"
    tmp_metrics = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with tmp_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    os.replace(tmp_metrics, metrics_path)
