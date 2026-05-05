from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.engines import init_engines
from bpa.trace import result_summary, write_json, write_jsonl

from .baselines import solve_variant
from .benchmark_eval import benchmark_eval_match
from .datasets import load_eval_dataset

VARIANTS = ("slm_only", "llm_only", "glimprouter_hinit")
MATH_DATASETS = {"math500", "aime24", "aime25"}


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


def _summary_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary.csv"


def _metrics_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary_metrics.json"


def _problem_root(output_root: Path, dataset: str, variant: str, problem_id) -> Path:
    return output_root / dataset / variant / str(problem_id)


def _problem_output_paths(output_root: Path, dataset: str, variant: str, problem_id) -> list[Path]:
    root = _problem_root(output_root, dataset, variant, problem_id)
    stem = str(problem_id)
    return [
        root / f"{stem}.problem.json",
        root / f"{stem}.steps.jsonl",
        root / f"{stem}.trace.json",
    ]


def has_complete_problem_outputs(output_root: Path, dataset: str, variant: str, problem_id) -> bool:
    return all(path.exists() for path in _problem_output_paths(output_root, dataset, variant, problem_id))


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


def _existing_row_from_problem_output(
    output_root: Path,
    dataset: str,
    variant: str,
    problem,
    config: BPAConfig,
    existing_summary_row: dict | None = None,
) -> dict | None:
    problem_json = _problem_root(output_root, dataset, variant, problem.problem_id) / f"{problem.problem_id}.problem.json"
    if not problem_json.exists():
        if existing_summary_row is not None:
            return existing_summary_row
        return None
    with problem_json.open("r", encoding="utf-8") as f:
        saved = json.load(f)

    if existing_summary_row is not None:
        row = dict(existing_summary_row)
        if problem.gold_answer is not None:
            row["correct"] = benchmark_eval_match(row.get("answer"), problem.gold_answer, dataset)
        if row.get("problem_wall_time") in (None, ""):
            row["problem_wall_time"] = saved.get("problem_wall_time") or saved.get("total_wall_time")
        return row

    row = {
        key: saved.get(key)
        for key in [
            "answer",
            "correct",
            "generation_protocol",
            "step_count",
            "total_wall_time",
            "slm_decode_tokens",
            "slm_prefill_tokens",
            "llm_decode_tokens",
            "llm_prefill_tokens",
            "slm_total_tokens",
            "llm_total_tokens",
            "total_model_tokens",
            "llm_token_share",
            "llm_decode_share",
            "slm_generate_calls",
            "llm_generate_calls",
            "llm_scoring_calls",
            "llm_full_calls",
            "slm_wall_time",
            "llm_generation_wall_time",
            "llm_scoring_wall_time",
            "llm_wall_time",
            "model_wall_time",
            "llm_wall_time_share",
            "stop_reason",
        ]
    }

    def numeric(key: str) -> float:
        value = row.get(key)
        if value in (None, ""):
            return 0.0
        return float(value)

    slm_total = numeric("slm_decode_tokens") + numeric("slm_prefill_tokens")
    llm_total = numeric("llm_decode_tokens") + numeric("llm_prefill_tokens")
    total_model_tokens = slm_total + llm_total
    total_decode_tokens = numeric("slm_decode_tokens") + numeric("llm_decode_tokens")
    llm_wall_time = numeric("llm_generation_wall_time") + numeric("llm_scoring_wall_time")
    model_wall_time = numeric("slm_wall_time") + llm_wall_time

    if row.get("generation_protocol") is None:
        row["generation_protocol"] = "oneshot" if variant in {"slm_only", "llm_only"} else "routed_stepwise"
    if row.get("slm_total_tokens") is None:
        row["slm_total_tokens"] = slm_total
    if row.get("llm_total_tokens") is None:
        row["llm_total_tokens"] = llm_total
    if row.get("total_model_tokens") is None:
        row["total_model_tokens"] = total_model_tokens
    if row.get("llm_token_share") is None:
        row["llm_token_share"] = (llm_total / total_model_tokens) if total_model_tokens else 0.0
    if row.get("llm_decode_share") is None:
        row["llm_decode_share"] = (numeric("llm_decode_tokens") / total_decode_tokens) if total_decode_tokens else 0.0
    if row.get("llm_generate_calls") is None:
        row["llm_generate_calls"] = row.get("llm_full_calls")
    if row.get("llm_wall_time") is None:
        row["llm_wall_time"] = llm_wall_time
    if row.get("model_wall_time") is None:
        row["model_wall_time"] = model_wall_time
    if row.get("llm_wall_time_share") is None:
        row["llm_wall_time_share"] = (llm_wall_time / model_wall_time) if model_wall_time else 0.0
    if problem.gold_answer is not None:
        predicted = row.get("answer")
        row["correct"] = benchmark_eval_match(predicted, problem.gold_answer, dataset)
    row.update(
        {
            "dataset": dataset,
            "variant": variant,
            "problem_id": problem.problem_id,
            "question_id": problem.question_id,
            "gold_answer": problem.gold_answer,
            "problem_wall_time": saved.get("problem_wall_time") or saved.get("total_wall_time"),
        }
    )
    return row


def _step_rows(result):
    for event in result.state.trace:
        if event.event == "step_logs":
            return event.data.get("steps", [])
    return []


def write_problem_outputs(root: Path, dataset: str, variant: str, problem, result, config: BPAConfig) -> None:
    problem_root = root / dataset / variant / str(problem.problem_id)
    write_json(problem_root / f"{problem.problem_id}.problem.json", {"raw": problem.raw, **result_summary(result)})
    step_rows = [
        {"problem_id": problem.problem_id, "question_id": problem.question_id, **row}
        for row in _step_rows(result)
    ]
    write_jsonl(problem_root / f"{problem.problem_id}.steps.jsonl", step_rows)
    write_json(problem_root / f"{problem.problem_id}.trace.json", result.state.trace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BPA v2.1 benchmarks.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip problems that already have complete per-problem outputs.")
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    slm, llm = init_engines(config)
    output_root = Path(config.output_dir)
    summary_path = _summary_path(output_root, args.dataset, args.variant)
    existing_summary_rows = load_summary_rows(summary_path) if args.resume else {}
    rows_by_problem_id: dict[str, dict] = {}
    skipped = 0
    if args.resume:
        for problem in problems:
            problem_id = str(problem.problem_id)
            if has_complete_problem_outputs(output_root, args.dataset, args.variant, problem.problem_id):
                row = _existing_row_from_problem_output(
                    output_root,
                    args.dataset,
                    args.variant,
                    problem,
                    config,
                    existing_summary_rows.get(problem_id),
                )
                if row is not None:
                    rows_by_problem_id[problem_id] = row
                    skipped += 1

    def ordered_rows() -> list[dict]:
        return [rows_by_problem_id[str(problem.problem_id)] for problem in problems if str(problem.problem_id) in rows_by_problem_id]

    dataset_start = time.time()

    for problem in tqdm(problems, desc=f"{args.variant}:{args.dataset}"):
        problem_id = str(problem.problem_id)
        if args.resume and problem_id in rows_by_problem_id:
            continue
        problem_start = time.time()
        result = solve_variant(problem.problem_text, args.variant, slm, llm, config)
        if problem.gold_answer is not None:
            predicted = result.answer if args.dataset in MATH_DATASETS else result.state.assistant_prefix_text
            result.correct = benchmark_eval_match(predicted, problem.gold_answer, args.dataset)
        summary = result_summary(result)
        summary.update(
            {
                "dataset": args.dataset,
                "variant": args.variant,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                "gold_answer": problem.gold_answer,
            }
        )
        write_problem_outputs(output_root, args.dataset, args.variant, problem, result, config)
        if config.reset_prefix_cache_after_problem:
            slm.clear_runtime_cache()
            llm.clear_runtime_cache()
        summary["problem_wall_time"] = time.time() - problem_start
        rows_by_problem_id[problem_id] = summary
        rows = ordered_rows()
        write_summary_files(
            summary_path,
            rows,
            build_summary_metrics(args.dataset, args.variant, rows, time.time() - dataset_start),
        )

    dataset_wall_time = time.time() - dataset_start
    rows = ordered_rows()
    write_summary_files(summary_path, rows, build_summary_metrics(args.dataset, args.variant, rows, dataset_wall_time))
    metrics_path = _metrics_path(output_root, args.dataset, args.variant)
    if args.resume:
        print(f"Skipped {skipped} completed problem(s).")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
