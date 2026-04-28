from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.engines import init_engines
from bpa.trace import result_summary, write_json, write_jsonl

from .baselines import solve_variant
from .benchmark_eval import benchmark_eval_match
from .datasets import load_eval_dataset

VARIANTS = ("slm_only", "llm_only", "glimprouter_hinit", "bpa_logging_only", "bpa_arbitration")
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
    }


def _step_rows(result):
    for event in result.state.trace:
        if event.event == "step_logs":
            return event.data.get("steps", [])
    return []


def write_problem_outputs(root: Path, dataset: str, variant: str, problem, result, config: BPAConfig) -> None:
    problem_root = root / dataset / variant / str(problem.problem_id)
    write_json(problem_root / f"{problem.problem_id}.problem.json", {"raw": problem.raw, **result_summary(result, config.slm_to_llm_flop_ratio)})
    step_rows = [
        {"problem_id": problem.problem_id, "question_id": problem.question_id, **row}
        for row in _step_rows(result)
    ]
    branch_rows = [
        {"problem_id": problem.problem_id, "question_id": problem.question_id, **row}
        for row in result.state.branch_logs
    ]
    write_jsonl(problem_root / f"{problem.problem_id}.steps.jsonl", step_rows)
    write_jsonl(problem_root / f"{problem.problem_id}.branches.jsonl", branch_rows)
    write_json(problem_root / f"{problem.problem_id}.trace.json", result.state.trace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BPA v2.1 benchmarks.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    slm, llm = init_engines(config)
    rows = []
    output_root = Path(config.output_dir)
    dataset_start = time.time()

    for problem in tqdm(problems, desc=f"{args.variant}:{args.dataset}"):
        problem_start = time.time()
        result = solve_variant(problem.problem_text, args.variant, slm, llm, config)
        if problem.gold_answer is not None:
            predicted = result.answer if args.dataset in MATH_DATASETS else result.state.assistant_prefix_text
            result.correct = benchmark_eval_match(predicted, problem.gold_answer, args.dataset)
        summary = result_summary(result, config.slm_to_llm_flop_ratio)
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
        rows.append(summary)

    dataset_wall_time = time.time() - dataset_start

    summary_path = output_root / args.dataset / args.variant / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = summary_path.parent / "summary_metrics.json"
    write_json(metrics_path, build_summary_metrics(args.dataset, args.variant, rows, dataset_wall_time))
    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
