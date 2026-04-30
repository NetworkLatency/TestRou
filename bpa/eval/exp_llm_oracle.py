from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.engines import init_engines
from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import load_eval_dataset
from bpa.pipeline import solve_engine_only
from bpa.trace import json_safe, result_summary


SUMMARY_FIELDS = [
    "dataset",
    "problem_id",
    "question_id",
    "gold_answer",
    "llm_answer",
    "llm_correct",
    "stop_reason",
    "total_wall_time",
    "llm_decode_tokens",
    "llm_prefill_tokens",
    "llm_generate_calls",
]


def write_oracle_outputs(out_dir: Path, summary_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "oracle_summary.csv"
    fieldnames = list(SUMMARY_FIELDS)
    extra_fields = sorted({key for row in summary_rows for key in row} - set(fieldnames))
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(json_safe(summary_rows))

    traces_path = out_dir / "oracle_traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-only oracle traces for boundary-label diagnostics.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    _, llm = init_engines(config)

    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for problem in tqdm(problems, desc=f"llm_oracle:{args.dataset}"):
        result = solve_engine_only(problem.problem_text, llm, config, account="llm")
        correct = None
        if problem.gold_answer is not None:
            correct = benchmark_eval_match(result.answer, problem.gold_answer, args.dataset)
            result.correct = correct
        summary = result_summary(result, config.slm_to_llm_flop_ratio)
        summary_rows.append(
            {
                "dataset": args.dataset,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                "gold_answer": problem.gold_answer,
                "llm_answer": result.answer,
                "llm_correct": correct,
                "stop_reason": result.state.stop_reason,
                "total_wall_time": result.total_wall_time,
                "llm_decode_tokens": result.llm_decode_tokens,
                "llm_prefill_tokens": result.llm_prefill_tokens,
                "llm_generate_calls": result.llm_full_calls,
                "llm_total_tokens": summary["llm_total_tokens"],
            }
        )
        trace_rows.append(
            {
                "dataset": args.dataset,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                "gold_answer": problem.gold_answer,
                "llm_answer": result.answer,
                "llm_correct": correct,
                "llm_text": result.state.assistant_prefix_text,
                "summary": summary,
            }
        )
        if config.reset_prefix_cache_after_problem:
            llm.clear_runtime_cache()

    out_dir = Path(config.output_dir) / "diagnostics" / "llm_oracle" / args.dataset
    write_oracle_outputs(out_dir, summary_rows, trace_rows)
    print(f"Wrote {out_dir / 'oracle_summary.csv'}")
    print(f"Wrote {out_dir / 'oracle_traces.jsonl'}")


if __name__ == "__main__":
    main()
