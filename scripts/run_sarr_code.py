#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import load_eval_dataset
from bpa.eval.main_benchmark import build_summary_metrics, load_summary_rows, write_summary_files
from bpa.trace import result_summary, write_json, write_jsonl
from sarr_code import SARRConfig, run_sarr_code


MATH_DATASETS = {"math500", "aime24", "aime25"}
DEFAULT_VARIANT = "pdi_step_window_controller_v0"


def _summary_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary.csv"


def _metrics_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary_metrics.json"


def _problem_root(output_root: Path, dataset: str, variant: str, problem_id: Any) -> Path:
    return output_root / dataset / variant / str(problem_id)


def _problem_complete(output_root: Path, dataset: str, variant: str, problem_id: Any) -> bool:
    root = _problem_root(output_root, dataset, variant, problem_id)
    stem = str(problem_id)
    required = [
        root / f"{stem}.problem.json",
        root / f"{stem}.steps.jsonl",
        root / f"{stem}.controller_events.jsonl",
        root / f"{stem}.transitions.jsonl",
        root / f"{stem}.trace.json",
    ]
    return all(path.exists() for path in required)


def _write_problem_outputs(
    output_root: Path,
    dataset: str,
    variant: str,
    problem,
    result,
    cfg: SARRConfig,
    *,
    step_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    problem_wall_time: float,
) -> None:
    root = _problem_root(output_root, dataset, variant, problem.problem_id)
    stem = str(problem.problem_id)
    summary = result_summary(result)
    metric_fields = _problem_sarr_metrics(result, step_rows, controller_rows)
    summary.update(
        {
            "raw": problem.raw,
            "dataset": dataset,
            "variant": variant,
            "problem_id": problem.problem_id,
            "question_id": problem.question_id,
            "gold_answer": problem.gold_answer,
            "problem_wall_time": problem_wall_time,
            "method": cfg.method,
            **metric_fields,
        }
    )
    write_json(root / f"{stem}.problem.json", summary)
    write_jsonl(
        root / f"{stem}.steps.jsonl",
        [
            {"problem_id": problem.problem_id, "question_id": problem.question_id, **row}
            for row in step_rows
        ],
    )
    write_jsonl(
        root / f"{stem}.controller_events.jsonl",
        [
            {"problem_id": problem.problem_id, "question_id": problem.question_id, **row}
            for row in controller_rows
        ],
    )
    write_jsonl(
        root / f"{stem}.transitions.jsonl",
        [
            {"problem_id": problem.problem_id, "question_id": problem.question_id, **row}
            for row in transition_rows
        ],
    )
    write_json(root / f"{stem}.trace.json", result.state.trace)


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _num(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _sarr_summary_from_trace(result) -> dict[str, Any]:
    for event in reversed(result.state.trace):
        if event.event == "sarr_summary":
            return dict(event.data)
    return {}


def _problem_sarr_metrics(result, step_rows: list[dict[str, Any]], controller_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _sarr_summary_from_trace(result)
    active_steps = [row for row in step_rows if _truthy(row.get("active"))]
    active_slm_tokens = sum(int(_num(row.get("token_count"))) for row in active_steps if row.get("source") == "SLM")
    active_llm_tokens = sum(int(_num(row.get("token_count"))) for row in active_steps if row.get("source") == "LLM")
    if not summary:
        summary = {
            "driver_switch_count": sum(1 for row in controller_rows if row.get("event") == "driver_switch"),
            "rollback_count": sum(1 for row in controller_rows if row.get("event") == "rollback"),
            "slm_thinking_tokens": active_slm_tokens,
            "llm_thinking_tokens": active_llm_tokens,
            "total_thinking_tokens": active_slm_tokens + active_llm_tokens,
        }
    return {
        "active_thinking_step_count": len(active_steps),
        "generated_thinking_attempt_count": len(step_rows),
        "slm_generated_thinking_tokens": int(_num(summary.get("slm_thinking_tokens") or active_slm_tokens)),
        "llm_generated_thinking_tokens": int(_num(summary.get("llm_thinking_tokens") or active_llm_tokens)),
        "total_generated_thinking_tokens": int(
            _num(summary.get("total_thinking_tokens") or active_slm_tokens + active_llm_tokens)
        ),
        "controller_mode": summary.get("controller_mode") or "pdi_step_window",
        "driver_switch_count": int(_num(summary.get("driver_switch_count"))),
        "llm_ownership_episodes": int(_num(summary.get("llm_ownership_episodes"))),
        "llm_repair_episodes": int(_num(summary.get("llm_repair_episodes"))),
        "rollback_count": int(_num(summary.get("rollback_count"))),
        "has_rollback": int(_num(summary.get("rollback_count"))) > 0,
        "handoff_attempt_count": int(_num(summary.get("handoff_attempt_count"))),
        "handoff_success_count": int(_num(summary.get("handoff_success_count"))),
        "handoff_failure_count": int(_num(summary.get("handoff_failure_count"))),
        "handoff_success_rate": float(_num(summary.get("handoff_success_rate"))),
        "probation_failure_count": int(_num(summary.get("probation_failure_count"))),
        "probation_failure_rate": float(_num(summary.get("probation_failure_rate"))),
        "reentry_failure_count": int(_num(summary.get("reentry_failure_count"))),
        "reentry_failure_rate": float(_num(summary.get("reentry_failure_rate"))),
        "early_stop_trigger_count": int(_num(summary.get("early_stop_trigger_count"))),
        "pdi_decision_count": int(_num(summary.get("pdi_decision_count"))),
        "no_valid_pdi_window_count": int(_num(summary.get("no_valid_pdi_window_count"))),
        "pdi_window_count": int(_num(summary.get("pdi_window_count"))),
        "trusted_buffer_size": int(_num(summary.get("trusted_buffer_size"))),
        "failure_buffer_size": int(_num(summary.get("failure_buffer_size"))),
        "prior_size": int(_num(summary.get("prior_size"))),
        "prior_weight": float(_num(summary.get("prior_weight"))),
        "llm_participation_rate": float(_num(summary.get("llm_participation_rate"))),
        "slm_scoring_overhead": float(_num(summary.get("slm_scoring_overhead"))),
        "slm_scoring_count": int(_num(summary.get("slm_scoring_count"))),
        "self_reentry_attempt_count": int(_num(summary.get("self_reentry_attempt_count"))),
        "self_reentry_accept_count": int(_num(summary.get("self_reentry_accept_count"))),
        "self_reentry_reject_count": int(_num(summary.get("self_reentry_reject_count"))),
        "self_reentry_accept_rate": float(_num(summary.get("self_reentry_accept_rate"))),
        "avg_self_reentry_pdi": summary.get("avg_self_reentry_pdi"),
        "avg_self_reentry_q": summary.get("avg_self_reentry_q"),
        "self_reentry_reject_reasons": summary.get("self_reentry_reject_reasons"),
        "shadow_old_handoff_ready_count": int(_num(summary.get("shadow_old_handoff_ready_count"))),
        "slm_thinking_tokens": int(_num(summary.get("slm_thinking_tokens"))),
        "llm_thinking_tokens": int(_num(summary.get("llm_thinking_tokens"))),
        "total_thinking_tokens": int(_num(summary.get("total_thinking_tokens"))),
        "slm_step_count": int(_num(summary.get("slm_step_count"))),
        "llm_step_count": int(_num(summary.get("llm_step_count"))),
        "slm_prefill_count": int(_num(summary.get("slm_prefill_count"))),
        "llm_prefill_count": int(_num(summary.get("llm_prefill_count"))),
        "final_answer_generator": summary.get("final_answer_generator"),
        "finish_reason": summary.get("finish_reason") or result.state.stop_reason,
    }


def _extra_sarr_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        return sum(vals) / len(vals) if vals else None

    def total(key: str) -> int:
        return sum(int(_num(row.get(key))) for row in rows)

    n = len(rows)
    return {
        "avg_driver_switch_count": avg("driver_switch_count"),
        "avg_llm_ownership_episodes": avg("llm_ownership_episodes"),
        "avg_handoff_attempt_count": avg("handoff_attempt_count"),
        "handoff_success_rate": (
            total("handoff_success_count") / total("handoff_attempt_count")
            if total("handoff_attempt_count")
            else 0.0
        ),
        "handoff_failure_rate": (
            total("handoff_failure_count") / total("handoff_attempt_count")
            if total("handoff_attempt_count")
            else 0.0
        ),
        "probation_failure_rate": avg("probation_failure_rate"),
        "llm_participation_rate": avg("llm_participation_rate"),
        "rollback_rate": (sum(1 for row in rows if _truthy(row.get("has_rollback"))) / n) if n else 0.0,
        "total_rollback_count": total("rollback_count"),
        "total_early_stop_trigger_count": total("early_stop_trigger_count"),
        "avg_pdi_decision_count": avg("pdi_decision_count"),
        "avg_no_valid_pdi_window_count": avg("no_valid_pdi_window_count"),
        "total_no_valid_pdi_window_count": total("no_valid_pdi_window_count"),
        "avg_pdi_window_count": avg("pdi_window_count"),
        "total_slm_scoring_count": total("slm_scoring_count"),
        "avg_slm_scoring_overhead": avg("slm_scoring_overhead"),
        "avg_self_reentry_attempt_count": avg("self_reentry_attempt_count"),
        "total_self_reentry_attempt_count": total("self_reentry_attempt_count"),
        "total_self_reentry_accept_count": total("self_reentry_accept_count"),
        "total_self_reentry_reject_count": total("self_reentry_reject_count"),
        "self_reentry_accept_rate": (
            total("self_reentry_accept_count") / total("self_reentry_attempt_count")
            if total("self_reentry_attempt_count")
            else 0.0
        ),
        "total_shadow_old_handoff_ready_count": total("shadow_old_handoff_ready_count"),
    }


def _write_summary(summary_path: Path, dataset: str, variant: str, rows: list[dict[str, Any]], dataset_wall_time: float) -> None:
    metrics = build_summary_metrics(dataset, variant, rows, dataset_wall_time)
    metrics.update(_extra_sarr_metrics(rows))
    metrics["method"] = variant
    write_summary_files(summary_path, rows, metrics)


def _validate_ownership_config(_cfg: SARRConfig) -> None:
    pass


def run_experiment(args: argparse.Namespace, cfg: SARRConfig) -> None:
    variant = args.variant or DEFAULT_VARIANT
    output_root = Path(args.output_root) if args.output_root else Path(cfg.output_dir)
    problems = load_eval_dataset(args.dataset, cfg, max_problems=args.max_problems)
    print(f"[sarr] loaded {len(problems)} problem(s) from {args.dataset}", flush=True)
    print(f"[sarr] variant={variant} output_root={output_root}", flush=True)
    print(
        f"[sarr] slm_backend={cfg.slm.backend} slm_device={cfg.slm.device} llm_backend={cfg.llm.backend} llm_endpoint={cfg.llm.api_base_url}",
        flush=True,
    )
    summary_path = _summary_path(output_root, args.dataset, variant)
    existing_summary_rows = load_summary_rows(summary_path) if args.resume else {}
    rows_by_problem_id: dict[str, dict[str, Any]] = {}
    skipped = 0

    if args.resume:
        for problem in problems:
            if _problem_complete(output_root, args.dataset, variant, problem.problem_id):
                row = existing_summary_rows.get(str(problem.problem_id))
                if row is not None:
                    rows_by_problem_id[str(problem.problem_id)] = row
                    skipped += 1

    def ordered_rows() -> list[dict[str, Any]]:
        return [
            rows_by_problem_id[str(problem.problem_id)]
            for problem in problems
            if str(problem.problem_id) in rows_by_problem_id
        ]

    _validate_ownership_config(cfg)
    print(
        "[sarr] controller=pdi_step_window "
        f"t_min={cfg.controller.t_min} "
        f"q_high={cfg.controller.q_high} "
        f"q_recover={cfg.controller.q_recover} "
        "handoff=repair_landing_index "
        f"final_answer_generator={cfg.generation.final_answer_generator}",
        flush=True,
    )
    from sarr_code.engines import build_llm, build_slm

    slm = build_slm(cfg.slm, cfg.runtime)
    llm = build_llm(cfg.llm, cfg.runtime)

    dataset_start = time.time()
    for problem in tqdm(problems, desc=f"{variant}:{args.dataset}"):
        problem_id = str(problem.problem_id)
        if args.resume and problem_id in rows_by_problem_id:
            continue

        problem_start = time.time()
        print(f"[sarr] running problem_id={problem_id}", flush=True)
        result, step_rows, controller_rows, transition_rows = run_sarr_code(
            problem_id=problem_id,
            problem_text=problem.problem_text,
            slm=slm,
            llm=llm,
            cfg=cfg,
        )
        if problem.gold_answer is not None:
            predicted = result.answer if args.dataset in MATH_DATASETS else result.state.assistant_prefix_text
            result.correct = benchmark_eval_match(predicted, problem.gold_answer, args.dataset)
        problem_wall_time = time.time() - problem_start
        _write_problem_outputs(
            output_root,
            args.dataset,
            variant,
            problem,
            result,
            cfg,
            step_rows=step_rows,
            controller_rows=controller_rows,
            transition_rows=transition_rows,
            problem_wall_time=problem_wall_time,
        )
        if cfg.runtime.reset_prefix_cache_after_problem:
            slm.clear_runtime_cache()
            llm.clear_runtime_cache()
        row = result_summary(result)
        metric_fields = _problem_sarr_metrics(result, step_rows, controller_rows)
        row.update(
            {
                "dataset": args.dataset,
                "variant": variant,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                "gold_answer": problem.gold_answer,
                "problem_wall_time": problem_wall_time,
                **metric_fields,
            }
        )
        rows_by_problem_id[problem_id] = row
        _write_summary(summary_path, args.dataset, variant, ordered_rows(), time.time() - dataset_start)

    _write_summary(summary_path, args.dataset, variant, ordered_rows(), time.time() - dataset_start)
    if args.resume and skipped:
        print(f"Skipped {skipped} completed problem(s).")
    print(f"Wrote {summary_path}")
    print(f"Wrote {_metrics_path(output_root, args.dataset, variant)}")


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def main() -> None:
    raise_csv_field_limit()
    parser = argparse.ArgumentParser(description="Run SARR-CoDE with SLM-led ownership control.")
    parser.add_argument("--config", required=True, help="Path to SARRConfig JSON.")
    parser.add_argument("--mode", default="run", choices=["run"], help="Compatibility flag; only run mode is supported.")
    parser.add_argument("--dataset", default="aime25", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = SARRConfig.from_json(args.config)
    run_experiment(args, cfg)


if __name__ == "__main__":
    main()
