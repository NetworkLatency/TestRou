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
from sarr_code.engines import build_llm, build_slm


MATH_DATASETS = {"math500", "aime24", "aime25"}
DEFAULT_VARIANT = "sarr_code_v2_raw_hcs_confirmed_rollback"


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
        root / f"{stem}.rollback_events.jsonl",
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
    rollback_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    problem_wall_time: float,
) -> None:
    root = _problem_root(output_root, dataset, variant, problem.problem_id)
    stem = str(problem.problem_id)
    summary = result_summary(result)
    metric_fields = _problem_sarr_metrics(result, step_rows, rollback_rows)
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
        root / f"{stem}.rollback_events.jsonl",
        [
            {"problem_id": problem.problem_id, "question_id": problem.question_id, **row}
            for row in rollback_rows
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


def _problem_sarr_metrics(result, step_rows: list[dict[str, Any]], rollback_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rollback_count = len(rollback_rows)
    startup_rollback_count = sum(1 for row in rollback_rows if row.get("type") == "STARTUP_ROLLBACK")
    post_stable_rollback_count = sum(1 for row in rollback_rows if row.get("type") == "POST_STABLE_ROLLBACK")
    hcs_rollback_count = sum(1 for row in rollback_rows if row.get("type") == "HCS_ROLLBACK")
    rollback_span_total = sum(int(row.get("rollback_span") or 0) for row in rollback_rows)
    recovery_steps_total = sum(int(row.get("recovery_actual_steps") or 0) for row in rollback_rows)
    recovery_ready_count = sum(1 for row in rollback_rows if row.get("stop_reason") == "SLM_READY")
    recovery_exhausted_count = sum(1 for row in rollback_rows if row.get("stop_reason") == "EXHAUSTED_FORCE_SLM")
    force_slm_after_recovery_count = sum(1 for row in rollback_rows if _truthy(row.get("force_next_step_slm")))
    force_slm_after_recovery_fail_count = sum(
        1 for row in rollback_rows if row.get("force_slm_after_recovery_failed") is True
    )
    active_thinking_step_count = sum(1 for row in step_rows if row.get("active") is True or str(row.get("active")).lower() == "true")
    generated_thinking_attempt_count = len(step_rows)
    forced_close_think = str(result.state.stop_reason or "").endswith("_forced_close_think")
    summary = result_summary(result)
    llm_token_ratio = summary["llm_token_share"]
    return {
        "active_thinking_step_count": active_thinking_step_count,
        "generated_thinking_attempt_count": generated_thinking_attempt_count,
        "rollback_count": rollback_count,
        "has_rollback": rollback_count > 0,
        "startup_rollback_count": startup_rollback_count,
        "has_startup_rollback": startup_rollback_count > 0,
        "post_stable_rollback_count": post_stable_rollback_count,
        "has_post_stable_rollback": post_stable_rollback_count > 0,
        "hcs_rollback_count": hcs_rollback_count,
        "has_hcs_rollback": hcs_rollback_count > 0,
        "hcs_suspect_count": sum(1 for row in step_rows if _truthy(row.get("hcs_suspect"))),
        "hcs_confirmed_count": sum(1 for row in step_rows if _truthy(row.get("hcs_confirmed"))),
        "anchor_zero_count": sum(1 for row in rollback_rows if row.get("anchor_step") == 0),
        "rollback_span_total": rollback_span_total,
        "avg_rollback_span": (rollback_span_total / rollback_count) if rollback_count else 0.0,
        "recovery_steps_total": recovery_steps_total,
        "avg_recovery_steps": (recovery_steps_total / rollback_count) if rollback_count else 0.0,
        "recovery_ready_count": recovery_ready_count,
        "recovery_ready_rate": (recovery_ready_count / rollback_count) if rollback_count else 0.0,
        "recovery_exhausted_count": recovery_exhausted_count,
        "recovery_exhausted_rate": (recovery_exhausted_count / rollback_count) if rollback_count else 0.0,
        "forced_close_think": forced_close_think,
        "force_slm_after_recovery_count": force_slm_after_recovery_count,
        "force_slm_after_recovery_fail_count": force_slm_after_recovery_fail_count,
        "force_slm_after_recovery_fail_rate": (
            force_slm_after_recovery_fail_count / force_slm_after_recovery_count
        )
        if force_slm_after_recovery_count
        else 0.0,
        "llm_token_ratio": llm_token_ratio,
    }


def _extra_sarr_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        return sum(vals) / len(vals) if vals else None

    n = len(rows)
    total_rollbacks = sum(int(_num(row.get("rollback_count"))) for row in rows)
    total_startup_rollbacks = sum(int(_num(row.get("startup_rollback_count"))) for row in rows)
    total_post_stable_rollbacks = sum(int(_num(row.get("post_stable_rollback_count"))) for row in rows)
    total_hcs_rollbacks = sum(int(_num(row.get("hcs_rollback_count"))) for row in rows)
    total_anchor_zero = sum(int(_num(row.get("anchor_zero_count"))) for row in rows)
    total_span = sum(_num(row.get("rollback_span_total")) for row in rows)
    total_recovery_steps = sum(_num(row.get("recovery_steps_total")) for row in rows)
    total_ready = sum(int(_num(row.get("recovery_ready_count"))) for row in rows)
    total_exhausted = sum(int(_num(row.get("recovery_exhausted_count"))) for row in rows)
    total_force_handoffs = sum(int(_num(row.get("force_slm_after_recovery_count"))) for row in rows)
    total_force_failures = sum(int(_num(row.get("force_slm_after_recovery_fail_count"))) for row in rows)
    total_llm_tokens = sum(_num(row.get("llm_total_tokens")) for row in rows)
    total_model_tokens = sum(_num(row.get("total_model_tokens")) for row in rows)
    return {
        "rollback_rate": (sum(1 for row in rows if _truthy(row.get("has_rollback"))) / n) if n else 0.0,
        "startup_rollback_rate": (sum(1 for row in rows if _truthy(row.get("has_startup_rollback"))) / n) if n else 0.0,
        "post_stable_rollback_rate": (sum(1 for row in rows if _truthy(row.get("has_post_stable_rollback"))) / n) if n else 0.0,
        "hcs_rollback_rate": (sum(1 for row in rows if _truthy(row.get("has_hcs_rollback"))) / n) if n else 0.0,
        "avg_rollback_count": avg("rollback_count"),
        "avg_startup_rollback_count": avg("startup_rollback_count"),
        "avg_post_stable_rollback_count": avg("post_stable_rollback_count"),
        "avg_hcs_rollback_count": avg("hcs_rollback_count"),
        "total_rollback_count": total_rollbacks,
        "total_startup_rollback_count": total_startup_rollbacks,
        "total_post_stable_rollback_count": total_post_stable_rollbacks,
        "total_hcs_rollback_count": total_hcs_rollbacks,
        "anchor_zero_rate_per_rollback": (total_anchor_zero / total_rollbacks) if total_rollbacks else 0.0,
        "avg_rollback_span": (total_span / total_rollbacks) if total_rollbacks else 0.0,
        "avg_recovery_steps": (total_recovery_steps / total_rollbacks) if total_rollbacks else 0.0,
        "recovery_ready_rate": (total_ready / total_rollbacks) if total_rollbacks else 0.0,
        "recovery_exhausted_rate_per_rollback": (total_exhausted / total_rollbacks) if total_rollbacks else 0.0,
        "recovery_exhausted_rate": (total_exhausted / total_rollbacks) if total_rollbacks else 0.0,
        "forced_close_think_rate": (sum(1 for row in rows if _truthy(row.get("forced_close_think"))) / n) if n else 0.0,
        "force_slm_after_recovery_fail_rate": (
            total_force_failures / total_force_handoffs
        )
        if total_force_handoffs
        else 0.0,
        "llm_token_ratio": (total_llm_tokens / total_model_tokens) if total_model_tokens else 0.0,
    }


def _write_summary(summary_path: Path, dataset: str, variant: str, rows: list[dict[str, Any]], dataset_wall_time: float) -> None:
    metrics = build_summary_metrics(dataset, variant, rows, dataset_wall_time)
    metrics.update(_extra_sarr_metrics(rows))
    metrics["method"] = variant
    write_summary_files(summary_path, rows, metrics)


def _load_normalizer(cfg: SARRConfig):
    if cfg.calibration.enabled or cfg.calibration.load_cdf or cfg.calibration.use_percentile:
        raise RuntimeError("This experiment disables calibration; run with calibration.enabled=false.")
    if cfg.confidence.calibration_path:
        raise RuntimeError("This experiment disables calibration; remove confidence.calibration_path.")
    return None


def run_calibration(args: argparse.Namespace, cfg: SARRConfig) -> None:
    raise RuntimeError("Calibration CDF construction is disabled for raw-readiness SARR-CoDE.")


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

    normalizer = _load_normalizer(cfg)
    print("[sarr] readiness_source=raw calibration_enabled=false", flush=True)
    slm = build_slm(cfg.slm, cfg.runtime)
    llm = build_llm(cfg.llm, cfg.runtime)

    dataset_start = time.time()
    for problem in tqdm(problems, desc=f"{variant}:{args.dataset}"):
        problem_id = str(problem.problem_id)
        if args.resume and problem_id in rows_by_problem_id:
            continue

        problem_start = time.time()
        print(f"[sarr] running problem_id={problem_id}", flush=True)
        result, step_rows, rollback_rows, transition_rows = run_sarr_code(
            problem_id=problem_id,
            problem_text=problem.problem_text,
            slm=slm,
            llm=llm,
            normalizer=normalizer,
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
            rollback_rows=rollback_rows,
            transition_rows=transition_rows,
            problem_wall_time=problem_wall_time,
        )
        if cfg.runtime.reset_prefix_cache_after_problem:
            slm.clear_runtime_cache()
            llm.clear_runtime_cache()
        row = result_summary(result)
        metric_fields = _problem_sarr_metrics(result, step_rows, rollback_rows)
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
    parser = argparse.ArgumentParser(description="Run SARR-CoDE raw-readiness HCS confirmed rollback collaboration.")
    parser.add_argument("--config", required=True, help="Path to SARRConfig JSON.")
    parser.add_argument("--mode", choices=["run", "calibrate"], default="run")
    parser.add_argument("--dataset", default="aime25", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibration-output", default=None)
    args = parser.parse_args()

    cfg = SARRConfig.from_json(args.config)
    if args.mode == "calibrate":
        run_calibration(args, cfg)
    else:
        run_experiment(args, cfg)


if __name__ == "__main__":
    main()
