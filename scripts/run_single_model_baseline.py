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

from sarr_code import GenerationState, Phase, SARRConfig, SARRResult, TraceEvent
from sarr_code.context_budget import ContextBudgetExceeded
from sarr_code.engines import build_llm, build_slm
from sarr_code.eval import benchmark_eval_match, build_summary_metrics, load_eval_dataset, load_summary_rows, write_summary_files
from sarr_code.safety import extract_answer_from_final_step
from sarr_code.trace import result_summary, write_json


SUPPORTED_DATASETS = ("math500", "aime24", "aime25", "gpqa", "gpqa_diamond", "humaneval")
MATH_DATASETS = {"math500", "aime24", "aime25"}
GPQA_DATASETS = {"gpqa", "gpqa_diamond"}
CODE_DATASETS = {"humaneval"}
DEFAULT_METHOD = "single_model_baseline_v0"


def _summary_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary.csv"


def _metrics_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary_metrics.json"


def _problem_root(output_root: Path, dataset: str, variant: str, problem_id: Any) -> Path:
    return output_root / dataset / variant / str(problem_id)


def _problem_complete(output_root: Path, dataset: str, variant: str, problem_id: Any) -> bool:
    root = _problem_root(output_root, dataset, variant, problem_id)
    stem = str(problem_id)
    return (root / f"{stem}.problem.json").exists() and (root / f"{stem}.generation.json").exists()


def _actual_token_count(output) -> int:
    return int(output.extra.get("actual_token_count") or output.token_count)


def _default_max_new_tokens(cfg: SARRConfig) -> int:
    return int(cfg.generation.think_token_budget) + int(cfg.generation.answer_token_budget)


def _normalize_output_root(path: str | None, cfg: SARRConfig) -> Path:
    output_root = Path(path or cfg.output_dir)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root


def _stop_delimiters(args: argparse.Namespace) -> list[str] | None:
    stops = [stop for stop in args.stop if stop]
    return stops or None


def _build_engine(args: argparse.Namespace, cfg: SARRConfig):
    if args.model_role == "slm":
        return build_slm(cfg.slm, cfg.runtime), cfg.slm
    return build_llm(cfg.llm, cfg.runtime), cfg.llm


def _account_generation_cost(state: GenerationState, model_role: str, output) -> None:
    token_count = _actual_token_count(output)
    if model_role == "slm":
        state.slm_generate_calls = 1
        state.slm_decode_tokens = token_count
        state.slm_prefill_tokens = int(output.prompt_tokens)
        state.slm_wall_time = float(output.wall_time)
        return
    state.llm_full_calls = 1
    state.llm_decode_tokens = token_count
    state.llm_prefill_tokens = int(output.prompt_tokens)
    state.llm_generation_wall_time = float(output.wall_time)


def _predicted_for_eval(dataset: str, output_text: str, extracted_answer: str | None) -> str | None:
    if dataset in MATH_DATASETS:
        return extracted_answer or output_text
    if dataset in GPQA_DATASETS:
        return output_text
    if dataset in CODE_DATASETS:
        return output_text
    return extracted_answer or output_text


def _run_problem(
    *,
    problem_id: str,
    problem_text: str,
    dataset: str,
    gold_answer: str | None,
    engine,
    model_role: str,
    max_new_tokens: int,
    stop_delimiters: list[str] | None,
) -> SARRResult:
    state = GenerationState(problem_text=problem_text, generation_protocol=DEFAULT_METHOD)
    start_time = time.time()
    try:
        output = engine.generate_text(
            problem_text,
            "",
            max_new_tokens=max_new_tokens,
            stop_delimiters=stop_delimiters,
            include_stop_str_in_output=True,
        )
        _account_generation_cost(state, model_role, output)
        state.assistant_prefix_text = output.text
        state.step_count = 1
        state.stop_reason = output.finish_reason or "finished"
        answer = extract_answer_from_final_step(output.text)
        predicted = _predicted_for_eval(dataset, output.text, answer)
        correct = benchmark_eval_match(predicted, gold_answer, dataset) if gold_answer is not None else None
        state.trace.append(
            TraceEvent(
                1,
                "single_model_generation",
                {
                    "problem_id": problem_id,
                    "model_role": model_role,
                    "max_new_tokens": max_new_tokens,
                    "stop_delimiters": stop_delimiters or [],
                    "finish_reason": output.finish_reason,
                    "prompt_tokens": output.prompt_tokens,
                    "token_count": output.token_count,
                    "actual_token_count": _actual_token_count(output),
                    "wall_time": output.wall_time,
                    "extra": output.extra,
                },
            )
        )
        state.phase = Phase.DONE
        return SARRResult(answer=answer, state=state, total_wall_time=time.time() - start_time, correct=correct)
    except ContextBudgetExceeded as exc:
        state.phase = Phase.DONE
        state.stop_reason = "context_budget"
        state.trace.append(TraceEvent(0, "context_budget_exhausted", exc.to_trace_data()))
        return SARRResult(answer=None, state=state, total_wall_time=time.time() - start_time, correct=None)
    except Exception as exc:
        state.phase = Phase.DONE
        state.stop_reason = "error"
        state.trace.append(TraceEvent(0, "generation_error", {"error": str(exc)}))
        return SARRResult(answer=None, state=state, total_wall_time=time.time() - start_time, correct=None)


def _write_problem_outputs(
    output_root: Path,
    dataset: str,
    variant: str,
    problem,
    result: SARRResult,
    *,
    model_role: str,
    model_backend: str,
    model_path: str,
    max_new_tokens: int,
    problem_wall_time: float,
) -> None:
    root = _problem_root(output_root, dataset, variant, problem.problem_id)
    stem = str(problem.problem_id)
    row = result_summary(result)
    row.update(
        {
            "raw": problem.raw,
            "dataset": dataset,
            "variant": variant,
            "problem_id": problem.problem_id,
            "question_id": problem.question_id,
            "gold_answer": problem.gold_answer,
            "problem_wall_time": problem_wall_time,
            "method": DEFAULT_METHOD,
            "model_role": model_role,
            "model_backend": model_backend,
            "model_path": model_path,
            "max_new_tokens": max_new_tokens,
        }
    )
    write_json(root / f"{stem}.problem.json", row)
    write_json(
        root / f"{stem}.generation.json",
        {
            "problem_id": problem.problem_id,
            "question_id": problem.question_id,
            "dataset": dataset,
            "model_role": model_role,
            "output_text": result.state.assistant_prefix_text,
            "extracted_answer": result.answer,
            "correct": result.correct,
            "trace": result.state.trace,
        },
    )


def _summary_row(
    *,
    dataset: str,
    variant: str,
    problem,
    result: SARRResult,
    model_role: str,
    model_backend: str,
    model_path: str,
    max_new_tokens: int,
    problem_wall_time: float,
) -> dict[str, Any]:
    row = result_summary(result)
    row.update(
        {
            "dataset": dataset,
            "variant": variant,
            "problem_id": problem.problem_id,
            "question_id": problem.question_id,
            "gold_answer": problem.gold_answer,
            "problem_wall_time": problem_wall_time,
            "method": DEFAULT_METHOD,
            "model_role": model_role,
            "model_backend": model_backend,
            "model_path": model_path,
            "max_new_tokens": max_new_tokens,
            "error": _trace_error(result),
        }
    )
    return row


def _trace_error(result: SARRResult) -> str | None:
    for event in reversed(result.state.trace):
        if event.event == "generation_error":
            return str(event.data.get("error") or "")
    return None


def _extra_baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_failed": sum(1 for row in rows if row.get("error")),
        "num_context_budget": sum(1 for row in rows if row.get("stop_reason") == "context_budget"),
    }


def _write_summary(
    summary_path: Path,
    dataset: str,
    variant: str,
    rows: list[dict[str, Any]],
    dataset_wall_time: float,
    *,
    model_role: str,
    model_backend: str,
    model_path: str,
    max_new_tokens: int,
) -> None:
    metrics = build_summary_metrics(dataset, variant, rows, dataset_wall_time)
    metrics.update(
        {
            "method": DEFAULT_METHOD,
            "model_role": model_role,
            "model_backend": model_backend,
            "model_path": model_path,
            "max_new_tokens": max_new_tokens,
            **_extra_baseline_metrics(rows),
        }
    )
    write_summary_files(summary_path, rows, metrics)


def run_experiment(args: argparse.Namespace, cfg: SARRConfig) -> None:
    variant = args.variant or f"{args.model_role}_{DEFAULT_METHOD}"
    output_root = _normalize_output_root(args.output_root, cfg)
    max_new_tokens = args.max_new_tokens or _default_max_new_tokens(cfg)
    stop_delimiters = _stop_delimiters(args)
    problems = load_eval_dataset(args.dataset, cfg, max_problems=args.max_problems)
    engine, model_cfg = _build_engine(args, cfg)
    model_backend = model_cfg.backend
    model_path = model_cfg.api_model or model_cfg.model_path

    print(f"[baseline] loaded {len(problems)} problem(s) from {args.dataset}", flush=True)
    print(f"[baseline] variant={variant} output_root={output_root}", flush=True)
    print(
        f"[baseline] model_role={args.model_role} backend={model_backend} model={model_path} max_new_tokens={max_new_tokens}",
        flush=True,
    )

    summary_path = _summary_path(output_root, args.dataset, variant)
    existing_summary_rows = load_summary_rows(summary_path) if args.resume else {}
    rows_by_problem_id: dict[str, dict[str, Any]] = {}
    skipped = 0
    if args.resume:
        for problem in problems:
            problem_id = str(problem.problem_id)
            if _problem_complete(output_root, args.dataset, variant, problem.problem_id):
                row = existing_summary_rows.get(problem_id)
                if row is not None:
                    rows_by_problem_id[problem_id] = row
                    skipped += 1

    def ordered_rows() -> list[dict[str, Any]]:
        return [
            rows_by_problem_id[str(problem.problem_id)]
            for problem in problems
            if str(problem.problem_id) in rows_by_problem_id
        ]

    dataset_start = time.time()
    for problem in tqdm(problems, desc=f"{variant}:{args.dataset}"):
        problem_id = str(problem.problem_id)
        if args.resume and problem_id in rows_by_problem_id:
            continue

        problem_start = time.time()
        print(f"[baseline] running problem_id={problem_id}", flush=True)
        result = _run_problem(
            problem_id=problem_id,
            problem_text=problem.problem_text,
            dataset=args.dataset,
            gold_answer=problem.gold_answer,
            engine=engine,
            model_role=args.model_role,
            max_new_tokens=max_new_tokens,
            stop_delimiters=stop_delimiters,
        )
        problem_wall_time = time.time() - problem_start
        _write_problem_outputs(
            output_root,
            args.dataset,
            variant,
            problem,
            result,
            model_role=args.model_role,
            model_backend=model_backend,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            problem_wall_time=problem_wall_time,
        )
        row = _summary_row(
            dataset=args.dataset,
            variant=variant,
            problem=problem,
            result=result,
            model_role=args.model_role,
            model_backend=model_backend,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            problem_wall_time=problem_wall_time,
        )
        rows_by_problem_id[problem_id] = row
        _write_summary(
            summary_path,
            args.dataset,
            variant,
            ordered_rows(),
            time.time() - dataset_start,
            model_role=args.model_role,
            model_backend=model_backend,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
        )
        if cfg.runtime.reset_prefix_cache_after_problem:
            engine.clear_runtime_cache()

    _write_summary(
        summary_path,
        args.dataset,
        variant,
        ordered_rows(),
        time.time() - dataset_start,
        model_role=args.model_role,
        model_backend=model_backend,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
    )
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
    parser = argparse.ArgumentParser(description="Run a pure single-model baseline on a local eval dataset.")
    parser.add_argument("--config", required=True, help="Path to SARRConfig JSON.")
    parser.add_argument("--dataset", default="aime25", choices=SUPPORTED_DATASETS)
    parser.add_argument("--model-role", default="llm", choices=["slm", "llm"], help="Which model config to evaluate.")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop", action="append", default=[], help="Optional stop delimiter; can be passed multiple times.")
    args = parser.parse_args()

    cfg = SARRConfig.from_json(args.config)
    run_experiment(args, cfg)


if __name__ == "__main__":
    main()
