#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_ROOT = Path(__file__).resolve().parent
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from specreason_core import SpecReasonHyperparams, SpecReasonRouter, extract_answer_text, parse_endpoints, route_stats
from baselines.model_pairs import DEFAULT_MODEL_PAIRS, apply_model_pair
from sarr_code import SARRConfig
from sarr_code.eval import benchmark_eval_match, build_summary_metrics, load_eval_dataset, write_summary_files
from sarr_code.eval.prompts import build_problem_text
from sarr_code.safety import extract_answer_from_final_step
from sarr_code.trace import write_json


SUPPORTED_DATASETS = ("math500", "aime24", "aime25", "gpqa", "gpqa_diamond")
DEFAULT_CONFIG: dict[str, Any] = {
    "repeat_num": 1,
    "score_threshold": 7.0,
    "score_method": "greedy",
    "token_budget": 16384,
    "output_dir": "specreason_results",
    "model_pair": "qwen3_1p7b_qwen3_32b",
    "model_pairs": DEFAULT_MODEL_PAIRS,
    "base_model_key": "base",
    "small_model_key": "small",
    "first_n_steps_base_model": 0,
    "prompt_style": "upstream",
    "step_max_tokens": 512,
    "stop_token": "\n\n",
    "generation_temperature": 0.6,
    "generation_top_p": 0.95,
    "score_temperature": 0.0,
    "score_max_tokens": 1,
    "score_top_logprobs": 10,
    "endpoints": {
        "base": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["base"],
        "small": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["small"],
    }
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return deepcopy(DEFAULT_CONFIG)
    source = Path(path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"SpecReason config must be a JSON object: {source}")
    return deep_update(DEFAULT_CONFIG, data)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, value in {
        "repeat_num": args.repeat_num,
        "score_threshold": args.score_threshold,
        "score_method": args.score_method,
        "token_budget": args.token_budget,
        "output_dir": args.output_root,
        "model_pair": args.model_pair,
        "base_model_key": args.base_model_key,
        "small_model_key": args.small_model_key,
        "prompt_style": args.prompt_style,
    }.items():
        if value is not None:
            config[key] = value
    apply_model_pair(
        config,
        config.get("model_pair"),
        small_key=str(config["small_model_key"]),
        base_key=str(config["base_model_key"]),
    )
    return config


def hyperparams_from_config(config: dict[str, Any]) -> SpecReasonHyperparams:
    return SpecReasonHyperparams(
        score_threshold=float(config["score_threshold"]),
        score_method=str(config["score_method"]),
        token_budget=int(config["token_budget"]),
        first_n_steps_base_model=int(config["first_n_steps_base_model"]),
        step_max_tokens=int(config["step_max_tokens"]),
        stop_token=str(config["stop_token"]),
        generation_temperature=float(config["generation_temperature"]),
        generation_top_p=float(config["generation_top_p"]),
        score_temperature=float(config["score_temperature"]),
        score_max_tokens=int(config["score_max_tokens"]),
        score_top_logprobs=int(config["score_top_logprobs"]),
    )


def output_root_from_config(config: dict[str, Any]) -> Path:
    output_root = Path(str(config["output_dir"]))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root


def variant_from_config(config: dict[str, Any], variant: str | None) -> str:
    if variant:
        return variant
    threshold = str(config["score_threshold"]).replace(".", "p")
    return f"specreason_{config['model_pair']}_{config['score_method']}_{threshold}"


def summary_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary.csv"


def metrics_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary_metrics.json"


def problem_root(output_root: Path, dataset: str, variant: str, problem_id: Any) -> Path:
    return output_root / dataset / variant / str(problem_id)


def run_id(problem_id: Any, repeat_id: int) -> str:
    return f"{problem_id}:{repeat_id}"


def problem_complete(output_root: Path, dataset: str, variant: str, problem_id: Any, repeat_id: int) -> bool:
    root = problem_root(output_root, dataset, variant, problem_id)
    return (root / f"{repeat_id}.metadata.json").exists() and (root / f"{repeat_id}.problem.json").exists()


def load_existing_summary_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            str(row.get("run_id") or run_id(row.get("problem_id"), int(row.get("repeat_id") or 0))): dict(row)
            for row in csv.DictReader(f)
        }


def ordered_rows(problems, rows_by_run_id: dict[str, dict[str, Any]], repeat_num: int) -> list[dict[str, Any]]:
    rows = []
    for problem in problems:
        for repeat_id in range(repeat_num):
            key = run_id(problem.problem_id, repeat_id)
            if key in rows_by_run_id:
                rows.append(rows_by_run_id[key])
    return rows


def choice_lines(raw: dict[str, Any]) -> str:
    choices = []
    for label in ["A", "B", "C", "D"]:
        value = raw.get(label) or raw.get(f"choice_{label.lower()}") or raw.get(f"Choice {label}")
        if value:
            choices.append((label, value))
    if not choices and raw.get("Correct Answer"):
        values = [
            raw.get("Correct Answer"),
            raw.get("Incorrect Answer 1"),
            raw.get("Incorrect Answer 2"),
            raw.get("Incorrect Answer 3"),
        ]
        choices = [(label, value) for label, value in zip(["A", "B", "C", "D"], values) if value]
    return "\n".join(f"({label}) {value}" for label, value in choices)


def build_upstream_prompt(raw: dict[str, Any], dataset: str) -> str:
    if dataset in {"math500", "aime24", "aime25"}:
        body = raw.get("problem") or raw.get("question") or raw.get("prompt")
        return (
            "Solve the following math problem efficiently and clearly. Please reason step by step,\n"
            "separate logical reasoning steps with two newline characters (\\n\\n), and put your final answer within \\boxed{{}}.\n"
            f"Problem: {body}"
        )
    if dataset in {"gpqa", "gpqa_diamond"}:
        body = raw.get("problem") or raw.get("Question") or raw.get("question")
        choices = choice_lines(raw)
        return (
            "What is the correct answer to the following problem? Please reason step by step.\n"
            "Separate logical reasoning steps with two newline characters (\\n\\n).\n"
            "Put the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).\n\n"
            "**Example output:** \\boxed{{A}}\n\n"
            f"Problem: {body}.\n"
            "Choices:\n"
            f"{choices}"
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_prompt(problem, dataset: str, prompt_style: str) -> str:
    if prompt_style == "project":
        return build_problem_text(problem.raw, dataset)
    if prompt_style == "upstream":
        return build_upstream_prompt(problem.raw, dataset)
    raise ValueError("prompt_style must be 'upstream' or 'project'.")


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return (sum(values) / len(values)) if values else None


def summary_row(
    *,
    dataset: str,
    variant: str,
    problem,
    repeat_id: int,
    metadata_list: list[dict[str, Any]],
    answer_text: str | None,
    correct: bool | None,
    config: dict[str, Any],
    wall_time: float,
    error: str | None = None,
) -> dict[str, Any]:
    stats = route_stats(metadata_list) if metadata_list else {}
    answer = extract_answer_from_final_step(answer_text) if answer_text else None
    row = {
        "run_id": run_id(problem.problem_id, repeat_id),
        "dataset": dataset,
        "variant": variant,
        "problem_id": problem.problem_id,
        "repeat_id": repeat_id,
        "question_id": problem.question_id,
        "gold_answer": problem.gold_answer,
        "answer": answer,
        "answer_text": answer_text,
        "correct": correct,
        "method": "specreason",
        "score_method": config["score_method"],
        "score_threshold": config["score_threshold"],
        "token_budget": config["token_budget"],
        "model_pair": config["model_pair"],
        "base_model_key": config["base_model_key"],
        "small_model_key": config["small_model_key"],
        "base_model": config["endpoints"][config["base_model_key"]]["model"],
        "small_model": config["endpoints"][config["small_model_key"]]["model"],
        "prompt_style": config["prompt_style"],
        "total_wall_time": wall_time,
        "problem_wall_time": wall_time,
        "error": error,
        **stats,
    }
    row.update(
        {
            "slm_decode_tokens": stats.get("small_decode_tokens", 0),
            "slm_prefill_tokens": stats.get("small_prefill_tokens", 0),
            "llm_decode_tokens": stats.get("base_decode_tokens", 0) + stats.get("score_decode_tokens", 0),
            "llm_prefill_tokens": stats.get("base_prefill_tokens", 0) + stats.get("score_prefill_tokens", 0),
            "slm_generate_calls": stats.get("small_accept_count", 0) + stats.get("base_fallback_count", 0),
            "llm_generate_calls": stats.get("base_fallback_count", 0) + stats.get("score_call_count", 0),
            "llm_full_calls": stats.get("base_fallback_count", 0) + stats.get("score_call_count", 0),
            "llm_scoring_calls": stats.get("score_call_count", 0),
        }
    )
    return row


def write_problem_outputs(
    *,
    output_root: Path,
    dataset: str,
    variant: str,
    problem,
    repeat_id: int,
    metadata_list: list[dict[str, Any]],
    row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    root = problem_root(output_root, dataset, variant, problem.problem_id)
    write_json(root / f"{repeat_id}.metadata.json", metadata_list)
    write_json(
        root / f"{repeat_id}.problem.json",
        {
            "problem_id": problem.problem_id,
            "repeat_id": repeat_id,
            "question_id": problem.question_id,
            "dataset": dataset,
            "variant": variant,
            "raw": problem.raw,
            "gold_answer": problem.gold_answer,
            "summary": row,
            "config": config,
        },
    )


def write_predictions(output_root: Path, dataset: str, variant: str, rows: list[dict[str, Any]], repeat_num: int) -> None:
    root = output_root / dataset / variant
    for repeat_id in range(repeat_num):
        predictions = [
            {
                "id": row["problem_id"],
                "repeat_id": repeat_id,
                "question_id": row.get("question_id"),
                "answer": row.get("answer_text"),
            }
            for row in rows
            if int(row.get("repeat_id") or 0) == repeat_id
        ]
        write_json(root / f"result_{repeat_id + 1}.json", predictions)


def write_summary(
    *,
    output_root: Path,
    dataset: str,
    variant: str,
    rows: list[dict[str, Any]],
    dataset_wall_time: float,
    config: dict[str, Any],
) -> None:
    metrics = build_summary_metrics(dataset, variant, rows, dataset_wall_time)
    metrics.update(
        {
            "method": "specreason",
            "score_method": config["score_method"],
            "score_threshold": config["score_threshold"],
            "token_budget": config["token_budget"],
            "model_pair": config["model_pair"],
            "base_model": config["endpoints"][config["base_model_key"]]["model"],
            "small_model": config["endpoints"][config["small_model_key"]]["model"],
            "prompt_style": config["prompt_style"],
            "repeat_num": config["repeat_num"],
            "num_failed": sum(1 for row in rows if row.get("error")),
            "avg_step_count": average(rows, "step_count"),
            "avg_score_call_count": average(rows, "score_call_count"),
            "avg_small_accept_count": average(rows, "small_accept_count"),
            "avg_base_fallback_count": average(rows, "base_fallback_count"),
            "avg_score": average(rows, "avg_score"),
            "total_small_decode_tokens": sum(int(row.get("small_decode_tokens") or 0) for row in rows),
            "total_base_decode_tokens": sum(int(row.get("base_decode_tokens") or 0) for row in rows),
            "total_score_decode_tokens": sum(int(row.get("score_decode_tokens") or 0) for row in rows),
        }
    )
    write_summary_files(summary_path(output_root, dataset, variant), rows, metrics)
    write_predictions(output_root, dataset, variant, rows, int(config["repeat_num"]))


def run_experiment(args: argparse.Namespace) -> None:
    config = apply_cli_overrides(load_config(args.specreason_config), args)
    sarr_cfg = SARRConfig.from_json(args.sarr_config)
    problems = load_eval_dataset(args.dataset, sarr_cfg, max_problems=args.max_problems)
    output_root = output_root_from_config(config)
    variant = variant_from_config(config, args.variant)
    repeat_num = int(config["repeat_num"])
    router = SpecReasonRouter(
        endpoints=parse_endpoints(config["endpoints"]),
        base_model_key=str(config["base_model_key"]),
        small_model_key=str(config["small_model_key"]),
        hyperparams=hyperparams_from_config(config),
    )

    print(f"[specreason] loaded {len(problems)} problem(s) from {args.dataset}", flush=True)
    print(f"[specreason] variant={variant} output_root={output_root}", flush=True)
    print(
        "[specreason] "
        f"base={config['base_model_key']}:{config['endpoints'][config['base_model_key']]['model']} "
        f"small={config['small_model_key']}:{config['endpoints'][config['small_model_key']]['model']} "
        f"score_threshold={config['score_threshold']} score_method={config['score_method']}",
        flush=True,
    )

    existing = load_existing_summary_rows(summary_path(output_root, args.dataset, variant)) if args.resume else {}
    rows_by_run_id: dict[str, dict[str, Any]] = {}
    skipped = 0
    if args.resume:
        for problem in problems:
            for repeat_id in range(repeat_num):
                key = run_id(problem.problem_id, repeat_id)
                if problem_complete(output_root, args.dataset, variant, problem.problem_id, repeat_id) and key in existing:
                    rows_by_run_id[key] = existing[key]
                    skipped += 1

    dataset_start = time.time()
    for problem in tqdm(problems, desc=f"{variant}:{args.dataset}"):
        prompt = build_prompt(problem, args.dataset, str(config["prompt_style"]))
        for repeat_id in range(repeat_num):
            key = run_id(problem.problem_id, repeat_id)
            if args.resume and key in rows_by_run_id:
                continue
            print(f"[specreason] running problem_id={problem.problem_id} repeat_id={repeat_id}", flush=True)
            wall_start = time.time()
            metadata_list: list[dict[str, Any]] = []
            error = None
            try:
                metadata_list = router.run(
                    problem_prompt=prompt,
                    dataset_name=args.dataset,
                    problem_id=problem.problem_id,
                    repeat_id=repeat_id,
                )
                answer_text = extract_answer_text(metadata_list)
                correct = (
                    benchmark_eval_match(answer_text, problem.gold_answer, args.dataset)
                    if problem.gold_answer is not None
                    else None
                )
            except Exception as exc:
                answer_text = None
                correct = None
                error = str(exc)
            wall_time = time.time() - wall_start
            row = summary_row(
                dataset=args.dataset,
                variant=variant,
                problem=problem,
                repeat_id=repeat_id,
                metadata_list=metadata_list,
                answer_text=answer_text,
                correct=correct,
                config=config,
                wall_time=wall_time,
                error=error,
            )
            rows_by_run_id[key] = row
            write_problem_outputs(
                output_root=output_root,
                dataset=args.dataset,
                variant=variant,
                problem=problem,
                repeat_id=repeat_id,
                metadata_list=metadata_list,
                row=row,
                config=config,
            )
            write_summary(
                output_root=output_root,
                dataset=args.dataset,
                variant=variant,
                rows=ordered_rows(problems, rows_by_run_id, repeat_num),
                dataset_wall_time=time.time() - dataset_start,
                config=config,
            )

    rows = ordered_rows(problems, rows_by_run_id, repeat_num)
    write_summary(
        output_root=output_root,
        dataset=args.dataset,
        variant=variant,
        rows=rows,
        dataset_wall_time=time.time() - dataset_start,
        config=config,
    )
    if skipped:
        print(f"Skipped {skipped} completed run(s).")
    print(f"Wrote {summary_path(output_root, args.dataset, variant)}")
    print(f"Wrote {metrics_path(output_root, args.dataset, variant)}")


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
    parser = argparse.ArgumentParser(description="Run the SpecReason comparison baseline.")
    parser.add_argument("--sarr-config", default="configs/sarr_code_aggressive.json", help="Project config used for dataset paths.")
    parser.add_argument("--specreason-config", default=str(BASELINE_ROOT / "config.example.json"), help="SpecReason endpoint and hyperparameter config.")
    parser.add_argument("--dataset", default="aime24", choices=SUPPORTED_DATASETS)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeat-num", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--score-method", choices=["greedy", "average"], default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--model-pair", default=None, help="One of the four Qwen3/DeepSeek model-pair ids.")
    parser.add_argument("--base-model-key", default=None)
    parser.add_argument("--small-model-key", default=None)
    parser.add_argument("--prompt-style", choices=["upstream", "project"], default=None)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
