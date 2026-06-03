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

from rsd_core import RSDHyperparams, RSDRouter, extract_answer_text, parse_endpoints, route_stats
from baselines.model_pairs import DEFAULT_MODEL_PAIRS, apply_model_pair
from sarr_code import SARRConfig
from sarr_code.eval import benchmark_eval_match, build_summary_metrics, load_eval_dataset, write_summary_files
from sarr_code.eval.prompts import build_problem_text
from sarr_code.safety import extract_answer_from_final_step
from sarr_code.trace import write_json


SUPPORTED_DATASETS = ("math500", "aime24", "aime25", "gpqa", "gpqa_diamond")
DEFAULT_CONFIG: dict[str, Any] = {
    "repeat_num": 1,
    "output_dir": "rsd_results",
    "prompt_type": "reasoning-chat",
    "prompt_style": "upstream",
    "model_pair": "qwen3_1p7b_qwen3_32b",
    "model_pairs": DEFAULT_MODEL_PAIRS,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens_per_call": 16384,
    "step_max_tokens": 512,
    "step_word": "\n\n",
    "prm_threshold": 0.7,
    "max_steps": 100,
    "patience": 5,
    "score_temperature": 0.0,
    "score_max_tokens": 1,
    "score_top_logprobs": 10,
    "enable_thinking": True,
    "endpoints": {
        "draft": {
            "model": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["small"]["model"],
            "base_url": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["small"]["base_url"],
            "api_key": "EMPTY",
        },
        "target": {
            "model": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["base"]["model"],
            "base_url": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["base"]["base_url"],
            "api_key": "EMPTY",
        },
    },
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
        raise ValueError(f"RSD config must be a JSON object: {source}")
    return deep_update(DEFAULT_CONFIG, data)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, value in {
        "repeat_num": args.repeat_num,
        "output_dir": args.output_root,
        "model_pair": args.model_pair,
        "prompt_style": args.prompt_style,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens_per_call": args.max_tokens_per_call,
        "step_max_tokens": args.step_max_tokens,
        "prm_threshold": args.prm_threshold,
        "max_steps": args.max_steps,
        "patience": args.patience,
    }.items():
        if value is not None:
            config[key] = value
    if float(config["temperature"]) == 0.0:
        config["top_p"] = 1.0
    apply_model_pair(config, config.get("model_pair"), small_key="draft", base_key="target")
    return config


def hyperparams_from_config(config: dict[str, Any]) -> RSDHyperparams:
    return RSDHyperparams(
        prm_threshold=float(config["prm_threshold"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        max_tokens_per_call=int(config["max_tokens_per_call"]),
        step_max_tokens=int(config["step_max_tokens"]),
        step_word=str(config["step_word"]),
        max_steps=int(config["max_steps"]),
        patience=int(config["patience"]),
        score_temperature=float(config["score_temperature"]),
        score_max_tokens=int(config["score_max_tokens"]),
        score_top_logprobs=int(config["score_top_logprobs"]),
        enable_thinking=bool(config["enable_thinking"]),
    )


def output_root_from_config(config: dict[str, Any]) -> Path:
    output_root = Path(str(config["output_dir"]))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root


def _model_slug(config: dict[str, Any], key: str) -> str:
    raw = config["endpoints"][key].get("model")
    return str(raw).split("/")[-1].replace(".", "p")


def variant_from_config(config: dict[str, Any], variant: str | None) -> str:
    if variant:
        return variant
    threshold = str(config["prm_threshold"]).replace(".", "p")
    return f"rsd_{config['model_pair']}_judge{threshold}"


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
    if isinstance(raw.get("choices"), list):
        choices.extend((label, value) for label, value in zip(["A", "B", "C", "D"], raw["choices"]) if value)
    for label in ["A", "B", "C", "D"]:
        value = raw.get(label) or raw.get(f"choice_{label.lower()}") or raw.get(f"Choice {label}")
        if value and not any(existing_label == label for existing_label, _ in choices):
            choices.append((label, value))
    if not choices and raw.get("Correct Answer"):
        values = [
            raw.get("Correct Answer"),
            raw.get("Incorrect Answer 1"),
            raw.get("Incorrect Answer 2"),
            raw.get("Incorrect Answer 3"),
        ]
        choices = [(label, value) for label, value in zip(["A", "B", "C", "D"], values) if value]
    return " ".join(f"({label}) {value}" for label, value in choices)


def build_upstream_problem_input(raw: dict[str, Any], dataset: str) -> str:
    if dataset in {"math500", "aime24", "aime25"}:
        return str(raw.get("problem") or raw.get("question") or raw.get("prompt") or "").strip()
    if dataset in {"gpqa", "gpqa_diamond"}:
        body = str(raw.get("problem") or raw.get("Question") or raw.get("question") or "").strip()
        choices = choice_lines(raw)
        return f"{body}\nAnswer Choices: {choices}".strip()
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_reasoning_chat_prompt(problem_input: str, dataset: str) -> str:
    if dataset in {"gpqa", "gpqa_diamond"}:
        return (
            "Please reason step by step. Put the final answer strictly in the format "
            "\\boxed{X}, where X is a single letter (A, B, C, or D).\n\n"
            f"Problem: {problem_input}"
        )
    return (
        "Please reason step by step, separate logical reasoning steps with two newline characters, "
        "and put your final answer within \\boxed{}.\n\n"
        f"Problem: {problem_input}"
    )


def build_prompt(problem, dataset: str, prompt_style: str) -> tuple[str, str]:
    prm_problem = build_upstream_problem_input(problem.raw, dataset)
    if prompt_style == "upstream":
        return build_reasoning_chat_prompt(prm_problem, dataset), prm_problem
    if prompt_style == "project":
        return build_problem_text(problem.raw, dataset), prm_problem
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
    draft_decode = int(stats.get("draft_decode_tokens") or 0)
    target_decode = int(stats.get("target_decode_tokens") or 0)
    total_decode = draft_decode + target_decode
    draft_wall = float(stats.get("draft_wall_time") or 0.0)
    target_wall = float(stats.get("target_wall_time") or 0.0)
    prm_wall = float(stats.get("prm_wall_time") or 0.0)
    total_generation_wall = draft_wall + target_wall
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
        "method": "rsd",
        "prompt_type": config["prompt_type"],
        "prompt_style": config["prompt_style"],
        "model_pair": config["model_pair"],
        "prm_threshold": config["prm_threshold"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens_per_call": config["max_tokens_per_call"],
        "max_steps": config["max_steps"],
        "patience": config["patience"],
        "draft_model": config["endpoints"]["draft"]["model"],
        "target_model": config["endpoints"]["target"]["model"],
        "judge_model": config["endpoints"]["target"]["model"],
        "total_wall_time": wall_time,
        "problem_wall_time": wall_time,
        "error": error,
        **stats,
    }
    row.update(
        {
            "slm_decode_tokens": draft_decode,
            "slm_prefill_tokens": 0,
            "llm_decode_tokens": target_decode,
            "llm_prefill_tokens": 0,
            "slm_generate_calls": stats.get("step_count", 0),
            "llm_generate_calls": stats.get("target_fallback_count", 0),
            "llm_full_calls": stats.get("target_fallback_count", 0),
            "llm_scoring_calls": 0,
            "prm_scoring_calls": stats.get("prm_call_count", 0),
            "excluded_judge_scoring_calls": stats.get("prm_call_count", 0),
            "excluded_judge_prompt_tokens": stats.get("excluded_judge_prompt_tokens", 0),
            "excluded_judge_decode_tokens": stats.get("excluded_judge_decode_tokens", 0),
            "excluded_judge_total_tokens": stats.get("excluded_judge_total_tokens", 0),
            "llm_token_share": (target_decode / total_decode) if total_decode else None,
            "llm_decode_share": (target_decode / total_decode) if total_decode else None,
            "llm_wall_time_share": (target_wall / total_generation_wall) if total_generation_wall else None,
            "slm_wall_time": draft_wall,
            "llm_generation_wall_time": target_wall,
            "llm_scoring_wall_time": 0.0,
            "excluded_judge_wall_time": prm_wall,
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
            "method": "rsd",
            "prompt_type": config["prompt_type"],
            "prompt_style": config["prompt_style"],
            "model_pair": config["model_pair"],
            "prm_threshold": config["prm_threshold"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_tokens_per_call": config["max_tokens_per_call"],
            "max_steps": config["max_steps"],
            "patience": config["patience"],
            "draft_model": config["endpoints"]["draft"]["model"],
            "target_model": config["endpoints"]["target"]["model"],
            "judge_model": config["endpoints"]["target"]["model"],
            "repeat_num": config["repeat_num"],
            "num_failed": sum(1 for row in rows if row.get("error")),
            "avg_draft_accept_count": average(rows, "draft_accept_count"),
            "avg_target_fallback_count": average(rows, "target_fallback_count"),
            "avg_prm_call_count": average(rows, "prm_call_count"),
            "avg_prm_reward": average(rows, "avg_prm_reward"),
            "total_draft_decode_tokens": sum(int(row.get("draft_decode_tokens") or 0) for row in rows),
            "total_target_decode_tokens": sum(int(row.get("target_decode_tokens") or 0) for row in rows),
            "total_draft_discarded_decode_tokens": sum(int(row.get("draft_discarded_decode_tokens") or 0) for row in rows),
            "total_prm_input_ids": sum(int(row.get("prm_input_ids") or 0) for row in rows),
            "excluded_judge_scoring_calls": sum(int(row.get("excluded_judge_scoring_calls") or 0) for row in rows),
            "excluded_judge_prompt_tokens": sum(int(row.get("excluded_judge_prompt_tokens") or 0) for row in rows),
            "excluded_judge_decode_tokens": sum(int(row.get("excluded_judge_decode_tokens") or 0) for row in rows),
            "excluded_judge_total_tokens": sum(int(row.get("excluded_judge_total_tokens") or 0) for row in rows),
            "excluded_judge_wall_time": sum(float(row.get("excluded_judge_wall_time") or 0.0) for row in rows),
        }
    )
    write_summary_files(summary_path(output_root, dataset, variant), rows, metrics)
    write_predictions(output_root, dataset, variant, rows, int(config["repeat_num"]))


def run_experiment(args: argparse.Namespace) -> None:
    config = apply_cli_overrides(load_config(args.rsd_config), args)
    sarr_cfg = SARRConfig.from_json(args.sarr_config)
    problems = load_eval_dataset(args.dataset, sarr_cfg, max_problems=args.max_problems)
    output_root = output_root_from_config(config)
    variant = variant_from_config(config, args.variant)
    repeat_num = int(config["repeat_num"])
    router = RSDRouter(
        endpoints=parse_endpoints(config["endpoints"]),
        hyperparams=hyperparams_from_config(config),
    )

    print(f"[rsd] loaded {len(problems)} problem(s) from {args.dataset}", flush=True)
    print(f"[rsd] variant={variant} output_root={output_root}", flush=True)
    print(
        "[rsd] "
        f"model_pair={config['model_pair']} "
        f"draft={config['endpoints']['draft']['model']} "
        f"target={config['endpoints']['target']['model']} "
        f"judge={config['endpoints']['target']['model']} "
        f"threshold={config['prm_threshold']}",
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
        prompt, prm_problem = build_prompt(problem, args.dataset, str(config["prompt_style"]))
        for repeat_id in range(repeat_num):
            key = run_id(problem.problem_id, repeat_id)
            if args.resume and key in rows_by_run_id:
                continue
            print(f"[rsd] running problem_id={problem.problem_id} repeat_id={repeat_id}", flush=True)
            wall_start = time.time()
            metadata_list: list[dict[str, Any]] = []
            error = None
            try:
                metadata_list = router.run(
                    problem_prompt=prompt,
                    prm_problem=prm_problem,
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
    parser = argparse.ArgumentParser(description="Run the RSD comparison baseline.")
    parser.add_argument("--sarr-config", default="configs/sarr_code_aggressive.json", help="Project config used for dataset paths.")
    parser.add_argument("--rsd-config", default=str(BASELINE_ROOT / "config.example.json"), help="RSD endpoint and hyperparameter config.")
    parser.add_argument("--dataset", default="aime24", choices=SUPPORTED_DATASETS)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeat-num", type=int, default=None)
    parser.add_argument("--prompt-style", choices=["upstream", "project"], default=None)
    parser.add_argument("--model-pair", default=None, help="One of the four Qwen3/DeepSeek model-pair ids.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens-per-call", type=int, default=None)
    parser.add_argument("--step-max-tokens", type=int, default=None)
    parser.add_argument("--prm-threshold", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
