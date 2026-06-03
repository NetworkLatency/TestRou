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

from relaygen_core import RelayGenHyperparams, RelayGenRouter, extract_answer_text, parse_endpoints, route_stats
from baselines.model_pairs import DEFAULT_MODEL_PAIRS, apply_model_pair, model_family
from sarr_code import SARRConfig
from sarr_code.eval import benchmark_eval_match, build_summary_metrics, load_eval_dataset, write_summary_files
from sarr_code.eval.prompts import build_problem_text
from sarr_code.safety import extract_answer_from_final_step
from sarr_code.trace import write_json


SUPPORTED_DATASETS = ("math500", "aime24", "aime25", "gpqa", "gpqa_diamond")
DEFAULT_CONFIG: dict[str, Any] = {
    "repeat_num": 1,
    "output_dir": "relaygen_results",
    "prompt_style": "upstream",
    "model_pair": "qwen3_1p7b_qwen3_32b",
    "model_pairs": DEFAULT_MODEL_PAIRS,
    "budget": 16384,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": None,
    "cue_family": "qwen3",
    "answer_model": "small",
    "min_tokens_large": 5,
    "enable_thinking": True,
    "include_stop_str_in_output": True,
    "endpoints": {
        "base": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["base"],
        "small": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["small"],
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
        raise ValueError(f"RelayGen config must be a JSON object: {source}")
    return deep_update(DEFAULT_CONFIG, data)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, value in {
        "repeat_num": args.repeat_num,
        "output_dir": args.output_root,
        "prompt_style": args.prompt_style,
        "model_pair": args.model_pair,
        "budget": args.budget,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "cue_family": args.cue_family,
        "answer_model": args.answer_model,
        "min_tokens_large": args.min_tokens_large,
    }.items():
        if value is not None:
            config[key] = value
    apply_model_pair(config, config.get("model_pair"), small_key="small", base_key="base")
    if args.cue_family is None:
        family = model_family(str(config["endpoints"]["base"]["model"]))
        if family in {"qwen3", "r1"}:
            config["cue_family"] = family
    return config


def hyperparams_from_config(config: dict[str, Any]) -> RelayGenHyperparams:
    top_k = config.get("top_k")
    return RelayGenHyperparams(
        budget=int(config["budget"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        top_k=None if top_k is None else int(top_k),
        presence_penalty=config.get("presence_penalty"),
        cue_family=str(config["cue_family"]),
        answer_model=str(config["answer_model"]),
        min_tokens_large=int(config["min_tokens_large"]),
        enable_thinking=bool(config["enable_thinking"]),
        include_stop_str_in_output=bool(config["include_stop_str_in_output"]),
    )


def output_root_from_config(config: dict[str, Any]) -> Path:
    output_root = Path(str(config["output_dir"]))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root


def _model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "p")


def variant_from_config(config: dict[str, Any], variant: str | None) -> str:
    if variant:
        return variant
    return f"relaygen_{config['model_pair']}_{config['cue_family']}_budget{config['budget']}"


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


def choice_options(raw: dict[str, Any]) -> dict[str, str] | None:
    if isinstance(raw.get("choices"), list):
        values = raw["choices"]
        if len(values) >= 4:
            return {label: str(value) for label, value in zip(["A", "B", "C", "D"], values)}
    values_by_key = {}
    for label in ["A", "B", "C", "D"]:
        value = raw.get(label) or raw.get(f"choice_{label.lower()}") or raw.get(f"Choice {label}")
        if value:
            values_by_key[label] = str(value)
    if len(values_by_key) == 4:
        return values_by_key
    if raw.get("Correct Answer"):
        values = [
            raw.get("Correct Answer"),
            raw.get("Incorrect Answer 1"),
            raw.get("Incorrect Answer 2"),
            raw.get("Incorrect Answer 3"),
        ]
        if all(value is not None for value in values):
            return {label: str(value) for label, value in zip(["A", "B", "C", "D"], values)}
    return None


def build_upstream_problem_input(raw: dict[str, Any], dataset: str) -> tuple[str, dict[str, str] | None]:
    if dataset in {"math500", "aime24", "aime25"}:
        body = raw.get("problem") or raw.get("question") or raw.get("prompt") or ""
        return str(body).strip(), None
    if dataset in {"gpqa", "gpqa_diamond"}:
        body = raw.get("problem") or raw.get("Question") or raw.get("question") or ""
        return str(body).strip(), choice_options(raw)
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_first_user_msg(problem: str, options: dict[str, str] | None = None) -> str:
    if options is None:
        return f"""
        {problem}
        Please reason step by step, and put your final answer within \\boxed{{}}.
        """
    return f"""
        Please solve this multiple choice question.

        Question: {problem}.

        Options: 
        A: {options["A"]}
        B: {options["B"]}
        C: {options["C"]}
        D: {options["D"]}

        Please provide your answer in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).
        """


def build_prompt(problem, dataset: str, prompt_style: str) -> str:
    if prompt_style == "upstream":
        problem_input, options = build_upstream_problem_input(problem.raw, dataset)
        return get_first_user_msg(problem_input, options=options)
    if prompt_style == "project":
        return build_problem_text(problem.raw, dataset)
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
        "method": "relaygen",
        "prompt_style": config["prompt_style"],
        "model_pair": config["model_pair"],
        "budget": config["budget"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config.get("top_k"),
        "cue_family": config["cue_family"],
        "answer_model": config["answer_model"],
        "base_model": config["endpoints"]["base"]["model"],
        "small_model": config["endpoints"]["small"]["model"],
        "total_wall_time": wall_time,
        "problem_wall_time": wall_time,
        "error": error,
        **stats,
    }
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
            "method": "relaygen",
            "prompt_style": config["prompt_style"],
            "model_pair": config["model_pair"],
            "budget": config["budget"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config.get("top_k"),
            "cue_family": config["cue_family"],
            "answer_model": config["answer_model"],
            "base_model": config["endpoints"]["base"]["model"],
            "small_model": config["endpoints"]["small"]["model"],
            "repeat_num": config["repeat_num"],
            "num_failed": sum(1 for row in rows if row.get("error")),
            "avg_total_switches": average(rows, "total_switches"),
            "avg_switch_rate": average(rows, "switch_rate"),
            "avg_large_model_percentage": average(rows, "large_model_percentage"),
            "avg_small_model_percentage": average(rows, "small_model_percentage"),
            "total_large_model_tokens": sum(int(row.get("large_model_tokens") or 0) for row in rows),
            "total_small_model_tokens": sum(int(row.get("small_model_tokens") or 0) for row in rows),
        }
    )
    write_summary_files(summary_path(output_root, dataset, variant), rows, metrics)
    write_predictions(output_root, dataset, variant, rows, int(config["repeat_num"]))


def run_experiment(args: argparse.Namespace) -> None:
    config = apply_cli_overrides(load_config(args.relaygen_config), args)
    sarr_cfg = SARRConfig.from_json(args.sarr_config)
    problems = load_eval_dataset(args.dataset, sarr_cfg, max_problems=args.max_problems)
    output_root = output_root_from_config(config)
    variant = variant_from_config(config, args.variant)
    repeat_num = int(config["repeat_num"])
    router = RelayGenRouter(
        endpoints=parse_endpoints(config["endpoints"]),
        hyperparams=hyperparams_from_config(config),
    )

    print(f"[relaygen] loaded {len(problems)} problem(s) from {args.dataset}", flush=True)
    print(f"[relaygen] variant={variant} output_root={output_root}", flush=True)
    print(
        "[relaygen] "
        f"base={config['endpoints']['base']['model']} "
        f"small={config['endpoints']['small']['model']} "
        f"budget={config['budget']} cue_family={config['cue_family']}",
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
            print(f"[relaygen] running problem_id={problem.problem_id} repeat_id={repeat_id}", flush=True)
            wall_start = time.time()
            metadata_list: list[dict[str, Any]] = []
            error = None
            try:
                metadata_list = router.run(
                    problem_prompt=prompt,
                    dataset_name=args.dataset,
                    problem_id=problem.problem_id,
                    repeat_id=repeat_id,
                    verbose=args.verbose,
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
    parser = argparse.ArgumentParser(description="Run the RelayGen comparison baseline.")
    parser.add_argument("--sarr-config", default="configs/sarr_code_aggressive.json", help="Project config used for dataset paths.")
    parser.add_argument("--relaygen-config", default=str(BASELINE_ROOT / "config.example.json"), help="RelayGen endpoint and hyperparameter config.")
    parser.add_argument("--dataset", default="aime25", choices=SUPPORTED_DATASETS)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeat-num", type=int, default=None)
    parser.add_argument("--prompt-style", choices=["upstream", "project"], default=None)
    parser.add_argument("--model-pair", default=None, help="One of the four Qwen3/DeepSeek model-pair ids.")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--cue-family", choices=["qwen3", "r1"], default=None)
    parser.add_argument("--answer-model", choices=["small", "base"], default=None)
    parser.add_argument("--min-tokens-large", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
