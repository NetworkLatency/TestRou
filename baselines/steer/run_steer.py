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

from steer_core import ChatGenerator, SteerBatchRouter, SteerHyperparams, parse_chat_endpoint
from baselines.model_pairs import DEFAULT_MODEL_PAIRS, apply_model_pair
from sarr_code import SARRConfig
from sarr_code.eval import benchmark_eval_match, build_summary_metrics, load_eval_dataset, write_summary_files
from sarr_code.eval.prompts import build_problem_text
from sarr_code.safety import extract_answer_from_final_step
from sarr_code.trace import write_json


SUPPORTED_DATASETS = ("math500", "aime24", "aime25", "gpqa", "gpqa_diamond")
DEFAULT_CONFIG: dict[str, Any] = {
    "output_dir": "steer_results",
    "prompt_type": "reasoning-chat",
    "prompt_style": "steer",
    "model_pair": "qwen3_1p7b_qwen3_32b",
    "model_pairs": DEFAULT_MODEL_PAIRS,
    "repeat_num": 1,
    "seed": 0,
    "temperature": 0.7,
    "top_p": 1.0,
    "n_sampling": 1,
    "max_tokens_per_call": 16384,
    "min_tokens": 2,
    "max_steps": 100,
    "patience": 5,
    "step_word": "\n\n",
    "logprobs": 5,
    "use_step_reliability": True,
    "reliability_metric": "R_eu",
    "reliability_mode": "math_only_avg",
    "reliability_k_top": 1,
    "reliability_target_usage": "gmm_responsibility",
    "draft_gmm_threshold": 0.4,
    "target_gmm_threshold": 0.4,
    "draft": {
        "model": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["small"]["model"],
        "base_url": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["small"]["base_url"],
        "api_key": "EMPTY",
        "timeout": 3600,
    },
    "target": {
        "model": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["base"]["model"],
        "base_url": DEFAULT_MODEL_PAIRS["qwen3_1p7b_qwen3_32b"]["base"]["base_url"],
        "api_key": "EMPTY",
        "timeout": 3600,
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
        raise ValueError(f"STEER config must be a JSON object: {source}")
    return deep_update(DEFAULT_CONFIG, data)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, value in {
        "output_dir": args.output_root,
        "model_pair": args.model_pair,
        "repeat_num": args.repeat_num,
        "temperature": args.temperature,
        "max_tokens_per_call": args.max_tokens_per_call,
        "max_steps": args.max_steps,
        "reliability_mode": args.reliability_mode,
        "reliability_k_top": args.reliability_k_top,
        "draft_gmm_threshold": args.draft_gmm_threshold,
        "target_gmm_threshold": args.target_gmm_threshold,
        "prompt_type": args.prompt_type,
        "prompt_style": args.prompt_style,
    }.items():
        if value is not None:
            config[key] = value
    apply_model_pair(config, config.get("model_pair"), small_key="draft", base_key="target")
    config["draft"] = deepcopy(config["endpoints"]["draft"])
    config["target"] = deepcopy(config["endpoints"]["target"])
    return config


def output_root_from_config(config: dict[str, Any]) -> Path:
    root = Path(str(config["output_dir"]))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def variant_from_config(config: dict[str, Any], variant: str | None) -> str:
    if variant:
        return variant
    dthr = str(config["draft_gmm_threshold"]).replace(".", "p")
    tthr = str(config["target_gmm_threshold"]).replace(".", "p")
    return f"steer_{config['model_pair']}_{config['reliability_mode']}_ktop{config['reliability_k_top']}_d{dthr}_t{tthr}"


def summary_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary.csv"


def metrics_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary_metrics.json"


def problem_root(output_root: Path, dataset: str, variant: str, problem_id: Any) -> Path:
    return output_root / dataset / variant / str(problem_id)


def run_id(problem_id: Any, repeat_id: int) -> str:
    return f"{problem_id}:{repeat_id}"


def load_existing_summary_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            str(row.get("run_id") or run_id(row.get("problem_id"), int(row.get("repeat_id") or 0))): dict(row)
            for row in csv.DictReader(f)
        }


def problem_complete(output_root: Path, dataset: str, variant: str, problem_id: Any, repeat_id: int) -> bool:
    root = problem_root(output_root, dataset, variant, problem_id)
    return (root / f"{repeat_id}.steps.json").exists() and (root / f"{repeat_id}.problem.json").exists()


def bare_problem_text(raw: dict[str, Any], dataset: str) -> str:
    if dataset in {"math500", "aime24", "aime25"}:
        return str(raw.get("problem") or raw.get("question") or raw.get("prompt") or "")
    if dataset in {"gpqa", "gpqa_diamond"}:
        body = str(raw.get("problem") or raw.get("Question") or raw.get("question") or "")
        choices = []
        for label in ["A", "B", "C", "D"]:
            value = raw.get(label) or raw.get(f"choice_{label.lower()}") or raw.get(f"Choice {label}")
            if value:
                choices.append(f"{label}. {value}")
        if not choices and raw.get("Correct Answer"):
            values = [
                raw.get("Correct Answer"),
                raw.get("Incorrect Answer 1"),
                raw.get("Incorrect Answer 2"),
                raw.get("Incorrect Answer 3"),
            ]
            choices = [f"{label}. {value}" for label, value in zip(["A", "B", "C", "D"], values) if value]
        return f"{body}\n\n" + "\n".join(choices) if choices else body
    raise ValueError(f"Unsupported dataset: {dataset}")


def steer_prompt(input_text: str, prompt_type: str) -> str:
    if prompt_type == "reasoning-chat":
        return (
            "Please reason step by step, separate logical reasoning steps with two newline characters, "
            "and put your final answer within \\boxed{}.\n\n"
            f"Problem: {input_text}"
        )
    raise ValueError("prompt_type must be 'reasoning-chat'.")


def build_prompt(problem, dataset: str, config: dict[str, Any]) -> str:
    if config["prompt_style"] == "project":
        return build_problem_text(problem.raw, dataset)
    if config["prompt_style"] == "steer":
        return steer_prompt(bare_problem_text(problem.raw, dataset), str(config["prompt_type"]))
    raise ValueError("prompt_style must be 'steer' or 'project'.")


def hyperparams_from_config(config: dict[str, Any]) -> SteerHyperparams:
    return SteerHyperparams(
        seed=int(config["seed"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        n_sampling=int(config["n_sampling"]),
        max_tokens_per_call=int(config["max_tokens_per_call"]),
        min_tokens=int(config["min_tokens"]),
        max_steps=int(config["max_steps"]),
        patience=int(config["patience"]),
        step_word=str(config["step_word"]),
        logprobs=int(config["logprobs"]),
        reliability_metric=str(config["reliability_metric"]),
        reliability_mode=str(config["reliability_mode"]),
        reliability_k_top=int(config["reliability_k_top"]),
        reliability_target_usage=str(config["reliability_target_usage"]),
        draft_gmm_threshold=float(config["draft_gmm_threshold"]),
        target_gmm_threshold=float(config["target_gmm_threshold"]),
    )


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return (sum(values) / len(values)) if values else None


def summary_row(*, dataset: str, variant: str, problem, repeat_id: int, state, correct: bool | None, config: dict[str, Any], wall_time: float) -> dict[str, Any]:
    answer = extract_answer_from_final_step(state.output)
    draft_tokens, target_tokens, discarded_tokens = state.token_counts
    row = {
        "run_id": run_id(problem.problem_id, repeat_id),
        "dataset": dataset,
        "variant": variant,
        "problem_id": problem.problem_id,
        "repeat_id": repeat_id,
        "question_id": problem.question_id,
        "gold_answer": problem.gold_answer,
        "answer": answer,
        "answer_text": state.output,
        "correct": correct,
        "method": "steer",
        "prompt_type": config["prompt_type"],
        "prompt_style": config["prompt_style"],
        "model_pair": config["model_pair"],
        "draft_model": config["draft"]["model"],
        "target_model": config["target"]["model"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens_per_call": config["max_tokens_per_call"],
        "max_steps": config["max_steps"],
        "reliability_metric": config["reliability_metric"],
        "reliability_mode": config["reliability_mode"],
        "reliability_k_top": config["reliability_k_top"],
        "draft_gmm_threshold": config["draft_gmm_threshold"],
        "target_gmm_threshold": config["target_gmm_threshold"],
        "step_count": len(state.steps),
        "draft_step_count": sum(1 for _, client_id in state.step_info if client_id == 1),
        "target_step_count": sum(1 for _, client_id in state.step_info if client_id == 2),
        "draft_decode_tokens": draft_tokens,
        "target_decode_tokens": target_tokens,
        "discarded_draft_tokens": discarded_tokens,
        "slm_decode_tokens": draft_tokens + discarded_tokens,
        "llm_decode_tokens": target_tokens,
        "slm_prefill_tokens": 0,
        "llm_prefill_tokens": 0,
        "slm_generate_calls": sum(1 for _, client_id in state.step_info if client_id == 1),
        "llm_generate_calls": sum(1 for _, client_id in state.step_info if client_id == 2),
        "llm_full_calls": sum(1 for _, client_id in state.step_info if client_id == 2),
        "llm_scoring_calls": 0,
        "route_sequence": ",".join("draft" if client_id == 1 else "target" for _, client_id in state.step_info),
        "avg_decision_value": average([{"v": item["decision_value"]} for item in state.reliabilities], "v"),
        "total_wall_time": wall_time,
        "problem_wall_time": wall_time,
        "stop_reason": "finished",
    }
    return row


def write_problem_outputs(*, output_root: Path, dataset: str, variant: str, problem, repeat_id: int, state, row: dict[str, Any], config: dict[str, Any]) -> None:
    root = problem_root(output_root, dataset, variant, problem.problem_id)
    write_json(root / f"{repeat_id}.steps.json", state.steps)
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
            "token_counts": state.token_counts,
            "turn_info": state.step_info,
            "reliabilities": state.reliabilities,
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


def write_summary(*, output_root: Path, dataset: str, variant: str, rows: list[dict[str, Any]], dataset_wall_time: float, config: dict[str, Any]) -> None:
    metrics = build_summary_metrics(dataset, variant, rows, dataset_wall_time)
    metrics.update(
        {
            "method": "steer",
            "prompt_type": config["prompt_type"],
            "prompt_style": config["prompt_style"],
            "repeat_num": config["repeat_num"],
            "model_pair": config["model_pair"],
            "draft_model": config["draft"]["model"],
            "target_model": config["target"]["model"],
            "reliability_metric": config["reliability_metric"],
            "reliability_mode": config["reliability_mode"],
            "reliability_k_top": config["reliability_k_top"],
            "draft_gmm_threshold": config["draft_gmm_threshold"],
            "target_gmm_threshold": config["target_gmm_threshold"],
            "avg_step_count": average(rows, "step_count"),
            "avg_draft_step_count": average(rows, "draft_step_count"),
            "avg_target_step_count": average(rows, "target_step_count"),
            "total_draft_decode_tokens": sum(int(row.get("draft_decode_tokens") or 0) for row in rows),
            "total_target_decode_tokens": sum(int(row.get("target_decode_tokens") or 0) for row in rows),
            "total_discarded_draft_tokens": sum(int(row.get("discarded_draft_tokens") or 0) for row in rows),
        }
    )
    write_summary_files(summary_path(output_root, dataset, variant), rows, metrics)
    write_predictions(output_root, dataset, variant, rows, int(config["repeat_num"]))


def ordered_rows(problems, rows_by_run_id: dict[str, dict[str, Any]], repeat_num: int) -> list[dict[str, Any]]:
    rows = []
    for problem in problems:
        for repeat_id in range(repeat_num):
            key = run_id(problem.problem_id, repeat_id)
            if key in rows_by_run_id:
                rows.append(rows_by_run_id[key])
    return rows


def run_experiment(args: argparse.Namespace) -> None:
    config = apply_cli_overrides(load_config(args.steer_config), args)
    repeat_num = int(config["repeat_num"])
    sarr_cfg = SARRConfig.from_json(args.sarr_config)
    problems = load_eval_dataset(args.dataset, sarr_cfg, max_problems=args.max_problems)
    output_root = output_root_from_config(config)
    variant = variant_from_config(config, args.variant)

    print(f"[steer] loaded {len(problems)} problem(s) from {args.dataset}", flush=True)
    print(f"[steer] variant={variant} output_root={output_root}", flush=True)
    print(
        f"[steer] model_pair={config['model_pair']} draft={config['draft']['model']} target={config['target']['model']} "
        f"mode={config['reliability_mode']} k_top={config['reliability_k_top']}",
        flush=True,
    )

    existing = load_existing_summary_rows(summary_path(output_root, args.dataset, variant)) if args.resume else {}
    pending_items: list[tuple[Any, int, str]] = []
    pending_lookup: list[tuple[Any, int]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    skipped = 0
    for problem in problems:
        for repeat_id in range(repeat_num):
            key = run_id(problem.problem_id, repeat_id)
            if args.resume and key in existing and problem_complete(output_root, args.dataset, variant, problem.problem_id, repeat_id):
                rows_by_id[key] = existing[key]
                skipped += 1
                continue
            pending_items.append((problem.problem_id, repeat_id, build_prompt(problem, args.dataset, config)))
            pending_lookup.append((problem, repeat_id))

    if pending_items:
        draft = ChatGenerator(parse_chat_endpoint(config["draft"]))
        target = ChatGenerator(parse_chat_endpoint(config["target"]))
        router = SteerBatchRouter(draft=draft, target=target, hyperparams=hyperparams_from_config(config))
        start = time.time()
        states = router.run(pending_items)
        total_time = time.time() - start
        avg_time = total_time / max(1, len(states))
        for state, (problem, repeat_id) in tqdm(list(zip(states, pending_lookup)), desc=f"{variant}:{args.dataset}"):
            correct = benchmark_eval_match(state.output, problem.gold_answer, args.dataset) if problem.gold_answer is not None else None
            row = summary_row(
                dataset=args.dataset,
                variant=variant,
                problem=problem,
                repeat_id=repeat_id,
                state=state,
                correct=correct,
                config=config,
                wall_time=avg_time,
            )
            rows_by_id[run_id(problem.problem_id, repeat_id)] = row
            write_problem_outputs(
                output_root=output_root,
                dataset=args.dataset,
                variant=variant,
                problem=problem,
                repeat_id=repeat_id,
                state=state,
                row=row,
                config=config,
            )

    rows = ordered_rows(problems, rows_by_id, repeat_num)
    write_summary(
        output_root=output_root,
        dataset=args.dataset,
        variant=variant,
        rows=rows,
        dataset_wall_time=sum(float(row.get("problem_wall_time") or 0.0) for row in rows),
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
    parser = argparse.ArgumentParser(description="Run the STEER comparison baseline.")
    parser.add_argument("--sarr-config", default="configs/sarr_code_aggressive.json", help="Project config used for dataset paths.")
    parser.add_argument("--steer-config", default=str(BASELINE_ROOT / "config.example.json"), help="STEER model and hyperparameter config.")
    parser.add_argument("--dataset", default="math500", choices=SUPPORTED_DATASETS)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeat-num", type=int, default=None)
    parser.add_argument("--model-pair", default=None, help="One of the four Qwen3/DeepSeek model-pair ids.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens-per-call", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--reliability-mode", default=None)
    parser.add_argument("--reliability-k-top", type=int, default=None)
    parser.add_argument("--draft-gmm-threshold", type=float, default=None)
    parser.add_argument("--target-gmm-threshold", type=float, default=None)
    parser.add_argument("--prompt-type", choices=["reasoning-chat"], default=None)
    parser.add_argument("--prompt-style", choices=["steer", "project"], default=None)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
