from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.context_budget import ContextBudgetExceeded
from bpa.engines import init_engines
from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import load_eval_dataset
from bpa.eval.exp_sampling_disagreement import _max, _mean, _sample_probe_rollouts
from bpa.pipeline import _is_eos_finish, _llm_generate_step, _slm_generate_step
from bpa.safety import ensure_step_terminator, extract_answer, update_strict_step_repetition
from bpa.state import GenerationState, Phase, RepetitionState, TraceEvent
from bpa.trace import BPAResult, json_safe


SUMMARY_FIELDS = [
    "dataset",
    "problem_id",
    "question_id",
    "gold_answer",
    "final_answer",
    "correct",
    "metric",
    "threshold",
    "threshold_quantile",
    "num_boundaries",
    "num_llm_routed_steps",
    "num_slm_steps",
    "max_metric_value",
    "mean_metric_value",
    "total_wall_time",
    "slm_decode_tokens",
    "slm_prefill_tokens",
    "llm_decode_tokens",
    "llm_prefill_tokens",
    "slm_generate_calls",
    "llm_generate_calls",
    "probe_decode_tokens",
    "probe_prefill_tokens",
    "probe_generate_calls",
    "probe_wall_time",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _default_probe_path(config: BPAConfig, dataset: str) -> Path:
    return Path(config.output_dir) / "diagnostics" / "sampling_disagreement" / dataset / "probes.jsonl"


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if value is True or text == "true":
        return True
    if value is False or text == "false":
        return False
    return None


def _is_initial_probe(row: dict[str, Any]) -> bool:
    parsed = _parse_bool(row.get("is_initial_probe"))
    if parsed is not None:
        return parsed
    try:
        if int(row.get("boundary_idx", 0)) < 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return int(row.get("prefix_char_len", 1)) == 0
    except (TypeError, ValueError):
        return False


def threshold_from_probes(path: Path, metric: str, quantile: float, *, include_initial_probe: bool = False) -> float:
    rows = _read_jsonl(path)
    values = []
    for row in rows:
        if not include_initial_probe and _is_initial_probe(row):
            continue
        value = row.get(metric)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        raise ValueError(f"No values for metric {metric!r} in {path}")
    return float(np.quantile(values, quantile))


def run_disagreement_routing(
    problem_text: str,
    slm,
    llm,
    config: BPAConfig,
    *,
    metric: str,
    threshold: float,
    probe_k: int = 4,
    probe_temperature: float = 0.7,
    probe_max_tokens: int = 32,
    probe_stop: str = "\n\n",
) -> tuple[BPAResult, list[dict[str, Any]], dict[str, float | int]]:
    state = GenerationState(problem_text=problem_text, generation_protocol="disagreement_top_quantile_routing")
    rep = RepetitionState()
    start_time = time.time()
    step_logs: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    probe_cost = {
        "probe_decode_tokens": 0,
        "probe_prefill_tokens": 0,
        "probe_generate_calls": 0,
        "probe_wall_time": 0.0,
    }
    boundary_idx = 0

    while state.phase != Phase.DONE:
        if state.slm_decode_tokens + state.llm_decode_tokens >= config.max_total_tokens:
            state.trace.append(TraceEvent(state.step_count, "total_token_budget_exhausted", {}))
            state.stop_reason = "total_token_budget"
            break

        route_to_llm = False
        metric_value = None
        try:
            probe_row, cost = _sample_probe_rollouts(
                state,
                slm,
                config,
                probe_k=probe_k,
                probe_temperature=probe_temperature,
                probe_max_tokens=probe_max_tokens,
                probe_stop=probe_stop,
            )
        except ContextBudgetExceeded as exc:
            state.phase = Phase.DONE
            state.stop_reason = "context_budget"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
            break
        is_initial_probe = len(state.assistant_prefix_text) == 0
        probe_row["boundary_idx"] = -1 if is_initial_probe else boundary_idx
        probe_row["target_step_idx"] = state.step_count
        probe_row["is_initial_probe"] = is_initial_probe
        if not is_initial_probe:
            boundary_idx += 1
        for key in probe_cost:
            probe_cost[key] += cost[key]
        metric_value = probe_row.get(metric)
        route_to_llm = (not is_initial_probe) and metric_value is not None and float(metric_value) >= threshold

        decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
        try:
            if route_to_llm:
                step_text, finish = _llm_generate_step(state, llm, config)
            else:
                step_text, finish = _slm_generate_step(state, slm, config)
        except ContextBudgetExceeded as exc:
            state.phase = Phase.DONE
            state.stop_reason = "context_budget"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
            break
        step_text_normalized = ensure_step_terminator(step_text, finish)
        generated_step_tokens = state.slm_decode_tokens + state.llm_decode_tokens - decode_tokens_before
        state.assistant_prefix_text += step_text_normalized
        state.step_count += 1

        probe_row["metric"] = metric
        probe_row["metric_value"] = metric_value
        probe_row["threshold"] = threshold
        probe_row["routed_to_llm"] = route_to_llm
        probe_row["main_step_text"] = step_text_normalized
        probe_row["main_step_finish_reason"] = finish
        boundary_rows.append(probe_row)

        log_row = {
            "step_idx": state.step_count - 1,
            "decision": "llm_disagreement_route" if route_to_llm else "slm_direct",
            "generation_source": "llm" if route_to_llm else "slm",
            "finish_reason": finish,
            "step_text": step_text_normalized,
            "generated_step_tokens": generated_step_tokens,
            "metric": metric if metric_value is not None else None,
            "metric_value": metric_value,
            "threshold": threshold if metric_value is not None else None,
        }
        step_logs.append(log_row)

        if generated_step_tokens <= 0 and not step_text_normalized.strip():
            state.phase = Phase.DONE
            state.stop_reason = "empty_step"
            state.trace.append(TraceEvent(state.step_count, "empty_step", {}))
            break

        trigger = update_strict_step_repetition(rep, step_text_normalized)
        if trigger is not None:
            state.phase = Phase.DONE
            state.stop_reason = trigger
            state.trace.append(TraceEvent(state.step_count, "step_repetition_stop", {"trigger_reason": trigger}))
            break

        if _is_eos_finish(finish):
            state.phase = Phase.DONE
            state.stop_reason = "eos"

    state.trace.append(TraceEvent(state.step_count, "step_logs", {"steps": step_logs}))
    result = BPAResult(answer=extract_answer(state.assistant_prefix_text), state=state, total_wall_time=time.time() - start_time)
    return result, boundary_rows, probe_cost


def _summary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("correct") not in (None, "")]
    correct = [row for row in evaluated if row.get("correct") is True or str(row.get("correct")).lower() == "true"]
    return {
        "num_problems": len(rows),
        "num_evaluated": len(evaluated),
        "num_correct": len(correct),
        "accuracy": len(correct) / len(evaluated) if evaluated else None,
        "avg_total_wall_time": _mean([row.get("total_wall_time") for row in rows]),
        "avg_num_llm_routed_steps": _mean([row.get("num_llm_routed_steps") for row in rows]),
        "total_llm_decode_tokens": sum(float(row.get("llm_decode_tokens") or 0.0) for row in rows),
        "total_llm_prefill_tokens": sum(float(row.get("llm_prefill_tokens") or 0.0) for row in rows),
        "total_slm_decode_tokens": sum(float(row.get("slm_decode_tokens") or 0.0) for row in rows),
        "total_probe_decode_tokens": sum(float(row.get("probe_decode_tokens") or 0.0) for row in rows),
    }


def write_outputs(out_dir: Path, summary_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    fieldnames = list(SUMMARY_FIELDS)
    extra_fields = sorted({key for row in summary_rows for key in row} - set(fieldnames))
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(json_safe(summary_rows))

    with (out_dir / "routing_boundaries.jsonl").open("w", encoding="utf-8") as f:
        for row in boundary_rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")

    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(_summary_metrics(summary_rows)), f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run top-quantile disagreement routing as a sanity check.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument("--threshold-source", default=None)
    parser.add_argument("--threshold-quantile", type=float, default=0.8)
    parser.add_argument("--metric", default="rhs_number_vote_disagreement")
    parser.add_argument("--probe-k", type=int, default=4)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--probe-max-tokens", type=int, default=32)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    threshold_source = Path(args.threshold_source) if args.threshold_source else _default_probe_path(config, args.dataset)
    threshold = threshold_from_probes(threshold_source, args.metric, args.threshold_quantile)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    slm, llm = init_engines(config)

    summary_rows: list[dict[str, Any]] = []
    all_boundary_rows: list[dict[str, Any]] = []
    for problem in tqdm(problems, desc=f"disagreement_routing:{args.dataset}"):
        result, boundary_rows, probe_cost = run_disagreement_routing(
            problem.problem_text,
            slm,
            llm,
            config,
            metric=args.metric,
            threshold=threshold,
            probe_k=args.probe_k,
            probe_temperature=args.probe_temperature,
            probe_max_tokens=args.probe_max_tokens,
        )
        correct = None
        if problem.gold_answer is not None:
            correct = benchmark_eval_match(result.answer, problem.gold_answer, args.dataset)
            result.correct = correct
        real_boundary_rows = [row for row in boundary_rows if not _is_initial_probe(row)]
        metric_values = [row.get(args.metric) for row in real_boundary_rows]
        summary_rows.append(
            {
                "dataset": args.dataset,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                "gold_answer": problem.gold_answer,
                "final_answer": result.answer,
                "correct": correct,
                "metric": args.metric,
                "threshold": threshold,
                "threshold_quantile": args.threshold_quantile,
                "num_boundaries": len(real_boundary_rows),
                "num_llm_routed_steps": sum(1 for row in real_boundary_rows if row.get("routed_to_llm")),
                "num_slm_steps": result.state.slm_generate_calls,
                "max_metric_value": _max(metric_values),
                "mean_metric_value": _mean(metric_values),
                "total_wall_time": result.total_wall_time,
                "slm_decode_tokens": result.state.slm_decode_tokens,
                "slm_prefill_tokens": result.state.slm_prefill_tokens,
                "llm_decode_tokens": result.state.llm_decode_tokens,
                "llm_prefill_tokens": result.state.llm_prefill_tokens,
                "slm_generate_calls": result.state.slm_generate_calls,
                "llm_generate_calls": result.state.llm_full_calls,
                **probe_cost,
            }
        )
        for row in boundary_rows:
            all_boundary_rows.append({"dataset": args.dataset, "problem_id": problem.problem_id, "question_id": problem.question_id, **row})
        if config.reset_prefix_cache_after_problem:
            slm.clear_runtime_cache()
            llm.clear_runtime_cache()

    out_dir = Path(config.output_dir) / "diagnostics" / "disagreement_routing" / args.dataset
    write_outputs(out_dir, summary_rows, all_boundary_rows)
    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'routing_boundaries.jsonl'}")


if __name__ == "__main__":
    main()
