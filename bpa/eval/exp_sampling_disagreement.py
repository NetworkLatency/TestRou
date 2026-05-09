from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.context_budget import ContextBudgetExceeded, generation_budget_for_rendered
from bpa.engines import init_engines, logprob_value
from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import EvalProblem, load_eval_dataset
from bpa.eval.sampling_disagreement import (
    extract_number_signature,
    extract_novel_number_signature,
    extract_operation_signature,
    extract_rhs_number_signature,
    extract_structured_signature,
    rollout_disagreement_metrics,
)
from bpa.pipeline import _is_eos_finish, _slm_generate_step
from bpa.render import render_for_continuation
from bpa.safety import ensure_step_terminator, extract_answer, update_strict_step_repetition
from bpa.state import GenerationState, Phase, RepetitionState, TraceEvent
from bpa.trace import BPAResult, json_safe


PROBLEM_SUMMARY_FIELDS = [
    "problem_id",
    "question_id",
    "gold_answer",
    "final_answer",
    "correct",
    "num_boundaries",
    "num_initial_probes",
    "mean_structured_disagreement",
    "max_structured_disagreement",
    "mean_operation_vote_disagreement",
    "max_operation_vote_disagreement",
    "mean_number_vote_disagreement",
    "max_number_vote_disagreement",
    "mean_novel_number_vote_disagreement",
    "max_novel_number_vote_disagreement",
    "mean_rhs_number_vote_disagreement",
    "max_rhs_number_vote_disagreement",
    "mean_self_bleu_disagreement",
    "max_self_bleu_disagreement",
    "mean_char_jaccard_disagreement",
    "max_char_jaccard_disagreement",
    "mean_score_variance",
    "max_score_variance",
    "total_wall_time",
    "slm_decode_tokens",
    "slm_prefill_tokens",
    "slm_generate_calls",
    "probe_decode_tokens",
    "probe_prefill_tokens",
    "probe_generate_calls",
    "probe_wall_time",
]


def _completion_mean_logprob(completion: Any) -> float | None:
    token_ids = list(getattr(completion, "token_ids", []) or [])
    logprob_steps = list(getattr(completion, "logprobs", []) or [])
    if not token_ids or not logprob_steps:
        return None

    values: list[float] = []
    for token_id, step in zip(token_ids, logprob_steps):
        if not isinstance(step, dict):
            continue
        record = step.get(token_id)
        if record is None:
            record = step.get(str(token_id))
        if record is None and len(step) == 1:
            record = next(iter(step.values()))
        if record is not None:
            values.append(logprob_value(record))
    return (sum(values) / len(values)) if values else None


def _sample_probe_rollouts(
    state: GenerationState,
    slm,
    config: BPAConfig,
    probe_k: int,
    probe_temperature: float,
    probe_max_tokens: int,
    probe_stop: str,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    tokenizer = slm.ensure_tokenizer()
    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, tokenizer)
    max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, slm, config, probe_max_tokens)
    sampling = slm.sampling_params(
        max_tokens=max_tokens,
        temperature=probe_temperature,
        stop=[probe_stop],
        include_stop_str_in_output=True,
        logprobs=1,
        n=probe_k,
    )
    generate_start = time.time()
    out = slm.generate(rendered, sampling)[0]
    probe_wall_time = time.time() - generate_start

    context_text = f"{state.problem_text}\n{state.assistant_prefix_text}"
    rollouts = []
    probe_decode_tokens = 0
    for completion in getattr(out, "outputs", []) or []:
        text = getattr(completion, "text", "") or ""
        token_ids = list(getattr(completion, "token_ids", []) or [])
        mean_logprob = _completion_mean_logprob(completion)
        signature = extract_structured_signature(text)
        operation_signature = extract_operation_signature(text)
        number_signature = extract_number_signature(text)
        novel_number_signature = extract_novel_number_signature(text, context_text)
        rhs_number_signature = extract_rhs_number_signature(text, context_text)
        probe_decode_tokens += len(token_ids)
        rollouts.append(
            {
                "rollout_idx": len(rollouts),
                "text": text,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "finish_reason": str(getattr(completion, "finish_reason", "") or ""),
                "mean_logprob": mean_logprob,
                "signature_type": signature["signature_type"],
                "signature_value": signature["signature_value"],
                "signature": signature["signature"],
                "operation_signature_type": operation_signature["signature_type"],
                "operation_signature_value": operation_signature["signature_value"],
                "operation_signature": operation_signature["signature"],
                "number_signature_type": number_signature["signature_type"],
                "number_signature_value": number_signature["signature_value"],
                "number_signature": number_signature["signature"],
                "novel_number_signature_type": novel_number_signature["signature_type"],
                "novel_number_signature_value": novel_number_signature["signature_value"],
                "novel_number_signature": novel_number_signature["signature"],
                "rhs_number_signature_type": rhs_number_signature["signature_type"],
                "rhs_number_signature_value": rhs_number_signature["signature_value"],
                "rhs_number_signature": rhs_number_signature["signature"],
            }
        )

    metrics = rollout_disagreement_metrics(
        [row["text"] for row in rollouts],
        [row["mean_logprob"] for row in rollouts],
        context_text=context_text,
    )
    row = {
        "assistant_prefix_text": state.assistant_prefix_text,
        "prefix_char_len": len(state.assistant_prefix_text),
        "prefix_token_len": prompt_tokens,
        "rollouts": rollouts,
        **metrics,
    }
    cost = {
        "probe_decode_tokens": probe_decode_tokens,
        "probe_prefill_tokens": prompt_tokens,
        "probe_generate_calls": 1,
        "probe_wall_time": probe_wall_time,
    }
    return row, cost


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _max(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def run_sampling_disagreement(
    problem_text: str,
    slm,
    config: BPAConfig,
    *,
    probe_k: int = 4,
    probe_temperature: float = 0.7,
    probe_max_tokens: int = 32,
    probe_stop: str = "\n\n",
) -> tuple[BPAResult, list[dict[str, Any]], dict[str, float | int]]:
    if probe_k < 3:
        raise ValueError("probe_k must be >= 3 for this diagnostic.")

    state = GenerationState(problem_text=problem_text, generation_protocol="slm_sampling_disagreement")
    rep = RepetitionState()
    start_time = time.time()
    step_logs: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    probe_cost = {
        "probe_decode_tokens": 0,
        "probe_prefill_tokens": 0,
        "probe_generate_calls": 0,
        "probe_wall_time": 0.0,
    }
    boundary_idx = 0

    while state.phase != Phase.DONE:
        if state.slm_decode_tokens >= config.max_total_tokens:
            state.trace.append(TraceEvent(state.step_count, "total_token_budget_exhausted", {}))
            state.stop_reason = "total_token_budget"
            break

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

        decode_tokens_before = state.slm_decode_tokens
        try:
            step_text, finish = _slm_generate_step(state, slm, config)
        except ContextBudgetExceeded as exc:
            state.phase = Phase.DONE
            state.stop_reason = "context_budget"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
            break
        step_text_normalized = ensure_step_terminator(step_text, finish)
        generated_step_tokens = state.slm_decode_tokens - decode_tokens_before
        state.assistant_prefix_text += step_text_normalized
        state.step_count += 1

        probe_row["main_step_text"] = step_text_normalized
        probe_row["main_step_finish_reason"] = finish
        probe_rows.append(probe_row)

        log_row = {
            "step_idx": state.step_count - 1,
            "decision": "slm_sampling_probe",
            "generation_source": "slm",
            "finish_reason": finish,
            "step_text": step_text_normalized,
            "generated_step_tokens": generated_step_tokens,
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
    result = BPAResult(
        answer=extract_answer(state.assistant_prefix_text),
        state=state,
        total_wall_time=time.time() - start_time,
    )
    return result, probe_rows, probe_cost


def build_problem_summary(
    problem: EvalProblem,
    result: BPAResult,
    probe_rows: list[dict[str, Any]],
    probe_cost: dict[str, float | int],
    dataset: str,
) -> dict[str, Any]:
    correct = None
    if problem.gold_answer is not None:
        correct = benchmark_eval_match(result.answer, problem.gold_answer, dataset)
        result.correct = correct
    real_probe_rows = [row for row in probe_rows if int(row.get("boundary_idx", -1)) >= 0 and not row.get("is_initial_probe")]
    num_initial_probes = sum(1 for row in probe_rows if row.get("is_initial_probe") or int(row.get("boundary_idx", 0)) < 0)

    return {
        "problem_id": problem.problem_id,
        "question_id": problem.question_id,
        "gold_answer": problem.gold_answer,
        "final_answer": result.answer,
        "correct": correct,
        "num_boundaries": len(real_probe_rows),
        "num_initial_probes": num_initial_probes,
        "mean_structured_disagreement": _mean([row.get("structured_disagreement") for row in real_probe_rows]),
        "max_structured_disagreement": _max([row.get("structured_disagreement") for row in real_probe_rows]),
        "mean_operation_vote_disagreement": _mean([row.get("operation_vote_disagreement") for row in real_probe_rows]),
        "max_operation_vote_disagreement": _max([row.get("operation_vote_disagreement") for row in real_probe_rows]),
        "mean_number_vote_disagreement": _mean([row.get("number_vote_disagreement") for row in real_probe_rows]),
        "max_number_vote_disagreement": _max([row.get("number_vote_disagreement") for row in real_probe_rows]),
        "mean_novel_number_vote_disagreement": _mean([row.get("novel_number_vote_disagreement") for row in real_probe_rows]),
        "max_novel_number_vote_disagreement": _max([row.get("novel_number_vote_disagreement") for row in real_probe_rows]),
        "mean_rhs_number_vote_disagreement": _mean([row.get("rhs_number_vote_disagreement") for row in real_probe_rows]),
        "max_rhs_number_vote_disagreement": _max([row.get("rhs_number_vote_disagreement") for row in real_probe_rows]),
        "mean_self_bleu_disagreement": _mean([row.get("self_bleu_disagreement") for row in real_probe_rows]),
        "max_self_bleu_disagreement": _max([row.get("self_bleu_disagreement") for row in real_probe_rows]),
        "mean_char_jaccard_disagreement": _mean([row.get("char_jaccard_disagreement") for row in real_probe_rows]),
        "max_char_jaccard_disagreement": _max([row.get("char_jaccard_disagreement") for row in real_probe_rows]),
        "mean_score_variance": _mean([row.get("score_variance") for row in real_probe_rows]),
        "max_score_variance": _max([row.get("score_variance") for row in real_probe_rows]),
        "total_wall_time": result.total_wall_time,
        "slm_decode_tokens": result.state.slm_decode_tokens,
        "slm_prefill_tokens": result.state.slm_prefill_tokens,
        "slm_generate_calls": result.state.slm_generate_calls,
        **probe_cost,
    }


def enrich_probe_rows(
    problem: EvalProblem,
    probe_rows: list[dict[str, Any]],
    result: BPAResult,
    dataset: str,
) -> list[dict[str, Any]]:
    final_correct = None
    if problem.gold_answer is not None:
        final_correct = benchmark_eval_match(result.answer, problem.gold_answer, dataset)
    enriched = []
    for row in probe_rows:
        enriched.append(
            {
                "dataset": dataset,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                **row,
                "final_answer": result.answer,
                "gold_answer": problem.gold_answer,
                "final_correct": final_correct,
                "stop_reason": result.state.stop_reason,
            }
        )
    return enriched


def write_sampling_outputs(out_dir: Path, probe_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    probes_path = out_dir / "probes.jsonl"
    with probes_path.open("w", encoding="utf-8") as f:
        for row in probe_rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")

    summary_path = out_dir / "problem_summary.csv"
    fieldnames = list(PROBLEM_SUMMARY_FIELDS)
    extra_fields = sorted({key for row in summary_rows for key in row} - set(fieldnames))
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(json_safe(summary_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure-SLM sampling disagreement diagnostics.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--probe-k", type=int, default=4)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--probe-max-tokens", type=int, default=32)
    parser.add_argument("--probe-stop", default="\n\n")
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    slm, _ = init_engines(config)

    all_probe_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for problem in tqdm(problems, desc=f"sampling_disagreement:{args.dataset}"):
        result, probe_rows, probe_cost = run_sampling_disagreement(
            problem.problem_text,
            slm,
            config,
            probe_k=args.probe_k,
            probe_temperature=args.probe_temperature,
            probe_max_tokens=args.probe_max_tokens,
            probe_stop=args.probe_stop,
        )
        summary_rows.append(build_problem_summary(problem, result, probe_rows, probe_cost, args.dataset))
        all_probe_rows.extend(enrich_probe_rows(problem, probe_rows, result, args.dataset))
        if config.reset_prefix_cache_after_problem:
            slm.clear_runtime_cache()

    out_dir = Path(config.output_dir) / "diagnostics" / "sampling_disagreement" / args.dataset
    write_sampling_outputs(out_dir, all_probe_rows, summary_rows)
    print(f"Wrote {out_dir / 'probes.jsonl'}")
    print(f"Wrote {out_dir / 'problem_summary.csv'}")


if __name__ == "__main__":
    main()
