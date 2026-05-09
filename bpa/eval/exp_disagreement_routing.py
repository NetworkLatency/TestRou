from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.context_budget import ContextBudgetExceeded
from bpa.engines import init_engines
from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import load_eval_dataset
from bpa.eval.exp_sampling_disagreement import _sample_probe_rollouts
from bpa.eval.sampling_disagreement import EVIDENCE_CHANNEL_PRIORITY
from bpa.pipeline import (
    _generate_step_with_engine,
    _in_final_answer_phase,
    _is_eos_finish,
    _llm_generate_step,
    _post_stop_lookahead,
    _slm_generate_final_answer,
    _slm_generate_step,
)
from bpa.safety import ensure_step_terminator, extract_answer_from_steps, update_strict_step_repetition
from bpa.state import GenerationState, Phase, RepetitionState, TraceEvent
from bpa.trace import BPAResult, json_safe


SUMMARY_FIELDS = [
    "dataset",
    "problem_id",
    "question_id",
    "gold_answer",
    "final_answer",
    "correct",
    "min_agreement_count",
    "post_stop_lookahead_tokens",
    "num_boundaries",
    "num_llm_routed_steps",
    "num_slm_steps",
    "num_probe_reused_steps",
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

def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _mean_logprob_sort_value(row: dict[str, Any]) -> float:
    value = row.get("mean_logprob")
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _rollout_idx_sort_value(row: dict[str, Any]) -> int:
    try:
        return int(row.get("rollout_idx") or 0)
    except (TypeError, ValueError):
        return 0


def _rollout_evidence_value(rollout: dict[str, Any], channel: str) -> str | None:
    value = rollout.get(f"evidence_{channel}")
    if value is None:
        evidence = rollout.get("step_evidence")
        if isinstance(evidence, dict):
            value = evidence.get(channel)
    if value in (None, ""):
        return None
    return str(value)


def _best_rollout(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rollouts,
        key=lambda rollout: (
            _mean_logprob_sort_value(rollout),
            -_rollout_idx_sort_value(rollout),
        ),
    )


def _selected_prefix_consensus_rollout(
    probe_row: dict[str, Any],
    *,
    min_agreement_count: int,
) -> dict[str, Any]:
    if min_agreement_count < 1:
        raise ValueError("min_agreement_count must be >= 1")

    rollouts = list(probe_row.get("rollouts") or [])
    if not rollouts:
        return {
            "prefix_anchor_idx": None,
            "prefix_anchor_mean_logprob": None,
            "prefix_consensus_channel": None,
            "prefix_consensus_value": None,
            "prefix_consensus_support_count": 0,
            "prefix_consensus_vote_fraction": None,
            "prefix_consensus_support_by_rollout": {},
            "prefix_consensus_group_counts": {},
            "selected_rollout": None,
        }

    for idx, rollout in enumerate(rollouts):
        if "rollout_idx" not in rollout:
            rollout["rollout_idx"] = idx

    best_channel = None
    best_value = None
    best_group: list[dict[str, Any]] = []
    best_group_counts: dict[str, int] = {}
    for channel in EVIDENCE_CHANNEL_PRIORITY:
        groups: dict[str, list[dict[str, Any]]] = {}
        for rollout in rollouts:
            value = _rollout_evidence_value(rollout, channel)
            if value is None:
                continue
            groups.setdefault(value, []).append(rollout)
        if not groups:
            continue
        channel_value, channel_group = max(
            groups.items(),
            key=lambda item: (
                len(item[1]),
                _mean_logprob_sort_value(_best_rollout(item[1])),
                item[0],
            ),
        )
        if len(channel_group) >= min_agreement_count:
            best_channel = channel
            best_value = channel_value
            best_group = channel_group
            best_group_counts = {value: len(group) for value, group in sorted(groups.items())}
            break
        if not best_group:
            best_channel = channel
            best_value = channel_value
            best_group = channel_group
            best_group_counts = {value: len(group) for value, group in sorted(groups.items())}

    selected_rollout = _best_rollout(best_group) if len(best_group) >= min_agreement_count else None
    support_by_rollout: dict[str, bool] = {}
    for rollout in rollouts:
        rollout_idx = _rollout_idx_sort_value(rollout)
        is_support = selected_rollout is not None and rollout in best_group
        support_by_rollout[str(rollout_idx)] = is_support
    support_count = len(best_group) if best_group else 0

    return {
        "prefix_anchor_idx": selected_rollout.get("rollout_idx") if selected_rollout else None,
        "prefix_anchor_mean_logprob": selected_rollout.get("mean_logprob") if selected_rollout else None,
        "prefix_consensus_channel": best_channel,
        "prefix_consensus_value": best_value,
        "prefix_consensus_support_count": support_count,
        "prefix_consensus_vote_fraction": support_count / len(rollouts) if rollouts else None,
        "prefix_consensus_support_by_rollout": support_by_rollout,
        "prefix_consensus_group_counts": best_group_counts,
        "selected_rollout": selected_rollout,
    }


def _probe_step_token_count(rollout: dict[str, Any], slm) -> int:
    token_count = rollout.get("token_count")
    try:
        count = int(token_count)
    except (TypeError, ValueError):
        count = 0
    if count > 0:
        return count
    text = str(rollout.get("text") or "")
    return len(slm.encode(text)) if text else 0


def _selected_probe_prefix_step(
    state: GenerationState,
    slm,
    config: BPAConfig,
    selected_rollout: dict[str, Any],
) -> tuple[str, str, int, bool, str, str]:
    prefix_text = str(selected_rollout.get("text") or "")
    prefix_finish = str(selected_rollout.get("finish_reason") or "")
    prefix_token_count = _probe_step_token_count(selected_rollout, slm)

    if prefix_finish in {"stop", "eos"}:
        state.slm_decode_tokens += prefix_token_count
        if prefix_finish == "stop":
            lookahead_text, lookahead_finish = _post_stop_lookahead(
                state,
                slm,
                config,
                account="slm",
                step_text=prefix_text,
            )
            if lookahead_text or lookahead_finish == "eos":
                return (
                    prefix_text + lookahead_text,
                    lookahead_finish,
                    prefix_token_count,
                    False,
                    prefix_text,
                    "",
                )
        return prefix_text, prefix_finish, prefix_token_count, False, prefix_text, ""

    current_decode_tokens = state.slm_decode_tokens + state.llm_decode_tokens
    remaining_step_tokens = config.max_step_tokens - prefix_token_count
    remaining_total_tokens = config.max_total_tokens - current_decode_tokens - prefix_token_count
    if remaining_step_tokens <= 0 or remaining_total_tokens <= 0:
        state.slm_decode_tokens += prefix_token_count
        return prefix_text, prefix_finish or "length", prefix_token_count, False, prefix_text, ""

    continuation_budget = min(remaining_step_tokens, remaining_total_tokens)
    continuation_text, finish = _generate_step_with_engine(
        state,
        slm,
        config,
        account="slm",
        prefix_extension=prefix_text,
        step_token_budget=continuation_budget,
        total_token_offset=prefix_token_count,
    )
    state.slm_decode_tokens += prefix_token_count
    return (
        prefix_text + continuation_text,
        finish,
        prefix_token_count,
        True,
        prefix_text,
        continuation_text,
    )


def run_disagreement_routing(
    problem_text: str,
    slm,
    llm,
    config: BPAConfig,
    *,
    min_agreement_count: int = 3,
    probe_k: int = 4,
    probe_temperature: float = 0.7,
    probe_max_tokens: int = 32,
    probe_stop: str = "\n\n",
) -> tuple[BPAResult, list[dict[str, Any]], dict[str, float | int]]:
    if min_agreement_count > probe_k:
        raise ValueError("min_agreement_count cannot exceed probe_k")

    protocol = "evidence_consensus_routing"
    state = GenerationState(problem_text=problem_text, generation_protocol=protocol)
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

        if not state.assistant_prefix_text:
            decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
            try:
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

            step_logs.append(
                {
                    "step_idx": state.step_count - 1,
                    "decision": "slm_initial",
                    "generation_source": "slm",
                    "finish_reason": finish,
                    "step_text": step_text_normalized,
                    "generated_step_tokens": generated_step_tokens,
                    "reused_probe_rollout": False,
                    "continued_probe_rollout": False,
                    "selected_rollout_idx": None,
                }
            )

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
            continue

        if _in_final_answer_phase(state, config):
            decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
            try:
                step_text, finish = _slm_generate_final_answer(state, slm, config)
            except ContextBudgetExceeded as exc:
                state.phase = Phase.DONE
                state.stop_reason = "context_budget"
                state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
                break
            step_text_normalized = ensure_step_terminator(step_text, finish)
            generated_step_tokens = state.slm_decode_tokens + state.llm_decode_tokens - decode_tokens_before
            state.assistant_prefix_text += step_text_normalized
            state.step_count += 1
            step_logs.append(
                {
                    "step_idx": state.step_count - 1,
                    "decision": "slm_final_answer",
                    "generation_source": "slm",
                    "finish_reason": finish,
                    "step_text": step_text_normalized,
                    "generated_step_tokens": generated_step_tokens,
                    "reused_probe_rollout": False,
                    "continued_probe_rollout": False,
                    "selected_rollout_idx": None,
                }
            )

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
            continue

        selected_rollout = None
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
        probe_row["boundary_idx"] = boundary_idx
        probe_row["target_step_idx"] = state.step_count
        probe_row["is_initial_probe"] = False
        boundary_idx += 1
        for key in probe_cost:
            probe_cost[key] += cost[key]
        prefix_consensus = _selected_prefix_consensus_rollout(
            probe_row,
            min_agreement_count=min_agreement_count,
        )
        selected_rollout = prefix_consensus["selected_rollout"]
        route_to_llm = selected_rollout is None

        decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
        selected_rollout_token_count = None
        continued_probe_rollout = False
        probe_prefix_text = None
        probe_continuation_text = None
        try:
            if route_to_llm:
                step_text, finish = _llm_generate_step(state, llm, config)
            elif selected_rollout is not None:
                (
                    step_text,
                    finish,
                    selected_rollout_token_count,
                    continued_probe_rollout,
                    probe_prefix_text,
                    probe_continuation_text,
                ) = _selected_probe_prefix_step(state, slm, config, selected_rollout)
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

        probe_row["min_agreement_count"] = min_agreement_count
        probe_row["prefix_anchor_idx"] = prefix_consensus["prefix_anchor_idx"]
        probe_row["prefix_anchor_mean_logprob"] = prefix_consensus["prefix_anchor_mean_logprob"]
        probe_row["prefix_consensus_channel"] = prefix_consensus["prefix_consensus_channel"]
        probe_row["prefix_consensus_value"] = prefix_consensus["prefix_consensus_value"]
        probe_row["prefix_consensus_support_count"] = prefix_consensus["prefix_consensus_support_count"]
        probe_row["prefix_consensus_vote_fraction"] = prefix_consensus["prefix_consensus_vote_fraction"]
        probe_row["prefix_consensus_support_by_rollout"] = prefix_consensus["prefix_consensus_support_by_rollout"]
        probe_row["prefix_consensus_group_counts"] = prefix_consensus["prefix_consensus_group_counts"]
        probe_row["routed_to_llm"] = route_to_llm
        probe_row["reused_probe_rollout"] = selected_rollout is not None
        probe_row["selected_rollout_idx"] = selected_rollout.get("rollout_idx") if probe_row["reused_probe_rollout"] else None
        probe_row["selected_rollout_mean_logprob"] = (
            selected_rollout.get("mean_logprob") if probe_row["reused_probe_rollout"] else None
        )
        probe_row["selected_rollout_finish_reason"] = (
            selected_rollout.get("finish_reason") if probe_row["reused_probe_rollout"] else None
        )
        probe_row["selected_rollout_token_count"] = selected_rollout_token_count if probe_row["reused_probe_rollout"] else None
        probe_row["continued_probe_rollout"] = continued_probe_rollout
        probe_row["probe_prefix_text"] = probe_prefix_text if probe_row["reused_probe_rollout"] else None
        probe_row["probe_continuation_text"] = probe_continuation_text if continued_probe_rollout else None
        probe_row["main_step_text"] = step_text_normalized
        probe_row["main_step_finish_reason"] = finish
        boundary_rows.append(probe_row)

        if route_to_llm:
            decision = "llm_consensus_fallback"
        elif probe_row["reused_probe_rollout"]:
            decision = "slm_probe_reuse"
        else:
            decision = "slm_direct"
        log_row = {
            "step_idx": state.step_count - 1,
            "decision": decision,
            "generation_source": "llm" if route_to_llm else "slm",
            "finish_reason": finish,
            "step_text": step_text_normalized,
            "generated_step_tokens": generated_step_tokens,
            "reused_probe_rollout": probe_row["reused_probe_rollout"],
            "continued_probe_rollout": continued_probe_rollout,
            "selected_rollout_idx": probe_row["selected_rollout_idx"],
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
        answer=extract_answer_from_steps(step_logs, state.assistant_prefix_text),
        state=state,
        total_wall_time=time.time() - start_time,
    )
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
        "avg_num_probe_reused_steps": _mean([row.get("num_probe_reused_steps") for row in rows]),
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
    parser = argparse.ArgumentParser(description="Run evidence-consensus routing.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument("--min-agreement-count", type=int, default=3)
    parser.add_argument("--probe-k", type=int, default=4)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--probe-max-tokens", type=int, default=32)
    parser.add_argument("--post-stop-lookahead-tokens", type=int, default=None)
    parser.add_argument("--output-name", default=None, help="Optional diagnostics output folder name under disagreement_routing/.")
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    if args.post_stop_lookahead_tokens is not None:
        config = config.with_updates(post_stop_lookahead_tokens=args.post_stop_lookahead_tokens)
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
            min_agreement_count=args.min_agreement_count,
            probe_k=args.probe_k,
            probe_temperature=args.probe_temperature,
            probe_max_tokens=args.probe_max_tokens,
        )
        correct = None
        if problem.gold_answer is not None:
            correct = benchmark_eval_match(result.answer, problem.gold_answer, args.dataset)
            result.correct = correct
        real_boundary_rows = list(boundary_rows)
        summary_rows.append(
            {
                "dataset": args.dataset,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                "gold_answer": problem.gold_answer,
                "final_answer": result.answer,
                "correct": correct,
                "min_agreement_count": args.min_agreement_count,
                "post_stop_lookahead_tokens": config.post_stop_lookahead_tokens,
                "num_boundaries": len(real_boundary_rows),
                "num_llm_routed_steps": sum(1 for row in real_boundary_rows if row.get("routed_to_llm")),
                "num_slm_steps": sum(1 for row in real_boundary_rows if not row.get("routed_to_llm")),
                "num_probe_reused_steps": sum(1 for row in real_boundary_rows if row.get("reused_probe_rollout")),
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

    out_dir = Path(config.output_dir) / "diagnostics" / "disagreement_routing" / (args.output_name or args.dataset)
    write_outputs(out_dir, summary_rows, all_boundary_rows)
    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'routing_boundaries.jsonl'}")


if __name__ == "__main__":
    main()
