from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.context_budget import ContextBudgetExceeded, generation_budget_for_rendered
from bpa.engines import init_engines
from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import load_eval_dataset
from bpa.eval.exp_sampling_disagreement import _sample_probe_rollouts
from bpa.eval.sampling_disagreement import ROUTING_EVIDENCE_CHANNEL_PRIORITY
from bpa.pipeline import (
    THINKING_RECOVERY_STOP_REASONS,
    _generate_step_with_engine,
    _final_answer_stop_reason,
    _append_forced_close_think,
    _can_force_final_answer_from_thinking,
    _in_final_answer_phase,
    _is_eos_finish,
    _llm_generate_step,
    _post_stop_lookahead,
    _slm_generate_final_answer,
    _slm_generate_step,
)
from bpa.render import render_for_continuation
from bpa.safety import ensure_step_terminator, extract_answer_from_steps, update_strict_step_repetition
from bpa.state import GenerationState, Phase, RepetitionState, TraceEvent
from bpa.trace import BPAResult, json_safe, result_summary, write_json, write_jsonl


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
    "num_llm_colon_continuation_steps",
    "num_slm_steps",
    "num_probe_reused_steps",
    "total_wall_time",
    "problem_wall_time",
    "slm_decode_tokens",
    "slm_prefill_tokens",
    "llm_decode_tokens",
    "llm_prefill_tokens",
    "main_decode_tokens",
    "total_decode_tokens_including_probe",
    "slm_generate_calls",
    "llm_generate_calls",
    "probe_decode_tokens",
    "probe_prefill_tokens",
    "probe_generate_calls",
    "probe_wall_time",
    "text_bridge_decode_tokens",
    "text_bridge_prefill_tokens",
    "text_bridge_generate_calls",
    "text_bridge_wall_time",
    "text_bridge_verify_decode_tokens",
    "text_bridge_verify_prefill_tokens",
    "text_bridge_verify_generate_calls",
    "text_bridge_verify_wall_time",
    "num_text_bridge_triggered",
    "num_text_bridge_accepted",
]

def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _is_evaluated(value: Any) -> bool:
    return value is not None and str(value) != ""


def _is_correct(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


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


def _empty_prefix_consensus() -> dict[str, Any]:
    return {
        "prefix_anchor_idx": None,
        "prefix_anchor_mean_logprob": None,
        "prefix_consensus_channel": None,
        "prefix_consensus_value": None,
        "prefix_consensus_support_count": 0,
        "prefix_consensus_vote_fraction": None,
        "prefix_consensus_support_by_rollout": {},
        "prefix_consensus_group_counts": {},
        "prefix_consensus_stage": None,
        "stage1_case": "empty",
        "selected_rollout": None,
    }


def _stage1_groups(rollouts: list[dict[str, Any]]) -> tuple[str | None, dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    for channel in ROUTING_EVIDENCE_CHANNEL_PRIORITY:
        groups: dict[str, list[dict[str, Any]]] = {}
        none_rollouts: list[dict[str, Any]] = []
        for rollout in rollouts:
            value = _rollout_evidence_value(rollout, channel)
            if value is None:
                none_rollouts.append(rollout)
                continue
            groups.setdefault(value, []).append(rollout)
        if groups:
            return channel, groups, none_rollouts
    return None, {}, list(rollouts)


def _stage1_case(
    rollouts: list[dict[str, Any]],
    *,
    min_agreement_count: int,
) -> dict[str, Any]:
    channel, groups, none_rollouts = _stage1_groups(rollouts)
    if channel is None:
        return {
            "case": "all_none",
            "channel": None,
            "value": None,
            "group": [],
            "none_rollouts": none_rollouts,
            "group_counts": {},
        }

    value, group = max(
        groups.items(),
        key=lambda item: (
            len(item[1]),
            _mean_logprob_sort_value(_best_rollout(item[1])),
            item[0],
        ),
    )
    group_counts = {group_value: len(group_rows) for group_value, group_rows in sorted(groups.items())}
    if len(group) >= min_agreement_count:
        case = "accepted"
    elif len(groups) == 1:
        case = "partial"
    else:
        case = "conflict"
    return {
        "case": case,
        "channel": channel,
        "value": value,
        "group": group,
        "none_rollouts": none_rollouts,
        "group_counts": group_counts,
    }


def _zero_bridge_cost() -> dict[str, float | int]:
    return {
        "probe_decode_tokens": 0,
        "probe_prefill_tokens": 0,
        "probe_generate_calls": 0,
        "probe_wall_time": 0.0,
        "text_bridge_decode_tokens": 0,
        "text_bridge_prefill_tokens": 0,
        "text_bridge_generate_calls": 0,
        "text_bridge_wall_time": 0.0,
        "text_bridge_verify_decode_tokens": 0,
        "text_bridge_verify_prefill_tokens": 0,
        "text_bridge_verify_generate_calls": 0,
        "text_bridge_verify_wall_time": 0.0,
    }


def _add_probe_cost(total: dict[str, float | int], delta: dict[str, float | int]) -> None:
    for key, value in delta.items():
        total[key] = total.get(key, 0) + value


def _bridge_defaults() -> dict[str, Any]:
    return {
        "text_bridge_triggered": False,
        "text_bridge_accepted": False,
        "text_bridge_reason": None,
        "text_bridge_seed_rollout_idx": None,
        "text_bridge_seed_mean_logprob": None,
        "text_bridge_seed_finish_reason": None,
        "text_bridge_text": None,
        "text_bridge_token_count": None,
        "text_bridge_continued": False,
        "text_bridge_continuation_text": None,
        "text_bridge_verify_consensus_channel": None,
        "text_bridge_verify_consensus_value": None,
        "text_bridge_verify_consensus_support_count": 0,
        "text_bridge_verify_consensus_vote_fraction": None,
        "text_bridge_verify_consensus_group_counts": {},
        "text_bridge_verify_stage1_case": None,
        "text_bridge_verify_selected_rollout_idx": None,
        "text_bridge_verify_boundary": None,
    }


def _step_ends_with_colon(step_text: str, finish: str) -> bool:
    if _is_eos_finish(finish):
        return False
    return step_text.rstrip().endswith(":")


def _zero_probe_cost() -> dict[str, float | int]:
    cost = _zero_bridge_cost()
    return cost


def _complete_text_bridge_step(
    state: GenerationState,
    slm,
    config: BPAConfig,
    seed_rollout: dict[str, Any],
    *,
    bridge_max_tokens: int,
    probe_stop: str,
) -> tuple[str, str, int, bool, str, dict[str, float | int]]:
    tokenizer = slm.ensure_tokenizer()
    seed_text = str(seed_rollout.get("text") or "")
    seed_finish = str(seed_rollout.get("finish_reason") or "")
    seed_token_count = _probe_step_token_count(seed_rollout, slm)
    cost = _zero_bridge_cost()
    if seed_finish in {"stop", "eos"}:
        return seed_text, seed_finish, seed_token_count, False, "", cost

    remaining_bridge_tokens = bridge_max_tokens - seed_token_count
    if remaining_bridge_tokens <= 0:
        return seed_text, seed_finish or "length", seed_token_count, False, "", cost

    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text + seed_text, tokenizer)
    max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, slm, config, remaining_bridge_tokens)
    sampling = slm.sampling_params(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=[probe_stop],
        include_stop_str_in_output=True,
        logprobs=1,
        n=1,
    )
    generate_start = time.time()
    out = slm.generate(rendered, sampling)[0]
    wall_time = time.time() - generate_start
    completion = (getattr(out, "outputs", []) or [None])[0]
    continuation_text = getattr(completion, "text", "") if completion is not None else ""
    token_ids = list(getattr(completion, "token_ids", []) or []) if completion is not None else []
    finish = str(getattr(completion, "finish_reason", "") or "") if completion is not None else ""
    token_count = len(token_ids)
    cost["probe_decode_tokens"] += token_count
    cost["probe_prefill_tokens"] += prompt_tokens
    cost["probe_generate_calls"] += 1
    cost["probe_wall_time"] += wall_time
    cost["text_bridge_decode_tokens"] += token_count
    cost["text_bridge_prefill_tokens"] += prompt_tokens
    cost["text_bridge_generate_calls"] += 1
    cost["text_bridge_wall_time"] += wall_time
    return seed_text + (continuation_text or ""), finish or seed_finish, seed_token_count + token_count, True, continuation_text or "", cost


def _selected_prefix_consensus_rollout(
    probe_row: dict[str, Any],
    *,
    min_agreement_count: int,
) -> dict[str, Any]:
    if min_agreement_count < 1:
        raise ValueError("min_agreement_count must be >= 1")

    rollouts = list(probe_row.get("rollouts") or [])
    if not rollouts:
        return _empty_prefix_consensus()

    for idx, rollout in enumerate(rollouts):
        if "rollout_idx" not in rollout:
            rollout["rollout_idx"] = idx

    stage1 = _stage1_case(rollouts, min_agreement_count=min_agreement_count)
    best_channel = stage1["channel"]
    best_value = stage1["value"]
    best_group = list(stage1["group"])
    best_group_counts = dict(stage1["group_counts"])
    selected_rollout = _best_rollout(best_group) if stage1["case"] == "accepted" else None
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
        "prefix_consensus_stage": 1 if selected_rollout else None,
        "stage1_case": stage1["case"],
        "selected_rollout": selected_rollout,
    }


def _copy_consensus_fields(target: dict[str, Any], consensus: dict[str, Any]) -> None:
    for key in (
        "prefix_anchor_idx",
        "prefix_anchor_mean_logprob",
        "prefix_consensus_channel",
        "prefix_consensus_value",
        "prefix_consensus_support_count",
        "prefix_consensus_vote_fraction",
        "prefix_consensus_support_by_rollout",
        "prefix_consensus_group_counts",
        "prefix_consensus_stage",
    ):
        target[key] = consensus.get(key)


def _verify_text_bridge(
    state: GenerationState,
    slm,
    config: BPAConfig,
    probe_row: dict[str, Any],
    prefix_consensus: dict[str, Any],
    *,
    min_agreement_count: int,
    probe_temperature: float,
    probe_max_tokens: int,
    probe_stop: str,
    text_bridge_max_tokens: int,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    if prefix_consensus.get("selected_rollout") is not None:
        return prefix_consensus, _zero_bridge_cost()

    rollouts = list(probe_row.get("rollouts") or [])
    if prefix_consensus.get("stage1_case") != "all_none" or not rollouts:
        return prefix_consensus, _zero_bridge_cost()

    bridge_info = _bridge_defaults()
    bridge_info["text_bridge_triggered"] = True
    seed_rollout = _best_rollout(rollouts)
    bridge_info["text_bridge_seed_rollout_idx"] = seed_rollout.get("rollout_idx")
    bridge_info["text_bridge_seed_mean_logprob"] = seed_rollout.get("mean_logprob")
    bridge_info["text_bridge_seed_finish_reason"] = seed_rollout.get("finish_reason")

    bridge_text, bridge_finish, bridge_token_count, bridge_continued, bridge_continuation_text, cost = _complete_text_bridge_step(
        state,
        slm,
        config,
        seed_rollout,
        bridge_max_tokens=text_bridge_max_tokens,
        probe_stop=probe_stop,
    )
    bridge_text_normalized = ensure_step_terminator(bridge_text, bridge_finish)
    bridge_info.update(
        {
            "text_bridge_text": bridge_text_normalized,
            "text_bridge_token_count": bridge_token_count,
            "text_bridge_continued": bridge_continued,
            "text_bridge_continuation_text": bridge_continuation_text,
        }
    )
    if bridge_finish != "stop":
        bridge_info["text_bridge_reason"] = "bridge_not_closed" if bridge_finish != "eos" else "bridge_eos_without_evidence"
        prefix_consensus.update(bridge_info)
        return prefix_consensus, cost

    verifier_state = GenerationState(problem_text=state.problem_text, generation_protocol=state.generation_protocol)
    verifier_state.assistant_prefix_text = state.assistant_prefix_text + bridge_text_normalized
    verifier_row, verifier_cost = _sample_probe_rollouts(
        verifier_state,
        slm,
        config,
        probe_k=len(rollouts),
        probe_temperature=probe_temperature,
        probe_max_tokens=probe_max_tokens,
        probe_stop=probe_stop,
    )
    _add_probe_cost(cost, verifier_cost)
    cost["text_bridge_verify_decode_tokens"] += verifier_cost.get("probe_decode_tokens", 0)
    cost["text_bridge_verify_prefill_tokens"] += verifier_cost.get("probe_prefill_tokens", 0)
    cost["text_bridge_verify_generate_calls"] += verifier_cost.get("probe_generate_calls", 0)
    cost["text_bridge_verify_wall_time"] += verifier_cost.get("probe_wall_time", 0.0)

    verifier_consensus = _selected_prefix_consensus_rollout(
        verifier_row,
        min_agreement_count=min_agreement_count,
    )
    bridge_info["text_bridge_verify_boundary"] = verifier_row
    bridge_info["text_bridge_verify_consensus_channel"] = verifier_consensus["prefix_consensus_channel"]
    bridge_info["text_bridge_verify_consensus_value"] = verifier_consensus["prefix_consensus_value"]
    bridge_info["text_bridge_verify_consensus_support_count"] = verifier_consensus["prefix_consensus_support_count"]
    bridge_info["text_bridge_verify_consensus_vote_fraction"] = verifier_consensus["prefix_consensus_vote_fraction"]
    bridge_info["text_bridge_verify_consensus_group_counts"] = verifier_consensus["prefix_consensus_group_counts"]
    bridge_info["text_bridge_verify_stage1_case"] = verifier_consensus["stage1_case"]

    verifier_rollout = verifier_consensus.get("selected_rollout")
    if verifier_rollout is None:
        bridge_info["text_bridge_reason"] = "no_post_bridge_consensus"
        prefix_consensus.update(bridge_info)
        return prefix_consensus, cost

    bridge_info["text_bridge_accepted"] = True
    bridge_info["text_bridge_reason"] = "post_bridge_consensus"
    bridge_info["text_bridge_verify_selected_rollout_idx"] = verifier_rollout.get("rollout_idx")
    verifier_text = str(verifier_rollout.get("text") or "")
    combined_rollout = dict(verifier_rollout)
    combined_rollout.update(
        {
            "rollout_idx": verifier_rollout.get("rollout_idx"),
            "text": bridge_text_normalized + verifier_text,
            "finish_reason": verifier_rollout.get("finish_reason"),
            "token_count": bridge_token_count + _probe_step_token_count(verifier_rollout, slm),
            "mean_logprob": verifier_rollout.get("mean_logprob"),
            "text_bridge_prefix_text": bridge_text_normalized,
            "text_bridge_verifier_text": verifier_text,
        }
    )
    _copy_consensus_fields(prefix_consensus, verifier_consensus)
    prefix_consensus["prefix_consensus_stage"] = "text_bridge"
    prefix_consensus["prefix_anchor_idx"] = verifier_rollout.get("rollout_idx")
    prefix_consensus["prefix_anchor_mean_logprob"] = verifier_rollout.get("mean_logprob")
    prefix_consensus["selected_rollout"] = combined_rollout
    prefix_consensus.update(bridge_info)
    return prefix_consensus, cost


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
        return (
            prefix_text,
            prefix_finish,
            prefix_token_count,
            bool(selected_rollout.get("text_bridge_prefix_text")),
            prefix_text,
            str(selected_rollout.get("text_bridge_verifier_text") or ""),
        )

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
    text_bridge_max_tokens: int | None = None,
) -> tuple[BPAResult, list[dict[str, Any]], dict[str, float | int]]:
    if min_agreement_count > probe_k:
        raise ValueError("min_agreement_count cannot exceed probe_k")
    if text_bridge_max_tokens is None:
        text_bridge_max_tokens = max(probe_max_tokens, min(config.max_step_tokens, 128))
    if text_bridge_max_tokens < probe_max_tokens:
        raise ValueError("text_bridge_max_tokens must be >= probe_max_tokens")

    protocol = "evidence_consensus_routing"
    state = GenerationState(problem_text=problem_text, generation_protocol=protocol)
    rep = RepetitionState()
    start_time = time.time()
    step_logs: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    probe_cost = _zero_probe_cost()
    boundary_idx = 0
    force_next_llm = False

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
                state.trace.append(TraceEvent(state.step_count, "step_repetition_stop", {"trigger_reason": trigger}))
                if trigger in THINKING_RECOVERY_STOP_REASONS and _can_force_final_answer_from_thinking(state, config):
                    _append_forced_close_think(state)
                    continue
                state.phase = Phase.DONE
                state.stop_reason = trigger
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

            state.phase = Phase.DONE
            state.stop_reason = _final_answer_stop_reason(finish)
            continue

        if force_next_llm:
            force_next_llm = False
            decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
            try:
                step_text, finish = _llm_generate_step(state, llm, config)
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
                    "decision": "llm_colon_continuation",
                    "generation_source": "llm",
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
                state.trace.append(TraceEvent(state.step_count, "step_repetition_stop", {"trigger_reason": trigger}))
                if trigger in THINKING_RECOVERY_STOP_REASONS and _can_force_final_answer_from_thinking(state, config):
                    _append_forced_close_think(state)
                    continue
                state.phase = Phase.DONE
                state.stop_reason = trigger
                break
            if _is_eos_finish(finish):
                state.phase = Phase.DONE
                state.stop_reason = "eos"
            else:
                force_next_llm = _step_ends_with_colon(step_text_normalized, finish)
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
        _add_probe_cost(probe_cost, cost)
        prefix_consensus = _selected_prefix_consensus_rollout(
            probe_row,
            min_agreement_count=min_agreement_count,
        )
        try:
            prefix_consensus, bridge_cost = _verify_text_bridge(
                state,
                slm,
                config,
                probe_row,
                prefix_consensus,
                min_agreement_count=min_agreement_count,
                probe_temperature=probe_temperature,
                probe_max_tokens=probe_max_tokens,
                probe_stop=probe_stop,
                text_bridge_max_tokens=text_bridge_max_tokens,
            )
            _add_probe_cost(probe_cost, bridge_cost)
        except ContextBudgetExceeded as exc:
            state.phase = Phase.DONE
            state.stop_reason = "context_budget"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
            break
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
        probe_row["prefix_consensus_stage"] = prefix_consensus["prefix_consensus_stage"]
        probe_row["stage1_case"] = prefix_consensus["stage1_case"]
        for key, value in _bridge_defaults().items():
            probe_row[key] = prefix_consensus.get(key, value)
        probe_row["routed_to_llm"] = route_to_llm
        probe_row["reused_probe_rollout"] = selected_rollout is not None
        probe_row["selected_rollout_idx"] = selected_rollout.get("rollout_idx") if probe_row["reused_probe_rollout"] else None
        probe_row["selected_rollout_mean_logprob"] = (
            selected_rollout.get("mean_logprob") if probe_row["reused_probe_rollout"] else None
        )
        probe_row["selected_rollout_finish_reason"] = (
            selected_rollout.get("finish_reason")
            if probe_row["reused_probe_rollout"]
            else None
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
            force_next_llm = _step_ends_with_colon(step_text_normalized, finish)
        elif probe_row["reused_probe_rollout"]:
            decision = "slm_text_bridge_reuse" if probe_row.get("text_bridge_accepted") else "slm_probe_reuse"
            force_next_llm = False
        else:
            decision = "slm_direct"
            force_next_llm = False
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
            state.trace.append(TraceEvent(state.step_count, "step_repetition_stop", {"trigger_reason": trigger}))
            if trigger in THINKING_RECOVERY_STOP_REASONS and _can_force_final_answer_from_thinking(state, config):
                _append_forced_close_think(state)
                continue
            state.phase = Phase.DONE
            state.stop_reason = trigger
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


def build_problem_summary(
    *,
    dataset: str,
    problem,
    result: BPAResult,
    boundary_rows: list[dict[str, Any]],
    probe_cost: dict[str, float | int],
    min_agreement_count: int,
    config: BPAConfig,
    problem_wall_time: float | None = None,
) -> dict[str, Any]:
    correct = None
    if problem.gold_answer is not None:
        correct = benchmark_eval_match(result.answer, problem.gold_answer, dataset)
        result.correct = correct
    main_decode_tokens = result.state.slm_decode_tokens + result.state.llm_decode_tokens
    probe_decode_tokens = int(probe_cost.get("probe_decode_tokens") or 0)
    real_boundary_rows = list(boundary_rows)
    step_rows = _step_rows(result)
    return {
        "dataset": dataset,
        "problem_id": problem.problem_id,
        "question_id": problem.question_id,
        "gold_answer": problem.gold_answer,
        "final_answer": result.answer,
        "correct": correct,
        "min_agreement_count": min_agreement_count,
        "post_stop_lookahead_tokens": config.post_stop_lookahead_tokens,
        "num_boundaries": len(real_boundary_rows),
        "num_llm_routed_steps": sum(1 for row in real_boundary_rows if row.get("routed_to_llm")),
        "num_llm_colon_continuation_steps": sum(1 for row in step_rows if row.get("decision") == "llm_colon_continuation"),
        "num_slm_steps": sum(1 for row in real_boundary_rows if not row.get("routed_to_llm")),
        "num_probe_reused_steps": sum(1 for row in real_boundary_rows if row.get("reused_probe_rollout")),
        "num_text_bridge_triggered": sum(1 for row in real_boundary_rows if row.get("text_bridge_triggered")),
        "num_text_bridge_accepted": sum(1 for row in real_boundary_rows if row.get("text_bridge_accepted")),
        "total_wall_time": result.total_wall_time,
        "problem_wall_time": problem_wall_time if problem_wall_time is not None else result.total_wall_time,
        "slm_decode_tokens": result.state.slm_decode_tokens,
        "slm_prefill_tokens": result.state.slm_prefill_tokens,
        "llm_decode_tokens": result.state.llm_decode_tokens,
        "llm_prefill_tokens": result.state.llm_prefill_tokens,
        "main_decode_tokens": main_decode_tokens,
        "total_decode_tokens_including_probe": main_decode_tokens + probe_decode_tokens,
        "slm_generate_calls": result.state.slm_generate_calls,
        "llm_generate_calls": result.state.llm_full_calls,
        **probe_cost,
    }


def _summary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if _is_evaluated(row.get("correct"))]
    correct = [row for row in evaluated if _is_correct(row.get("correct"))]
    return {
        "num_problems": len(rows),
        "num_evaluated": len(evaluated),
        "num_correct": len(correct),
        "accuracy": len(correct) / len(evaluated) if evaluated else None,
        "avg_total_wall_time": _mean([row.get("total_wall_time") for row in rows]),
        "avg_problem_wall_time": _mean([row.get("problem_wall_time") for row in rows]),
        "avg_num_llm_routed_steps": _mean([row.get("num_llm_routed_steps") for row in rows]),
        "avg_num_llm_colon_continuation_steps": _mean([row.get("num_llm_colon_continuation_steps") for row in rows]),
        "avg_num_probe_reused_steps": _mean([row.get("num_probe_reused_steps") for row in rows]),
        "avg_num_text_bridge_triggered": _mean([row.get("num_text_bridge_triggered") for row in rows]),
        "avg_num_text_bridge_accepted": _mean([row.get("num_text_bridge_accepted") for row in rows]),
        "avg_main_decode_tokens": _mean([row.get("main_decode_tokens") for row in rows]),
        "avg_total_decode_tokens_including_probe": _mean(
            [row.get("total_decode_tokens_including_probe") for row in rows]
        ),
        "total_llm_decode_tokens": sum(float(row.get("llm_decode_tokens") or 0.0) for row in rows),
        "total_llm_prefill_tokens": sum(float(row.get("llm_prefill_tokens") or 0.0) for row in rows),
        "total_slm_decode_tokens": sum(float(row.get("slm_decode_tokens") or 0.0) for row in rows),
        "total_probe_decode_tokens": sum(float(row.get("probe_decode_tokens") or 0.0) for row in rows),
        "total_text_bridge_decode_tokens": sum(float(row.get("text_bridge_decode_tokens") or 0.0) for row in rows),
        "total_text_bridge_verify_decode_tokens": sum(float(row.get("text_bridge_verify_decode_tokens") or 0.0) for row in rows),
        "total_main_decode_tokens": sum(float(row.get("main_decode_tokens") or 0.0) for row in rows),
        "total_decode_tokens_including_probe": sum(
            float(row.get("total_decode_tokens_including_probe") or 0.0) for row in rows
        ),
    }


def _problem_root(out_dir: Path, problem_id: Any) -> Path:
    return out_dir / str(problem_id)


def _problem_output_paths(out_dir: Path, problem_id: Any) -> list[Path]:
    root = _problem_root(out_dir, problem_id)
    stem = str(problem_id)
    return [
        root / f"{stem}.problem.json",
        root / f"{stem}.steps.jsonl",
        root / f"{stem}.boundaries.jsonl",
        root / f"{stem}.trace.json",
    ]


def has_complete_problem_outputs(out_dir: Path, problem_id: Any) -> bool:
    return all(path.exists() for path in _problem_output_paths(out_dir, problem_id))


def load_summary_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row["problem_id"]): row for row in csv.DictReader(f) if row.get("problem_id") not in (None, "")}


def _step_rows(result: BPAResult) -> list[dict[str, Any]]:
    for event in result.state.trace:
        if event.event == "step_logs":
            return list(event.data.get("steps", []))
    return []


def _compact_boundary_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _compact_boundary_value(v) for k, v in value.items() if k not in {"assistant_prefix_text", "token_ids"}}
    if isinstance(value, list):
        return [_compact_boundary_value(item) for item in value]
    return value


def compact_boundary_row(row: dict[str, Any]) -> dict[str, Any]:
    return _compact_boundary_value(row)


def write_problem_outputs(
    out_dir: Path,
    *,
    dataset: str,
    problem,
    result: BPAResult,
    boundary_rows: list[dict[str, Any]],
    probe_cost: dict[str, float | int],
    summary_row: dict[str, Any],
) -> None:
    root = _problem_root(out_dir, problem.problem_id)
    stem = str(problem.problem_id)
    write_json(
        root / f"{stem}.problem.json",
        {
            "raw": problem.raw,
            "summary": summary_row,
            "probe_cost": probe_cost,
            "result": result_summary(result),
        },
    )
    write_jsonl(
        root / f"{stem}.steps.jsonl",
        [
            {"dataset": dataset, "problem_id": problem.problem_id, "question_id": problem.question_id, **row}
            for row in _step_rows(result)
        ],
    )
    write_jsonl(
        root / f"{stem}.boundaries.jsonl",
        [
            {
                "dataset": dataset,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                **compact_boundary_row(row),
            }
            for row in boundary_rows
        ],
    )
    write_json(root / f"{stem}.trace.json", result.state.trace)


def existing_problem_summary(out_dir: Path, problem, dataset: str, existing_summary_row: dict[str, Any] | None = None) -> dict[str, Any] | None:
    problem_json = _problem_root(out_dir, problem.problem_id) / f"{problem.problem_id}.problem.json"
    if not problem_json.exists():
        return existing_summary_row
    with problem_json.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    row = dict(saved.get("summary") or existing_summary_row or {})
    if not row:
        result = saved.get("result") or {}
        row = {
            "final_answer": result.get("answer"),
            "total_wall_time": result.get("total_wall_time"),
            "slm_decode_tokens": result.get("slm_decode_tokens"),
            "slm_prefill_tokens": result.get("slm_prefill_tokens"),
            "llm_decode_tokens": result.get("llm_decode_tokens"),
            "llm_prefill_tokens": result.get("llm_prefill_tokens"),
            "slm_generate_calls": result.get("slm_generate_calls"),
            "llm_generate_calls": result.get("llm_full_calls") or result.get("llm_generate_calls"),
        }
    row.update(
        {
            "dataset": dataset,
            "problem_id": problem.problem_id,
            "question_id": problem.question_id,
            "gold_answer": problem.gold_answer,
        }
    )
    if problem.gold_answer is not None:
        row["correct"] = benchmark_eval_match(row.get("final_answer"), problem.gold_answer, dataset)
    return row


def write_summary_files(summary_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SUMMARY_FIELDS)
    extra_fields = sorted({key for row in summary_rows for key in row} - set(fieldnames))
    tmp_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with tmp_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(json_safe(summary_rows))
    os.replace(tmp_summary, summary_path)

    metrics_path = summary_path.parent / "summary_metrics.json"
    tmp_metrics = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with tmp_metrics.open("w", encoding="utf-8") as f:
        json.dump(json_safe(_summary_metrics(summary_rows)), f, ensure_ascii=False, indent=2)
    os.replace(tmp_metrics, metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-consensus routing.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument("--min-agreement-count", type=int, default=3)
    parser.add_argument("--probe-k", type=int, default=4)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--probe-max-tokens", type=int, default=32)
    parser.add_argument("--text-bridge-max-tokens", type=int, default=None)
    parser.add_argument("--post-stop-lookahead-tokens", type=int, default=None)
    parser.add_argument("--output-name", default=None, help="Optional diagnostics output folder name under disagreement_routing/.")
    parser.add_argument("--resume", action="store_true", help="Skip problems that already have complete per-problem outputs.")
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    if args.post_stop_lookahead_tokens is not None:
        config = config.with_updates(post_stop_lookahead_tokens=args.post_stop_lookahead_tokens)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    slm, llm = init_engines(config)
    out_dir = Path(config.output_dir) / "diagnostics" / "disagreement_routing" / (args.output_name or args.dataset)
    summary_path = out_dir / "summary.csv"
    existing_summary_rows = load_summary_rows(summary_path) if args.resume else {}
    rows_by_problem_id: dict[str, dict[str, Any]] = {}
    skipped = 0
    if args.resume:
        for problem in problems:
            problem_id = str(problem.problem_id)
            if has_complete_problem_outputs(out_dir, problem.problem_id):
                row = existing_problem_summary(out_dir, problem, args.dataset, existing_summary_rows.get(problem_id))
                if row is not None:
                    rows_by_problem_id[problem_id] = row
                    skipped += 1

    def ordered_rows() -> list[dict[str, Any]]:
        return [
            rows_by_problem_id[str(problem.problem_id)]
            for problem in problems
            if str(problem.problem_id) in rows_by_problem_id
        ]

    for problem in tqdm(problems, desc=f"disagreement_routing:{args.dataset}"):
        problem_id = str(problem.problem_id)
        if args.resume and problem_id in rows_by_problem_id:
            continue
        problem_start = time.time()
        result, boundary_rows, probe_cost = run_disagreement_routing(
            problem.problem_text,
            slm,
            llm,
            config,
            min_agreement_count=args.min_agreement_count,
            probe_k=args.probe_k,
            probe_temperature=args.probe_temperature,
            probe_max_tokens=args.probe_max_tokens,
            text_bridge_max_tokens=args.text_bridge_max_tokens,
        )
        summary = build_problem_summary(
            dataset=args.dataset,
            problem=problem,
            result=result,
            boundary_rows=boundary_rows,
            probe_cost=probe_cost,
            min_agreement_count=args.min_agreement_count,
            config=config,
            problem_wall_time=time.time() - problem_start,
        )
        write_problem_outputs(
            out_dir,
            dataset=args.dataset,
            problem=problem,
            result=result,
            boundary_rows=boundary_rows,
            probe_cost=probe_cost,
            summary_row=summary,
        )
        if config.reset_prefix_cache_after_problem:
            slm.clear_runtime_cache()
            llm.clear_runtime_cache()
        rows_by_problem_id[problem_id] = summary
        write_summary_files(summary_path, ordered_rows())

    write_summary_files(summary_path, ordered_rows())
    if args.resume:
        print(f"Skipped {skipped} completed problem(s).")
    print(f"Wrote {summary_path}")
    print(f"Wrote {summary_path.parent / 'summary_metrics.json'}")
    print(f"Wrote per-problem outputs under {out_dir}")


if __name__ == "__main__":
    main()
