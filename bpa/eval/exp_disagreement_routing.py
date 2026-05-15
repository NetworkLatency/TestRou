from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from bpa.safety import ensure_step_terminator, extract_answer_from_steps, normalize_step_skeleton, update_strict_step_repetition
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
    "num_step_type_reused_steps",
    "num_dynamics_reused_steps",
    "num_pure_text_reused_steps",
    "step_type_routing_enabled",
    "step_type_min_agreement_count",
    "max_consecutive_step_type_exec",
    "dynamics_routing_enabled",
    "dynamics_min_agreement_count",
    "probe_logprobs_topk",
    "dynamics_flat_max_mean_surprisal",
    "dynamics_flat_max_var",
    "dynamics_flat_max_delta",
    "dynamics_unstable_min_delta",
    "dynamics_unstable_min_max_jump",
    "dynamics_unstable_min_spike_ratio",
    "dynamics_structured_min_var",
    "dynamics_structured_min_max_jump",
    "pure_text_slm_fallback_enabled",
    "branch_cut_recovery_enabled",
    "num_branch_cut_recoveries",
    "branch_cut_recovery_steps",
    "branch_cut_skeleton_window",
    "branch_cut_skeleton_threshold",
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
]


STEP_TYPE_EXECUTION = "execution"
STEP_TYPE_REFLECTION_TRANSITION = "reflection_transition"
STEP_TYPE_UNKNOWN = "unknown"

DYNAMICS_STRUCTURED = "structured"
DYNAMICS_FLAT_LOW = "flat_low"
DYNAMICS_UNSTABLE = "unstable"
DYNAMICS_UNKNOWN = "unknown"

_STEP_TYPE_REFLECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("wait", re.compile(r"^\s*(?:wait|hmm|uh|oh|oops)\b", re.IGNORECASE)),
    ("correction", re.compile(r"^\s*(?:actually|but|however|although|nevertheless)\b", re.IGNORECASE)),
    ("transition", re.compile(r"^\s*(?:alternatively|instead|another|otherwise|or)\b", re.IGNORECASE)),
    ("self_directed", re.compile(r"^\s*(?:let\s+me|let\s+us|let's|lets|try|maybe|perhaps|i\s+think|i\s+should|we\s+should)\b", re.IGNORECASE)),
    ("new_setup", re.compile(r"^\s*(?:let|suppose|assume|consider)\b", re.IGNORECASE)),
)

_STEP_TYPE_CONTAMINATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hidden_reflection", re.compile(r"\b(?:wait|hmm|actually|but|however|mistake|wrong|conflict(?:ing)?|not\s+sure|wonder|check|verify|recheck)\b", re.IGNORECASE)),
    ("hidden_self_directed", re.compile(r"\b(?:let\s+me|let\s+us|let['’]?s|lets|try|maybe|perhaps|might|i\s+(?:think|need|should|will)|we\s+(?:need|should))\b", re.IGNORECASE)),
    ("hidden_transition", re.compile(r"\b(?:alternative|alternatively|instead|another|different|suppose|assume|consider|case|cases|subcase|give\s+up|tedious)\b", re.IGNORECASE)),
)

_STEP_TYPE_EXECUTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("math_symbol", re.compile(r"^\s*(?:[=+\-*/^()\[\]{}]|\d|\\(?:frac|sqrt|cdot|times|begin))", re.IGNORECASE)),
    ("derivation_marker", re.compile(r"^\s*(?:so|therefore|thus|hence|then)\b", re.IGNORECASE)),
    ("mechanical_verb", re.compile(r"^\s*(?:substituting|plugging|computing|calculating|expanding|simplifying|factoring|multiplying|dividing|adding|subtracting|combining|solving|rearranging|reducing|evaluating)\b", re.IGNORECASE)),
    ("result_phrase", re.compile(r"^\s*(?:we\s+(?:get|obtain|have|find|see)|this\s+(?:gives|means|implies|becomes|is)|which\s+(?:gives|means|implies)|it\s+(?:follows|is))\b", re.IGNORECASE)),
)

BRANCH_CUT_RECOVERY_BRIDGE = (
    "\n\nThe previous branch is repeating the same reasoning pattern without progress. "
    "I will abandon that local branch and restart from the last stable point. "
    "I will not continue enumerating similar candidates. "
    "I will use a different representation, invariant, or counting argument, and each next step must produce a concrete new constraint or conclusion.\n\n"
)

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


def _step_type_head(text: Any, *, max_tokens: int = 4) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return ""
    pieces = re.findall(r"\\[A-Za-z]+|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+|[^\w\s]", normalized)
    return " ".join(pieces[:max_tokens])


def _step_type_scan_window(text: str, *, max_tokens: int = 16) -> str:
    pieces = re.findall(r"\\[A-Za-z]+|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+|[^\w\s]", text)
    head = " ".join(pieces[:max_tokens])
    head = head.replace("’", "'")
    head = re.sub(r"\s*'\s*", "'", head)
    return re.sub(r"\s+", " ", head).strip()


def _classify_step_type(text: Any) -> dict[str, str]:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    head = _step_type_head(normalized)
    scan_head = _step_type_scan_window(normalized)
    if not normalized:
        return {"step_type": STEP_TYPE_UNKNOWN, "step_type_signal": "empty", "step_type_head": head}

    for signal, pattern in _STEP_TYPE_REFLECTION_PATTERNS:
        if pattern.search(normalized):
            return {
                "step_type": STEP_TYPE_REFLECTION_TRANSITION,
                "step_type_signal": signal,
                "step_type_head": head,
            }
    for signal, pattern in _STEP_TYPE_CONTAMINATION_PATTERNS:
        if pattern.search(scan_head):
            return {
                "step_type": STEP_TYPE_REFLECTION_TRANSITION,
                "step_type_signal": signal,
                "step_type_head": head,
            }
    for signal, pattern in _STEP_TYPE_EXECUTION_PATTERNS:
        if pattern.search(normalized):
            return {"step_type": STEP_TYPE_EXECUTION, "step_type_signal": signal, "step_type_head": head}
    return {"step_type": STEP_TYPE_UNKNOWN, "step_type_signal": "no_match", "step_type_head": head}


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


def _empty_step_type_consensus(reason: str) -> dict[str, Any]:
    return {
        "step_type_consensus_type": None,
        "step_type_consensus_signal": None,
        "step_type_consensus_support_count": 0,
        "step_type_consensus_vote_fraction": None,
        "step_type_consensus_group_counts": {},
        "step_type_consensus_heads": {},
        "step_type_consensus_signals": {},
        "step_type_consensus_reject_reason": reason,
        "step_type_selected_rollout": None,
    }


def _empty_dynamics_consensus(reason: str) -> dict[str, Any]:
    return {
        "dynamics_consensus_type": None,
        "dynamics_consensus_signal": None,
        "dynamics_consensus_support_count": 0,
        "dynamics_consensus_vote_fraction": None,
        "dynamics_consensus_group_counts": {},
        "dynamics_consensus_reject_reason": reason,
        "dynamics_selected_rollout": None,
    }


def _float_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_dynamics(
    rollout: dict[str, Any],
    *,
    flat_max_mean_surprisal: float,
    flat_max_var: float,
    flat_max_delta: float,
    unstable_min_delta: float,
    unstable_min_max_jump: float,
    unstable_min_spike_ratio: float,
    structured_min_var: float,
    structured_min_max_jump: float,
) -> dict[str, str]:
    scored_tokens = int(rollout.get("dynamics_num_scored_tokens") or 0)
    if scored_tokens <= 0:
        return {"dynamics_category": DYNAMICS_UNKNOWN, "dynamics_signal": "no_scores"}

    mean_surprisal = _float_metric(rollout, "surprisal_mean")
    var = _float_metric(rollout, "surprisal_var")
    mean_abs_delta = _float_metric(rollout, "surprisal_mean_abs_delta")
    max_abs_delta = _float_metric(rollout, "surprisal_max_abs_delta")
    spike_ratio = _float_metric(rollout, "surprisal_spike_ratio")

    if (
        mean_surprisal is not None
        and var is not None
        and mean_surprisal <= flat_max_mean_surprisal
        and var <= flat_max_var
        and (mean_abs_delta is None or mean_abs_delta <= flat_max_delta)
    ):
        return {"dynamics_category": DYNAMICS_FLAT_LOW, "dynamics_signal": "low_flat_surprisal"}

    if (
        (mean_abs_delta is not None and mean_abs_delta >= unstable_min_delta)
        or (max_abs_delta is not None and max_abs_delta >= unstable_min_max_jump)
        or (spike_ratio is not None and spike_ratio >= unstable_min_spike_ratio)
    ):
        return {"dynamics_category": DYNAMICS_UNSTABLE, "dynamics_signal": "unstable_surprisal"}

    if (
        (var is not None and var >= structured_min_var)
        or (max_abs_delta is not None and max_abs_delta >= structured_min_max_jump)
    ):
        return {"dynamics_category": DYNAMICS_STRUCTURED, "dynamics_signal": "structured_surprisal"}

    return {"dynamics_category": DYNAMICS_UNKNOWN, "dynamics_signal": "low_information"}


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


def _add_probe_cost(total: dict[str, float | int], delta: dict[str, float | int]) -> None:
    for key, value in delta.items():
        total[key] = total.get(key, 0) + value


def _zero_probe_cost() -> dict[str, float | int]:
    return {
        "probe_decode_tokens": 0,
        "probe_prefill_tokens": 0,
        "probe_generate_calls": 0,
        "probe_wall_time": 0.0,
    }


def _step_ends_with_colon(step_text: str, finish: str) -> bool:
    if _is_eos_finish(finish):
        return False
    return step_text.rstrip().endswith(":")


def _active_step_logs(step_logs: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any], str]]:
    rows: list[tuple[int, dict[str, Any], str]] = []
    for idx, row in enumerate(step_logs):
        if row.get("discarded_by_branch_cut"):
            continue
        text = str(row.get("step_text") or "")
        skeleton = str(row.get("step_skeleton") or normalize_step_skeleton(text))
        row["step_skeleton"] = skeleton
        if skeleton:
            rows.append((idx, row, skeleton))
    return rows


def _branch_cut_repeat_candidate(
    step_logs: list[dict[str, Any]],
    *,
    window: int,
    threshold: int,
) -> dict[str, Any] | None:
    if window < 2 or threshold < 2:
        return None
    recent = _active_step_logs(step_logs)[-window:]
    if len(recent) < threshold:
        return None

    by_skeleton: dict[str, list[tuple[int, dict[str, Any], str]]] = {}
    for item in recent:
        by_skeleton.setdefault(item[2], []).append(item)
    current_skeleton = recent[-1][2]
    candidates = [
        (skeleton, items)
        for skeleton, items in by_skeleton.items()
        if len(items) >= threshold and skeleton == current_skeleton
    ]
    if not candidates:
        candidates = [(skeleton, items) for skeleton, items in by_skeleton.items() if len(items) >= threshold]
    if not candidates:
        return None

    skeleton, items = max(candidates, key=lambda item: (len(item[1]), item[1][-1][0]))
    return {
        "skeleton": skeleton,
        "support_count": len(items),
        "support_step_indices": [int(item[1].get("step_idx") or 0) for item in items],
        "cut_log_idx": items[0][0],
    }


def _apply_branch_cut_recovery(
    state: GenerationState,
    step_logs: list[dict[str, Any]],
    *,
    recovery_id: int,
    trigger_reason: str,
    window: int,
    threshold: int,
) -> bool:
    candidate = _branch_cut_repeat_candidate(step_logs, window=window, threshold=threshold)
    if candidate is None:
        return False

    cut_log_idx = int(candidate["cut_log_idx"])
    discarded: list[dict[str, Any]] = []
    suffix_parts: list[str] = []
    for row in step_logs[cut_log_idx:]:
        if row.get("discarded_by_branch_cut"):
            continue
        row["discarded_by_branch_cut"] = True
        row["branch_cut_recovery_id"] = recovery_id
        discarded.append(row)
        suffix_parts.append(str(row.get("step_text") or ""))

    suffix = "".join(suffix_parts)
    if not suffix or not state.assistant_prefix_text.endswith(suffix):
        state.trace.append(
            TraceEvent(
                state.step_count,
                "branch_cut_recovery_failed",
                {
                    "recovery_id": recovery_id,
                    "trigger_reason": trigger_reason,
                    "skeleton": candidate["skeleton"],
                    "reason": "prefix_suffix_mismatch",
                },
            )
        )
        for row in discarded:
            row.pop("discarded_by_branch_cut", None)
            row.pop("branch_cut_recovery_id", None)
        return False

    state.assistant_prefix_text = state.assistant_prefix_text[: -len(suffix)] + BRANCH_CUT_RECOVERY_BRIDGE
    state.trace.append(
        TraceEvent(
            state.step_count,
            "branch_cut_recovery",
            {
                "recovery_id": recovery_id,
                "trigger_reason": trigger_reason,
                "skeleton": candidate["skeleton"],
                "support_count": candidate["support_count"],
                "support_step_indices": candidate["support_step_indices"],
                "discarded_step_indices": [int(row.get("step_idx") or 0) for row in discarded],
                "bridge": BRANCH_CUT_RECOVERY_BRIDGE,
                "window": window,
                "threshold": threshold,
            },
        )
    )
    return True


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


def _selected_step_type_consensus_rollout(
    probe_row: dict[str, Any],
    *,
    enabled: bool,
    min_agreement_count: int,
    consecutive_step_type_exec: int,
    max_consecutive_step_type_exec: int,
) -> dict[str, Any]:
    rollouts = list(probe_row.get("rollouts") or [])
    if not rollouts:
        return _empty_step_type_consensus("empty")

    groups: dict[str, list[dict[str, Any]]] = {}
    heads: dict[str, str] = {}
    signals: dict[str, str] = {}
    for idx, rollout in enumerate(rollouts):
        if "rollout_idx" not in rollout:
            rollout["rollout_idx"] = idx
        rollout_idx = str(_rollout_idx_sort_value(rollout))
        classified = _classify_step_type(rollout.get("text"))
        step_type = classified["step_type"]
        rollout["step_type"] = step_type
        rollout["step_type_signal"] = classified["step_type_signal"]
        rollout["step_type_head"] = classified["step_type_head"]
        heads[rollout_idx] = classified["step_type_head"]
        signals[rollout_idx] = classified["step_type_signal"]
        groups.setdefault(step_type, []).append(rollout)

    group_counts = {step_type: len(group_rows) for step_type, group_rows in sorted(groups.items())}
    execution_group = groups.get(STEP_TYPE_EXECUTION, [])
    support_count = len(execution_group)
    vote_fraction = support_count / len(rollouts) if rollouts else None
    result = {
        "step_type_consensus_type": STEP_TYPE_EXECUTION if execution_group else None,
        "step_type_consensus_signal": "execution_prefix" if execution_group else None,
        "step_type_consensus_support_count": support_count,
        "step_type_consensus_vote_fraction": vote_fraction,
        "step_type_consensus_group_counts": group_counts,
        "step_type_consensus_heads": heads,
        "step_type_consensus_signals": signals,
        "step_type_consensus_reject_reason": None,
        "step_type_selected_rollout": None,
    }

    if not enabled:
        result["step_type_consensus_reject_reason"] = "disabled_or_not_all_none"
        return result
    if max_consecutive_step_type_exec < 1:
        result["step_type_consensus_reject_reason"] = "budget_disabled"
        return result
    if consecutive_step_type_exec >= max_consecutive_step_type_exec:
        result["step_type_consensus_reject_reason"] = "consecutive_budget"
        return result
    if support_count < min_agreement_count:
        result["step_type_consensus_reject_reason"] = "insufficient_execution_agreement"
        return result

    selected_rollout = _best_rollout(execution_group)
    result["step_type_selected_rollout"] = selected_rollout
    return result


def _selected_dynamics_consensus_rollout(
    probe_row: dict[str, Any],
    *,
    enabled: bool,
    min_agreement_count: int,
    flat_max_mean_surprisal: float,
    flat_max_var: float,
    flat_max_delta: float,
    unstable_min_delta: float,
    unstable_min_max_jump: float,
    unstable_min_spike_ratio: float,
    structured_min_var: float,
    structured_min_max_jump: float,
) -> dict[str, Any]:
    rollouts = list(probe_row.get("rollouts") or [])
    if not rollouts:
        return _empty_dynamics_consensus("empty")

    groups: dict[str, list[dict[str, Any]]] = {}
    for idx, rollout in enumerate(rollouts):
        if "rollout_idx" not in rollout:
            rollout["rollout_idx"] = idx
        if "step_type" not in rollout:
            classified = _classify_step_type(rollout.get("text"))
            rollout["step_type"] = classified["step_type"]
            rollout["step_type_signal"] = classified["step_type_signal"]
            rollout["step_type_head"] = classified["step_type_head"]
        dynamics = _classify_dynamics(
            rollout,
            flat_max_mean_surprisal=flat_max_mean_surprisal,
            flat_max_var=flat_max_var,
            flat_max_delta=flat_max_delta,
            unstable_min_delta=unstable_min_delta,
            unstable_min_max_jump=unstable_min_max_jump,
            unstable_min_spike_ratio=unstable_min_spike_ratio,
            structured_min_var=structured_min_var,
            structured_min_max_jump=structured_min_max_jump,
        )
        rollout["dynamics_category"] = dynamics["dynamics_category"]
        rollout["dynamics_signal"] = dynamics["dynamics_signal"]
        groups.setdefault(dynamics["dynamics_category"], []).append(rollout)

    group_counts = {category: len(group_rows) for category, group_rows in sorted(groups.items())}
    structured_group = [
        rollout
        for rollout in groups.get(DYNAMICS_STRUCTURED, [])
        if rollout.get("step_type") != STEP_TYPE_REFLECTION_TRANSITION
    ]
    support_count = len(structured_group)
    vote_fraction = support_count / len(rollouts) if rollouts else None
    result = {
        "dynamics_consensus_type": DYNAMICS_STRUCTURED if structured_group else None,
        "dynamics_consensus_signal": "structured_surprisal" if structured_group else None,
        "dynamics_consensus_support_count": support_count,
        "dynamics_consensus_vote_fraction": vote_fraction,
        "dynamics_consensus_group_counts": group_counts,
        "dynamics_consensus_reject_reason": None,
        "dynamics_selected_rollout": None,
    }

    if not enabled:
        result["dynamics_consensus_reject_reason"] = "disabled_or_not_all_none"
        return result
    if support_count < min_agreement_count:
        result["dynamics_consensus_reject_reason"] = "insufficient_structured_agreement"
        return result

    result["dynamics_selected_rollout"] = _best_rollout(structured_group)
    return result


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
            False,
            prefix_text,
            "",
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
    probe_logprobs_topk: int = 1,
    enable_step_type_routing: bool = False,
    step_type_min_agreement_count: int | None = None,
    max_consecutive_step_type_exec: int = 2,
    enable_dynamics_routing: bool = False,
    dynamics_min_agreement_count: int | None = None,
    dynamics_flat_max_mean_surprisal: float = 0.35,
    dynamics_flat_max_var: float = 0.02,
    dynamics_flat_max_delta: float = 0.08,
    dynamics_unstable_min_delta: float = 1.0,
    dynamics_unstable_min_max_jump: float = 2.0,
    dynamics_unstable_min_spike_ratio: float = 0.25,
    dynamics_structured_min_var: float = 0.002,
    dynamics_structured_min_max_jump: float = 0.15,
    enable_pure_text_slm_fallback: bool = False,
    enable_branch_cut_recovery: bool = False,
    branch_cut_skeleton_window: int = 12,
    branch_cut_skeleton_threshold: int = 3,
    branch_cut_recovery_steps: int = 5,
    max_branch_cut_recoveries: int = 2,
) -> tuple[BPAResult, list[dict[str, Any]], dict[str, float | int]]:
    if min_agreement_count > probe_k:
        raise ValueError("min_agreement_count cannot exceed probe_k")
    if step_type_min_agreement_count is None:
        step_type_min_agreement_count = min_agreement_count
    if step_type_min_agreement_count > probe_k:
        raise ValueError("step_type_min_agreement_count cannot exceed probe_k")
    if dynamics_min_agreement_count is None:
        dynamics_min_agreement_count = min_agreement_count
    if dynamics_min_agreement_count > probe_k:
        raise ValueError("dynamics_min_agreement_count cannot exceed probe_k")

    protocol = "evidence_consensus_routing"
    state = GenerationState(problem_text=problem_text, generation_protocol=protocol)
    rep = RepetitionState()
    start_time = time.time()
    step_logs: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    probe_cost = _zero_probe_cost()
    boundary_idx = 0
    force_next_llm = False
    consecutive_step_type_exec = 0
    branch_cut_recoveries = 0
    branch_recovery_steps_remaining = 0

    def maybe_start_branch_cut_recovery(trigger_reason: str, *, threshold: int | None = None) -> bool:
        nonlocal branch_cut_recoveries, branch_recovery_steps_remaining, rep, force_next_llm, consecutive_step_type_exec
        if not enable_branch_cut_recovery:
            return False
        if branch_recovery_steps_remaining > 0:
            return False
        if branch_cut_recoveries >= max_branch_cut_recoveries:
            return False
        recovery_id = branch_cut_recoveries + 1
        applied = _apply_branch_cut_recovery(
            state,
            step_logs,
            recovery_id=recovery_id,
            trigger_reason=trigger_reason,
            window=branch_cut_skeleton_window,
            threshold=threshold if threshold is not None else branch_cut_skeleton_threshold,
        )
        if not applied:
            return False
        branch_cut_recoveries = recovery_id
        branch_recovery_steps_remaining = max(branch_cut_recovery_steps, 0)
        rep = RepetitionState()
        force_next_llm = False
        consecutive_step_type_exec = 0
        return True

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
                    "step_skeleton": normalize_step_skeleton(step_text_normalized),
                }
            )

            if generated_step_tokens <= 0 and not step_text_normalized.strip():
                state.phase = Phase.DONE
                state.stop_reason = "empty_step"
                state.trace.append(TraceEvent(state.step_count, "empty_step", {}))
                break

            trigger = update_strict_step_repetition(rep, step_text_normalized)
            if trigger is not None:
                if maybe_start_branch_cut_recovery(trigger, threshold=2):
                    continue
                state.trace.append(TraceEvent(state.step_count, "step_repetition_stop", {"trigger_reason": trigger}))
                if trigger in THINKING_RECOVERY_STOP_REASONS and _can_force_final_answer_from_thinking(state, config):
                    _append_forced_close_think(state)
                    continue
                state.phase = Phase.DONE
                state.stop_reason = trigger
                break
            if maybe_start_branch_cut_recovery("skeleton_repeat"):
                continue

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
                    "step_skeleton": normalize_step_skeleton(step_text_normalized),
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

        if branch_recovery_steps_remaining > 0:
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
            branch_recovery_steps_remaining -= 1
            consecutive_step_type_exec = 0
            step_logs.append(
                {
                    "step_idx": state.step_count - 1,
                    "decision": "llm_branch_recovery",
                    "generation_source": "llm",
                    "finish_reason": finish,
                    "step_text": step_text_normalized,
                    "generated_step_tokens": generated_step_tokens,
                    "reused_probe_rollout": False,
                    "continued_probe_rollout": False,
                    "selected_rollout_idx": None,
                    "branch_cut_recovery_id": branch_cut_recoveries,
                    "branch_recovery_steps_remaining": branch_recovery_steps_remaining,
                    "step_skeleton": normalize_step_skeleton(step_text_normalized),
                }
            )
            if generated_step_tokens <= 0 and not step_text_normalized.strip():
                state.phase = Phase.DONE
                state.stop_reason = "empty_step"
                state.trace.append(TraceEvent(state.step_count, "empty_step", {}))
                break
            if _is_eos_finish(finish):
                state.phase = Phase.DONE
                state.stop_reason = "eos"
            else:
                force_next_llm = branch_recovery_steps_remaining <= 0 and _step_ends_with_colon(step_text_normalized, finish)
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
            consecutive_step_type_exec = 0
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
                    "step_skeleton": normalize_step_skeleton(step_text_normalized),
                }
            )
            if generated_step_tokens <= 0 and not step_text_normalized.strip():
                state.phase = Phase.DONE
                state.stop_reason = "empty_step"
                state.trace.append(TraceEvent(state.step_count, "empty_step", {}))
                break
            trigger = update_strict_step_repetition(rep, step_text_normalized)
            if trigger is not None:
                if maybe_start_branch_cut_recovery(trigger, threshold=2):
                    continue
                state.trace.append(TraceEvent(state.step_count, "step_repetition_stop", {"trigger_reason": trigger}))
                if trigger in THINKING_RECOVERY_STOP_REASONS and _can_force_final_answer_from_thinking(state, config):
                    _append_forced_close_think(state)
                    continue
                state.phase = Phase.DONE
                state.stop_reason = trigger
                break
            if maybe_start_branch_cut_recovery("skeleton_repeat"):
                continue
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
                probe_logprobs_topk=probe_logprobs_topk,
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
        selected_rollout = prefix_consensus["selected_rollout"]
        step_type_consensus = _selected_step_type_consensus_rollout(
            probe_row,
            enabled=enable_step_type_routing and prefix_consensus["stage1_case"] == "all_none",
            min_agreement_count=step_type_min_agreement_count,
            consecutive_step_type_exec=consecutive_step_type_exec,
            max_consecutive_step_type_exec=max_consecutive_step_type_exec,
        )
        dynamics_consensus = _selected_dynamics_consensus_rollout(
            probe_row,
            enabled=enable_dynamics_routing and prefix_consensus["stage1_case"] == "all_none",
            min_agreement_count=dynamics_min_agreement_count,
            flat_max_mean_surprisal=dynamics_flat_max_mean_surprisal,
            flat_max_var=dynamics_flat_max_var,
            flat_max_delta=dynamics_flat_max_delta,
            unstable_min_delta=dynamics_unstable_min_delta,
            unstable_min_max_jump=dynamics_unstable_min_max_jump,
            unstable_min_spike_ratio=dynamics_unstable_min_spike_ratio,
            structured_min_var=dynamics_structured_min_var,
            structured_min_max_jump=dynamics_structured_min_max_jump,
        )
        if selected_rollout is None and step_type_consensus["step_type_selected_rollout"] is not None:
            selected_rollout = step_type_consensus["step_type_selected_rollout"]
            prefix_consensus = {
                **prefix_consensus,
                "prefix_anchor_idx": selected_rollout.get("rollout_idx"),
                "prefix_anchor_mean_logprob": selected_rollout.get("mean_logprob"),
                "prefix_consensus_channel": "step_type",
                "prefix_consensus_value": STEP_TYPE_EXECUTION,
                "prefix_consensus_support_count": step_type_consensus["step_type_consensus_support_count"],
                "prefix_consensus_vote_fraction": step_type_consensus["step_type_consensus_vote_fraction"],
                "prefix_consensus_support_by_rollout": {
                    str(_rollout_idx_sort_value(rollout)): rollout.get("step_type") == STEP_TYPE_EXECUTION
                    for rollout in probe_row.get("rollouts") or []
                },
                "prefix_consensus_group_counts": step_type_consensus["step_type_consensus_group_counts"],
                "prefix_consensus_stage": 1.5,
            }
        if selected_rollout is None and dynamics_consensus["dynamics_selected_rollout"] is not None:
            selected_rollout = dynamics_consensus["dynamics_selected_rollout"]
            prefix_consensus = {
                **prefix_consensus,
                "prefix_anchor_idx": selected_rollout.get("rollout_idx"),
                "prefix_anchor_mean_logprob": selected_rollout.get("mean_logprob"),
                "prefix_consensus_channel": "dynamics",
                "prefix_consensus_value": DYNAMICS_STRUCTURED,
                "prefix_consensus_support_count": dynamics_consensus["dynamics_consensus_support_count"],
                "prefix_consensus_vote_fraction": dynamics_consensus["dynamics_consensus_vote_fraction"],
                "prefix_consensus_support_by_rollout": {
                    str(_rollout_idx_sort_value(rollout)): (
                        rollout.get("dynamics_category") == DYNAMICS_STRUCTURED
                        and rollout.get("step_type") != STEP_TYPE_REFLECTION_TRANSITION
                    )
                    for rollout in probe_row.get("rollouts") or []
                },
                "prefix_consensus_group_counts": dynamics_consensus["dynamics_consensus_group_counts"],
                "prefix_consensus_stage": 1.55,
            }
        if (
            selected_rollout is None
            and enable_pure_text_slm_fallback
            and prefix_consensus["stage1_case"] == "all_none"
        ):
            pure_text_rollouts = list(probe_row.get("rollouts") or [])
            if pure_text_rollouts:
                selected_rollout = _best_rollout(pure_text_rollouts)
                prefix_consensus = {
                    **prefix_consensus,
                    "prefix_anchor_idx": selected_rollout.get("rollout_idx"),
                    "prefix_anchor_mean_logprob": selected_rollout.get("mean_logprob"),
                    "prefix_consensus_channel": "pure_text",
                    "prefix_consensus_value": "all_none_best_logprob",
                    "prefix_consensus_support_count": len(pure_text_rollouts),
                    "prefix_consensus_vote_fraction": 1.0,
                    "prefix_consensus_support_by_rollout": {
                        str(_rollout_idx_sort_value(rollout)): True for rollout in pure_text_rollouts
                    },
                    "prefix_consensus_group_counts": {"all_none": len(pure_text_rollouts)},
                    "prefix_consensus_stage": 1.6,
                }
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
        probe_row["step_type_routing_enabled"] = enable_step_type_routing
        probe_row["step_type_min_agreement_count"] = step_type_min_agreement_count
        probe_row["max_consecutive_step_type_exec"] = max_consecutive_step_type_exec
        probe_row["consecutive_step_type_exec_before"] = consecutive_step_type_exec
        probe_row["step_type_consensus_type"] = step_type_consensus["step_type_consensus_type"]
        probe_row["step_type_consensus_signal"] = step_type_consensus["step_type_consensus_signal"]
        probe_row["step_type_consensus_support_count"] = step_type_consensus["step_type_consensus_support_count"]
        probe_row["step_type_consensus_vote_fraction"] = step_type_consensus["step_type_consensus_vote_fraction"]
        probe_row["step_type_consensus_group_counts"] = step_type_consensus["step_type_consensus_group_counts"]
        probe_row["step_type_consensus_heads"] = step_type_consensus["step_type_consensus_heads"]
        probe_row["step_type_consensus_signals"] = step_type_consensus["step_type_consensus_signals"]
        probe_row["step_type_consensus_reject_reason"] = step_type_consensus["step_type_consensus_reject_reason"]
        probe_row["dynamics_routing_enabled"] = enable_dynamics_routing
        probe_row["dynamics_min_agreement_count"] = dynamics_min_agreement_count
        probe_row["probe_logprobs_topk"] = probe_logprobs_topk
        probe_row["dynamics_consensus_type"] = dynamics_consensus["dynamics_consensus_type"]
        probe_row["dynamics_consensus_signal"] = dynamics_consensus["dynamics_consensus_signal"]
        probe_row["dynamics_consensus_support_count"] = dynamics_consensus["dynamics_consensus_support_count"]
        probe_row["dynamics_consensus_vote_fraction"] = dynamics_consensus["dynamics_consensus_vote_fraction"]
        probe_row["dynamics_consensus_group_counts"] = dynamics_consensus["dynamics_consensus_group_counts"]
        probe_row["dynamics_consensus_reject_reason"] = dynamics_consensus["dynamics_consensus_reject_reason"]
        probe_row["pure_text_slm_fallback_enabled"] = enable_pure_text_slm_fallback
        probe_row["pure_text_slm_fallback_used"] = prefix_consensus["prefix_consensus_stage"] == 1.6
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
            consecutive_step_type_exec = 0
        elif probe_row["prefix_consensus_stage"] == 1.5:
            decision = "slm_step_type_reuse"
            force_next_llm = False
            consecutive_step_type_exec += 1
        elif probe_row["prefix_consensus_stage"] == 1.55:
            decision = "slm_dynamics_reuse"
            force_next_llm = False
            consecutive_step_type_exec = 0
        elif probe_row["prefix_consensus_stage"] == 1.6:
            decision = "slm_pure_text_reuse"
            force_next_llm = False
            consecutive_step_type_exec = 0
        elif probe_row["reused_probe_rollout"]:
            decision = "slm_probe_reuse"
            force_next_llm = False
            consecutive_step_type_exec = 0
        else:
            decision = "slm_direct"
            force_next_llm = False
            consecutive_step_type_exec = 0
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
            "prefix_consensus_stage": probe_row["prefix_consensus_stage"],
            "step_skeleton": normalize_step_skeleton(step_text_normalized),
        }
        step_logs.append(log_row)

        if generated_step_tokens <= 0 and not step_text_normalized.strip():
            state.phase = Phase.DONE
            state.stop_reason = "empty_step"
            state.trace.append(TraceEvent(state.step_count, "empty_step", {}))
            break

        trigger = update_strict_step_repetition(rep, step_text_normalized)
        if trigger is not None:
            if maybe_start_branch_cut_recovery(trigger, threshold=2):
                continue
            state.trace.append(TraceEvent(state.step_count, "step_repetition_stop", {"trigger_reason": trigger}))
            if trigger in THINKING_RECOVERY_STOP_REASONS and _can_force_final_answer_from_thinking(state, config):
                _append_forced_close_think(state)
                continue
            state.phase = Phase.DONE
            state.stop_reason = trigger
            break
        if maybe_start_branch_cut_recovery("skeleton_repeat"):
            continue

        if _is_eos_finish(finish):
            state.phase = Phase.DONE
            state.stop_reason = "eos"

    state.trace.append(TraceEvent(state.step_count, "step_logs", {"steps": step_logs}))
    active_step_logs = [row for row in step_logs if not row.get("discarded_by_branch_cut")]
    result = BPAResult(
        answer=extract_answer_from_steps(active_step_logs, state.assistant_prefix_text),
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
    step_type_routing_enabled: bool = False,
    step_type_min_agreement_count: int | None = None,
    max_consecutive_step_type_exec: int = 2,
    dynamics_routing_enabled: bool = False,
    dynamics_min_agreement_count: int | None = None,
    probe_logprobs_topk: int = 1,
    dynamics_flat_max_mean_surprisal: float = 0.35,
    dynamics_flat_max_var: float = 0.02,
    dynamics_flat_max_delta: float = 0.08,
    dynamics_unstable_min_delta: float = 1.0,
    dynamics_unstable_min_max_jump: float = 2.0,
    dynamics_unstable_min_spike_ratio: float = 0.25,
    dynamics_structured_min_var: float = 0.002,
    dynamics_structured_min_max_jump: float = 0.15,
    pure_text_slm_fallback_enabled: bool = False,
    branch_cut_recovery_enabled: bool = False,
    branch_cut_recovery_steps: int = 5,
    branch_cut_skeleton_window: int = 12,
    branch_cut_skeleton_threshold: int = 3,
) -> dict[str, Any]:
    correct = None
    if problem.gold_answer is not None:
        correct = benchmark_eval_match(result.answer, problem.gold_answer, dataset)
        result.correct = correct
    main_decode_tokens = result.state.slm_decode_tokens + result.state.llm_decode_tokens
    probe_decode_tokens = int(probe_cost.get("probe_decode_tokens") or 0)
    real_boundary_rows = list(boundary_rows)
    step_rows = _step_rows(result)
    num_branch_cut_recoveries = sum(1 for event in result.state.trace if event.event == "branch_cut_recovery")
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
        "num_step_type_reused_steps": sum(1 for row in real_boundary_rows if row.get("prefix_consensus_stage") == 1.5),
        "num_dynamics_reused_steps": sum(1 for row in real_boundary_rows if row.get("prefix_consensus_stage") == 1.55),
        "num_pure_text_reused_steps": sum(1 for row in real_boundary_rows if row.get("prefix_consensus_stage") == 1.6),
        "step_type_routing_enabled": step_type_routing_enabled,
        "step_type_min_agreement_count": (
            step_type_min_agreement_count if step_type_min_agreement_count is not None else min_agreement_count
        ),
        "max_consecutive_step_type_exec": max_consecutive_step_type_exec,
        "dynamics_routing_enabled": dynamics_routing_enabled,
        "dynamics_min_agreement_count": (
            dynamics_min_agreement_count if dynamics_min_agreement_count is not None else min_agreement_count
        ),
        "probe_logprobs_topk": probe_logprobs_topk,
        "dynamics_flat_max_mean_surprisal": dynamics_flat_max_mean_surprisal,
        "dynamics_flat_max_var": dynamics_flat_max_var,
        "dynamics_flat_max_delta": dynamics_flat_max_delta,
        "dynamics_unstable_min_delta": dynamics_unstable_min_delta,
        "dynamics_unstable_min_max_jump": dynamics_unstable_min_max_jump,
        "dynamics_unstable_min_spike_ratio": dynamics_unstable_min_spike_ratio,
        "dynamics_structured_min_var": dynamics_structured_min_var,
        "dynamics_structured_min_max_jump": dynamics_structured_min_max_jump,
        "pure_text_slm_fallback_enabled": pure_text_slm_fallback_enabled,
        "branch_cut_recovery_enabled": branch_cut_recovery_enabled,
        "num_branch_cut_recoveries": num_branch_cut_recoveries,
        "branch_cut_recovery_steps": branch_cut_recovery_steps,
        "branch_cut_skeleton_window": branch_cut_skeleton_window,
        "branch_cut_skeleton_threshold": branch_cut_skeleton_threshold,
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
        "avg_num_step_type_reused_steps": _mean([row.get("num_step_type_reused_steps") for row in rows]),
        "avg_num_dynamics_reused_steps": _mean([row.get("num_dynamics_reused_steps") for row in rows]),
        "avg_num_pure_text_reused_steps": _mean([row.get("num_pure_text_reused_steps") for row in rows]),
        "avg_num_branch_cut_recoveries": _mean([row.get("num_branch_cut_recoveries") for row in rows]),
        "avg_main_decode_tokens": _mean([row.get("main_decode_tokens") for row in rows]),
        "avg_total_decode_tokens_including_probe": _mean(
            [row.get("total_decode_tokens_including_probe") for row in rows]
        ),
        "total_llm_decode_tokens": sum(float(row.get("llm_decode_tokens") or 0.0) for row in rows),
        "total_llm_prefill_tokens": sum(float(row.get("llm_prefill_tokens") or 0.0) for row in rows),
        "total_slm_decode_tokens": sum(float(row.get("slm_decode_tokens") or 0.0) for row in rows),
        "total_probe_decode_tokens": sum(float(row.get("probe_decode_tokens") or 0.0) for row in rows),
        "total_step_type_reused_steps": sum(float(row.get("num_step_type_reused_steps") or 0.0) for row in rows),
        "total_dynamics_reused_steps": sum(float(row.get("num_dynamics_reused_steps") or 0.0) for row in rows),
        "total_pure_text_reused_steps": sum(float(row.get("num_pure_text_reused_steps") or 0.0) for row in rows),
        "total_branch_cut_recoveries": sum(float(row.get("num_branch_cut_recoveries") or 0.0) for row in rows),
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
    parser.add_argument("--probe-logprobs-topk", type=int, default=1, help="Top-k logprobs to request for each probe token; 1 records sampled-token surprisal only, >1 also approximates entropy.")
    parser.add_argument("--post-stop-lookahead-tokens", type=int, default=None)
    parser.add_argument("--enable-step-type-routing", action="store_true", help="Allow all-none execution-style probe consensus to reuse an SLM rollout.")
    parser.add_argument("--step-type-min-agreement-count", type=int, default=None, help="Agreement count for Tier 1.5 execution consensus; defaults to --min-agreement-count.")
    parser.add_argument("--max-consecutive-step-type-exec", type=int, default=2, help="Maximum consecutive Tier 1.5 SLM execution-style reuses before forcing an LLM check-in.")
    parser.add_argument("--enable-dynamics-routing", action="store_true", help="Allow all-none structured-surprisal probe consensus to reuse an SLM rollout.")
    parser.add_argument("--dynamics-min-agreement-count", type=int, default=None, help="Agreement count for structured dynamics consensus; defaults to --min-agreement-count.")
    parser.add_argument("--dynamics-flat-max-mean-surprisal", type=float, default=0.35)
    parser.add_argument("--dynamics-flat-max-var", type=float, default=0.02)
    parser.add_argument("--dynamics-flat-max-delta", type=float, default=0.08)
    parser.add_argument("--dynamics-unstable-min-delta", type=float, default=1.0)
    parser.add_argument("--dynamics-unstable-min-max-jump", type=float, default=2.0)
    parser.add_argument("--dynamics-unstable-min-spike-ratio", type=float, default=0.25)
    parser.add_argument("--dynamics-structured-min-var", type=float, default=0.002)
    parser.add_argument("--dynamics-structured-min-max-jump", type=float, default=0.15)
    parser.add_argument("--enable-pure-text-slm-fallback", action="store_true", help="For all-none pure-text probes not accepted by Tier 1.5, reuse the best SLM probe rollout instead of routing to the LLM.")
    parser.add_argument("--enable-branch-cut-recovery", action="store_true", help="When a repeated reasoning skeleton is detected, cut the repeated branch and let the LLM recover for a few steps.")
    parser.add_argument("--branch-cut-skeleton-window", type=int, default=12)
    parser.add_argument("--branch-cut-skeleton-threshold", type=int, default=3)
    parser.add_argument("--branch-cut-recovery-steps", type=int, default=5)
    parser.add_argument("--max-branch-cut-recoveries", type=int, default=2)
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
            probe_logprobs_topk=args.probe_logprobs_topk,
            enable_step_type_routing=args.enable_step_type_routing,
            step_type_min_agreement_count=args.step_type_min_agreement_count,
            max_consecutive_step_type_exec=args.max_consecutive_step_type_exec,
            enable_dynamics_routing=args.enable_dynamics_routing,
            dynamics_min_agreement_count=args.dynamics_min_agreement_count,
            dynamics_flat_max_mean_surprisal=args.dynamics_flat_max_mean_surprisal,
            dynamics_flat_max_var=args.dynamics_flat_max_var,
            dynamics_flat_max_delta=args.dynamics_flat_max_delta,
            dynamics_unstable_min_delta=args.dynamics_unstable_min_delta,
            dynamics_unstable_min_max_jump=args.dynamics_unstable_min_max_jump,
            dynamics_unstable_min_spike_ratio=args.dynamics_unstable_min_spike_ratio,
            dynamics_structured_min_var=args.dynamics_structured_min_var,
            dynamics_structured_min_max_jump=args.dynamics_structured_min_max_jump,
            enable_pure_text_slm_fallback=args.enable_pure_text_slm_fallback,
            enable_branch_cut_recovery=args.enable_branch_cut_recovery,
            branch_cut_skeleton_window=args.branch_cut_skeleton_window,
            branch_cut_skeleton_threshold=args.branch_cut_skeleton_threshold,
            branch_cut_recovery_steps=args.branch_cut_recovery_steps,
            max_branch_cut_recoveries=args.max_branch_cut_recoveries,
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
            step_type_routing_enabled=args.enable_step_type_routing,
            step_type_min_agreement_count=args.step_type_min_agreement_count,
            max_consecutive_step_type_exec=args.max_consecutive_step_type_exec,
            dynamics_routing_enabled=args.enable_dynamics_routing,
            dynamics_min_agreement_count=args.dynamics_min_agreement_count,
            probe_logprobs_topk=args.probe_logprobs_topk,
            dynamics_flat_max_mean_surprisal=args.dynamics_flat_max_mean_surprisal,
            dynamics_flat_max_var=args.dynamics_flat_max_var,
            dynamics_flat_max_delta=args.dynamics_flat_max_delta,
            dynamics_unstable_min_delta=args.dynamics_unstable_min_delta,
            dynamics_unstable_min_max_jump=args.dynamics_unstable_min_max_jump,
            dynamics_unstable_min_spike_ratio=args.dynamics_unstable_min_spike_ratio,
            dynamics_structured_min_var=args.dynamics_structured_min_var,
            dynamics_structured_min_max_jump=args.dynamics_structured_min_max_jump,
            pure_text_slm_fallback_enabled=args.enable_pure_text_slm_fallback,
            branch_cut_recovery_enabled=args.enable_branch_cut_recovery,
            branch_cut_recovery_steps=args.branch_cut_recovery_steps,
            branch_cut_skeleton_window=args.branch_cut_skeleton_window,
            branch_cut_skeleton_threshold=args.branch_cut_skeleton_threshold,
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
