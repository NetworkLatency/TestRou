from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.engines import completion, generated_text, generated_token_ids, init_engines, logprob_value
from bpa.eval.datasets import load_eval_dataset
from bpa.eval.exp_disagreement_routing import _step_expression_flags
from bpa.render import render_for_continuation


CSV_FIELDS = [
    "dataset",
    "problem_id",
    "boundary_idx",
    "target_step_idx",
    "problem_correct",
    "problem_wall_time",
    "num_boundaries",
    "long_tail",
    "forced_or_budget",
    "original_decision",
    "stage1_case",
    "prefix_token_len",
    "lookahead_text",
    "lookahead_token_count",
    "h_init_topk",
    "mean_token_logprob",
    "p10_token_logprob",
    "token_logprob_min",
    "token_logprob_max",
    "entropy_mean",
    "entropy_var",
    "entropy_max",
    "p_step_end_onset",
    "p_step_end_max",
    "p_step_end_mean",
    "p_short_step_end",
    "p_newline_onset",
    "p_newline_max",
    "p_short_newline",
    "p_think_end_onset",
    "p_think_end_max",
    "p_short_think_end",
    "lookahead_uncertain",
    "lookahead_transition",
    "lookahead_commit",
    "lookahead_bridge_operation",
    "lookahead_step_type",
    "next_step_uncertain",
    "next_step_transition",
    "next_step_commit",
    "next_step_bridge_operation",
    "next_step_type",
    "closure_state",
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_exp(logprob: float | None) -> float:
    if logprob is None or not math.isfinite(logprob):
        return 0.0
    return math.exp(max(-80.0, min(0.0, logprob)))


def _logprob_record_value(record: Any, token_id: int) -> float | None:
    if not isinstance(record, dict):
        return None
    value = record.get(token_id)
    if value is None:
        value = record.get(str(token_id))
    if value is None and len(record) == 1:
        value = next(iter(record.values()))
    if value is None:
        return None
    try:
        return logprob_value(value)
    except (TypeError, ValueError):
        return None


def _entropy_from_topk(record: Any) -> float | None:
    if not isinstance(record, dict) or not record:
        return None
    values: list[float] = []
    for value in record.values():
        try:
            lp = logprob_value(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(lp):
            values.append(lp)
    if len(values) < 2:
        return None
    max_lp = max(values)
    weights = [math.exp(lp - max_lp) for lp in values]
    total = sum(weights)
    if total <= 0:
        return None
    probs = [w / total for w in weights]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _series_var(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _p10(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.floor(0.1 * (len(ordered) - 1)))))
    return ordered[idx]


def _topk_entropy(record: Any) -> float | None:
    return _entropy_from_topk(record)


def _topk_prob_sum(
    record: Any,
    *,
    token_ids: set[int],
    decoded_predicate,
    tokenizer: Any,
) -> float:
    if not isinstance(record, dict) or not record:
        return 0.0
    total = 0.0
    seen: set[int] = set()
    for key, value in record.items():
        try:
            token_id = int(key)
        except (TypeError, ValueError):
            continue
        if token_id in seen:
            continue
        seen.add(token_id)
        matched = token_id in token_ids
        if not matched:
            try:
                decoded = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                decoded = ""
            matched = bool(decoded_predicate(decoded))
        if matched:
            try:
                total += _safe_exp(logprob_value(value))
            except (TypeError, ValueError):
                continue
    return min(1.0, total)


def _candidate_single_token_ids(tokenizer: Any, texts: list[str]) -> set[int]:
    ids: set[int] = set()
    for text in texts:
        for variant in {text, " " + text}:
            try:
                encoded = list(tokenizer.encode(variant, add_special_tokens=False))
            except Exception:
                encoded = []
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
    return ids


def _event_flags(problem_dir: Path, problem_id: int) -> dict[str, bool]:
    trace_path = problem_dir / f"{problem_id}.trace.json"
    if not trace_path.exists():
        return {"forced_or_budget": False}
    try:
        trace = _read_json(trace_path)
    except Exception:
        return {"forced_or_budget": False}
    events = trace if isinstance(trace, list) else trace.get("events", []) if isinstance(trace, dict) else []
    event_names = {str(event.get("event")) for event in events if isinstance(event, dict)}
    return {
        "forced_or_budget": bool(
            {"step_repetition_stop", "total_token_budget_exhausted", "context_budget_exhausted"} & event_names
        )
    }


def _problem_summary(problem_dir: Path, problem_id: int) -> dict[str, Any]:
    problem_path = problem_dir / f"{problem_id}.problem.json"
    if not problem_path.exists():
        return {}
    data = _read_json(problem_path)
    return dict(data.get("summary") or {})


def _decision_from_boundary(row: dict[str, Any]) -> str:
    if row.get("reused_probe_rollout"):
        stage = row.get("prefix_consensus_stage")
        if stage == 1 or stage == "1":
            return "hard_evidence_reuse"
        if str(stage) == "1.5":
            return "step_type_reuse"
        if str(stage) == "1.55":
            return "dynamics_reuse"
        if str(stage) == "1.6":
            return "pure_text_reuse"
        return "slm_reuse"
    if row.get("routed_to_llm"):
        return "llm_after_probe"
    return "other"


def _short_prob(probs: list[float]) -> float:
    product = 1.0
    for prob in probs:
        product *= max(0.0, 1.0 - min(1.0, prob))
    return 1.0 - product


def _closure_state(
    *,
    p_short: float,
    p_max: float,
    flags: dict[str, Any],
    entropy_var: float | None,
    high: float,
    low: float,
    trap_max: float,
) -> str:
    has_commit = bool(flags.get("expr_commit"))
    is_transition = bool(flags.get("expr_uncertain") or flags.get("expr_transition"))
    is_execution_like = str(flags.get("monitor_step_type")) == "execution" or has_commit
    unstable = entropy_var is not None and entropy_var >= 0.25
    if p_short >= high and is_execution_like and not is_transition:
        return "closed_execution"
    if p_short >= high and (is_transition or unstable):
        return "premature_closure"
    if p_short <= low and p_max <= trap_max and is_transition:
        return "wandering"
    if p_short <= low and is_execution_like and not unstable:
        return "open_execution"
    return "middle"


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        output = [0.0] * len(values)
        idx = 0
        while idx < len(indexed):
            end = idx + 1
            while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
                end += 1
            rank = (idx + 1 + end) / 2.0
            for original_idx, _ in indexed[idx:end]:
                output[original_idx] = rank
            idx = end
        return output

    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def collect_termination_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = BPAConfig.from_json(args.config)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    problem_by_id = {int(problem.problem_id): problem for problem in problems}
    slm, _ = init_engines(config)
    tokenizer = slm.ensure_tokenizer()
    step_end_ids = _candidate_single_token_ids(tokenizer, ["\n\n", "\r\n\r\n"])
    newline_ids = _candidate_single_token_ids(tokenizer, ["\n", "\r\n"])
    think_end_ids = _candidate_single_token_ids(tokenizer, ["</think>"])
    debug_info = {
        "step_end_token_ids": sorted(step_end_ids),
        "newline_token_ids": sorted(newline_ids),
        "think_end_token_ids": sorted(think_end_ids),
        "lookahead_tokens": args.lookahead_tokens,
        "logprobs_topk": args.logprobs_topk,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "token_sets.json").write_text(json.dumps(debug_info, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    problem_dirs = sorted(
        [path for path in args.trace_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    if args.max_problems is not None:
        problem_dirs = problem_dirs[: args.max_problems]

    sampling = slm.sampling_params(
        max_tokens=args.lookahead_tokens,
        temperature=args.temperature,
        logprobs=args.logprobs_topk,
    )

    for problem_dir in tqdm(problem_dirs, desc="termination-probe"):
        problem_id = int(problem_dir.name)
        problem = problem_by_id.get(problem_id)
        if problem is None:
            continue
        steps = _read_jsonl(problem_dir / f"{problem_id}.steps.jsonl")
        boundaries = _read_jsonl(problem_dir / f"{problem_id}.boundaries.jsonl")
        if args.max_boundaries_per_problem is not None:
            boundaries = boundaries[: args.max_boundaries_per_problem]
        summary = _problem_summary(problem_dir, problem_id)
        event_flags = _event_flags(problem_dir, problem_id)
        problem_correct = _bool(summary.get("correct"))
        problem_wall_time = _float(summary.get("total_wall_time"))
        num_boundaries = int(_float(summary.get("num_boundaries")))
        long_tail = problem_wall_time >= args.long_tail_time or num_boundaries >= args.long_tail_boundaries

        step_text_by_idx = {int(row.get("step_idx", idx)): str(row.get("step_text") or "") for idx, row in enumerate(steps)}
        for boundary in boundaries:
            target_step_idx = int(boundary.get("target_step_idx") or (int(boundary.get("boundary_idx") or 0) + 1))
            prefix = "".join(step_text_by_idx[idx] for idx in sorted(step_text_by_idx) if idx < target_step_idx)
            rendered = render_for_continuation(problem.problem_text, prefix, tokenizer)
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(prompt_ids) + args.lookahead_tokens >= config.max_model_len:
                continue
            started = time.time()
            out = slm.generate(rendered, sampling)[0]
            _ = time.time() - started
            comp = completion(out)
            token_ids = generated_token_ids(out)
            text = generated_text(out)
            logprob_steps = list(getattr(comp, "logprobs", []) or [])
            token_lps: list[float] = []
            entropies: list[float] = []
            p_step_end: list[float] = []
            p_newline: list[float] = []
            p_think_end: list[float] = []
            for token_id, record in zip(token_ids, logprob_steps):
                lp = _logprob_record_value(record, int(token_id))
                if lp is not None and math.isfinite(lp):
                    token_lps.append(lp)
                entropy = _topk_entropy(record)
                if entropy is not None:
                    entropies.append(entropy)
                p_step_end.append(
                    _topk_prob_sum(
                        record,
                        token_ids=step_end_ids,
                        decoded_predicate=lambda value: "\n\n" in value or "\r\n\r\n" in value,
                        tokenizer=tokenizer,
                    )
                )
                p_newline.append(
                    _topk_prob_sum(
                        record,
                        token_ids=newline_ids,
                        decoded_predicate=lambda value: value in {"\n", "\r\n"},
                        tokenizer=tokenizer,
                    )
                )
                p_think_end.append(
                    _topk_prob_sum(
                        record,
                        token_ids=think_end_ids,
                        decoded_predicate=lambda value: "</think" in value,
                        tokenizer=tokenizer,
                    )
                )

            lookahead_flags = _step_expression_flags(text)
            next_step_text = step_text_by_idx.get(target_step_idx, "")
            next_step_flags = _step_expression_flags(next_step_text)
            entropy_var = _series_var(entropies)
            h_init = entropies[0] if entropies else None
            p_short = _short_prob(p_step_end)
            p_max = max(p_step_end, default=0.0)
            row = {
                "dataset": args.dataset,
                "problem_id": problem_id,
                "boundary_idx": boundary.get("boundary_idx"),
                "target_step_idx": target_step_idx,
                "problem_correct": problem_correct,
                "problem_wall_time": problem_wall_time,
                "num_boundaries": num_boundaries,
                "long_tail": long_tail,
                "forced_or_budget": event_flags["forced_or_budget"],
                "original_decision": _decision_from_boundary(boundary),
                "stage1_case": boundary.get("stage1_case"),
                "prefix_token_len": len(prompt_ids),
                "lookahead_text": text.replace("\r", "\\r").replace("\n", "\\n"),
                "lookahead_token_count": len(token_ids),
                "h_init_topk": h_init,
                "mean_token_logprob": sum(token_lps) / len(token_lps) if token_lps else None,
                "p10_token_logprob": _p10(token_lps),
                "token_logprob_min": min(token_lps) if token_lps else None,
                "token_logprob_max": max(token_lps) if token_lps else None,
                "entropy_mean": sum(entropies) / len(entropies) if entropies else None,
                "entropy_var": entropy_var,
                "entropy_max": max(entropies) if entropies else None,
                "p_step_end_onset": p_step_end[0] if p_step_end else None,
                "p_step_end_max": p_max,
                "p_step_end_mean": sum(p_step_end) / len(p_step_end) if p_step_end else None,
                "p_short_step_end": p_short,
                "p_newline_onset": p_newline[0] if p_newline else None,
                "p_newline_max": max(p_newline, default=0.0),
                "p_short_newline": _short_prob(p_newline),
                "p_think_end_onset": p_think_end[0] if p_think_end else None,
                "p_think_end_max": max(p_think_end, default=0.0),
                "p_short_think_end": _short_prob(p_think_end),
                "lookahead_uncertain": lookahead_flags["expr_uncertain"],
                "lookahead_transition": lookahead_flags["expr_transition"],
                "lookahead_commit": lookahead_flags["expr_commit"],
                "lookahead_bridge_operation": lookahead_flags["expr_bridge_operation"],
                "lookahead_step_type": lookahead_flags["monitor_step_type"],
                "next_step_uncertain": next_step_flags["expr_uncertain"],
                "next_step_transition": next_step_flags["expr_transition"],
                "next_step_commit": next_step_flags["expr_commit"],
                "next_step_bridge_operation": next_step_flags["expr_bridge_operation"],
                "next_step_type": next_step_flags["monitor_step_type"],
                "closure_state": _closure_state(
                    p_short=p_short,
                    p_max=p_max,
                    flags=lookahead_flags,
                    entropy_var=entropy_var,
                    high=args.p_short_high,
                    low=args.p_short_low,
                    trap_max=args.trap_max_end,
                ),
            }
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric = ["p_short_step_end", "p_step_end_max", "p_short_newline", "h_init_topk", "entropy_var", "mean_token_logprob"]

    def as_float(row: dict[str, Any], key: str) -> float | None:
        try:
            value = row.get(key)
            if value is None or value == "":
                return None
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    summary: dict[str, Any] = {
        "num_boundaries": len(rows),
        "num_problems": len({row.get("problem_id") for row in rows}),
        "closure_state_counts": dict(Counter(str(row.get("closure_state")) for row in rows)),
    }
    for label, predicate in {
        "correct": lambda row: _bool(row.get("problem_correct")),
        "wrong": lambda row: not _bool(row.get("problem_correct")),
        "long_tail": lambda row: _bool(row.get("long_tail")),
        "forced_or_budget": lambda row: _bool(row.get("forced_or_budget")),
    }.items():
        group = [row for row in rows if predicate(row)]
        summary[f"{label}_count"] = len(group)
        for key in numeric:
            values = [as_float(row, key) for row in group]
            values = [value for value in values if value is not None]
            if values:
                summary[f"{label}_{key}_mean"] = statistics.mean(values)
                summary[f"{label}_{key}_median"] = statistics.median(values)
    xs = [as_float(row, "p_short_step_end") for row in rows]
    ys = [as_float(row, "h_init_topk") for row in rows]
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    summary["spearman_p_short_vs_h_init"] = _spearman([x for x, _ in pairs], [y for _, y in pairs]) if pairs else None
    summary["intervention_curve"] = intervention_curve(rows, args)
    return summary


def intervention_curve(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, float]]:
    values = sorted({_float(row.get("p_short_step_end")) for row in rows})
    if not values:
        return []
    thresholds = []
    for idx in range(args.curve_points):
        q = idx / max(1, args.curve_points - 1)
        thresholds.append(values[min(len(values) - 1, int(q * (len(values) - 1)))])
    failure = [_bool(row.get("long_tail")) or _bool(row.get("forced_or_budget")) or not _bool(row.get("problem_correct")) for row in rows]
    num_failure = sum(1 for flag in failure if flag)
    curve = []
    for threshold in thresholds:
        flagged = [
            _float(row.get("p_short_step_end")) <= threshold
            and _float(row.get("p_step_end_max")) <= args.trap_max_end
            for row in rows
        ]
        intervention_rate = sum(flagged) / len(flagged)
        recall = (sum(1 for flag, fail in zip(flagged, failure) if flag and fail) / num_failure) if num_failure else 0.0
        precision = (
            sum(1 for flag, fail in zip(flagged, failure) if flag and fail) / sum(flagged)
            if sum(flagged)
            else 0.0
        )
        curve.append(
            {
                "threshold": float(threshold),
                "intervention_rate": float(intervention_rate),
                "failure_recall": float(recall),
                "failure_precision": float(precision),
            }
        )
    return curve


def plot_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib unavailable; skipped plots: {exc}")
        return

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def values(key: str, predicate=lambda row: True) -> list[float]:
        output = []
        for row in rows:
            if predicate(row):
                value = _float(row.get(key), default=float("nan"))
                if math.isfinite(value):
                    output.append(value)
        return output

    plt.figure(figsize=(8, 5))
    plt.hist(values("p_short_step_end"), bins=50, alpha=0.8, label="all")
    plt.xlabel("p_short_step_end")
    plt.ylabel("boundary count")
    plt.title("P(end within lookahead) distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "p_short_hist_all.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(values("p_short_step_end", lambda row: _bool(row.get("problem_correct"))), bins=40, alpha=0.55, label="correct")
    plt.hist(values("p_short_step_end", lambda row: not _bool(row.get("problem_correct"))), bins=40, alpha=0.55, label="wrong")
    plt.xlabel("p_short_step_end")
    plt.ylabel("boundary count")
    plt.title("P(end) by final correctness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "p_short_by_correctness.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(values("p_short_step_end", lambda row: _bool(row.get("long_tail"))), bins=40, alpha=0.55, label="long tail")
    plt.hist(values("p_short_step_end", lambda row: not _bool(row.get("long_tail"))), bins=40, alpha=0.55, label="normal")
    plt.xlabel("p_short_step_end")
    plt.ylabel("boundary count")
    plt.title("P(end) by long-tail status")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "p_short_by_long_tail.png", dpi=180)
    plt.close()

    xs = values("p_short_step_end")
    ys = values("h_init_topk")
    if xs and ys:
        paired = [
            (_float(row.get("p_short_step_end"), default=float("nan")), _float(row.get("h_init_topk"), default=float("nan")))
            for row in rows
        ]
        paired = [(x, y) for x, y in paired if math.isfinite(x) and math.isfinite(y)]
        plt.figure(figsize=(6, 5))
        plt.scatter([x for x, _ in paired], [y for _, y in paired], s=8, alpha=0.35)
        rho = summary.get("spearman_p_short_vs_h_init")
        plt.xlabel("p_short_step_end")
        plt.ylabel("H_init top-k entropy")
        plt.title(f"P(end) vs H_init (Spearman={rho:.3f})" if isinstance(rho, float) else "P(end) vs H_init")
        plt.tight_layout()
        plt.savefig(plot_dir / "p_short_vs_hinit.png", dpi=180)
        plt.close()

    curve = summary.get("intervention_curve") or []
    if curve:
        plt.figure(figsize=(7, 5))
        plt.plot([row["intervention_rate"] for row in curve], [row["failure_recall"] for row in curve], marker="o", ms=3)
        plt.xlabel("intervention rate")
        plt.ylabel("failure recall")
        plt.title("Low P(end) trap detector curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "intervention_rate_vs_failure_recall.png", dpi=180)
        plt.close()

    by_correct: dict[str, Counter] = {"correct": Counter(), "wrong": Counter()}
    for row in rows:
        key = "correct" if _bool(row.get("problem_correct")) else "wrong"
        by_correct[key][str(row.get("closure_state"))] += 1
    states = sorted(set(by_correct["correct"]) | set(by_correct["wrong"]))
    x = list(range(len(states)))
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar([idx - width / 2 for idx in x], [by_correct["correct"][state] for state in states], width, label="correct")
    plt.bar([idx + width / 2 for idx in x], [by_correct["wrong"][state] for state in states], width, label="wrong")
    plt.xticks(x, states, rotation=25, ha="right")
    plt.ylabel("boundary count")
    plt.title("Closure-state distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "closure_state_by_correctness.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and plot termination/closure-state diagnostics for SLM routing.")
    parser.add_argument("--config", required=False, help="Path to BPAConfig JSON; required unless --input-csv is used.")
    parser.add_argument("--dataset", default="aime25")
    parser.add_argument("--trace-dir", type=Path, help="Existing run directory with per-problem trace outputs.")
    parser.add_argument("--input-csv", type=Path, help="Existing termination_boundaries.csv; skips model replay.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--max-boundaries-per-problem", type=int, default=None)
    parser.add_argument("--lookahead-tokens", type=int, default=8)
    parser.add_argument("--logprobs-topk", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--long-tail-time", type=float, default=300.0)
    parser.add_argument("--long-tail-boundaries", type=int, default=300)
    parser.add_argument("--p-short-high", type=float, default=0.5)
    parser.add_argument("--p-short-low", type=float, default=0.02)
    parser.add_argument("--trap-max-end", type=float, default=0.005)
    parser.add_argument("--curve-points", type=int, default=40)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_csv:
        rows = _read_csv(args.input_csv)
    else:
        if not args.config or not args.trace_dir:
            raise SystemExit("--config and --trace-dir are required unless --input-csv is provided.")
        rows = collect_termination_rows(args)
        _write_csv(args.output_dir / "termination_boundaries.csv", rows)

    summary = summarize(rows, args)
    (args.output_dir / "termination_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.no_plots:
        plot_outputs(rows, summary, args)
    print(f"Wrote {args.output_dir / 'termination_boundaries.csv'}")
    print(f"Wrote {args.output_dir / 'termination_summary.json'}")
    if not args.no_plots:
        print(f"Wrote plots under {args.output_dir / 'plots'}")


if __name__ == "__main__":
    main()
