from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import load_eval_dataset
from bpa.render import render_for_continuation
from bpa.safety import extract_answer


MULTI_HORIZON_VALUES = [4, 8, 16, 32]
MULTI_HORIZON_BOUNDARY_FIELDS = [
    field
    for horizon in MULTI_HORIZON_VALUES
    for field in (
        f"p_end_mean_h{horizon}",
        f"p_end_max_h{horizon}",
        f"p_short_h{horizon}",
        f"log_p_short_h{horizon}",
        f"end_in_topk_rate_h{horizon}",
    )
]
MULTI_HORIZON_WINDOW_FIELDS = [
    field
    for horizon in MULTI_HORIZON_VALUES
    for field in (
        f"p_short_h{horizon}_mean",
        f"p_short_h{horizon}_median",
        f"log_p_short_h{horizon}_mean",
        f"log_p_short_h{horizon}_median",
        f"end_in_topk_rate_h{horizon}",
    )
]


BOUNDARY_FIELDS = [
    "dataset",
    "problem_id",
    "boundary_idx",
    "boundary_token_idx",
    "problem_correct",
    "problem_long_by_boundaries",
    "problem_long_by_tokens",
    "problem_wrong_long",
    "problem_num_boundaries",
    "problem_generated_tokens",
    "problem_wall_time",
    "step_token_len",
    "prefix_generated_tokens",
    "p_end_onset",
    "p_end_mean",
    "p_end_max",
    "p_short",
    "log_p_end_onset",
    "log_p_short",
    "log_survival",
    "p_newline_onset",
    "p_newline_max",
    "p_think_end_onset",
    "p_think_end_max",
    *MULTI_HORIZON_BOUNDARY_FIELDS,
    "token_logprob_onset",
    "token_logprob_mean",
    "topk_entropy_onset",
    "topk_entropy_mean",
    "end_in_topk_onset",
    "end_in_topk_rate",
    "newline_in_topk_onset",
    "think_end_in_topk_onset",
    "lookahead_token_count",
    "lookahead_text",
]


PROBLEM_FIELDS = [
    "dataset",
    "problem_id",
    "gold_answer",
    "predicted_answer",
    "correct",
    "long_by_boundaries",
    "long_by_tokens",
    "wrong_long",
    "num_boundaries",
    "generated_tokens",
    "wall_time",
    "finish_reason",
    "p_short_mean",
    "p_short_median",
    "p_short_min",
    "p_short_max",
    "log_p_short_mean",
    "log_p_short_median",
    "log_p_short_var",
    "p_short_topk_censored_rate",
    "p_end_onset_topk_censored_rate",
]


WINDOW_FIELDS = [
    "dataset",
    "problem_id",
    "window_size",
    "used_boundaries",
    "problem_correct",
    "problem_wrong",
    "long_by_boundaries",
    "long_by_tokens",
    "wrong_long",
    "problem_num_boundaries",
    "problem_generated_tokens",
    "p_short_mean",
    "p_short_median",
    "p_short_var",
    "log_p_short_mean",
    "log_p_short_median",
    "log_p_short_var",
    "log_p_short_slope",
    "problem_z_mean",
    "problem_z_abs_mean",
    "problem_z_var",
    "global_z_mean",
    "global_z_abs_mean",
    "global_z_var",
    "end_in_topk_rate",
    "topk_entropy_mean",
    "token_logprob_mean",
    *MULTI_HORIZON_WINDOW_FIELDS,
]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_log(value: float, eps: float) -> float:
    return math.log(max(eps, min(1.0, float(value))))


def _mean(values: list[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _var(values: list[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    avg = sum(values) / len(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def _mad_sigma(values: list[float], *, sigma_floor: float) -> tuple[float, float]:
    center = _median(values)
    if center is None:
        return 0.0, sigma_floor
    deviations = [abs(value - center) for value in values if math.isfinite(value)]
    mad = _median(deviations) or 0.0
    return center, max(1.4826 * mad, sigma_floor)


def _slope(values: list[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    if len(values) < 2:
        return None
    xs = list(range(len(values)))
    mx = sum(xs) / len(xs)
    my = sum(values) / len(values)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    pairs = [(score, bool(label)) for score, label in zip(scores, labels) if math.isfinite(score)]
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank = 1
    pos_rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx) - 1) / 2.0
        pos_rank_sum += avg_rank * sum(1 for _, label in pairs[idx:end] if label)
        rank += end - idx
        idx = end
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _short_prob(probs: list[float]) -> float:
    product = 1.0
    for prob in probs:
        product *= max(0.0, 1.0 - min(1.0, prob))
    return 1.0 - product


def _candidate_single_token_ids(tokenizer: Any, texts: list[str]) -> set[int]:
    ids: set[int] = set()
    for text in texts:
        for variant in {text, " " + text}:
            try:
                encoded = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                encoded = []
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
    return ids


def _scan_vocab_token_ids(tokenizer: Any, predicate, *, limit: int = 200000) -> set[int]:
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    ids: set[int] = set()
    for _, token_id in vocab.items():
        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            continue
        if token_id < 0 or token_id > limit:
            continue
        try:
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            continue
        if predicate(decoded):
            ids.add(token_id)
    return ids


def _token_id_set_tensor(torch, token_ids: set[int], *, device: Any):
    if not token_ids:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.tensor(sorted(token_ids), dtype=torch.long, device=device)


def _set_probability(torch, logits, token_ids_tensor) -> float:
    if token_ids_tensor.numel() == 0:
        return 0.0
    denom = torch.logsumexp(logits, dim=-1)
    selected = logits.index_select(0, token_ids_tensor)
    return float(torch.exp(torch.logsumexp(selected, dim=-1) - denom).item())


def _topk_entropy_and_hits(torch, logits, *, topk: int, id_sets: dict[str, set[int]]) -> dict[str, Any]:
    k = min(int(topk), int(logits.numel()))
    values, indices = torch.topk(logits, k=k)
    probs = torch.softmax(values, dim=-1)
    entropy = float((-(probs * torch.log(torch.clamp(probs, min=1e-30))).sum()).item())
    index_set = {int(value) for value in indices.detach().cpu().tolist()}

    def hit(name: str) -> bool:
        ids = id_sets[name]
        if not ids:
            return False
        return any(value in ids for value in index_set)

    return {
        "topk_entropy": entropy,
        "end_in_topk": hit("step_end"),
        "newline_in_topk": hit("newline"),
        "think_end_in_topk": hit("think_end"),
    }


def _sample_next_token(torch, logits, *, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    scaled = logits / max(temperature, 1e-6)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        if mask.numel() > 1:
            mask[1:] = mask[:-1].clone()
            mask[0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(sorted_indices[sampled].item())
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _new_boundary_token_indices(tokenizer: Any, token_ids: list[int]) -> list[int]:
    indices: list[int] = []
    text = ""
    search_from = 0
    for idx, token_id in enumerate(token_ids):
        try:
            piece = tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            piece = ""
        text += piece
        while True:
            found = text.find("\n\n", search_from)
            if found < 0:
                break
            indices.append(idx)
            search_from = found + 2
    return indices


def _token_text(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _load_hf_model_and_tokenizer(args: argparse.Namespace, config: BPAConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path or config.slm_model_path
    tokenizer_path = args.tokenizer_path or config.slm_tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=config.trust_remote_code, use_fast=True)

    dtype = None
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    kwargs = {"trust_remote_code": config.trust_remote_code}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if args.device_map:
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not args.device_map:
        device = torch.device(args.device)
        model.to(device)
    model.eval()
    return torch, model, tokenizer


def _generate_problem(args: argparse.Namespace, config: BPAConfig, problem, torch, model, tokenizer, token_sets):
    device = next(model.parameters()).device
    rendered = render_for_continuation(problem.problem_text, "", tokenizer)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(prompt_ids) >= config.max_model_len:
        raise RuntimeError(f"Prompt length {len(prompt_ids)} exceeds max_model_len={config.max_model_len}")

    id_tensors = {
        name: _token_id_set_tensor(torch, ids, device=device)
        for name, ids in token_sets.items()
    }
    id_sets = {name: set(ids) for name, ids in token_sets.items()}

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    stats: list[dict[str, Any]] = []
    generated_ids: list[int] = []
    finish_reason = "length"
    start_time = time.time()
    eos_ids = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        eos_ids.add(eos)
    elif eos is not None:
        try:
            eos_ids.update(int(value) for value in eos if value is not None)
        except TypeError:
            pass

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :].squeeze(0).float()
        for token_idx in range(args.max_new_tokens):
            if len(prompt_ids) + token_idx >= config.max_model_len:
                finish_reason = "context_budget"
                break
            denom = torch.logsumexp(logits, dim=-1)
            p_end = _set_probability(torch, logits, id_tensors["step_end"])
            p_newline = _set_probability(torch, logits, id_tensors["newline"])
            p_think_end = _set_probability(torch, logits, id_tensors["think_end"])
            topk = _topk_entropy_and_hits(torch, logits, topk=args.topk_for_censoring, id_sets=id_sets)
            next_id = _sample_next_token(torch, logits, temperature=args.temperature, top_p=args.top_p)
            token_logprob = float((logits[next_id] - denom).item())
            stats.append(
                {
                    "p_end": p_end,
                    "p_newline": p_newline,
                    "p_think_end": p_think_end,
                    "token_logprob": token_logprob,
                    **topk,
                }
            )
            generated_ids.append(next_id)
            if next_id in eos_ids:
                finish_reason = "eos"
                break
            next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
            outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :].squeeze(0).float()

    wall_time = time.time() - start_time
    generated_text = _token_text(tokenizer, generated_ids)
    answer = extract_answer(generated_text)
    correct = benchmark_eval_match(answer, problem.gold_answer, args.dataset) if problem.gold_answer is not None else None
    boundary_indices = _new_boundary_token_indices(tokenizer, generated_ids)
    long_by_boundaries = len(boundary_indices) >= args.long_tail_boundaries
    long_by_tokens = len(generated_ids) >= args.long_tail_tokens
    wrong_long = bool(correct is False and (long_by_boundaries or long_by_tokens))

    problem_row = {
        "dataset": args.dataset,
        "problem_id": problem.problem_id,
        "gold_answer": problem.gold_answer,
        "predicted_answer": answer,
        "correct": correct,
        "long_by_boundaries": long_by_boundaries,
        "long_by_tokens": long_by_tokens,
        "wrong_long": wrong_long,
        "num_boundaries": len(boundary_indices),
        "generated_tokens": len(generated_ids),
        "wall_time": wall_time,
        "finish_reason": finish_reason,
    }

    boundary_rows: list[dict[str, Any]] = []
    prev_boundary_idx = -1
    for boundary_idx, token_complete_idx in enumerate(boundary_indices):
        onset = token_complete_idx + 1
        lookahead = stats[onset : min(len(stats), onset + args.lookahead_tokens)]
        if not lookahead:
            continue
        p_end_values = [row["p_end"] for row in lookahead]
        p_newline_values = [row["p_newline"] for row in lookahead]
        p_think_end_values = [row["p_think_end"] for row in lookahead]
        token_lps = [row["token_logprob"] for row in lookahead]
        entropies = [row["topk_entropy"] for row in lookahead]
        generated_slice = generated_ids[onset : min(len(generated_ids), onset + args.lookahead_tokens)]
        p_short = _short_prob(p_end_values)
        log_survival = sum(math.log(max(args.epsilon, 1.0 - min(1.0, value))) for value in p_end_values)
        row = {
                "dataset": args.dataset,
                "problem_id": problem.problem_id,
                "boundary_idx": boundary_idx,
                "boundary_token_idx": token_complete_idx,
                "problem_correct": correct,
                "problem_long_by_boundaries": long_by_boundaries,
                "problem_long_by_tokens": long_by_tokens,
                "problem_wrong_long": wrong_long,
                "problem_num_boundaries": len(boundary_indices),
                "problem_generated_tokens": len(generated_ids),
                "problem_wall_time": wall_time,
                "step_token_len": token_complete_idx - prev_boundary_idx,
                "prefix_generated_tokens": onset,
                "p_end_onset": p_end_values[0],
                "p_end_mean": _mean(p_end_values),
                "p_end_max": max(p_end_values),
                "p_short": p_short,
                "log_p_end_onset": _safe_log(p_end_values[0], args.epsilon),
                "log_p_short": _safe_log(p_short, args.epsilon),
                "log_survival": log_survival,
                "p_newline_onset": p_newline_values[0],
                "p_newline_max": max(p_newline_values),
                "p_think_end_onset": p_think_end_values[0],
                "p_think_end_max": max(p_think_end_values),
                "token_logprob_onset": token_lps[0],
                "token_logprob_mean": _mean(token_lps),
                "topk_entropy_onset": entropies[0],
                "topk_entropy_mean": _mean(entropies),
                "end_in_topk_onset": lookahead[0]["end_in_topk"],
                "end_in_topk_rate": sum(1 for row in lookahead if row["end_in_topk"]) / len(lookahead),
                "newline_in_topk_onset": lookahead[0]["newline_in_topk"],
                "think_end_in_topk_onset": lookahead[0]["think_end_in_topk"],
                "lookahead_token_count": len(lookahead),
                "lookahead_text": _token_text(tokenizer, generated_slice).replace("\r", "\\r").replace("\n", "\\n"),
            }
        for horizon in args.lookahead_horizons:
            horizon_rows = stats[onset : min(len(stats), onset + horizon)]
            horizon_p_end = [item["p_end"] for item in horizon_rows]
            if horizon_p_end:
                horizon_p_short = _short_prob(horizon_p_end)
                row[f"p_end_mean_h{horizon}"] = _mean(horizon_p_end)
                row[f"p_end_max_h{horizon}"] = max(horizon_p_end)
                row[f"p_short_h{horizon}"] = horizon_p_short
                row[f"log_p_short_h{horizon}"] = _safe_log(horizon_p_short, args.epsilon)
                row[f"end_in_topk_rate_h{horizon}"] = sum(1 for item in horizon_rows if item["end_in_topk"]) / len(horizon_rows)
        boundary_rows.append(row)
        prev_boundary_idx = token_complete_idx

    p_short_values = [_float(row["p_short"]) for row in boundary_rows]
    log_p_short_values = [_float(row["log_p_short"]) for row in boundary_rows]
    problem_row.update(
        {
            "p_short_mean": _mean(p_short_values),
            "p_short_median": _median(p_short_values),
            "p_short_min": min(p_short_values) if p_short_values else None,
            "p_short_max": max(p_short_values) if p_short_values else None,
            "log_p_short_mean": _mean(log_p_short_values),
            "log_p_short_median": _median(log_p_short_values),
            "log_p_short_var": _var(log_p_short_values),
            "p_short_topk_censored_rate": _mean([1.0 if _float(row["end_in_topk_rate"]) == 0.0 else 0.0 for row in boundary_rows]),
            "p_end_onset_topk_censored_rate": _mean([0.0 if row["end_in_topk_onset"] else 1.0 for row in boundary_rows]),
        }
    )
    return problem_row, boundary_rows, generated_text


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _add_window_features(args: argparse.Namespace, boundary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_problem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in boundary_rows:
        rows_by_problem[int(row["problem_id"])].append(row)
    for rows in rows_by_problem.values():
        rows.sort(key=lambda row: int(row["boundary_idx"]))

    all_logs = [_float(row["log_p_short"], float("nan")) for row in boundary_rows]
    global_center, global_sigma = _mad_sigma(all_logs, sigma_floor=args.robust_sigma_floor)

    problem_baselines: dict[int, tuple[float, float]] = {}
    for problem_id, rows in rows_by_problem.items():
        baseline_values = [_float(row["log_p_short"], float("nan")) for row in rows[: args.baseline_boundaries]]
        problem_baselines[problem_id] = _mad_sigma(baseline_values, sigma_floor=args.robust_sigma_floor)

    window_rows: list[dict[str, Any]] = []
    for problem_id, rows in rows_by_problem.items():
        problem_center, problem_sigma = problem_baselines[problem_id]
        for window_size in args.window_sizes:
            if len(rows) < args.min_window_boundaries:
                continue
            used = rows[: min(window_size, len(rows))]
            if len(used) < args.min_window_boundaries:
                continue
            logs = [_float(row["log_p_short"], float("nan")) for row in used]
            p_short = [_float(row["p_short"], float("nan")) for row in used]
            problem_z = [(value - problem_center) / problem_sigma for value in logs if math.isfinite(value)]
            global_z = [(value - global_center) / global_sigma for value in logs if math.isfinite(value)]
            first = used[0]
            window_rows.append(
                row := {
                    "dataset": args.dataset,
                    "problem_id": problem_id,
                    "window_size": window_size,
                    "used_boundaries": len(used),
                    "problem_correct": first["problem_correct"],
                    "problem_wrong": first["problem_correct"] is False,
                    "long_by_boundaries": first["problem_long_by_boundaries"],
                    "long_by_tokens": first["problem_long_by_tokens"],
                    "wrong_long": first["problem_wrong_long"],
                    "problem_num_boundaries": first["problem_num_boundaries"],
                    "problem_generated_tokens": first["problem_generated_tokens"],
                    "p_short_mean": _mean(p_short),
                    "p_short_median": _median(p_short),
                    "p_short_var": _var(p_short),
                    "log_p_short_mean": _mean(logs),
                    "log_p_short_median": _median(logs),
                    "log_p_short_var": _var(logs),
                    "log_p_short_slope": _slope(logs),
                    "problem_z_mean": _mean(problem_z),
                    "problem_z_abs_mean": _mean([abs(value) for value in problem_z]),
                    "problem_z_var": _var(problem_z),
                    "global_z_mean": _mean(global_z),
                    "global_z_abs_mean": _mean([abs(value) for value in global_z]),
                    "global_z_var": _var(global_z),
                    "end_in_topk_rate": _mean([_float(row["end_in_topk_rate"]) for row in used]),
                    "topk_entropy_mean": _mean([_float(row["topk_entropy_mean"], float("nan")) for row in used]),
                    "token_logprob_mean": _mean([_float(row["token_logprob_mean"], float("nan")) for row in used]),
                }
            )
            for horizon in args.lookahead_horizons:
                row[f"p_short_h{horizon}_mean"] = _mean([_float(item.get(f"p_short_h{horizon}"), float("nan")) for item in used])
                row[f"p_short_h{horizon}_median"] = _median([_float(item.get(f"p_short_h{horizon}"), float("nan")) for item in used])
                row[f"log_p_short_h{horizon}_mean"] = _mean([_float(item.get(f"log_p_short_h{horizon}"), float("nan")) for item in used])
                row[f"log_p_short_h{horizon}_median"] = _median([_float(item.get(f"log_p_short_h{horizon}"), float("nan")) for item in used])
                row[f"end_in_topk_rate_h{horizon}"] = _mean([_float(item.get(f"end_in_topk_rate_h{horizon}"), float("nan")) for item in used])
    return window_rows


def _summarize(args: argparse.Namespace, problem_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]], token_info: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "num_problems": len(problem_rows),
        "num_boundaries": len(boundary_rows),
        "num_windows": len(window_rows),
        "accuracy": _mean([1.0 if row.get("correct") else 0.0 for row in problem_rows]),
        "avg_generated_tokens": _mean([_float(row.get("generated_tokens")) for row in problem_rows]),
        "avg_num_boundaries": _mean([_float(row.get("num_boundaries")) for row in problem_rows]),
        "token_sets": token_info,
        "boundary_p_short_topk_censored_rate": _mean([1.0 if _float(row.get("end_in_topk_rate")) == 0.0 else 0.0 for row in boundary_rows]),
        "boundary_p_end_onset_topk_censored_rate": _mean([0.0 if row.get("end_in_topk_onset") else 1.0 for row in boundary_rows]),
    }
    metrics = [
        "p_short_mean",
        "p_short_median",
        "log_p_short_mean",
        "log_p_short_median",
        "log_p_short_slope",
        "problem_z_abs_mean",
        "problem_z_var",
        "global_z_abs_mean",
        "global_z_var",
        "end_in_topk_rate",
        "topk_entropy_mean",
        "token_logprob_mean",
    ]
    for horizon in args.lookahead_horizons:
        metrics.extend(
            [
                f"p_short_h{horizon}_mean",
                f"p_short_h{horizon}_median",
                f"log_p_short_h{horizon}_mean",
                f"log_p_short_h{horizon}_median",
                f"end_in_topk_rate_h{horizon}",
            ]
        )
    auc_rows: list[dict[str, Any]] = []
    for window_size in args.window_sizes:
        scoped = [row for row in window_rows if int(row["window_size"]) == window_size]
        for metric in metrics:
            scores = [_float(row.get(metric), float("nan")) for row in scoped]
            for target in ["problem_wrong", "long_by_boundaries", "long_by_tokens", "wrong_long"]:
                value = _auc(scores, [bool(row.get(target)) for row in scoped])
                auc_rows.append(
                    {
                        "window_size": window_size,
                        "metric": metric,
                        "target": target,
                        "auc_high_score": value,
                        "auc_abs_from_random": abs(value - 0.5) if value is not None else None,
                    }
                )
    summary["auc"] = auc_rows
    return summary


def _write_plots(output_dir: Path, problem_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(repr(exc), encoding="utf-8")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    correct_values = [_float(row["log_p_short"]) for row in boundary_rows if row["problem_correct"] is True]
    wrong_values = [_float(row["log_p_short"]) for row in boundary_rows if row["problem_correct"] is False]
    plt.figure(figsize=(8, 4.8))
    plt.hist(correct_values, bins=50, alpha=0.55, label="correct")
    plt.hist(wrong_values, bins=50, alpha=0.55, label="wrong")
    plt.xlabel("log p_short exact")
    plt.ylabel("boundary count")
    plt.title("Exact P(end) trajectory values by final correctness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "log_p_short_by_correctness.png", dpi=170)
    plt.close()

    for target in ["problem_wrong", "wrong_long"]:
        auc_rows = [row for row in summary.get("auc", []) if row["target"] == target and row["window_size"] == 50 and row["auc_high_score"] is not None]
        auc_rows.sort(key=lambda row: abs(row["auc_high_score"] - 0.5), reverse=True)
        auc_rows = auc_rows[:12]
        plt.figure(figsize=(10, 4.8))
        plt.bar(range(len(auc_rows)), [row["auc_high_score"] for row in auc_rows])
        plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
        plt.xticks(range(len(auc_rows)), [row["metric"] for row in auc_rows], rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.title(f"Phase 0 window-50 AUC: high metric predicts {target}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"window50_auc_{target}.png", dpi=170)
        plt.close()

    selected = sorted(problem_rows, key=lambda row: (_float(row.get("wall_time")), _float(row.get("num_boundaries"))), reverse=True)[:8]
    selected_ids = {int(row["problem_id"]) for row in selected}
    for metric in ["log_p_short", "p_short", "end_in_topk_rate"]:
        plt.figure(figsize=(10, 5))
        for problem_id in selected_ids:
            rows = [row for row in boundary_rows if int(row["problem_id"]) == problem_id]
            rows.sort(key=lambda row: int(row["boundary_idx"]))
            if not rows:
                continue
            correct = rows[0]["problem_correct"]
            label = f"p{problem_id} " + ("C" if correct else "W")
            plt.plot([int(row["boundary_idx"]) for row in rows], [_float(row[metric]) for row in rows], linewidth=1, label=label)
        plt.xlabel("boundary idx")
        plt.ylabel(metric)
        plt.title(f"Phase 0 trajectories: {metric}")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"trajectory_{metric}.png", dpi=170)
        plt.close()


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    config = BPAConfig.from_json(args.config)
    if args.max_model_len is not None:
        config = config.with_updates(max_model_len=args.max_model_len)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "problems").mkdir(exist_ok=True)

    torch, model, tokenizer = _load_hf_model_and_tokenizer(args, config)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    step_end_ids = _candidate_single_token_ids(tokenizer, ["\n\n", "\r\n\r\n", "\n\n\n", "\n\n\n\n"])
    newline_ids = _candidate_single_token_ids(tokenizer, ["\n", "\r\n"])
    think_end_ids = _candidate_single_token_ids(tokenizer, ["</think>"])
    if args.scan_vocab_for_boundary_tokens:
        step_end_ids |= _scan_vocab_token_ids(tokenizer, lambda value: "\n\n" in value or "\r\n\r\n" in value)
        newline_ids |= _scan_vocab_token_ids(tokenizer, lambda value: "\n" in value or "\r\n" in value)
        think_end_ids |= _scan_vocab_token_ids(tokenizer, lambda value: "</think" in value)
    token_sets = {"step_end": step_end_ids, "newline": newline_ids, "think_end": think_end_ids}
    token_info = {
        "step_end_token_ids": sorted(step_end_ids),
        "newline_token_ids": sorted(newline_ids),
        "think_end_token_ids": sorted(think_end_ids),
        "lookahead_tokens": args.lookahead_tokens,
        "lookahead_horizons": args.lookahead_horizons,
        "topk_for_censoring": args.topk_for_censoring,
        "exact_probability": True,
    }
    (output_dir / "token_sets.json").write_text(json.dumps(token_info, indent=2), encoding="utf-8")

    problem_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for problem in tqdm(problems, desc="phase0-pend"):
        try:
            problem_row, problem_boundary_rows, generated_text = _generate_problem(
                args,
                config,
                problem,
                torch,
                model,
                tokenizer,
                token_sets,
            )
        except Exception as exc:
            problem_row = {
                "dataset": args.dataset,
                "problem_id": problem.problem_id,
                "gold_answer": problem.gold_answer,
                "predicted_answer": None,
                "correct": None,
                "long_by_boundaries": None,
                "long_by_tokens": None,
                "wrong_long": None,
                "num_boundaries": 0,
                "generated_tokens": 0,
                "wall_time": 0,
                "finish_reason": f"error:{exc}",
            }
            problem_boundary_rows = []
            generated_text = ""
        problem_rows.append(problem_row)
        boundary_rows.extend(problem_boundary_rows)
        (output_dir / "problems" / f"{problem.problem_id}.txt").write_text(generated_text, encoding="utf-8")
        (output_dir / "problems" / f"{problem.problem_id}.json").write_text(
            json.dumps(problem_row, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    window_rows = _add_window_features(args, boundary_rows)
    summary = _summarize(args, problem_rows, boundary_rows, window_rows, token_info)
    _write_csv(output_dir / "phase0_problems.csv", problem_rows, PROBLEM_FIELDS)
    _write_csv(output_dir / "phase0_boundaries.csv", boundary_rows, BOUNDARY_FIELDS)
    _write_csv(output_dir / "phase0_windows.csv", window_rows, WINDOW_FIELDS)
    (output_dir / "phase0_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_plots(output_dir, problem_rows, boundary_rows, window_rows, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 diagnostic for exact P(end) trajectories on pure SLM traces.")
    parser.add_argument("--config", default="configs/bpa_default.json")
    parser.add_argument("--dataset", default="aime25")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--model-path", default=None, help="Override config.slm_model_path.")
    parser.add_argument("--tokenizer-path", default=None, help="Override config.slm_tokenizer_path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default=None, help="Optional transformers device_map, e.g. auto.")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional context-length override for this diagnostic run.")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead-tokens", type=int, default=8)
    parser.add_argument("--lookahead-horizons", type=int, nargs="+", default=MULTI_HORIZON_VALUES)
    parser.add_argument("--topk-for-censoring", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--baseline-boundaries", type=int, default=8)
    parser.add_argument("--robust-sigma-floor", type=float, default=0.5)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--min-window-boundaries", type=int, default=8)
    parser.add_argument("--long-tail-boundaries", type=int, default=300)
    parser.add_argument("--long-tail-tokens", type=int, default=6000)
    parser.add_argument("--scan-vocab-for-boundary-tokens", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.lookahead_horizons = sorted(
        horizon
        for horizon in set(int(value) for value in args.lookahead_horizons)
        if horizon in set(MULTI_HORIZON_VALUES)
    )
    if args.lookahead_tokens not in args.lookahead_horizons and args.lookahead_tokens in set(MULTI_HORIZON_VALUES):
        args.lookahead_horizons.append(args.lookahead_tokens)
        args.lookahead_horizons = sorted(args.lookahead_horizons)
    run(args)


if __name__ == "__main__":
    main()
