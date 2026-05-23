#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_REFLECTION_STRINGS = [
    "Wait",
    " wait",
    "But",
    " but",
    "However",
    " however",
    "Alternatively",
    " alternatively",
    "Hmm",
    " hmm",
    "actually",
    " check",
    " verify",
    "reconsider",
    "again",
]


@dataclass
class StepSignal:
    problem_id: str
    step_id: int
    progress: float
    forced_close: bool
    correct: bool | None
    entropy_topk: float | None
    margin_topk: float | None
    top1_prob_topk: float | None
    pref_topk: float | None
    h_topk: float | None
    h_no_ref_topk: float | None
    delta_h_ref_topk: float | None
    reflection_rank_min: int | None
    generated_starts_with_reflection: bool
    generated_contains_reflection: bool
    token_count: int
    remaining_tokens_after_step: int
    text: str


def _truthy(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if value is True:
        return True
    if value is False:
        return False
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_summary(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "summary.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row.get("problem_id") or row.get("id") or ""): row for row in csv.DictReader(f)}


def _problem_sort_key(path: Path) -> tuple[int, str]:
    return (0, f"{int(path.name):09d}") if path.name.isdigit() else (1, path.name)


def _steps_path(problem_dir: Path) -> Path | None:
    preferred = problem_dir / f"{problem_dir.name}.steps.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(problem_dir.glob("*.steps.jsonl"))
    return matches[0] if matches else None


def _token_entropy(probs: list[float]) -> float | None:
    vals = [p for p in probs if p > 0.0 and math.isfinite(p)]
    if not vals:
        return None
    return -sum(p * math.log(p + 1e-12) for p in vals)


def _renorm_entropy_without_ref(probs: list[float], is_ref: list[bool]) -> float | None:
    keep = [p for p, ref in zip(probs, is_ref) if not ref and p > 0.0 and math.isfinite(p)]
    total = sum(keep)
    if total <= 0.0:
        return None
    renorm = [p / total for p in keep]
    return _token_entropy(renorm)


def _margin(probs: list[float]) -> float | None:
    if len(probs) < 2:
        return None
    ordered = sorted(probs, reverse=True)
    return ordered[0] - ordered[1]


def _safe_mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _safe_corr(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _load_tokenizer(tokenizer_path: str | None, chat_template_path: str | None = None):
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    if chat_template_path:
        tok.chat_template = Path(chat_template_path).read_text(encoding="utf-8")
    return tok


def reflection_token_ids(tokenizer: Any | None, reflection_strings: list[str]) -> dict[str, list[int]]:
    if tokenizer is None:
        return {}
    out: dict[str, list[int]] = {}
    for text in reflection_strings:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        out[text] = [int(x) for x in ids]
    return out


def _single_token_ref_ids(mapping: dict[str, list[int]]) -> set[int]:
    return {ids[0] for ids in mapping.values() if len(ids) == 1}


def _token_matches_ref_text(token_text: str, reflection_strings: list[str]) -> bool:
    return token_text in reflection_strings


def _text_starts_with_reflection(text: str, reflection_strings: list[str]) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(s.strip()) for s in reflection_strings)


def _text_contains_reflection(text: str, reflection_strings: list[str]) -> bool:
    padded = f" {text} "
    needles = {s.strip() for s in reflection_strings}
    return any(f" {s} " in padded or padded.lstrip().startswith(f"{s} ") for s in needles if s)


def load_checkpoint_signals(
    root: Path,
    *,
    reflection_strings: list[str],
    tokenizer: Any | None,
    late_window_fraction: float,
) -> tuple[list[StepSignal], dict[str, Any]]:
    summary = _read_summary(root)
    ref_mapping = reflection_token_ids(tokenizer, reflection_strings)
    ref_single_ids = _single_token_ref_ids(ref_mapping)
    used_tokenizer_mapping = bool(ref_single_ids)

    signals: list[StepSignal] = []
    for problem_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=_problem_sort_key):
        steps_file = _steps_path(problem_dir)
        if steps_file is None:
            continue
        pid = problem_dir.name
        rows = [r for r in _read_jsonl(steps_file) if not r.get("is_final_answer")]
        scored_rows = [r for r in rows if (r.get("extra") or {}).get("confidence")]
        total_visible_tokens = sum(int(r.get("token_count") or 0) for r in rows)
        token_prefix = 0
        row_summary = summary.get(pid, {})
        forced_close = str(row_summary.get("stop_reason") or "").endswith("_forced_close_think")
        correct = _truthy(row_summary.get("correct"))
        scored_count = len(scored_rows)
        scored_index = 0
        for row in rows:
            text = str(row.get("text") or "")
            token_count = int(row.get("token_count") or 0)
            token_prefix += token_count
            conf = (row.get("extra") or {}).get("confidence") or {}
            if not conf:
                continue
            scored_index += 1
            top_ids = [int(x) for x in conf.get("top_ids") or []]
            top_tokens = [str(x) for x in conf.get("top_tokens") or []]
            probs = [_float(x) for x in conf.get("top_probs") or []]
            probs = [float(p) for p in probs if p is not None]
            if len(probs) != len(top_ids):
                probs = probs[: len(top_ids)]
            is_ref: list[bool] = []
            for idx, token_id in enumerate(top_ids[: len(probs)]):
                match_by_id = token_id in ref_single_ids if used_tokenizer_mapping else False
                match_by_text = (
                    _token_matches_ref_text(top_tokens[idx], reflection_strings)
                    if idx < len(top_tokens)
                    else False
                )
                is_ref.append(match_by_id or match_by_text)
            pref = sum(p for p, ref in zip(probs, is_ref) if ref) if probs else None
            h = _token_entropy(probs)
            h_no_ref = _renorm_entropy_without_ref(probs, is_ref) if probs else None
            ref_ranks = [i + 1 for i, ref in enumerate(is_ref) if ref]
            signals.append(
                StepSignal(
                    problem_id=pid,
                    step_id=int(row.get("step_id") or scored_index),
                    progress=(scored_index - 1) / max(1, scored_count - 1),
                    forced_close=forced_close,
                    correct=correct,
                    entropy_topk=_float(conf.get("norm_entropy")),
                    margin_topk=_margin(probs),
                    top1_prob_topk=probs[0] if probs else None,
                    pref_topk=pref,
                    h_topk=h,
                    h_no_ref_topk=h_no_ref,
                    delta_h_ref_topk=(h - h_no_ref) if h is not None and h_no_ref is not None else None,
                    reflection_rank_min=min(ref_ranks) if ref_ranks else None,
                    generated_starts_with_reflection=_text_starts_with_reflection(text, reflection_strings),
                    generated_contains_reflection=_text_contains_reflection(text, reflection_strings),
                    token_count=token_count,
                    remaining_tokens_after_step=max(0, total_visible_tokens - token_prefix),
                    text=text,
                )
            )

    metadata = {
        "reflection_strings": reflection_strings,
        "reflection_token_id_mapping": ref_mapping,
        "single_token_ref_ids": sorted(ref_single_ids),
        "probability_note": (
            "checkpoint mode uses logged top-k probabilities. In current SARR logs, top_probs are normalized within top-k, "
            "so pref_topk and delta_h_ref_topk are top-k-local approximations, not full-vocabulary probabilities."
        ),
        "late_window_fraction": late_window_fraction,
    }
    return signals, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _signal_dict(s: StepSignal) -> dict[str, Any]:
    return {
        "problem_id": s.problem_id,
        "step_id": s.step_id,
        "progress": s.progress,
        "forced_close": s.forced_close,
        "correct": s.correct,
        "entropy_topk": s.entropy_topk,
        "margin_topk": s.margin_topk,
        "top1_prob_topk": s.top1_prob_topk,
        "pref_topk": s.pref_topk,
        "h_topk": s.h_topk,
        "h_no_ref_topk": s.h_no_ref_topk,
        "delta_h_ref_topk": s.delta_h_ref_topk,
        "reflection_rank_min": s.reflection_rank_min,
        "generated_starts_with_reflection": s.generated_starts_with_reflection,
        "generated_contains_reflection": s.generated_contains_reflection,
        "token_count": s.token_count,
        "remaining_tokens_after_step": s.remaining_tokens_after_step,
    }


def aggregate_by_progress(signals: list[StepSignal], bins: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[StepSignal]] = defaultdict(list)
    for s in signals:
        group = "budget-hit" if s.forced_close else "normal-end"
        idx = min(bins - 1, max(0, int(s.progress * bins)))
        buckets[(group, idx)].append(s)
    rows: list[dict[str, Any]] = []
    for group in ["budget-hit", "normal-end"]:
        for idx in range(bins):
            bucket = buckets.get((group, idx), [])
            rows.append(
                {
                    "group": group,
                    "bin": idx,
                    "progress_mid": (idx + 0.5) / bins,
                    "n": len(bucket),
                    "pref_topk_mean": _safe_mean([s.pref_topk for s in bucket]),
                    "entropy_topk_mean": _safe_mean([s.entropy_topk for s in bucket]),
                    "margin_topk_mean": _safe_mean([s.margin_topk for s in bucket]),
                    "delta_h_ref_topk_mean": _safe_mean([s.delta_h_ref_topk for s in bucket]),
                    "reflection_topk_hit_rate": (
                        sum(1 for s in bucket if (s.pref_topk or 0.0) > 0.0) / len(bucket)
                        if bucket
                        else None
                    ),
                }
            )
    return rows


def aggregate_problem_level(signals: list[StepSignal], late_window_fraction: float) -> list[dict[str, Any]]:
    by_problem: dict[str, list[StepSignal]] = defaultdict(list)
    for s in signals:
        by_problem[s.problem_id].append(s)
    rows: list[dict[str, Any]] = []
    for pid, items in sorted(by_problem.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        items = sorted(items, key=lambda s: s.step_id)
        cutoff = max(0.0, 1.0 - late_window_fraction)
        late = [s for s in items if s.progress >= cutoff]
        rows.append(
            {
                "problem_id": pid,
                "forced_close": items[0].forced_close if items else None,
                "correct": items[0].correct if items else None,
                "checkpoint_count": len(items),
                "pref_topk_mean": _safe_mean([s.pref_topk for s in items]),
                "pref_topk_late_mean": _safe_mean([s.pref_topk for s in late]),
                "pref_topk_max": max([s.pref_topk or 0.0 for s in items], default=None),
                "entropy_topk_mean": _safe_mean([s.entropy_topk for s in items]),
                "entropy_topk_late_mean": _safe_mean([s.entropy_topk for s in late]),
                "margin_topk_late_mean": _safe_mean([s.margin_topk for s in late]),
                "delta_h_ref_topk_mean": _safe_mean([s.delta_h_ref_topk for s in items]),
                "delta_h_ref_topk_late_mean": _safe_mean([s.delta_h_ref_topk for s in late]),
                "remaining_corr_pref": _safe_corr(
                    [s.pref_topk for s in items],
                    [float(s.remaining_tokens_after_step) for s in items],
                ),
                "reflection_start_step_count": sum(1 for s in items if s.generated_starts_with_reflection),
                "reflection_topk_hit_count": sum(1 for s in items if (s.pref_topk or 0.0) > 0.0),
            }
        )
    return rows


def event_aligned(signals: list[StepSignal], window_before: int, window_after: int) -> list[dict[str, Any]]:
    by_problem: dict[str, list[StepSignal]] = defaultdict(list)
    for s in signals:
        by_problem[s.problem_id].append(s)
    aligned: dict[int, list[StepSignal]] = defaultdict(list)
    for items in by_problem.values():
        items = sorted(items, key=lambda s: s.step_id)
        index_by_step = {s.step_id: i for i, s in enumerate(items)}
        event_steps = [
            s.step_id
            for s in items
            if s.generated_starts_with_reflection and (s.pref_topk or 0.0) > 0.0
        ]
        for event_step in event_steps:
            center = index_by_step[event_step]
            for offset in range(-window_before, window_after + 1):
                idx = center + offset
                if 0 <= idx < len(items):
                    aligned[offset].append(items[idx])
    rows: list[dict[str, Any]] = []
    for offset in range(-window_before, window_after + 1):
        bucket = aligned.get(offset, [])
        rows.append(
            {
                "offset": offset,
                "n": len(bucket),
                "pref_topk_mean": _safe_mean([s.pref_topk for s in bucket]),
                "entropy_topk_mean": _safe_mean([s.entropy_topk for s in bucket]),
                "margin_topk_mean": _safe_mean([s.margin_topk for s in bucket]),
                "top1_prob_topk_mean": _safe_mean([s.top1_prob_topk for s in bucket]),
                "delta_h_ref_topk_mean": _safe_mean([s.delta_h_ref_topk for s in bucket]),
            }
        )
    return rows


def plot_progress(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"budget-hit": "#b23a48", "normal-end": "#2a9d8f"}
    for group in ["budget-hit", "normal-end"]:
        subset = [r for r in rows if r["group"] == group and r["pref_topk_mean"] is not None]
        if not subset:
            continue
        ax.plot(
            [float(r["progress_mid"]) for r in subset],
            [float(r["pref_topk_mean"]) for r in subset],
            label=group,
            color=colors[group],
            linewidth=1.8,
        )
    ax.set_title("Reflection mass by reasoning progress")
    ax.set_xlabel("normalized reasoning progress")
    ax.set_ylabel("P_ref within logged top-k")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reflection_mass_by_progress.png", dpi=180)
    plt.close(fig)


def plot_event(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    metrics = [
        ("entropy_topk_mean", "entropy", "#365c8d"),
        ("margin_topk_mean", "margin", "#6a4c93"),
        ("top1_prob_topk_mean", "top1 prob", "#2a9d8f"),
        ("pref_topk_mean", "P_ref", "#b23a48"),
    ]
    for ax, (field, label, color) in zip(axes, metrics):
        subset = [r for r in rows if r[field] is not None]
        ax.plot([int(r["offset"]) for r in subset], [float(r[field]) for r in subset], color=color, linewidth=1.6)
        ax.axvline(0, color="#222222", linestyle="--", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("checkpoint offset from generated reflection-start event")
    fig.suptitle("Event-aligned checkpoint signals around reflection starts", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "reflection_event_aligned_topk.png", dpi=180)
    plt.close(fig)


def plot_problem_scatter(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        x = row.get("pref_topk_late_mean")
        y = row.get("entropy_topk_late_mean")
        if x is None or y is None:
            continue
        color = "#b23a48" if row.get("forced_close") else "#2a9d8f"
        marker = "x" if row.get("correct") is False else "o"
        ax.scatter(float(x), float(y), color=color, marker=marker, s=60)
        ax.text(float(x), float(y), str(row["problem_id"]), fontsize=8, ha="left", va="bottom")
    ax.set_title("Late reflection mass vs late entropy")
    ax.set_xlabel("late mean P_ref within top-k")
    ax.set_ylabel("late mean entropy")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "late_reflection_mass_vs_entropy.png", dpi=180)
    plt.close(fig)


def compute_full_vocab_checkpoints(
    root: Path,
    *,
    config_path: Path,
    reflection_strings: list[str],
    stride: int,
    max_checkpoints_per_problem: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from bpa.render import render_for_continuation
    from sarr_code import SARRConfig
    from sarr_code.engines import LocalTransformersSLM

    cfg = SARRConfig.from_json(config_path)
    slm = LocalTransformersSLM(cfg.slm, cfg.runtime).load()
    tokenizer = slm.ensure_tokenizer()
    ref_mapping = reflection_token_ids(tokenizer, reflection_strings)
    ref_ids = sorted(_single_token_ref_ids(ref_mapping))
    if not ref_ids:
        raise SystemExit("No reflection strings mapped to single tokenizer ids; cannot compute next-token P_ref.")

    summary = _read_summary(root)
    rows_out: list[dict[str, Any]] = []
    for problem_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=_problem_sort_key):
        steps_file = _steps_path(problem_dir)
        problem_file = problem_dir / f"{problem_dir.name}.problem.json"
        if steps_file is None or not problem_file.exists():
            continue
        pid = problem_dir.name
        problem_meta = json.loads(problem_file.read_text(encoding="utf-8"))
        problem_text = str((problem_meta.get("raw") or {}).get("question") or problem_meta.get("question") or "")
        if not problem_text:
            continue

        steps = [r for r in _read_jsonl(steps_file) if not r.get("is_final_answer")]
        scored_indices = [
            i
            for i, row in enumerate(steps)
            if row.get("generator") == "slm" and (row.get("extra") or {}).get("confidence")
        ]
        selected: set[int] = set(scored_indices[:: max(1, stride)])
        # Always keep the last 20% so plateau behavior is not missed.
        tail_start = int(len(scored_indices) * 0.8)
        selected.update(scored_indices[tail_start:])
        if max_checkpoints_per_problem > 0 and len(selected) > max_checkpoints_per_problem:
            ordered = sorted(selected)
            keep_every = max(1, math.ceil(len(ordered) / max_checkpoints_per_problem))
            selected = set(ordered[::keep_every])
            selected.add(ordered[-1])

        assistant_prefix = ""
        row_summary = summary.get(pid, {})
        forced_close = str(row_summary.get("stop_reason") or "").endswith("_forced_close_think")
        correct = _truthy(row_summary.get("correct"))
        for idx, row in enumerate(steps):
            assistant_prefix += str(row.get("text") or "")
            if idx not in selected:
                continue
            rendered = render_for_continuation(problem_text, assistant_prefix, tokenizer)
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to(slm.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(slm.device)
            with torch.inference_mode():
                outputs = slm.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            p_ref = probs[ref_ids].sum()
            entropy_full = -(probs * torch.log(probs + 1e-12)).sum()
            keep_probs = probs.clone()
            keep_probs[ref_ids] = 0.0
            keep_mass = keep_probs.sum()
            entropy_no_ref = None
            if float(keep_mass.detach().cpu()) > 0.0:
                renorm = keep_probs / keep_mass
                entropy_no_ref_tensor = -(renorm * torch.log(renorm + 1e-12)).sum()
                entropy_no_ref = float(entropy_no_ref_tensor.detach().cpu())
            top_probs, top_ids = torch.topk(probs, k=2)
            vocab_size = int(probs.shape[-1])
            rows_out.append(
                {
                    "problem_id": pid,
                    "step_id": int(row.get("step_id") or idx + 1),
                    "progress": idx / max(1, len(steps) - 1),
                    "forced_close": forced_close,
                    "correct": correct,
                    "p_ref_full": float(p_ref.detach().cpu()),
                    "h_full": float(entropy_full.detach().cpu()),
                    "h_full_norm": float((entropy_full / math.log(vocab_size)).detach().cpu()),
                    "h_no_ref": entropy_no_ref,
                    "h_no_ref_norm": (entropy_no_ref / math.log(vocab_size)) if entropy_no_ref is not None else None,
                    "delta_h_ref": (
                        float(entropy_full.detach().cpu()) - entropy_no_ref
                        if entropy_no_ref is not None
                        else None
                    ),
                    "top1_prob_full": float(top_probs[0].detach().cpu()),
                    "margin_full": float((top_probs[0] - top_probs[1]).detach().cpu()),
                    "top1_id": int(top_ids[0].detach().cpu()),
                    "top2_id": int(top_ids[1].detach().cpu()),
                    "prompt_tokens": int(input_ids.shape[-1]),
                }
            )
            del outputs, logits, probs

    metadata = {
        "reflection_strings": reflection_strings,
        "reflection_token_id_mapping": ref_mapping,
        "single_token_ref_ids": ref_ids,
        "probability_note": "full-logits mode uses full-vocabulary softmax at selected step checkpoints.",
        "stride": stride,
        "max_checkpoints_per_problem": max_checkpoints_per_problem,
    }
    return rows_out, metadata


def plot_full_progress(rows: list[dict[str, Any]], out_dir: Path, bins: int) -> None:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = "budget-hit" if row.get("forced_close") else "normal-end"
        idx = min(bins - 1, max(0, int(float(row["progress"]) * bins)))
        buckets[(group, idx)].append(row)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fields = [
        ("p_ref_full", "P_ref full", "#b23a48"),
        ("h_full_norm", "H_full norm", "#365c8d"),
        ("delta_h_ref", "Delta H_ref", "#6a4c93"),
    ]
    for ax, (field, label, color) in zip(axes, fields):
        for group, linestyle in [("budget-hit", "-"), ("normal-end", "--")]:
            xs: list[float] = []
            ys: list[float] = []
            for idx in range(bins):
                bucket = buckets.get((group, idx), [])
                value = _safe_mean([_float(r.get(field)) for r in bucket])
                if value is not None:
                    xs.append((idx + 0.5) / bins)
                    ys.append(value)
            if xs:
                ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.7, label=group)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("normalized reasoning progress")
    fig.suptitle("Full-vocabulary reflection signals by progress", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "reflection_full_vocab_by_progress.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze reflection-trigger probability mass from SARR step checkpoints. "
            "This mode uses logged top-k probability mass; use the output note to avoid treating it as full-vocab mass."
        )
    )
    parser.add_argument("--input-root", required=True, help="Experiment result directory with summary.csv and problem subdirs.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path for exact reflection token-id mapping.")
    parser.add_argument("--chat-template-path", default=None, help="Optional chat template path, recorded for tokenizer consistency.")
    parser.add_argument("--reflection-token", action="append", default=None, help="Override/add reflection strings; can be repeated.")
    parser.add_argument("--progress-bins", type=int, default=20)
    parser.add_argument("--late-window-fraction", type=float, default=0.20)
    parser.add_argument("--event-window-before", type=int, default=16)
    parser.add_argument("--event-window-after", type=int, default=32)
    parser.add_argument(
        "--full-logits-config",
        default=None,
        help="Optional SARR config path. If set, recompute full-vocabulary logits at selected checkpoints.",
    )
    parser.add_argument("--full-logits-stride", type=int, default=8)
    parser.add_argument(
        "--max-full-checkpoints-per-problem",
        type=int,
        default=120,
        help="Limit full-logits checkpoints per problem; 0 means no limit.",
    )
    args = parser.parse_args()

    root = Path(args.input_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "reflection_logit_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    reflection_strings = args.reflection_token or DEFAULT_REFLECTION_STRINGS
    tokenizer = _load_tokenizer(args.tokenizer_path, args.chat_template_path) if args.tokenizer_path else None
    signals, metadata = load_checkpoint_signals(
        root,
        reflection_strings=reflection_strings,
        tokenizer=tokenizer,
        late_window_fraction=args.late_window_fraction,
    )
    if not signals:
        raise SystemExit(f"No checkpoint confidence signals found under {root}")

    step_rows = [_signal_dict(s) for s in signals]
    progress_rows = aggregate_by_progress(signals, args.progress_bins)
    problem_rows = aggregate_problem_level(signals, args.late_window_fraction)
    event_rows = event_aligned(signals, args.event_window_before, args.event_window_after)

    write_csv(out_dir / "reflection_checkpoint_signals.csv", step_rows)
    write_csv(out_dir / "reflection_by_progress.csv", progress_rows)
    write_csv(out_dir / "reflection_problem_summary.csv", problem_rows)
    write_csv(out_dir / "reflection_event_aligned_topk.csv", event_rows)
    (out_dir / "reflection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_progress(progress_rows, out_dir)
    plot_event(event_rows, out_dir)
    plot_problem_scatter(problem_rows, out_dir)

    if args.full_logits_config:
        full_rows, full_metadata = compute_full_vocab_checkpoints(
            root,
            config_path=Path(args.full_logits_config),
            reflection_strings=reflection_strings,
            stride=args.full_logits_stride,
            max_checkpoints_per_problem=args.max_full_checkpoints_per_problem,
        )
        write_csv(out_dir / "reflection_full_checkpoint_signals.csv", full_rows)
        (out_dir / "reflection_full_metadata.json").write_text(
            json.dumps(full_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if full_rows:
            plot_full_progress(full_rows, out_dir, bins=args.progress_bins)

    forced = [r for r in problem_rows if r["forced_close"]]
    normal = [r for r in problem_rows if not r["forced_close"]]
    report = {
        "problem_count": len(problem_rows),
        "checkpoint_count": len(signals),
        "budget_hit_count": len(forced),
        "normal_end_count": len(normal),
        "mean_late_pref_topk_budget_hit": _safe_mean([r["pref_topk_late_mean"] for r in forced]),
        "mean_late_pref_topk_normal_end": _safe_mean([r["pref_topk_late_mean"] for r in normal]),
        "mean_late_entropy_budget_hit": _safe_mean([r["entropy_topk_late_mean"] for r in forced]),
        "mean_late_entropy_normal_end": _safe_mean([r["entropy_topk_late_mean"] for r in normal]),
        "mean_remaining_corr_pref": _safe_mean([r["remaining_corr_pref"] for r in problem_rows]),
    }
    (out_dir / "reflection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reflection] wrote: {out_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
