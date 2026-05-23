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


REFLECTION_STRINGS = {
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
}


@dataclass
class RPDIPoint:
    problem_id: str
    step_id: int
    progress: float
    phase: str
    forced_close: bool
    correct: bool | None
    entropy: float
    h_local: float
    h_global_problem: float
    h_global_cumulative: float
    rpdi_problem: float
    rpdi_cumulative: float
    margin: float | None
    pref_topk: float | None
    reflection_start: bool
    llm_switch: bool
    action: str
    remaining_tokens_after_step: int
    text: str


def _truthy(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if value is True:
        return True
    if value is False:
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
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


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _pstdev(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def _quantile(values: list[float | None], q: float) -> float | None:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    idx = q * (len(vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def _corr(xs: list[float | None], ys: list[float | None]) -> float | None:
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
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


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


def _phase(progress: float) -> str:
    if progress < 0.25:
        return "early"
    if progress >= 0.75:
        return "late"
    return "middle"


def _reflection_start(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(token.strip()) for token in REFLECTION_STRINGS if token.strip())


def _pref_topk(conf: dict[str, Any]) -> float | None:
    probs = conf.get("top_probs") or []
    tokens = conf.get("top_tokens") or []
    if not probs or not tokens:
        return None
    total = 0.0
    matched = False
    for prob, token in zip(probs, tokens):
        if str(token) in REFLECTION_STRINGS:
            value = _float(prob)
            if value is not None:
                total += value
                matched = True
    return total if matched else 0.0


def _margin(conf: dict[str, Any]) -> float | None:
    direct = _float(conf.get("margin"))
    if direct is not None:
        return direct
    probs = [_float(x) for x in (conf.get("top_probs") or [])]
    vals = sorted([float(p) for p in probs if p is not None], reverse=True)
    if len(vals) < 2:
        return None
    return vals[0] - vals[1]


def load_rpdi_points(root: Path, *, local_window: int, eps: float) -> list[RPDIPoint]:
    summary = _read_summary(root)
    out: list[RPDIPoint] = []
    for problem_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=_problem_sort_key):
        steps_file = _steps_path(problem_dir)
        if steps_file is None:
            continue
        pid = problem_dir.name
        rows = [r for r in _read_jsonl(steps_file) if not r.get("is_final_answer")]
        total_tokens = sum(int(r.get("token_count") or 0) for r in rows)
        token_prefix = 0
        scored: list[dict[str, Any]] = []
        for row in rows:
            token_prefix += int(row.get("token_count") or 0)
            if row.get("generator") != "slm":
                continue
            signals = row.get("observed_signals") if isinstance(row.get("observed_signals"), dict) else {}
            conf = (row.get("extra") or {}).get("confidence") or {}
            entropy = _float(signals.get("entropy"))
            if entropy is None:
                entropy = _float(conf.get("entropy") or conf.get("mean_token_entropy") or conf.get("norm_entropy"))
            if entropy is None:
                continue
            row = dict(row)
            row["_entropy"] = entropy
            row["_token_prefix"] = token_prefix
            scored.append(row)
        if not scored:
            continue
        entropies = [float(row["_entropy"]) for row in scored]
        h_global_problem = sum(entropies) / len(entropies)
        row_summary = summary.get(pid, {})
        forced_close = str(row_summary.get("stop_reason") or "").endswith("_forced_close_think")
        correct = _truthy(row_summary.get("correct"))
        for idx, row in enumerate(scored):
            progress = idx / max(1, len(scored) - 1)
            start = max(0, idx - local_window + 1)
            local_values = entropies[start : idx + 1]
            h_local = sum(local_values) / len(local_values)
            cumulative_values = entropies[: idx + 1]
            h_global_cumulative = sum(cumulative_values) / len(cumulative_values)
            signals = row.get("observed_signals") if isinstance(row.get("observed_signals"), dict) else {}
            conf = (row.get("extra") or {}).get("confidence") or {}
            if signals:
                conf = {**conf, **signals}
            action = str(row.get("action") or "")
            source = str(row.get("source") or row.get("generator") or "").upper()
            out.append(
                RPDIPoint(
                    problem_id=pid,
                    step_id=int(row.get("step_id") or idx + 1),
                    progress=progress,
                    phase=_phase(progress),
                    forced_close=forced_close,
                    correct=correct,
                    entropy=float(row["_entropy"]),
                    h_local=h_local,
                    h_global_problem=h_global_problem,
                    h_global_cumulative=h_global_cumulative,
                    rpdi_problem=h_local / (h_global_problem + eps),
                    rpdi_cumulative=h_local / (h_global_cumulative + eps),
                    margin=_margin(conf),
                    pref_topk=_pref_topk(conf),
                    reflection_start=_reflection_start(str(row.get("text") or "")),
                    llm_switch=source == "LLM" or action.startswith("LLM_") or action.startswith("HANDOFF_"),
                    action=action,
                    remaining_tokens_after_step=max(0, total_tokens - int(row["_token_prefix"])),
                    text=str(row.get("text") or ""),
                )
            )
    return out


def _point_dict(point: RPDIPoint) -> dict[str, Any]:
    return {
        "problem_id": point.problem_id,
        "step_id": point.step_id,
        "progress": point.progress,
        "phase": point.phase,
        "forced_close": point.forced_close,
        "correct": point.correct,
        "entropy": point.entropy,
        "h_local": point.h_local,
        "h_global_problem": point.h_global_problem,
        "h_global_cumulative": point.h_global_cumulative,
        "rpdi_problem": point.rpdi_problem,
        "rpdi_cumulative": point.rpdi_cumulative,
        "margin": point.margin,
        "pref_topk": point.pref_topk,
        "reflection_start": point.reflection_start,
        "llm_switch": point.llm_switch,
        "action": point.action,
        "remaining_tokens_after_step": point.remaining_tokens_after_step,
    }


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


def aggregate_problem(points: list[RPDIPoint], high_rpdi_threshold: float) -> list[dict[str, Any]]:
    by_problem: dict[str, list[RPDIPoint]] = defaultdict(list)
    for point in points:
        by_problem[point.problem_id].append(point)
    rows: list[dict[str, Any]] = []
    for pid, group in sorted(by_problem.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        early = [p for p in group if p.phase == "early"]
        late = [p for p in group if p.phase == "late"]
        ref = [p for p in group if p.reflection_start]
        nonref = [p for p in group if not p.reflection_start]
        switch = [p for p in group if p.llm_switch]
        rows.append(
            {
                "problem_id": pid,
                "forced_close": group[0].forced_close,
                "correct": group[0].correct,
                "checkpoint_count": len(group),
                "global_entropy": group[0].h_global_problem,
                "rpdi_early_mean": _mean([p.rpdi_problem for p in early]),
                "rpdi_early_p75": _quantile([p.rpdi_problem for p in early], 0.75),
                "rpdi_late_mean": _mean([p.rpdi_problem for p in late]),
                "rpdi_late_p75": _quantile([p.rpdi_problem for p in late], 0.75),
                "rpdi_late_max": max([p.rpdi_problem for p in late], default=None),
                "rpdi_reflection_start_mean": _mean([p.rpdi_problem for p in ref]),
                "rpdi_non_reflection_mean": _mean([p.rpdi_problem for p in nonref]),
                "rpdi_switch_mean": _mean([p.rpdi_problem for p in switch]),
                "high_rpdi_count": sum(1 for p in group if p.rpdi_problem >= high_rpdi_threshold),
                "late_high_rpdi_count": sum(
                    1 for p in late if p.rpdi_problem >= high_rpdi_threshold
                ),
                "remaining_corr_rpdi": _corr(
                    [p.rpdi_problem for p in group],
                    [float(p.remaining_tokens_after_step) for p in group],
                ),
                "pref_corr_rpdi": _corr([p.rpdi_problem for p in group], [p.pref_topk for p in group]),
            }
        )
    return rows


def aggregate_progress(points: list[RPDIPoint], bins: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[RPDIPoint]] = defaultdict(list)
    for point in points:
        group = "budget-hit" if point.forced_close else "normal-end"
        idx = min(bins - 1, max(0, int(point.progress * bins)))
        buckets[(group, idx)].append(point)
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
                    "rpdi_mean": _mean([p.rpdi_problem for p in bucket]),
                    "rpdi_p75": _quantile([p.rpdi_problem for p in bucket], 0.75),
                    "entropy_mean": _mean([p.entropy for p in bucket]),
                    "pref_topk_mean": _mean([p.pref_topk for p in bucket]),
                    "margin_mean": _mean([p.margin for p in bucket]),
                    "reflection_start_rate": (
                        sum(1 for p in bucket if p.reflection_start) / len(bucket)
                        if bucket
                        else None
                    ),
                    "llm_switch_count": sum(1 for p in bucket if p.llm_switch),
                }
            )
    return rows


def switch_context(points: list[RPDIPoint], window: int) -> list[dict[str, Any]]:
    by_problem: dict[str, list[RPDIPoint]] = defaultdict(list)
    for point in points:
        by_problem[point.problem_id].append(point)
    rows: list[dict[str, Any]] = []
    for pid, group in by_problem.items():
        group = sorted(group, key=lambda p: p.step_id)
        index = {p.step_id: i for i, p in enumerate(group)}
        for point in group:
            if not point.llm_switch:
                continue
            i = index[point.step_id]
            before = group[max(0, i - window) : i]
            after = group[i + 1 : min(len(group), i + window + 1)]
            rows.append(
                {
                    "problem_id": pid,
                    "step_id": point.step_id,
                    "progress": point.progress,
                    "action": point.action,
                    "rpdi_at_switch": point.rpdi_problem,
                    "entropy_at_switch": point.entropy,
                    "pref_topk_at_switch": point.pref_topk,
                    "margin_at_switch": point.margin,
                    "rpdi_before_mean": _mean([p.rpdi_problem for p in before]),
                    "rpdi_after_mean": _mean([p.rpdi_problem for p in after]),
                    "entropy_before_mean": _mean([p.entropy for p in before]),
                    "entropy_after_mean": _mean([p.entropy for p in after]),
                    "pref_before_mean": _mean([p.pref_topk for p in before]),
                    "pref_after_mean": _mean([p.pref_topk for p in after]),
                    "text": point.text.replace("\n", " ")[:220],
                }
            )
    return rows


def plot_progress(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"budget-hit": "#b23a48", "normal-end": "#2a9d8f"}
    for group in ["budget-hit", "normal-end"]:
        subset = [r for r in rows if r["group"] == group and r["rpdi_mean"] is not None]
        if not subset:
            continue
        ax.plot(
            [float(r["progress_mid"]) for r in subset],
            [float(r["rpdi_mean"]) for r in subset],
            color=colors[group],
            linewidth=1.8,
            label=group,
        )
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_title("RPDI by reasoning progress")
    ax.set_xlabel("normalized reasoning progress")
    ax.set_ylabel("RPDI = local entropy / problem entropy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rpdi_by_progress.png", dpi=180)
    plt.close(fig)


def plot_phase_box(points: list[RPDIPoint], out_dir: Path) -> None:
    labels = []
    values = []
    for group_name, forced in [("budget early", True), ("budget late", True), ("normal early", False), ("normal late", False)]:
        phase = "early" if "early" in group_name else "late"
        vals = [p.rpdi_problem for p in points if p.forced_close == forced and p.phase == phase]
        if vals:
            labels.append(group_name)
            values.append(vals)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_title("Early vs late RPDI distribution")
    ax.set_ylabel("RPDI")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "rpdi_early_late_boxplot.png", dpi=180)
    plt.close(fig)


def plot_reflection_scatter(points: list[RPDIPoint], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for point in points:
        if point.pref_topk is None:
            continue
        color = "#b23a48" if point.forced_close else "#2a9d8f"
        alpha = 0.65 if point.phase == "late" else 0.25
        marker = "x" if point.reflection_start else "o"
        ax.scatter(point.rpdi_problem, point.pref_topk, color=color, alpha=alpha, marker=marker, s=22)
    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_title("RPDI vs reflection mass")
    ax.set_xlabel("RPDI")
    ax.set_ylabel("P_ref within logged top-k")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "rpdi_vs_reflection_mass.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local/global entropy ratio (RPDI) in SARR traces.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-window", type=int, default=16)
    parser.add_argument("--progress-bins", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--high-rpdi-threshold", type=float, default=1.5)
    parser.add_argument("--switch-context-window", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.input_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "rpdi_entropy_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    points = load_rpdi_points(root, local_window=args.local_window, eps=args.eps)
    if not points:
        raise SystemExit(f"No SLM entropy checkpoints found under {root}")
    checkpoint_rows = [_point_dict(p) for p in points]
    problem_rows = aggregate_problem(points, args.high_rpdi_threshold)
    progress_rows = aggregate_progress(points, args.progress_bins)
    switch_rows = switch_context(points, args.switch_context_window)

    write_csv(out_dir / "rpdi_checkpoint_signals.csv", checkpoint_rows)
    write_csv(out_dir / "rpdi_problem_summary.csv", problem_rows)
    write_csv(out_dir / "rpdi_by_progress.csv", progress_rows)
    write_csv(out_dir / "rpdi_switch_context.csv", switch_rows)

    plot_progress(progress_rows, out_dir)
    plot_phase_box(points, out_dir)
    plot_reflection_scatter(points, out_dir)

    report = {
        "problem_count": len(problem_rows),
        "checkpoint_count": len(points),
        "local_window": args.local_window,
        "high_rpdi_threshold": args.high_rpdi_threshold,
        "rpdi_early_mean": _mean([p.rpdi_problem for p in points if p.phase == "early"]),
        "rpdi_late_mean": _mean([p.rpdi_problem for p in points if p.phase == "late"]),
        "rpdi_reflection_start_mean": _mean([p.rpdi_problem for p in points if p.reflection_start]),
        "rpdi_non_reflection_mean": _mean([p.rpdi_problem for p in points if not p.reflection_start]),
        "rpdi_llm_switch_mean": _mean([p.rpdi_problem for p in points if p.llm_switch]),
        "rpdi_budget_late_mean": _mean([p.rpdi_problem for p in points if p.forced_close and p.phase == "late"]),
        "rpdi_normal_late_mean": _mean([p.rpdi_problem for p in points if not p.forced_close and p.phase == "late"]),
        "remaining_corr_rpdi_all": _corr(
            [p.rpdi_problem for p in points],
            [float(p.remaining_tokens_after_step) for p in points],
        ),
    }
    (out_dir / "rpdi_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rpdi] wrote: {out_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
