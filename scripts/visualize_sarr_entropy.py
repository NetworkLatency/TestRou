#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class StepPoint:
    problem_id: str
    step_id: int
    source: str
    status: str
    action: str
    entropy: float | None
    confidence: float | None
    margin: float | None
    degeneration_score: float | None
    low_new_information_score: float | None
    text: str


@dataclass
class ProblemSeries:
    problem_id: str
    correct: bool | None
    finish_reason: str
    steps: list[StepPoint]


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _problem_sort_key(path: Path) -> tuple[int, str]:
    return (0, f"{int(path.name):09d}") if path.name.isdigit() else (1, path.name)


def _load_summary(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "summary.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row.get("problem_id") or row.get("id") or ""): row for row in csv.DictReader(f)}


def _steps_path(problem_dir: Path) -> Path | None:
    preferred = problem_dir / f"{problem_dir.name}.steps.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(problem_dir.glob("*.steps.jsonl"))
    return matches[0] if matches else None


def _signals(row: dict[str, Any]) -> dict[str, Any]:
    signals = row.get("observed_signals")
    if isinstance(signals, dict):
        return signals
    conf = (row.get("extra") or {}).get("confidence") or {}
    return {
        "raw_next_token_confidence": conf.get("raw_next_token_confidence") or conf.get("mean_token_confidence"),
        "entropy": conf.get("entropy") or conf.get("mean_token_entropy") or conf.get("norm_entropy"),
        "margin": conf.get("margin") or conf.get("mean_token_margin"),
    }


def load_problem_series(root: Path) -> list[ProblemSeries]:
    summary = _load_summary(root)
    out: list[ProblemSeries] = []
    for problem_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=_problem_sort_key):
        steps_file = _steps_path(problem_dir)
        if steps_file is None:
            continue
        pid = problem_dir.name
        points: list[StepPoint] = []
        for row in _read_jsonl(steps_file):
            if row.get("is_final_answer"):
                continue
            signals = _signals(row)
            source = str(row.get("source") or row.get("generator") or "").upper()
            points.append(
                StepPoint(
                    problem_id=pid,
                    step_id=int(row.get("step_id") or len(points) + 1),
                    source=source,
                    status=str(row.get("status") or ""),
                    action=str(row.get("action") or ""),
                    entropy=_float(signals.get("entropy")),
                    confidence=_float(signals.get("raw_next_token_confidence")),
                    margin=_float(signals.get("margin")),
                    degeneration_score=_float(signals.get("degeneration_score")),
                    low_new_information_score=_float(signals.get("low_new_information_score")),
                    text=str(row.get("text") or ""),
                )
            )
        row = summary.get(pid, {})
        out.append(
            ProblemSeries(
                problem_id=pid,
                correct=_truthy(row.get("correct")),
                finish_reason=str(row.get("finish_reason") or row.get("stop_reason") or ""),
                steps=points,
            )
        )
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_problem(series: ProblemSeries, *, late_window: int) -> dict[str, Any]:
    slm = [s for s in series.steps if s.source == "SLM" and s.status == "active"]
    ent = [float(s.entropy) for s in slm if s.entropy is not None]
    late = slm[-late_window:] if late_window > 0 else slm
    late_ent = [float(s.entropy) for s in late if s.entropy is not None]
    late_deg = [float(s.degeneration_score) for s in late if s.degeneration_score is not None]
    return {
        "problem_id": series.problem_id,
        "correct": series.correct,
        "finish_reason": series.finish_reason,
        "step_count": len(series.steps),
        "active_slm_step_count": len(slm),
        "active_llm_step_count": sum(1 for s in series.steps if s.source == "LLM" and s.status == "active"),
        "discarded_probe_count": sum(1 for s in series.steps if s.status == "probe_discarded"),
        "entropy_mean": _mean(ent),
        "entropy_last_window_mean": _mean(late_ent),
        "degeneration_last_window_mean": _mean(late_deg),
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


def _finite_xy(points: list[StepPoint], field: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for point in points:
        value = getattr(point, field)
        if value is not None:
            xs.append(point.step_id)
            ys.append(float(value))
    return xs, ys


def plot_problem(series: ProblemSeries, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        f"Problem {series.problem_id} | correct={series.correct} | finish={series.finish_reason}",
        fontsize=12,
    )

    xs, ent = _finite_xy(series.steps, "entropy")
    axes[0].plot(xs, ent, color="#365c8d", linewidth=1.4, label="entropy")
    axes[0].set_ylabel("entropy")
    axes[0].legend(loc="upper right", fontsize=8)

    xs, margin = _finite_xy(series.steps, "margin")
    axes[1].plot(xs, margin, color="#6a4c93", linewidth=1.2, label="margin")
    xs, confidence = _finite_xy(series.steps, "confidence")
    if xs:
        axes[1].plot(xs, confidence, color="#2a9d8f", linewidth=1.0, alpha=0.9, label="confidence")
    axes[1].set_ylabel("confidence")
    axes[1].legend(loc="lower right", fontsize=8)

    xs, degeneration = _finite_xy(series.steps, "degeneration_score")
    axes[2].plot(xs, degeneration, color="#b23a48", linewidth=1.2, label="degeneration")
    xs, low_info = _finite_xy(series.steps, "low_new_information_score")
    if xs:
        axes[2].plot(xs, low_info, color="#d97706", linewidth=1.0, alpha=0.9, label="low information")
    axes[2].set_ylabel("signals")
    axes[2].legend(loc="upper right", fontsize=8)

    active_sources = [s for s in series.steps if s.status == "active"]
    for point in active_sources:
        y = 1 if point.source == "LLM" else 0
        axes[3].scatter(point.step_id, y, color="#2f855a" if y else "#4a5568", s=18)
    for point in series.steps:
        if point.status == "probe_discarded":
            axes[3].scatter(point.step_id, 0.5, color="#b23a48", marker="x", s=30)
    axes[3].set_yticks([0, 0.5, 1], ["SLM", "discard", "LLM"])
    axes[3].set_ylabel("owner")
    axes[3].set_xlabel("step")

    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"{series.problem_id}_ownership_timeline.png", dpi=180)
    plt.close(fig)


def plot_aggregate(series_list: list[ProblemSeries], out_dir: Path, bins: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entropy_by_bin: list[list[float]] = [[] for _ in range(bins)]
    degeneration_by_bin: list[list[float]] = [[] for _ in range(bins)]
    for series in series_list:
        slm = [s for s in series.steps if s.source == "SLM" and s.status == "active"]
        n = len(slm)
        if n == 0:
            continue
        for i, step in enumerate(slm):
            idx = min(bins - 1, int((i / max(1, n - 1)) * bins))
            if step.entropy is not None:
                entropy_by_bin[idx].append(float(step.entropy))
            if step.degeneration_score is not None:
                degeneration_by_bin[idx].append(float(step.degeneration_score))

    xs = [(i + 0.5) / bins for i in range(bins)]
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, buckets, color in [
        ("entropy", entropy_by_bin, "#365c8d"),
        ("degeneration", degeneration_by_bin, "#b23a48"),
    ]:
        means = [_mean(bucket) for bucket in buckets]
        if any(v is not None for v in means):
            ax.plot(xs, [float("nan") if v is None else v for v in means], label=label, color=color, linewidth=1.8)
    ax.set_title("Observable signals by normalized SLM progress")
    ax.set_xlabel("normalized active SLM progress")
    ax.set_ylabel("mean signal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_signals_by_progress.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SARR ownership-controller signals.")
    parser.add_argument("--input-root", required=True, help="Experiment result directory containing summary.csv and problem subdirs.")
    parser.add_argument("--output-dir", default=None, help="Directory for figures and CSV summaries.")
    parser.add_argument("--late-window", type=int, default=20, help="Final active SLM steps used for late-window metrics.")
    parser.add_argument("--progress-bins", type=int, default=20, help="Bins for normalized-progress aggregate plot.")
    parser.add_argument("--max-problem-plots", type=int, default=0, help="0 means plot every problem.")
    args = parser.parse_args()

    root = Path(args.input_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "ownership_visualizations"
    series_list = load_problem_series(root)
    if not series_list:
        raise SystemExit(f"No problem step files found under {root}")

    summaries = [summarize_problem(series, late_window=args.late_window) for series in series_list]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "problem_signal_summary.csv", summaries)
    (out_dir / "problem_signal_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_problem_dir = out_dir / "per_problem"
    limit = len(series_list) if args.max_problem_plots == 0 else max(0, args.max_problem_plots)
    for series in series_list[:limit]:
        plot_problem(series, per_problem_dir)
    plot_aggregate(series_list, out_dir, bins=args.progress_bins)

    discarded_count = sum(row["discarded_probe_count"] for row in summaries)
    print(f"[ownership-viz] problems={len(summaries)} discarded_probes={discarded_count}")
    print(f"[ownership-viz] wrote: {out_dir}")


if __name__ == "__main__":
    main()
