#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass
class ProblemSummary:
    problem_id: str
    correct: bool
    forced_close: bool
    terminal_step: int


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_summary(root: Path) -> dict[str, ProblemSummary]:
    rows: dict[str, ProblemSummary] = {}
    summary_path = root / "summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row["problem_id"])
            rows[pid] = ProblemSummary(
                problem_id=pid,
                correct=_truthy(row.get("correct")),
                forced_close=str(row.get("stop_reason") or "").endswith("_forced_close_think"),
                terminal_step=int(float(row.get("step_count") or 0)),
            )
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def problem_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def bin_index(value: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def bin_label(edges: list[float], idx: int) -> str:
    return f"[{edges[idx]:.2f},{edges[idx + 1]:.2f})"


def fixed_bins(max_value: int, width: int) -> list[tuple[int, int]]:
    if max_value <= 0:
        return [(0, width - 1)]
    bins = []
    start = 0
    while start <= max_value:
        end = start + width - 1
        bins.append((start, end))
        start += width
    return bins


def int_bin_label(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


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


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row.get(key)) / len(rows) if rows else float("nan")


def pct(rows: list[dict[str, Any]], key: str) -> float:
    value = rate(rows, key)
    return value * 100.0 if not math.isnan(value) else float("nan")


def build_step_rows(root: Path, horizon: int, raw_low_threshold: float, smooth_low_threshold: float) -> list[dict[str, Any]]:
    summaries = load_summary(root)
    out: list[dict[str, Any]] = []
    for d in problem_dirs(root):
        pid = d.name
        summary = summaries.get(pid)
        if summary is None:
            continue
        steps = load_jsonl(d / f"{pid}.steps.jsonl")
        events = load_jsonl(d / f"{pid}.rollback_events.jsonl")
        intervention_steps = sorted(
            int(e.get("trigger_step") or 0)
            for e in events
            if e.get("event") == "llm_lease" or e.get("type") not in (None, "", "LLM_LEASE")
        )
        scored = [
            s
            for s in steps
            if s.get("generator") == "slm" and s.get("c_raw") is not None and s.get("removed_by_rollback") is not True
        ]
        high_run = 0
        masked_cum = 0
        masked_before_by_step: dict[int, int] = {}
        high_run_by_step: dict[int, int] = {}
        for s in scored:
            sid = int(s.get("step_id") or 0)
            c_raw = _float(s.get("c_raw"), 0.0) or 0.0
            readiness = _float(
                s.get("readiness_value"),
                _float(s.get("readiness"), _float(s.get("readiness_raw_smooth"), c_raw)),
            )
            readiness = c_raw if readiness is None else readiness
            masked_before_by_step[sid] = masked_cum
            high_run = high_run + 1 if _truthy(s.get("readiness_high")) else 0
            high_run_by_step[sid] = high_run
            is_masked = c_raw <= raw_low_threshold and readiness > smooth_low_threshold
            if is_masked:
                masked_cum += 1

        for s in scored:
            sid = int(s.get("step_id") or 0)
            c_raw = _float(s.get("c_raw"), 0.0) or 0.0
            readiness = _float(
                s.get("readiness_value"),
                _float(s.get("readiness"), _float(s.get("readiness_raw_smooth"), c_raw)),
            )
            readiness = c_raw if readiness is None else readiness
            future_intervention = any(sid < x <= sid + horizon for x in intervention_steps)
            terminal_in_horizon = sid < summary.terminal_step <= sid + horizon
            forced_close_h = bool(summary.forced_close and terminal_in_horizon)
            wrong_h = bool((not summary.correct) and terminal_in_horizon)
            no_close_within_h = summary.terminal_step > sid + horizon
            out.append(
                {
                    "problem_id": pid,
                    "step_id": sid,
                    "c_raw": c_raw,
                    "readiness_value": readiness,
                    "readiness_high": _truthy(s.get("readiness_high")),
                    "readiness_low": _truthy(s.get("readiness_low")),
                    "masked_uncertainty_before": masked_before_by_step[sid],
                    "high_conf_run": high_run_by_step[sid],
                    "anchor_age": sid - int(s.get("clean_autonomy_anchor") or 0),
                    "state_before": s.get("state_before"),
                    "action": s.get("action"),
                    "future_intervention_h": future_intervention,
                    "future_forced_close_h": forced_close_h,
                    "future_wrong_h": wrong_h,
                    "future_failure_h": future_intervention or forced_close_h or wrong_h,
                    "no_close_within_h": no_close_within_h,
                    "problem_wrong": not summary.correct,
                    "problem_forced_close": summary.forced_close,
                }
            )
    return out


def aggregate_confidence(rows: list[dict[str, Any]], edges: list[float], field: str) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[bin_index(float(row[field]), edges)].append(row)
    out = []
    for idx in range(len(edges) - 1):
        bucket = buckets.get(idx, [])
        out.append(
            {
                "signal": field,
                "bin": bin_label(edges, idx),
                "n": len(bucket),
                "future_failure_rate": rate(bucket, "future_failure_h"),
                "future_intervention_rate": rate(bucket, "future_intervention_h"),
                "future_forced_close_rate": rate(bucket, "future_forced_close_h"),
                "future_wrong_rate": rate(bucket, "future_wrong_h"),
            }
        )
    return out


def aggregate_int_bins(rows: list[dict[str, Any]], field: str, width: int) -> list[dict[str, Any]]:
    max_value = max((int(row[field]) for row in rows), default=0)
    bins = fixed_bins(max_value, width)
    out = []
    for start, end in bins:
        bucket = [row for row in rows if start <= int(row[field]) <= end]
        out.append(
            {
                "bin": int_bin_label(start, end),
                "start": start,
                "end": end,
                "n": len(bucket),
                "future_failure_rate": rate(bucket, "future_failure_h"),
                "future_intervention_rate": rate(bucket, "future_intervention_h"),
                "future_forced_close_rate": rate(bucket, "future_forced_close_h"),
                "future_wrong_rate": rate(bucket, "future_wrong_h"),
                "no_close_within_h_rate": rate(bucket, "no_close_within_h"),
            }
        )
    return out


def aggregate_heatmap(rows: list[dict[str, Any]], x_width: int, y_width: int) -> list[dict[str, Any]]:
    max_x = max((int(row["high_conf_run"]) for row in rows), default=0)
    max_y = max((int(row["masked_uncertainty_before"]) for row in rows), default=0)
    x_bins = fixed_bins(max_x, x_width)
    y_bins = fixed_bins(max_y, y_width)
    out = []
    for y0, y1 in y_bins:
        for x0, x1 in x_bins:
            bucket = [
                row
                for row in rows
                if x0 <= int(row["high_conf_run"]) <= x1 and y0 <= int(row["masked_uncertainty_before"]) <= y1
            ]
            out.append(
                {
                    "high_conf_run_bin": int_bin_label(x0, x1),
                    "masked_uncertainty_bin": int_bin_label(y0, y1),
                    "x_start": x0,
                    "x_end": x1,
                    "y_start": y0,
                    "y_end": y1,
                    "n": len(bucket),
                    "future_failure_rate": rate(bucket, "future_failure_h"),
                }
            )
    return out


def plot_confidence(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for signal, marker in [("c_raw", "o"), ("readiness_value", "s")]:
        data = [row for row in rows if row["signal"] == signal and row["n"] > 0]
        ax.plot([row["bin"] for row in data], [row["future_failure_rate"] for row in data], marker=marker, label=signal)
    ax.set_title("Confidence Bin vs Future Failure Rate")
    ax.set_xlabel("confidence bin")
    ax.set_ylabel("future failure rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_int_bins(rows: list[dict[str, Any]], field_name: str, title: str, x_label: str, path: Path) -> None:
    data = [row for row in rows if row["n"] > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [row["bin"] for row in data]
    ax.plot(xs, [row["future_failure_rate"] for row in data], marker="o", label="future failure")
    ax.plot(xs, [row["future_intervention_rate"] for row in data], marker="s", label="future intervention")
    if "no_close_within_h_rate" in data[0]:
        ax.plot(xs, [row["no_close_within_h_rate"] for row in data], marker="^", label="no close within horizon")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    x_labels = sorted({row["high_conf_run_bin"] for row in rows}, key=lambda x: int(x.split("-")[0]))
    y_labels = sorted({row["masked_uncertainty_bin"] for row in rows}, key=lambda x: int(x.split("-")[0]))
    matrix = []
    counts = []
    for y in y_labels:
        values = []
        ns = []
        for x in x_labels:
            match = next(row for row in rows if row["high_conf_run_bin"] == x and row["masked_uncertainty_bin"] == y)
            values.append(float("nan") if match["n"] == 0 else float(match["future_failure_rate"]))
            ns.append(int(match["n"]))
        matrix.append(values)
        counts.append(ns)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma", aspect="auto", origin="lower")
    ax.set_title("Future Failure Probability by High-Confidence Dwell and Masked Uncertainty")
    ax.set_xlabel("high-confidence dwell length")
    ax.set_ylabel("masked uncertainty count")
    ax.set_xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)), y_labels)
    for yi, row in enumerate(matrix):
        for xi, value in enumerate(row):
            n = counts[yi][xi]
            if n:
                ax.text(xi, yi, f"{value:.2f}\nn={n}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, label="future failure probability")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SARR-CoDE confidence and non-convergence signal analysis.")
    parser.add_argument("--root", required=True, help="SARR result directory containing summary.csv and problem folders.")
    parser.add_argument("--output", required=True, help="Directory for CSV and plot outputs.")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--raw-low-threshold", type=float, default=0.35)
    parser.add_argument("--smooth-low-threshold", type=float, default=0.35)
    parser.add_argument("--confidence-bin-width", type=float, default=0.1)
    parser.add_argument("--masked-bin-width", type=int, default=5)
    parser.add_argument("--high-run-bin-width", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    confidence_edges = [round(i * args.confidence_bin_width, 10) for i in range(int(1 / args.confidence_bin_width) + 1)]
    if confidence_edges[-1] < 1.0:
        confidence_edges.append(1.0)
    confidence_edges[-1] = 1.0000001

    step_rows = build_step_rows(root, args.horizon, args.raw_low_threshold, args.smooth_low_threshold)
    write_csv(output / "step_signal_rows.csv", step_rows)

    confidence_rows = aggregate_confidence(step_rows, confidence_edges, "c_raw")
    confidence_rows += aggregate_confidence(step_rows, confidence_edges, "readiness_value")
    write_csv(output / "confidence_bins.csv", confidence_rows)
    plot_confidence(confidence_rows, output / "confidence_bin_failure_rate.png")

    masked_rows = aggregate_int_bins(step_rows, "masked_uncertainty_before", args.masked_bin_width)
    write_csv(output / "masked_uncertainty_bins.csv", masked_rows)
    plot_int_bins(
        masked_rows,
        "masked_uncertainty_before",
        "Masked Uncertainty Accumulation vs Future Failure",
        "cumulative masked uncertainty spikes before step t",
        output / "masked_uncertainty_failure_rate.png",
    )

    high_run_rows = aggregate_int_bins(step_rows, "high_conf_run", args.high_run_bin_width)
    write_csv(output / "high_conf_run_bins.csv", high_run_rows)
    plot_int_bins(
        high_run_rows,
        "high_conf_run",
        "High-Confidence Run Length vs Non-Convergence",
        "consecutive high-readiness length",
        output / "high_conf_run_failure_rate.png",
    )

    heatmap_rows = aggregate_heatmap(step_rows, args.high_run_bin_width, args.masked_bin_width)
    write_csv(output / "high_conf_x_masked_heatmap.csv", heatmap_rows)
    plot_heatmap(heatmap_rows, output / "high_conf_x_masked_heatmap.png")

    metadata = {
        "root": str(root),
        "output": str(output),
        "horizon": args.horizon,
        "raw_low_threshold": args.raw_low_threshold,
        "smooth_low_threshold": args.smooth_low_threshold,
        "num_step_rows": len(step_rows),
        "definitions": {
            "future_failure_h": "future intervention within h steps OR terminal forced close/wrong within h steps",
            "masked_uncertainty_spike": "c_raw <= raw_low_threshold and readiness_value > smooth_low_threshold",
            "high_conf_run": "consecutive scored SLM steps with readiness_high=true ending at step t",
        },
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
