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


SIGNAL_FIELDS = [
    "entropy",
    "margin",
    "degeneration_score",
    "low_new_information_score",
    "reflection_pattern_count",
    "repeated_verification_pattern_count",
    "repeated_answer_mention_count",
]


@dataclass
class ProblemSummary:
    problem_id: str
    correct: bool | None
    finish_reason: str
    terminal_step: int


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


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_summary(root: Path) -> dict[str, ProblemSummary]:
    rows: dict[str, ProblemSummary] = {}
    summary_path = root / "summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid = str(row["problem_id"])
            rows[pid] = ProblemSummary(
                problem_id=pid,
                correct=_truthy(row.get("correct")),
                finish_reason=str(row.get("finish_reason") or row.get("stop_reason") or ""),
                terminal_step=int(float(row.get("step_count") or row.get("active_thinking_step_count") or 0)),
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
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else p.name)


def bin_index(value: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def bin_label(edges: list[float], idx: int) -> str:
    return f"[{edges[idx]:.2f},{edges[idx + 1]:.2f})"


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


def _controller_events_path(problem_dir: Path) -> Path | None:
    preferred = problem_dir / f"{problem_dir.name}.controller_events.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(problem_dir.glob("*.controller_events.jsonl"))
    return matches[0] if matches else None


def _steps_path(problem_dir: Path) -> Path | None:
    preferred = problem_dir / f"{problem_dir.name}.steps.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(problem_dir.glob("*.steps.jsonl"))
    return matches[0] if matches else None


def build_step_rows(root: Path, horizon: int) -> list[dict[str, Any]]:
    summaries = load_summary(root)
    out: list[dict[str, Any]] = []
    for problem_dir in problem_dirs(root):
        pid = problem_dir.name
        summary = summaries.get(pid)
        steps_path = _steps_path(problem_dir)
        events_path = _controller_events_path(problem_dir)
        if summary is None or steps_path is None:
            continue
        steps = load_jsonl(steps_path)
        events = load_jsonl(events_path) if events_path else []
        ownership_transfer_steps = sorted(
            int(event.get("step_id") or 0)
            for event in events
            if event.get("event") == "driver_switch"
            and str(event.get("to_state") or "").startswith("LLM_")
        )

        for step in steps:
            if step.get("is_final_answer") or step.get("status") != "active":
                continue
            source = str(step.get("source") or step.get("generator") or "").upper()
            if source != "SLM":
                continue
            sid = int(step.get("step_id") or 0)
            signals = step.get("observed_signals") if isinstance(step.get("observed_signals"), dict) else {}
            future_transfer = any(sid < x <= sid + horizon for x in ownership_transfer_steps)
            terminal_in_horizon = sid < summary.terminal_step <= sid + horizon
            row = {
                "problem_id": pid,
                "step_id": sid,
                "action": step.get("action"),
                "future_ownership_transfer_h": future_transfer,
                "future_close_h": terminal_in_horizon,
                "future_wrong_h": bool(summary.correct is False and terminal_in_horizon),
                "future_signal_event_h": future_transfer or terminal_in_horizon,
                "problem_wrong": summary.correct is False,
                "finish_reason": summary.finish_reason,
            }
            for field in SIGNAL_FIELDS:
                row[field] = _float(signals.get(field), 0.0)
            out.append(row)
    return out


def aggregate_signal(rows: list[dict[str, Any]], field: str, edges: list[float]) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = _float(row.get(field))
        if value is None:
            continue
        buckets[bin_index(value, edges)].append(row)
    out = []
    for idx in range(len(edges) - 1):
        bucket = buckets.get(idx, [])
        out.append(
            {
                "signal": field,
                "bin": bin_label(edges, idx),
                "n": len(bucket),
                "future_signal_event_rate": rate(bucket, "future_signal_event_h"),
                "future_ownership_transfer_rate": rate(bucket, "future_ownership_transfer_h"),
                "future_close_rate": rate(bucket, "future_close_h"),
                "future_wrong_rate": rate(bucket, "future_wrong_h"),
            }
        )
    return out


def plot_signal(rows: list[dict[str, Any]], field: str, path: Path) -> None:
    data = [row for row in rows if row["signal"] == field and row["n"] > 0]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [row["bin"] for row in data]
    ax.plot(xs, [row["future_signal_event_rate"] for row in data], marker="o", label="future signal event")
    ax.plot(xs, [row["future_ownership_transfer_rate"] for row in data], marker="s", label="future LLM ownership")
    ax.plot(xs, [row["future_close_rate"] for row in data], marker="^", label="future close")
    ax.set_title(f"{field} vs future controller event")
    ax.set_xlabel(field)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SARR-CoDE ownership-signal analysis.")
    parser.add_argument("--root", required=True, help="SARR result directory containing summary.csv and problem folders.")
    parser.add_argument("--output", required=True, help="Directory for CSV and plot outputs.")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--bin-width", type=float, default=0.1)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    edges = [round(i * args.bin_width, 10) for i in range(int(1 / args.bin_width) + 1)]
    if edges[-1] < 1.0:
        edges.append(1.0)
    edges[-1] = 1.0000001

    step_rows = build_step_rows(root, args.horizon)
    write_csv(output / "step_signal_rows.csv", step_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for field in SIGNAL_FIELDS:
        aggregate_rows.extend(aggregate_signal(step_rows, field, edges))
    write_csv(output / "signal_bins.csv", aggregate_rows)
    for field in SIGNAL_FIELDS:
        plot_signal(aggregate_rows, field, output / f"{field}_future_event_rate.png")

    metadata = {
        "root": str(root),
        "output": str(output),
        "horizon": args.horizon,
        "num_step_rows": len(step_rows),
        "signals": SIGNAL_FIELDS,
        "definitions": {
            "future_signal_event_h": "future LLM ownership transfer or close within h active steps",
            "future_ownership_transfer_h": "driver switch into an LLM ownership state within h active steps",
        },
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
