"""Offline validation for trajectory-conditioned routing signals.

This script consumes outputs produced by ``phase0_pend_trajectory.py`` and
evaluates whether trajectory-level signals add information beyond local
single-step confidence signals.

It is intentionally read-only with respect to experiment directories. The
combined scores are fixed, interpretable averages of oriented z-scores rather
than trained classifiers, so the ablation remains easy to audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LOCAL_BOUNDARY_METRICS = [
    "topk_entropy_onset",
    "token_logprob_onset",
    "p_end_onset",
    "p_short_h4",
    "p_short_h8",
    "end_in_topk_rate_h8",
]

WINDOW_METRIC_FAMILIES: dict[str, list[tuple[str, int]]] = {
    # Higher entropy and lower generated-token logprob indicate instability.
    "confidence": [
        ("topk_entropy_mean", 1),
        ("token_logprob_mean", -1),
    ],
    # Successful trajectories tend to keep closure/end tokens visible.
    "closure": [
        ("end_in_topk_rate", -1),
        ("end_in_topk_rate_h16", -1),
        ("end_in_topk_rate_h32", -1),
    ],
    # Failed trajectories are often flatter: fewer P(end) spikes and lower
    # P(end) variability. These directions are fixed from the hypothesis.
    "pend_shape": [
        ("log_p_short_var", -1),
        ("p_short_mean", -1),
        ("p_short_h8_mean", -1),
        ("p_short_h32_mean", -1),
    ],
}

ABLATION_SPECS: dict[str, list[str]] = {
    "confidence_only": ["confidence"],
    "closure_only": ["closure"],
    "pend_shape_only": ["pend_shape"],
    "confidence_plus_closure": ["confidence", "closure"],
    "confidence_plus_shape": ["confidence", "pend_shape"],
    "closure_plus_shape": ["closure", "pend_shape"],
    "full": ["confidence", "closure", "pend_shape"],
    "full_minus_confidence": ["closure", "pend_shape"],
    "full_minus_closure": ["confidence", "pend_shape"],
    "full_minus_shape": ["confidence", "closure"],
}

TARGET_COLUMNS = {
    "wrong": "problem_wrong",
    "long_by_tokens": "long_by_tokens",
    "long_by_boundaries": "long_by_boundaries",
    "wrong_long": "wrong_long",
}


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    pairs = [(score, int(label)) for score, label in zip(scores, labels) if math.isfinite(score)]
    if not pairs:
        return None
    n_pos = sum(label for _, label in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    pairs.sort(key=lambda item: item[0])
    rank = 1
    idx = 0
    pos_rank_sum = 0.0
    while idx < len(pairs):
        end = idx
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx) - 1) / 2.0
        for tied_idx in range(idx, end):
            if pairs[tied_idx][1]:
                pos_rank_sum += avg_rank
        rank += end - idx
        idx = end
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _bootstrap_auc(
    rows: list[dict[str, Any]],
    score_key: str,
    target_key: str,
    *,
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None, float | None, float | None, int]:
    if iterations <= 0 or not rows:
        return (None, None, None, None, 0)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        auc = _auc([_float(row.get(score_key)) for row in sample], [_bool(row.get(target_key)) for row in sample])
        if auc is not None:
            values.append(auc)
    if not values:
        return (None, None, None, None, 0)
    values.sort()
    return (
        statistics.mean(values),
        values[int(0.025 * (len(values) - 1))],
        values[int(0.5 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
        len(values),
    )


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _std(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return 0.0
    return statistics.pstdev(finite)


def _z_scores(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = [_float(row.get(metric)) for row in rows]
    center = _mean(values)
    scale = _std(values)
    if not math.isfinite(center) or scale <= 1e-12:
        return [float("nan") for _ in values]
    return [(value - center) / scale if math.isfinite(value) else float("nan") for value in values]


def _add_oriented_scores(rows: list[dict[str, Any]]) -> None:
    """Add family and ablation scores in-place for a single run/window subset."""
    oriented_cache: dict[tuple[str, int], list[float]] = {}
    for family_metrics in WINDOW_METRIC_FAMILIES.values():
        for metric, direction in family_metrics:
            if (metric, direction) in oriented_cache:
                continue
            z_values = _z_scores(rows, metric)
            oriented_cache[(metric, direction)] = [
                direction * value if math.isfinite(value) else float("nan") for value in z_values
            ]

    for family, family_metrics in WINDOW_METRIC_FAMILIES.items():
        for idx, row in enumerate(rows):
            row[f"score_{family}"] = _mean(oriented_cache[(metric, direction)][idx] for metric, direction in family_metrics)

    for score_name, families in ABLATION_SPECS.items():
        for row in rows:
            row[f"score_{score_name}"] = _mean(_float(row.get(f"score_{family}")) for family in families)


def _parse_run(values: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for value in values:
        if "=" in value:
            label, raw_path = value.split("=", 1)
        else:
            raw_path = value
            label = Path(value).name
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Run directory not found: {path}")
        runs.append((label, path))
    if not runs:
        raise ValueError("At least one --run label=path is required")
    return runs


def _load_run(label: str, path: Path) -> dict[str, Any]:
    problem_rows = _read_csv(path / "phase0_problems.csv")
    boundary_rows = _read_csv(path / "phase0_boundaries.csv")
    window_rows = _read_csv(path / "phase0_windows.csv")

    for row in problem_rows:
        row["run"] = label
        row["problem_wrong"] = str(not _bool(row.get("correct")))
    for row in boundary_rows:
        row["run"] = label
        row["problem_wrong"] = str(not _bool(row.get("problem_correct")))
    for row in window_rows:
        row["run"] = label
    return {
        "label": label,
        "path": str(path),
        "problem_rows": problem_rows,
        "boundary_rows": boundary_rows,
        "window_rows": window_rows,
    }


def _summarize_problem_rows(run: dict[str, Any]) -> dict[str, Any]:
    rows = run["problem_rows"]
    finish_counts = Counter(row.get("finish_reason") for row in rows)
    correct = sum(1 for row in rows if _bool(row.get("correct")))
    return {
        "run": run["label"],
        "num_problems": len(rows),
        "num_correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "num_eos": finish_counts.get("eos", 0),
        "num_length": finish_counts.get("length", 0),
        "avg_generated_tokens": _mean(_float(row.get("generated_tokens")) for row in rows),
        "avg_num_boundaries": _mean(_float(row.get("num_boundaries")) for row in rows),
        "correct_eos": sum(1 for row in rows if _bool(row.get("correct")) and row.get("finish_reason") == "eos"),
        "wrong_eos": sum(1 for row in rows if (not _bool(row.get("correct"))) and row.get("finish_reason") == "eos"),
        "correct_length": sum(1 for row in rows if _bool(row.get("correct")) and row.get("finish_reason") == "length"),
        "wrong_length": sum(1 for row in rows if (not _bool(row.get("correct"))) and row.get("finish_reason") == "length"),
    }


def _boundary_auc_rows(run: dict[str, Any], *, targets: list[str]) -> list[dict[str, Any]]:
    rows = run["boundary_rows"]
    output: list[dict[str, Any]] = []
    scopes = {
        "all": lambda row: True,
        "early25": lambda row: int(_float(row.get("boundary_idx"), -1)) < 25,
        "early50": lambda row: int(_float(row.get("boundary_idx"), -1)) < 50,
        "early100": lambda row: int(_float(row.get("boundary_idx"), -1)) < 100,
    }
    for scope_name, predicate in scopes.items():
        scoped = [row for row in rows if predicate(row)]
        if not scoped:
            continue
        for target in targets:
            target_key = TARGET_COLUMNS[target]
            labels = [_bool(row.get(target_key)) for row in scoped]
            if all(labels) or not any(labels):
                continue
            for metric in LOCAL_BOUNDARY_METRICS:
                if metric not in scoped[0]:
                    continue
                auc = _auc([_float(row.get(metric)) for row in scoped], labels)
                if auc is None:
                    continue
                output.append(
                    {
                        "run": run["label"],
                        "scope": scope_name,
                        "target": target,
                        "metric": metric,
                        "n": len(scoped),
                        "positive": sum(labels),
                        "auc_high_score": auc,
                        "auc_best_direction": max(auc, 1.0 - auc),
                        "best_direction": "high" if auc >= 0.5 else "low",
                    }
                )
    return output


def _window_auc_and_ablation_rows(
    run: dict[str, Any],
    *,
    targets: list[str],
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = run["window_rows"]
    metric_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    window_sizes = sorted({int(_float(row.get("window_size"))) for row in rows})

    raw_metrics: list[str] = []
    for family_metrics in WINDOW_METRIC_FAMILIES.values():
        for metric, _direction in family_metrics:
            if metric not in raw_metrics:
                raw_metrics.append(metric)

    for window_size in window_sizes:
        scoped = [dict(row) for row in rows if int(_float(row.get("window_size"))) == window_size]
        if not scoped:
            continue
        _add_oriented_scores(scoped)
        scored_rows.extend(scoped)

        for target in targets:
            target_key = TARGET_COLUMNS[target]
            labels = [_bool(row.get(target_key)) for row in scoped]
            if all(labels) or not any(labels):
                continue

            for metric in raw_metrics:
                if metric not in scoped[0]:
                    continue
                auc = _auc([_float(row.get(metric)) for row in scoped], labels)
                if auc is None:
                    continue
                metric_rows.append(
                    {
                        "run": run["label"],
                        "window_size": window_size,
                        "target": target,
                        "metric": metric,
                        "n": len(scoped),
                        "positive": sum(labels),
                        "auc_high_score": auc,
                        "auc_best_direction": max(auc, 1.0 - auc),
                        "best_direction": "high" if auc >= 0.5 else "low",
                    }
                )

            for score_name in ABLATION_SPECS:
                score_key = f"score_{score_name}"
                auc = _auc([_float(row.get(score_key)) for row in scoped], labels)
                if auc is None:
                    continue
                ci_mean, ci_low, ci_median, ci_high, ci_n = _bootstrap_auc(
                    scoped,
                    score_key,
                    target_key,
                    iterations=bootstrap,
                    seed=seed + window_size + len(score_name),
                )
                ablation_rows.append(
                    {
                        "run": run["label"],
                        "window_size": window_size,
                        "target": target,
                        "score": score_name,
                        "families": "+".join(ABLATION_SPECS[score_name]),
                        "n": len(scoped),
                        "positive": sum(labels),
                        "auc_high_score": auc,
                        "auc_best_direction": max(auc, 1.0 - auc),
                        "best_direction": "high" if auc >= 0.5 else "low",
                        "bootstrap_mean": ci_mean,
                        "bootstrap_low_95": ci_low,
                        "bootstrap_median": ci_median,
                        "bootstrap_high_95": ci_high,
                        "bootstrap_n": ci_n,
                    }
                )
    return metric_rows, ablation_rows, scored_rows


def _write_plots(output_dir: Path, ablation_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]], targets: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        for window_size in sorted({int(row["window_size"]) for row in ablation_rows if row["target"] == target}):
            scoped = [row for row in ablation_rows if row["target"] == target and int(row["window_size"]) == window_size]
            if not scoped:
                continue
            labels = sorted({row["run"] for row in scoped})
            score_names = [
                "confidence_only",
                "closure_only",
                "pend_shape_only",
                "confidence_plus_closure",
                "full",
                "full_minus_confidence",
                "full_minus_closure",
                "full_minus_shape",
            ]
            width = 0.8 / max(1, len(labels))
            x_positions = list(range(len(score_names)))
            plt.figure(figsize=(max(9, len(score_names) * 1.1), 5))
            for label_idx, label in enumerate(labels):
                values = []
                for score_name in score_names:
                    match = next((row for row in scoped if row["run"] == label and row["score"] == score_name), None)
                    values.append(float(match["auc_high_score"]) if match else float("nan"))
                xs = [x + label_idx * width for x in x_positions]
                plt.bar(xs, values, width=width, label=label)
            plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
            plt.ylim(0.0, 1.0)
            plt.xticks([x + width * (len(labels) - 1) / 2 for x in x_positions], score_names, rotation=35, ha="right")
            plt.ylabel("AUC: high score predicts target")
            plt.title(f"Trajectory signal ablation, target={target}, window={window_size}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"ablation_{target}_w{window_size}.png", dpi=170)
            plt.close()

    for target in targets:
        scoped = [row for row in metric_rows if row["target"] == target]
        if not scoped:
            continue
        for window_size in sorted({int(row["window_size"]) for row in scoped}):
            rows = [row for row in scoped if int(row["window_size"]) == window_size]
            rows.sort(key=lambda row: row["auc_best_direction"], reverse=True)
            rows = rows[:16]
            plt.figure(figsize=(10, 5))
            labels = [f'{row["run"]}:{row["metric"]}' for row in rows]
            values = [float(row["auc_best_direction"]) for row in rows]
            plt.bar(range(len(rows)), values)
            plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
            plt.ylim(0.0, 1.0)
            plt.xticks(range(len(rows)), labels, rotation=45, ha="right")
            plt.ylabel("Best-direction AUC")
            plt.title(f"Top raw window metrics, target={target}, window={window_size}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"raw_metrics_{target}_w{window_size}.png", dpi=170)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="Run directory, optionally label=path. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGET_COLUMNS), default=["wrong", "wrong_long"])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(label, path) for label, path in _parse_run(args.run)]

    problem_summary_rows = [_summarize_problem_rows(run) for run in runs]
    local_auc_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    scored_window_rows: list[dict[str, Any]] = []

    for run in runs:
        local_auc_rows.extend(_boundary_auc_rows(run, targets=args.targets))
        metric_rows, run_ablation_rows, run_scored_rows = _window_auc_and_ablation_rows(
            run,
            targets=args.targets,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        window_metric_rows.extend(metric_rows)
        ablation_rows.extend(run_ablation_rows)
        scored_window_rows.extend(run_scored_rows)

    _write_csv(
        output_dir / "problem_summary.csv",
        problem_summary_rows,
        [
            "run",
            "num_problems",
            "num_correct",
            "accuracy",
            "num_eos",
            "num_length",
            "avg_generated_tokens",
            "avg_num_boundaries",
            "correct_eos",
            "wrong_eos",
            "correct_length",
            "wrong_length",
        ],
    )
    _write_csv(
        output_dir / "local_boundary_auc.csv",
        local_auc_rows,
        ["run", "scope", "target", "metric", "n", "positive", "auc_high_score", "auc_best_direction", "best_direction"],
    )
    _write_csv(
        output_dir / "window_metric_auc.csv",
        window_metric_rows,
        ["run", "window_size", "target", "metric", "n", "positive", "auc_high_score", "auc_best_direction", "best_direction"],
    )
    _write_csv(
        output_dir / "window_ablation_auc.csv",
        ablation_rows,
        [
            "run",
            "window_size",
            "target",
            "score",
            "families",
            "n",
            "positive",
            "auc_high_score",
            "auc_best_direction",
            "best_direction",
            "bootstrap_mean",
            "bootstrap_low_95",
            "bootstrap_median",
            "bootstrap_high_95",
            "bootstrap_n",
        ],
    )
    _write_csv(
        output_dir / "scored_windows.csv",
        scored_window_rows,
        sorted({key for row in scored_window_rows for key in row.keys()}),
    )

    summary = {
        "runs": [{"label": run["label"], "path": run["path"]} for run in runs],
        "targets": args.targets,
        "bootstrap": args.bootstrap,
        "problem_summary": problem_summary_rows,
        "local_boundary_auc_top": sorted(local_auc_rows, key=lambda row: row["auc_best_direction"], reverse=True)[:20],
        "window_metric_auc_top": sorted(window_metric_rows, key=lambda row: row["auc_best_direction"], reverse=True)[:20],
        "ablation_top": sorted(ablation_rows, key=lambda row: row["auc_best_direction"], reverse=True)[:20],
        "score_families": {
            family: [{"metric": metric, "direction": direction} for metric, direction in metrics]
            for family, metrics in WINDOW_METRIC_FAMILIES.items()
        },
        "ablation_specs": ABLATION_SPECS,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_plots:
        _write_plots(output_dir, ablation_rows, window_metric_rows, args.targets)

    print(f"Wrote validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
