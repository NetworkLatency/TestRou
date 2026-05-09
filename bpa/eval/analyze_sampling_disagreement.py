from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from bpa.config import BPAConfig
from bpa.trace import json_safe


DEFAULT_METRICS = [
    "prefix_consensus_support_count",
    "prefix_consensus_vote_fraction",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if value is True or text == "true":
        return True
    if value is False or text == "false":
        return False
    return None


def _is_initial_probe(row: dict[str, Any]) -> bool:
    parsed = _parse_bool(row.get("is_initial_probe"))
    if parsed is not None:
        return parsed
    try:
        if int(row.get("boundary_idx", 0)) < 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return int(row.get("prefix_char_len", 1)) == 0
    except (TypeError, ValueError):
        return False


def _default_probe_path(config: BPAConfig, dataset: str) -> Path:
    return Path(config.output_dir) / "diagnostics" / "sampling_disagreement" / dataset / "probes.jsonl"


def _default_label_path(config: BPAConfig, dataset: str) -> Path:
    return Path(config.output_dir) / "diagnostics" / "boundary_continuation" / dataset / "boundary_labels.csv"


def _merge_labels(probes: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_map = {
        (str(row.get("problem_id")), str(row.get("boundary_idx"))): row
        for row in labels
        if row.get("problem_id") not in (None, "") and row.get("boundary_idx") not in (None, "")
    }
    merged = []
    for row in probes:
        key = (str(row.get("problem_id")), str(row.get("boundary_idx")))
        if key in label_map:
            merged.append({**row, **label_map[key]})
    return merged


def auroc(values: list[float], labels: list[bool]) -> float | None:
    pairs = [(value, label) for value, label in zip(values, labels) if value is not None and label is not None]
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    sorted_pairs = sorted(enumerate(pairs), key=lambda item: item[1][0])
    ranks = [0.0] * len(pairs)
    idx = 0
    while idx < len(sorted_pairs):
        end = idx + 1
        while end < len(sorted_pairs) and sorted_pairs[end][1][0] == sorted_pairs[idx][1][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            original_idx = sorted_pairs[pos][0]
            ranks[original_idx] = avg_rank
        idx = end

    sum_positive_ranks = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (sum_positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)


def hartigan_dip_test(values: list[float]) -> dict[str, Any]:
    if len(values) < 4:
        return {"available": False, "dip": None, "p_value": None, "reason": "too_few_values"}
    try:
        from diptest import diptest
    except Exception:
        return {"available": False, "dip": None, "p_value": None, "reason": "install diptest for Hartigan dip p-values"}
    dip, p_value = diptest(np.array(values, dtype=float))
    return {"available": True, "dip": float(dip), "p_value": float(p_value), "reason": None}


def quantile_rows(values: list[float], labels: list[bool], num_bins: int) -> list[dict[str, Any]]:
    pairs = sorted((value, label) for value, label in zip(values, labels) if value is not None and label is not None)
    if not pairs:
        return []
    rows = []
    for bin_idx in range(num_bins):
        start = round(bin_idx * len(pairs) / num_bins)
        end = round((bin_idx + 1) * len(pairs) / num_bins)
        subset = pairs[start:end]
        if not subset:
            continue
        positives = sum(1 for _, label in subset if label)
        rows.append(
            {
                "quantile_bin": bin_idx,
                "count": len(subset),
                "metric_min": subset[0][0],
                "metric_max": subset[-1][0],
                "critical_count": positives,
                "critical_rate": positives / len(subset),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def _plot_distribution(path: Path, values: list[float], metric: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=30, color="#4C78A8", edgecolor="white")
    plt.xlabel(metric)
    plt.ylabel("Boundary count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_quantiles(path: Path, rows: list[dict[str, Any]], metric: str) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([row["quantile_bin"] for row in rows], [row["critical_rate"] for row in rows], marker="o")
    plt.xlabel(f"{metric} quantile")
    plt.ylabel("P(critical)")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def analyze(
    probes: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    metrics: list[str],
    num_bins: int,
    out_dir: Path,
    *,
    include_initial_probe: bool = False,
) -> dict[str, Any]:
    raw_probe_count = len(probes)
    if not include_initial_probe:
        probes = [row for row in probes if not _is_initial_probe(row)]
    labeled = _merge_labels(probes, labels) if labels else []
    summary: dict[str, Any] = {
        "num_probes_raw": raw_probe_count,
        "num_probes": len(probes),
        "num_labeled": len(labeled),
        "include_initial_probe": include_initial_probe,
        "metrics": {},
    }
    quantile_tables: dict[str, list[dict[str, Any]]] = {}
    distribution_rows: list[dict[str, Any]] = []

    for metric in metrics:
        values = [_parse_float(row.get(metric)) for row in probes]
        metric_values = [value for value in values if value is not None]
        if not metric_values:
            continue
        dip = hartigan_dip_test(metric_values)
        metric_summary = {
            "count": len(metric_values),
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
            "min": float(np.min(metric_values)),
            "max": float(np.max(metric_values)),
            "dip_test": dip,
        }
        distribution_rows.append(
            {
                "metric": metric,
                **{key: value for key, value in metric_summary.items() if key != "dip_test"},
                "dip_available": dip["available"],
                "dip": dip["dip"],
                "dip_p_value": dip["p_value"],
                "dip_reason": dip["reason"],
            }
        )
        _plot_distribution(out_dir / f"distribution_{metric}.png", metric_values, metric)

        if labeled:
            labeled_values = [_parse_float(row.get(metric)) for row in labeled]
            labeled_targets = [_parse_bool(row.get("critical")) for row in labeled]
            auc = auroc(labeled_values, labeled_targets)
            labeled_pairs = [
                (value, label)
                for value, label in zip(labeled_values, labeled_targets)
                if value is not None and label is not None
            ]
            q_rows = quantile_rows([value for value, _ in labeled_pairs], [label for _, label in labeled_pairs], num_bins)
            metric_summary["auroc"] = auc
            metric_summary["num_labeled_with_metric"] = len(labeled_pairs)
            quantile_tables[metric] = q_rows
            _plot_quantiles(out_dir / f"critical_by_quantile_{metric}.png", q_rows, metric)

        summary["metrics"][metric] = metric_summary

    _write_csv(out_dir / "distribution_summary.csv", distribution_rows)
    for metric, rows in quantile_tables.items():
        _write_csv(out_dir / f"critical_by_quantile_{metric}.csv", rows)
    with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze numeric probe fields and critical-boundary labels.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--probes-path", default=None)
    parser.add_argument("--labels-path", default=None)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--include-initial-probe", action="store_true", help="Include boundary_idx=-1 diagnostics in plots and metrics.")
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    probes_path = Path(args.probes_path) if args.probes_path else _default_probe_path(config, args.dataset)
    labels_path = Path(args.labels_path) if args.labels_path else _default_label_path(config, args.dataset)
    probes = _read_jsonl(probes_path)
    labels = _read_csv(labels_path)
    out_dir = Path(config.output_dir) / "diagnostics" / "sampling_analysis" / args.dataset
    analyze(probes, labels, args.metrics, args.num_bins, out_dir, include_initial_probe=args.include_initial_probe)
    print(f"Wrote {out_dir / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
