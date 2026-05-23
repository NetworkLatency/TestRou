#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_SCRIPT = REPO_ROOT / "scripts" / "run_sarr_code.py"


@dataclass(frozen=True)
class SweepVariant:
    name: str
    local_entropy_delta: float
    prefix_bad_ratio: float
    degeneration_score_threshold: float
    low_new_information_threshold: float
    handoff_max_risk_rank: int


SWEEP_VARIANTS = [
    SweepVariant("O1_balanced", 0.15, 0.66, 0.62, 0.72, 1),
    SweepVariant("O2_conservative_handoff", 0.15, 0.66, 0.62, 0.72, 0),
    SweepVariant("O3_sensitive_prefix", 0.15, 0.50, 0.62, 0.72, 1),
    SweepVariant("O4_conservative_prefix", 0.18, 0.75, 0.68, 0.78, 1),
    SweepVariant("O5_sensitive_local", 0.10, 0.66, 0.62, 0.72, 1),
    SweepVariant("O6_loop_sensitive", 0.15, 0.66, 0.55, 0.65, 1),
]


SUMMARY_KEYS = [
    "accuracy",
    "num_problems",
    "num_evaluated",
    "avg_driver_switch_count",
    "avg_llm_ownership_episodes",
    "avg_handoff_probe_count",
    "handoff_success_rate",
    "handoff_failure_rate",
    "answer_stability_rate",
    "degenerative_loop_rate",
    "prefix_contamination_rate",
    "rollback_rate",
    "total_rollback_count",
    "total_sealed_interval_count",
    "total_repeated_rollback_blocked_count",
    "avg_confidence_forward_count",
    "avg_lookahead_count",
    "avg_problem_wall_time",
    "avg_llm_token_share",
    "total_slm_decode_tokens",
    "total_llm_decode_tokens",
    "total_llm_prefill_tokens",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply_variant(base_config: dict[str, Any], variant: SweepVariant) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))
    risk = cfg.setdefault("risk", {})
    risk["local_entropy_delta"] = variant.local_entropy_delta
    risk["prefix_bad_ratio"] = variant.prefix_bad_ratio
    risk["degeneration_score_threshold"] = variant.degeneration_score_threshold
    risk["low_new_information_threshold"] = variant.low_new_information_threshold
    risk["handoff_max_risk_rank"] = variant.handoff_max_risk_rank
    cfg.setdefault("metadata", {})
    cfg["metadata"]["sweep_variant"] = asdict(variant)
    return cfg


def selected_variants(only: str | None) -> list[SweepVariant]:
    if not only:
        return list(SWEEP_VARIANTS)
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    variants = [variant for variant in SWEEP_VARIANTS if variant.name in wanted]
    missing = sorted(wanted - {variant.name for variant in variants})
    if missing:
        raise SystemExit(f"Unknown variant name(s): {', '.join(missing)}")
    return variants


def run_variant(
    *,
    python_bin: str,
    run_script: Path,
    config_path: Path,
    dataset: str,
    output_root: Path,
    variant_name: str,
    max_problems: int | None,
    resume: bool,
    dry_run: bool,
) -> int:
    cmd = [
        python_bin,
        str(run_script),
        "--config",
        str(config_path),
        "--dataset",
        dataset,
        "--output-root",
        str(output_root),
        "--variant",
        variant_name,
    ]
    if max_problems is not None:
        cmd.extend(["--max-problems", str(max_problems)])
    if resume:
        cmd.append("--resume")

    print("\n==>", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


def metrics_path(output_root: Path, dataset: str, variant_name: str) -> Path:
    return output_root / dataset / variant_name / "summary_metrics.json"


def collect_sweep_summary(output_root: Path, dataset: str, variants: list[SweepVariant]) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        row: dict[str, Any] = asdict(variant)
        path = metrics_path(output_root, dataset, variant.name)
        row["metrics_path"] = str(path)
        if path.exists():
            metrics = load_json(path)
            for key in SUMMARY_KEYS:
                row[key] = metrics.get(key)
        else:
            row["missing_metrics"] = True
        rows.append(row)
    return rows


def write_sweep_summary(output_root: Path, dataset: str, rows: list[dict[str, Any]]) -> None:
    summary_dir = output_root / dataset / "sarr_sweep"
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "sweep_summary.json"
    csv_path = summary_dir / "sweep_summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


def write_variant_manifest(output_root: Path, dataset: str, variants: list[SweepVariant]) -> None:
    manifest_path = output_root / dataset / "sarr_sweep" / "variant_manifest.json"
    write_json(
        manifest_path,
        {
            "variants": [asdict(variant) for variant in variants],
            "summary_keys": SUMMARY_KEYS,
        },
    )
    print(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the predefined SARR-CoDE ownership-controller sweep.")
    parser.add_argument("--base-config", default="configs/sarr_code_aggressive.json")
    parser.add_argument("--dataset", default="aime25", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", default=None, help="Comma-separated variant names to run.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-script", default=str(DEFAULT_RUN_SCRIPT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = REPO_ROOT / base_config_path
    base_config = load_json(base_config_path)
    output_root = Path(args.output_root or base_config.get("output_dir") or "sarr_results")
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    variants = selected_variants(args.only)
    config_dir = output_root / args.dataset / "sarr_sweep" / "configs"
    run_script = Path(args.run_script)
    if not run_script.is_absolute():
        run_script = REPO_ROOT / run_script

    write_variant_manifest(output_root, args.dataset, variants)

    failures: list[tuple[str, int]] = []
    for variant in variants:
        variant_config = apply_variant(base_config, variant)
        variant_config_path = config_dir / f"{variant.name}.json"
        write_json(variant_config_path, variant_config)
        rc = run_variant(
            python_bin=args.python_bin,
            run_script=run_script,
            config_path=variant_config_path,
            dataset=args.dataset,
            output_root=output_root,
            variant_name=variant.name,
            max_problems=args.max_problems,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failures.append((variant.name, rc))
            if args.stop_on_failure:
                break

    rows = collect_sweep_summary(output_root, args.dataset, variants)
    write_sweep_summary(output_root, args.dataset, rows)

    if failures:
        detail = ", ".join(f"{name}:{rc}" for name, rc in failures)
        raise SystemExit(f"Sweep finished with failed variant(s): {detail}")


if __name__ == "__main__":
    main()
