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
    delta: float
    theta_s: float
    tau_start: int
    tau_D: int
    B_max: int


SWEEP_VARIANTS = [
    SweepVariant("D1_balanced_055", delta=0.55, theta_s=0.70, tau_start=1, tau_D=1, B_max=5),
    SweepVariant("D2_balanced_050", delta=0.50, theta_s=0.70, tau_start=1, tau_D=1, B_max=5),
    SweepVariant("D3_balanced_060", delta=0.60, theta_s=0.70, tau_start=1, tau_D=1, B_max=5),
    SweepVariant("D4_aggressive_startup", delta=0.55, theta_s=0.75, tau_start=1, tau_D=1, B_max=4),
    SweepVariant("D5_conservative", delta=0.50, theta_s=0.65, tau_start=2, tau_D=2, B_max=6),
    SweepVariant("D6_event_aggressive", delta=0.65, theta_s=0.70, tau_start=1, tau_D=1, B_max=4),
    SweepVariant("D7_high_theta_post_conservative", delta=0.55, theta_s=0.75, tau_start=1, tau_D=2, B_max=5),
    SweepVariant("D8_low_theta_post_repair", delta=0.60, theta_s=0.60, tau_start=2, tau_D=1, B_max=6),
]


SUMMARY_KEYS = [
    "accuracy",
    "num_problems",
    "num_evaluated",
    "rollback_rate",
    "startup_rollback_rate",
    "post_stable_rollback_rate",
    "avg_stagnation_rollback_count",
    "avg_llm_lease_count",
    "avg_rollback_span",
    "avg_recovery_steps",
    "recovery_ready_rate",
    "recovery_exhausted_rate",
    "forced_close_think_rate",
    "force_slm_after_recovery_fail_rate",
    "llm_token_ratio",
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
    cfg.setdefault("confidence", {})["delta"] = variant.delta
    cfg.setdefault("stable", {})["theta_s"] = variant.theta_s
    cfg.setdefault("stable", {})["tau_D"] = variant.tau_D
    cfg.setdefault("startup", {})["tau_start"] = variant.tau_start
    cfg.setdefault("startup", {})["B_max"] = variant.B_max
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
    parser = argparse.ArgumentParser(description="Run the predefined SARR-CoDE D1-D8 parameter sweep.")
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
