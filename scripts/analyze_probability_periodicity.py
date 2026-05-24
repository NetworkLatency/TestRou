#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class StepProbPoint:
    problem_id: str
    step_id: int
    source: str
    status: str
    text: str
    token_ids: list[int]
    logprobs: list[float]
    mean_logprob: float | None
    std_logprob: float | None
    min_logprob: float | None
    mean_prob: float | None
    repeated_ngram_ratio: float | None
    degeneration_score: float | None


@dataclass
class UnitPoint:
    problem_id: str
    start_step_id: int
    end_step_id: int
    text: str
    logprobs: list[float]

    @property
    def mean_logprob(self) -> float | None:
        return _mean(self.logprobs)

    @property
    def std_logprob(self) -> float | None:
        return _pstdev(self.logprobs)


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _pstdev(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((value - mu) ** 2 for value in clean) / len(clean))


def _percentile(values: list[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    idx = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
    return clean[idx]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _word_ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(str(text or "").lower())
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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


def _steps_path(problem_dir: Path) -> Path | None:
    if problem_dir.is_file() and problem_dir.name.endswith(".steps.jsonl"):
        return problem_dir
    preferred = problem_dir / f"{problem_dir.name}.steps.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(problem_dir.glob("*.steps.jsonl"))
    return matches[0] if matches else None


def _problem_sort_key(path: Path) -> tuple[int, str]:
    return (0, f"{int(path.name):09d}") if path.name.isdigit() else (1, path.name)


def _problem_dirs(root: Path) -> list[Path]:
    if root.is_file() and root.name.endswith(".steps.jsonl"):
        return [root]
    if _steps_path(root) is not None:
        return [root]
    return sorted([path for path in root.iterdir() if path.is_dir() and _steps_path(path) is not None], key=_problem_sort_key)


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        parsed = _float(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _step_probability_summary(extra: dict[str, Any], logprobs: list[float]) -> dict[str, float | None]:
    summary = extra.get("token_probability") if isinstance(extra.get("token_probability"), dict) else {}
    if logprobs:
        probs = [math.exp(value) for value in logprobs]
        return {
            "mean_logprob": _mean(logprobs),
            "std_logprob": _pstdev(logprobs),
            "min_logprob": min(logprobs),
            "p10_logprob": _percentile(logprobs, 0.10),
            "mean_prob": _mean(probs),
        }
    return {
        "mean_logprob": _float(summary.get("mean_logprob")),
        "std_logprob": _float(summary.get("std_logprob")),
        "min_logprob": _float(summary.get("min_logprob")),
        "p10_logprob": None,
        "mean_prob": _float(summary.get("mean_prob")),
    }


def load_step_points(problem_dir: Path, *, source: str) -> list[StepProbPoint]:
    path = _steps_path(problem_dir)
    if path is None:
        return []
    problem_id = problem_dir.stem if problem_dir.is_file() else problem_dir.name
    points: list[StepProbPoint] = []
    for row in _read_jsonl(path):
        if row.get("is_final_answer") or row.get("status") != "active":
            continue
        row_source = str(row.get("source") or row.get("generator") or "").upper()
        if source != "ALL" and row_source != source:
            continue
        extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        token_ids = _as_int_list(row.get("token_ids"))
        logprobs = _as_float_list(extra.get("generated_token_logprobs"))
        if token_ids and logprobs:
            limit = min(len(token_ids), len(logprobs))
            token_ids = token_ids[:limit]
            logprobs = logprobs[:limit]
        signals = row.get("observed_signals") if isinstance(row.get("observed_signals"), dict) else {}
        summary = _step_probability_summary(extra, logprobs)
        points.append(
            StepProbPoint(
                problem_id=problem_id,
                step_id=int(row.get("step_id") or len(points) + 1),
                source=row_source,
                status=str(row.get("status") or ""),
                text=str(row.get("text") or ""),
                token_ids=token_ids,
                logprobs=logprobs,
                mean_logprob=summary["mean_logprob"],
                std_logprob=summary["std_logprob"],
                min_logprob=summary["min_logprob"],
                mean_prob=summary["mean_prob"],
                repeated_ngram_ratio=_float(signals.get("repeated_ngram_ratio")),
                degeneration_score=_float(signals.get("degeneration_score")),
            )
        )
    return points


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def step_rows(points: list[StepProbPoint]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in points:
        rows.append(
            {
                "problem_id": point.problem_id,
                "step_id": point.step_id,
                "source": point.source,
                "token_count": len(point.token_ids),
                "has_token_logprobs": bool(point.logprobs),
                "mean_logprob": point.mean_logprob,
                "std_logprob": point.std_logprob,
                "min_logprob": point.min_logprob,
                "mean_prob": point.mean_prob,
                "repeated_ngram_ratio": point.repeated_ngram_ratio,
                "degeneration_score": point.degeneration_score,
                "text_preview": point.text.replace("\n", "\\n")[:120],
            }
        )
    return rows


def _unit_from_step(point: StepProbPoint) -> UnitPoint:
    return UnitPoint(
        problem_id=point.problem_id,
        start_step_id=point.step_id,
        end_step_id=point.step_id,
        text=point.text,
        logprobs=list(point.logprobs),
    )


def make_step_units(points: list[StepProbPoint]) -> list[UnitPoint]:
    return [_unit_from_step(point) for point in points]


def make_chunk_units(points: list[StepProbPoint], chunk_size: int) -> list[UnitPoint]:
    out: list[UnitPoint] = []
    if chunk_size < 1:
        return out
    for idx in range(0, len(points), chunk_size):
        chunk = points[idx : idx + chunk_size]
        if not chunk:
            continue
        logprobs: list[float] = []
        for point in chunk:
            logprobs.extend(point.logprobs)
        out.append(
            UnitPoint(
                problem_id=chunk[0].problem_id,
                start_step_id=chunk[0].step_id,
                end_step_id=chunk[-1].step_id,
                text="".join(point.text for point in chunk),
                logprobs=logprobs,
            )
        )
    return out


def lag_summary(problem_id: str, granularity: str, units: list[UnitPoint], max_lag: int) -> list[dict[str, Any]]:
    grams = [_word_ngrams(unit.text) for unit in units]
    means = [unit.mean_logprob for unit in units]
    rows: list[dict[str, Any]] = []
    for lag in range(1, min(max_lag, len(units) - 1) + 1):
        sims: list[float] = []
        current_probs: list[float] = []
        lagged_probs: list[float] = []
        abs_delta: list[float] = []
        for idx in range(lag, len(units)):
            sims.append(_jaccard(grams[idx], grams[idx - lag]))
            curr = means[idx]
            prev = means[idx - lag]
            if curr is not None and prev is not None:
                current_probs.append(curr)
                lagged_probs.append(prev)
                abs_delta.append(abs(curr - prev))
        rows.append(
            {
                "problem_id": problem_id,
                "granularity": granularity,
                "lag": lag,
                "n_pairs": len(sims),
                "text_similarity_mean": _mean(sims),
                "text_similarity_max": max(sims) if sims else None,
                "logprob_corr": _pearson(current_probs, lagged_probs),
                "mean_abs_logprob_delta": _mean(abs_delta),
                "mean_logprob": _mean([value for value in means if value is not None]),
            }
        )
    return rows


def rolling_windows(
    problem_id: str,
    granularity: str,
    units: list[UnitPoint],
    *,
    window_size: int,
    max_lag: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if window_size < 2 or len(units) < window_size:
        return rows
    grams = [_word_ngrams(unit.text) for unit in units]
    means = [unit.mean_logprob for unit in units]
    for start in range(0, len(units) - window_size + 1):
        end = start + window_size
        best_row: dict[str, Any] | None = None
        for lag in range(1, min(max_lag, window_size - 1) + 1):
            sims: list[float] = []
            curr_vals: list[float] = []
            prev_vals: list[float] = []
            for idx in range(start + lag, end):
                sims.append(_jaccard(grams[idx], grams[idx - lag]))
                curr = means[idx]
                prev = means[idx - lag]
                if curr is not None and prev is not None:
                    curr_vals.append(curr)
                    prev_vals.append(prev)
            candidate = {
                "problem_id": problem_id,
                "granularity": granularity,
                "window_size": window_size,
                "window_start_step": units[start].start_step_id,
                "window_end_step": units[end - 1].end_step_id,
                "best_lag": lag,
                "best_text_similarity": _mean(sims),
                "best_text_similarity_max": max(sims) if sims else None,
                "logprob_corr_at_best_lag": _pearson(curr_vals, prev_vals),
                "mean_logprob": _mean([value for value in means[start:end] if value is not None]),
                "std_logprob": _pstdev([value for value in means[start:end] if value is not None]),
            }
            if best_row is None or (candidate["best_text_similarity"] or 0.0) > (best_row["best_text_similarity"] or 0.0):
                best_row = candidate
        if best_row is not None:
            rows.append(best_row)
    return rows


def flatten_tokens(points: list[StepProbPoint]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    pos = 0
    for point in points:
        for idx, (token_id, logprob) in enumerate(zip(point.token_ids, point.logprobs)):
            tokens.append(
                {
                    "problem_id": point.problem_id,
                    "token_pos": pos,
                    "step_id": point.step_id,
                    "token_index_in_step": idx,
                    "token_id": token_id,
                    "logprob": logprob,
                }
            )
            pos += 1
    return tokens


def token_lag_summary(problem_id: str, tokens: list[dict[str, Any]], max_lag: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(tokens) < 2:
        return rows
    ids = [int(token["token_id"]) for token in tokens]
    logprobs = [float(token["logprob"]) for token in tokens]
    for lag in range(1, min(max_lag, len(tokens) - 1) + 1):
        match = [1.0 if ids[idx] == ids[idx - lag] else 0.0 for idx in range(lag, len(ids))]
        rows.append(
            {
                "problem_id": problem_id,
                "granularity": "token",
                "lag": lag,
                "n_pairs": len(match),
                "token_match_rate": _mean(match),
                "logprob_corr": _pearson(logprobs[lag:], logprobs[:-lag]),
                "mean_abs_logprob_delta": _mean([abs(logprobs[idx] - logprobs[idx - lag]) for idx in range(lag, len(logprobs))]),
                "mean_logprob": _mean(logprobs),
            }
        )
    return rows


def token_rolling_windows(
    problem_id: str,
    tokens: list[dict[str, Any]],
    *,
    window_size: int,
    stride: int,
    max_lag: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if window_size < 2 or len(tokens) < window_size:
        return rows
    stride = max(1, int(stride))
    ids = [int(token["token_id"]) for token in tokens]
    logprobs = [float(token["logprob"]) for token in tokens]
    for start in range(0, len(tokens) - window_size + 1, stride):
        end = start + window_size
        best: dict[str, Any] | None = None
        for lag in range(1, min(max_lag, window_size - 1) + 1):
            match = [1.0 if ids[idx] == ids[idx - lag] else 0.0 for idx in range(start + lag, end)]
            curr = logprobs[start + lag : end]
            prev = logprobs[start : end - lag]
            candidate = {
                "problem_id": problem_id,
                "granularity": "token",
                "window_size": window_size,
                "window_start_token": start,
                "window_end_token": end - 1,
                "window_start_step": tokens[start]["step_id"],
                "window_end_step": tokens[end - 1]["step_id"],
                "best_lag": lag,
                "best_token_match_rate": _mean(match),
                "logprob_corr_at_best_lag": _pearson(curr, prev),
                "mean_logprob": _mean(logprobs[start:end]),
                "std_logprob": _pstdev(logprobs[start:end]),
            }
            if best is None or (candidate["best_token_match_rate"] or 0.0) > (best["best_token_match_rate"] or 0.0):
                best = candidate
        if best is not None:
            rows.append(best)
    return rows


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text or "").split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated-token probability periodicity in SARR step traces.")
    parser.add_argument("--input", required=True, help="Experiment root, problem directory, or *.steps.jsonl file.")
    parser.add_argument("--output", required=True, help="Output directory for CSV/JSON analysis files.")
    parser.add_argument("--source", default="SLM", choices=["SLM", "LLM", "ALL"])
    parser.add_argument("--max-step-lag", type=int, default=12)
    parser.add_argument("--max-token-lag", type=int, default=256)
    parser.add_argument("--step-window-sizes", default="4,8,12")
    parser.add_argument("--chunk-sizes", default="2,4,8")
    parser.add_argument("--token-window", type=int, default=256)
    parser.add_argument("--token-window-stride", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_windows = _parse_int_list(args.step_window_sizes)
    chunk_sizes = _parse_int_list(args.chunk_sizes)

    all_step_points: list[StepProbPoint] = []
    step_csv_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    token_roll_rows: list[dict[str, Any]] = []

    problem_count = 0
    for problem_dir in _problem_dirs(root):
        points = load_step_points(problem_dir, source=args.source)
        if not points:
            continue
        problem_count += 1
        problem_id = points[0].problem_id
        all_step_points.extend(points)
        step_csv_rows.extend(step_rows(points))

        step_units = make_step_units(points)
        lag_rows.extend(lag_summary(problem_id, "step", step_units, args.max_step_lag))
        for window_size in step_windows:
            rolling_rows.extend(
                rolling_windows(
                    problem_id,
                    "step",
                    step_units,
                    window_size=window_size,
                    max_lag=args.max_step_lag,
                )
            )

        for chunk_size in chunk_sizes:
            chunk_units = make_chunk_units(points, chunk_size)
            granularity = f"chunk_{chunk_size}_steps"
            lag_rows.extend(lag_summary(problem_id, granularity, chunk_units, args.max_step_lag))
            for window_size in step_windows:
                rolling_rows.extend(
                    rolling_windows(
                        problem_id,
                        granularity,
                        chunk_units,
                        window_size=window_size,
                        max_lag=args.max_step_lag,
                    )
                )

        tokens = flatten_tokens(points)
        token_rows.extend(token_lag_summary(problem_id, tokens, args.max_token_lag))
        token_roll_rows.extend(
            token_rolling_windows(
                problem_id,
                tokens,
                window_size=args.token_window,
                stride=args.token_window_stride,
                max_lag=args.max_token_lag,
            )
        )

    _write_csv(out_dir / "probability_step_rows.csv", step_csv_rows)
    _write_csv(out_dir / "probability_lag_summary.csv", lag_rows)
    _write_csv(out_dir / "rolling_periodicity_windows.csv", rolling_rows)
    _write_csv(out_dir / "token_lag_summary.csv", token_rows)
    _write_csv(out_dir / "token_rolling_windows.csv", token_roll_rows)

    steps_with_token_logprobs = sum(1 for point in all_step_points if point.logprobs)
    metadata = {
        "input": str(root),
        "output": str(out_dir),
        "source": args.source,
        "problem_count": problem_count,
        "step_count": len(all_step_points),
        "steps_with_token_logprobs": steps_with_token_logprobs,
        "max_step_lag": args.max_step_lag,
        "max_token_lag": args.max_token_lag,
        "step_window_sizes": step_windows,
        "chunk_sizes": chunk_sizes,
        "token_window": args.token_window,
        "token_window_stride": args.token_window_stride,
        "notes": {
            "text_similarity": "Jaccard similarity over word 4-grams for step/chunk units.",
            "token_match_rate": "Exact token-id match rate at a token lag.",
            "probability_signal": "Generated-token logprob; probability can be recovered as exp(logprob).",
        },
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
