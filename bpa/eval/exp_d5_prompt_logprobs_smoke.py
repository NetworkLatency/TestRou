from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from bpa.arbitration import score_branch
from bpa.cascade.l0 import l0_filter
from bpa.cascade.l1 import l1_shadow_rollout
from bpa.config import BPAConfig
from bpa.engines import init_engines
from bpa.pipeline import _slm_generate_step
from bpa.safety import ensure_step_terminator
from bpa.state import GenerationState
from bpa.trace import write_jsonl

from .datasets import load_eval_dataset


class SmokeRunAborted(RuntimeError):
    def __init__(self, message: str, rows: list[dict], error_record: dict):
        super().__init__(message)
        self.rows = rows
        self.error_record = error_record


def _is_fatal_engine_error(exc: Exception) -> bool:
    fatal_names = {"EngineDeadError", "RuntimeError"}
    text = str(exc)
    return (
        type(exc).__name__ in fatal_names
        and (
            "EngineCore encountered an issue" in text
            or "Engine core initialization failed" in text
            or "EngineDeadError" in text
            or "died unexpectedly" in text
        )
    )


def run_smoke(
    config: BPAConfig,
    dataset: str,
    max_problems: int,
    max_steps: int,
    max_triggers_per_problem: int,
) -> list[dict]:
    slm, llm = init_engines(config)
    rows = []
    sweep = sorted(set(int(k) for k in config.prompt_logprobs_sweep))
    for problem in tqdm(load_eval_dataset(dataset, config, max_problems=max_problems), desc="D5 smoke"):
        state = GenerationState(problem_text=problem.problem_text)
        triggers_seen = 0
        for _ in range(max_steps):
            l0 = l0_filter(state, slm, config)
            if l0.passed and len(l0.top_logprobs) >= 2:
                if triggers_seen >= max_triggers_per_problem:
                    text, finish = _slm_generate_step(state, slm, config)
                    state.assistant_prefix_text += ensure_step_terminator(text, finish)
                    state.step_count += 1
                    if finish == "eos" or state.slm_decode_tokens >= config.max_total_tokens:
                        break
                    continue
                triggers_seen += 1
                b1, b2 = l1_shadow_rollout(state, slm, config, l0)
                for k in sweep:
                    k_config = config.with_updates(prompt_logprobs_topk=k)
                    for branch_idx, branch in [(1, b1), (2, b2)]:
                        before_prefill = state.llm_prefill_tokens
                        try:
                            score = score_branch(state, llm, branch.step_branch_text, k_config)
                        except Exception as exc:
                            error_record = {
                                "problem_id": problem.problem_id,
                                "step_idx": state.step_count,
                                "branch_idx": branch_idx,
                                "prompt_logprobs_topk": k,
                                "exception_type": type(exc).__name__,
                                "exception": str(exc),
                                "fatal": _is_fatal_engine_error(exc),
                            }
                            if error_record["fatal"]:
                                raise SmokeRunAborted(
                                    "Fatal vLLM engine error during D5 scoring. "
                                    "Partial records were written; inspect the terminal log above for vLLM root cause.",
                                    rows,
                                    error_record,
                                ) from exc
                            rows.append(
                                {
                                    "problem_id": problem.problem_id,
                                    "step_idx": state.step_count,
                                    "branch_idx": branch_idx,
                                    "prompt_logprobs_topk": k,
                                    "branch_token_count": 0,
                                    "missing_count": 0,
                                    "missing_ratio": 1.0,
                                    "is_invalid": True,
                                    "invalid_reason": f"score_exception:{type(exc).__name__}:{exc}",
                                    "prefill_tokens": state.llm_prefill_tokens - before_prefill,
                                }
                            )
                            continue
                        rows.append(
                            {
                                "problem_id": problem.problem_id,
                                "step_idx": state.step_count,
                                "branch_idx": branch_idx,
                                "prompt_logprobs_topk": k,
                                "branch_token_count": score.branch_token_count,
                                "missing_count": score.missing_count,
                                "missing_ratio": score.missing_ratio,
                                "is_invalid": score.is_invalid,
                                "invalid_reason": score.invalid_reason,
                                "prefill_tokens": state.llm_prefill_tokens - before_prefill,
                            }
                        )
            text, finish = _slm_generate_step(state, slm, config)
            state.assistant_prefix_text += ensure_step_terminator(text, finish)
            state.step_count += 1
            if finish == "eos" or state.slm_decode_tokens >= config.max_total_tokens:
                break
        if config.reset_prefix_cache_after_problem:
            slm.clear_runtime_cache()
            llm.clear_runtime_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt logprobs smoke test for BPA arbitration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--max-problems", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-triggers-per-problem", type=int, default=5)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    out_root = Path(config.output_dir) / "diagnostics" / "d5_prompt_logprobs_smoke"
    out_root.mkdir(parents=True, exist_ok=True)
    aborted = None
    try:
        rows = run_smoke(
            config,
            args.dataset,
            args.max_problems,
            args.max_steps,
            args.max_triggers_per_problem,
        )
    except SmokeRunAborted as exc:
        rows = exc.rows
        aborted = exc.error_record
    write_jsonl(out_root / "records.jsonl", rows)
    if aborted is not None:
        (out_root / "aborted.json").write_text(json.dumps(aborted, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_logprobs_topk",
                "branches",
                "per_token_missing_rate",
                "per_branch_invalid_rate",
            ],
        )
        writer.writeheader()
        for k in sorted(set(int(k) for k in config.prompt_logprobs_sweep)):
            subset = [row for row in rows if row["prompt_logprobs_topk"] == k]
            total_tokens = sum(row["branch_token_count"] for row in subset)
            missing = sum(row["missing_count"] for row in subset)
            invalid = sum(1 for row in subset if row["is_invalid"])
            writer.writerow(
                {
                    "prompt_logprobs_topk": k,
                    "branches": len(subset),
                    "per_token_missing_rate": missing / total_tokens if total_tokens else 0.0,
                    "per_branch_invalid_rate": invalid / len(subset) if subset else 0.0,
                }
            )
    print(f"Wrote {csv_path}")
    if aborted is not None:
        print(f"D5 aborted after fatal vLLM engine error. Wrote {out_root / 'aborted.json'}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
