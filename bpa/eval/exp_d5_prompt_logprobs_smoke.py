from __future__ import annotations

import argparse
import csv
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


def run_smoke(config: BPAConfig, dataset: str, max_problems: int) -> list[dict]:
    slm, llm = init_engines(config)
    rows = []
    for problem in tqdm(load_eval_dataset(dataset, config, max_problems=max_problems), desc="D5 smoke"):
        state = GenerationState(problem_text=problem.problem_text)
        for _ in range(20):
            l0 = l0_filter(state, slm, config)
            if l0.passed and len(l0.top_logprobs) >= 2:
                b1, b2 = l1_shadow_rollout(state, slm, config, l0)
                for k in [1, 5, 20, 50]:
                    k_config = config.with_updates(prompt_logprobs_topk=k)
                    for branch_idx, branch in [(1, b1), (2, b2)]:
                        before_prefill = state.llm_prefill_tokens
                        score = score_branch(state, llm, branch.step_branch_text, k_config)
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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt logprobs smoke test for BPA arbitration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--max-problems", type=int, default=5)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    rows = run_smoke(config, args.dataset, args.max_problems)
    out_root = Path(config.output_dir) / "diagnostics" / "d5_prompt_logprobs_smoke"
    out_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_root / "records.jsonl", rows)
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
        for k in [1, 5, 20, 50]:
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


if __name__ == "__main__":
    main()
