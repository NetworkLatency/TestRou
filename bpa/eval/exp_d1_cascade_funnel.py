from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.engines import init_engines
from bpa.pipeline import _slm_generate_step, run_cascade
from bpa.safety import ensure_step_terminator
from bpa.state import Decision, GenerationState

from .datasets import load_eval_dataset


def run_funnel(config: BPAConfig, dataset: str, max_problems: int | None, max_steps: int) -> list[dict]:
    rows: list[dict] = []
    for rollout_length in [8, 16, 32]:
        sweep_config = config.with_updates(rollout_length=rollout_length, apply_arbitration=False)
        slm, llm = init_engines(sweep_config)
        for problem in tqdm(load_eval_dataset(dataset, sweep_config, max_problems=max_problems), desc=f"D1 rollout={rollout_length}"):
            state = GenerationState(problem_text=problem.problem_text)
            t0 = time.time()
            l0_pass = l1_invoked = l2_trigger = arb_calls = llm_full = 0
            for _ in range(max_steps):
                cascade = run_cascade(state, slm, llm, sweep_config)
                l0_pass += int(cascade.l0.passed)
                l1_invoked += int(cascade.l1 is not None)
                l2_trigger += int(cascade.l2.triggered_arbitration if cascade.l2 else False)
                arb_calls += int(cascade.arbitration is not None)
                llm_full += int(cascade.decision == Decision.LLM_FULL)
                text, finish = _slm_generate_step(state, slm, sweep_config)
                state.assistant_prefix_text += ensure_step_terminator(text, finish)
                state.step_count += 1
                if finish == "eos":
                    break
            rows.append(
                {
                    "dataset": dataset,
                    "problem_id": problem.problem_id,
                    "rollout_length": rollout_length,
                    "boundary_count": state.step_count,
                    "l0_pass": l0_pass,
                    "l1_invocation": l1_invoked,
                    "l2_trigger": l2_trigger,
                    "llm_arbitration_call": arb_calls,
                    "llm_full_call": llm_full,
                    "wall_time_s": time.time() - t0,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp-D1 cascade funnel and rollout sweep.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--max-problems", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()
    config = BPAConfig.from_json(args.config)
    rows = run_funnel(config, args.dataset, args.max_problems, args.max_steps)
    out = Path(config.output_dir) / "diagnostics" / "d1_cascade_funnel" / f"{args.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
