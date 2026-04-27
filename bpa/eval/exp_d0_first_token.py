from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from bpa.cascade.l0 import classify_first_char, l0_filter
from bpa.config import BPAConfig
from bpa.engines import init_engines
from bpa.pipeline import _slm_generate_step
from bpa.safety import ensure_step_terminator
from bpa.state import GenerationState
from bpa.trace import write_jsonl

from .datasets import load_eval_dataset

SKIP_CLASSES = {"whitespace", "latex_command", "markdown"}


def run_d0(config: BPAConfig, dataset: str, max_problems: int | None, max_steps: int) -> list[dict]:
    slm, _ = init_engines(config)
    rows: list[dict] = []
    for problem in tqdm(load_eval_dataset(dataset, config, max_problems=max_problems), desc="D0 first-token"):
        state = GenerationState(problem_text=problem.problem_text)
        for _ in range(max_steps):
            boundary = len(state.assistant_prefix_text)
            l0 = l0_filter(state, slm, config)
            top_classes = [classify_first_char(tok) for tok in l0.top_token_strs]
            skipped_idx = next((i for i, cls in enumerate(top_classes) if cls not in SKIP_CLASSES), None)
            rows.append(
                {
                    "problem_id": problem.problem_id,
                    "step_idx": state.step_count,
                    "boundary_pos_in_assistant_prefix": boundary,
                    "main_first_token_str": l0.top_token_strs[0] if l0.top_token_strs else "",
                    "main_first_char_class": l0.first_char_class,
                    "main_h_init": l0.h_init,
                    "main_margin": l0.margin,
                    "main_l0_passed": l0.passed,
                    "skipped_first_token_str": l0.top_token_strs[skipped_idx] if skipped_idx is not None else None,
                    "skipped_first_char_class": top_classes[skipped_idx] if skipped_idx is not None else None,
                    "skipped_h_init": None,
                    "skipped_margin": None,
                    "skipped_l0_passed": None,
                    "skip_changed_decision": False,
                    "top_k_token_strs": l0.top_token_strs,
                    "top_k_char_classes": top_classes,
                    "arbitration_triggered": False,
                    "arbitration_changed_winner": False,
                    "final_correct": None,
                }
            )
            text, finish = _slm_generate_step(state, slm, config)
            state.assistant_prefix_text += ensure_step_terminator(text, finish)
            state.step_count += 1
            if finish == "eos":
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp-D0 first-token diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    rows = run_d0(config, args.dataset, args.max_problems, args.max_steps)
    out = Path(config.output_dir) / "diagnostics" / "d0_first_token" / f"{args.dataset}.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
