from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.engines import finish_reason, generated_text, generated_token_ids, init_engines
from bpa.render import render_for_continuation
from bpa.trace import write_jsonl

from .benchmark_eval import benchmark_eval_match


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def continue_branch(problem_text: str, assistant_prefix_text: str, branch_text: str, slm, config: BPAConfig) -> str:
    prefix = assistant_prefix_text + branch_text
    rendered = render_for_continuation(problem_text, prefix, slm.ensure_tokenizer())
    sampling = slm.sampling_params(max_tokens=4096, temperature=0.0)
    out = slm.generate(rendered, sampling)[0]
    _ = generated_token_ids(out), finish_reason(out)
    return prefix + generated_text(out)


def label_pair(traj1: str, traj2: str, gold_answer: str, dataset: str) -> str:
    ok1 = benchmark_eval_match(traj1, gold_answer, dataset)
    ok2 = benchmark_eval_match(traj2, gold_answer, dataset)
    if ok1 and ok2:
        return "both_correct"
    if ok1:
        return "only_1"
    if ok2:
        return "only_2"
    return "both_wrong"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp-D2 L2 oracle calibration from branch logs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--branches-jsonl", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gold-json", required=True, help="JSON map from problem_id to gold answer.")
    parser.add_argument("--max-records", type=int, default=550)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    slm, _ = init_engines(config)
    branch_rows = _read_jsonl(Path(args.branches_jsonl))[: args.max_records]
    gold_map = json.loads(Path(args.gold_json).read_text(encoding="utf-8"))
    rows = []
    for row in tqdm(branch_rows, desc="D2 oracle"):
        problem_id = str(row["problem_id"]) if row.get("problem_id") is not None else str(row.get("step_idx"))
        gold = gold_map.get(problem_id)
        if gold is None:
            continue
        traj1 = continue_branch(row["problem_text"], row["assistant_prefix_text"], row["branch1"]["step_branch_text"], slm, config)
        traj2 = continue_branch(row["problem_text"], row["assistant_prefix_text"], row["branch2"]["step_branch_text"], slm, config)
        rows.append(
            {
                "problem_id": problem_id,
                "step_idx": row["step_idx"],
                "oracle_label": label_pair(traj1, traj2, gold, args.dataset),
                "l2": row["l2"],
                "traj1": traj1,
                "traj2": traj2,
            }
        )
    out = Path(config.output_dir) / "diagnostics" / "d2_oracle_l2" / f"{args.dataset}.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
