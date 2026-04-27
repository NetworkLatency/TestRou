from __future__ import annotations

import argparse
import json
from pathlib import Path

from bpa.cascade.l2 import char_ngram_jaccard
from bpa.trace import write_jsonl


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp-D3 loser commitment recurrence analysis.")
    parser.add_argument("--branches-jsonl", required=True)
    parser.add_argument("--steps-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    branches = _read_jsonl(Path(args.branches_jsonl))
    steps = _read_jsonl(Path(args.steps_jsonl))
    rows = []
    for branch in branches:
        arb = branch.get("arbitration") or {}
        if arb.get("is_invalid") or "winner_idx" not in arb:
            continue
        winner_idx = arb["winner_idx"]
        loser_key = "branch2" if winner_idx == 0 else "branch1"
        loser_text = branch[loser_key]["step_branch_text"]
        later = [s for s in steps if s.get("step_idx", -1) > branch["step_idx"]]
        best = max((char_ngram_jaccard(loser_text, s.get("step_text", ""), n=3) for s in later), default=0.0)
        rows.append(
            {
                "step_idx": branch["step_idx"],
                "loser_text": loser_text,
                "max_future_jaccard": best,
                "recurred": best >= args.threshold,
            }
        )
    write_jsonl(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
