from __future__ import annotations

import argparse
import json
from pathlib import Path

from bpa.safety import update_repetition
from bpa.state import RepetitionState
from bpa.trace import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp-D4 repetition diagnostics over step logs.")
    parser.add_argument("--steps-jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rep = RepetitionState()
    rows = []
    with Path(args.steps_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            step = json.loads(line)
            trigger = update_repetition(rep, step.get("step_text", ""))
            rows.append({"step_idx": step.get("step_idx"), "trigger": trigger, "step_text": step.get("step_text", "")})
    write_jsonl(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
