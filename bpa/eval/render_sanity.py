from __future__ import annotations

import argparse
import json
from pathlib import Path

from bpa.config import BPAConfig
from bpa.engines import ModelEngine
from bpa.render import chat_template_hash, rendered_initial_assistant_marker, render_for_continuation


def inspect_model(name: str, model_path: str, tokenizer_path: str | None, problem_text: str) -> dict:
    engine = ModelEngine(name=name, model_path=model_path, tokenizer_path=tokenizer_path)
    tokenizer = engine.ensure_tokenizer()
    rendered_empty = render_for_continuation(problem_text, "", tokenizer)
    rendered_close = render_for_continuation(problem_text, "</think>\n\n", tokenizer)
    return {
        "name": name,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "chat_template_hash": chat_template_hash(tokenizer),
        "rendered_initial_assistant_marker": rendered_initial_assistant_marker(problem_text, tokenizer),
        "contains_think_marker_with_empty_prefix": "<think>" in rendered_empty,
        "contains_double_think_with_empty_prefix": rendered_empty.count("<think>") > 1,
        "empty_prefix_preview": rendered_empty[-500:],
        "close_think_prefix_preview": rendered_close[-500:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-1 render_for_continuation sanity check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--problem", default="What is 1+1? Put the final answer in \\boxed{}.")
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    rows = [
        inspect_model("slm", config.slm_model_path, config.slm_tokenizer_path, args.problem),
        inspect_model("llm", config.llm_model_path, config.llm_tokenizer_path, args.problem),
    ]
    out = Path(config.output_dir) / "diagnostics" / "render_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
