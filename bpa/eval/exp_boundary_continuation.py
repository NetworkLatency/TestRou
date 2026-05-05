from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bpa.config import BPAConfig
from bpa.context_budget import ContextBudgetExceeded, generation_budget_for_rendered
from bpa.engines import finish_reason, generated_text, generated_token_ids, init_engines
from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import EvalProblem, load_eval_dataset
from bpa.render import render_for_continuation
from bpa.safety import extract_answer
from bpa.trace import json_safe


CSV_FIELDS = [
    "dataset",
    "problem_id",
    "question_id",
    "boundary_idx",
    "selected_rank",
    "prefix_char_len",
    "prefix_token_len",
    "slm_final_answer",
    "slm_final_correct",
    "llm_oracle_answer",
    "llm_oracle_correct",
    "llm_continuation_answer",
    "llm_continuation_correct",
    "critical",
    "label_reason",
    "continuation_wall_time",
    "continuation_decode_tokens",
    "continuation_prefill_tokens",
    "operation_vote_disagreement",
    "number_vote_disagreement",
    "novel_number_vote_disagreement",
    "rhs_number_vote_disagreement",
    "self_bleu_disagreement",
    "char_jaccard_disagreement",
    "structured_disagreement",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv_by_problem(path: Path | None, key: str = "problem_id") -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row[key]): row for row in csv.DictReader(f) if row.get(key) not in (None, "")}


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if value is True or str(value).strip().lower() == "true":
        return True
    if value is False or str(value).strip().lower() == "false":
        return False
    return None


def _is_initial_probe(row: dict[str, Any]) -> bool:
    parsed = _parse_bool(row.get("is_initial_probe"))
    if parsed is not None:
        return parsed
    try:
        if int(row.get("boundary_idx", 0)) < 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return int(row.get("prefix_char_len", 1)) == 0
    except (TypeError, ValueError):
        return False


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def select_evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    rows = [row for row in rows if not _is_initial_probe(row)]
    rows = sorted(rows, key=lambda row: int(row.get("boundary_idx", 0)))
    if count <= 0 or len(rows) <= count:
        return rows
    if count == 1:
        return [rows[0]]
    selected_indices = []
    for idx in range(count):
        pos = round(idx * (len(rows) - 1) / (count - 1))
        if pos not in selected_indices:
            selected_indices.append(pos)
    return [rows[pos] for pos in selected_indices]


def _default_probe_path(config: BPAConfig, dataset: str) -> Path:
    return Path(config.output_dir) / "diagnostics" / "sampling_disagreement" / dataset / "probes.jsonl"


def _default_problem_summary_path(config: BPAConfig, dataset: str) -> Path:
    return Path(config.output_dir) / "diagnostics" / "sampling_disagreement" / dataset / "problem_summary.csv"


def _default_oracle_summary_path(config: BPAConfig, dataset: str) -> Path:
    return Path(config.output_dir) / "diagnostics" / "llm_oracle" / dataset / "oracle_summary.csv"


def continue_boundary_with_llm(
    problem_text: str,
    assistant_prefix_text: str,
    llm,
    config: BPAConfig,
    max_tokens: int,
) -> dict[str, Any]:
    rendered = render_for_continuation(problem_text, assistant_prefix_text, llm.ensure_tokenizer())
    try:
        generation_max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, llm, config, max_tokens)
    except ContextBudgetExceeded as exc:
        return {
            "continuation_text": "",
            "full_text": assistant_prefix_text,
            "answer": extract_answer(assistant_prefix_text),
            "finish_reason": "context_budget",
            "wall_time": 0.0,
            "decode_tokens": 0,
            "prefill_tokens": exc.prompt_tokens,
        }
    sampling = llm.sampling_params(max_tokens=generation_max_tokens, temperature=0.0)
    generate_start = time.time()
    out = llm.generate(rendered, sampling)[0]
    wall_time = time.time() - generate_start
    continuation_text = generated_text(out)
    full_text = assistant_prefix_text + continuation_text
    return {
        "continuation_text": continuation_text,
        "full_text": full_text,
        "answer": extract_answer(full_text),
        "finish_reason": finish_reason(out),
        "wall_time": wall_time,
        "decode_tokens": len(generated_token_ids(out)),
        "prefill_tokens": prompt_tokens,
    }


def make_boundary_label(
    *,
    slm_final_correct: bool | None,
    llm_oracle_correct: bool | None,
    llm_continuation_correct: bool | None,
) -> tuple[bool | None, str]:
    if slm_final_correct is None or llm_continuation_correct is None:
        return None, "missing_gold_or_continuation_eval"
    if llm_oracle_correct is False:
        return False, "oracle_incorrect"
    if slm_final_correct:
        return False, "slm_already_correct"
    if llm_continuation_correct:
        return True, "slm_wrong_llm_boundary_recovers"
    return False, "slm_wrong_llm_boundary_not_recovered"


def build_boundary_label_rows(
    *,
    dataset: str,
    problems: list[EvalProblem],
    probes: list[dict[str, Any]],
    problem_summary: dict[str, dict[str, Any]],
    oracle_summary: dict[str, dict[str, Any]],
    llm,
    config: BPAConfig,
    boundaries_per_problem: int,
    continuation_max_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    problems_by_id = {str(problem.problem_id): problem for problem in problems}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in probes:
        problem_id = str(row.get("problem_id"))
        if problem_id in problems_by_id:
            grouped[problem_id].append(row)

    csv_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    selected_items: list[tuple[str, int, dict[str, Any]]] = []
    for problem_id, rows in grouped.items():
        for selected_rank, row in enumerate(select_evenly_spaced(rows, boundaries_per_problem)):
            selected_items.append((problem_id, selected_rank, row))

    for problem_id, selected_rank, probe in tqdm(selected_items, desc=f"boundary_continuation:{dataset}"):
        problem = problems_by_id[problem_id]
        problem_row = problem_summary.get(problem_id, {})
        oracle_row = oracle_summary.get(problem_id, {})
        continuation = continue_boundary_with_llm(
            problem.problem_text,
            str(probe.get("assistant_prefix_text") or ""),
            llm,
            config,
            max_tokens=continuation_max_tokens,
        )
        llm_continuation_correct = None
        if problem.gold_answer is not None:
            llm_continuation_correct = benchmark_eval_match(continuation["answer"], problem.gold_answer, dataset)

        slm_final_correct = _parse_bool(_first_present(problem_row.get("correct"), probe.get("final_correct")))
        llm_oracle_correct = _parse_bool(oracle_row.get("llm_correct"))
        critical, label_reason = make_boundary_label(
            slm_final_correct=slm_final_correct,
            llm_oracle_correct=llm_oracle_correct,
            llm_continuation_correct=llm_continuation_correct,
        )
        base = {
            "dataset": dataset,
            "problem_id": problem.problem_id,
            "question_id": problem.question_id,
            "boundary_idx": probe.get("boundary_idx"),
            "selected_rank": selected_rank,
            "prefix_char_len": probe.get("prefix_char_len"),
            "prefix_token_len": probe.get("prefix_token_len"),
            "slm_final_answer": _first_present(problem_row.get("final_answer"), probe.get("final_answer")),
            "slm_final_correct": slm_final_correct,
            "llm_oracle_answer": oracle_row.get("llm_answer"),
            "llm_oracle_correct": llm_oracle_correct,
            "llm_continuation_answer": continuation["answer"],
            "llm_continuation_correct": llm_continuation_correct,
            "critical": critical,
            "label_reason": label_reason,
            "continuation_wall_time": continuation["wall_time"],
            "continuation_decode_tokens": continuation["decode_tokens"],
            "continuation_prefill_tokens": continuation["prefill_tokens"],
            "operation_vote_disagreement": probe.get("operation_vote_disagreement"),
            "number_vote_disagreement": probe.get("number_vote_disagreement"),
            "novel_number_vote_disagreement": probe.get("novel_number_vote_disagreement"),
            "rhs_number_vote_disagreement": probe.get("rhs_number_vote_disagreement"),
            "self_bleu_disagreement": probe.get("self_bleu_disagreement"),
            "char_jaccard_disagreement": probe.get("char_jaccard_disagreement"),
            "structured_disagreement": probe.get("structured_disagreement"),
        }
        csv_rows.append(base)
        jsonl_rows.append(
            {
                **base,
                "gold_answer": problem.gold_answer,
                "assistant_prefix_text": probe.get("assistant_prefix_text"),
                "main_step_text": probe.get("main_step_text"),
                "continuation_text": continuation["continuation_text"],
                "full_text": continuation["full_text"],
                "continuation_finish_reason": continuation["finish_reason"],
                "probe": probe,
            }
        )
    return csv_rows, jsonl_rows


def write_boundary_label_outputs(out_dir: Path, csv_rows: list[dict[str, Any]], jsonl_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "boundary_labels.csv"
    fieldnames = list(CSV_FIELDS)
    extra_fields = sorted({key for row in csv_rows for key in row} - set(fieldnames))
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(json_safe(csv_rows))

    jsonl_path = out_dir / "boundary_labels.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label sampled SLM boundaries by asking the LLM to continue from each prefix.")
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--probes-path", default=None)
    parser.add_argument("--problem-summary", default=None)
    parser.add_argument("--oracle-summary", default=None)
    parser.add_argument("--boundaries-per-problem", type=int, default=5)
    parser.add_argument("--continuation-max-tokens", type=int, default=2048)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    probe_path = Path(args.probes_path) if args.probes_path else _default_probe_path(config, args.dataset)
    problem_summary_path = Path(args.problem_summary) if args.problem_summary else _default_problem_summary_path(config, args.dataset)
    oracle_summary_path = Path(args.oracle_summary) if args.oracle_summary else _default_oracle_summary_path(config, args.dataset)
    probes = _read_jsonl(probe_path)
    problem_summary = _read_csv_by_problem(problem_summary_path)
    oracle_summary = _read_csv_by_problem(oracle_summary_path)

    _, llm = init_engines(config)
    csv_rows, jsonl_rows = build_boundary_label_rows(
        dataset=args.dataset,
        problems=problems,
        probes=probes,
        problem_summary=problem_summary,
        oracle_summary=oracle_summary,
        llm=llm,
        config=config,
        boundaries_per_problem=args.boundaries_per_problem,
        continuation_max_tokens=args.continuation_max_tokens,
    )

    out_dir = Path(config.output_dir) / "diagnostics" / "boundary_continuation" / args.dataset
    write_boundary_label_outputs(out_dir, csv_rows, jsonl_rows)
    print(f"Wrote {out_dir / 'boundary_labels.csv'}")
    print(f"Wrote {out_dir / 'boundary_labels.jsonl'}")


if __name__ == "__main__":
    main()
