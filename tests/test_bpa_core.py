from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import unittest

from bpa.config import BPAConfig
from bpa.cascade.l0 import entropy_and_margin
from bpa.eval.benchmark_eval import benchmark_eval_match, normalize_math_expr
from bpa.eval.datasets import load_eval_dataset, load_local_rows
from bpa.eval.main_benchmark import (
    _existing_row_from_problem_output,
    build_summary_metrics,
    has_complete_problem_outputs,
    write_summary_files,
)
from bpa.engines import ModelEngine, completion_logprobs, generated_text, generated_token_ids
from bpa.pipeline import bpa_solve
from bpa.render import render_for_continuation
from bpa.safety import (
    clean_latex_answer,
    ensure_step_terminator,
    extract_answer,
    extract_answer_from_final_step,
    extract_answer_from_steps,
    update_repetition,
    update_strict_step_repetition,
)
from bpa.state import RepetitionState


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTokenizer:
    is_fast = True

    def __init__(self, table=None, eos_token_id=None):
        self.table = table or {}
        self.eos_token_id = eos_token_id

    def apply_chat_template(self, messages, tokenize=False, continue_final_message=True, add_generation_prompt=False):
        if add_generation_prompt:
            assert continue_final_message is False
            return f"USER:{messages[0]['content']}\nASSISTANT:"
        assert continue_final_message is True
        assert add_generation_prompt is False
        return f"USER:{messages[0]['content']}\nASSISTANT:{messages[1]['content']}"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        pieces = []
        for i in ids:
            if skip_special_tokens and self.eos_token_id is not None and i == self.eos_token_id:
                continue
            pieces.append(self.table.get(i, chr(i)))
        return "".join(pieces)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {
            "input_ids": [ord(ch) for ch in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class SequencedEngine:
    def __init__(self, outputs=None, fail_on_generate=False):
        self.tokenizer = FakeTokenizer()
        self.outputs = list(outputs or [])
        self.fail_on_generate = fail_on_generate
        self.generate_calls = 0
        self.sampling_history = []

    def ensure_tokenizer(self):
        return self.tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def sampling_params(self, **kwargs):
        return kwargs

    def tokens_prompt(self, prompt_token_ids):
        return prompt_token_ids

    def generate(self, prompts, sampling_params):
        self.generate_calls += 1
        self.sampling_history.append(dict(sampling_params))
        if self.fail_on_generate:
            raise AssertionError("generate should not have been called")
        if sampling_params.get("max_tokens") == 1 and sampling_params.get("logprobs") is not None:
            return [
                Obj(
                    outputs=[
                        Obj(
                            text="A",
                            token_ids=[ord("A")],
                            finish_reason="length",
                            logprobs=[{ord("A"): Obj(logprob=0.0)}],
                        )
                    ],
                )
            ]
        if not self.outputs:
            raise AssertionError("no queued generation output")
        text, finish = self.outputs.pop(0)
        token_ids = list(range(len(text)))
        return [
            Obj(
                outputs=[
                    Obj(
                        text=text,
                        token_ids=token_ids,
                        finish_reason=finish,
                        logprobs=[{token_id: Obj(logprob=-0.1)} for token_id in token_ids],
                    )
                ],
            )
        ]


class EmptyAssistantRejectingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, continue_final_message=True, add_generation_prompt=False):
        if continue_final_message and messages[-1]["role"] == "assistant":
            if messages[-1]["content"] == "" or messages[-1]["content"] == "bad":
                raise ValueError(
                    "continue_final_message is set but the final message does not appear in the chat after applying the chat template!"
                )
        return super().apply_chat_template(messages, tokenize, continue_final_message, add_generation_prompt)


class CoreTests(unittest.TestCase):
    def test_render_empty_prefix_uses_generation_prompt(self):
        tok = EmptyAssistantRejectingTokenizer()
        self.assertEqual(render_for_continuation("p", "", tok), "USER:p\nASSISTANT:<think>")
        self.assertEqual(render_for_continuation("p", "bad", tok), "USER:p\nASSISTANT:<think>bad")

    def test_step_terminator_and_repetition(self):
        self.assertEqual(ensure_step_terminator("abc", "stop"), "abc\n\n")
        self.assertEqual(ensure_step_terminator("abc", "eos"), "abc")
        rep = RepetitionState()
        self.assertIsNone(update_repetition(rep, "This is a long enough step."))
        self.assertEqual(update_repetition(rep, "This is a long enough step."), "duplicate_step")

        strict_rep = RepetitionState()
        internally_repetitive = r"\frac{1}{2} + \frac{1}{2} + \frac{1}{2} + \frac{1}{2}."
        self.assertIsNone(update_strict_step_repetition(strict_rep, internally_repetitive))
        self.assertEqual(update_strict_step_repetition(strict_rep, internally_repetitive), "duplicate_step")

    def test_l0_entropy_margin(self):
        h, margin = entropy_and_margin({1: 0.0, 2: math.log(0.5)})
        self.assertGreater(h, 0.0)
        self.assertGreater(margin, 0.0)

    def test_openai_backend_wraps_completion_like_vllm_output(self):
        engine = ModelEngine(
            name="llm",
            model_path="Qwen3-14B",
            backend="openai",
            api_base_url="http://192.168.3.13:8080/v1",
            api_model="Qwen3-14B",
        )
        engine.tokenizer = FakeTokenizer()
        sampling = engine.sampling_params(
            max_tokens=8,
            temperature=0.0,
            stop=["\n\n"],
            include_stop_str_in_output=True,
            logprobs=1,
        )
        kwargs = engine._openai_completion_kwargs(sampling)
        self.assertEqual(kwargs["max_tokens"], 8)
        self.assertEqual(kwargs["logprobs"], 1)
        self.assertTrue(kwargs["extra_body"]["include_stop_str_in_output"])

        completion = engine._openai_choice_to_completion(
            Obj(
                text="AB",
                finish_reason="stop",
                logprobs=Obj(
                    tokens=["A", "B"],
                    token_logprobs=[-0.1, -0.2],
                    top_logprobs=[{"A": -0.1}, {"B": -0.2}],
                ),
            )
        )
        output = Obj(outputs=[completion])
        self.assertEqual(generated_text(output), "AB")
        self.assertEqual(generated_token_ids(output), [ord("A"), ord("B")])
        self.assertAlmostEqual(completion_logprobs(output)[0][ord("A")].logprob, -0.1)

    def test_answer_eval(self):
        self.assertEqual(normalize_math_expr(r"\frac{4}{2}"), "2")
        self.assertTrue(benchmark_eval_match(r"final \boxed{42}", "42", "aime25"))
        self.assertTrue(benchmark_eval_match(r"final \boxed{B}", "B", "gpqa"))

    def test_latex_answer_cleanup_is_minimal(self):
        self.assertEqual(clean_latex_answer(r"$\dfrac{14}{3}$"), r"\frac{14}{3}")
        self.assertEqual(clean_latex_answer(r"11\sqrt2"), r"11\sqrt{2}")
        self.assertTrue(benchmark_eval_match(r"\dfrac{14}{3}", r"\frac{14}{3}", "math500"))
        self.assertTrue(benchmark_eval_match(r"11\sqrt2", r"11\sqrt{2}", "math500"))
        self.assertFalse(benchmark_eval_match("5", "x=5", "math500"))
        self.assertFalse(benchmark_eval_match("52", "52_8", "math500"))

    def test_extract_answer_cleans_latex_only(self):
        self.assertEqual(extract_answer(r"done \boxed{\dfrac{3}{56}}"), r"\frac{3}{56}")
        self.assertEqual(extract_answer(r"reasoning</think> $11\sqrt2$"), r"11\sqrt{2}")

    def test_extract_answer_from_steps_uses_only_final_step(self):
        steps = [
            {"step_text": r"tentative \boxed{999}\n\n"},
            {"step_text": r"corrected final \boxed{\dfrac{3}{56}}"},
        ]
        self.assertEqual(extract_answer_from_steps(steps, r"fallback \boxed{0}"), r"\frac{3}{56}")
        self.assertEqual(extract_answer_from_steps([{"step_text": "No boxed final."}], r"\boxed{0}"), "No boxed final.")
        self.assertIsNone(extract_answer_from_steps([{"step_text": r"old \boxed{9}"}, {"step_text": ""}], r"\boxed{0}"))

    def test_extract_answer_from_final_step_reads_labeled_answer(self):
        self.assertEqual(extract_answer_from_final_step("The final answer is 16."), "16")
        self.assertEqual(extract_answer_from_final_step(r"Answer: \frac{3}{4}."), r"\frac{3}{4}")
        self.assertEqual(extract_answer_from_final_step(r"Final answer: \boxed{16"), "16")
        self.assertEqual(
            extract_answer_from_final_step(r"scratch \boxed{9}</think> The final answer is 16."),
            "16",
        )

    def test_summary_metrics(self):
        rows = [
            {"correct": True, "total_wall_time": 1.0, "problem_wall_time": 2.0},
            {"correct": False, "total_wall_time": 3.0, "problem_wall_time": 4.0},
            {"correct": None, "total_wall_time": 5.0, "problem_wall_time": 6.0},
        ]
        metrics = build_summary_metrics("math500", "slm_only", rows, dataset_wall_time=12.0)
        self.assertEqual(metrics["num_problems"], 3)
        self.assertEqual(metrics["num_evaluated"], 2)
        self.assertEqual(metrics["num_correct"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["avg_total_wall_time"], 3.0)
        self.assertEqual(metrics["avg_problem_wall_time"], 4.0)
        self.assertEqual(metrics["dataset_wall_time"], 12.0)

    def test_resume_helpers_reconstruct_existing_problem_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problem = Obj(problem_id=7, question_id="", gold_answer=r"\frac{14}{3}")
            problem_root = root / "math500" / "glimprouter_hinit" / "7"
            problem_root.mkdir(parents=True)
            saved = {
                "answer": r"\dfrac{14}{3}",
                "correct": False,
                "total_wall_time": 1.5,
                "slm_decode_tokens": 1,
                "slm_prefill_tokens": 2,
                "llm_decode_tokens": 3,
                "llm_prefill_tokens": 4,
                "llm_scoring_calls": 5,
                "llm_full_calls": 6,
                "stop_reason": "final_eos",
            }
            (problem_root / "7.problem.json").write_text(json.dumps(saved), encoding="utf-8")
            (problem_root / "7.steps.jsonl").write_text("", encoding="utf-8")
            (problem_root / "7.trace.json").write_text("[]", encoding="utf-8")

            self.assertTrue(has_complete_problem_outputs(root, "math500", "glimprouter_hinit", 7))
            row = _existing_row_from_problem_output(root, "math500", "glimprouter_hinit", problem, BPAConfig())
            self.assertTrue(row["correct"])
            self.assertEqual(row["problem_wall_time"], 1.5)
            self.assertEqual(row["problem_id"], 7)

    def test_write_summary_files_writes_csv_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "math500" / "slm_only" / "summary.csv"
            rows = [{"problem_id": 1, "correct": True, "total_wall_time": 1.0, "problem_wall_time": 1.2}]
            metrics = build_summary_metrics("math500", "slm_only", rows, dataset_wall_time=1.3)
            write_summary_files(summary_path, rows, metrics)
            self.assertTrue(summary_path.exists())
            self.assertTrue((summary_path.parent / "summary_metrics.json").exists())

    def test_local_dataset_loader_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "math500.jsonl"
            path.write_text(json.dumps({"id": 7, "problem": "1+1?", "answer": "2"}) + "\n", encoding="utf-8")
            config = BPAConfig(dataset_paths={"math500": str(path)})
            rows = load_local_rows(path)
            problems = load_eval_dataset("math500", config)
            self.assertEqual(rows[0]["problem"], "1+1?")
            self.assertEqual(problems[0].problem_id, 7)
            self.assertEqual(problems[0].gold_answer, "2")
            self.assertIn("Problem: 1+1?", problems[0].problem_text)

    def test_local_dataset_loader_json_wrapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aime24.json"
            path.write_text(json.dumps({"data": [{"problem": "x", "target": "y"}]}), encoding="utf-8")
            config = BPAConfig(dataset_paths={"aime24": str(path)})
            problems = load_eval_dataset("aime24", config)
            self.assertEqual(problems[0].gold_answer, "y")

    def test_unified_stepwise_continues_after_close_think(self):
        slm = SequencedEngine([("</think>\n\n", "stop"), ("The answer is 42.", "eos")])
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=200))
        self.assertEqual(result.state.stop_reason, "eos")
        self.assertEqual(llm.generate_calls, 0)
        self.assertIn("The answer is 42.", result.state.assistant_prefix_text)

    def test_unified_final_answer_phase_runs_once_on_stop(self):
        slm = SequencedEngine([("</think>\n\n", "stop"), ("The answer is 42.\n\n", "stop")])
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=200))
        self.assertEqual(result.answer, "42")
        self.assertEqual(result.state.stop_reason, "final_answer_stop")
        self.assertEqual(result.state.slm_generate_calls, 3)
        self.assertEqual(result.state.step_count, 2)

    def test_final_answer_budget_matches_thinking_decode_tokens(self):
        slm = SequencedEngine([("</think>\n\n", "stop"), ("The answer is 42.\n\n", "stop")])
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=200))
        self.assertEqual(result.answer, "42")
        self.assertEqual(slm.sampling_history[-1]["max_tokens"], 11)

    def test_post_stop_lookahead_stops_on_eos(self):
        slm = SequencedEngine([(r"Final answer: \boxed{2}\n\n", "stop"), ("", "stop")])
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=200, post_stop_lookahead_tokens=4))
        self.assertEqual(result.answer, "2")
        self.assertEqual(result.state.stop_reason, "eos")
        self.assertEqual(result.state.step_count, 1)

    def test_post_stop_lookahead_captures_close_think(self):
        slm = SequencedEngine([("reasoning\n\n", "stop"), ("</think>", "length"), (r"Final \boxed{2}", "eos")])
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=200, post_stop_lookahead_tokens=8))
        self.assertEqual(result.answer, "2")
        self.assertIn("</think>", result.state.assistant_prefix_text)
        self.assertEqual(result.state.stop_reason, "eos")

    def test_unified_does_not_stop_on_internal_ngram_repetition(self):
        repeated_phrase = "alpha alpha alpha alpha alpha alpha alpha alpha.\n\n"
        slm = SequencedEngine(
            [
                (repeated_phrase, "stop"),
                ("Different final step.", "eos"),
            ]
        )
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=300))
        self.assertEqual(result.state.stop_reason, "eos")
        self.assertIn("Different final step.", result.state.assistant_prefix_text)
        self.assertFalse(any(event.event == "step_repetition_stop" for event in result.state.trace))

    def test_unified_forces_final_answer_after_thinking_duplicate(self):
        slm = SequencedEngine(
            [
                ("Same thinking step.\n\n", "stop"),
                ("Same thinking step.\n\n", "stop"),
                ("The answer is 42.\n\n", "stop"),
            ]
        )
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=300))
        self.assertEqual(result.answer, "42")
        self.assertEqual(result.state.stop_reason, "final_answer_stop")
        self.assertIn("</think>", result.state.assistant_prefix_text)
        self.assertIn("Do not restart the solution after </think>.", result.state.assistant_prefix_text)
        self.assertTrue(any(event.event == "forced_close_think_for_final_answer" for event in result.state.trace))

    def test_unified_context_budget_stops_before_generation(self):
        slm = SequencedEngine(fail_on_generate=True)
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("long prompt", slm, llm, BPAConfig(max_model_len=8, max_total_tokens=100))
        self.assertEqual(result.state.stop_reason, "context_budget")
        self.assertEqual(slm.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
