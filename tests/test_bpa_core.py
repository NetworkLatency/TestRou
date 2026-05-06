from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import unittest

from bpa.config import BPAConfig
from bpa.cascade.l0 import entropy_and_margin
from bpa.eval.benchmark_eval import benchmark_eval_match, normalize_math_expr
from bpa.eval.datasets import EvalProblem, load_eval_dataset, load_local_rows
from bpa.eval.analyze_sampling_disagreement import analyze, auroc, quantile_rows
from bpa.eval.exp_boundary_continuation import (
    build_boundary_label_rows,
    make_boundary_label,
    select_evenly_spaced,
)
from bpa.eval.exp_disagreement_routing import run_disagreement_routing, threshold_from_probes
from bpa.eval.exp_sampling_disagreement import (
    build_problem_summary,
    enrich_probe_rows,
    run_sampling_disagreement,
    write_sampling_outputs,
)
from bpa.eval.main_benchmark import (
    _existing_row_from_problem_output,
    build_summary_metrics,
    has_complete_problem_outputs,
    write_summary_files,
)
from bpa.eval.sampling_disagreement import (
    char_jaccard_disagreement,
    compute_vote_disagreement,
    extract_novel_number_signature,
    extract_number_signature,
    extract_operation_signature,
    extract_rhs_number_signature,
    extract_structured_signature,
    novel_number_vote_disagreement,
    number_vote_disagreement,
    operation_vote_disagreement,
    rhs_number_vote_disagreement,
    rollout_disagreement_metrics,
    score_variance,
    self_bleu_disagreement,
)
from bpa.engines import ModelEngine, completion_logprobs, generated_text, generated_token_ids
from bpa.pipeline import bpa_solve
from bpa.render import render_for_continuation
from bpa.safety import clean_latex_answer, ensure_step_terminator, extract_answer, update_repetition, update_strict_step_repetition
from bpa.state import GenerationState, RepetitionState


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


class SamplingProbeEngine(SequencedEngine):
    def __init__(self, outputs=None, probe_outputs=None):
        super().__init__(outputs)
        self.probe_outputs = list(probe_outputs or [])

    def generate(self, prompts, sampling_params):
        if sampling_params.get("n"):
            self.generate_calls += 1
            k = int(sampling_params["n"])
            if len(self.probe_outputs) < k:
                raise AssertionError("no queued probe outputs")
            completions = []
            for _ in range(k):
                text = self.probe_outputs.pop(0)
                token_ids = list(range(len(text)))
                completions.append(
                    Obj(
                        text=text,
                        token_ids=token_ids,
                        finish_reason="stop",
                        logprobs=[{token_id: Obj(logprob=-0.2 - 0.01 * token_id)} for token_id in token_ids],
                    )
                )
            return [Obj(outputs=completions)]
        return super().generate(prompts, sampling_params)


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
        self.assertEqual(render_for_continuation("p", "", tok), "USER:p\nASSISTANT:")
        self.assertEqual(render_for_continuation("p", "bad", tok), "USER:p\nASSISTANT:bad")

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

    def test_structured_signature_priority(self):
        self.assertEqual(extract_structured_signature(r"final \boxed{42}")["signature"], "boxed:42")
        self.assertEqual(extract_structured_signature("Use x + 1 = 3 first.")["signature_type"], "equation")
        self.assertEqual(extract_structured_signature(r"Take \frac{1}{2} of both sides.")["signature_type"], "number")
        self.assertEqual(extract_structured_signature("We substitute the value.")["signature"], "operator:substitute")
        self.assertEqual(extract_structured_signature("No math event here.")["signature"], "none:")
        self.assertEqual(extract_number_signature("First value is 17.")["signature"], "number:17")
        self.assertEqual(extract_operation_signature("We simplify the equation.")["signature"], "operator:simplify")

    def test_context_aware_number_signatures(self):
        context = "Problem: the point is (0, 3)."
        self.assertEqual(extract_number_signature("Copy (0, 3), then y = 7.")["signature"], "number:0")
        self.assertEqual(extract_novel_number_signature("Copy (0, 3), then y = 7.", context)["signature"], "number:7")
        self.assertEqual(extract_rhs_number_signature("Copy (0, 3), then y = 7.", context)["signature"], "number:7")

        texts = ["copy 0 then y = 5", "copy 0 then y = 5", "copy 0 then y = 6", "copy 0 then y = 5"]
        self.assertAlmostEqual(novel_number_vote_disagreement(texts, "Problem gives 0")["novel_number_vote_disagreement"], 0.25)
        self.assertAlmostEqual(rhs_number_vote_disagreement(texts, "Problem gives 0")["rhs_number_vote_disagreement"], 0.25)

    def test_vote_disagreement(self):
        result = compute_vote_disagreement(["a", "a", "a", "b"])
        self.assertEqual(result["signature_counts"], {"a": 3, "b": 1})
        self.assertAlmostEqual(result["vote_fraction"], 0.75)
        self.assertAlmostEqual(result["structured_disagreement"], 0.25)

        result = compute_vote_disagreement(["a", "b", "c", "d"])
        self.assertAlmostEqual(result["structured_disagreement"], 0.75)

    def test_char_jaccard_disagreement(self):
        self.assertAlmostEqual(char_jaccard_disagreement(["abc def", "abc def"]), 0.0)
        self.assertGreater(char_jaccard_disagreement(["abc", "xyz"]), 0.9)

    def test_operation_number_and_self_bleu_metrics(self):
        texts = ["solve x = 1", "solve x = 1", "calculate x = 2", "solve x = 1"]
        self.assertAlmostEqual(operation_vote_disagreement(texts)["operation_vote_disagreement"], 0.25)
        self.assertAlmostEqual(number_vote_disagreement(texts)["number_vote_disagreement"], 0.25)
        self.assertLess(self_bleu_disagreement(["same tokens", "same tokens"]), 0.1)
        self.assertGreater(self_bleu_disagreement(["alpha beta", "gamma delta"]), 0.0)
        metrics = rollout_disagreement_metrics(texts, [-1.0, -1.1, -1.2, -1.0])
        self.assertIn("operation_vote_disagreement", metrics)
        self.assertIn("number_vote_disagreement", metrics)
        self.assertIn("novel_number_vote_disagreement", metrics)
        self.assertIn("rhs_number_vote_disagreement", metrics)
        self.assertIn("self_bleu_disagreement", metrics)

    def test_score_variance(self):
        self.assertAlmostEqual(score_variance([-1.0, -2.0, None]), 0.25)
        self.assertIsNone(score_variance([None, -1.0]))

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

    def test_sampling_disagreement_fake_problem_writes_outputs(self):
        slm = SamplingProbeEngine(
            outputs=[
                ("Let x = 1.\n\n", "stop"),
                ("</think>\n\n", "stop"),
                (r"Final answer: \boxed{1}", "eos"),
            ],
            probe_outputs=[
                "x = 1",
                "x = 1",
                "x = 2",
                "x = 1",
                "solve for x",
                "solve for x",
                "calculate 2",
                "solve for x",
                "answer is 1",
                "answer is 1",
                "answer is 2",
                "answer is 1",
            ],
        )
        result, probes, probe_cost = run_sampling_disagreement(
            "Problem: x?",
            slm,
            BPAConfig(max_total_tokens=200),
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "1")
        self.assertEqual(len(probes), 3)
        self.assertEqual(probes[0]["boundary_idx"], -1)
        self.assertTrue(probes[0]["is_initial_probe"])
        self.assertEqual(probes[0]["prefix_char_len"], 0)
        self.assertEqual(probes[1]["boundary_idx"], 0)
        self.assertFalse(probes[1]["is_initial_probe"])
        self.assertEqual(probes[2]["boundary_idx"], 1)
        self.assertAlmostEqual(probes[0]["structured_disagreement"], 0.25)
        self.assertEqual(probe_cost["probe_generate_calls"], 3)

        problem = EvalProblem(problem_id=1, question_id="q1", problem_text="Problem: x?", raw={}, gold_answer="1")
        summary = build_problem_summary(problem, result, probes, probe_cost, "math500")
        self.assertEqual(summary["num_boundaries"], 2)
        self.assertEqual(summary["num_initial_probes"], 1)
        self.assertNotIn("num_thinking_boundaries", summary)
        self.assertNotIn("num_final_answer_boundaries", summary)
        enriched = enrich_probe_rows(problem, probes, result, "math500")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "sampling"
            write_sampling_outputs(out_dir, enriched, [summary])
            self.assertTrue((out_dir / "probes.jsonl").exists())
            self.assertTrue((out_dir / "problem_summary.csv").exists())
            self.assertEqual(len((out_dir / "probes.jsonl").read_text(encoding="utf-8").splitlines()), 3)

    def test_boundary_label_helpers(self):
        rows = [{"boundary_idx": -1, "is_initial_probe": True}] + [{"boundary_idx": idx} for idx in range(10)]
        selected = select_evenly_spaced(rows, 5)
        self.assertEqual([row["boundary_idx"] for row in selected], [0, 2, 4, 7, 9])
        self.assertEqual(make_boundary_label(slm_final_correct=False, llm_oracle_correct=True, llm_continuation_correct=True)[0], True)
        self.assertEqual(make_boundary_label(slm_final_correct=True, llm_oracle_correct=True, llm_continuation_correct=True)[0], False)

    def test_boundary_continuation_fake_label(self):
        problem = EvalProblem(problem_id=1, question_id="q1", problem_text="Problem: x?", raw={}, gold_answer="1")
        initial_probe = {
            "problem_id": 1,
            "question_id": "q1",
            "boundary_idx": -1,
            "is_initial_probe": True,
            "assistant_prefix_text": "",
            "prefix_char_len": 0,
            "prefix_token_len": 10,
            "operation_vote_disagreement": 1.0,
            "number_vote_disagreement": 1.0,
            "novel_number_vote_disagreement": 1.0,
            "rhs_number_vote_disagreement": 1.0,
            "self_bleu_disagreement": 1.0,
            "char_jaccard_disagreement": 1.0,
            "structured_disagreement": 1.0,
        }
        probe = {
            "problem_id": 1,
            "question_id": "q1",
            "boundary_idx": 0,
            "assistant_prefix_text": "Let x be unknown.\n\n",
            "prefix_char_len": 20,
            "prefix_token_len": 20,
            "operation_vote_disagreement": 0.25,
            "number_vote_disagreement": 0.75,
            "novel_number_vote_disagreement": 0.5,
            "rhs_number_vote_disagreement": 0.5,
            "self_bleu_disagreement": 0.5,
            "char_jaccard_disagreement": 0.5,
            "structured_disagreement": 0.75,
        }
        llm = SequencedEngine([(r"Thus \boxed{1}", "eos")])
        csv_rows, jsonl_rows = build_boundary_label_rows(
            dataset="math500",
            problems=[problem],
            probes=[initial_probe, probe],
            problem_summary={"1": {"correct": False, "final_answer": "2"}},
            oracle_summary={"1": {"llm_correct": True, "llm_answer": "1"}},
            llm=llm,
            config=BPAConfig(),
            boundaries_per_problem=1,
            continuation_max_tokens=64,
        )
        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(csv_rows[0]["boundary_idx"], 0)
        self.assertTrue(csv_rows[0]["critical"])
        self.assertEqual(csv_rows[0]["rhs_number_vote_disagreement"], 0.5)
        self.assertIn("full_text", jsonl_rows[0])

    def test_analysis_helpers(self):
        self.assertAlmostEqual(auroc([0.1, 0.2, 0.9, 1.0], [False, False, True, True]), 1.0)
        rows = quantile_rows([0.1, 0.2, 0.9, 1.0], [False, False, True, True], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["critical_rate"], 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            probes = [
                {"boundary_idx": -1, "is_initial_probe": True},
                {"boundary_idx": 0, "is_initial_probe": False},
            ]
            summary = analyze(probes, [], [], 10, Path(tmp) / "analysis")
            self.assertEqual(summary["num_probes_raw"], 2)
            self.assertEqual(summary["num_probes"], 1)
            summary = analyze(probes, [], [], 10, Path(tmp) / "analysis_include", include_initial_probe=True)
            self.assertEqual(summary["num_probes"], 2)

    def test_disagreement_routing_fake_problem(self):
        slm = SamplingProbeEngine(
            outputs=[
                ("Let x = 0.\n\n", "stop"),
                (r"\boxed{1}", "eos"),
            ],
            probe_outputs=[
                "x = 1",
                "x = 1",
                "x = 1",
                "x = 1",
                "x = 1",
                "x = 2",
                "x = 3",
                "x = 4",
                "answer = 1",
                "answer = 1",
                "answer = 1",
                "answer = 1",
            ],
        )
        llm = SequencedEngine([("</think>\n\n", "stop")])
        result, boundaries, probe_cost = run_disagreement_routing(
            "Problem: x?",
            slm,
            llm,
            BPAConfig(max_total_tokens=200),
            metric="rhs_number_vote_disagreement",
            threshold=0.5,
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "1")
        self.assertEqual(len(boundaries), 3)
        self.assertEqual(boundaries[0]["boundary_idx"], -1)
        self.assertFalse(boundaries[0]["routed_to_llm"])
        self.assertTrue(boundaries[1]["routed_to_llm"])
        self.assertFalse(boundaries[2]["routed_to_llm"])
        self.assertEqual(probe_cost["probe_generate_calls"], 3)

    def test_threshold_from_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probes.jsonl"
            rows = [
                {"boundary_idx": -1, "is_initial_probe": True, "rhs_number_vote_disagreement": 1.0},
                {"boundary_idx": 0, "is_initial_probe": False, "rhs_number_vote_disagreement": 0.0},
                {"boundary_idx": 1, "is_initial_probe": False, "rhs_number_vote_disagreement": 0.25},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            self.assertAlmostEqual(threshold_from_probes(path, "rhs_number_vote_disagreement", 0.5), 0.125)
            self.assertAlmostEqual(
                threshold_from_probes(path, "rhs_number_vote_disagreement", 0.5, include_initial_probe=True),
                0.25,
            )

    def test_unified_stepwise_continues_after_close_think(self):
        slm = SequencedEngine([("</think>\n\n", "stop"), ("The answer is 42.", "eos")])
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=200))
        self.assertEqual(result.state.stop_reason, "eos")
        self.assertEqual(llm.generate_calls, 0)
        self.assertIn("The answer is 42.", result.state.assistant_prefix_text)

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

    def test_unified_stops_on_duplicate_step(self):
        slm = SequencedEngine(
            [
                ("Same final step.\n\n", "stop"),
                ("Same final step.", "stop"),
            ]
        )
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=300))
        self.assertEqual(result.state.stop_reason, "duplicate_step")
        self.assertTrue(any(event.event == "step_repetition_stop" for event in result.state.trace))

    def test_unified_context_budget_stops_before_generation(self):
        slm = SequencedEngine(fail_on_generate=True)
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("long prompt", slm, llm, BPAConfig(max_model_len=8, max_total_tokens=100))
        self.assertEqual(result.state.stop_reason, "context_budget")
        self.assertEqual(slm.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
