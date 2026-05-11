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
from bpa.eval.exp_disagreement_routing import (
    _selected_prefix_consensus_rollout,
    build_problem_summary as build_routing_problem_summary,
    compact_boundary_row,
    existing_problem_summary as existing_routing_problem_summary,
    has_complete_problem_outputs as has_complete_routing_problem_outputs,
    run_disagreement_routing,
    write_problem_outputs as write_routing_problem_outputs,
)
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
    extract_content_anchors,
    extract_step_evidence,
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
                token_ids = [ord(ch) for ch in text]
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


class WeightedSamplingProbeEngine(SequencedEngine):
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
                item = self.probe_outputs.pop(0)
                text, mean_logprob, finish = item
                token_ids = [ord(ch) for ch in text]
                completions.append(
                    Obj(
                        text=text,
                        token_ids=token_ids,
                        finish_reason=finish,
                        logprobs=[{token_id: Obj(logprob=mean_logprob)} for token_id in token_ids],
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

    def test_step_evidence_splits_numbers_and_intent(self):
        evidence = extract_step_evidence("Now solve x = 3/4.\n\n", "Problem mentions 100.")
        self.assertEqual(evidence["rhs_novel_number"], "rhs_novel_number:3/4")
        self.assertEqual(evidence["equation_claim"], "equation_claim:x=3/4")
        self.assertEqual(evidence["operation_intent"], "operation_intent:solve_equation")
        self.assertIn("rhs_novel_number", evidence["evidence_channels"])

        latex = extract_step_evidence(r"Thus \left(\frac{x}{2}\right) = 7.", "")
        self.assertEqual(latex["equation_claim"], r"equation_claim:(\frac{x}{2})=7")

        rejected = extract_step_evidence("Set the expression equal to 0.", "")
        self.assertIsNone(rejected["equation_claim"])

        generic = extract_step_evidence("Now we continue.\n\n", "Problem mentions 100.")
        self.assertEqual(generic["evidence_channels"], [])

    def test_content_anchors_drop_generic_templates(self):
        self.assertEqual(extract_content_anchors("Now we need to solve this problem."), [])
        anchors = extract_content_anchors("Split into even and odd cases for the remaining values.")
        self.assertIn("split", anchors)
        self.assertIn("even", anchors)
        self.assertIn("odd", anchors)
        self.assertIn("cases", anchors)

    def test_text_anchor_consensus_accepts_new_shared_content(self):
        probe_row = {
            "assistant_prefix_text": "Let n be an integer.\n\n",
            "rollouts": [
                {"rollout_idx": 0, "text": "Split into even and odd cases.", "mean_logprob": -0.2},
                {"rollout_idx": 1, "text": "We split into even and odd cases next.", "mean_logprob": -0.1},
                {"rollout_idx": 2, "text": "Now split the problem into odd and even cases.", "mean_logprob": -0.3},
                {"rollout_idx": 3, "text": "Use a geometric diagram.", "mean_logprob": -0.4},
            ],
        }
        selected = _selected_prefix_consensus_rollout(probe_row, min_agreement_count=3)
        self.assertEqual(selected["prefix_consensus_channel"], "content_anchor")
        self.assertEqual(selected["prefix_consensus_support_count"], 3)
        self.assertEqual(selected["prefix_anchor_idx"], 1)
        self.assertGreater(selected["content_anchor_residual_agreement"], 0.0)

    def test_text_anchor_consensus_rejects_prefix_repetition(self):
        probe_row = {
            "assistant_prefix_text": "Split into even and odd cases.\n\n",
            "rollouts": [
                {"rollout_idx": 0, "text": "Split into even and odd cases.", "mean_logprob": -0.1},
                {"rollout_idx": 1, "text": "We split into even and odd cases.", "mean_logprob": -0.2},
                {"rollout_idx": 2, "text": "Split into odd and even cases.", "mean_logprob": -0.3},
                {"rollout_idx": 3, "text": "Split into even and odd cases.", "mean_logprob": -0.4},
            ],
        }
        selected = _selected_prefix_consensus_rollout(probe_row, min_agreement_count=3)
        self.assertIsNone(selected["selected_rollout"])
        self.assertIsNone(selected["prefix_consensus_channel"])
        self.assertLessEqual(selected["content_anchor_residual_agreement"], 0.0)

    def test_explicit_hard_conflict_skips_text_fallback(self):
        probe_row = {
            "assistant_prefix_text": "Let x be unknown.\n\n",
            "rollouts": [
                {"rollout_idx": 0, "text": r"Final \boxed{1}. Then split into cases.", "mean_logprob": -0.1},
                {"rollout_idx": 1, "text": r"Final \boxed{2}. Then split into cases.", "mean_logprob": -0.2},
                {"rollout_idx": 2, "text": "Then split into cases.", "mean_logprob": -0.3},
                {"rollout_idx": 3, "text": "Then split into cases.", "mean_logprob": -0.4},
            ],
        }
        for rollout in probe_row["rollouts"]:
            rollout["step_evidence"] = extract_step_evidence(rollout["text"], "")
            for key, value in rollout["step_evidence"].items():
                if key != "evidence_channels":
                    rollout[f"evidence_{key}"] = value
        selected = _selected_prefix_consensus_rollout(probe_row, min_agreement_count=3)
        self.assertTrue(selected["prefix_hard_conflict"])
        self.assertIsNone(selected["selected_rollout"])
        self.assertIsNone(selected["content_anchor_inter_agreement"])

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
        self.assertEqual(len(probes), 2)
        self.assertEqual(probes[0]["boundary_idx"], -1)
        self.assertTrue(probes[0]["is_initial_probe"])
        self.assertEqual(probes[0]["prefix_char_len"], 0)
        self.assertEqual(probes[1]["boundary_idx"], 0)
        self.assertFalse(probes[1]["is_initial_probe"])
        self.assertIn("rollouts", probes[0])
        self.assertEqual(probe_cost["probe_generate_calls"], 2)

        problem = EvalProblem(problem_id=1, question_id="q1", problem_text="Problem: x?", raw={}, gold_answer="1")
        summary = build_problem_summary(problem, result, probes, probe_cost, "math500")
        self.assertEqual(summary["num_boundaries"], 1)
        self.assertEqual(summary["num_initial_probes"], 1)
        self.assertNotIn("num_thinking_boundaries", summary)
        self.assertNotIn("num_final_answer_boundaries", summary)
        enriched = enrich_probe_rows(problem, probes, result, "math500")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "sampling"
            write_sampling_outputs(out_dir, enriched, [summary])
            self.assertTrue((out_dir / "probes.jsonl").exists())
            self.assertTrue((out_dir / "problem_summary.csv").exists())
            self.assertEqual(len((out_dir / "probes.jsonl").read_text(encoding="utf-8").splitlines()), 2)

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
            "prefix_consensus_support_count": 0,
            "prefix_consensus_vote_fraction": None,
        }
        probe = {
            "problem_id": 1,
            "question_id": "q1",
            "boundary_idx": 0,
            "assistant_prefix_text": "Let x be unknown.\n\n",
            "prefix_char_len": 20,
            "prefix_token_len": 20,
            "prefix_consensus_channel": "rhs_novel_number",
            "prefix_consensus_value": "rhs_novel_number:1",
            "prefix_consensus_support_count": 3,
            "prefix_consensus_vote_fraction": 0.75,
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
        self.assertEqual(csv_rows[0]["prefix_consensus_support_count"], 3)
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

    def test_consensus_routing_fake_problem(self):
        slm = SamplingProbeEngine(
            outputs=[
                ("Let x = 0.\n\n", "stop"),
            ],
            probe_outputs=[
                "x = 1",
                "x = 2",
                "x = 3",
                "x = 4",
            ],
        )
        llm = SequencedEngine([(r"\boxed{9}", "eos")])
        result, boundaries, probe_cost = run_disagreement_routing(
            "Problem: x?",
            slm,
            llm,
            BPAConfig(max_total_tokens=200),
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "9")
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0]["boundary_idx"], 0)
        self.assertTrue(boundaries[0]["routed_to_llm"])
        self.assertEqual(probe_cost["probe_generate_calls"], 1)

    def test_routing_problem_outputs_are_resumable_and_compact(self):
        slm = SamplingProbeEngine(
            outputs=[
                ("Let x = 0.\n\n", "stop"),
            ],
            probe_outputs=[
                "x = 1",
                "x = 2",
                "x = 3",
                "x = 4",
            ],
        )
        llm = SequencedEngine([(r"\boxed{9}", "eos")])
        problem = EvalProblem(
            problem_id=11,
            question_id="q11",
            problem_text="Problem: x?",
            raw={"id": 11},
            gold_answer="9",
        )
        config = BPAConfig(max_total_tokens=200)
        result, boundaries, probe_cost = run_disagreement_routing(
            problem.problem_text,
            slm,
            llm,
            config,
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        summary = build_routing_problem_summary(
            dataset="aime25",
            problem=problem,
            result=result,
            boundary_rows=boundaries,
            probe_cost=probe_cost,
            min_agreement_count=3,
            config=config,
            problem_wall_time=1.25,
        )
        compact = compact_boundary_row({"assistant_prefix_text": "old", "rollouts": [{"token_ids": [1], "text": "x=1"}]})
        self.assertNotIn("assistant_prefix_text", compact)
        self.assertNotIn("token_ids", compact["rollouts"][0])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "routing"
            write_routing_problem_outputs(
                out_dir,
                dataset="aime25",
                problem=problem,
                result=result,
                boundary_rows=boundaries,
                probe_cost=probe_cost,
                summary_row=summary,
            )
            self.assertTrue(has_complete_routing_problem_outputs(out_dir, problem.problem_id))
            boundary_text = (out_dir / "11" / "11.boundaries.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("assistant_prefix_text", boundary_text)
            self.assertNotIn("token_ids", boundary_text)
            resumed = existing_routing_problem_summary(out_dir, problem, "aime25")
            self.assertEqual(resumed["final_answer"], "9")
            self.assertTrue(resumed["correct"])
            self.assertEqual(float(resumed["problem_wall_time"]), 1.25)

    def test_prefix_consensus_reuses_anchor_probe_prefix(self):
        slm = WeightedSamplingProbeEngine(
            outputs=[
                ("First step.\n\n", "stop"),
                (r" and final \boxed{1}", "eos"),
            ],
            probe_outputs=[
                ("x = 1", -0.2, "length"),
                ("x = 1", -0.05, "length"),
                ("x = 1", -0.3, "length"),
                ("Try y = 2", -0.4, "length"),
            ],
        )
        llm = SequencedEngine(fail_on_generate=True)
        result, boundaries, _ = run_disagreement_routing(
            "Problem: x?",
            slm,
            llm,
            BPAConfig(max_total_tokens=200),
            min_agreement_count=3,
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "1")
        self.assertEqual(len(boundaries), 1)
        self.assertTrue(boundaries[0]["reused_probe_rollout"])
        self.assertTrue(boundaries[0]["continued_probe_rollout"])
        self.assertEqual(boundaries[0]["prefix_anchor_idx"], 1)
        self.assertEqual(boundaries[0]["prefix_consensus_channel"], "rhs_novel_number")
        self.assertEqual(boundaries[0]["prefix_consensus_value"], "rhs_novel_number:1")
        self.assertEqual(boundaries[0]["prefix_consensus_support_count"], 3)
        self.assertEqual(boundaries[0]["selected_rollout_idx"], 1)
        self.assertEqual(boundaries[0]["probe_prefix_text"], "x = 1")
        self.assertIn(r"x = 1 and final \boxed{1}", result.state.assistant_prefix_text)

    def test_prefix_consensus_routes_unstable_prefixes_to_llm(self):
        slm = WeightedSamplingProbeEngine(
            outputs=[
                ("First step.\n\n", "stop"),
            ],
            probe_outputs=[
                ("Let x = 1", -0.05, "length"),
                ("Try y = 2", -0.1, "length"),
                ("Compute z", -0.2, "length"),
                ("Maybe use geometry", -0.3, "length"),
            ],
        )
        llm = SequencedEngine([(r"\boxed{9}", "eos")])
        result, boundaries, _ = run_disagreement_routing(
            "Problem: x?",
            slm,
            llm,
            BPAConfig(max_total_tokens=200),
            min_agreement_count=3,
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "9")
        self.assertEqual(len(boundaries), 1)
        self.assertTrue(boundaries[0]["routed_to_llm"])
        self.assertFalse(boundaries[0]["reused_probe_rollout"])
        self.assertIsNone(boundaries[0]["prefix_anchor_idx"])
        self.assertEqual(boundaries[0]["prefix_consensus_support_count"], 1)
        self.assertEqual(llm.generate_calls, 1)

    def test_operation_intent_is_diagnostic_not_routing_evidence(self):
        slm = WeightedSamplingProbeEngine(
            outputs=[
                ("First step.\n\n", "stop"),
            ],
            probe_outputs=[
                ("So, solve the equation.\n\n", -0.1, "stop"),
                ("Now solve this equation.\n\n", -0.2, "stop"),
                ("We solve the equation.\n\n", -0.3, "stop"),
                ("Let us solve the equation.\n\n", -0.4, "stop"),
            ],
        )
        llm = SequencedEngine([(r"\boxed{9}", "eos")])
        result, boundaries, _ = run_disagreement_routing(
            "Problem: x?",
            slm,
            llm,
            BPAConfig(max_total_tokens=200),
            min_agreement_count=3,
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "9")
        self.assertTrue(boundaries[0]["routed_to_llm"])
        self.assertFalse(boundaries[0]["reused_probe_rollout"])
        self.assertIsNone(boundaries[0]["prefix_consensus_channel"])
        self.assertEqual(boundaries[0]["rollouts"][0]["evidence_operation_intent"], "operation_intent:solve_equation")

    def test_prefix_consensus_selected_stop_probe_uses_eos_lookahead(self):
        slm = WeightedSamplingProbeEngine(
            outputs=[
                ("First step.\n\n", "length"),
                ("", "eos"),
            ],
            probe_outputs=[
                (r"Final \boxed{1}\n\n", -0.05, "stop"),
                (r"Final \boxed{1}\n\n", -0.1, "stop"),
                (r"Final \boxed{1}\n\n", -0.2, "stop"),
                ("Try something else", -0.3, "stop"),
            ],
        )
        llm = SequencedEngine(fail_on_generate=True)
        result, boundaries, _ = run_disagreement_routing(
            "Problem: x?",
            slm,
            llm,
            BPAConfig(max_total_tokens=200, post_stop_lookahead_tokens=4),
            min_agreement_count=3,
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "1")
        self.assertEqual(result.state.stop_reason, "eos")
        self.assertEqual(len(boundaries), 1)
        self.assertTrue(boundaries[0]["reused_probe_rollout"])
        self.assertEqual(boundaries[0]["main_step_finish_reason"], "eos")

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
        # L0 probe uses one token, then the close-think step uses its text length.
        self.assertEqual(slm.sampling_history[-1]["max_tokens"], 11)

    def test_disagreement_routing_final_answer_phase_runs_once_on_stop(self):
        slm = SamplingProbeEngine(
            outputs=[
                ("</think>\n\n", "stop"),
                ("The answer is 42.\n\n", "stop"),
            ],
            probe_outputs=[],
        )
        llm = SequencedEngine(fail_on_generate=True)
        result, boundaries, _ = run_disagreement_routing(
            "Problem: x?",
            slm,
            llm,
            BPAConfig(max_total_tokens=200),
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "42")
        self.assertEqual(result.state.stop_reason, "final_answer_stop")
        self.assertEqual(slm.generate_calls, 2)
        self.assertEqual(result.state.step_count, 2)
        self.assertEqual(boundaries, [])

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

    def test_unified_recovers_from_duplicate_step(self):
        slm = SequencedEngine(
            [
                ("Same final step.\n\n", "stop"),
                ("Same final step.", "stop"),
                ("The answer is 42.", "eos"),
            ]
        )
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("p", slm, llm, BPAConfig(max_total_tokens=300))
        self.assertEqual(result.answer, "42")
        self.assertEqual(result.state.stop_reason, "eos")
        self.assertTrue(any(event.event == "step_repetition_stop" for event in result.state.trace))
        self.assertTrue(any(event.event == "forced_close_think_for_final_answer" for event in result.state.trace))

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

    def test_disagreement_routing_forces_final_answer_after_thinking_duplicate(self):
        slm = SamplingProbeEngine(
            outputs=[
                ("First step.\n\n", "stop"),
                ("The answer is 42.\n\n", "stop"),
            ],
            probe_outputs=[
                "alpha",
                "beta",
                "gamma",
                "delta",
                "alpha",
                "beta",
                "gamma",
                "delta",
            ],
        )
        llm = SequencedEngine(
            [
                ("Repeated thinking step.\n\n", "stop"),
                ("Repeated thinking step.\n\n", "stop"),
            ]
        )
        result, boundaries, _ = run_disagreement_routing(
            "Problem: x?",
            slm,
            llm,
            BPAConfig(max_total_tokens=300),
            probe_k=4,
            probe_temperature=0.7,
            probe_max_tokens=32,
        )
        self.assertEqual(result.answer, "42")
        self.assertEqual(result.state.stop_reason, "final_answer_stop")
        self.assertEqual(len(boundaries), 2)
        self.assertTrue(any(event.event == "forced_close_think_for_final_answer" for event in result.state.trace))

    def test_unified_context_budget_stops_before_generation(self):
        slm = SequencedEngine(fail_on_generate=True)
        llm = SequencedEngine(fail_on_generate=True)
        result = bpa_solve("long prompt", slm, llm, BPAConfig(max_model_len=8, max_total_tokens=100))
        self.assertEqual(result.state.stop_reason, "context_budget")
        self.assertEqual(slm.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
