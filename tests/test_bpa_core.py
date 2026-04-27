from __future__ import annotations

import math
import unittest

from bpa.arbitration import locate_branch_token_span, score_branch
from bpa.cascade.l0 import entropy_and_margin
from bpa.cascade.l1 import build_branch
from bpa.cascade.l2 import char_ngram_jaccard, l2_compute
from bpa.config import BPAConfig
from bpa.eval.benchmark_eval import benchmark_eval_match, normalize_math_expr
from bpa.phase_machine import check_and_transition_phase, detect_close_think
from bpa.safety import ensure_step_terminator, update_repetition
from bpa.state import BranchCandidate, GenerationState, RepetitionState


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTokenizer:
    is_fast = True

    def __init__(self, table=None):
        self.table = table or {}

    def apply_chat_template(self, messages, tokenize=False, continue_final_message=True, add_generation_prompt=False):
        assert continue_final_message is True
        assert add_generation_prompt is False
        return f"USER:{messages[0]['content']}\nASSISTANT:{messages[1]['content']}"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.table.get(i, chr(i)) for i in ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {
            "input_ids": [ord(ch) for ch in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class FakeEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def ensure_tokenizer(self):
        return self.tokenizer

    def sampling_params(self, **kwargs):
        return kwargs

    def tokens_prompt(self, prompt_token_ids):
        return prompt_token_ids

    def generate(self, prompts, sampling_params):
        prompt_ids = prompts
        plp = [{tid: Obj(logprob=-0.1)} for tid in prompt_ids]
        return [Obj(prompt_logprobs=plp, outputs=[Obj(text="x", token_ids=[120], finish_reason="length")])]


class CoreTests(unittest.TestCase):
    def test_detect_close_think_cross_boundary(self):
        found, rel = detect_close_think("abc</thi", "nk>tail")
        self.assertTrue(found)
        self.assertEqual(rel, -5)

        state = GenerationState(problem_text="p", assistant_prefix_text="abc</think>tail")
        check_and_transition_phase(state, "nk>tail")
        self.assertEqual(state.assistant_prefix_text, "abc</think>")
        self.assertTrue(state.has_seen_close_think)

    def test_step_terminator_and_repetition(self):
        self.assertEqual(ensure_step_terminator("abc", "stop"), "abc\n\n")
        self.assertEqual(ensure_step_terminator("abc", "eos"), "abc")
        rep = RepetitionState()
        self.assertIsNone(update_repetition(rep, "This is a long enough step."))
        self.assertEqual(update_repetition(rep, "This is a long enough step."), "duplicate_step")

    def test_l0_entropy_margin(self):
        h, margin = entropy_and_margin({1: 0.0, 2: math.log(0.5)})
        self.assertGreater(h, 0.0)
        self.assertGreater(margin, 0.0)

    def test_l1_branch_truncation(self):
        tok = FakeTokenizer({1: "A", 2: "B", 3: "\n", 4: "\n", 5: "C"})
        out = Obj(outputs=[Obj(token_ids=[2, 3, 4, 5], logprobs=[{2: Obj(logprob=-0.2)}, {3: Obj(logprob=-0.3)}, {4: Obj(logprob=-0.4)}, {5: Obj(logprob=-0.5)}])])
        branch = build_branch(1, -0.1, out, tok)
        self.assertEqual(branch.raw_rollout_text, "AB\n\nC")
        self.assertEqual(branch.step_branch_text, "AB")
        self.assertTrue(branch.step_branch_was_truncated)
        self.assertAlmostEqual(branch.sum_logprob_step, -0.3)

    def test_l2_statistics(self):
        b1 = BranchCandidate(1, "A", "alpha path", [1, 2], "alpha", False, [-0.2], -0.1, -0.3, -0.3)
        b2 = BranchCandidate(3, "B", "beta path", [3, 4], "beta", False, [-0.2], -0.1, -0.3, -0.3)
        result = l2_compute(b1, b2, BPAConfig())
        self.assertTrue(result.triggered_arbitration)
        self.assertEqual(result.trigger_reason, "delta_raw_low")
        self.assertLess(char_ngram_jaccard("abc", "xyz"), 1.0)

    def test_arbitration_span_and_score(self):
        tokenizer = FakeTokenizer()
        locate = locate_branch_token_span("p", "abc", "XYZ", tokenizer)
        self.assertFalse(locate.is_invalid)
        self.assertEqual(locate.branch_end_token - locate.branch_start_token, 3)

        state = GenerationState(problem_text="p", assistant_prefix_text="abc")
        score = score_branch(state, FakeEngine(tokenizer), "XYZ", BPAConfig())
        self.assertFalse(score.is_invalid)
        self.assertEqual(score.branch_token_count, 3)
        self.assertEqual(score.missing_count, 0)

    def test_answer_eval(self):
        self.assertEqual(normalize_math_expr(r"\frac{4}{2}"), "2")
        self.assertTrue(benchmark_eval_match(r"final \boxed{42}", "42", "aime25"))
        self.assertTrue(benchmark_eval_match(r"final \boxed{B}", "B", "gpqa"))


if __name__ == "__main__":
    unittest.main()
