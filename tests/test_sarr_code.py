from __future__ import annotations

import unittest

from sarr_code.algorithm import choose_best_prefix_anchor, run_sarr_code
from sarr_code.calibration import IdentityNormalizer, PercentileNormalizer, code_style_degeneration_event, smooth_confidence
from sarr_code.config import (
    ConfidenceConfig,
    GenerationConfig,
    ModelRuntimeConfig,
    RollbackConfig,
    RuntimeConfig,
    SARRConfig,
    StableConfig,
    StartupConfig,
)
from sarr_code.records import StepOutput, StepRecord
from scripts.run_sarr_code import _extra_sarr_metrics
from scripts.run_sarr_sweep import SWEEP_VARIANTS, apply_variant


class FakeEngine:
    def __init__(self, outputs=None, confidences=None):
        self.outputs = list(outputs or [])
        self.confidences = list(confidences or [])
        self.generate_calls = 0
        self.confidence_calls = 0

    def _pop_output(self):
        if not self.outputs:
            raise AssertionError("no queued output")
        text = self.outputs.pop(0)
        token_ids = list(range(len(text)))
        return StepOutput(text=text, token_ids=token_ids, finish_reason="stop", prompt_tokens=3, wall_time=0.01)

    def generate_step(self, *args, **kwargs):
        self.generate_calls += 1
        return self._pop_output()

    def generate_text(self, *args, **kwargs):
        self.generate_calls += 1
        return self._pop_output()

    def continuation_confidence(self, *args, **kwargs):
        self.confidence_calls += 1
        if not self.confidences:
            raise AssertionError("no queued confidence")
        value = float(self.confidences.pop(0))
        return value, {"norm_entropy": 1.0 - value, "top_ids": [1], "top_probs": [1.0]}

    def clear_runtime_cache(self):
        return True


def make_cfg() -> SARRConfig:
    return SARRConfig(
        slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
        llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
        generation=GenerationConfig(
            max_new_tokens_per_step=32,
            think_token_budget=512,
            answer_token_budget=64,
            force_close_think_on_budget=True,
        ),
        confidence=ConfidenceConfig(
            topk_entropy=20,
            calibration_path=None,
            allow_identity_normalizer=True,
            smooth_window=2,
            delta=0.55,
        ),
        startup=StartupConfig(B_min=2, B_max=3, tau_start=1),
        stable=StableConfig(theta_s=0.70, tau_D=1),
        rollback=RollbackConfig(M_max=5),
        runtime=RuntimeConfig(max_model_len=4096),
    )


class SARRCodeTests(unittest.TestCase):
    def test_percentile_normalizer_and_degradation(self):
        normalizer = PercentileNormalizer([0.1, 0.2, 0.4, 0.8])
        self.assertEqual(normalizer.transform(0.2), 0.5)
        self.assertEqual(smooth_confidence([0.2], W=2), None)
        self.assertAlmostEqual(smooth_confidence([0.2, 0.6], W=2), 0.4)
        self.assertEqual(code_style_degeneration_event(0.7, 0.5, delta=0.55), 1)
        self.assertEqual(code_style_degeneration_event(0.5, 0.7, delta=0.55), 0)

    def test_choose_best_prefix_anchor_allows_zero(self):
        records = [
            StepRecord("p", 1, "slm", "a", [], c_smooth=None),
            StepRecord("p", 2, "slm", "b", [], c_smooth=0.4),
            StepRecord("p", 3, "slm", "c", [], c_smooth=0.3),
        ]
        self.assertEqual(choose_best_prefix_anchor(records), 2)
        self.assertEqual(choose_best_prefix_anchor([records[0]]), 0)

    def test_run_sarr_code_rolls_back_and_recovers(self):
        cfg = make_cfg()
        slm = FakeEngine(
            outputs=[
                "first\n\n",
                "second\n\n",
                "bad\n\n",
                "</think>\n\n",
            ],
            confidences=[0.5, 0.4, 0.3, 0.8, 0.8],
        )
        llm = FakeEngine(outputs=["recovered\n\n", "Final answer: 42."])

        result, steps, rollbacks, transitions = run_sarr_code(
            problem_id="p1",
            problem_text="Problem: 6*7?",
            slm=slm,
            llm=llm,
            normalizer=IdentityNormalizer(),
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(len(rollbacks), 1)
        self.assertEqual(rollbacks[0]["type"], "STARTUP_ROLLBACK")
        self.assertEqual(rollbacks[0]["anchor_step"], 2)
        self.assertEqual(rollbacks[0]["rollback_span"], 1)
        self.assertEqual(rollbacks[0]["stop_reason"], "SLM_READY")
        removed = [row for row in steps if row["removed_by_rollback"]]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["text"], "bad\n\n")
        self.assertTrue(any(row["transition_type"] == "slm->llm" for row in steps))
        self.assertTrue(transitions)
        self.assertIn("recovered", result.state.assistant_prefix_text)
        self.assertNotIn("bad\n\nrecovered", result.state.assistant_prefix_text)

    def test_post_stable_suspect_recovers_without_rollback(self):
        cfg = make_cfg()
        slm = FakeEngine(
            outputs=[
                "stable-a\n\n",
                "stable-b\n\n",
                "correct-but-low-confidence\n\n",
                "confirmation-step\n\n",
                "</think>\n\n",
            ],
            confidences=[0.9, 0.9, 0.1, 0.9, 0.9],
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        result, steps, rollbacks, _ = run_sarr_code(
            problem_id="suspect-recover",
            problem_text="Problem: test",
            slm=slm,
            llm=llm,
            normalizer=IdentityNormalizer(),
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(len(rollbacks), 0)
        self.assertFalse(any(row["removed_by_rollback"] for row in steps))
        self.assertTrue(any(row["action"] == "ENTER_SUSPECT" for row in steps))
        self.assertTrue(any(row["action"] == "SUSPECT_RECOVERED" for row in steps))
        self.assertIn("correct-but-low-confidence", result.state.assistant_prefix_text)

    def test_post_stable_suspect_confirms_then_rolls_back(self):
        cfg = SARRConfig(
            slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
            llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
            generation=GenerationConfig(
                max_new_tokens_per_step=32,
                think_token_budget=512,
                answer_token_budget=64,
            ),
            confidence=ConfidenceConfig(
                topk_entropy=20,
                calibration_path=None,
                allow_identity_normalizer=True,
                smooth_window=2,
                delta=0.55,
            ),
            startup=StartupConfig(B_min=2, B_max=3, tau_start=1),
            stable=StableConfig(theta_s=0.70, tau_D=1),
            rollback=RollbackConfig(
                M_max=5,
                suspect_confirm_steps=1,
                suspect_max_steps=2,
                tau_confirm=1,
            ),
            runtime=RuntimeConfig(max_model_len=4096),
        )
        slm = FakeEngine(
            outputs=[
                "stable-a\n\n",
                "stable-b\n\n",
                "bad-a\n\n",
                "bad-b\n\n",
                "</think>\n\n",
            ],
            confidences=[
                0.9,
                0.9,
                0.1,
                0.0,
                0.8,
                0.8,
            ],
        )
        llm = FakeEngine(outputs=["repair\n\n", "Final answer: 42."])

        result, steps, rollbacks, _ = run_sarr_code(
            problem_id="suspect-confirm",
            problem_text="Problem: test",
            slm=slm,
            llm=llm,
            normalizer=IdentityNormalizer(),
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(len(rollbacks), 1)
        self.assertEqual(rollbacks[0]["type"], "POST_STABLE_ROLLBACK")
        self.assertEqual(rollbacks[0]["reason"], "POST_STABLE_CONFIRMED_DEGENERATION")
        self.assertEqual(rollbacks[0]["suspect_steps"], 1)
        removed_text = "".join(row["text"] for row in rollbacks[0]["removed_steps"])
        self.assertIn("bad-a", removed_text)
        self.assertIn("bad-b", removed_text)
        self.assertNotIn("bad-a\n\nbad-b", result.state.assistant_prefix_text)

    def test_summary_metrics_include_required_sarr_fields(self):
        rows = [
            {
                "has_rollback": True,
                "has_startup_rollback": True,
                "has_post_stable_rollback": False,
                "rollback_count": 2,
                "startup_rollback_count": 2,
                "post_stable_rollback_count": 0,
                "anchor_zero_count": 1,
                "rollback_span_total": 5,
                "recovery_steps_total": 6,
                "recovery_ready_count": 1,
                "recovery_exhausted_count": 1,
                "forced_close_think": True,
                "force_slm_after_recovery_count": 2,
                "force_slm_after_recovery_fail_count": 1,
                "llm_total_tokens": 30,
                "total_model_tokens": 100,
            },
            {
                "has_rollback": False,
                "has_startup_rollback": False,
                "has_post_stable_rollback": False,
                "rollback_count": 0,
                "startup_rollback_count": 0,
                "post_stable_rollback_count": 0,
                "anchor_zero_count": 0,
                "rollback_span_total": 0,
                "recovery_steps_total": 0,
                "recovery_ready_count": 0,
                "recovery_exhausted_count": 0,
                "forced_close_think": False,
                "force_slm_after_recovery_count": 0,
                "force_slm_after_recovery_fail_count": 0,
                "llm_total_tokens": 10,
                "total_model_tokens": 100,
            },
        ]
        metrics = _extra_sarr_metrics(rows)
        for key in [
            "rollback_rate",
            "startup_rollback_rate",
            "post_stable_rollback_rate",
            "avg_rollback_span",
            "avg_recovery_steps",
            "recovery_ready_rate",
            "recovery_exhausted_rate",
            "forced_close_think_rate",
            "force_slm_after_recovery_fail_rate",
            "llm_token_ratio",
        ]:
            self.assertIn(key, metrics)
        self.assertEqual(metrics["rollback_rate"], 0.5)
        self.assertEqual(metrics["avg_rollback_span"], 2.5)
        self.assertEqual(metrics["llm_token_ratio"], 0.2)

    def test_sweep_variants_apply_expected_parameters(self):
        self.assertEqual([variant.name for variant in SWEEP_VARIANTS], [
            "D1_balanced_055",
            "D2_balanced_050",
            "D3_balanced_060",
            "D4_aggressive_startup",
            "D5_conservative",
            "D6_event_aggressive",
            "D7_high_theta_post_conservative",
            "D8_low_theta_post_repair",
        ])
        cfg = apply_variant(
            {
                "confidence": {"delta": 0.1},
                "stable": {"theta_s": 0.1, "tau_D": 9},
                "startup": {"tau_start": 9, "B_max": 9},
            },
            SWEEP_VARIANTS[3],
        )
        self.assertEqual(cfg["confidence"]["delta"], 0.55)
        self.assertEqual(cfg["stable"]["theta_s"], 0.75)
        self.assertEqual(cfg["stable"]["tau_D"], 1)
        self.assertEqual(cfg["startup"]["tau_start"], 1)
        self.assertEqual(cfg["startup"]["B_max"], 4)


if __name__ == "__main__":
    unittest.main()
