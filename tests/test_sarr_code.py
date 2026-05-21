from __future__ import annotations

import unittest

from sarr_code.algorithm import choose_best_prefix_anchor, run_sarr_code
from sarr_code.calibration import PercentileNormalizer, code_style_degeneration_event, smooth_confidence
from sarr_code.config import (
    BudgetConfig,
    ConfidenceConfig,
    ConfidenceProcessConfig,
    GenerationConfig,
    HCSConfig,
    HCSRecoveryConfig,
    LLMLeaseConfig,
    LowConfidenceConfig,
    LowReadinessConfig,
    ModelRuntimeConfig,
    ReadinessConfig,
    RollbackConfig,
    RoutingConfig,
    RuntimeConfig,
    SARRConfig,
    StableConfig,
    StagnationConfig,
    StartupConfig,
)
from sarr_code.records import StepOutput, StepRecord
from scripts.run_sarr_code import _confidence_process_metrics, _extra_sarr_metrics
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
            smooth_window=2,
            delta=0.55,
        ),
        readiness=ReadinessConfig(smooth_window=1),
        startup=StartupConfig(B_min=2, B_max=3, tau_start=1),
        stable=StableConfig(theta_s=0.70, tau_D=1),
        rollback=RollbackConfig(M_max=5),
        low_confidence=LowConfidenceConfig(useful_exploration_grace_blocks=0, collapse_patience_blocks=1),
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
            StepRecord("p", 1, "slm", "a", [], readiness_raw_smooth=None),
            StepRecord("p", 2, "slm", "b", [], readiness_raw_smooth=0.4),
            StepRecord("p", 3, "slm", "c", [], readiness_raw_smooth=0.3),
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
            confidences=[0.5, 0.6, 0.3, 0.8, 0.8],
        )
        llm = FakeEngine(outputs=["recovered\n\n", "Final answer: 42."])

        result, steps, rollbacks, transitions = run_sarr_code(
            problem_id="p1",
            problem_text="Problem: 6*7?",
            slm=slm,
            llm=llm,
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

    def test_close_think_step_skips_confidence_forward(self):
        cfg = make_cfg()
        slm = FakeEngine(outputs=["Done.\n</think>\n\n"], confidences=[])
        llm = FakeEngine(outputs=["Final answer: 42."])

        result, steps, rollbacks, _ = run_sarr_code(
            problem_id="close",
            problem_text="Problem: 6*7?",
            slm=slm,
            llm=llm,
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(len(rollbacks), 0)
        self.assertEqual(slm.confidence_calls, 0)
        self.assertEqual(steps[0]["action"], "FINISHED")
        self.assertTrue(steps[0]["extra"]["confidence_skipped"])
        self.assertEqual(steps[0]["extra"]["confidence_skipped_reason"], "finished")

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
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(len(rollbacks), 0)
        self.assertFalse(any(row["removed_by_rollback"] for row in steps))
        self.assertTrue(any(row["action"] == "USEFUL_EXPLORATION" for row in steps))
        self.assertTrue(any(row["action"] == "REFRESH_STABLE_ANCHOR" for row in steps))
        self.assertIn("correct-but-low-confidence", result.state.assistant_prefix_text)

    def test_persistent_low_readiness_triggers_llm_lease_without_rollback(self):
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
                smooth_window=2,
                delta=0.55,
            ),
            startup=StartupConfig(B_min=2, B_max=3, tau_start=1),
            stable=StableConfig(theta_s=0.70, tau_D=1),
            readiness=ReadinessConfig(smooth_window=1),
            low_readiness=LowReadinessConfig(useful_exploration_grace_steps=1),
            llm_lease=LLMLeaseConfig(persistent_uncertainty_steps=2, max_tokens_per_step=16),
            rollback=RollbackConfig(M_max=5),
            low_confidence=LowConfidenceConfig(useful_exploration_grace_blocks=0, collapse_patience_blocks=1),
            runtime=RuntimeConfig(max_model_len=4096),
        )
        slm = FakeEngine(
            outputs=[
                "stable-a\n\n",
                "stable-b\n\n",
                "bad-a\n\n",
                "bad-b\n\n",
                "after-lease-observe\n\n",
                "</think>\n\n",
            ],
            confidences=[
                0.9,
                0.9,
                0.1,
                0.0,
                0.8,
            ],
        )
        llm = FakeEngine(outputs=["lease-a\n\n", "lease-b\n\n", "Final answer: 42."])

        result, steps, rollbacks, _ = run_sarr_code(
            problem_id="suspect-confirm",
            problem_text="Problem: test",
            slm=slm,
            llm=llm,
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(len(rollbacks), 1)
        self.assertEqual(rollbacks[0]["event"], "llm_lease")
        self.assertEqual(rollbacks[0]["type"], "LLM_LEASE")
        self.assertEqual(rollbacks[0]["reason"], "PERSISTENT_UNCERTAINTY")
        self.assertFalse(rollbacks[0]["rollback_before_lease"])
        self.assertEqual(rollbacks[0]["recovery_actual_steps"], 2)
        self.assertFalse(any(row["removed_by_rollback"] for row in steps))
        self.assertIn("bad-a\n\nbad-b\n\nlease-a", result.state.assistant_prefix_text)

    def test_hcs_confirmed_rolls_back_to_clean_anchor_with_normal_recovery(self):
        cfg = SARRConfig(
            slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
            llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
            generation=GenerationConfig(
                max_new_tokens_per_step=64,
                think_token_budget=1024,
                answer_token_budget=64,
            ),
            confidence=ConfidenceConfig(
                topk_entropy=20,
                calibration_path=None,
                smooth_window=1,
                delta=0.55,
            ),
            startup=StartupConfig(B_min=1, B_max=5, tau_start=1),
            stable=StableConfig(theta_s=0.70, tau_D=1),
            stagnation=StagnationConfig(
                enabled=True,
                block_min_tokens=8,
                block_max_steps=1,
                repeat_window=10,
                high_threshold=0.85,
            ),
            hcs=HCSConfig(enabled=True, suspect_patience=3, max_hcs_rollbacks_per_problem=2),
            hcs_recovery=HCSRecoveryConfig(max_llm_steps=2, max_tokens_per_step=16),
            llm_lease=LLMLeaseConfig(confirmed_stagnation_steps=3, max_tokens_per_step=16),
            rollback=RollbackConfig(M_max=1, anchor_repeat_policy="suppress"),
            runtime=RuntimeConfig(max_model_len=4096),
        )
        repeated = "we repeat the same confident local calculation and anchor phrase again\n\n"
        slm = FakeEngine(
            outputs=[
                repeated,
                repeated,
                repeated,
                repeated,
                "</think>\n\n",
            ],
            confidences=[0.9, 0.9, 0.9, 0.9, 0.8, 0.8],
        )
        llm = FakeEngine(
            outputs=[
                "ordinary continuation\n\n",
                "second continuation\n\n",
                "third continuation\n\n",
                "Final answer: 42.",
            ]
        )

        result, steps, rollbacks, _ = run_sarr_code(
            problem_id="hcs",
            problem_text="Problem: test",
            slm=slm,
            llm=llm,
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(len(rollbacks), 1)
        rollback = rollbacks[0]
        self.assertEqual(rollback["event"], "llm_lease")
        self.assertEqual(rollback["type"], "STAGNATION_ROLLBACK")
        self.assertEqual(rollback["reason"], "CONFIRMED_STAGNATION")
        self.assertTrue(rollback["rollback_before_lease"])
        self.assertEqual(rollback["readiness_source"], "raw")
        self.assertFalse(rollback["calibration_enabled"])
        self.assertEqual(rollback["anchor_step"], 1)
        self.assertEqual(rollback["clean_anchor_step"], 1)
        self.assertEqual(rollback["rollback_span"], 3)
        self.assertEqual(rollback["hcs_rollback_count"], 1)
        self.assertEqual(rollback["llm_recovery_prompt_type"], "normal_continuation")
        self.assertFalse(rollback["mention_stagnation"])
        self.assertTrue(rollback["return_to_slm"])
        self.assertEqual(rollback["recovery_actual_steps"], 3)

        suspect_rows = [row for row in steps if row["hcs_suspect"]]
        self.assertEqual(len(suspect_rows), 3)
        self.assertFalse(suspect_rows[0]["anchor_refresh_allowed"])
        self.assertEqual(suspect_rows[0]["anchor_refresh_blocked_reason"], "STAGNATION_SUSPECT")
        self.assertEqual(suspect_rows[0]["clean_autonomy_anchor"], 1)
        self.assertTrue(any(row["stagnation_confirmed"] for row in suspect_rows))
        removed_text = "".join(row["text"] for row in rollback["removed_steps"])
        self.assertEqual(removed_text, repeated * 3)
        self.assertIn("ordinary continuation", result.state.assistant_prefix_text)
        self.assertNotIn(repeated * 2, result.state.assistant_prefix_text)

    def test_mid_confidence_stagnation_confirms_and_rolls_back(self):
        cfg = SARRConfig(
            slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
            llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
            generation=GenerationConfig(max_new_tokens_per_step=64, think_token_budget=1024, answer_token_budget=64),
            confidence=ConfidenceConfig(topk_entropy=20, calibration_path=None),
            readiness=ReadinessConfig(smooth_window=1, high_threshold=0.70, low_threshold=0.35),
            startup=StartupConfig(B_min=1, B_max=5, tau_start=1),
            stagnation=StagnationConfig(
                enabled=True,
                block_min_tokens=8,
                block_max_steps=1,
                repeat_window=10,
                high_threshold=0.85,
                patience=3,
            ),
            llm_lease=LLMLeaseConfig(confirmed_stagnation_steps=3, max_tokens_per_step=16),
            rollback=RollbackConfig(M_max=1),
            runtime=RuntimeConfig(max_model_len=4096),
        )
        repeated = "repeat the same mid confidence derivation fragment again\n\n"
        slm = FakeEngine(
            outputs=[
                "clean confident anchor step with enough unique tokens\n\n",
                repeated,
                repeated,
                repeated,
                repeated,
                "</think>\n\n",
            ],
            confidences=[0.9, 0.6, 0.6, 0.6, 0.6, 0.8],
        )
        llm = FakeEngine(outputs=["lease one\n\n", "lease two\n\n", "lease three\n\n", "Final answer: 42."])

        result, steps, rollbacks, _ = run_sarr_code(
            problem_id="mid-stagnation",
            problem_text="Problem: test",
            slm=slm,
            llm=llm,
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(rollbacks[0]["type"], "STAGNATION_ROLLBACK")
        self.assertEqual(rollbacks[0]["anchor_step"], 1)
        mid_rows = [row for row in steps if row["stagnation_suspect"] and row["readiness_mid"]]
        self.assertTrue(mid_rows)
        self.assertTrue(any(row["autonomy_state"] == "MID_CONF_STAGNATION" for row in mid_rows))
        self.assertFalse(any(row["hcs_suspect"] for row in mid_rows))

    def test_routing_budget_exceeded_returns_to_slm_active(self):
        cfg = SARRConfig(
            slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
            llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
            generation=GenerationConfig(max_new_tokens_per_step=32, think_token_budget=512, answer_token_budget=64),
            confidence=ConfidenceConfig(topk_entropy=20, calibration_path=None),
            readiness=ReadinessConfig(smooth_window=1),
            startup=StartupConfig(B_min=2, B_max=3, tau_start=1),
            rollback=RollbackConfig(M_max=5, suspect_confirm_steps=1, suspect_max_steps=2),
            low_readiness=LowReadinessConfig(useful_exploration_grace_steps=0),
            low_confidence=LowConfidenceConfig(useful_exploration_grace_blocks=0, collapse_patience_blocks=1),
            llm_lease=LLMLeaseConfig(low_conf_rollback_steps=2, max_events_per_problem=0),
            budget=BudgetConfig(max_llm_lease_events_per_problem=0),
            runtime=RuntimeConfig(max_model_len=4096),
        )
        slm = FakeEngine(
            outputs=[
                "stable-a\n\n",
                "stable-b\n\n",
                "drop-a\n\n",
                "drop-b\n\n",
                "drop-c\n\n",
                "after-budget\n\n",
                "</think>\n\n",
            ],
            confidences=[0.9, 0.9, 0.1, 0.1, 0.1, 0.9],
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        result, steps, rollbacks, transitions = run_sarr_code(
            problem_id="budget-exceeded",
            problem_text="Problem: test",
            slm=slm,
            llm=llm,
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(rollbacks, [])
        budget_rows = [row for row in steps if row["action"] == "ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM"]
        self.assertEqual(len(budget_rows), 1)
        self.assertEqual(budget_rows[0]["state_after"], "SLM_ACTIVE")
        self.assertTrue(budget_rows[0]["extra"]["routing_budget_exceeded"])
        self.assertFalse(budget_rows[0]["invalid_rollback_recovery_state"])
        next_row = steps[steps.index(budget_rows[0]) + 1]
        self.assertEqual(next_row["state_before"], "SLM_ACTIVE")
        self.assertNotEqual(next_row["anchor_refresh_blocked_reason"], "STATE_ROLLBACK_RECOVERY")
        self.assertTrue(
            any(
                row.get("to") == "SLM_ACTIVE"
                and row.get("reason") == "ROUTING_BUDGET_EXCEEDED_CONTINUE_SLM"
                for row in transitions
            )
        )

    def test_confidence_process_shadow_logging(self):
        cfg = SARRConfig(
            slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
            llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
            generation=GenerationConfig(max_new_tokens_per_step=32, think_token_budget=512, answer_token_budget=64),
            confidence=ConfidenceConfig(
                topk_entropy=20,
                calibration_path=None,
                delta=0.0,
            ),
            confidence_process=ConfidenceProcessConfig(r0=2),
            readiness=ReadinessConfig(smooth_window=3, high_threshold=0.70, low_threshold=0.35),
            startup=StartupConfig(B_min=1, B_max=10, tau_start=1),
            routing=RoutingConfig(enabled=False),
            rollback=RollbackConfig(M_max=5),
            low_confidence=LowConfidenceConfig(useful_exploration_grace_blocks=0, collapse_patience_blocks=1),
            runtime=RuntimeConfig(max_model_len=4096),
        )
        slm = FakeEngine(
            outputs=[
                "h1\n\n",
                "h2\n\n",
                "masked\n\n",
                "m1\n\n",
                "m2\n\n",
                "h3\n\n",
                "h4\n\n",
                "h5\n\n",
                "</think>\n\n",
            ],
            confidences=[0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9],
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        result, steps, _, _ = run_sarr_code(
            problem_id="confidence-process",
            problem_text="Problem: test",
            slm=slm,
            llm=llm,
            cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        for row in steps:
            self.assertIn("confidence_process", row["extra"])

        masked = steps[2]["extra"]["confidence_process"]
        self.assertTrue(masked["raw_low"])
        self.assertFalse(masked["smooth_low"])
        self.assertTrue(masked["masked_uncertainty"])
        self.assertEqual(masked["masked_uncertainty_count"], 1)

        triggered = [row for row in steps if row["extra"]["confidence_process"]["ciod_shadow_trigger"]]
        self.assertTrue(triggered)
        first_trigger = triggered[0]["extra"]["confidence_process"]
        self.assertEqual(first_trigger["masked_memory_at_high_run_start"], 1)
        self.assertGreater(first_trigger["ciod_risk"], 0.0)

        summary = next(event.data for event in result.state.trace if event.event == "sarr_summary")
        self.assertEqual(summary["raw_low_count"], 1)
        self.assertEqual(summary["masked_uncertainty_count"], 1)
        self.assertEqual(summary["masked_uncertainty_gap"], 1)
        self.assertGreaterEqual(summary["max_high_run_length"], 3)
        self.assertGreater(summary["max_ciod_risk"], 0.0)
        self.assertEqual(summary["first_ciod_shadow_trigger_step"], triggered[0]["step_id"])

        problem_metrics = _confidence_process_metrics(steps)
        for key in [
            "raw_low_count",
            "smooth_low_count",
            "masked_uncertainty_count",
            "masked_uncertainty_gap",
            "max_high_run_length",
            "max_ciod_risk",
            "ciod_shadow_trigger_count",
            "first_ciod_shadow_trigger_step",
        ]:
            self.assertIn(key, problem_metrics)
        self.assertGreater(problem_metrics["max_ciod_risk"], 0.0)

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
