from __future__ import annotations

import unittest

from sarr_code.algorithm import choose_best_prefix_anchor, run_sarr_code
from sarr_code.calibration import PercentileNormalizer, code_style_degeneration_event, smooth_confidence
from sarr_code.config import (
    CIODConfig,
    ConfidenceConfig,
    ControllerConfig,
    GenerationConfig,
    LoggingConfig,
    ModelRuntimeConfig,
    RuntimeConfig,
    SARRConfig,
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


def make_cfg(**ciod_kwargs) -> SARRConfig:
    """Minimal config for unit tests. CIODConfig with on_threshold=99 disables CI-OD routing."""
    return SARRConfig(
        slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
        llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
        generation=GenerationConfig(
            max_new_tokens_per_step=32,
            think_token_budget=512,
            answer_token_budget=64,
            final_answer_generator="llm",
            force_close_think_on_budget=True,
        ),
        confidence=ConfidenceConfig(top_k=20, smooth_window=2),
        ciod=CIODConfig(on_threshold=0.99, **ciod_kwargs),  # disable CIOD by default
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

    def test_choose_best_prefix_anchor_uses_c_smooth(self):
        records = [
            StepRecord("p", 1, "slm", "a", [], c_smooth=None),
            StepRecord("p", 2, "slm", "b", [], c_smooth=0.4),
            StepRecord("p", 3, "slm", "c", [], c_smooth=0.3),
        ]
        self.assertEqual(choose_best_prefix_anchor(records), 2)
        self.assertEqual(choose_best_prefix_anchor([records[0]]), 0)

    def test_low_confidence_step_stays_in_trace(self):
        """Low c_raw steps produce no routing action; all text is kept in trace."""
        cfg = make_cfg()
        slm = FakeEngine(
            outputs=["first\n\n", "second\n\n", "bad\n\n", "</think>\n\n"],
            confidences=[0.5, 0.6, 0.3, 0.8],
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        result, steps, driver_switches, _ = run_sarr_code(
            problem_id="p1", problem_text="6*7?", slm=slm, llm=llm, cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(driver_switches, [])
        self.assertFalse(any(row["removed_by_rollback"] for row in steps))
        self.assertIn("bad", result.state.assistant_prefix_text)
        # c_smooth is logged for SLM steps
        slm_steps = [s for s in steps if s["generator"] == "slm" and s["c_raw"] is not None]
        self.assertTrue(all(s["c_smooth"] is not None for s in slm_steps))

    def test_close_think_step_skips_confidence(self):
        cfg = make_cfg()
        slm = FakeEngine(outputs=["Done.\n</think>\n\n"], confidences=[])
        llm = FakeEngine(outputs=["Final answer: 42."])

        result, steps, _, _ = run_sarr_code(
            problem_id="close", problem_text="6*7?", slm=slm, llm=llm, cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(slm.confidence_calls, 0)
        self.assertEqual(steps[0]["action"], "FINISHED")
        self.assertTrue(steps[0]["extra"]["confidence_skipped"])

    def test_no_llm_routing_without_ciod_trigger(self):
        """All SLM steps complete without LLM when CI-OD risk stays below threshold."""
        cfg = make_cfg()  # on_threshold=0.99 → never triggers
        slm = FakeEngine(
            outputs=["step1\n\n", "step2\n\n", "step3\n\n", "</think>\n\n"],
            confidences=[0.9, 0.1, 0.9],  # spike but risk won't build to 0.99
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        result, steps, driver_switches, _ = run_sarr_code(
            problem_id="no-routing", problem_text="test", slm=slm, llm=llm, cfg=cfg,
        )

        self.assertEqual(result.answer, "42")
        self.assertEqual(driver_switches, [])
        self.assertFalse(any(
            row["generator"] == "llm" and not row.get("is_final_answer") for row in steps
        ))
        for text in ["step1", "step2", "step3"]:
            self.assertIn(text, result.state.assistant_prefix_text)

    def test_anchor_refreshed_on_slm_active_steps(self):
        """clean_autonomy_anchor advances on each SLM step in SLM_ACTIVE."""
        cfg = make_cfg()
        slm = FakeEngine(
            outputs=["s1\n\n", "s2\n\n", "s3\n\n", "</think>\n\n"],
            confidences=[0.8, 0.8, 0.8],
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        _, steps, _, _ = run_sarr_code(
            problem_id="anchor", problem_text="test", slm=slm, llm=llm, cfg=cfg,
        )

        slm_steps = [s for s in steps if s["generator"] == "slm" and s["c_raw"] is not None]
        anchors = [s["extra"]["clean_autonomy_anchor"] for s in slm_steps]
        # Each step advances the anchor to its own step_id
        for i, (step, anchor) in enumerate(zip(slm_steps, anchors)):
            self.assertEqual(anchor, step["step_id"])

    def test_c_smooth_is_moving_average(self):
        """c_smooth equals rolling mean of last smooth_window c_raw values."""
        cfg = make_cfg()
        # smooth_window=2: c_smooth[i] = mean(c_raw[i-1], c_raw[i])
        slm = FakeEngine(
            outputs=["s1\n\n", "s2\n\n", "s3\n\n", "</think>\n\n"],
            confidences=[0.8, 0.4, 0.6],
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        _, steps, _, _ = run_sarr_code(
            problem_id="smooth", problem_text="test", slm=slm, llm=llm, cfg=cfg,
        )

        slm_steps = [s for s in steps if s["generator"] == "slm" and s["c_raw"] is not None]
        # step 1: only 1 value → smooth = 0.8
        self.assertAlmostEqual(slm_steps[0]["c_smooth"], 0.8, places=5)
        # step 2: mean(0.8, 0.4) = 0.6
        self.assertAlmostEqual(slm_steps[1]["c_smooth"], 0.6, places=5)
        # step 3: mean(0.4, 0.6) = 0.5
        self.assertAlmostEqual(slm_steps[2]["c_smooth"], 0.5, places=5)

    def test_ciod_driver_switching(self):
        """CI-OD risk triggers SWITCH_TO_LLM_BY_CIOD; probe passes → SWITCH_TO_SLM_BY_REENTRY_RISK.

        Parameters chosen so CIOD triggers at step 4 (h3) and re-entry probe passes
        after exactly 2 LLM steps (probe_c=0.5, exposure_decay=0.70 drains exposure
        below exposure_e0=0.5 within 2 steps).
        """
        cfg = SARRConfig(
            slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
            llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
            generation=GenerationConfig(
                max_new_tokens_per_step=32, think_token_budget=512,
                answer_token_budget=64, final_answer_generator="llm",
            ),
            confidence=ConfidenceConfig(top_k=20, smooth_window=2),
            ciod=CIODConfig(
                masked_low_threshold=0.35,
                exposure_threshold=0.60,
                masked_decay=0.95,
                exposure_decay=0.70,  # fast drain so probe passes in 2 steps
                min_masked_memory=0.5,
                exposure_e0=0.5,
                hazard_scale=1.0,
                on_threshold=0.10,
                off_threshold=0.01,
            ),
            runtime=RuntimeConfig(max_model_len=4096),
        )
        # SLM outputs: h1, masked, h2, h3(trigger), h4(new segment), </think>
        # SLM confidences: gen×4, probe×2, gen×1
        slm = FakeEngine(
            outputs=["h1\n\n", "masked\n\n", "h2\n\n", "h3\n\n", "h4\n\n", "</think>\n\n"],
            confidences=[0.9, 0.1, 0.9, 0.9, 0.5, 0.5, 0.9],
        )
        llm = FakeEngine(outputs=["l1\n\n", "l2\n\n", "Final answer: 42."])

        result, steps, driver_switches, _ = run_sarr_code(
            problem_id="ciod-switch", problem_text="test", slm=slm, llm=llm, cfg=cfg,
        )

        self.assertEqual(result.answer, "42")

        # One CIOD trigger on SLM side
        ciod_triggers = [s for s in steps if s["action"] == "SWITCH_TO_LLM_BY_CIOD"]
        self.assertEqual(len(ciod_triggers), 1)
        self.assertEqual(ciod_triggers[0]["generator"], "slm")

        # Two LLM steps: first keeps, second switches back
        llm_steps = [s for s in steps if s["generator"] == "llm" and not s.get("is_final_answer")]
        self.assertEqual(len(llm_steps), 2)
        self.assertEqual(llm_steps[0]["action"], "KEEP_LLM_BY_REENTRY_RISK")
        self.assertEqual(llm_steps[1]["action"], "SWITCH_TO_SLM_BY_REENTRY_RISK")

        # slm_reentry_risk logged on LLM steps
        self.assertIsNotNone(llm_steps[0]["extra"]["slm_reentry_risk"])
        self.assertGreater(llm_steps[0]["extra"]["slm_reentry_risk"], cfg.ciod.off_threshold)
        self.assertLessEqual(llm_steps[1]["extra"]["slm_reentry_risk"], cfg.ciod.off_threshold)

        # Two driver switch events (SLM→LLM, LLM→SLM)
        self.assertEqual(len(driver_switches), 2)
        self.assertEqual(driver_switches[0]["from"], "SLM_ACTIVE")
        self.assertEqual(driver_switches[0]["to"], "LLM_ACTIVE")
        self.assertEqual(driver_switches[1]["from"], "LLM_ACTIVE")
        self.assertEqual(driver_switches[1]["to"], "SLM_ACTIVE")

        # SLM steps after return are in new segment (segment_id=2)
        post_return_slm = [
            s for s in steps
            if s["generator"] == "slm"
            and s["extra"].get("segment_id", 1) == 2
        ]
        self.assertTrue(post_return_slm)

    def test_ciod_segment_tracker_risk_computation(self):
        """Segment CI-OD risk in extra matches expected formula values."""
        cfg = SARRConfig(
            slm=ModelRuntimeConfig(model_path="unused", backend="transformers"),
            llm=ModelRuntimeConfig(model_path="unused", backend="openai", api_base_url="http://127.0.0.1:8000/v1"),
            generation=GenerationConfig(
                max_new_tokens_per_step=32, think_token_budget=512,
                answer_token_budget=64, final_answer_generator="llm",
            ),
            confidence=ConfidenceConfig(top_k=20, smooth_window=2),
            ciod=CIODConfig(
                on_threshold=0.99,   # disable routing so we can inspect all steps
                min_masked_memory=0.5,
                exposure_e0=0.5,
                hazard_scale=1.0,
            ),
            runtime=RuntimeConfig(max_model_len=4096),
        )
        slm = FakeEngine(
            outputs=["h1\n\n", "masked\n\n", "h2\n\n", "h3\n\n", "</think>\n\n"],
            confidences=[0.9, 0.1, 0.9, 0.9],
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        _, steps, driver_switches, _ = run_sarr_code(
            problem_id="risk-check", problem_text="test", slm=slm, llm=llm, cfg=cfg,
        )

        self.assertEqual(driver_switches, [])

        slm_steps = [s for s in steps if s["generator"] == "slm" and s["c_raw"] is not None]

        # Step 1 (h1): no masked → masked_memory=0 → risk=0
        self.assertAlmostEqual(slm_steps[0]["extra"]["masked_memory"], 0.0, places=5)
        self.assertAlmostEqual(slm_steps[0]["extra"]["ciod_risk"], 0.0, places=5)

        # Step 2 (masked): raw_low=True, smooth_low=False (smooth=(0.9+0.1)/2=0.5>0.35)
        # → masked_uncertainty=True → masked_memory = 0.98*0 + 1 = 1.0 (with default decay=0.98)
        # Wait: test uses default CIODConfig except on_threshold.
        # masked_decay=0.98 (default). masked_memory after step 2 = 0.98*0 + 1 = 1.0
        mm_step2 = slm_steps[1]["extra"]["masked_memory"]
        self.assertGreater(mm_step2, 0.9)   # should be close to 1.0
        self.assertLessEqual(mm_step2, 1.0)

        # Step 4 (h3): c_smooth = mean(0.9, 0.9) = 0.9 >= exposure_threshold=0.60
        # → exposure accumulates → ciod_risk should be > 0 (assuming mm >= min_masked_memory=0.5)
        risk_step4 = slm_steps[3]["extra"]["ciod_risk"]
        self.assertGreater(risk_step4, 0.0)

    def test_budget_exhausted_forces_close_think(self):
        """When think_token_budget is hit, force_close_think_text is appended."""
        cfg = make_cfg()
        # Use small budget to force exhaustion
        cfg.generation.think_token_budget = 20
        slm = FakeEngine(
            outputs=["longer step text here\n\n", "more text here\n\n"],
            confidences=[0.8, 0.8],
        )
        llm = FakeEngine(outputs=["Final answer: 42."])

        result, steps, _, _ = run_sarr_code(
            problem_id="budget", problem_text="test", slm=slm, llm=llm, cfg=cfg,
        )

        self.assertIn("</think>", result.state.assistant_prefix_text)
        self.assertIn("think_token_budget", result.state.stop_reason)

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
