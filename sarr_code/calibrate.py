from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bpa.context_budget import ContextBudgetExceeded
from bpa.safety import CLOSE_THINK_TAG

from .calibration import PercentileNormalizer
from .config import SARRConfig


@dataclass
class CalibrationTrace:
    problem_id: str
    values: list[float] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None


def collect_slm_calibration_trace(problem_id: str, problem_text: str, slm, cfg: SARRConfig) -> CalibrationTrace:
    trace = CalibrationTrace(problem_id=problem_id)
    assistant_prefix = ""
    visible_tokens = 0

    try:
        step_id = 0
        while visible_tokens < cfg.generation.think_token_budget:
            step_id += 1
            remaining = cfg.generation.think_token_budget - visible_tokens
            if remaining <= 0:
                trace.stop_reason = "think_token_budget"
                break
            output = slm.generate_step(
                problem_text,
                assistant_prefix,
                max_new_tokens=max(1, min(cfg.generation.max_new_tokens_per_step, remaining)),
                stop_delimiters=cfg.generation.step_delimiters,
                capture_token_entropy=cfg.confidence.capture_slm_token_entropy,
                topk_entropy=cfg.confidence.topk_entropy,
            )
            assistant_prefix += output.text
            visible_tokens += output.token_count
            c_raw, c_info = slm.continuation_confidence(
                problem_text,
                assistant_prefix,
                topk=cfg.confidence.topk_entropy,
            )
            trace.values.append(float(c_raw))
            trace.steps.append(
                {
                    "problem_id": problem_id,
                    "step_id": step_id,
                    "text": output.text,
                    "token_count": output.token_count,
                    "finish_reason": output.finish_reason,
                    "c_raw": float(c_raw),
                    "confidence": c_info,
                }
            )
            if output.token_count <= 0 and not output.text.strip():
                trace.stop_reason = "empty_step"
                break
            if CLOSE_THINK_TAG in output.text:
                trace.stop_reason = "finished"
                break
            if output.finish_reason == "eos":
                trace.stop_reason = "eos"
                break
        if trace.stop_reason is None:
            trace.stop_reason = "think_token_budget"
    except ContextBudgetExceeded:
        trace.stop_reason = "context_budget"
    return trace


def build_calibration_payload(traces: list[CalibrationTrace], cfg: SARRConfig) -> dict[str, Any]:
    values = [value for trace in traces for value in trace.values]
    if not values:
        raise RuntimeError("Calibration produced no confidence values. Check local dataset paths and SLM generation.")
    normalizer = PercentileNormalizer(values)
    payload = normalizer.to_dict(topk_entropy=cfg.confidence.topk_entropy)
    payload["method"] = cfg.method
    payload["num_traces"] = len(traces)
    payload["trace_summaries"] = [
        {
            "problem_id": trace.problem_id,
            "num_values": len(trace.values),
            "stop_reason": trace.stop_reason,
        }
        for trace in traces
    ]
    return payload
