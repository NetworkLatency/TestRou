#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bpa.config import BPAConfig
from bpa.context_budget import ContextBudgetExceeded, generation_budget_for_rendered
from bpa.engines import (
    completion_logprobs,
    finish_reason,
    generated_text,
    generated_token_ids,
    init_engines,
    logprob_value,
)
from bpa.eval.benchmark_eval import benchmark_eval_match
from bpa.eval.datasets import load_eval_dataset
from bpa.eval.main_benchmark import (
    _existing_row_from_problem_output,
    build_summary_metrics,
    has_complete_problem_outputs,
    load_summary_rows,
    write_problem_outputs,
    write_summary_files,
)
from bpa.pipeline import (
    _account_generation_cost,
    solve_engine_only,
)
from bpa.render import render_for_continuation
from bpa.safety import CLOSE_THINK_TAG, ensure_step_terminator, extract_answer_from_steps
from bpa.state import Decision, GenerationState, Phase, TraceEvent
from bpa.trace import BPAResult, result_summary


MATH_DATASETS = {"math500", "aime24", "aime25"}
DEFAULT_GLIMP_THRESHOLDS = "0.3,0.6,1.0,1.5,2.0"
DEFAULT_DASR_THRESHOLDS = "-4.0,-3.0,-2.0,-1.5,-1.0,-0.75,-0.5,-0.25,-0.1"
ROUTING_POLICIES = {"glimprouter_hinit", "dasr_sk"}


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


@dataclass(frozen=True)
class ExperimentSpec:
    policy: str
    threshold: float
    variant: str


@dataclass
class ProbeResult:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    finish: str
    top_logprobs: dict[int, float]
    entropy: float | None = None
    s_k: float | None = None

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass
class StrictGeneration:
    text: str
    finish: str
    token_count: int


@dataclass
class DriftPreview:
    prev_window_size: int
    prev_mean: float | None
    delta: float | None
    volatility: float | None
    change_points: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_prev_window_size": self.prev_window_size,
            "drift_prev_mean": self.prev_mean,
            "drift_delta": self.delta,
            "drift_volatility": self.volatility,
            "drift_change_points": self.change_points,
        }


class DriftTracker:
    """Tracks DASR drift over accepted SLM steps only."""

    def __init__(self, window_size: int):
        self.window_size = int(window_size)
        self.s_window: deque[float] = deque(maxlen=max(1, self.window_size))
        self.delta_window: deque[float] = deque(maxlen=max(1, self.window_size))

    def reset(self) -> None:
        self.s_window.clear()
        self.delta_window.clear()

    def preview(self, s_k: float | None) -> DriftPreview:
        if s_k is None or not math.isfinite(s_k):
            return DriftPreview(len(self.s_window), None, None, None, None)

        prev_values = list(self.s_window)[-(self.window_size - 1) :] if self.window_size > 1 else []
        prev_mean = mean(prev_values) if prev_values else None
        delta = s_k - prev_mean if prev_mean is not None else None
        window_with_current = (prev_values + [s_k])[-self.window_size :]
        volatility = pstdev(window_with_current) if len(window_with_current) >= 2 else 0.0

        candidate_deltas = list(self.delta_window)
        if delta is not None and math.isfinite(delta):
            candidate_deltas.append(delta)
        candidate_deltas = candidate_deltas[-self.window_size :]
        change_points = _count_sign_changes(candidate_deltas) if candidate_deltas else 0
        return DriftPreview(
            prev_window_size=len(prev_values),
            prev_mean=prev_mean,
            delta=delta,
            volatility=volatility,
            change_points=change_points,
        )

    def accept(self, s_k: float | None, delta: float | None) -> None:
        if s_k is not None and math.isfinite(s_k):
            self.s_window.append(float(s_k))
        if delta is not None and math.isfinite(delta):
            self.delta_window.append(float(delta))


@dataclass
class RouteChoice:
    decision: Decision
    probe: ProbeResult
    log: dict[str, Any]


def _count_sign_changes(values: list[float]) -> int:
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in values if math.isfinite(value)]
    signs = [sign for sign in signs if sign != 0]
    if len(signs) < 2:
        return 0
    return sum(1 for prev, cur in zip(signs, signs[1:]) if prev != cur)


def parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(float(piece))
    return values


def threshold_slug(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def entropy_from_logprobs(logprobs: list[float]) -> float:
    finite = [float(lp) for lp in logprobs if lp is not None and math.isfinite(float(lp))]
    if not finite:
        return 0.0
    max_lp = max(finite)
    weights = [math.exp(lp - max_lp) for lp in finite]
    total = sum(weights)
    if total <= 0:
        return 0.0
    probs = [weight / total for weight in weights]
    return -sum(prob * math.log(prob + 1e-45) for prob in probs)


def _coerce_token_id(token_id: Any) -> int | None:
    try:
        return int(token_id)
    except (TypeError, ValueError):
        return None


def _top_logprobs_for_step(output: Any, step_idx: int = 0) -> dict[int, float]:
    steps = completion_logprobs(output)
    if step_idx >= len(steps) or not steps[step_idx]:
        token_ids = generated_token_ids(output)
        if step_idx < len(token_ids):
            return {int(token_ids[step_idx]): 0.0}
        return {}

    top: dict[int, float] = {}
    for token_id, record in dict(steps[step_idx]).items():
        coerced = _coerce_token_id(token_id)
        if coerced is not None:
            top[coerced] = logprob_value(record)
    return top


def _chosen_token_logprob(output: Any, step_idx: int, token_id: int) -> float:
    steps = completion_logprobs(output)
    if step_idx >= len(steps) or not steps[step_idx]:
        raise RuntimeError(
            "Model output did not include logprobs. Check that the backend supports logprobs "
            "and that --dasr-logprobs-topk/--hinit-logprobs-topk is at least 1."
        )

    step = dict(steps[step_idx])
    if token_id in step:
        return logprob_value(step[token_id])
    token_id_str = str(token_id)
    if token_id_str in step:
        return logprob_value(step[token_id_str])
    if len(step) == 1:
        return logprob_value(next(iter(step.values())))
    raise RuntimeError(f"Chosen token id {token_id} is missing from returned logprobs at probe position {step_idx}.")


def _probe_slm(
    state: GenerationState,
    slm,
    config: BPAConfig,
    *,
    max_tokens: int,
    logprobs_topk: int,
    prefix_extension: str = "",
    stop_on_step_boundary: bool = False,
) -> Any:
    rendered = render_for_continuation(
        state.problem_text,
        state.assistant_prefix_text + prefix_extension,
        slm.ensure_tokenizer(),
    )
    requested_tokens = max(1, int(max_tokens))
    max_tokens_budget, prompt_tokens = generation_budget_for_rendered(rendered, slm, config, requested_tokens)
    sampling_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens_budget,
        "temperature": 0.0,
        "logprobs": max(1, int(logprobs_topk)),
    }
    if stop_on_step_boundary:
        sampling_kwargs["stop"] = ["\n\n"]
        sampling_kwargs["include_stop_str_in_output"] = True
    sampling = slm.sampling_params(**sampling_kwargs)
    start = time.time()
    output = slm.generate(rendered, sampling)[0]
    wall_time = time.time() - start
    token_count = len(generated_token_ids(output))
    _account_generation_cost(
        state,
        "slm",
        wall_time=wall_time,
        token_count=token_count,
        prompt_tokens=prompt_tokens,
    )
    return output


def _generate_strict_step_with_engine(
    state: GenerationState,
    engine,
    config: BPAConfig,
    *,
    account: str,
    step_token_budget: int,
    prefix_extension: str = "",
) -> StrictGeneration:
    rendered = render_for_continuation(
        state.problem_text,
        state.assistant_prefix_text + prefix_extension,
        engine.ensure_tokenizer(),
    )
    max_tokens, prompt_tokens = generation_budget_for_rendered(
        rendered,
        engine,
        config,
        max(1, int(step_token_budget)),
    )
    sampling = engine.sampling_params(
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=1,
        stop=["\n\n"],
        include_stop_str_in_output=True,
    )
    start = time.time()
    output = engine.generate(rendered, sampling)[0]
    wall_time = time.time() - start
    token_count = len(generated_token_ids(output))
    _account_generation_cost(
        state,
        account,
        wall_time=wall_time,
        token_count=token_count,
        prompt_tokens=prompt_tokens,
    )
    return StrictGeneration(
        text=generated_text(output),
        finish=finish_reason(output),
        token_count=token_count,
    )


def _final_answer_prefix(thinking_text: str) -> str:
    thinking = thinking_text.split(CLOSE_THINK_TAG, 1)[0] if CLOSE_THINK_TAG in thinking_text else thinking_text
    return f"{thinking.rstrip()}\n{CLOSE_THINK_TAG}\n\n"


def _generate_strict_final_answer(
    state: GenerationState,
    engine,
    config: BPAConfig,
    *,
    account: str,
    answer_token_budget: int,
) -> StrictGeneration:
    rendered = render_for_continuation(
        state.problem_text,
        _final_answer_prefix(state.assistant_prefix_text),
        engine.ensure_tokenizer(),
    )
    max_tokens, prompt_tokens = generation_budget_for_rendered(
        rendered,
        engine,
        config,
        max(1, int(answer_token_budget)),
    )
    sampling = engine.sampling_params(max_tokens=max_tokens, temperature=0.0)
    start = time.time()
    output = engine.generate(rendered, sampling)[0]
    wall_time = time.time() - start
    token_count = len(generated_token_ids(output))
    _account_generation_cost(
        state,
        account,
        wall_time=wall_time,
        token_count=token_count,
        prompt_tokens=prompt_tokens,
    )
    return StrictGeneration(
        text=generated_text(output),
        finish=finish_reason(output),
        token_count=token_count,
    )


def compute_hinit_probe(state: GenerationState, slm, config: BPAConfig, *, logprobs_topk: int) -> ProbeResult:
    output = _probe_slm(
        state,
        slm,
        config,
        max_tokens=1,
        logprobs_topk=logprobs_topk,
        stop_on_step_boundary=False,
    )
    token_ids = generated_token_ids(output)
    top_logprobs = _top_logprobs_for_step(output, 0)
    token_logprobs = []
    for idx, token_id in enumerate(token_ids):
        token_logprobs.append(_chosen_token_logprob(output, idx, int(token_id)))
    entropy = entropy_from_logprobs(list(top_logprobs.values()))
    return ProbeResult(
        text=generated_text(output),
        token_ids=[int(token_id) for token_id in token_ids],
        token_logprobs=token_logprobs,
        finish=finish_reason(output),
        top_logprobs=top_logprobs,
        entropy=entropy,
    )


def compute_sk_probe(
    state: GenerationState,
    slm,
    config: BPAConfig,
    *,
    probe_tokens: int,
    logprobs_topk: int,
) -> ProbeResult:
    output = _probe_slm(
        state,
        slm,
        config,
        max_tokens=probe_tokens,
        logprobs_topk=logprobs_topk,
        stop_on_step_boundary=True,
    )
    token_ids = [int(token_id) for token_id in generated_token_ids(output)]
    token_logprobs = [
        _chosen_token_logprob(output, idx, token_id)
        for idx, token_id in enumerate(token_ids)
    ]
    s_k = (sum(token_logprobs) / len(token_logprobs)) if token_logprobs else float("-inf")
    return ProbeResult(
        text=generated_text(output),
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        finish=finish_reason(output),
        top_logprobs=_top_logprobs_for_step(output, 0),
        s_k=s_k,
    )


def choose_route(
    state: GenerationState,
    slm,
    config: BPAConfig,
    spec: ExperimentSpec,
    *,
    hinit_logprobs_topk: int,
    dasr_probe_tokens: int,
    dasr_logprobs_topk: int,
    drift_tracker: DriftTracker,
    remaining_think_tokens: int,
) -> RouteChoice:
    if spec.policy == "glimprouter_hinit":
        probe = compute_hinit_probe(state, slm, config, logprobs_topk=hinit_logprobs_topk)
        entropy = probe.entropy if probe.entropy is not None else float("inf")
        decision = Decision.SLM_DIRECT if entropy < spec.threshold else Decision.LLM_FULL
        sorted_top = sorted(probe.top_logprobs.items(), key=lambda item: item[1], reverse=True)
        return RouteChoice(
            decision=decision,
            probe=probe,
            log={
                "routing_policy": spec.policy,
                "threshold": spec.threshold,
                "h_init": entropy,
                "hinit_topk": hinit_logprobs_topk,
                "probe_text": probe.text,
                "probe_token_ids": probe.token_ids,
                "probe_token_logprobs": probe.token_logprobs,
                "probe_finish_reason": probe.finish,
                "top_token_ids": [token_id for token_id, _ in sorted_top],
                "top_logprobs": {str(token_id): lp for token_id, lp in sorted_top},
            },
        )

    if spec.policy == "dasr_sk":
        probe = compute_sk_probe(
            state,
            slm,
            config,
            probe_tokens=max(1, min(int(dasr_probe_tokens), int(remaining_think_tokens))),
            logprobs_topk=dasr_logprobs_topk,
        )
        s_k = probe.s_k if probe.s_k is not None else float("-inf")
        drift = drift_tracker.preview(s_k)
        decision = Decision.SLM_DIRECT if s_k >= spec.threshold else Decision.LLM_FULL
        log = {
            "routing_policy": spec.policy,
            "threshold": spec.threshold,
            "s_k": s_k,
            "probe_k": dasr_probe_tokens,
            "probe_len": len(probe.token_ids),
            "probe_text": probe.text,
            "probe_token_ids": probe.token_ids,
            "probe_token_logprobs": probe.token_logprobs,
            "probe_finish_reason": probe.finish,
        }
        log.update(drift.to_dict())
        return RouteChoice(decision=decision, probe=probe, log=log)

    raise ValueError(f"Unknown routing policy: {spec.policy}")


def _probe_already_finished_step(probe: ProbeResult) -> bool:
    if probe.finish in {"stop", "eos"}:
        return True
    return False


def _continue_trusted_probe(
    state: GenerationState,
    slm,
    config: BPAConfig,
    probe: ProbeResult,
    *,
    remaining_think_tokens: int,
    step_token_budget: int,
) -> StrictGeneration:
    if _probe_already_finished_step(probe):
        return StrictGeneration(probe.text, probe.finish, min(probe.token_count, remaining_think_tokens))

    remaining_after_probe = int(remaining_think_tokens) - probe.token_count
    if remaining_after_probe <= 0:
        return StrictGeneration(probe.text, "length", probe.token_count)

    continuation_budget = max(1, min(int(step_token_budget), remaining_after_probe))
    continuation = _generate_strict_step_with_engine(
        state,
        slm,
        config,
        account="slm",
        prefix_extension=probe.text,
        step_token_budget=continuation_budget,
    )
    return StrictGeneration(
        probe.text + continuation.text,
        continuation.finish,
        probe.token_count + continuation.token_count,
    )


def _append_strict_thinking_step(
    state: GenerationState,
    step_logs: list[dict[str, Any]],
    *,
    generation: StrictGeneration,
    decision: str,
    generation_source: str,
    generated_step_tokens: int,
    visible_think_tokens: int,
    step_token_budget: int,
    think_token_budget: int,
    extra_log: dict[str, Any] | None = None,
) -> str:
    step_text = ensure_step_terminator(generation.text, generation.finish)
    step_logs.append(
        {
            "step_idx": state.step_count,
            "decision": decision,
            "generation_source": generation_source,
            "finish_reason": generation.finish,
            "step_text": step_text,
            "generated_step_tokens": generated_step_tokens,
            "appended_think_tokens": generation.token_count,
            "visible_think_tokens_so_far": visible_think_tokens,
            "step_token_budget": step_token_budget,
            "think_token_budget": think_token_budget,
            "is_final_answer": False,
            **(extra_log or {}),
        }
    )
    state.assistant_prefix_text += step_text
    state.step_count += 1
    return step_text


def _strict_thinking_stop_reason(generation: StrictGeneration, step_text: str) -> str | None:
    if generation.token_count <= 0 and not step_text.strip():
        return "empty_step"
    if CLOSE_THINK_TAG in step_text:
        return "finished"
    if generation.finish == "eos":
        return "eos"
    return None


def _append_strict_final_answer(
    state: GenerationState,
    step_logs: list[dict[str, Any]],
    engine,
    config: BPAConfig,
    *,
    account: str,
    decision: str,
    answer_token_budget: int,
) -> None:
    before = state.slm_decode_tokens + state.llm_decode_tokens
    final_generation = _generate_strict_final_answer(
        state,
        engine,
        config,
        account=account,
        answer_token_budget=answer_token_budget,
    )
    step_logs.append(
        {
            "step_idx": state.step_count,
            "decision": decision,
            "generation_source": account,
            "finish_reason": final_generation.finish,
            "step_text": final_generation.text,
            "generated_step_tokens": state.slm_decode_tokens + state.llm_decode_tokens - before,
            "appended_answer_tokens": final_generation.token_count,
            "answer_token_budget": answer_token_budget,
            "is_final_answer": True,
        }
    )
    state.assistant_prefix_text = _final_answer_prefix(state.assistant_prefix_text) + final_generation.text
    state.step_count += 1


def _finish_strict_result(
    state: GenerationState,
    step_logs: list[dict[str, Any]],
    *,
    start_time: float,
    step_token_budget: int,
    think_token_budget: int,
    answer_token_budget: int,
    visible_think_tokens: int,
    include_routing_summary: bool,
) -> BPAResult:
    state.phase = Phase.DONE
    state.trace.append(
        TraceEvent(
            state.step_count,
            "strict_two_phase_budget",
            {
                "step_token_budget": step_token_budget,
                "think_token_budget": think_token_budget,
                "answer_token_budget": answer_token_budget,
                "visible_think_tokens": visible_think_tokens,
                "stop_reason": state.stop_reason,
            },
        )
    )
    state.trace.append(TraceEvent(state.step_count, "step_logs", {"steps": step_logs}))
    if include_routing_summary:
        state.trace.append(TraceEvent(state.step_count, "routing_summary", summarize_step_logs(step_logs)))
    return BPAResult(
        answer=extract_answer_from_steps(step_logs, state.assistant_prefix_text),
        state=state,
        total_wall_time=time.time() - start_time,
    )


def solve_routed_validation(
    problem_text: str,
    slm,
    llm,
    config: BPAConfig,
    spec: ExperimentSpec,
    *,
    hinit_logprobs_topk: int,
    dasr_probe_tokens: int,
    dasr_logprobs_topk: int,
    drift_window: int,
    step_token_budget: int,
    think_token_budget: int,
    answer_token_budget: int,
) -> BPAResult:
    state = GenerationState(problem_text=problem_text, generation_protocol=f"{spec.policy}_strict_two_phase")
    drift_tracker = DriftTracker(drift_window)
    start_time = time.time()
    step_logs: list[dict[str, Any]] = []
    visible_think_tokens = 0

    while visible_think_tokens < int(think_token_budget):
        remaining_think_tokens = int(think_token_budget) - visible_think_tokens
        if remaining_think_tokens <= 0:
            state.stop_reason = "think_token_budget"
            break

        try:
            decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
            route = choose_route(
                state,
                slm,
                config,
                spec,
                hinit_logprobs_topk=hinit_logprobs_topk,
                dasr_probe_tokens=dasr_probe_tokens,
                dasr_logprobs_topk=dasr_logprobs_topk,
                drift_tracker=drift_tracker,
                remaining_think_tokens=remaining_think_tokens,
            )

            if spec.policy == "glimprouter_hinit":
                # Source-faithful GlimpRouter: H_init is a scoring glimpse only.
                # The routed model then generates the whole step from the original context.
                step_budget = max(1, min(int(step_token_budget), remaining_think_tokens))
                if route.decision == Decision.SLM_DIRECT:
                    generation = _generate_strict_step_with_engine(
                        state,
                        slm,
                        config,
                        account="slm",
                        step_token_budget=step_budget,
                    )
                else:
                    generation = _generate_strict_step_with_engine(
                        state,
                        llm,
                        config,
                        account="llm",
                        step_token_budget=step_budget,
                    )
                appended_step_tokens = generation.token_count
                probe_reused = False
            elif route.decision == Decision.SLM_DIRECT:
                generation = _continue_trusted_probe(
                    state,
                    slm,
                    config,
                    route.probe,
                    remaining_think_tokens=remaining_think_tokens,
                    step_token_budget=step_token_budget,
                )
                appended_step_tokens = generation.token_count
                probe_reused = True
                drift_tracker.accept(route.probe.s_k, route.log.get("drift_delta"))
            else:
                drift_tracker.reset()
                step_budget = max(1, min(int(step_token_budget), remaining_think_tokens))
                generation = _generate_strict_step_with_engine(
                    state,
                    llm,
                    config,
                    account="llm",
                    step_token_budget=step_budget,
                )
                appended_step_tokens = generation.token_count
                probe_reused = False
        except ContextBudgetExceeded as exc:
            state.phase = Phase.DONE
            state.stop_reason = "context_budget"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
            break

        generated_step_tokens = state.slm_decode_tokens + state.llm_decode_tokens - decode_tokens_before
        visible_think_tokens += appended_step_tokens
        step_text = _append_strict_thinking_step(
            state,
            step_logs,
            generation=generation,
            decision=route.decision.value,
            generation_source="slm" if route.decision == Decision.SLM_DIRECT else "llm",
            generated_step_tokens=generated_step_tokens,
            visible_think_tokens=visible_think_tokens,
            step_token_budget=step_token_budget,
            think_token_budget=think_token_budget,
            extra_log={"probe_reused": probe_reused, **route.log},
        )

        stop_reason = _strict_thinking_stop_reason(
            StrictGeneration(generation.text, generation.finish, appended_step_tokens),
            step_text,
        )
        if stop_reason is not None:
            state.stop_reason = stop_reason
            if stop_reason == "empty_step":
                state.trace.append(TraceEvent(state.step_count, "empty_step", {}))
            break

    if state.stop_reason is None:
        state.stop_reason = "think_token_budget"

    if state.stop_reason != "context_budget":
        try:
            _append_strict_final_answer(
                state,
                step_logs,
                llm,
                config,
                account="llm",
                decision="llm_final_answer",
                answer_token_budget=answer_token_budget,
            )
        except ContextBudgetExceeded as exc:
            state.stop_reason = "context_budget_final_answer"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    return _finish_strict_result(
        state,
        step_logs,
        start_time=start_time,
        step_token_budget=step_token_budget,
        think_token_budget=think_token_budget,
        answer_token_budget=answer_token_budget,
        visible_think_tokens=visible_think_tokens,
        include_routing_summary=True,
    )


def summarize_step_logs(step_logs: list[dict[str, Any]]) -> dict[str, Any]:
    routed = [row for row in step_logs if row.get("routing_policy") in ROUTING_POLICIES]
    trust = [row for row in routed if row.get("decision") == Decision.SLM_DIRECT.value]
    verify = [row for row in routed if row.get("decision") == Decision.LLM_FULL.value]
    sk_values = [float(row["s_k"]) for row in routed if _finite_field(row, "s_k")]
    h_values = [float(row["h_init"]) for row in routed if _finite_field(row, "h_init")]
    abs_delta_values = [abs(float(row["drift_delta"])) for row in trust if _finite_field(row, "drift_delta")]
    volatility_values = [float(row["drift_volatility"]) for row in trust if _finite_field(row, "drift_volatility")]
    return {
        "routing_decision_count": len(routed),
        "trust_steps": len(trust),
        "verify_steps": len(verify),
        "llm_call_rate": (len(verify) / len(routed)) if routed else 0.0,
        "avg_s_k": mean(sk_values) if sk_values else None,
        "avg_h_init": mean(h_values) if h_values else None,
        "avg_abs_drift_delta": mean(abs_delta_values) if abs_delta_values else None,
        "avg_drift_volatility": mean(volatility_values) if volatility_values else None,
        "max_drift_change_points": max(
            (int(row["drift_change_points"]) for row in trust if _finite_field(row, "drift_change_points")),
            default=None,
        ),
    }


def _finite_field(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if value in (None, ""):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _problem_step_jsonl(output_root: Path, dataset: str, variant: str, problem_id: Any) -> Path:
    return output_root / dataset / variant / str(problem_id) / f"{problem_id}.steps.jsonl"


def load_problem_step_logs(output_root: Path, dataset: str, variant: str, problem_id: Any) -> list[dict[str, Any]]:
    path = _problem_step_jsonl(output_root, dataset, variant, problem_id)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sum_numeric(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _percentile(values: list[float], pct: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (float(pct) / 100.0) * (len(finite) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return finite[lo]
    frac = rank - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def iter_run_step_logs(output_root: Path, dataset: str, variant: str, rows: list[dict[str, Any]]):
    for row in rows:
        problem_id = row.get("problem_id")
        if problem_id in (None, ""):
            continue
        yield from load_problem_step_logs(output_root, dataset, variant, problem_id)


def build_validation_metrics(
    output_root: Path,
    dataset: str,
    spec: ExperimentSpec,
    rows: list[dict[str, Any]],
    dataset_wall_time: float,
) -> dict[str, Any]:
    metrics = build_summary_metrics(dataset, spec.variant, rows, dataset_wall_time)
    route_total = _sum_numeric(rows, "routing_decision_count")
    verify_total = _sum_numeric(rows, "verify_steps")
    trust_total = _sum_numeric(rows, "trust_steps")
    metrics.update(
        {
            "policy": spec.policy,
            "threshold": spec.threshold,
            "routing_decision_count": route_total,
            "trust_steps": trust_total,
            "verify_steps": verify_total,
            "llm_call_rate": (verify_total / route_total) if route_total else 0.0,
        }
    )

    step_rows = list(iter_run_step_logs(output_root, dataset, spec.variant, rows))
    trusted_dasr = [
        row
        for row in step_rows
        if row.get("routing_policy") == "dasr_sk"
        and row.get("generation_source") == "slm"
    ]
    abs_delta_values = [abs(float(row["drift_delta"])) for row in trusted_dasr if _finite_field(row, "drift_delta")]
    volatility_values = [
        float(row["drift_volatility"])
        for row in trusted_dasr
        if _finite_field(row, "drift_volatility")
    ]
    metrics.update(
        {
            "drift_abs_delta_p70": _percentile(abs_delta_values, 70.0),
            "drift_volatility_p70": _percentile(volatility_values, 70.0),
            "drift_change_points_threshold": 3,
        }
    )
    return metrics


def _summary_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary.csv"


def _metrics_path(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / variant / "summary_metrics.json"


def enrich_row_with_routing_summary(
    output_root: Path,
    dataset: str,
    variant: str,
    problem_id: Any,
    row: dict[str, Any],
    step_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    step_logs = step_logs if step_logs is not None else load_problem_step_logs(output_root, dataset, variant, problem_id)
    enriched = dict(row)
    enriched.update(summarize_step_logs(step_logs))
    return enriched


def run_spec(
    spec: ExperimentSpec,
    *,
    config: BPAConfig,
    problems,
    slm,
    llm,
    output_root: Path,
    dataset: str,
    resume: bool,
    hinit_logprobs_topk: int,
    dasr_probe_tokens: int,
    dasr_logprobs_topk: int,
    drift_window: int,
    step_token_budget: int,
    think_token_budget: int,
    answer_token_budget: int,
) -> dict[str, Any]:
    summary_path = _summary_path(output_root, dataset, spec.variant)
    existing_summary_rows = load_summary_rows(summary_path) if resume else {}
    rows_by_problem_id: dict[str, dict[str, Any]] = {}
    skipped = 0

    if resume:
        for problem in problems:
            problem_id = str(problem.problem_id)
            if has_complete_problem_outputs(output_root, dataset, spec.variant, problem.problem_id):
                row = _existing_row_from_problem_output(
                    output_root,
                    dataset,
                    spec.variant,
                    problem,
                    config,
                    existing_summary_rows.get(problem_id),
                )
                if row is not None:
                    rows_by_problem_id[problem_id] = enrich_row_with_routing_summary(
                        output_root,
                        dataset,
                        spec.variant,
                        problem.problem_id,
                        row,
                    )
                    skipped += 1

    def ordered_rows() -> list[dict[str, Any]]:
        return [
            rows_by_problem_id[str(problem.problem_id)]
            for problem in problems
            if str(problem.problem_id) in rows_by_problem_id
        ]

    dataset_start = time.time()
    progress = tqdm(problems, desc=f"{spec.variant}:{dataset}")
    for problem in progress:
        problem_id = str(problem.problem_id)
        if resume and problem_id in rows_by_problem_id:
            continue

        problem_start = time.time()
        result = solve_routed_validation(
            problem.problem_text,
            slm,
            llm,
            config,
            spec,
            hinit_logprobs_topk=hinit_logprobs_topk,
            dasr_probe_tokens=dasr_probe_tokens,
            dasr_logprobs_topk=dasr_logprobs_topk,
            drift_window=drift_window,
            step_token_budget=step_token_budget,
            think_token_budget=think_token_budget,
            answer_token_budget=answer_token_budget,
        )
        if problem.gold_answer is not None:
            predicted = result.answer if dataset in MATH_DATASETS else result.state.assistant_prefix_text
            result.correct = benchmark_eval_match(predicted, problem.gold_answer, dataset)

        step_logs = _step_rows(result)
        summary = result_summary(result)
        summary.update(
            {
                "dataset": dataset,
                "variant": spec.variant,
                "policy": spec.policy,
                "threshold": spec.threshold,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                "gold_answer": problem.gold_answer,
            }
        )
        summary.update(summarize_step_logs(step_logs))
        write_problem_outputs(output_root, dataset, spec.variant, problem, result, config)
        if config.reset_prefix_cache_after_problem:
            slm.clear_runtime_cache()
            llm.clear_runtime_cache()
        summary["problem_wall_time"] = time.time() - problem_start
        rows_by_problem_id[problem_id] = summary
        rows = ordered_rows()
        write_summary_files(
            summary_path,
            rows,
            build_validation_metrics(output_root, dataset, spec, rows, time.time() - dataset_start),
        )

    rows = ordered_rows()
    metrics = build_validation_metrics(output_root, dataset, spec, rows, time.time() - dataset_start)
    write_summary_files(summary_path, rows, metrics)
    if resume and skipped:
        print(f"{spec.variant}: skipped {skipped} completed problem(s).")
    print(f"Wrote {summary_path}")
    print(f"Wrote {_metrics_path(output_root, dataset, spec.variant)}")
    return metrics


def solve_strict_engine_only(
    problem_text: str,
    engine,
    config: BPAConfig,
    *,
    account: str,
    variant: str,
    step_token_budget: int,
    think_token_budget: int,
    answer_token_budget: int,
) -> BPAResult:
    state = GenerationState(problem_text=problem_text, generation_protocol=f"{variant}_strict_two_phase")
    start_time = time.time()
    step_logs: list[dict[str, Any]] = []
    visible_think_tokens = 0

    while visible_think_tokens < int(think_token_budget):
        remaining = int(think_token_budget) - visible_think_tokens
        if remaining <= 0:
            break
        try:
            before = state.slm_decode_tokens + state.llm_decode_tokens
            generation = _generate_strict_step_with_engine(
                state,
                engine,
                config,
                account=account,
                step_token_budget=max(1, min(int(step_token_budget), remaining)),
            )
        except ContextBudgetExceeded as exc:
            state.stop_reason = "context_budget"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))
            break

        visible_think_tokens += generation.token_count
        step_text = _append_strict_thinking_step(
            state,
            step_logs,
            generation=generation,
            decision=variant,
            generation_source=account,
            generated_step_tokens=state.slm_decode_tokens + state.llm_decode_tokens - before,
            visible_think_tokens=visible_think_tokens,
            step_token_budget=step_token_budget,
            think_token_budget=think_token_budget,
        )

        stop_reason = _strict_thinking_stop_reason(generation, step_text)
        if stop_reason is not None:
            state.stop_reason = stop_reason
            break

    if state.stop_reason is None:
        state.stop_reason = "think_token_budget"

    if state.stop_reason != "context_budget":
        try:
            _append_strict_final_answer(
                state,
                step_logs,
                engine,
                config,
                account=account,
                decision=f"{variant}_final_answer",
                answer_token_budget=answer_token_budget,
            )
        except ContextBudgetExceeded as exc:
            state.stop_reason = "context_budget_final_answer"
            state.trace.append(TraceEvent(state.step_count, "context_budget_exhausted", exc.to_trace_data()))

    return _finish_strict_result(
        state,
        step_logs,
        start_time=start_time,
        step_token_budget=step_token_budget,
        think_token_budget=think_token_budget,
        answer_token_budget=answer_token_budget,
        visible_think_tokens=visible_think_tokens,
        include_routing_summary=False,
    )


def run_engine_baseline(
    variant: str,
    *,
    engine,
    account: str,
    config: BPAConfig,
    problems,
    output_root: Path,
    dataset: str,
    resume: bool,
    baseline_protocol: str,
    step_token_budget: int,
    think_token_budget: int,
    answer_token_budget: int,
) -> dict[str, Any]:
    summary_path = _summary_path(output_root, dataset, variant)
    existing_summary_rows = load_summary_rows(summary_path) if resume else {}
    rows_by_problem_id: dict[str, dict[str, Any]] = {}
    skipped = 0

    if resume:
        for problem in problems:
            problem_id = str(problem.problem_id)
            if has_complete_problem_outputs(output_root, dataset, variant, problem.problem_id):
                row = _existing_row_from_problem_output(
                    output_root,
                    dataset,
                    variant,
                    problem,
                    config,
                    existing_summary_rows.get(problem_id),
                )
                if row is not None:
                    rows_by_problem_id[problem_id] = row
                    skipped += 1

    def ordered_rows() -> list[dict[str, Any]]:
        return [
            rows_by_problem_id[str(problem.problem_id)]
            for problem in problems
            if str(problem.problem_id) in rows_by_problem_id
        ]

    dataset_start = time.time()
    for problem in tqdm(problems, desc=f"{variant}:{dataset}"):
        problem_id = str(problem.problem_id)
        if resume and problem_id in rows_by_problem_id:
            continue

        problem_start = time.time()
        if baseline_protocol == "strict":
            result = solve_strict_engine_only(
                problem.problem_text,
                engine,
                config,
                account=account,
                variant=variant,
                step_token_budget=step_token_budget,
                think_token_budget=think_token_budget,
                answer_token_budget=answer_token_budget,
            )
        else:
            result = solve_engine_only(problem.problem_text, engine, config, account=account)
        if problem.gold_answer is not None:
            predicted = result.answer if dataset in MATH_DATASETS else result.state.assistant_prefix_text
            result.correct = benchmark_eval_match(predicted, problem.gold_answer, dataset)

        summary = result_summary(result)
        summary.update(
            {
                "dataset": dataset,
                "variant": variant,
                "policy": variant,
                "threshold": None,
                "problem_id": problem.problem_id,
                "question_id": problem.question_id,
                "gold_answer": problem.gold_answer,
            }
        )
        write_problem_outputs(output_root, dataset, variant, problem, result, config)
        if config.reset_prefix_cache_after_problem:
            engine.clear_runtime_cache()
        summary["problem_wall_time"] = time.time() - problem_start
        rows_by_problem_id[problem_id] = summary

        rows = ordered_rows()
        write_summary_files(
            summary_path,
            rows,
            build_summary_metrics(dataset, variant, rows, time.time() - dataset_start),
        )

    rows = ordered_rows()
    metrics = build_summary_metrics(dataset, variant, rows, time.time() - dataset_start)
    metrics["policy"] = variant
    metrics["threshold"] = None
    write_summary_files(summary_path, rows, metrics)
    if resume and skipped:
        print(f"{variant}: skipped {skipped} completed problem(s).")
    print(f"Wrote {summary_path}")
    print(f"Wrote {_metrics_path(output_root, dataset, variant)}")
    return metrics


def _step_rows(result: BPAResult) -> list[dict[str, Any]]:
    for event in result.state.trace:
        if event.event == "step_logs":
            return event.data.get("steps", [])
    return []


def load_baseline_metrics(args: argparse.Namespace) -> dict[str, Any]:
    def read_metrics(path_text: str | None) -> dict[str, Any]:
        if not path_text:
            return {}
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    slm_metrics = read_metrics(args.slm_only_metrics)
    llm_metrics = read_metrics(args.llm_only_metrics)
    slm_accuracy = args.slm_only_accuracy
    if slm_accuracy is None and slm_metrics:
        slm_accuracy = _optional_float(slm_metrics.get("accuracy"))

    llm_latency = args.llm_only_latency
    if llm_latency is None and llm_metrics:
        llm_latency = _optional_float(llm_metrics.get(args.latency_key))
        if llm_latency is None:
            llm_latency = _optional_float(llm_metrics.get("avg_problem_wall_time"))
        if llm_latency is None:
            llm_latency = _optional_float(llm_metrics.get("avg_total_wall_time"))

    return {
        "slm_only_accuracy": slm_accuracy,
        "llm_only_latency": llm_latency,
        "latency_key": args.latency_key,
        "slm_only_metrics": slm_metrics,
        "llm_only_metrics": llm_metrics,
    }


def _metric_latency(metrics: dict[str, Any], latency_key: str) -> float | None:
    latency = _optional_float(metrics.get(latency_key))
    if latency is not None:
        return latency
    latency = _optional_float(metrics.get("avg_problem_wall_time"))
    if latency is not None:
        return latency
    return _optional_float(metrics.get("avg_total_wall_time"))


def complete_baseline_metrics(
    baselines: dict[str, Any],
    generated_metrics: dict[str, dict[str, Any]],
    *,
    latency_key: str,
) -> dict[str, Any]:
    merged = dict(baselines)
    slm_generated = generated_metrics.get("slm_only") or {}
    llm_generated = generated_metrics.get("llm_only") or {}
    if merged.get("slm_only_accuracy") is None and slm_generated:
        merged["slm_only_accuracy"] = _optional_float(slm_generated.get("accuracy"))
        merged["slm_only_metrics"] = slm_generated
    if merged.get("llm_only_latency") is None and llm_generated:
        merged["llm_only_latency"] = _metric_latency(llm_generated, latency_key)
        merged["llm_only_metrics"] = llm_generated
    return merged


def baseline_variants_to_run(args: argparse.Namespace, supplied_baselines: dict[str, Any]) -> set[str]:
    if args.baselines == "never":
        return set()
    if args.baselines == "always":
        return {"slm_only", "llm_only"}

    needed = set()
    if supplied_baselines.get("slm_only_accuracy") is None:
        needed.add("slm_only")
    if supplied_baselines.get("llm_only_latency") is None:
        needed.add("llm_only")
    return needed


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_gate_report(
    metrics_rows: list[dict[str, Any]],
    baselines: dict[str, Any],
    *,
    stage1_acc_margin: float,
    stage1_latency_frac: float,
    call_rate_tolerance: float,
    latency_key: str,
) -> dict[str, Any]:
    glimp = [row for row in metrics_rows if row.get("policy") == "glimprouter_hinit"]
    dasr = [row for row in metrics_rows if row.get("policy") == "dasr_sk"]

    slm_acc = baselines.get("slm_only_accuracy")
    llm_latency = baselines.get("llm_only_latency")
    stage1_candidates = []
    for row in glimp:
        accuracy = _optional_float(row.get("accuracy"))
        latency = _optional_float(row.get(latency_key)) or _optional_float(row.get("avg_problem_wall_time"))
        pass_gate = None
        if slm_acc is not None and llm_latency is not None and accuracy is not None and latency is not None:
            pass_gate = (accuracy >= slm_acc + stage1_acc_margin) and (latency <= llm_latency * stage1_latency_frac)
        stage1_candidates.append(
            {
                "variant": row.get("variant"),
                "threshold": row.get("threshold"),
                "accuracy": accuracy,
                "latency": latency,
                "llm_call_rate": _optional_float(row.get("llm_call_rate")),
                "pass": pass_gate,
            }
        )

    stage2_pairs = []
    for g_row in glimp:
        g_rate = _optional_float(g_row.get("llm_call_rate"))
        if g_rate is None or not dasr:
            continue
        nearest = min(
            dasr,
            key=lambda row: abs((_optional_float(row.get("llm_call_rate")) or 0.0) - g_rate),
        )
        d_rate = _optional_float(nearest.get("llm_call_rate")) or 0.0
        rate_diff = abs(d_rate - g_rate)
        g_acc = _optional_float(g_row.get("accuracy"))
        d_acc = _optional_float(nearest.get("accuracy"))
        pass_gate = (
            rate_diff <= call_rate_tolerance
            and g_acc is not None
            and d_acc is not None
            and d_acc >= g_acc
        )
        stage2_pairs.append(
            {
                "glimprouter_variant": g_row.get("variant"),
                "glimprouter_threshold": g_row.get("threshold"),
                "glimprouter_accuracy": g_acc,
                "glimprouter_llm_call_rate": g_rate,
                "dasr_variant": nearest.get("variant"),
                "dasr_threshold": nearest.get("threshold"),
                "dasr_accuracy": d_acc,
                "dasr_llm_call_rate": d_rate,
                "llm_call_rate_diff": rate_diff,
                "pass": pass_gate,
            }
        )

    stage1_go = None
    if stage1_candidates and all(candidate["pass"] is not None for candidate in stage1_candidates):
        stage1_go = any(candidate["pass"] for candidate in stage1_candidates)

    stage2_go = None
    if stage2_pairs:
        stage2_go = any(pair["pass"] for pair in stage2_pairs)

    return {
        "baselines": {
            "slm_only_accuracy": slm_acc,
            "llm_only_latency": llm_latency,
            "latency_key": latency_key,
        },
        "stage1_gate": {
            "accuracy_margin": stage1_acc_margin,
            "latency_fraction": stage1_latency_frac,
            "go": stage1_go,
            "candidates": stage1_candidates,
            "note": "go is null when SLM-only accuracy or LLM-only latency was not supplied.",
        },
        "stage2_gate": {
            "call_rate_tolerance": call_rate_tolerance,
            "go": stage2_go,
            "pairs": stage2_pairs,
        },
    }


def write_gate_outputs(output_root: Path, dataset: str, gate_report: dict[str, Any]) -> None:
    root = output_root / dataset
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "dasr_validation_gate_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(gate_report, f, ensure_ascii=False, indent=2)

    pairs = gate_report.get("stage2_gate", {}).get("pairs", [])
    if pairs:
        comparison_path = root / "dasr_stage2_call_rate_comparison.csv"
        fieldnames = sorted({key for row in pairs for key in row})
        with comparison_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pairs)
        print(f"Wrote {comparison_path}")
    print(f"Wrote {report_path}")


def build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs = []
    if args.stage in {"stage1", "both"}:
        for tau in parse_float_list(args.glimp_thresholds):
            specs.append(
                ExperimentSpec(
                    policy="glimprouter_hinit",
                    threshold=tau,
                    variant=f"glimprouter_tau_{threshold_slug(tau)}",
                )
            )
    if args.stage in {"stage2", "both"}:
        for threshold in parse_float_list(args.dasr_thresholds):
            specs.append(
                ExperimentSpec(
                    policy="dasr_sk",
                    threshold=threshold,
                    variant=f"dasr_sk_{threshold_slug(threshold)}",
                )
            )
    return specs


def main() -> None:
    raise_csv_field_limit()

    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage-1 GlimpRouter H_init sweep and Stage-2 DASR s_k validation "
            "with optional same-format SLM-only/LLM-only baselines."
        )
    )
    parser.add_argument("--config", required=True, help="Path to BPAConfig JSON.")
    parser.add_argument("--dataset", default="aime25", choices=["math500", "aime24", "aime25", "gpqa", "gpqa_diamond"])
    parser.add_argument("--max-problems", type=int, default=30)
    parser.add_argument("--stage", choices=["stage1", "stage2", "both"], default="both")
    parser.add_argument("--output-root", default=None, help="Override config.output_dir for this validation run.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-problem outputs.")
    parser.add_argument("--glimp-thresholds", default=DEFAULT_GLIMP_THRESHOLDS)
    parser.add_argument(
        "--dasr-thresholds",
        default=DEFAULT_DASR_THRESHOLDS,
        help="Comma-separated s_k thresholds. Use --dasr-thresholds=-2,-1,... when values start with '-'.",
    )
    parser.add_argument("--hinit-logprobs-topk", type=int, default=20)
    parser.add_argument("--dasr-probe-tokens", type=int, default=8)
    parser.add_argument("--dasr-logprobs-topk", type=int, default=1)
    parser.add_argument("--drift-window", type=int, default=5)
    parser.add_argument(
        "--step-token-budget",
        type=int,
        default=512,
        help="Strict per-step token budget; the official GlimpRouter source uses 512.",
    )
    parser.add_argument(
        "--think-token-budget",
        type=int,
        default=8192,
        help="Strict thinking-token budget for GlimpRouter/DASR source-style two-phase generation.",
    )
    parser.add_argument(
        "--answer-token-budget",
        type=int,
        default=2048,
        help="Strict final-answer token budget; the official GlimpRouter source uses 2048.",
    )
    parser.add_argument("--slm-only-metrics", default=None, help="Optional SLM-only summary_metrics.json.")
    parser.add_argument("--llm-only-metrics", default=None, help="Optional LLM-only summary_metrics.json.")
    parser.add_argument("--slm-only-accuracy", type=float, default=None)
    parser.add_argument("--llm-only-latency", type=float, default=None)
    parser.add_argument(
        "--baselines",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Run SLM-only/LLM-only inside this script. auto runs whichever baseline is not supplied "
            "through --slm-only-* or --llm-only-*."
        ),
    )
    parser.add_argument(
        "--baseline-protocol",
        choices=["strict", "oneshot"],
        default="strict",
        help="strict uses the same think/answer token budgets; oneshot preserves the older solve_engine_only baseline.",
    )
    parser.add_argument("--latency-key", default="avg_problem_wall_time")
    parser.add_argument("--stage1-acc-margin", type=float, default=0.05)
    parser.add_argument("--stage1-latency-frac", type=float, default=0.7)
    parser.add_argument("--call-rate-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    output_root = Path(args.output_root) if args.output_root else Path(config.output_dir) / "dasr_validation"
    problems = load_eval_dataset(args.dataset, config, max_problems=args.max_problems)
    specs = build_specs(args)
    if not specs:
        raise SystemExit("No experiment specs were selected.")

    print(f"Loaded {len(problems)} problem(s) from {args.dataset}.")
    print(f"Running {len(specs)} validation spec(s). Output root: {output_root}")
    slm, llm = init_engines(config)

    supplied_baselines = load_baseline_metrics(args)
    generated_baseline_metrics: dict[str, dict[str, Any]] = {}
    baseline_runs = baseline_variants_to_run(args, supplied_baselines)
    if baseline_runs:
        print(f"Running baseline(s): {', '.join(sorted(baseline_runs))}")
    if "slm_only" in baseline_runs:
        generated_baseline_metrics["slm_only"] = run_engine_baseline(
            "slm_only",
            engine=slm,
            account="slm",
            config=config,
            problems=problems,
            output_root=output_root,
            dataset=args.dataset,
            resume=args.resume,
            baseline_protocol=args.baseline_protocol,
            step_token_budget=args.step_token_budget,
            think_token_budget=args.think_token_budget,
            answer_token_budget=args.answer_token_budget,
        )
    if "llm_only" in baseline_runs:
        generated_baseline_metrics["llm_only"] = run_engine_baseline(
            "llm_only",
            engine=llm,
            account="llm",
            config=config,
            problems=problems,
            output_root=output_root,
            dataset=args.dataset,
            resume=args.resume,
            baseline_protocol=args.baseline_protocol,
            step_token_budget=args.step_token_budget,
            think_token_budget=args.think_token_budget,
            answer_token_budget=args.answer_token_budget,
        )

    metrics_rows = []
    for spec in specs:
        metrics = run_spec(
            spec,
            config=config,
            problems=problems,
            slm=slm,
            llm=llm,
            output_root=output_root,
            dataset=args.dataset,
            resume=args.resume,
            hinit_logprobs_topk=args.hinit_logprobs_topk,
            dasr_probe_tokens=args.dasr_probe_tokens,
            dasr_logprobs_topk=args.dasr_logprobs_topk,
            drift_window=args.drift_window,
            step_token_budget=args.step_token_budget,
            think_token_budget=args.think_token_budget,
            answer_token_budget=args.answer_token_budget,
        )
        metrics_rows.append(metrics)

    gate_report = build_gate_report(
        metrics_rows,
        complete_baseline_metrics(
            supplied_baselines,
            generated_baseline_metrics,
            latency_key=args.latency_key,
        ),
        stage1_acc_margin=args.stage1_acc_margin,
        stage1_latency_frac=args.stage1_latency_frac,
        call_rate_tolerance=args.call_rate_tolerance,
        latency_key=args.latency_key,
    )
    write_gate_outputs(output_root, args.dataset, gate_report)


if __name__ == "__main__":
    main()
