from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .arbitration import llm_arbitrate
from .cascade.l0 import l0_filter
from .cascade.l1 import l1_shadow_rollout
from .cascade.l2 import l2_compute
from .config import BPAConfig
from .engines import finish_reason, generated_text, generated_token_ids
from .phase_machine import CLOSE_THINK_TAG, check_and_transition_phase
from .render import render_for_continuation
from .safety import clean_latex_answer, ensure_step_terminator, extract_answer, extract_last_boxed, update_repetition
from .state import CascadeResult, Decision, GenerationState, Phase, RejectedBranch, RepetitionState, TraceEvent
from .trace import BPAResult


EOS_FINISH_REASONS = {"eos", "branch_eos"}


def run_cascade(state: GenerationState, slm, llm, config: BPAConfig, disabled: bool = False) -> CascadeResult:
    l0 = l0_filter(state, slm, config)
    state.trace.append(TraceEvent(state.step_count, "l0", l0.to_dict()))

    if config.cascade_mode == "hinit":
        decision = Decision.LLM_FULL if (not disabled and l0.passed) else Decision.SLM_DIRECT
        return CascadeResult(decision=decision, l0=l0)

    if disabled or not l0.passed:
        return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0)

    try:
        branch1, branch2 = l1_shadow_rollout(state, slm, config, l0)
    except ValueError as exc:
        state.trace.append(TraceEvent(state.step_count, "l1_skipped", {"reason": str(exc)}))
        return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0)

    state.trace.append(
        TraceEvent(
            state.step_count,
            "l1",
            {
                "b1_text": branch1.raw_rollout_text[:200],
                "b2_text": branch2.raw_rollout_text[:200],
                "b1_truncated": branch1.step_branch_was_truncated,
                "b2_truncated": branch2.step_branch_was_truncated,
                "b1_finish_reason": branch1.finish_reason,
                "b2_finish_reason": branch2.finish_reason,
                "b1_ended_by_eos": branch1.ended_by_eos,
                "b2_ended_by_eos": branch2.ended_by_eos,
            },
        )
    )

    l2 = l2_compute(branch1, branch2, config)
    state.trace.append(TraceEvent(state.step_count, "l2", l2.to_dict()))

    arbitration = None
    if l2.triggered_arbitration:
        arbitration = llm_arbitrate(state, llm, branch1, branch2, config)
        state.trace.append(TraceEvent(state.step_count, "arbitration", arbitration.to_dict()))

    if config.collect_branch_logs:
        state.branch_logs.append(
            {
                "step_idx": state.step_count,
                "problem_text": state.problem_text,
                "assistant_prefix_text": state.assistant_prefix_text,
                "l0": l0.to_dict(),
                "branch1": branch1.to_dict(),
                "branch2": branch2.to_dict(),
                "l2": l2.to_dict(),
                "arbitration": arbitration.to_dict() if arbitration is not None else None,
            }
        )

    if not l2.triggered_arbitration:
        return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0, l1=(branch1, branch2), l2=l2)

    if arbitration is None or arbitration.is_invalid:
        if config.invalid_fallback == "llm_full":
            return CascadeResult(
                decision=Decision.LLM_FULL,
                l0=l0,
                l1=(branch1, branch2),
                l2=l2,
                arbitration=arbitration,
            )
        return CascadeResult(
            decision=Decision.SLM_DIRECT,
            l0=l0,
            l1=(branch1, branch2),
            l2=l2,
            arbitration=arbitration,
        )

    winner = branch1 if arbitration.winner_idx == 0 else branch2
    loser = branch2 if arbitration.winner_idx == 0 else branch1
    state.rejected_branches_log.append(
        RejectedBranch(
            step_idx=state.step_count,
            loser_text=loser.step_branch_text,
            winner_text=winner.step_branch_text,
            l2=l2,
        )
    )
    return CascadeResult(
        decision=Decision.LLM_ARBITRATE,
        l0=l0,
        l1=(branch1, branch2),
        l2=l2,
        arbitration=arbitration,
        winner_branch=winner,
    )


def _generate_step_with_engine(state: GenerationState, engine, config: BPAConfig, account: str, prefix_extension: str = "") -> tuple[str, str]:
    rendered = render_for_continuation(
        state.problem_text,
        state.assistant_prefix_text + prefix_extension,
        engine.ensure_tokenizer(),
    )
    sampling = engine.sampling_params(
        max_tokens=config.max_step_tokens,
        temperature=0.0,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    out = engine.generate(rendered, sampling)[0]
    token_count = len(generated_token_ids(out))
    if account == "slm":
        state.slm_decode_tokens += token_count
        state.slm_prefill_tokens += len(engine.encode(rendered))
    else:
        state.llm_decode_tokens += token_count
        state.llm_prefill_tokens += len(engine.encode(rendered))
        state.llm_full_calls += 1
    return generated_text(out), finish_reason(out)


def _slm_generate_step(state: GenerationState, slm, config: BPAConfig) -> tuple[str, str]:
    return _generate_step_with_engine(state, slm, config, account="slm")


def _slm_continue_step(state: GenerationState, slm, config: BPAConfig, prefix_extension: str) -> tuple[str, str]:
    return _generate_step_with_engine(state, slm, config, account="slm", prefix_extension=prefix_extension)


def _llm_generate_step(state: GenerationState, llm, config: BPAConfig) -> tuple[str, str]:
    return _generate_step_with_engine(state, llm, config, account="llm")


def generate_one_step(state: GenerationState, slm, llm, config: BPAConfig, cascade: CascadeResult) -> tuple[str, str]:
    if cascade.decision == Decision.SLM_DIRECT:
        return _slm_generate_step(state, slm, config)

    if cascade.decision == Decision.LLM_FULL:
        return _llm_generate_step(state, llm, config)

    if cascade.decision == Decision.LLM_ARBITRATE:
        if not config.apply_arbitration:
            return _slm_generate_step(state, slm, config)
        if cascade.winner_branch is None:
            return _slm_generate_step(state, slm, config)
        winner = cascade.winner_branch
        if winner.ended_by_eos:
            return winner.step_branch_text, "branch_eos"
        if winner.step_branch_was_truncated:
            return winner.step_branch_text + "\n\n", "stop_in_branch"
        suffix_text, finish = _slm_continue_step(state, slm, config, prefix_extension=winner.step_branch_text)
        return winner.step_branch_text + suffix_text, finish

    raise ValueError(f"Unknown cascade decision: {cascade.decision}")


def _ends_with_chunk_stop(text: str, stops: list[str]) -> bool:
    return any(text.endswith(stop) for stop in stops)


def llm_generate_final_with_rep_guard(state: GenerationState, llm, config: BPAConfig) -> tuple[str, str]:
    rep = RepetitionState()
    accumulated_text = ""
    accumulated_tokens = 0
    chunk_stops = [".\n", "!\n", "?\n", "\n\n"]

    while accumulated_tokens < config.final_answer_max_tokens:
        rendered = render_for_continuation(
            state.problem_text,
            state.assistant_prefix_text + accumulated_text,
            llm.ensure_tokenizer(),
        )
        remaining = config.final_answer_max_tokens - accumulated_tokens
        sampling = llm.sampling_params(
            max_tokens=min(remaining, config.final_answer_chunk_tokens),
            temperature=0.0,
            stop=chunk_stops,
            include_stop_str_in_output=True,
        )
        out = llm.generate(rendered, sampling)[0]
        chunk_text = generated_text(out)
        chunk_tokens = len(generated_token_ids(out))
        chunk_finish = finish_reason(out)

        state.llm_decode_tokens += chunk_tokens
        state.llm_prefill_tokens += len(llm.encode(rendered))
        state.llm_full_calls += 1

        accumulated_text += chunk_text
        accumulated_tokens += chunk_tokens

        if chunk_finish == "eos" or (chunk_finish == "stop" and not _ends_with_chunk_stop(chunk_text, chunk_stops)):
            return accumulated_text, "eos"

        trigger = update_repetition(
            rep,
            chunk_text,
            config.repetition_ngram_size,
            config.repetition_ngram_threshold,
        )
        if trigger is not None:
            return accumulated_text, "repetition"

        if not chunk_text or chunk_tokens == 0:
            return accumulated_text, "empty_chunk"

    return accumulated_text, "max_tokens"


def _normalized_step_for_duplicate(step_text: str) -> str:
    return step_text.rstrip()


def _is_eos_finish(finish: str) -> bool:
    return finish in EOS_FINISH_REASONS


def _extract_final_answer_candidate(step_text: str) -> str | None:
    boxed = extract_last_boxed(step_text)
    if boxed is not None:
        return clean_latex_answer(boxed)
    return extract_answer(step_text)


def _has_final_answer_marker(step_text: str) -> bool:
    lower = step_text.lower()
    if "final answer" in lower:
        return True
    if re.search(r"\banswer\s*:", lower):
        return True
    if re.search(r"\bthe\s+answer\s+is\b", lower):
        return True
    if re.search(r"\btherefore\b[\s\S]{0,120}\\boxed", lower):
        return True
    return False


def _looks_like_restart_after_answer(step_text: str) -> bool:
    lower = step_text.lstrip().lower()
    if not lower:
        return False
    restart_prefixes = (
        "step-by-step",
        "step by step",
        "let's solve",
        "lets solve",
        "we need",
        "first,",
        "first ",
        "to solve",
        "solution:",
        "reasoning",
        "we start",
    )
    return lower.startswith(restart_prefixes)


def _generation_source(cascade: CascadeResult, config: BPAConfig) -> str:
    if cascade.decision == Decision.LLM_FULL:
        return "llm"
    if cascade.decision == Decision.LLM_ARBITRATE and config.apply_arbitration and cascade.winner_branch is not None:
        return "slm_branch_llm_arbitrated"
    return "slm"


@dataclass
class FinalStopDecision:
    stop_reason: str | None = None
    append_step: bool = True
    signal: str | None = None
    event_data: dict | None = None


@dataclass
class FinalStopState:
    previous_step: str | None = None
    previous_candidate: str | None = None
    candidate_repeat_count: int = 0
    candidate_marker_seen: bool = False
    seen_marked_candidate: str | None = None
    final_steps: int = 0
    final_tokens: int = 0

    def observe(self, step_text: str, finish: str, generated_tokens: int, config: BPAConfig) -> FinalStopDecision:
        self.final_steps += 1
        self.final_tokens += max(generated_tokens, 0)

        normalized_step = _normalized_step_for_duplicate(step_text)
        candidate = _extract_final_answer_candidate(step_text)
        has_marker = _has_final_answer_marker(step_text)

        if candidate is not None and candidate == self.previous_candidate:
            self.candidate_repeat_count += 1
            self.candidate_marker_seen = self.candidate_marker_seen or has_marker
        elif candidate is not None:
            self.candidate_repeat_count = 1
            self.candidate_marker_seen = has_marker
        else:
            self.candidate_repeat_count = 0
            self.candidate_marker_seen = False

        if candidate is not None and has_marker:
            self.seen_marked_candidate = candidate

        decision = FinalStopDecision(event_data={"finish_reason": finish})

        if _is_eos_finish(finish):
            decision.stop_reason = "final_eos"
            decision.signal = "final_eos"
        elif self.previous_step is not None and normalized_step == self.previous_step:
            decision.stop_reason = "final_duplicate_step"
            decision.signal = "final_duplicate_step"
            decision.event_data = {"step_text": normalized_step[:200]}
        elif (
            candidate is not None
            and self.candidate_repeat_count >= config.final_answer_stability_repeats
            and self.candidate_marker_seen
        ):
            decision.stop_reason = "final_answer_stable"
            decision.signal = "final_answer_stable"
            decision.event_data = {
                "answer_candidate": candidate,
                "repeat_count": self.candidate_repeat_count,
            }
        elif (
            self.seen_marked_candidate is not None
            and _looks_like_restart_after_answer(step_text)
            and candidate != self.seen_marked_candidate
        ):
            decision.stop_reason = "final_restarted_after_answer"
            decision.signal = "final_restarted_after_answer"
            decision.append_step = False
            decision.event_data = {
                "previous_answer_candidate": self.seen_marked_candidate,
                "restart_preview": normalized_step[:200],
            }
        elif self.final_steps >= config.max_final_steps:
            decision.stop_reason = "final_step_budget"
            decision.signal = "final_step_budget"
            decision.event_data = {"final_steps": self.final_steps}
        elif self.final_tokens >= config.max_final_tokens:
            decision.stop_reason = "final_token_budget"
            decision.signal = "final_token_budget"
            decision.event_data = {"final_tokens": self.final_tokens}

        self.previous_step = normalized_step
        self.previous_candidate = candidate
        return decision


def bpa_solve(problem_text: str, slm, llm, config: BPAConfig) -> BPAResult:
    state = GenerationState(problem_text=problem_text)
    rep = RepetitionState()
    final_stop = FinalStopState()
    start_time = time.time()
    step_logs: list[dict] = []

    while state.phase != Phase.DONE:
        if state.slm_decode_tokens + state.llm_decode_tokens >= config.max_total_tokens:
            state.trace.append(TraceEvent(state.step_count, "budget_exhausted_total", {}))
            state.stop_reason = "budget"
            break

        phase_before_step = state.phase

        if state.phase == Phase.FINAL_ANSWER and config.final_answer_mode == "llm_chunked":
            final_text, final_stop_reason = llm_generate_final_with_rep_guard(state, llm, config)
            state.assistant_prefix_text += final_text
            state.phase = Phase.DONE
            state.stop_reason = f"final_{final_stop_reason}"
            if final_stop_reason == "repetition":
                state.trace.append(TraceEvent(state.step_count, "final_answer_stopped_by_repetition", {}))
            break

        intervention_disabled = state.llm_scoring_calls >= config.max_llm_interventions * 2
        cascade = run_cascade(state, slm, llm, config, disabled=intervention_disabled)
        decode_tokens_before = state.slm_decode_tokens + state.llm_decode_tokens
        step_text, finish = generate_one_step(state, slm, llm, config, cascade)
        step_text_normalized = ensure_step_terminator(step_text, finish)
        generated_step_tokens = state.slm_decode_tokens + state.llm_decode_tokens - decode_tokens_before
        final_decision: FinalStopDecision | None = None

        if phase_before_step == Phase.FINAL_ANSWER:
            final_decision = final_stop.observe(step_text_normalized, finish, generated_step_tokens, config)

        if final_decision is None or final_decision.append_step:
            state.assistant_prefix_text += step_text_normalized
        state.step_count += 1

        if final_decision is None or final_decision.append_step:
            check_and_transition_phase(state, step_text_normalized)

        if phase_before_step == Phase.FINAL_ANSWER:
            if final_decision is not None and final_decision.stop_reason is not None:
                state.phase = Phase.DONE
                state.stop_reason = final_decision.stop_reason
                if final_decision.signal is not None:
                    state.trace.append(
                        TraceEvent(
                            state.step_count,
                            final_decision.signal,
                            final_decision.event_data or {},
                        )
                    )

        log_row = {
            "step_idx": state.step_count - 1,
            "decision": cascade.decision.value,
            "generation_source": _generation_source(cascade, config),
            "finish_reason": finish,
            "step_text": step_text_normalized,
            "phase_after": state.phase.value,
        }
        if cascade.winner_branch is not None:
            log_row["branch_finish_reason"] = cascade.winner_branch.finish_reason
        if final_decision is not None and final_decision.signal is not None:
            log_row["final_stop_signal"] = final_decision.signal
        step_logs.append(log_row)

        if state.phase == Phase.THINKING:
            trigger = update_repetition(
                rep,
                step_text_normalized,
                config.repetition_ngram_size,
                config.repetition_ngram_threshold,
            )
            if trigger is not None:
                state.assistant_prefix_text += CLOSE_THINK_TAG + "\n\n"
                state.has_seen_close_think = True
                state.phase = Phase.FINAL_ANSWER
                state.trace.append(
                    TraceEvent(
                        state.step_count,
                        "thinking_repetition_force_close",
                        {"trigger_reason": trigger},
                    )
                )
                rep = RepetitionState()

        if _is_eos_finish(finish) and phase_before_step != Phase.FINAL_ANSWER:
            state.phase = Phase.DONE
            state.stop_reason = "eos"

    state.trace.append(TraceEvent(state.step_count, "step_logs", {"steps": step_logs}))
    return BPAResult(
        answer=extract_answer(state.assistant_prefix_text),
        state=state,
        total_wall_time=time.time() - start_time,
    )


def solve_engine_only(problem_text: str, engine, config: BPAConfig, account: str) -> BPAResult:
    state = GenerationState(problem_text=problem_text)
    start_time = time.time()
    rendered = render_for_continuation(problem_text, "", engine.ensure_tokenizer())
    sampling = engine.sampling_params(max_tokens=config.max_total_tokens, temperature=0.0)
    out = engine.generate(rendered, sampling)[0]
    text = generated_text(out)
    token_count = len(generated_token_ids(out))
    if account == "slm":
        state.slm_decode_tokens += token_count
        state.slm_prefill_tokens += len(engine.encode(rendered))
    else:
        state.llm_decode_tokens += token_count
        state.llm_prefill_tokens += len(engine.encode(rendered))
        state.llm_full_calls += 1
    state.assistant_prefix_text = text
    state.phase = Phase.DONE
    state.stop_reason = finish_reason(out) or "done"
    return BPAResult(answer=extract_answer(text), state=state, total_wall_time=time.time() - start_time)
