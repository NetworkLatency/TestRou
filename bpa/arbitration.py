from __future__ import annotations

import time
from difflib import SequenceMatcher

from .config import BPAConfig
from .engines import logprob_value, prompt_logprobs
from .render import render_for_continuation
from .state import ArbitrationResult, BranchCandidate, BranchScore, GenerationState, SpanLocateResult


def _apply_scoring_context_window(locate: SpanLocateResult, window: int) -> SpanLocateResult:
    if window <= 0 or locate.branch_start_token <= window:
        return locate
    trim = locate.branch_start_token - window
    token_ids = locate.token_ids[trim:]
    return SpanLocateResult(
        token_ids=token_ids,
        branch_start_token=locate.branch_start_token - trim,
        branch_end_token=locate.branch_end_token - trim,
        span_method=locate.span_method,
        has_boundary_crossing_token=locate.has_boundary_crossing_token,
        char_start=locate.char_start,
        char_end=locate.char_end,
        is_invalid=locate.is_invalid,
        invalid_reason=locate.invalid_reason,
    )


def _longest_common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _char_overlap_ratio(decoded_span: str, branch_text: str) -> float:
    if not decoded_span and not branch_text:
        return 1.0
    return SequenceMatcher(None, decoded_span, branch_text).ratio()


def locate_branch_token_span(
    problem_text: str,
    assistant_prefix_text: str,
    branch_text: str,
    llm_tokenizer,
) -> SpanLocateResult:
    rendered_prefix = render_for_continuation(problem_text, assistant_prefix_text, llm_tokenizer)
    rendered_full = render_for_continuation(problem_text, assistant_prefix_text + branch_text, llm_tokenizer)

    lcp_len = _longest_common_prefix_len(rendered_prefix, rendered_full)
    char_start_lcp = lcp_len
    char_end_lcp = char_start_lcp + len(branch_text)
    if char_end_lcp <= len(rendered_full) and rendered_full[char_start_lcp:char_end_lcp] == branch_text:
        char_start, char_end, span_method = char_start_lcp, char_end_lcp, "lcp"
    else:
        idx = rendered_full.rfind(branch_text)
        if idx >= 0:
            char_start, char_end, span_method = idx, idx + len(branch_text), "rfind_after_lcp_fail"
        else:
            char_start, char_end, span_method = lcp_len, len(rendered_full), "prefix_len_fallback"

    if not getattr(llm_tokenizer, "is_fast", False):
        return SpanLocateResult(
            token_ids=[],
            branch_start_token=0,
            branch_end_token=0,
            span_method="invalid",
            has_boundary_crossing_token=False,
            char_start=char_start,
            char_end=char_end,
            is_invalid=True,
            invalid_reason="tokenizer_not_fast",
        )

    encoding = llm_tokenizer(rendered_full, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(encoding["input_ids"])
    offsets = list(encoding["offset_mapping"])

    branch_token_idxs = []
    has_crossing = False
    for tok_idx, (start, end) in enumerate(offsets):
        if end <= char_start:
            continue
        if start >= char_end:
            break
        if start < char_start or end > char_end:
            has_crossing = True
        branch_token_idxs.append(tok_idx)

    if not branch_token_idxs:
        return SpanLocateResult(
            token_ids=token_ids,
            branch_start_token=0,
            branch_end_token=0,
            span_method=span_method,
            has_boundary_crossing_token=False,
            char_start=char_start,
            char_end=char_end,
            is_invalid=True,
            invalid_reason="no_tokens_in_span",
        )

    decoded_span = llm_tokenizer.decode(
        [token_ids[i] for i in branch_token_idxs],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    overlap = _char_overlap_ratio(decoded_span, branch_text)
    is_invalid = overlap < 0.7
    return SpanLocateResult(
        token_ids=token_ids,
        branch_start_token=branch_token_idxs[0],
        branch_end_token=branch_token_idxs[-1] + 1,
        span_method=span_method,
        has_boundary_crossing_token=has_crossing,
        char_start=char_start,
        char_end=char_end,
        is_invalid=is_invalid,
        invalid_reason=None if not is_invalid else f"low_decode_overlap_{overlap:.2f}",
    )


def score_branch(state: GenerationState, llm, branch_text: str, config: BPAConfig) -> BranchScore:
    locate = locate_branch_token_span(state.problem_text, state.assistant_prefix_text, branch_text, llm.ensure_tokenizer())
    if locate.is_invalid:
        return BranchScore(
            mean_logprob=None,
            branch_token_count=0,
            span_locate=locate,
            is_invalid=True,
            invalid_reason=locate.invalid_reason,
            prefill_tokens=0,
        )
    locate = _apply_scoring_context_window(locate, config.llm_scoring_context_window)

    sampling = llm.sampling_params(max_tokens=1, temperature=0.0, prompt_logprobs=config.prompt_logprobs_topk)
    generate_start = time.time()
    out = llm.generate(llm.tokens_prompt(locate.token_ids), sampling)[0]
    state.llm_scoring_wall_time += time.time() - generate_start
    state.llm_prefill_tokens += len(locate.token_ids)
    state.llm_decode_tokens += 1
    state.llm_scoring_calls += 1

    plp = prompt_logprobs(out)
    branch_lps: list[float] = []
    missing_count = 0
    for i in range(locate.branch_start_token, locate.branch_end_token):
        if i >= len(plp) or plp[i] is None:
            missing_count += 1
            continue
        actual = plp[i].get(locate.token_ids[i])
        if actual is None:
            missing_count += 1
            continue
        branch_lps.append(logprob_value(actual))

    branch_token_count = locate.branch_end_token - locate.branch_start_token
    if branch_token_count == 0:
        return BranchScore(
            mean_logprob=None,
            branch_token_count=0,
            span_locate=locate,
            is_invalid=True,
            invalid_reason="empty_span",
            prefill_tokens=len(locate.token_ids),
        )
    missing_ratio = missing_count / branch_token_count
    if missing_ratio > config.score_missing_ratio_thresh or not branch_lps:
        return BranchScore(
            mean_logprob=None,
            branch_token_count=branch_token_count,
            span_locate=locate,
            is_invalid=True,
            invalid_reason=f"missing_ratio_{missing_ratio:.2f}",
            prefill_tokens=len(locate.token_ids),
            missing_count=missing_count,
            missing_ratio=missing_ratio,
        )

    return BranchScore(
        mean_logprob=sum(branch_lps) / len(branch_lps),
        branch_token_count=branch_token_count,
        span_locate=locate,
        is_invalid=False,
        invalid_reason=None,
        prefill_tokens=len(locate.token_ids),
        missing_count=missing_count,
        missing_ratio=missing_ratio,
    )


def _apply_scoring_context_window(locate: SpanLocateResult, context_window: int) -> SpanLocateResult:
    if context_window <= 0:
        return locate
    if locate.branch_start_token <= context_window:
        return locate
    crop_start = locate.branch_start_token - context_window
    return SpanLocateResult(
        token_ids=locate.token_ids[crop_start:],
        branch_start_token=locate.branch_start_token - crop_start,
        branch_end_token=locate.branch_end_token - crop_start,
        span_method=f"{locate.span_method}_context_window_{context_window}",
        has_boundary_crossing_token=locate.has_boundary_crossing_token,
        char_start=locate.char_start,
        char_end=locate.char_end,
        is_invalid=locate.is_invalid,
        invalid_reason=locate.invalid_reason,
    )


def llm_arbitrate(
    state: GenerationState,
    llm,
    b1: BranchCandidate,
    b2: BranchCandidate,
    config: BPAConfig,
) -> ArbitrationResult:
    s1 = score_branch(state, llm, b1.step_branch_text, config)
    s2 = score_branch(state, llm, b2.step_branch_text, config)
    if s1.is_invalid or s2.is_invalid:
        return ArbitrationResult(
            score1=s1,
            score2=s2,
            winner_idx=0,
            is_invalid=True,
            invalid_reason=f"s1={s1.invalid_reason} s2={s2.invalid_reason}",
        )
    if abs((s1.mean_logprob or 0.0) - (s2.mean_logprob or 0.0)) < config.arbitration_tie_margin:
        winner_idx = 0
    else:
        winner_idx = 0 if (s1.mean_logprob or float("-inf")) >= (s2.mean_logprob or float("-inf")) else 1
    return ArbitrationResult(score1=s1, score2=s2, winner_idx=winner_idx, is_invalid=False, invalid_reason=None)
