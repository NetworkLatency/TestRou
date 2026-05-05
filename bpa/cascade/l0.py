from __future__ import annotations

import math
import time

import numpy as np

from bpa.config import BPAConfig
from bpa.context_budget import generation_budget_for_rendered
from bpa.engines import completion_logprobs, generated_token_ids, logprob_value
from bpa.render import render_for_continuation
from bpa.state import GenerationState, L0Result


GLIMPROUTER_HINIT_TOPK = 10
GLIMPROUTER_HINIT_MARGIN_THRESH = 0.4
GLIMPROUTER_HINIT_ENTROPY_THRESH = 0.5


def classify_first_char(token_str: str) -> str:
    if not token_str:
        return "empty"
    s = token_str.lstrip()
    if not s:
        return "whitespace"
    c = s[0]
    if c.isalpha():
        return "alpha"
    if c.isdigit():
        return "digit"
    if c == "\\" and any(s.startswith(p) for p in [r"\frac", r"\sqrt", r"\sum", r"\int", r"\(", r"\["]):
        return "latex_command"
    if c in "*_#`>~|-":
        return "markdown"
    if c == "<":
        return "special_tag"
    return "other_symbol"


def entropy_and_margin(top_logprobs: dict[int, float]) -> tuple[float, float]:
    if not top_logprobs:
        return 0.0, 1.0
    sorted_lps = sorted(top_logprobs.values(), reverse=True)
    max_lp = max(sorted_lps)
    weights = np.array([math.exp(lp - max_lp) for lp in sorted_lps], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0, 1.0
    probs = weights / total
    h = float(-np.sum(probs * np.log(probs + 1e-10)) / np.log(max(len(probs), 2)))
    margin = float(probs[0] - probs[1]) if len(probs) > 1 else 1.0
    return h, margin


def _top_logprobs_from_output(output) -> dict[int, float]:
    logprobs = completion_logprobs(output)
    if not logprobs:
        ids = generated_token_ids(output)
        return {ids[0]: 0.0} if ids else {}
    first = logprobs[0] or {}
    return {int(tok_id): logprob_value(record) for tok_id, record in first.items()}


def l0_filter(state: GenerationState, slm, config: BPAConfig) -> L0Result:
    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, slm.ensure_tokenizer())
    max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, slm, config, 1)
    sampling = slm.sampling_params(max_tokens=max_tokens, temperature=0.0, logprobs=GLIMPROUTER_HINIT_TOPK)
    generate_start = time.time()
    out = slm.generate(rendered, sampling)[0]
    state.slm_wall_time += time.time() - generate_start
    state.slm_generate_calls += 1
    token_ids = generated_token_ids(out)
    state.slm_decode_tokens += len(token_ids) or 1
    state.slm_prefill_tokens += prompt_tokens

    top_logprobs = _top_logprobs_from_output(out)
    h_init, margin = entropy_and_margin(top_logprobs)
    sorted_ids = [tok_id for tok_id, _ in sorted(top_logprobs.items(), key=lambda kv: kv[1], reverse=True)]
    top_token_strs = [slm.decode([tok_id]) for tok_id in sorted_ids]
    first_char_class = classify_first_char(top_token_strs[0] if top_token_strs else "")
    passed = (margin < GLIMPROUTER_HINIT_MARGIN_THRESH) or (h_init > GLIMPROUTER_HINIT_ENTROPY_THRESH)
    return L0Result(
        passed=passed,
        h_init=h_init,
        margin=margin,
        top_logprobs=top_logprobs,
        top_token_strs=top_token_strs,
        first_char_class=first_char_class,
    )
