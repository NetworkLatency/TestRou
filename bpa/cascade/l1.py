from __future__ import annotations

from bpa.config import BPAConfig
from bpa.engines import completion_logprobs, generated_token_ids, logprob_value
from bpa.render import render_for_continuation
from bpa.state import BranchCandidate, GenerationState, L0Result


def _actual_logprobs(output, token_ids: list[int]) -> list[float]:
    values: list[float] = []
    for idx, lp_map in enumerate(completion_logprobs(output)):
        if not lp_map:
            values.append(0.0)
            continue
        actual_id = token_ids[idx] if idx < len(token_ids) else None
        record = lp_map.get(actual_id) if actual_id is not None else None
        if record is None:
            record = next(iter(lp_map.values()))
        values.append(logprob_value(record))
    return values


def _token_cutoff_before_double_newline(raw_ids: list[int], idx: int, tokenizer) -> int:
    cutoff = 0
    for i in range(len(raw_ids)):
        piece = tokenizer.decode(
            raw_ids[: i + 1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if len(piece) > idx:
            return i
        cutoff = i + 1
    return cutoff


def build_branch(first_tok_id: int, first_tok_lp: float, vllm_out, tokenizer) -> BranchCandidate:
    continuation_ids = generated_token_ids(vllm_out)
    rollout_logprobs = _actual_logprobs(vllm_out, continuation_ids)
    raw_ids = [first_tok_id] + continuation_ids
    raw_rollout_text = tokenizer.decode(
        raw_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    idx = raw_rollout_text.find("\n\n")
    if idx >= 0:
        step_branch_text = raw_rollout_text[:idx]
        step_branch_was_truncated = True
        cutoff_tok_count = _token_cutoff_before_double_newline(raw_ids, idx, tokenizer)
        if cutoff_tok_count <= 0:
            sum_logprob_step = 0.0
        elif cutoff_tok_count == 1:
            sum_logprob_step = first_tok_lp
        else:
            sum_logprob_step = first_tok_lp + sum(rollout_logprobs[: cutoff_tok_count - 1])
    else:
        step_branch_text = raw_rollout_text
        step_branch_was_truncated = False
        cutoff_tok_count = None
        sum_logprob_step = first_tok_lp + sum(rollout_logprobs)

    return BranchCandidate(
        first_token_id=first_tok_id,
        first_token_str=tokenizer.decode(
            [first_tok_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        raw_rollout_text=raw_rollout_text,
        raw_rollout_token_ids=raw_ids,
        step_branch_text=step_branch_text,
        step_branch_was_truncated=step_branch_was_truncated,
        rollout_logprobs=rollout_logprobs,
        first_token_logprob=first_tok_lp,
        sum_logprob_raw=first_tok_lp + sum(rollout_logprobs),
        sum_logprob_step=sum_logprob_step,
        cutoff_tok_count=cutoff_tok_count,
    )


def l1_shadow_rollout(state: GenerationState, slm, config: BPAConfig, l0: L0Result) -> tuple[BranchCandidate, BranchCandidate]:
    sorted_tokens = sorted(l0.top_logprobs.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_tokens) < 2:
        raise ValueError("L1 requires at least two L0 candidate tokens.")
    (tok1, lp1), (tok2, lp2) = sorted_tokens[0], sorted_tokens[1]

    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, slm.ensure_tokenizer())
    rendered_ids = slm.encode(rendered)
    sampling = slm.sampling_params(max_tokens=config.rollout_length, temperature=0.0, logprobs=1)
    prompts = [
        slm.tokens_prompt(rendered_ids + [tok1]),
        slm.tokens_prompt(rendered_ids + [tok2]),
    ]
    outs = slm.generate(prompts, sampling)

    state.slm_decode_tokens += sum(len(generated_token_ids(out)) for out in outs)
    state.slm_prefill_tokens += len(rendered_ids) + 1

    tokenizer = slm.ensure_tokenizer()
    return build_branch(tok1, lp1, outs[0], tokenizer), build_branch(tok2, lp2, outs[1], tokenizer)
