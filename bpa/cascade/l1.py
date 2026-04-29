from __future__ import annotations

import time

from bpa.config import BPAConfig
from bpa.engines import completion_logprobs, finish_reason, generated_token_ids, logprob_value
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


def _tokenizer_eos_token_ids(tokenizer) -> set[int]:
    eos_ids: set[int] = set()

    def add_id(value) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            eos_ids.add(int(value))
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_id(item)
            return
        try:
            eos_ids.add(int(value))
        except (TypeError, ValueError):
            return

    add_id(getattr(tokenizer, "eos_token_id", None))
    add_id(getattr(tokenizer, "eos_token_ids", None))

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        for token in ("<｜end▁of▁sentence｜>", "<|endoftext|>", "<|im_end|>"):
            try:
                token_id = convert(token)
            except Exception:
                continue
            if unk_token_id is not None and token_id == unk_token_id:
                continue
            add_id(token_id)

    return eos_ids


def _first_eos_index(raw_ids: list[int], tokenizer) -> int | None:
    eos_ids = _tokenizer_eos_token_ids(tokenizer)
    if not eos_ids:
        return None
    for idx, token_id in enumerate(raw_ids):
        if token_id in eos_ids:
            return idx
    return None


def _decode_visible(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _sum_logprob_for_visible_tokens(first_tok_lp: float, rollout_logprobs: list[float], token_count: int) -> float:
    if token_count <= 0:
        return 0.0
    if token_count == 1:
        return first_tok_lp
    return first_tok_lp + sum(rollout_logprobs[: token_count - 1])


def build_branch(first_tok_id: int, first_tok_lp: float, vllm_out, tokenizer) -> BranchCandidate:
    continuation_ids = generated_token_ids(vllm_out)
    rollout_logprobs = _actual_logprobs(vllm_out, continuation_ids)
    raw_ids = [first_tok_id] + continuation_ids
    raw_rollout_text = tokenizer.decode(
        raw_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    eos_cutoff_tok_count = _first_eos_index(raw_ids, tokenizer)
    idx = raw_rollout_text.find("\n\n")
    newline_cutoff_tok_count = _token_cutoff_before_double_newline(raw_ids, idx, tokenizer) if idx >= 0 else None

    if newline_cutoff_tok_count is not None and (
        eos_cutoff_tok_count is None or newline_cutoff_tok_count <= eos_cutoff_tok_count
    ):
        step_branch_text = _decode_visible(tokenizer, raw_ids[:newline_cutoff_tok_count])
        step_branch_was_truncated = True
        cutoff_tok_count = newline_cutoff_tok_count
        ended_by_eos = False
        branch_finish_reason = "stop_in_branch"
        sum_logprob_step = _sum_logprob_for_visible_tokens(first_tok_lp, rollout_logprobs, cutoff_tok_count)
    elif eos_cutoff_tok_count is not None:
        step_branch_text = _decode_visible(tokenizer, raw_ids[:eos_cutoff_tok_count])
        step_branch_was_truncated = False
        cutoff_tok_count = eos_cutoff_tok_count
        ended_by_eos = True
        branch_finish_reason = "branch_eos"
        sum_logprob_step = _sum_logprob_for_visible_tokens(first_tok_lp, rollout_logprobs, cutoff_tok_count)
    else:
        step_branch_text = _decode_visible(tokenizer, raw_ids)
        step_branch_was_truncated = False
        cutoff_tok_count = None
        ended_by_eos = False
        branch_finish_reason = finish_reason(vllm_out)
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
        ended_by_eos=ended_by_eos,
        finish_reason=branch_finish_reason,
    )


def l1_shadow_rollout(state: GenerationState, slm, config: BPAConfig, l0: L0Result) -> tuple[BranchCandidate, BranchCandidate]:
    sorted_tokens = sorted(l0.top_logprobs.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_tokens) < 2:
        raise ValueError("L1 requires at least two L0 candidate tokens.")
    (tok1, lp1), (tok2, lp2) = sorted_tokens[0], sorted_tokens[1]

    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, slm.ensure_tokenizer())
    rendered_ids = slm.encode(rendered)
    sampling_kwargs = {"max_tokens": config.rollout_length, "temperature": 0.0, "logprobs": 1}
    eos_token_ids = sorted(_tokenizer_eos_token_ids(slm.ensure_tokenizer()))
    if eos_token_ids:
        sampling_kwargs["stop_token_ids"] = eos_token_ids
    sampling = slm.sampling_params(**sampling_kwargs)
    prompts = [
        slm.tokens_prompt(rendered_ids + [tok1]),
        slm.tokens_prompt(rendered_ids + [tok2]),
    ]
    generate_start = time.time()
    outs = slm.generate(prompts, sampling)
    state.slm_wall_time += time.time() - generate_start
    state.slm_generate_calls += 1

    state.slm_decode_tokens += sum(len(generated_token_ids(out)) for out in outs)
    state.slm_prefill_tokens += len(rendered_ids) + 1

    tokenizer = slm.ensure_tokenizer()
    return build_branch(tok1, lp1, outs[0], tokenizer), build_branch(tok2, lp2, outs[1], tokenizer)
