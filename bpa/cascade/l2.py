from __future__ import annotations

from bpa.config import BPAConfig
from bpa.state import BranchCandidate, L2Result


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def grams(x: str) -> set[str]:
        if len(x) < n:
            return {x} if x else set()
        return {x[i : i + n] for i in range(len(x) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    union = ga | gb
    if not union:
        return 0.0
    return len(ga & gb) / len(union)


def _first_diverge_pos(a: list[int], b: list[int]) -> int:
    for idx, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return idx
    return min(len(a), len(b))


def count_step_tokens(branch: BranchCandidate) -> int:
    if not branch.step_branch_was_truncated:
        return max(len(branch.raw_rollout_token_ids), 1)
    return max(1, min(len(branch.raw_rollout_token_ids), len(branch.rollout_logprobs) + 1))


def l2_compute(b1: BranchCandidate, b2: BranchCandidate, config: BPAConfig) -> L2Result:
    n1_raw = max(len(b1.raw_rollout_token_ids), 1)
    n2_raw = max(len(b2.raw_rollout_token_ids), 1)
    avg_raw_1 = b1.sum_logprob_raw / n1_raw
    avg_raw_2 = b2.sum_logprob_raw / n2_raw
    delta_raw = abs(avg_raw_1 - avg_raw_2)

    n1_step = count_step_tokens(b1)
    n2_step = count_step_tokens(b2)
    avg_step_1 = b1.sum_logprob_step / n1_step
    avg_step_2 = b2.sum_logprob_step / n2_step
    delta_step = abs(avg_step_1 - avg_step_2)

    j_raw = char_ngram_jaccard(b1.raw_rollout_text, b2.raw_rollout_text, n=3)
    j_step = char_ngram_jaccard(b1.step_branch_text, b2.step_branch_text, n=3)
    diverge_pos = _first_diverge_pos(b1.raw_rollout_token_ids, b2.raw_rollout_token_ids)

    triggered = False
    reason = "no_trigger"
    if delta_raw < config.l2_divergence_thresh:
        triggered = True
        reason = "delta_raw_low"
    elif j_raw < config.l2_text_jaccard_thresh:
        triggered = True
        reason = "text_jaccard_low"

    return L2Result(
        avg_lp_raw_1=avg_raw_1,
        avg_lp_raw_2=avg_raw_2,
        delta_avg_lp_raw=delta_raw,
        avg_lp_step_1=avg_step_1,
        avg_lp_step_2=avg_step_2,
        delta_avg_lp_step=delta_step,
        text_jaccard_3gram_raw=j_raw,
        text_jaccard_3gram_step=j_step,
        branches_diverged_at_token=diverge_pos,
        triggered_arbitration=triggered,
        trigger_reason=reason,
    )
