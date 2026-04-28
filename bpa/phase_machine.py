from __future__ import annotations

from .state import GenerationState, Phase, TraceEvent

CLOSE_THINK_TAG = "</think>"


def detect_close_think(prev_assistant_prefix: str, newly_added_text: str) -> tuple[bool, int]:
    combined = prev_assistant_prefix + newly_added_text
    start = max(0, len(prev_assistant_prefix) - len(CLOSE_THINK_TAG))
    idx = combined.find(CLOSE_THINK_TAG, start)
    if idx < 0:
        return False, -1
    rel_idx = idx - len(prev_assistant_prefix)
    return True, rel_idx


def check_and_transition_phase(state: GenerationState, newly_added_text: str) -> None:
    if state.phase != Phase.THINKING:
        return
    if not newly_added_text:
        return
    prev_prefix = state.assistant_prefix_text[: -len(newly_added_text)]
    found, rel_idx = detect_close_think(prev_prefix, newly_added_text)
    if not found:
        return

    state.has_seen_close_think = True
    close_end = len(prev_prefix) + rel_idx + len(CLOSE_THINK_TAG)
    discarded_tail = state.assistant_prefix_text[close_end:]
    state.assistant_prefix_text = state.assistant_prefix_text[:close_end]

    if discarded_tail:
        state.trace.append(
            TraceEvent(
                state.step_count,
                "phase_to_final_answer_discard_tail",
                {"discarded_chars": len(discarded_tail), "discarded_preview": discarded_tail[:200]},
            )
        )
    else:
        state.trace.append(TraceEvent(state.step_count, "phase_to_final_answer", {}))
    state.phase = Phase.FINAL_ANSWER
