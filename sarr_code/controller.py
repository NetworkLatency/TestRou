from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

from bpa.safety import CLOSE_THINK_TAG, clean_latex_answer, extract_last_boxed, normalize_step_skeleton

from .config import RiskConfig
from .records import StepOutput, StepRecord


SLM_ACTIVE = "SLM_ACTIVE"
LLM_FORWARD_OWNERSHIP = "LLM_FORWARD_OWNERSHIP"
LLM_REPAIR_OWNERSHIP = "LLM_REPAIR_OWNERSHIP"
HANDOFF_PROBE = "HANDOFF_PROBE"
CLOSE_OR_FINALIZE = "CLOSE_OR_FINALIZE"

ACTIVE = "active"
REMOVED = "removed"
SEALED = "sealed"
PROBE_DISCARDED = "probe_discarded"

SOURCE_SLM = "SLM"
SOURCE_LLM = "LLM"
SOURCE_SYSTEM = "SYSTEM"


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.M)
_MATH_EQ_RE = re.compile(r"(?:=|\\frac|\\sqrt|\\sum|\\prod|\\binom|\d+\s*[+\-*/^]\s*\d+)")
_CASE_SPLIT_RE = re.compile(r"\b(?:case|if|suppose|assume|otherwise|when)\b", re.I)
_CONSTRAINT_RE = re.compile(r"\b(?:must|given|constraint|condition|require|since|because)\b", re.I)
_REFLECTION_PATTERNS = [
    "wait",
    "let me check",
    "check again",
    "verify",
    "maybe",
    "but",
    "however",
    "reconsider",
    "let's see",
    "alternatively",
]
_VERIFICATION_PATTERNS = [
    "check",
    "verify",
    "again",
    "confirm",
    "make sure",
    "wait",
    "maybe",
]
_ANSWER_PATTERNS = [
    re.compile(
        r"(?is)\b(?:final\s+answer|the\s+answer|answer|result)\b\s*(?:is|=|:)?\s*([^\n.;]{1,100})"
    ),
    re.compile(r"(?is)\b(?:therefore|thus|so)\b\s+([^\n.;]{1,100})"),
]


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(str(text or "").lower())


def _ngram_counter(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"</?think>", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", "#", text)
    text = re.sub(r"[^a-z0-9#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_candidate(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = clean_latex_answer(str(value).strip())
    if not candidate:
        return None
    candidate = candidate.strip(" \t\r\n\"'`")
    candidate = re.sub(r"^(?:is|=|:)\s*", "", candidate, flags=re.I).strip()
    candidate = re.split(r"[\r\n]", candidate, maxsplit=1)[0].strip()
    candidate = candidate.strip(" ,;:.")
    if not candidate:
        return None
    return candidate[:100]


def extract_candidate_answer(text: str) -> str | None:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return _normalize_candidate(boxed)

    for pattern in _ANSWER_PATTERNS:
        matches = list(pattern.finditer(text or ""))
        for match in reversed(matches):
            candidate = _short_candidate(match.group(1))
            if candidate is not None:
                return candidate
    return None


def _short_candidate(raw: str) -> str | None:
    text = str(raw or "").strip()
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return _normalize_candidate(boxed)
    math_match = re.search(
        r"(\\frac\s*\{[^{}\r\n]{1,80}\}\s*\{[^{}\r\n]{1,80}\}|[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?|\b[ABCD]\b)",
        text,
        flags=re.I,
    )
    if math_match:
        return _normalize_candidate(math_match.group(1))
    expression = re.split(r"(?<!\d)\.(?!\d)|[,;]", text, maxsplit=1)[0].strip()
    if re.search(r"\d|\\frac|\\sqrt|[=+\-*/^]", expression):
        return _normalize_candidate(expression)
    return None


def _candidate_mention_count(text: str, candidate: str | None) -> int:
    if not candidate:
        return 0
    raw = str(text or "")
    if not raw:
        return 0
    return raw.lower().count(candidate.lower())


def _contains_candidate(text: str, candidate: str | None) -> bool:
    return _candidate_mention_count(text, candidate) > 0


def _sentence_fingerprints(text: str) -> list[str]:
    values = []
    for match in _SENTENCE_RE.finditer(text or ""):
        sent = _normalize_text(match.group(0))
        if len(sent) >= 12:
            values.append(sent)
    return values


def _jaccard_words(a: str, b: str) -> float:
    wa = set(_word_tokens(a))
    wb = set(_word_tokens(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclass
class StepSignals:
    raw_next_token_confidence: float | None = None
    entropy: float | None = None
    margin: float | None = None
    repeated_ngram_ratio: float = 0.0
    repeated_sentence_count: int = 0
    repeated_phrase_count: int = 0
    repeated_verification_pattern_count: int = 0
    repeated_answer_mention_count: int = 0
    low_new_information_score: float = 0.0
    reflection_pattern_count: int = 0
    has_new_equation: bool = False
    has_new_case_split: bool = False
    has_new_constraint: bool = False
    has_candidate_answer: bool = False
    candidate_answer_value: str | None = None
    repeats_existing_candidate_answer: bool = False
    degeneration_score: float = 0.0
    has_progress: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    triggered: bool = False
    reason: str = ""
    onset_step_id: int | None = None
    loop_type: str | None = None
    candidate_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateHistory:
    value: str
    first_seen_step: int
    mention_count: int = 0
    last_seen_step: int = 0
    step_ids: list[int] = field(default_factory=list)
    surrounding_verification_text: list[str] = field(default_factory=list)

    @property
    def distinct_step_count(self) -> int:
        return len(set(self.step_ids))

    def snapshot(self) -> dict[str, Any]:
        return {
            "candidate": self.value,
            "first_seen_step": self.first_seen_step,
            "mention_count": self.mention_count,
            "last_seen_step": self.last_seen_step,
            "distinct_step_count": self.distinct_step_count,
        }


@dataclass
class SealedInterval:
    anchor_step_id: int
    removed_start_step_id: int
    removed_end_step_id: int
    reason: str
    repair_horizon: int
    removed_signatures: list[str] = field(default_factory=list)

    def overlaps(self, start_step_id: int, end_step_id: int) -> bool:
        return not (end_step_id < self.removed_start_step_id or start_step_id > self.removed_end_step_id)


class TrajectoryState:
    def __init__(self, problem_id: str) -> None:
        self.problem_id = problem_id
        self.records: list[StepRecord] = []
        self.next_step_id = 1
        self.sealed_intervals: list[SealedInterval] = []

    def append_active_step(
        self,
        *,
        output: StepOutput,
        source: str,
        driver_state: str,
        observed_signals: StepSignals,
        action: str,
        attempt_id: int,
        extra: dict[str, Any] | None = None,
    ) -> StepRecord:
        record = StepRecord(
            problem_id=self.problem_id,
            step_id=self.next_step_id,
            text=output.text,
            token_ids=list(output.token_ids),
            source=source,
            status=ACTIVE,
            driver_state_when_generated=driver_state,
            observed_signals=observed_signals.to_dict(),
            created_at=time.time(),
            action=action,
            finish_reason=output.finish_reason,
            prompt_tokens=output.prompt_tokens,
            wall_time=output.wall_time,
            attempt_id=attempt_id,
            extra=dict(extra or {}),
        )
        self.records.append(record)
        self.next_step_id += 1
        return record

    def append_probe_discarded(
        self,
        *,
        output: StepOutput,
        source: str,
        driver_state: str,
        observed_signals: StepSignals,
        action: str,
        attempt_id: int,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> StepRecord:
        payload = dict(extra or {})
        payload["discard_reason"] = reason
        record = StepRecord(
            problem_id=self.problem_id,
            step_id=self.next_step_id,
            text=output.text,
            token_ids=list(output.token_ids),
            source=source,
            status=PROBE_DISCARDED,
            driver_state_when_generated=driver_state,
            observed_signals=observed_signals.to_dict(),
            created_at=time.time(),
            action=action,
            finish_reason=output.finish_reason,
            prompt_tokens=output.prompt_tokens,
            wall_time=output.wall_time,
            attempt_id=attempt_id,
            extra=payload,
        )
        self.records.append(record)
        self.next_step_id += 1
        return record

    def rollback_to_step(self, anchor_step_id: int) -> list[StepRecord]:
        removed: list[StepRecord] = []
        for record in self.records:
            if record.status == ACTIVE and record.step_id > anchor_step_id:
                record.status = REMOVED
                record.action = "REMOVED_BY_PREFIX_CONTAMINATION_ROLLBACK"
                removed.append(record)
        return removed

    def seal_interval(
        self,
        *,
        anchor_step_id: int,
        removed_start_step_id: int,
        removed_end_step_id: int,
        reason: str,
        repair_horizon: int,
        removed_steps: list[StepRecord],
    ) -> SealedInterval:
        for record in removed_steps:
            if removed_start_step_id <= record.step_id <= removed_end_step_id:
                record.status = SEALED
        interval = SealedInterval(
            anchor_step_id=anchor_step_id,
            removed_start_step_id=removed_start_step_id,
            removed_end_step_id=removed_end_step_id,
            reason=reason,
            repair_horizon=repair_horizon,
            removed_signatures=[normalize_step_skeleton(record.text) for record in removed_steps if record.text.strip()],
        )
        self.sealed_intervals.append(interval)
        return interval

    def get_active_prefix(self) -> str:
        return "".join(record.text for record in self.records if record.status == ACTIVE)

    def get_recent_steps(self, k: int, *, source: str | None = None, active_only: bool = True) -> list[StepRecord]:
        rows = [
            record
            for record in self.records
            if (not active_only or record.status == ACTIVE) and (source is None or record.source == source)
        ]
        return rows[-k:]

    def get_active_steps_between(self, a: int, b: int) -> list[StepRecord]:
        return [record for record in self.records if record.status == ACTIVE and a <= record.step_id <= b]

    def count_active_steps_between(self, a: int, b: int) -> int:
        return len(self.get_active_steps_between(a, b))

    def active_steps(self) -> list[StepRecord]:
        return [record for record in self.records if record.status == ACTIVE]

    def detect_finish_marker(self) -> bool:
        return CLOSE_THINK_TAG in self.get_active_prefix()

    def visible_token_count(self) -> int:
        return sum(record.token_count for record in self.records if record.status == ACTIVE)

    def source_token_count(self, source: str) -> int:
        return sum(record.token_count for record in self.records if record.status == ACTIVE and record.source == source)

    def source_step_count(self, source: str) -> int:
        return sum(1 for record in self.records if record.status == ACTIVE and record.source == source)


class ObservableSignals:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def compute(
        self,
        *,
        text: str,
        source: str,
        output_extra: dict[str, Any] | None,
        trajectory: TrajectoryState,
        known_candidates: list[str],
    ) -> StepSignals:
        extra = output_extra or {}
        confidence = extra.get("confidence") if isinstance(extra.get("confidence"), dict) else {}
        recent = trajectory.get_recent_steps(self.cfg.recent_window)
        recent_text = "\n".join(record.text for record in recent)
        tokens = _word_tokens(text)
        recent_tokens = _word_tokens(recent_text)

        ngram_ratio = self._repeated_ngram_ratio(tokens, recent_tokens)
        repeated_sentences = self._repeated_sentence_count(text, recent)
        repeated_phrases = self._repeated_phrase_count(tokens, recent_tokens)
        verification_count = self._pattern_count(text, _VERIFICATION_PATTERNS)
        reflection_count = self._pattern_count(text, _REFLECTION_PATTERNS)
        candidate = extract_candidate_answer(text)
        repeated_answer_count = self._known_candidate_mentions(text, candidate, known_candidates, recent)
        low_new_info = self._low_new_information_score(tokens, recent_tokens, text, recent_text)

        has_new_equation = bool(_MATH_EQ_RE.search(text or ""))
        has_new_case_split = bool(_CASE_SPLIT_RE.search(text or ""))
        has_new_constraint = bool(_CONSTRAINT_RE.search(text or ""))
        repeats_candidate = bool(candidate and any(_contains_candidate(record.text, candidate) for record in recent))
        has_progress = (
            has_new_equation
            or has_new_case_split
            or has_new_constraint
            or (candidate is not None and not repeats_candidate)
        )

        degeneration_score = self._degeneration_score(
            ngram_ratio=ngram_ratio,
            repeated_sentences=repeated_sentences,
            repeated_phrases=repeated_phrases,
            verification_count=verification_count,
            repeated_answer_count=repeated_answer_count,
            low_new_info=low_new_info,
            reflection_count=reflection_count,
            has_progress=has_progress,
        )

        return StepSignals(
            raw_next_token_confidence=_as_float(confidence.get("raw_next_token_confidence")),
            entropy=_as_float(confidence.get("entropy")),
            margin=_as_float(confidence.get("margin")),
            repeated_ngram_ratio=ngram_ratio,
            repeated_sentence_count=repeated_sentences,
            repeated_phrase_count=repeated_phrases,
            repeated_verification_pattern_count=verification_count,
            repeated_answer_mention_count=repeated_answer_count,
            low_new_information_score=low_new_info,
            reflection_pattern_count=reflection_count,
            has_new_equation=has_new_equation,
            has_new_case_split=has_new_case_split,
            has_new_constraint=has_new_constraint,
            has_candidate_answer=candidate is not None,
            candidate_answer_value=candidate,
            repeats_existing_candidate_answer=repeats_candidate,
            degeneration_score=degeneration_score,
            has_progress=has_progress,
        )

    def _repeated_ngram_ratio(self, tokens: list[str], recent_tokens: list[str], n: int = 4) -> float:
        grams = _ngram_counter(tokens, n)
        if not grams:
            return 0.0
        repeated_inside = sum(count - 1 for count in grams.values() if count > 1)
        recent_grams = set(_ngram_counter(recent_tokens, n))
        overlap = sum(count for gram, count in grams.items() if gram in recent_grams)
        return min(1.0, (repeated_inside + overlap) / max(1, sum(grams.values())))

    def _repeated_sentence_count(self, text: str, recent: list[StepRecord]) -> int:
        current = _sentence_fingerprints(text)
        if not current:
            return 0
        recent_sentences = set()
        for record in recent:
            recent_sentences.update(_sentence_fingerprints(record.text))
        inside = len(current) - len(set(current))
        overlap = sum(1 for sent in current if sent in recent_sentences)
        return inside + overlap

    def _repeated_phrase_count(self, tokens: list[str], recent_tokens: list[str], n: int = 6) -> int:
        grams = _ngram_counter(tokens, n)
        if not grams:
            return 0
        recent_grams = set(_ngram_counter(recent_tokens, n))
        return sum(1 for gram in grams if gram in recent_grams)

    def _pattern_count(self, text: str, patterns: list[str]) -> int:
        lowered = str(text or "").lower()
        return sum(lowered.count(pattern) for pattern in patterns)

    def _known_candidate_mentions(
        self,
        text: str,
        candidate: str | None,
        known_candidates: list[str],
        recent: list[StepRecord],
    ) -> int:
        candidates = [c for c in known_candidates if c]
        if candidate:
            candidates.append(candidate)
        seen = set()
        count = 0
        for value in candidates:
            norm = value.lower()
            if norm in seen:
                continue
            seen.add(norm)
            count += _candidate_mention_count(text, value)
            count += sum(1 for record in recent if _contains_candidate(record.text, value))
        return count

    def _low_new_information_score(
        self,
        tokens: list[str],
        recent_tokens: list[str],
        text: str,
        recent_text: str,
    ) -> float:
        if not tokens:
            return 1.0
        if not recent_tokens:
            return 0.0
        recent_set = set(recent_tokens)
        new_count = sum(1 for token in tokens if token not in recent_set)
        new_ratio = new_count / max(1, len(tokens))
        jaccard = _jaccard_words(text, recent_text)
        return max(0.0, min(1.0, 0.55 * (1.0 - new_ratio) + 0.45 * jaccard))

    def _degeneration_score(
        self,
        *,
        ngram_ratio: float,
        repeated_sentences: int,
        repeated_phrases: int,
        verification_count: int,
        repeated_answer_count: int,
        low_new_info: float,
        reflection_count: int,
        has_progress: bool,
    ) -> float:
        score = 0.0
        score += 0.24 * ngram_ratio
        score += 0.14 * min(1.0, repeated_sentences / 2)
        score += 0.12 * min(1.0, repeated_phrases / 3)
        score += 0.14 * min(1.0, verification_count / 3)
        score += 0.14 * min(1.0, reflection_count / 3)
        score += 0.12 * min(1.0, repeated_answer_count / 3)
        score += 0.18 * low_new_info
        if has_progress:
            score -= 0.18
        return max(0.0, min(1.0, score))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class StableStepMemory:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self._records: list[tuple[int, StepSignals]] = []

    def add_if_stable(self, record: StepRecord, signals: StepSignals) -> bool:
        if record.source != SOURCE_SLM or record.status != ACTIVE:
            return False
        if signals.degeneration_score >= self.cfg.degeneration_score_threshold:
            return False
        if signals.repeated_verification_pattern_count >= 3 and not signals.has_progress:
            return False
        if signals.repeats_existing_candidate_answer and signals.low_new_information_score >= 0.55:
            return False
        if not signals.has_progress and signals.low_new_information_score >= self.cfg.low_new_information_threshold:
            return False
        self._records.append((record.step_id, signals))
        return True

    def remove_after(self, anchor_step_id: int) -> None:
        self._records = [(step_id, signals) for step_id, signals in self._records if step_id <= anchor_step_id]

    def has_enough_reference(self) -> bool:
        return len(self._records) >= self.cfg.stable_reference_min_steps

    def get_reference_distribution(self) -> dict[str, float]:
        if not self._records:
            return {}

        def values(name: str) -> list[float]:
            out = []
            for _, signals in self._records:
                value = getattr(signals, name)
                if value is not None:
                    out.append(float(value))
            return out

        distribution = {}
        for name in ["entropy", "margin", "degeneration_score", "low_new_information_score"]:
            vals = values(name)
            if vals:
                distribution[name] = float(median(vals))
        return distribution

    def compare_to_stable(self, signals: StepSignals) -> dict[str, Any]:
        ref = self.get_reference_distribution()
        entropy_worse = False
        margin_worse = False
        degeneration_worse = False
        low_info_worse = False

        if self.has_enough_reference():
            if signals.entropy is not None and "entropy" in ref:
                entropy_worse = signals.entropy > ref["entropy"] + self.cfg.local_entropy_delta
            if signals.margin is not None and "margin" in ref:
                margin_worse = signals.margin < ref["margin"] * self.cfg.local_margin_ratio
            if "degeneration_score" in ref:
                degeneration_worse = signals.degeneration_score > max(
                    self.cfg.degeneration_score_threshold,
                    ref["degeneration_score"] + 0.22,
                )
            if "low_new_information_score" in ref:
                low_info_worse = signals.low_new_information_score > max(
                    self.cfg.low_new_information_threshold,
                    ref["low_new_information_score"] + 0.25,
                )
        else:
            degeneration_worse = signals.degeneration_score >= self.cfg.degeneration_score_threshold
            low_info_worse = signals.low_new_information_score >= self.cfg.low_new_information_threshold

        risk_rank = sum([entropy_worse, margin_worse, degeneration_worse, low_info_worse])
        return {
            "reference": ref,
            "entropy_worse": entropy_worse,
            "margin_worse": margin_worse,
            "degeneration_worse": degeneration_worse,
            "low_info_worse": low_info_worse,
            "risk_rank": risk_rank,
        }

    def latest_anchor_before(self, step_id: int) -> int | None:
        anchors = [anchor_id for anchor_id, _ in self._records if anchor_id < step_id]
        return max(anchors) if anchors else None


class AnswerStabilityDetector:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.histories: dict[str, CandidateHistory] = {}
        self.stable_candidate: str | None = None
        self.trigger_count = 0
        self.last_reason: str = ""

    def known_candidates(self) -> list[str]:
        return list(self.histories)

    def update(self, record: StepRecord, signals: StepSignals, trajectory: TrajectoryState) -> DetectionResult:
        current_candidate = signals.candidate_answer_value
        candidates_to_update: list[str] = []
        if current_candidate:
            candidates_to_update.append(current_candidate)
        for candidate in self.histories:
            if _contains_candidate(record.text, candidate):
                candidates_to_update.append(candidate)

        for candidate in dict.fromkeys(candidates_to_update):
            mentions = max(1, _candidate_mention_count(record.text, candidate))
            history = self.histories.get(candidate)
            if history is None:
                history = CandidateHistory(
                    value=candidate,
                    first_seen_step=record.step_id,
                    last_seen_step=record.step_id,
                )
                self.histories[candidate] = history
            history.mention_count += mentions
            history.last_seen_step = record.step_id
            history.step_ids.append(record.step_id)
            if signals.repeated_verification_pattern_count or signals.reflection_pattern_count:
                history.surrounding_verification_text.append(record.text[:240])

        result = self._detect(trajectory)
        if result.triggered:
            self.stable_candidate = result.candidate_answer
            self.trigger_count += 1
            self.last_reason = result.reason
        return result

    def _detect(self, trajectory: TrajectoryState) -> DetectionResult:
        if not self.histories:
            return DetectionResult()
        recent = trajectory.get_recent_steps(self.cfg.answer_stability_recent_window)
        if not recent:
            return DetectionResult()
        recent_start = recent[0].step_id
        active_candidates = [
            history
            for history in self.histories.values()
            if history.mention_count >= self.cfg.answer_stability_min_mentions
            and history.distinct_step_count >= 2
        ]
        if not active_candidates:
            return DetectionResult()
        active_candidates.sort(key=lambda h: (h.mention_count, h.last_seen_step), reverse=True)
        best = active_candidates[0]
        conflicts = [
            history
            for history in self.histories.values()
            if history.value != best.value and history.last_seen_step >= recent_start
        ]
        if conflicts:
            return DetectionResult()

        candidate_recent_mentions = sum(1 for record in recent if _contains_candidate(record.text, best.value))
        recent_signals = [record.observed_signals for record in recent if isinstance(record.observed_signals, dict)]
        verification_steps = sum(
            1
            for sig in recent_signals
            if int(sig.get("repeated_verification_pattern_count") or 0) > 0
            or int(sig.get("reflection_pattern_count") or 0) > 0
        )
        low_info_steps = sum(
            1
            for sig in recent_signals
            if float(sig.get("low_new_information_score") or 0.0) >= self.cfg.low_new_information_threshold * 0.75
        )
        progress_steps = sum(1 for sig in recent_signals if bool(sig.get("has_progress")))
        repeats = sum(1 for sig in recent_signals if bool(sig.get("repeats_existing_candidate_answer")))

        if candidate_recent_mentions >= 2 and (verification_steps >= 2 or low_info_steps >= 2 or repeats >= 1):
            if progress_steps <= max(1, len(recent) // 2):
                return DetectionResult(
                    triggered=True,
                    reason="candidate_answer_stable_but_thinking_continues",
                    onset_step_id=best.first_seen_step,
                    candidate_answer=best.value,
                )
        return DetectionResult()

    def snapshot(self) -> dict[str, Any]:
        return {
            "stable_candidate": self.stable_candidate,
            "candidates": [history.snapshot() for history in self.histories.values()],
        }


class RiskDetector:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def local_difficulty(
        self,
        record: StepRecord,
        signals: StepSignals,
        stable_memory: StableStepMemory,
    ) -> DetectionResult:
        if record.source != SOURCE_SLM:
            return DetectionResult()
        if not stable_memory.has_enough_reference():
            return DetectionResult()
        comparison = stable_memory.compare_to_stable(signals)
        if comparison["risk_rank"] >= 2 and signals.degeneration_score < self.cfg.degeneration_score_threshold:
            return DetectionResult(
                triggered=True,
                reason="slm_step_confidence_drift_without_prefix_loop",
                onset_step_id=record.step_id,
            )
        return DetectionResult()

    def prefix_contamination(
        self,
        trajectory: TrajectoryState,
        stable_memory: StableStepMemory,
    ) -> DetectionResult:
        recent = trajectory.get_recent_steps(self.cfg.prefix_recent_steps, source=SOURCE_SLM)
        if len(recent) < min(2, self.cfg.prefix_recent_steps):
            return DetectionResult()
        bad: list[StepRecord] = []
        low_progress_count = 0
        for record in recent:
            signals = _signals_from_record(record)
            comparison = stable_memory.compare_to_stable(signals)
            low_progress = (not signals.has_progress) or signals.low_new_information_score >= self.cfg.low_new_information_threshold
            if (
                comparison["risk_rank"] >= 2
                or signals.degeneration_score >= self.cfg.degeneration_score_threshold
                or (comparison["risk_rank"] >= 1 and low_progress)
            ):
                bad.append(record)
            if low_progress:
                low_progress_count += 1

        bad_ratio = len(bad) / max(1, len(recent))
        if bad_ratio >= self.cfg.prefix_bad_ratio and low_progress_count >= max(1, len(recent) // 2):
            return DetectionResult(
                triggered=True,
                reason="recent_slm_steps_drifted_from_stable_memory",
                onset_step_id=bad[0].step_id if bad else recent[0].step_id,
            )

        candidate_values = [
            str(_signals_from_record(record).candidate_answer_value)
            for record in recent
            if _signals_from_record(record).candidate_answer_value
        ]
        if candidate_values:
            most_common, count = Counter(candidate_values).most_common(1)[0]
            verification = sum(_signals_from_record(record).repeated_verification_pattern_count for record in recent)
            if count >= 2 and verification >= 2 and low_progress_count >= 2:
                return DetectionResult(
                    triggered=True,
                    reason="recent_slm_steps_repeated_candidate_verification",
                    onset_step_id=recent[0].step_id,
                    candidate_answer=most_common,
                )
        return DetectionResult()

    def degenerative_loop(
        self,
        trajectory: TrajectoryState,
        answer_detector: AnswerStabilityDetector,
    ) -> DetectionResult:
        recent = trajectory.get_recent_steps(self.cfg.recent_window)
        if len(recent) < 2:
            return DetectionResult()
        recent_signals = [_signals_from_record(record) for record in recent]
        stable_candidate = answer_detector.stable_candidate
        if stable_candidate:
            mentions = sum(1 for record in recent if _contains_candidate(record.text, stable_candidate))
            verification = sum(sig.repeated_verification_pattern_count + sig.reflection_pattern_count for sig in recent_signals)
            low_info = sum(1 for sig in recent_signals if sig.low_new_information_score >= self.cfg.low_new_information_threshold * 0.75)
            if mentions >= 2 and (verification >= 2 or low_info >= 2):
                return DetectionResult(
                    triggered=True,
                    reason="stable_candidate_repeated_in_verification_loop",
                    onset_step_id=recent[0].step_id,
                    loop_type="answer_verification_loop",
                    candidate_answer=stable_candidate,
                )

        skeletons = [normalize_step_skeleton(record.text) for record in recent if len(record.text.strip()) >= 20]
        if len(skeletons) >= 2 and (skeletons[-1] == skeletons[-2] or (len(skeletons) >= 3 and skeletons[-1] == skeletons[-3])):
            return DetectionResult(
                triggered=True,
                reason="repeated_reasoning_fragment",
                onset_step_id=recent[-2].step_id,
                loop_type="repeated_fragment",
            )

        reflection_total = sum(sig.reflection_pattern_count for sig in recent_signals)
        low_info_count = sum(1 for sig in recent_signals if sig.low_new_information_score >= self.cfg.low_new_information_threshold)
        repeated_template_count = sum(
            1
            for sig in recent_signals
            if sig.repeated_verification_pattern_count > 0 or sig.repeated_phrase_count > 0 or sig.repeated_sentence_count > 0
        )
        if reflection_total >= 3 and low_info_count >= 2 and repeated_template_count >= 2:
            return DetectionResult(
                triggered=True,
                reason="reflection_and_low_information_loop",
                onset_step_id=recent[0].step_id,
                loop_type="reflection_loop",
            )
        return DetectionResult()


def _signals_from_record(record: StepRecord) -> StepSignals:
    data = record.observed_signals if isinstance(record.observed_signals, dict) else {}
    valid_fields = StepSignals.__dataclass_fields__
    return StepSignals(**{key: data.get(key) for key in valid_fields if key in data})


class SealedIntervalLock:
    def __init__(self) -> None:
        self.intervals: list[SealedInterval] = []

    def add(self, interval: SealedInterval) -> None:
        self.intervals.append(interval)

    def blocks(self, onset_step_id: int, current_step_id: int) -> bool:
        return any(interval.overlaps(onset_step_id, current_step_id) for interval in self.intervals)

    def reopens_sealed_problem(self, text: str) -> bool:
        signature = normalize_step_skeleton(text)
        if not signature:
            return False
        for interval in self.intervals:
            for removed in interval.removed_signatures:
                if removed and _jaccard_words(signature, removed) >= 0.55:
                    return True
        return False


class HandoffReadiness:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        *,
        probe_text: str,
        probe_signals: StepSignals,
        stable_memory: StableStepMemory,
        sealed_lock: SealedIntervalLock,
    ) -> DetectionResult:
        comparison = stable_memory.compare_to_stable(probe_signals)
        if stable_memory.has_enough_reference() and comparison["risk_rank"] > self.cfg.handoff_max_risk_rank:
            return DetectionResult(triggered=False, reason="probe_worse_than_stable_memory")
        if probe_signals.degeneration_score >= self.cfg.degeneration_score_threshold:
            return DetectionResult(triggered=False, reason="probe_degenerated")
        if (
            probe_signals.low_new_information_score >= self.cfg.low_new_information_threshold
            and not probe_signals.has_progress
        ):
            return DetectionResult(triggered=False, reason="probe_low_information_without_progress")
        if probe_signals.reflection_pattern_count >= 3 and not probe_signals.has_progress:
            return DetectionResult(triggered=False, reason="probe_immediate_hesitation_loop")
        if sealed_lock.reopens_sealed_problem(probe_text):
            return DetectionResult(triggered=False, reason="probe_reopens_sealed_interval")
        return DetectionResult(triggered=True, reason="probe_continuation_stable")


class OwnershipController:
    def __init__(self, problem_id: str, cfg: RiskConfig, initial_driver: str = "slm") -> None:
        self.problem_id = problem_id
        self.cfg = cfg
        self.driver_state = SLM_ACTIVE if initial_driver.lower() == "slm" else initial_driver.upper()
        self.previous_ownership_state: str | None = None
        self.repair_horizon = 0
        self.repair_generated_steps = 0
        self.events: list[dict[str, Any]] = []
        self.transition_events: list[dict[str, Any]] = []

        self.driver_switch_count = 0
        self.llm_forward_episodes = 0
        self.llm_repair_episodes = 0
        self.handoff_probe_count = 0
        self.handoff_success_count = 0
        self.handoff_failure_count = 0
        self.local_difficulty_count = 0
        self.prefix_contamination_count = 0
        self.degenerative_loop_count = 0
        self.answer_stability_count = 0
        self.rollback_count = 0
        self.repeated_rollback_blocked_count = 0
        self.confidence_forward_count = 0
        self.handoff_probe_forward_count = 0
        self.lookahead_count = 0

    @property
    def llm_ownership_episodes(self) -> int:
        return self.llm_forward_episodes + self.llm_repair_episodes

    def switch(self, to_state: str, *, step_id: int | None, reason: str, data: dict[str, Any] | None = None) -> None:
        from_state = self.driver_state
        if from_state == to_state:
            return
        self.driver_state = to_state
        self.driver_switch_count += 1
        event = {
            "problem_id": self.problem_id,
            "event": "driver_switch",
            "step_id": step_id,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
            "data": dict(data or {}),
        }
        self.events.append(event)
        self.transition_events.append(event)
        if to_state == LLM_FORWARD_OWNERSHIP and from_state != HANDOFF_PROBE:
            self.llm_forward_episodes += 1
        elif to_state == LLM_REPAIR_OWNERSHIP and from_state != HANDOFF_PROBE:
            self.llm_repair_episodes += 1

    def note_event(self, event: str, *, step_id: int | None, reason: str, data: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "problem_id": self.problem_id,
                "event": event,
                "step_id": step_id,
                "reason": reason,
                "data": dict(data or {}),
            }
        )

    def start_repair(self, *, repair_horizon: int) -> None:
        self.repair_horizon = max(1, repair_horizon)
        self.repair_generated_steps = 0

    def note_repair_step(self) -> None:
        self.repair_generated_steps += 1

    def repair_horizon_satisfied(self) -> bool:
        return self.repair_generated_steps >= self.repair_horizon

    def summary(
        self,
        *,
        problem_id: str,
        finish_reason: str,
        final_answer: str | None,
        final_answer_generator: str | None,
        trajectory: TrajectoryState,
        total_wall_time: float,
        slm_wall_time: float,
        llm_wall_time: float,
        slm_prefill_count: int,
        llm_prefill_count: int,
    ) -> dict[str, Any]:
        slm_tokens = trajectory.source_token_count(SOURCE_SLM)
        llm_tokens = trajectory.source_token_count(SOURCE_LLM)
        return {
            "problem_id": problem_id,
            "finish_reason": finish_reason,
            "final_answer": final_answer,
            "final_answer_generator": final_answer_generator,
            "driver_state": self.driver_state,
            "driver_switch_count": self.driver_switch_count,
            "llm_ownership_episodes": self.llm_ownership_episodes,
            "llm_forward_episodes": self.llm_forward_episodes,
            "llm_repair_episodes": self.llm_repair_episodes,
            "handoff_probe_count": self.handoff_probe_count,
            "handoff_success_count": self.handoff_success_count,
            "handoff_failure_count": self.handoff_failure_count,
            "local_difficulty_count": self.local_difficulty_count,
            "prefix_contamination_count": self.prefix_contamination_count,
            "degenerative_loop_count": self.degenerative_loop_count,
            "answer_stability_count": self.answer_stability_count,
            "rollback_count": self.rollback_count,
            "sealed_interval_count": len(trajectory.sealed_intervals),
            "repeated_rollback_blocked_count": self.repeated_rollback_blocked_count,
            "slm_thinking_tokens": slm_tokens,
            "llm_thinking_tokens": llm_tokens,
            "total_thinking_tokens": slm_tokens + llm_tokens,
            "slm_step_count": trajectory.source_step_count(SOURCE_SLM),
            "llm_step_count": trajectory.source_step_count(SOURCE_LLM),
            "confidence_forward_count": self.confidence_forward_count,
            "handoff_probe_forward_count": self.handoff_probe_forward_count,
            "lookahead_count": self.lookahead_count,
            "slm_prefill_count": slm_prefill_count,
            "llm_prefill_count": llm_prefill_count,
            "total_wall_time": total_wall_time,
            "slm_wall_time": slm_wall_time,
            "llm_wall_time": llm_wall_time,
        }
