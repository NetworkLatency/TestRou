from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BPAConfig:
    # Model and output configuration.
    slm_model_path: str = "DeepSeek-R1-Distill-Qwen-1.5B"
    llm_model_path: str = "Qwen-32B"
    slm_tokenizer_path: str | None = None
    llm_tokenizer_path: str | None = None
    output_dir: str = "bpa_results"
    dataset_paths: dict[str, str] = field(default_factory=dict)
    system_prompt: str | None = None

    # vLLM engine kwargs. These are passed directly to vllm.LLM.
    slm_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    llm_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    slm_device: str | None = None  # e.g. "cuda:0"
    llm_device: str | None = None  # e.g. "cuda:1"
    trust_remote_code: bool = True
    max_model_len: int = 16384
    enable_prefix_caching: bool = True

    # BPA defaults from the implementation spec.
    l0_topk: int = 10
    l0_margin_thresh: float = 0.4
    l0_entropy_thresh: float = 0.5
    rollout_length: int = 16
    l2_divergence_thresh: float = 0.15
    l2_text_jaccard_thresh: float = 0.4
    prompt_logprobs_topk: int = 20
    prompt_logprobs_sweep: list[int] = field(default_factory=lambda: [1, 5, 20])
    llm_scoring_context_window: int = 0  # 0 means exact full-prefix scoring.
    score_missing_ratio_thresh: float = 0.2
    invalid_fallback: str = "skip"
    max_step_tokens: int = 1024
    max_total_tokens: int = 16384
    max_llm_interventions: int = 8
    final_answer_max_tokens: int = 1024
    final_answer_chunk_tokens: int = 128
    final_answer_mode: str = "routed"  # "routed" | "llm_chunked"
    max_final_steps: int = 64
    max_final_tokens: int = 2048
    final_answer_stability_repeats: int = 2
    repetition_ngram_size: int = 8
    repetition_ngram_threshold: int = 4
    slm_to_llm_flop_ratio: float = 0.05
    arbitration_tie_margin: float = 0.05

    # Runtime switches used by baselines and diagnostic runs.
    cascade_mode: str = "bpa"  # "bpa" | "hinit"
    apply_arbitration: bool = True
    collect_branch_logs: bool = True
    reset_prefix_cache_after_problem: bool = True

    def __post_init__(self) -> None:
        if self.final_answer_mode not in {"routed", "llm_chunked"}:
            raise ValueError("final_answer_mode must be one of: 'routed', 'llm_chunked'")
        if self.max_final_steps < 1:
            raise ValueError("max_final_steps must be >= 1")
        if self.max_final_tokens < 1:
            raise ValueError("max_final_tokens must be >= 1")
        if self.final_answer_stability_repeats < 1:
            raise ValueError("final_answer_stability_repeats must be >= 1")

    @classmethod
    def from_json(cls, path: str | Path) -> "BPAConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BPAConfig":
        valid = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown BPAConfig keys: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **updates: Any) -> "BPAConfig":
        data = self.to_dict()
        data.update(updates)
        return BPAConfig.from_dict(data)
