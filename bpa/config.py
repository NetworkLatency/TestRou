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

    max_step_tokens: int = 1024
    max_total_tokens: int = 14336

    # Runtime switches used by baselines and diagnostic runs.
    reset_prefix_cache_after_problem: bool = True

    def __post_init__(self) -> None:
        if self.max_total_tokens < 1:
            raise ValueError("max_total_tokens must be >= 1")
        if self.max_step_tokens < 1:
            raise ValueError("max_step_tokens must be >= 1")

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
