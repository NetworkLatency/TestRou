from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelRuntimeConfig:
    model_path: str
    tokenizer_path: str | None = None
    chat_template_path: str | None = None
    device: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = True
    backend: str = "transformers"
    api_base_url: str | None = None
    api_key: str = "EMPTY"
    api_model: str | None = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in {"transformers", "openai", "vllm"}:
            raise ValueError(f"Unsupported backend: {self.backend!r}")


@dataclass
class GenerationConfig:
    max_new_tokens_per_step: int = 512
    min_new_tokens_per_step: int = 64
    think_token_budget: int = 8192
    answer_token_budget: int = 2048
    step_delimiters: list[str] = field(default_factory=lambda: ["\n\n"])
    final_answer_generator: str = "slm"
    force_close_think_text: str = "\n</think>\n\n"

    def __post_init__(self) -> None:
        if self.max_new_tokens_per_step < 1:
            raise ValueError("generation.max_new_tokens_per_step must be >= 1")
        if self.min_new_tokens_per_step < 1:
            raise ValueError("generation.min_new_tokens_per_step must be >= 1")
        if self.min_new_tokens_per_step > self.max_new_tokens_per_step:
            raise ValueError("generation.min_new_tokens_per_step must be <= max_new_tokens_per_step")
        if self.think_token_budget < 1:
            raise ValueError("generation.think_token_budget must be >= 1")
        if self.answer_token_budget < 1:
            raise ValueError("generation.answer_token_budget must be >= 1")
        if self.final_answer_generator not in {"slm", "llm"}:
            raise ValueError("generation.final_answer_generator must be 'slm' or 'llm'")


@dataclass
class ControllerConfig:
    mode: str = "ownership_controller"
    initial_driver: str = "slm"

    def __post_init__(self) -> None:
        if self.mode != "ownership_controller":
            raise ValueError("controller.mode must be 'ownership_controller'")
        if self.initial_driver != "slm":
            raise ValueError("controller.initial_driver must be 'slm'")


@dataclass
class ConfidenceConfig:
    """Top-k logits captured during SLM generation for local confidence signals."""

    top_k: int = 20
    capture_topk_entropy: bool = True
    capture_token_logprobs: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 2:
            raise ValueError("confidence.top_k must be >= 2")


@dataclass
class RiskConfig:
    enable_local_difficulty_routing: bool = False
    stable_reference_min_steps: int = 3
    recent_window: int = 4
    prefix_recent_steps: int = 3
    handoff_probe_strategy: str = "eager"
    handoff_probe_interval: int = 1
    handoff_probe_warmup_steps: int = 2

    def __post_init__(self) -> None:
        if self.stable_reference_min_steps < 1:
            raise ValueError("risk.stable_reference_min_steps must be >= 1")
        if self.recent_window < 1:
            raise ValueError("risk.recent_window must be >= 1")
        if self.prefix_recent_steps < 1:
            raise ValueError("risk.prefix_recent_steps must be >= 1")
        if self.handoff_probe_strategy not in {"eager", "periodic", "hybrid"}:
            raise ValueError("risk.handoff_probe_strategy must be one of: eager, periodic, hybrid")
        if self.handoff_probe_interval < 1:
            raise ValueError("risk.handoff_probe_interval must be >= 1")
        if self.handoff_probe_warmup_steps < 0:
            raise ValueError("risk.handoff_probe_warmup_steps must be >= 0")


@dataclass
class LoggingConfig:
    save_step_records: bool = True
    save_driver_switch_events: bool = True
    save_transition_stats: bool = True


@dataclass
class RuntimeConfig:
    max_model_len: int = 16384
    reset_prefix_cache_after_problem: bool = True

    def __post_init__(self) -> None:
        if self.max_model_len < 1:
            raise ValueError("runtime.max_model_len must be >= 1")


@dataclass
class SARRConfig:
    method: str = "sarr_code_v5_ownership_controller"
    metadata: dict[str, Any] = field(default_factory=dict)
    slm: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(model_path=""))
    llm: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(model_path="", backend="openai"))
    output_dir: str = "sarr_results"
    dataset_paths: dict[str, str] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "SARRConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SARRConfig":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown SARRConfig keys: {unknown}")
        kwargs = dict(data)
        if isinstance(kwargs.get("slm"), dict):
            kwargs["slm"] = ModelRuntimeConfig(**kwargs["slm"])
        if isinstance(kwargs.get("llm"), dict):
            kwargs["llm"] = ModelRuntimeConfig(**kwargs["llm"])
        for key, cls_ in [
            ("generation", GenerationConfig),
            ("controller", ControllerConfig),
            ("confidence", ConfidenceConfig),
            ("risk", RiskConfig),
            ("logging", LoggingConfig),
            ("runtime", RuntimeConfig),
        ]:
            if isinstance(kwargs.get(key), dict):
                kwargs[key] = cls_(**kwargs[key])
        cfg = cls(**kwargs)
        if cfg.slm.backend != "transformers":
            raise ValueError("SARR-CoDE requires slm.backend='transformers' for local logits.")
        if cfg.llm.backend == "transformers":
            raise ValueError("Use llm.backend='openai' or 'vllm'. SLM is the local transformers model.")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
