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
    max_new_tokens_per_step: int = 256
    think_token_budget: int = 8192
    answer_token_budget: int = 2048
    step_delimiters: list[str] = field(default_factory=lambda: ["\n\n"])
    final_answer_generator: str = "active"
    force_close_think_on_budget: bool = True
    force_close_think_text: str = (
        "\nWe are out of reliable reasoning budget. Stop reasoning now. "
        "Do not restart the solution after </think>. After </think>, give only the final answer "
        "using the strongest conclusion above; if uncertain, make the best concise guess.\n</think>\n\n"
    )
    close_tag_lookahead_tokens: int = 16

    def __post_init__(self) -> None:
        if self.max_new_tokens_per_step < 1:
            raise ValueError("generation.max_new_tokens_per_step must be >= 1")
        if self.think_token_budget < 1:
            raise ValueError("generation.think_token_budget must be >= 1")
        if self.answer_token_budget < 1:
            raise ValueError("generation.answer_token_budget must be >= 1")
        if self.close_tag_lookahead_tokens < 0:
            raise ValueError("generation.close_tag_lookahead_tokens must be >= 0")
        if self.final_answer_generator not in {"slm", "llm", "active"}:
            raise ValueError("generation.final_answer_generator must be 'slm', 'llm', or 'active'")


@dataclass
class ControllerConfig:
    mode: str = "ciod_driver_switching"
    initial_driver: str = "slm"

    def __post_init__(self) -> None:
        if self.mode != "ciod_driver_switching":
            raise ValueError("controller.mode must be 'ciod_driver_switching'")
        if self.initial_driver != "slm":
            raise ValueError("controller.initial_driver must be 'slm'")


@dataclass
class ConfidenceConfig:
    """Confidence observation config. score_type is metadata; top_k and smooth_window are active."""

    score_type: str = "normalized_topk_entropy"
    top_k: int = 20
    smooth_window: int = 3

    def __post_init__(self) -> None:
        if self.score_type != "normalized_topk_entropy":
            raise ValueError("confidence.score_type must be 'normalized_topk_entropy'")
        if self.top_k < 2:
            raise ValueError("confidence.top_k must be >= 2")
        if self.smooth_window < 1:
            raise ValueError("confidence.smooth_window must be >= 1")


@dataclass
class CIODConfig:
    """CI-OD (Confidence-based Intervention on Degeneration) driver switching config.

    Risk formula (v2):
        hazard = hazard_scale * (1 + masked_memory) * max(0, post_masked_exposure - exposure_e0)^2
        ciod_risk = 1 - exp(-hazard)

    State equations per SLM step:
        masked_uncertainty = (c_raw <= masked_low_threshold) AND (c_smooth > masked_low_threshold)
        masked_memory_t = masked_decay * masked_memory_{t-1} + 1[masked_uncertainty]
        exp_inc = 1 if (masked_memory > 0 AND c_smooth >= exposure_threshold) else 0
        post_masked_exposure_t = exposure_decay * post_masked_exposure_{t-1} + exp_inc
    """

    masked_low_threshold: float = 0.35
    exposure_threshold: float = 0.60
    masked_decay: float = 0.98
    exposure_decay: float = 0.98
    min_masked_memory: float = 3.0
    exposure_e0: float = 8.0
    hazard_scale: float = 0.005
    on_threshold: float = 0.10
    off_threshold: float = 0.03

    def __post_init__(self) -> None:
        for name in [
            "masked_low_threshold",
            "exposure_threshold",
            "masked_decay",
            "exposure_decay",
            "on_threshold",
            "off_threshold",
        ]:
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"ciod.{name} must be in [0, 1]")
        if self.min_masked_memory < 0.0:
            raise ValueError("ciod.min_masked_memory must be >= 0")
        if self.exposure_e0 < 0.0:
            raise ValueError("ciod.exposure_e0 must be >= 0")
        if self.hazard_scale < 0.0:
            raise ValueError("ciod.hazard_scale must be >= 0")
        if self.off_threshold > self.on_threshold:
            raise ValueError("ciod.off_threshold must be <= ciod.on_threshold")
        if self.exposure_threshold < self.masked_low_threshold:
            raise ValueError("ciod.exposure_threshold must be >= ciod.masked_low_threshold")


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
    method: str = "sarr_code_v4_ciod_driver"
    metadata: dict[str, Any] = field(default_factory=dict)
    slm: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(model_path=""))
    llm: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(model_path="", backend="openai"))
    output_dir: str = "sarr_results"
    dataset_paths: dict[str, str] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    ciod: CIODConfig = field(default_factory=CIODConfig)
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
            ("ciod", CIODConfig),
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
