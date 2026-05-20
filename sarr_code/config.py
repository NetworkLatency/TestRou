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
    step_delimiters: list[str] = field(default_factory=lambda: ["\n\n"])
    max_new_tokens_per_step: int = 256
    think_token_budget: int = 8192
    answer_token_budget: int = 2048
    final_answer_generator: str = "llm"
    force_close_think_on_budget: bool = True
    force_close_think_text: str = (
        "\nWe are out of reliable reasoning budget. Stop reasoning now. "
        "Do not restart the solution after </think>. After </think>, give only the final answer "
        "using the strongest conclusion above; if uncertain, make the best concise guess.\n</think>\n\n"
    )

    def __post_init__(self) -> None:
        if self.max_new_tokens_per_step < 1:
            raise ValueError("generation.max_new_tokens_per_step must be >= 1")
        if self.think_token_budget < 1:
            raise ValueError("generation.think_token_budget must be >= 1")
        if self.answer_token_budget < 1:
            raise ValueError("generation.answer_token_budget must be >= 1")
        if self.final_answer_generator not in {"slm", "llm"}:
            raise ValueError("generation.final_answer_generator must be 'slm' or 'llm'")


@dataclass
class ConfidenceConfig:
    topk_entropy: int = 20
    percentile_normalization: bool = True
    calibration_path: str | None = None
    allow_identity_normalizer: bool = False
    smooth_window: int = 2
    delta: float = 0.55
    capture_slm_token_entropy: bool = False

    def __post_init__(self) -> None:
        if self.topk_entropy < 2:
            raise ValueError("confidence.topk_entropy must be >= 2")
        if self.smooth_window < 1:
            raise ValueError("confidence.smooth_window must be >= 1")


@dataclass
class StartupConfig:
    B_min: int = 2
    B_max: int = 5
    tau_start: int = 1

    def __post_init__(self) -> None:
        if self.B_min < 1:
            raise ValueError("startup.B_min must be >= 1")
        if self.B_max < self.B_min:
            raise ValueError("startup.B_max must be >= startup.B_min")
        if self.tau_start < 1:
            raise ValueError("startup.tau_start must be >= 1")


@dataclass
class StableConfig:
    theta_s: float = 0.70
    tau_D: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta_s <= 1.0:
            raise ValueError("stable.theta_s must be in [0, 1]")
        if self.tau_D < 1:
            raise ValueError("stable.tau_D must be >= 1")


@dataclass
class RollbackConfig:
    M_max: int = 5
    recovery_max_policy: str = "m_plus_1"
    confidence_gated_recovery: bool = True
    force_slm_after_recovery: bool = True
    long_span_policy: str = "fallback_once_then_rollback"
    max_long_span_fallbacks_per_anchor: int = 1
    long_span_recovery_steps: int = 1
    anchor_repeat_backoff_after: int = 1
    anchor_repeat_backoff_steps: int = 1
    max_root_rollbacks: int = 2
    root_rollback_action: str = "force_close_think"

    def __post_init__(self) -> None:
        if self.M_max < 1:
            raise ValueError("rollback.M_max must be >= 1")
        if self.recovery_max_policy != "m_plus_1":
            raise ValueError("Only rollback.recovery_max_policy='m_plus_1' is implemented.")
        if self.long_span_policy not in {"fallback_no_delete", "rollback_to_anchor", "fallback_once_then_rollback"}:
            raise ValueError(
                "rollback.long_span_policy must be one of "
                "'fallback_no_delete', 'rollback_to_anchor', or 'fallback_once_then_rollback'."
            )
        if self.max_long_span_fallbacks_per_anchor < 0:
            raise ValueError("rollback.max_long_span_fallbacks_per_anchor must be >= 0")
        if self.long_span_recovery_steps < 1:
            raise ValueError("rollback.long_span_recovery_steps must be >= 1")
        if self.anchor_repeat_backoff_after < 1:
            raise ValueError("rollback.anchor_repeat_backoff_after must be >= 1")
        if self.anchor_repeat_backoff_steps < 0:
            raise ValueError("rollback.anchor_repeat_backoff_steps must be >= 0")
        if self.max_root_rollbacks < 0:
            raise ValueError("rollback.max_root_rollbacks must be >= 0")
        if self.root_rollback_action not in {"force_close_think", "allow"}:
            raise ValueError("rollback.root_rollback_action must be 'force_close_think' or 'allow'")


@dataclass
class LoggingConfig:
    save_step_records: bool = True
    save_rollback_events: bool = True
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
    method: str = "sarr_code_aggressive_prefix"
    metadata: dict[str, Any] = field(default_factory=dict)
    slm: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(model_path=""))
    llm: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(model_path="", backend="openai"))
    output_dir: str = "sarr_results"
    dataset_paths: dict[str, str] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    startup: StartupConfig = field(default_factory=StartupConfig)
    stable: StableConfig = field(default_factory=StableConfig)
    rollback: RollbackConfig = field(default_factory=RollbackConfig)
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
            ("confidence", ConfidenceConfig),
            ("startup", StartupConfig),
            ("stable", StableConfig),
            ("rollback", RollbackConfig),
            ("logging", LoggingConfig),
            ("runtime", RuntimeConfig),
        ]:
            if isinstance(kwargs.get(key), dict):
                kwargs[key] = cls_(**kwargs[key])
        cfg = cls(**kwargs)
        if cfg.slm.backend != "transformers":
            raise ValueError("SARR-CoDE requires slm.backend='transformers' for local logits diagnostics.")
        if cfg.llm.backend == "transformers":
            raise ValueError("Use llm.backend='openai' or 'vllm'. SLM is the local transformers model.")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
