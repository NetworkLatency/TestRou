from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_DEPRECATED_CONTROLLER_KEYS = {
    "lambda0_cross",
    "q_handoff",
    "q_handoff_cross",
    "cross_prior_distribution",
    "cross_prior_distribution_path",
    "self_reentry_min_tokens",
    "self_reentry_max_tokens",
    "self_reentry_agg_quantile",
}


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
    think_token_budget: int = 8192
    answer_token_budget: int = 2048
    step_delimiters: list[str] = field(default_factory=lambda: ["\n\n"])
    final_answer_generator: str = "slm"
    force_close_think_text: str = "\n</think>\n\n"
    llm_repair_instruction: str = (
        "Repair the existing reasoning locally and produce a concise continuation "
        "that a smaller model can continue from. Do not restart the solution. "
        "Do not perform long self-reflection. End after one or two constructive "
        "mathematical steps."
    )

    def __post_init__(self) -> None:
        if self.think_token_budget < 1:
            raise ValueError("generation.think_token_budget must be >= 1")
        if self.answer_token_budget < 1:
            raise ValueError("generation.answer_token_budget must be >= 1")
        if self.final_answer_generator not in {"slm", "llm"}:
            raise ValueError("generation.final_answer_generator must be 'slm' or 'llm'")


@dataclass
class ControllerConfig:
    mode: str = "pdi_step_window_controller"
    initial_driver: str = "slm"
    t_min: int = 32
    lambda0: float = 3.0
    lambda0_self: float | None = None
    n_min: int = 3
    q_high: float = 0.90
    q_recover: float = 0.75
    transition_grace_windows: int = 2
    r_upper: int = 2
    eta_upper: float = 0.50
    r_handoff: int = 1
    m_probation: int = 2
    m_reentry: int = 3
    reentry_transition_grace: int = 2
    q_low: float = 0.10
    r_low: int = 2
    max_llm_repair_steps: int = 5
    handoff_strategy: str = "self_reentry_certification"
    self_reentry_min_scored_tokens: int | None = None
    self_reentry_max_attempt_steps: int = 3
    self_reentry_q_threshold: float = 0.80
    commit_self_reentry_step: bool = True
    prior_distribution: list[float] = field(default_factory=lambda: [0.12, 0.19, 0.29, 0.36])
    prior_distribution_path: str | None = None
    self_prior_distribution: list[float] | None = None
    self_prior_distribution_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"pdi_step_window_controller", "ownership_controller"}:
            raise ValueError("controller.mode must be 'pdi_step_window_controller'")
        if self.initial_driver != "slm":
            raise ValueError("controller.initial_driver must be 'slm'")
        if self.t_min < 1:
            raise ValueError("controller.t_min must be >= 1")
        if self.lambda0 < 0:
            raise ValueError("controller.lambda0 must be >= 0")
        if self.lambda0_self is not None and self.lambda0_self < 0:
            raise ValueError("controller.lambda0_self must be >= 0")
        if self.n_min < 1:
            raise ValueError("controller.n_min must be >= 1")
        for name in ("q_high", "q_recover", "q_low"):
            value = getattr(self, name)
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"controller.{name} must be in (0, 1)")
        if self.q_recover >= self.q_high:
            raise ValueError("controller.q_recover must be lower than controller.q_high")
        if self.transition_grace_windows < 1:
            raise ValueError("controller.transition_grace_windows must be >= 1")
        if self.r_upper < 1:
            raise ValueError("controller.r_upper must be >= 1")
        if self.r_handoff < 1:
            raise ValueError("controller.r_handoff must be >= 1")
        if self.m_probation < 1:
            raise ValueError("controller.m_probation must be >= 1")
        if self.m_reentry < 1:
            raise ValueError("controller.m_reentry must be >= 1")
        if self.reentry_transition_grace < 1:
            raise ValueError("controller.reentry_transition_grace must be >= 1")
        if self.r_low < 1:
            raise ValueError("controller.r_low must be >= 1")
        if self.max_llm_repair_steps < 1:
            raise ValueError("controller.max_llm_repair_steps must be >= 1")
        if self.handoff_strategy not in {"repair_landing_index", "self_reentry_certification"}:
            raise ValueError("controller.handoff_strategy must be 'repair_landing_index' or 'self_reentry_certification'")
        if self.self_reentry_min_scored_tokens is not None and self.self_reentry_min_scored_tokens < 1:
            raise ValueError("controller.self_reentry_min_scored_tokens must be >= 1")
        if self.self_reentry_max_attempt_steps < 1:
            raise ValueError("controller.self_reentry_max_attempt_steps must be >= 1")
        if not 0.0 < self.self_reentry_q_threshold < 1.0:
            raise ValueError("controller.self_reentry_q_threshold must be in (0, 1)")


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
    method: str = "pdi_step_window_controller_v0"
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
                if key == "controller":
                    kwargs[key] = {
                        item_key: item_value
                        for item_key, item_value in kwargs[key].items()
                        if item_key not in _DEPRECATED_CONTROLLER_KEYS
                    }
                kwargs[key] = cls_(**kwargs[key])
        cfg = cls(**kwargs)
        if cfg.slm.backend != "transformers":
            raise ValueError("SARR-CoDE requires slm.backend='transformers' for local logits.")
        if cfg.llm.backend == "transformers" and not cfg.llm.model_path:
            raise ValueError("llm.model_path is required when llm.backend='transformers'.")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
