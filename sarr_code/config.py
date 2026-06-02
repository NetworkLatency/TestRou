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
    "q_high",
    "q_recover",
    "transition_grace_windows",
    "r_upper",
    "eta_upper",
    "r_handoff",
    "m_probation",
    "reentry_transition_grace",
    "q_low",
    "r_low",
    "handoff_strategy",
    "self_reentry_min_scored_tokens",
    "self_reentry_min_tokens",
    "self_reentry_max_attempt_steps",
    "self_reentry_max_tokens",
    "self_reentry_q_threshold",
    "self_reentry_agg_quantile",
    "commit_self_reentry_step",
    "msm_repair_handoff_decay_factor",
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
    slm_high_q: float = 0.82
    slm_recover_q: float = 0.62
    m_reentry: int = 3
    max_llm_repair_steps: int = 5
    msm_initial_posterior: dict[str, float] = field(default_factory=lambda: {
        "stable": 0.85,
        "transition-risk": 0.10,
        "llm-confirmed": 0.03,
        "reentry-ready": 0.02,
    })
    msm_transition_matrix: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "stable": {"stable": 0.86, "transition-risk": 0.14},
        "transition-risk": {"stable": 0.76, "llm-confirmed": 0.05, "reentry-ready": 0.19},
        "llm-confirmed": {"llm-confirmed": 0.72, "reentry-ready": 0.28},
        "reentry-ready": {"stable": 0.72, "llm-confirmed": 0.28},
    })
    msm_action_thresholds: dict[str, float] = field(default_factory=lambda: {
        "llm_repair": 0.45,
        "transition_watch": 0.40,
        "handoff_back": 0.65,
        "slm_continue": 0.45,
    })
    msm_emission_floor: float = 0.03
    msm_llm_beneficial_boost: float = 8.0
    msm_reentry_ready_boost: float = 8.0
    msm_stable_boost: float = 4.0
    msm_llm_repair_confirm_steps: int = 1
    msm_repair_min_steps_before_reentry: int = 2
    msm_repair_reentry_boost: float = 4.0
    msm_repair_stable_boost: float = 2.0
    delta_llm_beneficial_threshold: float = 0.15
    delta_reentry_threshold: float = -0.15
    llm_diagnostic_enabled: bool = True
    repeat_finalize_enabled: bool = True
    step_text_repeat_min_occurrences: int = 3
    pdi_repeat_window: int = 6
    msm_trend_alpha: float = 2.0
    msm_repair_handoff_q_threshold: float = 0.70
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
        for name in ("slm_high_q", "slm_recover_q"):
            value = getattr(self, name)
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"controller.{name} must be in (0, 1)")
        if self.slm_recover_q >= self.slm_high_q:
            raise ValueError("controller.slm_recover_q must be lower than controller.slm_high_q")
        if self.m_reentry < 1:
            raise ValueError("controller.m_reentry must be >= 1")
        if self.max_llm_repair_steps < 1:
            raise ValueError("controller.max_llm_repair_steps must be >= 1")
        msm_states = {"stable", "transition-risk", "llm-confirmed", "reentry-ready"}
        if set(self.msm_initial_posterior) != msm_states:
            raise ValueError("controller.msm_initial_posterior must contain exactly the MSM state keys")
        if any(float(value) < 0 for value in self.msm_initial_posterior.values()):
            raise ValueError("controller.msm_initial_posterior values must be non-negative")
        if sum(float(value) for value in self.msm_initial_posterior.values()) <= 0:
            raise ValueError("controller.msm_initial_posterior must have positive total mass")
        if set(self.msm_transition_matrix) != msm_states:
            raise ValueError("controller.msm_transition_matrix must contain exactly the MSM source state keys")
        for source, row in self.msm_transition_matrix.items():
            unknown_targets = set(row) - msm_states
            if unknown_targets:
                raise ValueError(f"controller.msm_transition_matrix[{source!r}] has unknown targets: {sorted(unknown_targets)}")
            if any(float(value) < 0 for value in row.values()):
                raise ValueError(f"controller.msm_transition_matrix[{source!r}] values must be non-negative")
            if sum(float(value) for value in row.values()) <= 0:
                raise ValueError(f"controller.msm_transition_matrix[{source!r}] must have positive total mass")
        required_actions = {"llm_repair", "transition_watch", "handoff_back", "slm_continue"}
        if set(self.msm_action_thresholds) != required_actions:
            raise ValueError("controller.msm_action_thresholds must contain llm_repair, transition_watch, handoff_back, slm_continue")
        if self.msm_repair_min_steps_before_reentry < 1:
            raise ValueError("controller.msm_repair_min_steps_before_reentry must be >= 1")
        if self.step_text_repeat_min_occurrences < 2:
            raise ValueError("controller.step_text_repeat_min_occurrences must be >= 2")
        for name, value in self.msm_action_thresholds.items():
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"controller.msm_action_thresholds[{name!r}] must be in (0, 1)")
        for name in (
            "msm_emission_floor",
            "msm_llm_beneficial_boost",
            "msm_reentry_ready_boost",
            "msm_stable_boost",
            "msm_repair_reentry_boost",
            "msm_repair_stable_boost",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"controller.{name} must be > 0")
        if self.delta_llm_beneficial_threshold < 0:
            raise ValueError("controller.delta_llm_beneficial_threshold must be >= 0")
        if self.delta_reentry_threshold > 0:
            raise ValueError("controller.delta_reentry_threshold must be <= 0")
        if self.msm_trend_alpha < 0:
            raise ValueError("controller.msm_trend_alpha must be >= 0")
        if not 0.0 < self.msm_repair_handoff_q_threshold < 1.0:
            raise ValueError("controller.msm_repair_handoff_q_threshold must be in (0, 1)")



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
