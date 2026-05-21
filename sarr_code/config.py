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
    close_tag_lookahead_tokens: int = 16
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
        if self.close_tag_lookahead_tokens < 0:
            raise ValueError("generation.close_tag_lookahead_tokens must be >= 0")
        if self.think_token_budget < 1:
            raise ValueError("generation.think_token_budget must be >= 1")
        if self.answer_token_budget < 1:
            raise ValueError("generation.answer_token_budget must be >= 1")
        if self.final_answer_generator not in {"slm", "llm"}:
            raise ValueError("generation.final_answer_generator must be 'slm' or 'llm'")


@dataclass
class ConfidenceConfig:
    topk_entropy: int = 20
    percentile_normalization: bool = False
    calibration_path: str | None = None
    smooth_window: int = 2
    delta: float = 0.55
    capture_slm_token_entropy: bool = False

    def __post_init__(self) -> None:
        if self.topk_entropy < 2:
            raise ValueError("confidence.topk_entropy must be >= 2")
        if self.smooth_window < 1:
            raise ValueError("confidence.smooth_window must be >= 1")


@dataclass
class ConfidenceProcessConfig:
    lambda0: float = 0.002
    alpha: float = 1.0
    r0: int = 20
    power: float = 2.0
    high_threshold: float = 0.70
    raw_low_threshold: float = 0.35
    smooth_low_threshold: float = 0.35

    def __post_init__(self) -> None:
        if self.lambda0 < 0.0:
            raise ValueError("confidence_process.lambda0 must be >= 0")
        if self.alpha < 0.0:
            raise ValueError("confidence_process.alpha must be >= 0")
        if self.r0 < 0:
            raise ValueError("confidence_process.r0 must be >= 0")
        if self.power < 0.0:
            raise ValueError("confidence_process.power must be >= 0")
        for name in ["high_threshold", "raw_low_threshold", "smooth_low_threshold"]:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence_process.{name} must be in [0, 1]")


@dataclass
class CalibrationConfig:
    enabled: bool = False
    build_cdf: bool = False
    load_cdf: bool = False
    use_percentile: bool = False

    def __post_init__(self) -> None:
        if self.enabled or self.build_cdf or self.load_cdf or self.use_percentile:
            raise ValueError("This experiment disables calibration; calibration.* must all be false")


@dataclass
class ReadinessConfig:
    signal: str = "continuation_confidence"
    normalization: str = "raw"
    use_calibration: bool = False
    value_field: str = "readiness_smooth_or_raw"
    high_threshold: float = 0.70
    low_threshold: float = 0.35
    smooth_window: int = 3

    def __post_init__(self) -> None:
        if self.signal != "continuation_confidence":
            raise ValueError("readiness.signal must be 'continuation_confidence'")
        if self.normalization != "raw":
            raise ValueError("This experiment disables calibration; readiness.normalization must be 'raw'")
        if self.use_calibration:
            raise ValueError("This experiment disables calibration; readiness.use_calibration must be false")
        if self.value_field not in {"readiness_smooth_or_raw", "c_raw"}:
            raise ValueError(
                "This experiment disables calibration; readiness.value_field must be "
                "'readiness_smooth_or_raw' or 'c_raw'"
            )
        if not 0.0 <= self.high_threshold <= 1.0:
            raise ValueError("readiness.high_threshold must be in [0, 1]")
        if not 0.0 <= self.low_threshold <= 1.0:
            raise ValueError("readiness.low_threshold must be in [0, 1]")
        if self.low_threshold > self.high_threshold:
            raise ValueError("readiness.low_threshold must be <= readiness.high_threshold")
        if self.smooth_window < 1:
            raise ValueError("readiness.smooth_window must be >= 1")


@dataclass
class StagnationConfig:
    enabled: bool = True
    unit: str = "step_or_small_block"
    block_min_tokens: int = 32
    block_max_steps: int = 2
    metric: str = "word_3gram_jaccard"
    repeat_window: int = 10
    ngram_n: int = 3
    high_threshold: float = 0.85
    patience: int = 3
    include_mid_readiness: bool = True

    def __post_init__(self) -> None:
        if self.unit != "step_or_small_block":
            raise ValueError("stagnation.unit must be 'step_or_small_block'")
        if self.block_min_tokens < 1:
            raise ValueError("stagnation.block_min_tokens must be >= 1")
        if self.block_max_steps < 1:
            raise ValueError("stagnation.block_max_steps must be >= 1")
        if self.metric != "word_3gram_jaccard":
            raise ValueError("stagnation.metric must be 'word_3gram_jaccard'")
        if self.repeat_window < 1:
            raise ValueError("stagnation.repeat_window must be >= 1")
        if self.ngram_n < 1:
            raise ValueError("stagnation.ngram_n must be >= 1")
        if not 0.0 <= self.high_threshold <= 1.0:
            raise ValueError("stagnation.high_threshold must be in [0, 1]")
        if self.patience < 1:
            raise ValueError("stagnation.patience must be >= 1")


@dataclass
class AnchorConfig:
    type: str = "clean_autonomy_anchor"
    refresh_condition: str = "slm_step_and_readiness_high_and_not_stagnation_suspect_and_slm_active"
    freeze_on_hcs_suspect: bool = True
    freeze_on_stagnation_suspect: bool = True
    llm_steps_do_not_refresh: bool = True
    fallback: str = "startup_anchor_or_zero"

    def __post_init__(self) -> None:
        if self.type != "clean_autonomy_anchor":
            raise ValueError("anchor.type must be 'clean_autonomy_anchor'")
        if self.refresh_condition not in {
            "raw_readiness_high_and_not_hcs_suspect",
            "slm_step_and_readiness_high_and_not_stagnation_suspect_and_slm_active",
        }:
            raise ValueError(
                "anchor.refresh_condition must be "
                "'slm_step_and_readiness_high_and_not_stagnation_suspect_and_slm_active'"
            )
        if self.fallback not in {"startup_anchor_or_zero", "zero"}:
            raise ValueError("anchor.fallback must be 'startup_anchor_or_zero' or 'zero'")


@dataclass
class RoutingConfig:
    enabled: bool = True


@dataclass
class LowReadinessConfig:
    useful_exploration_grace_steps: int = 2
    persistent_low_after_grace_action: str = "llm_lease_no_rollback"

    def __post_init__(self) -> None:
        if self.useful_exploration_grace_steps < 0:
            raise ValueError("low_readiness.useful_exploration_grace_steps must be >= 0")
        if self.persistent_low_after_grace_action != "llm_lease_no_rollback":
            raise ValueError(
                "low_readiness.persistent_low_after_grace_action must be 'llm_lease_no_rollback'"
            )


@dataclass
class ConfirmedStagnationConfig:
    action: str = "rollback_to_clean_anchor_then_llm_lease"
    include_mid_readiness: bool = True
    max_rollbacks_per_problem: int = 2

    def __post_init__(self) -> None:
        if self.action != "rollback_to_clean_anchor_then_llm_lease":
            raise ValueError(
                "confirmed_stagnation.action must be 'rollback_to_clean_anchor_then_llm_lease'"
            )
        if self.max_rollbacks_per_problem < 0:
            raise ValueError("confirmed_stagnation.max_rollbacks_per_problem must be >= 0")


@dataclass
class LLMLeaseConfig:
    enabled: bool = True
    prompt_type: str = "normal_continuation"
    mention_uncertainty: bool = False
    mention_stagnation: bool = False
    mention_repetition: bool = False
    mention_error: bool = False
    persistent_uncertainty_steps: int = 2
    confirmed_stagnation_steps: int = 3
    low_conf_rollback_steps: int = 2
    post_recovery_stabilization_steps: int = 1
    max_tokens_per_step: int = 128
    max_events_per_problem: int = 4
    max_total_tokens_per_problem: int = 1024
    return_to_slm: bool = True

    def __post_init__(self) -> None:
        if self.prompt_type != "normal_continuation":
            raise ValueError("llm_lease.prompt_type must be 'normal_continuation'")
        if self.mention_uncertainty or self.mention_stagnation or self.mention_repetition or self.mention_error:
            raise ValueError("LLM lease prompts must not mention uncertainty, stagnation, repetition, or errors")
        for name in [
            "persistent_uncertainty_steps",
            "confirmed_stagnation_steps",
            "low_conf_rollback_steps",
            "post_recovery_stabilization_steps",
            "max_tokens_per_step",
            "max_events_per_problem",
            "max_total_tokens_per_problem",
        ]:
            if getattr(self, name) < 0:
                raise ValueError(f"llm_lease.{name} must be >= 0")
        if self.max_tokens_per_step < 1:
            raise ValueError("llm_lease.max_tokens_per_step must be >= 1")


@dataclass
class PostLeaseObserveConfig:
    observe_slm_blocks: int = 2
    suppress_startup_rollback: bool = True
    suppress_immediate_rollback: bool = True

    def __post_init__(self) -> None:
        if self.observe_slm_blocks < 0:
            raise ValueError("post_lease_observe.observe_slm_blocks must be >= 0")


@dataclass
class HCSConfig:
    enabled: bool = True
    enable_after_clean_anchor: bool = True
    suspect_condition: str = "raw_readiness_high_and_stagnation_high"
    suspect_patience: int = 3
    action: str = "rollback_to_clean_anchor"
    max_hcs_rollbacks_per_problem: int = 2

    def __post_init__(self) -> None:
        if self.suspect_condition != "raw_readiness_high_and_stagnation_high":
            raise ValueError("hcs.suspect_condition must be 'raw_readiness_high_and_stagnation_high'")
        if self.suspect_patience < 1:
            raise ValueError("hcs.suspect_patience must be >= 1")
        if self.action != "rollback_to_clean_anchor":
            raise ValueError("hcs.action must be 'rollback_to_clean_anchor'")
        if self.max_hcs_rollbacks_per_problem < 0:
            raise ValueError("hcs.max_hcs_rollbacks_per_problem must be >= 0")


@dataclass
class HCSRecoveryConfig:
    generator: str = "llm"
    prompt_type: str = "normal_continuation"
    mention_stagnation: bool = False
    mention_repetition: bool = False
    max_llm_steps: int = 2
    max_tokens_per_step: int = 128
    return_to_slm_after_recovery: bool = True

    def __post_init__(self) -> None:
        if self.generator != "llm":
            raise ValueError("hcs_recovery.generator must be 'llm'")
        if self.prompt_type != "normal_continuation":
            raise ValueError("hcs_recovery.prompt_type must be 'normal_continuation'")
        if self.mention_stagnation:
            raise ValueError("hcs_recovery.mention_stagnation must remain false")
        if self.mention_repetition:
            raise ValueError("hcs_recovery.mention_repetition must remain false")
        if self.max_llm_steps < 1:
            raise ValueError("hcs_recovery.max_llm_steps must be >= 1")
        if self.max_tokens_per_step < 1:
            raise ValueError("hcs_recovery.max_tokens_per_step must be >= 1")


@dataclass
class LowConfidenceConfig:
    useful_exploration_grace_blocks: int = 2
    collapse_patience_blocks: int = 3
    action_after_patience: str = "existing_rollback_recovery"

    def __post_init__(self) -> None:
        if self.useful_exploration_grace_blocks < 0:
            raise ValueError("low_confidence.useful_exploration_grace_blocks must be >= 0")
        if self.collapse_patience_blocks < 1:
            raise ValueError("low_confidence.collapse_patience_blocks must be >= 1")
        if self.collapse_patience_blocks <= self.useful_exploration_grace_blocks:
            raise ValueError("low_confidence.collapse_patience_blocks must be > useful_exploration_grace_blocks")
        if self.action_after_patience != "existing_rollback_recovery":
            raise ValueError("low_confidence.action_after_patience must be 'existing_rollback_recovery'")


@dataclass
class StartupGuardConfig:
    hcs_enabled: bool = False
    enable_hcs_after_clean_anchor: bool = True


@dataclass
class BudgetConfig:
    max_total_hcs_llm_tokens_per_problem: int = 512
    max_total_llm_recovery_tokens_per_problem: int = 1024
    max_llm_lease_events_per_problem: int = 4
    max_llm_lease_tokens_per_problem: int = 1024
    max_rollbacks_per_problem: int = 4
    max_stagnation_rollbacks_per_problem: int = 2

    def __post_init__(self) -> None:
        if self.max_total_hcs_llm_tokens_per_problem < 0:
            raise ValueError("budget.max_total_hcs_llm_tokens_per_problem must be >= 0")
        if self.max_total_llm_recovery_tokens_per_problem < 0:
            raise ValueError("budget.max_total_llm_recovery_tokens_per_problem must be >= 0")
        if self.max_llm_lease_events_per_problem < 0:
            raise ValueError("budget.max_llm_lease_events_per_problem must be >= 0")
        if self.max_llm_lease_tokens_per_problem < 0:
            raise ValueError("budget.max_llm_lease_tokens_per_problem must be >= 0")
        if self.max_rollbacks_per_problem < 0:
            raise ValueError("budget.max_rollbacks_per_problem must be >= 0")
        if self.max_stagnation_rollbacks_per_problem < 0:
            raise ValueError("budget.max_stagnation_rollbacks_per_problem must be >= 0")


@dataclass
class StartupConfig:
    B_min: int = 2
    B_max: int = 5
    tau_start: int = 1
    max_steps: int | None = None
    never_reenter_after_recovery: bool = True
    never_reenter_after_llm_lease: bool = True
    never_reenter_after_clean_anchor: bool = True

    def __post_init__(self) -> None:
        if self.max_steps is not None:
            self.B_max = self.max_steps
        if self.B_min < 1:
            raise ValueError("startup.B_min must be >= 1")
        if self.B_max < self.B_min:
            raise ValueError("startup.B_max must be >= startup.B_min")
        if self.tau_start < 1:
            raise ValueError("startup.tau_start must be >= 1")
        self.max_steps = self.B_max


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
    post_stable_intervention_policy: str = "suspect_confirmed_rollback"
    suspect_confirm_steps: int = 1
    suspect_max_steps: int = 2
    tau_confirm: int = 1
    anchor_repeat_policy: str = "suppress"
    anchor_repeat_backoff_after: int = 1
    anchor_repeat_backoff_steps: int = 1
    max_root_rollbacks: int = 2
    root_rollback_action: str = "force_close_think"
    max_rollbacks_per_problem: int = 4
    max_stagnation_rollbacks_per_problem: int = 2
    fallback_if_no_clean_anchor: str = "startup_anchor_or_zero"

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
        if self.post_stable_intervention_policy not in {
            "suspect_confirmed_rollback",
            "rollback_to_anchor",
        }:
            raise ValueError(
                "rollback.post_stable_intervention_policy must be "
                "'suspect_confirmed_rollback' or 'rollback_to_anchor'"
            )
        if self.suspect_confirm_steps < 1:
            raise ValueError("rollback.suspect_confirm_steps must be >= 1")
        if self.suspect_max_steps < self.suspect_confirm_steps:
            raise ValueError("rollback.suspect_max_steps must be >= suspect_confirm_steps")
        if self.tau_confirm < 1:
            raise ValueError("rollback.tau_confirm must be >= 1")
        if self.anchor_repeat_policy not in {"suppress", "backoff", "allow"}:
            raise ValueError("rollback.anchor_repeat_policy must be 'suppress', 'backoff', or 'allow'")
        if self.anchor_repeat_backoff_after < 1:
            raise ValueError("rollback.anchor_repeat_backoff_after must be >= 1")
        if self.anchor_repeat_backoff_steps < 0:
            raise ValueError("rollback.anchor_repeat_backoff_steps must be >= 0")
        if self.max_root_rollbacks < 0:
            raise ValueError("rollback.max_root_rollbacks must be >= 0")
        if self.root_rollback_action not in {"force_close_think", "allow"}:
            raise ValueError("rollback.root_rollback_action must be 'force_close_think' or 'allow'")
        if self.max_rollbacks_per_problem < 0:
            raise ValueError("rollback.max_rollbacks_per_problem must be >= 0")
        if self.max_stagnation_rollbacks_per_problem < 0:
            raise ValueError("rollback.max_stagnation_rollbacks_per_problem must be >= 0")
        if self.fallback_if_no_clean_anchor not in {"startup_anchor_or_zero", "zero"}:
            raise ValueError("rollback.fallback_if_no_clean_anchor must be 'startup_anchor_or_zero' or 'zero'")


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
    method: str = "sarr_code_v3_state_aware_routing_rollback"
    metadata: dict[str, Any] = field(default_factory=dict)
    slm: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(model_path=""))
    llm: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(model_path="", backend="openai"))
    output_dir: str = "sarr_results"
    dataset_paths: dict[str, str] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    confidence_process: ConfidenceProcessConfig = field(default_factory=ConfidenceProcessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    low_readiness: LowReadinessConfig = field(default_factory=LowReadinessConfig)
    confirmed_stagnation: ConfirmedStagnationConfig = field(default_factory=ConfirmedStagnationConfig)
    llm_lease: LLMLeaseConfig = field(default_factory=LLMLeaseConfig)
    post_lease_observe: PostLeaseObserveConfig = field(default_factory=PostLeaseObserveConfig)
    hcs: HCSConfig = field(default_factory=HCSConfig)
    hcs_recovery: HCSRecoveryConfig = field(default_factory=HCSRecoveryConfig)
    low_confidence: LowConfidenceConfig = field(default_factory=LowConfidenceConfig)
    startup_guard: StartupGuardConfig = field(default_factory=StartupGuardConfig)
    startup: StartupConfig = field(default_factory=StartupConfig)
    stable: StableConfig = field(default_factory=StableConfig)
    rollback: RollbackConfig = field(default_factory=RollbackConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
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
            ("confidence_process", ConfidenceProcessConfig),
            ("calibration", CalibrationConfig),
            ("readiness", ReadinessConfig),
            ("stagnation", StagnationConfig),
            ("anchor", AnchorConfig),
            ("routing", RoutingConfig),
            ("low_readiness", LowReadinessConfig),
            ("confirmed_stagnation", ConfirmedStagnationConfig),
            ("llm_lease", LLMLeaseConfig),
            ("post_lease_observe", PostLeaseObserveConfig),
            ("hcs", HCSConfig),
            ("hcs_recovery", HCSRecoveryConfig),
            ("low_confidence", LowConfidenceConfig),
            ("startup_guard", StartupGuardConfig),
            ("startup", StartupConfig),
            ("stable", StableConfig),
            ("rollback", RollbackConfig),
            ("budget", BudgetConfig),
            ("logging", LoggingConfig),
            ("runtime", RuntimeConfig),
        ]:
            if isinstance(kwargs.get(key), dict):
                kwargs[key] = cls_(**kwargs[key])
        cfg = cls(**kwargs)
        if cfg.confidence.percentile_normalization:
            raise ValueError("This experiment disables calibration; confidence.percentile_normalization must be false")
        if cfg.confidence.calibration_path:
            raise ValueError("This experiment disables calibration; confidence.calibration_path must be null")
        if cfg.slm.backend != "transformers":
            raise ValueError("SARR-CoDE requires slm.backend='transformers' for local logits diagnostics.")
        if cfg.llm.backend == "transformers":
            raise ValueError("Use llm.backend='openai' or 'vllm'. SLM is the local transformers model.")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
