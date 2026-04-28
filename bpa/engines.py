from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoTokenizer

from .config import BPAConfig


@contextmanager
def _cuda_device_scope(devices: str | None):
    if devices is None:
        yield
        return
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


@dataclass
class ModelEngine:
    name: str
    model_path: str
    tokenizer_path: str | None = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    cuda_visible_devices: str | None = None
    llm: Any | None = None
    tokenizer: Any | None = None

    def load(self) -> "ModelEngine":
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path or self.model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        if self.llm is None:
            try:
                from vllm import LLM
            except ImportError as exc:
                raise RuntimeError(
                    "vLLM is required for runtime generation. Install it on the GPU experiment host."
                ) from exc
            try:
                with _cuda_device_scope(self.cuda_visible_devices):
                    self.llm = LLM(model=self.model_path, tokenizer=self.tokenizer_path or self.model_path, **self.engine_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize vLLM engine {self.name!r} for model {self.model_path!r}. "
                    "If another BPA engine is already loaded in the same process, vLLM's default "
                    "gpu_memory_utilization=0.9 can reserve almost all GPU memory. Set "
                    "slm_engine_kwargs/llm_engine_kwargs.gpu_memory_utilization in the config, "
                    "reduce max_model_len, free other GPU processes, or use tensor parallelism / a smaller model."
                ) from exc
        return self

    def ensure_tokenizer(self) -> Any:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path or self.model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        return self.tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.ensure_tokenizer().encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return self.ensure_tokenizer().decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def sampling_params(self, **kwargs: Any) -> Any:
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM SamplingParams is unavailable in this environment.") from exc
        return SamplingParams(**kwargs)

    def tokens_prompt(self, prompt_token_ids: list[int]) -> Any:
        try:
            from vllm.inputs import TokensPrompt
        except ImportError:
            return {"prompt_token_ids": prompt_token_ids}
        return TokensPrompt(prompt_token_ids=prompt_token_ids)

    def generate(self, prompts: Any, sampling_params: Any) -> Any:
        self.load()
        return self.llm.generate(prompts, sampling_params)


def _engine_kwargs(config: BPAConfig, specific: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "max_model_len": config.max_model_len,
        "enable_prefix_caching": config.enable_prefix_caching,
    }
    kwargs.update(specific)
    return kwargs


def init_engines(config: BPAConfig) -> tuple[ModelEngine, ModelEngine]:
    slm = ModelEngine(
        name="slm",
        model_path=config.slm_model_path,
        tokenizer_path=config.slm_tokenizer_path,
        engine_kwargs=_engine_kwargs(config, config.slm_engine_kwargs),
        cuda_visible_devices=config.slm_device,
    )
    llm = ModelEngine(
        name="llm",
        model_path=config.llm_model_path,
        tokenizer_path=config.llm_tokenizer_path,
        engine_kwargs=_engine_kwargs(config, config.llm_engine_kwargs),
        cuda_visible_devices=config.llm_device,
    )
    return slm, llm


def completion(output: Any) -> Any:
    return output.outputs[0]


def generated_text(output: Any) -> str:
    return getattr(completion(output), "text", "") or ""


def generated_token_ids(output: Any) -> list[int]:
    return list(getattr(completion(output), "token_ids", []) or [])


def finish_reason(output: Any) -> str:
    return str(getattr(completion(output), "finish_reason", "") or "")


def completion_logprobs(output: Any) -> list[Any]:
    return list(getattr(completion(output), "logprobs", []) or [])


def prompt_logprobs(output: Any) -> list[Any]:
    return list(getattr(output, "prompt_logprobs", []) or [])


def logprob_value(record: Any) -> float:
    return float(getattr(record, "logprob", record))
