from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

from .config import BPAConfig


@contextmanager
def _cuda_device_scope(devices: str | None):
    if devices is None:
        yield
        return
    devices = devices.removeprefix("cuda:")
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
    backend: str = "vllm"
    api_base_url: str | None = None
    api_key: str = "EMPTY"
    api_model: str | None = None
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
            if self.backend == "openai":
                if not self.api_base_url:
                    raise RuntimeError(f"OpenAI-compatible backend for {self.name!r} requires api_base_url.")
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError(
                        "The openai package is required for remote OpenAI-compatible vLLM endpoints. "
                        "Install it on the experiment host."
                    ) from exc
                self.llm = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
                return self

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
        if self.backend == "openai":
            return dict(kwargs)
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
        if self.backend == "openai":
            return self._generate_openai(prompts, sampling_params)
        return self.llm.generate(prompts, sampling_params)

    def reset_prefix_cache(self) -> bool:
        if self.backend == "openai":
            return True
        if self.llm is None:
            return True
        reset = getattr(self.llm, "reset_prefix_cache", None)
        if reset is None:
            return False
        for args in ((), (False,), (False, False)):
            try:
                return bool(reset(*args))
            except TypeError:
                continue
            except Exception:
                return False
        return False

    def clear_runtime_cache(self) -> bool:
        ok = self.reset_prefix_cache()
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return ok

    def _generate_openai(self, prompts: Any, sampling_params: Any) -> list[Any]:
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        return [self._generate_openai_one(prompt, sampling_params) for prompt in prompt_list]

    def _generate_openai_one(self, prompt: Any, sampling_params: Any) -> Any:
        prompt_text = self._prompt_to_text(prompt)
        kwargs = self._openai_completion_kwargs(sampling_params)
        response = self.llm.completions.create(
            model=self.api_model or self.model_path,
            prompt=prompt_text,
            **kwargs,
        )
        choices = sorted(list(getattr(response, "choices", []) or []), key=lambda choice: getattr(choice, "index", 0))
        return SimpleNamespace(outputs=[self._openai_choice_to_completion(choice) for choice in choices])

    def _prompt_to_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            return self.decode(list(prompt["prompt_token_ids"]))
        token_ids = getattr(prompt, "prompt_token_ids", None)
        if token_ids is not None:
            return self.decode(list(token_ids))
        return str(prompt)

    @staticmethod
    def _sampling_value(sampling_params: Any, key: str, default: Any = None) -> Any:
        if isinstance(sampling_params, dict):
            return sampling_params.get(key, default)
        return getattr(sampling_params, key, default)

    def _openai_completion_kwargs(self, sampling_params: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        passthrough = (
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        )
        for key in passthrough:
            value = self._sampling_value(sampling_params, key)
            if value is not None:
                kwargs[key] = value

        logprobs = self._sampling_value(sampling_params, "logprobs")
        if logprobs is not None:
            kwargs["logprobs"] = int(logprobs)

        extra_body: dict[str, Any] = {}
        include_stop = self._sampling_value(sampling_params, "include_stop_str_in_output")
        if include_stop is not None:
            extra_body["include_stop_str_in_output"] = bool(include_stop)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _openai_choice_to_completion(self, choice: Any) -> Any:
        text = str(getattr(choice, "text", "") or "")
        token_ids, logprob_steps = self._openai_logprobs_to_vllm_shape(getattr(choice, "logprobs", None), text)
        return SimpleNamespace(
            text=text,
            token_ids=token_ids,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            logprobs=logprob_steps,
        )

    def _token_id_for_openai_token(self, token: str) -> int:
        token_ids = self.encode(token)
        if not token_ids:
            return 0
        return int(token_ids[0])

    def _openai_logprobs_to_vllm_shape(self, logprobs: Any, text: str) -> tuple[list[int], list[dict[int, Any]]]:
        if logprobs is None:
            return self.encode(text), []

        tokens = list(getattr(logprobs, "tokens", None) or [])
        token_logprobs = list(getattr(logprobs, "token_logprobs", None) or [])
        top_logprobs = list(getattr(logprobs, "top_logprobs", None) or [])
        if not tokens:
            return self.encode(text), []

        token_ids: list[int] = []
        logprob_steps: list[dict[int, Any]] = []
        for idx, token in enumerate(tokens):
            token_id = self._token_id_for_openai_token(str(token))
            token_ids.append(token_id)
            step: dict[int, Any] = {}
            if idx < len(top_logprobs) and top_logprobs[idx]:
                for top_token, logprob in dict(top_logprobs[idx]).items():
                    step[self._token_id_for_openai_token(str(top_token))] = SimpleNamespace(logprob=float(logprob))
            if idx < len(token_logprobs) and token_logprobs[idx] is not None:
                step.setdefault(token_id, SimpleNamespace(logprob=float(token_logprobs[idx])))
            logprob_steps.append(step)
        return token_ids, logprob_steps


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
        backend="openai" if config.slm_api_base_url else config.slm_backend,
        api_base_url=config.slm_api_base_url,
        api_key=config.slm_api_key,
        api_model=config.slm_api_model,
        engine_kwargs=_engine_kwargs(config, config.slm_engine_kwargs),
        cuda_visible_devices=config.slm_device,
    )
    llm = ModelEngine(
        name="llm",
        model_path=config.llm_model_path,
        tokenizer_path=config.llm_tokenizer_path,
        backend="openai" if config.llm_api_base_url else config.llm_backend,
        api_base_url=config.llm_api_base_url,
        api_key=config.llm_api_key,
        api_model=config.llm_api_model,
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


def logprob_value(record: Any) -> float:
    return float(getattr(record, "logprob", record))
