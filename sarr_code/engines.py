from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bpa.context_budget import generation_budget_for_rendered
from bpa.render import render_for_continuation

from .config import ModelRuntimeConfig, RuntimeConfig
from .records import StepOutput


def _read_chat_template(path: str | None) -> str | None:
    if not path:
        return None
    template_path = Path(path)
    if not template_path.exists():
        raise FileNotFoundError(f"Chat template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _validate_local_path(path: str, *, label: str, local_files_only: bool) -> None:
    if local_files_only and path and not Path(path).exists():
        raise FileNotFoundError(
            f"{label} must be a local path when local_files_only=true: {path}"
        )


def _device_string(device: str | None) -> str:
    if device is None or device == "":
        return "cuda:0"
    if device.isdigit():
        return f"cuda:{device}"
    return device


def _torch_dtype(dtype: str):
    import torch

    normalized = (dtype or "auto").lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def _load_tokenizer(cfg: ModelRuntimeConfig):
    from transformers import AutoTokenizer

    source = cfg.tokenizer_path or cfg.model_path
    _validate_local_path(source, label="tokenizer_path/model_path", local_files_only=cfg.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
        local_files_only=cfg.local_files_only,
    )
    template = _read_chat_template(cfg.chat_template_path)
    if template is not None:
        tokenizer.chat_template = template
    return tokenizer


def _decode(tokenizer: Any, token_ids: list[int] | tuple[int, ...]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _topk_entropy_from_logits(logits, topk: int) -> tuple[float, dict[str, Any]]:
    import torch

    k = min(int(topk), int(logits.shape[-1]))
    top_logits, top_ids = torch.topk(logits.float(), k=k)
    probs = torch.softmax(top_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    norm_entropy = entropy / math.log(k)
    confidence = 1.0 - float(norm_entropy)
    return confidence, {
        "top_ids": [int(x) for x in top_ids.detach().cpu().tolist()],
        "top_probs": [float(x) for x in probs.detach().cpu().tolist()],
        "norm_entropy": float(norm_entropy.detach().cpu()),
    }


class _StopOnSubstrings:
    def __init__(self, tokenizer: Any, prompt_len: int, stop_strings: list[str]):
        from transformers import StoppingCriteria

        class Stopper(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
                generated = input_ids[0, prompt_len:].detach().cpu().tolist()
                if not generated:
                    return False
                text = _decode(tokenizer, generated)
                return any(stop in text for stop in stop_strings)

        self.criteria = Stopper()


@dataclass
class LocalTransformersSLM:
    cfg: ModelRuntimeConfig
    runtime: RuntimeConfig
    tokenizer: Any | None = None
    model: Any | None = None
    device: str | None = None

    def load(self) -> "LocalTransformersSLM":
        if self.tokenizer is None:
            self.tokenizer = _load_tokenizer(self.cfg)
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM

            _validate_local_path(self.cfg.model_path, label="slm.model_path", local_files_only=self.cfg.local_files_only)
            self.device = _device_string(self.cfg.device)
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Requested {self.device}, but torch.cuda.is_available() is false.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_path,
                trust_remote_code=self.cfg.trust_remote_code,
                torch_dtype=_torch_dtype(self.cfg.dtype),
                local_files_only=self.cfg.local_files_only,
            )
            self.model.to(self.device)
            self.model.eval()
        return self

    def ensure_tokenizer(self) -> Any:
        self.load()
        return self.tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.ensure_tokenizer().encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return _decode(self.ensure_tokenizer(), list(token_ids))

    def render(self, problem_text: str, assistant_prefix_text: str) -> str:
        return render_for_continuation(problem_text, assistant_prefix_text, self.ensure_tokenizer())

    def generate_step(
        self,
        problem_text: str,
        assistant_prefix_text: str,
        *,
        max_new_tokens: int,
        stop_delimiters: list[str] | None,
        capture_token_entropy: bool = False,
        topk_entropy: int = 20,
    ) -> StepOutput:
        import torch
        from transformers import StoppingCriteriaList

        self.load()
        rendered = self.render(problem_text, assistant_prefix_text)
        max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, self, self.runtime, max_new_tokens)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        stop_strings = list(stop_delimiters or [])
        stopping = StoppingCriteriaList()
        if stop_strings:
            stopping.append(_StopOnSubstrings(self.tokenizer, input_ids.shape[-1], stop_strings).criteria)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        start = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                stopping_criteria=stopping,
                return_dict_in_generate=True,
                output_scores=bool(capture_token_entropy),
            )
        wall_time = time.time() - start

        sequence = output.sequences[0]
        new_token_ids = [int(x) for x in sequence[input_ids.shape[-1] :].detach().cpu().tolist()]
        text = self.decode(new_token_ids)
        finish = "length"
        if stop_strings and any(stop in text for stop in stop_strings):
            finish = "stop"
        elif eos_token_id is not None and new_token_ids and new_token_ids[-1] == int(eos_token_id):
            finish = "eos"

        extra: dict[str, Any] = {}
        if capture_token_entropy and getattr(output, "scores", None):
            token_entropies = []
            for score in output.scores:
                _, info = _topk_entropy_from_logits(score[0], topk_entropy)
                token_entropies.append(info["norm_entropy"])
            extra["token_norm_entropies"] = token_entropies

        return StepOutput(
            text=text,
            token_ids=new_token_ids,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            wall_time=wall_time,
            extra=extra,
        )

    def generate_text(
        self,
        problem_text: str,
        assistant_prefix_text: str,
        *,
        max_new_tokens: int,
        stop_delimiters: list[str] | None = None,
        include_stop_str_in_output: bool = True,
    ) -> StepOutput:
        return self.generate_step(
            problem_text,
            assistant_prefix_text,
            max_new_tokens=max_new_tokens,
            stop_delimiters=stop_delimiters,
            capture_token_entropy=False,
            topk_entropy=20,
        )

    def continuation_confidence(
        self,
        problem_text: str,
        assistant_prefix_text: str,
        *,
        topk: int,
    ) -> tuple[float, dict[str, Any]]:
        import torch

        self.load()
        rendered = self.render(problem_text, assistant_prefix_text)
        generation_budget_for_rendered(rendered, self, self.runtime, 1)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        start = time.time()
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        wall_time = time.time() - start
        logits = outputs.logits[0, -1, :]
        confidence, info = _topk_entropy_from_logits(logits, topk)
        info["wall_time"] = wall_time
        info["prompt_tokens"] = int(input_ids.shape[-1])
        try:
            info["top_tokens"] = [self.decode([token_id]) for token_id in info["top_ids"]]
        except Exception:
            info["top_tokens"] = []
        return confidence, info

    def clear_runtime_cache(self) -> bool:
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return False
        return True


@contextmanager
def _cuda_visible_devices(devices: str | None):
    if devices is None or devices == "":
        yield
        return
    normalized = devices.removeprefix("cuda:")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


@dataclass
class CompletionEngine:
    cfg: ModelRuntimeConfig
    runtime: RuntimeConfig
    name: str = "llm"
    tokenizer: Any | None = None
    llm: Any | None = None
    _openai_client: Any | None = None

    def load(self) -> "CompletionEngine":
        if self.tokenizer is None:
            self.tokenizer = _load_tokenizer(self.cfg)
        if self.llm is None and self.cfg.backend == "vllm":
            try:
                from vllm import LLM
            except ImportError as exc:
                raise RuntimeError("vLLM is required for llm.backend='vllm'.") from exc
            kwargs = {
                "trust_remote_code": self.cfg.trust_remote_code,
                "max_model_len": self.runtime.max_model_len,
                **self.cfg.engine_kwargs,
            }
            with _cuda_visible_devices(self.cfg.device):
                self.llm = LLM(
                    model=self.cfg.model_path,
                    tokenizer=self.cfg.tokenizer_path or self.cfg.model_path,
                    **kwargs,
                )
        elif self._openai_client is None and self.cfg.backend == "openai":
            if not self.cfg.api_base_url:
                raise RuntimeError("llm.backend='openai' requires llm.api_base_url.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("The openai package is required for OpenAI-compatible vLLM endpoints.") from exc
            self._openai_client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.api_base_url)
        return self

    def ensure_tokenizer(self) -> Any:
        self.load()
        return self.tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.ensure_tokenizer().encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return _decode(self.ensure_tokenizer(), list(token_ids))

    def render(self, problem_text: str, assistant_prefix_text: str) -> str:
        return render_for_continuation(problem_text, assistant_prefix_text, self.ensure_tokenizer())

    def _sampling_params(self, **kwargs: Any) -> Any:
        if self.cfg.backend == "openai":
            return kwargs
        from vllm import SamplingParams

        return SamplingParams(**kwargs)

    def generate_text(
        self,
        problem_text: str,
        assistant_prefix_text: str,
        *,
        max_new_tokens: int,
        stop_delimiters: list[str] | None = None,
        include_stop_str_in_output: bool = True,
    ) -> StepOutput:
        self.load()
        rendered = self.render(problem_text, assistant_prefix_text)
        max_tokens, prompt_tokens = generation_budget_for_rendered(rendered, self, self.runtime, max_new_tokens)
        start = time.time()
        if self.cfg.backend == "openai":
            output = self._generate_openai(
                rendered,
                max_tokens=max_tokens,
                stop=stop_delimiters,
                include_stop_str_in_output=include_stop_str_in_output,
            )
        else:
            output = self._generate_vllm(
                rendered,
                max_tokens=max_tokens,
                stop=stop_delimiters,
                include_stop_str_in_output=include_stop_str_in_output,
            )
        output.wall_time = time.time() - start
        output.prompt_tokens = prompt_tokens
        return output

    def generate_step(
        self,
        problem_text: str,
        assistant_prefix_text: str,
        *,
        max_new_tokens: int,
        stop_delimiters: list[str] | None,
    ) -> StepOutput:
        return self.generate_text(
            problem_text,
            assistant_prefix_text,
            max_new_tokens=max_new_tokens,
            stop_delimiters=stop_delimiters,
            include_stop_str_in_output=True,
        )

    def _generate_openai(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None,
        include_stop_str_in_output: bool,
    ) -> StepOutput:
        kwargs: dict[str, Any] = {
            "model": self.cfg.api_model or self.cfg.model_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if stop:
            kwargs["stop"] = stop
            kwargs["extra_body"] = {"include_stop_str_in_output": include_stop_str_in_output}
        response = self._openai_client.completions.create(**kwargs)
        choices = sorted(list(getattr(response, "choices", []) or []), key=lambda c: getattr(c, "index", 0))
        choice = choices[0] if choices else SimpleNamespace(text="", finish_reason="")
        text = str(getattr(choice, "text", "") or "")
        finish = str(getattr(choice, "finish_reason", "") or "")
        return StepOutput(
            text=text,
            token_ids=self.encode(text),
            finish_reason=finish,
        )

    def _generate_vllm(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None,
        include_stop_str_in_output: bool,
    ) -> StepOutput:
        sampling = self._sampling_params(
            max_tokens=max_tokens,
            temperature=0.0,
            stop=stop,
            include_stop_str_in_output=include_stop_str_in_output,
        )
        out = self.llm.generate(prompt, sampling)[0]
        completion = out.outputs[0]
        text = str(getattr(completion, "text", "") or "")
        token_ids = [int(x) for x in (getattr(completion, "token_ids", []) or [])]
        if not token_ids:
            token_ids = self.encode(text)
        finish = str(getattr(completion, "finish_reason", "") or "")
        return StepOutput(text=text, token_ids=token_ids, finish_reason=finish)

    def clear_runtime_cache(self) -> bool:
        if self.cfg.backend == "openai":
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


def build_slm(cfg: ModelRuntimeConfig, runtime: RuntimeConfig) -> LocalTransformersSLM:
    return LocalTransformersSLM(cfg=cfg, runtime=runtime)


def build_llm(cfg: ModelRuntimeConfig, runtime: RuntimeConfig) -> CompletionEngine:
    return CompletionEngine(cfg=cfg, runtime=runtime, name="llm")
