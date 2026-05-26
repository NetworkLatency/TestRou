from __future__ import annotations

import math
import os
import sys
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


def _hf_source(path: str) -> str:
    # A trailing slash can make transformers' dynamic remote-code module name
    # resolve to an empty string for local paths with trust_remote_code=True.
    if not path:
        return path
    return Path(path).as_posix().rstrip("/\\") or path


def _read_chat_template(path: str | None) -> str | None:
    if not path:
        return None
    template_path = Path(_hf_source(path))
    if not template_path.exists():
        raise FileNotFoundError(f"Chat template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _validate_local_path(path: str, *, label: str, local_files_only: bool) -> None:
    normalized = _hf_source(path)
    if local_files_only and normalized and not Path(normalized).exists():
        raise FileNotFoundError(
            f"{label} must be a local path when local_files_only=true: {normalized}"
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

    source = _hf_source(cfg.tokenizer_path or cfg.model_path)
    _validate_local_path(source, label="tokenizer_path/model_path", local_files_only=cfg.local_files_only)
    print(f"[sarr] loading tokenizer: {source}", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
        local_files_only=cfg.local_files_only,
    )
    template = _read_chat_template(cfg.chat_template_path)
    if template is not None:
        tokenizer.chat_template = template
        print(f"[sarr] loaded chat template: {cfg.chat_template_path}", file=sys.stderr, flush=True)
    return tokenizer


def _load_causal_lm(model_path: str, cfg: ModelRuntimeConfig, dtype_value: Any):
    from transformers import AutoModelForCausalLM

    source = _hf_source(model_path)
    kwargs = {
        "trust_remote_code": cfg.trust_remote_code,
        "local_files_only": cfg.local_files_only,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            source,
            dtype=dtype_value,
            **kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=dtype_value,
            **kwargs,
        )


def _sanitize_greedy_generation_config(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    for key in ("temperature", "top_p", "top_k", "typical_p"):
        if hasattr(generation_config, key):
            try:
                setattr(generation_config, key, None)
            except Exception:
                pass
    if hasattr(generation_config, "do_sample"):
        try:
            generation_config.do_sample = False
        except Exception:
            pass


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
    top_probs = [float(x) for x in probs.detach().cpu().tolist()]
    margin = top_probs[0] - top_probs[1] if len(top_probs) >= 2 else top_probs[0]
    return confidence, {
        "top_ids": [int(x) for x in top_ids.detach().cpu().tolist()],
        "top_probs": top_probs,
        "top1_prob": top_probs[0] if top_probs else 0.0,
        "top2_prob": top_probs[1] if len(top_probs) >= 2 else 0.0,
        "margin": float(margin),
        "norm_entropy": float(norm_entropy.detach().cpu()),
    }


def _generated_token_logprob_from_logits(logits, token_id: int) -> float | None:
    import torch

    if int(token_id) < 0 or int(token_id) >= int(logits.shape[-1]):
        return None
    values = logits.float()
    logprob = values[int(token_id)] - torch.logsumexp(values, dim=-1)
    return float(logprob.detach().cpu())


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    mu = sum(values) / len(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def _token_probability_payload(generated_token_logprobs: list[float]) -> dict[str, Any]:
    generated_token_probs = [math.exp(value) for value in generated_token_logprobs]
    return {
        "generated_token_logprobs": generated_token_logprobs,
        "token_probability": {
            "first_logprob": generated_token_logprobs[0] if generated_token_logprobs else None,
            "first_prob": generated_token_probs[0] if generated_token_probs else None,
            "mean_logprob": _mean(generated_token_logprobs),
            "mean_prob": _mean(generated_token_probs),
            "std_logprob": _pstdev(generated_token_logprobs),
            "min_logprob": min(generated_token_logprobs) if generated_token_logprobs else None,
            "token_count": len(generated_token_logprobs),
        },
    }


def _logprob_value(value: Any) -> float | None:
    if value is None:
        return None
    raw = getattr(value, "logprob", value)
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


class _StopOnSubstrings:
    def __init__(
        self,
        tokenizer: Any,
        prompt_len: int,
        stop_strings: list[str],
        *,
        immediate_stop_strings: list[str] | None = None,
    ):
        from transformers import StoppingCriteria

        self.stop_reason: str | None = None
        self.stop_text: str | None = None
        self.immediate_stop_strings = [s for s in (immediate_stop_strings or []) if s]
        outer = self

        class Stopper(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
                generated = input_ids[0, prompt_len:].detach().cpu().tolist()
                if not generated:
                    return False
                text = _decode(tokenizer, generated)
                for stop in outer.immediate_stop_strings:
                    if stop in text:
                        outer.stop_reason = "immediate_stop"
                        outer.stop_text = stop
                        return True
                if any(text.endswith(stop) for stop in stop_strings):
                    outer.stop_reason = "stop"
                    outer.stop_text = next(stop for stop in stop_strings if text.endswith(stop))
                    return True
                return False

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

            model_source = _hf_source(self.cfg.model_path)
            _validate_local_path(model_source, label="slm.model_path", local_files_only=self.cfg.local_files_only)
            self.device = _device_string(self.cfg.device)
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Requested {self.device}, but torch.cuda.is_available() is false.")
            print(
                f"[sarr] loading SLM model: {model_source} -> {self.device} dtype={self.cfg.dtype}",
                file=sys.stderr,
                flush=True,
            )
            self.model = _load_causal_lm(model_source, self.cfg, _torch_dtype(self.cfg.dtype))
            _sanitize_greedy_generation_config(self.model)
            self.model.to(self.device)
            self.model.eval()
            print("[sarr] SLM model loaded", file=sys.stderr, flush=True)
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
        capture_token_logprobs: bool = False,
        topk_entropy: int = 20,
        topk_logprobs: int | None = None,
        immediate_stop_strings: list[str] | None = None,
    ) -> StepOutput:
        del topk_logprobs
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
        stop_tracker = None
        if stop_strings:
            stop_tracker = _StopOnSubstrings(
                self.tokenizer,
                input_ids.shape[-1],
                stop_strings,
                immediate_stop_strings=immediate_stop_strings,
            )
            stopping.append(stop_tracker.criteria)
        elif immediate_stop_strings:
            stop_tracker = _StopOnSubstrings(
                self.tokenizer,
                input_ids.shape[-1],
                [],
                immediate_stop_strings=immediate_stop_strings,
            )
            stopping.append(stop_tracker.criteria)

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
                temperature=None,
                top_p=None,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                stopping_criteria=stopping,
                return_dict_in_generate=True,
                output_scores=bool(capture_token_entropy or capture_token_logprobs),
            )
        wall_time = time.time() - start

        sequence = output.sequences[0]
        new_token_ids = [int(x) for x in sequence[input_ids.shape[-1] :].detach().cpu().tolist()]
        text = self.decode(new_token_ids)
        visible_token_ids = new_token_ids
        finish = "length"
        if stop_tracker is not None and stop_tracker.stop_reason in {"immediate_stop", "stop"}:
            finish = "stop"
        elif eos_token_id is not None and new_token_ids and new_token_ids[-1] == int(eos_token_id):
            finish = "eos"

        extra: dict[str, Any] = {
            "actual_token_count": len(new_token_ids),
        }
        if stop_tracker is not None and stop_tracker.stop_reason:
            extra["stop_reason_detail"] = stop_tracker.stop_reason
        if (capture_token_entropy or capture_token_logprobs) and getattr(output, "scores", None):
            token_entropies = []
            token_confidences = []
            token_margins = []
            generated_token_logprobs = []
            first_info: dict[str, Any] | None = None
            for idx, score in enumerate(output.scores):
                logits = score[0]
                if capture_token_entropy:
                    confidence, info = _topk_entropy_from_logits(logits, topk_entropy)
                    if first_info is None:
                        first_info = dict(info)
                    token_entropies.append(info["norm_entropy"])
                    token_confidences.append(confidence)
                    token_margins.append(info["margin"])
                if capture_token_logprobs and idx < len(new_token_ids):
                    logprob = _generated_token_logprob_from_logits(logits, new_token_ids[idx])
                    if logprob is not None:
                        generated_token_logprobs.append(logprob)
            if capture_token_entropy:
                extra["token_norm_entropies"] = token_entropies
                extra["token_margins"] = token_margins
                extra["confidence"] = {
                    "raw_next_token_confidence": token_confidences[0] if token_confidences else None,
                    "entropy": token_entropies[0] if token_entropies else None,
                    "margin": token_margins[0] if token_margins else None,
                    "mean_token_confidence": _mean(token_confidences),
                    "mean_token_entropy": _mean(token_entropies),
                    "mean_token_margin": _mean(token_margins),
                    "first_token": first_info or {},
                }
            if capture_token_logprobs:
                extra.update(_token_probability_payload(generated_token_logprobs))

        return StepOutput(
            text=text,
            token_ids=visible_token_ids,
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

    def score_suffix_pdi(
        self,
        problem_text: str,
        prefix_text: str,
        suffix_text: str,
    ) -> dict[str, Any]:
        import torch

        self.load()
        rendered_prefix = self.render(problem_text, prefix_text)
        rendered_full = self.render(problem_text, prefix_text + suffix_text)
        if not rendered_full.startswith(rendered_prefix):
            raise RuntimeError("Rendered full prompt does not extend the rendered prefix.")

        generation_budget_for_rendered(rendered_full, self, self.runtime, 1)
        boundary = len(rendered_prefix)
        start = time.time()

        encoded = None
        offsets = None
        try:
            encoded = self.tokenizer(
                rendered_full,
                return_tensors="pt",
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
        except Exception:
            encoded = self.tokenizer(rendered_full, return_tensors="pt", add_special_tokens=False)

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if offsets is not None:
            target_positions = [
                idx
                for idx, (_start, end) in enumerate(offsets)
                if idx > 0 and int(end) > boundary
            ]
        else:
            prefix_ids = self.tokenizer(rendered_prefix, add_special_tokens=False)["input_ids"]
            target_positions = list(range(max(1, len(prefix_ids)), int(input_ids.shape[-1])))

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits[0]
        logprobs: list[float] = []
        target_token_ids: list[int] = []
        target_token_offsets: list[list[int]] = []
        for pos in target_positions:
            token_id = int(input_ids[0, pos].detach().cpu())
            target_token_ids.append(token_id)
            if offsets is not None:
                target_token_offsets.append([int(offsets[pos][0]), int(offsets[pos][1])])
            values = logits[pos - 1].float()
            logprob = values[token_id] - torch.logsumexp(values, dim=-1)
            logprobs.append(float(logprob.detach().cpu()))

        wall_time = time.time() - start
        pdi = -sum(logprobs) / len(logprobs) if logprobs else float("inf")
        try:
            target_tokens = [self.decode([token_id]) for token_id in target_token_ids]
        except Exception:
            target_tokens = []
        return {
            "pdi": float(pdi),
            "token_count": len(logprobs),
            "logprobs": logprobs,
            "token_ids": target_token_ids,
            "tokens": target_tokens,
            "token_offsets": target_token_offsets,
            "prompt_tokens": int(input_ids.shape[-1]),
            "wall_time": wall_time,
        }

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
            print(
                f"[sarr] loading local vLLM engine: {self.cfg.model_path} device={self.cfg.device}",
                file=sys.stderr,
                flush=True,
            )
            with _cuda_visible_devices(self.cfg.device):
                self.llm = LLM(
                    model=self.cfg.model_path,
                    tokenizer=self.cfg.tokenizer_path or self.cfg.model_path,
                    **kwargs,
                )
            print("[sarr] local vLLM engine loaded", file=sys.stderr, flush=True)
        elif self._openai_client is None and self.cfg.backend == "openai":
            if not self.cfg.api_base_url:
                raise RuntimeError("llm.backend='openai' requires llm.api_base_url.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("The openai package is required for OpenAI-compatible vLLM endpoints.") from exc
            print(
                f"[sarr] using OpenAI-compatible LLM endpoint: {self.cfg.api_base_url} model={self.cfg.api_model or self.cfg.model_path}",
                file=sys.stderr,
                flush=True,
            )
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
        capture_token_logprobs: bool = False,
        topk_logprobs: int = 1,
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
                capture_token_logprobs=capture_token_logprobs,
                topk_logprobs=topk_logprobs,
            )
        else:
            output = self._generate_vllm(
                rendered,
                max_tokens=max_tokens,
                stop=stop_delimiters,
                include_stop_str_in_output=include_stop_str_in_output,
                capture_token_logprobs=capture_token_logprobs,
                topk_logprobs=topk_logprobs,
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
        capture_token_logprobs: bool = False,
        topk_logprobs: int = 1,
    ) -> StepOutput:
        return self.generate_text(
            problem_text,
            assistant_prefix_text,
            max_new_tokens=max_new_tokens,
            stop_delimiters=stop_delimiters,
            include_stop_str_in_output=True,
            capture_token_logprobs=capture_token_logprobs,
            topk_logprobs=topk_logprobs,
        )

    def _generate_openai(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None,
        include_stop_str_in_output: bool,
        capture_token_logprobs: bool,
        topk_logprobs: int,
    ) -> StepOutput:
        kwargs: dict[str, Any] = {
            "model": self.cfg.api_model or self.cfg.model_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if capture_token_logprobs:
            kwargs["logprobs"] = max(1, int(topk_logprobs))
        if stop:
            kwargs["stop"] = stop
            kwargs["extra_body"] = {"include_stop_str_in_output": include_stop_str_in_output}
        response = self._openai_client.completions.create(**kwargs)
        choices = sorted(list(getattr(response, "choices", []) or []), key=lambda c: getattr(c, "index", 0))
        choice = choices[0] if choices else SimpleNamespace(text="", finish_reason="")
        text = str(getattr(choice, "text", "") or "")
        finish = str(getattr(choice, "finish_reason", "") or "")
        token_ids = self.encode(text)
        extra: dict[str, Any] = {}
        if capture_token_logprobs:
            logprobs_obj = getattr(choice, "logprobs", None)
            tokens = list(getattr(logprobs_obj, "tokens", None) or []) if logprobs_obj is not None else []
            token_logprobs = (
                list(getattr(logprobs_obj, "token_logprobs", None) or []) if logprobs_obj is not None else []
            )
            if tokens:
                token_ids = [self._token_id_for_openai_token(str(token)) for token in tokens]
            generated_token_logprobs = [
                parsed
                for value in token_logprobs
                for parsed in [_logprob_value(value)]
                if parsed is not None
            ]
            extra.update(_token_probability_payload(generated_token_logprobs))
        return StepOutput(
            text=text,
            token_ids=token_ids,
            finish_reason=finish,
            extra=extra,
        )

    def _generate_vllm(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None,
        include_stop_str_in_output: bool,
        capture_token_logprobs: bool,
        topk_logprobs: int,
    ) -> StepOutput:
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stop": stop,
            "include_stop_str_in_output": include_stop_str_in_output,
        }
        if capture_token_logprobs:
            sampling_kwargs["logprobs"] = max(1, int(topk_logprobs))
        sampling = self._sampling_params(**sampling_kwargs)
        out = self.llm.generate(prompt, sampling)[0]
        completion = out.outputs[0]
        text = str(getattr(completion, "text", "") or "")
        token_ids = [int(x) for x in (getattr(completion, "token_ids", []) or [])]
        if not token_ids:
            token_ids = self.encode(text)
        finish = str(getattr(completion, "finish_reason", "") or "")
        extra: dict[str, Any] = {}
        if capture_token_logprobs:
            generated_token_logprobs = self._generated_logprobs_from_vllm_completion(completion, token_ids)
            extra.update(_token_probability_payload(generated_token_logprobs))
        return StepOutput(text=text, token_ids=token_ids, finish_reason=finish, extra=extra)

    def _token_id_for_openai_token(self, token: str) -> int:
        token_ids = self.encode(token)
        return int(token_ids[0]) if token_ids else 0

    def _generated_logprobs_from_vllm_completion(self, completion: Any, token_ids: list[int]) -> list[float]:
        logprob_steps = list(getattr(completion, "logprobs", None) or [])
        generated: list[float] = []
        for idx, token_id in enumerate(token_ids):
            if idx >= len(logprob_steps):
                break
            step = logprob_steps[idx] or {}
            value = None
            if isinstance(step, dict):
                value = step.get(token_id)
                if value is None:
                    value = step.get(str(token_id))
            parsed = _logprob_value(value)
            if parsed is not None:
                generated.append(parsed)
        return generated

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


def build_llm(cfg: ModelRuntimeConfig, runtime: RuntimeConfig) -> CompletionEngine | LocalTransformersSLM:
    if cfg.backend == "transformers":
        return LocalTransformersSLM(cfg=cfg, runtime=runtime)
    return CompletionEngine(cfg=cfg, runtime=runtime, name="llm")
