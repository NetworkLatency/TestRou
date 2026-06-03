from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


FINISH_MARKERS = ("boxed", "Answer:", "ANSWER:")


@dataclass(frozen=True)
class EndpointConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"


@dataclass(frozen=True)
class SpecReasonHyperparams:
    score_threshold: float = 7.0
    score_method: str = "greedy"
    token_budget: int = 16384
    first_n_steps_base_model: int = 0
    step_max_tokens: int = 512
    stop_token: str = "\n\n"
    generation_temperature: float = 0.6
    generation_top_p: float = 0.95
    score_temperature: float = 0.0
    score_max_tokens: int = 1
    score_top_logprobs: int = 10

    def __post_init__(self) -> None:
        if self.score_method not in {"greedy", "average"}:
            raise ValueError("score_method must be 'greedy' or 'average'.")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive.")
        if self.step_max_tokens < 1:
            raise ValueError("step_max_tokens must be positive.")
        if self.score_max_tokens != 1:
            raise ValueError("SpecReason scoring expects score_max_tokens=1.")


def parse_endpoints(data: dict[str, Any]) -> dict[str, EndpointConfig]:
    endpoints: dict[str, EndpointConfig] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Endpoint {name!r} must be an object.")
        endpoints[name] = EndpointConfig(
            model=str(raw["model"]),
            base_url=str(raw["base_url"]),
            api_key=str(raw.get("api_key") or "EMPTY"),
        )
    return endpoints


def usage_tokens(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def first_choice_text(response: Any) -> str:
    choices = list(getattr(response, "choices", []) or [])
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    return str(getattr(message, "content", "") or "")


def first_token_score_logprobs(response: Any) -> tuple[str, dict[str, float]]:
    choices = list(getattr(response, "choices", []) or [])
    if not choices:
        raise ValueError("Scoring response has no choices.")
    logprobs_obj = getattr(choices[0], "logprobs", None)
    content = list(getattr(logprobs_obj, "content", []) or []) if logprobs_obj is not None else []
    if len(content) != 1:
        raise ValueError(f"Expected exactly one scoring token, got {len(content)}.")
    first = content[0]
    token = str(getattr(first, "token", "") or "")
    top = list(getattr(first, "top_logprobs", []) or [])
    token_logprobs = {
        str(getattr(item, "token", "") or ""): float(getattr(item, "logprob"))
        for item in top
        if getattr(item, "logprob", None) is not None
    }
    return token, token_logprobs


def process_score_response(response: Any, method: str, *, temp: float = 1.0) -> tuple[float, dict[str, Any]]:
    token, raw_logprobs = first_token_score_logprobs(response)
    digit_logprobs = {token: logprob for token, logprob in raw_logprobs.items() if str(token).isdigit()}
    if method == "greedy":
        score = float(int(token)) if token.isdigit() else 0.0
        return score, {
            "score_token": token,
            "raw_top_logprobs": raw_logprobs,
            "digit_top_logprobs": digit_logprobs,
        }
    if method == "average":
        probs = {tok: math.exp(logprob / temp) for tok, logprob in digit_logprobs.items()}
        total = sum(probs.values())
        if total <= 0:
            return 0.0, {
                "score_token": token,
                "raw_top_logprobs": raw_logprobs,
                "digit_top_logprobs": digit_logprobs,
            }
        probs = {tok: prob / total for tok, prob in probs.items()}
        for idx in range(10):
            probs.setdefault(str(idx), 0.0)
        score = sum(int(tok) * prob for tok, prob in probs.items())
        return float(score), {
            "score_token": token,
            "raw_top_logprobs": raw_logprobs,
            "digit_top_logprobs": digit_logprobs,
            "digit_probs": probs,
        }
    raise ValueError(f"Unsupported score method: {method}")


def build_step_messages(problem_prompt: str, steps_so_far: list[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not steps_so_far:
        return [{"role": "user", "content": problem_prompt}], {"add_generation_prompt": True}
    steps = "\n\n".join(steps_so_far) + "\n\n"
    return (
        [
            {"role": "user", "content": problem_prompt},
            {"role": "assistant", "content": f"<think>{steps}"},
        ],
        {"add_generation_prompt": False, "continue_final_message": True},
    )


def build_score_messages(problem_prompt: str, steps_with_candidate: list[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    steps = "\n\n".join(steps_with_candidate) + "\n\n"
    return (
        [
            {"role": "user", "content": problem_prompt},
            {"role": "assistant", "content": f"<think>{steps}"},
            {
                "role": "user",
                "content": (
                    "Evaluate the last reasoning step solely based on factual correctness and logical validity. "
                    "Ignore style, phrasing, or overall usefulness--only judge whether the step is objectively "
                    "correct and logically follows from prior steps. Assign a score from 0 to 9."
                ),
            },
            {"role": "assistant", "content": "<think>I think the quality score is: "},
        ],
        {"add_generation_prompt": False, "continue_final_message": True},
    )


class SpecReasonRouter:
    def __init__(
        self,
        *,
        endpoints: dict[str, EndpointConfig],
        base_model_key: str = "32b",
        small_model_key: str = "1.5b",
        hyperparams: SpecReasonHyperparams | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required to run SpecReason against OpenAI-compatible vLLM endpoints."
            ) from exc
        self.endpoints = endpoints
        self.base_model_key = base_model_key
        self.small_model_key = small_model_key
        self.hyperparams = hyperparams or SpecReasonHyperparams()
        for key in (base_model_key, small_model_key):
            if key not in endpoints:
                raise ValueError(f"Missing endpoint config for model key {key!r}.")
        self.clients = {
            key: OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url)
            for key, endpoint in endpoints.items()
        }

    def model_name(self, key: str) -> str:
        return self.endpoints[key].model

    def generate_new_step(self, problem_prompt: str, steps_so_far: list[str], model_key: str) -> dict[str, Any]:
        hp = self.hyperparams
        messages, extra_body = build_step_messages(problem_prompt, steps_so_far)
        start = time.perf_counter()
        response = self.clients[model_key].chat.completions.create(
            model=self.model_name(model_key),
            messages=messages,
            temperature=hp.generation_temperature,
            top_p=hp.generation_top_p,
            max_tokens=hp.step_max_tokens,
            stop=[hp.stop_token],
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - start
        text = first_choice_text(response)
        return {
            "text": text,
            "finished": any(marker in text for marker in FINISH_MARKERS),
            "usage": usage_tokens(response),
            "wall_time": elapsed,
        }

    def score_step(self, problem_prompt: str, steps_with_candidate: list[str]) -> dict[str, Any]:
        hp = self.hyperparams
        messages, extra_body = build_score_messages(problem_prompt, steps_with_candidate)
        start = time.perf_counter()
        response = self.clients[self.base_model_key].chat.completions.create(
            model=self.model_name(self.base_model_key),
            messages=messages,
            temperature=hp.score_temperature,
            max_tokens=hp.score_max_tokens,
            logprobs=True,
            top_logprobs=hp.score_top_logprobs,
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - start
        score, score_payload = process_score_response(response, hp.score_method)
        return {
            "score": score,
            "justification": first_choice_text(response),
            "usage": usage_tokens(response),
            "wall_time": elapsed,
            **score_payload,
        }

    def run(self, *, problem_prompt: str, dataset_name: str, problem_id: Any, repeat_id: int = 0) -> list[dict[str, Any]]:
        del dataset_name
        hp = self.hyperparams
        steps_so_far: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        step_id = 0

        while True:
            warning_flag = False
            step_time = 0.0
            if step_id < hp.first_n_steps_base_model:
                base = self.generate_new_step(problem_prompt, steps_so_far, self.base_model_key)
                small = None
                score = None
                score_payload = None
                chosen = base
                selected_model = "base"
                step_time += float(base["wall_time"])
            else:
                small = self.generate_new_step(problem_prompt, steps_so_far, self.small_model_key)
                step_time += float(small["wall_time"])
                score_payload = self.score_step(problem_prompt, steps_so_far + [small["text"]])
                step_time += float(score_payload["wall_time"])
                score = float(score_payload["score"])
                if score >= hp.score_threshold:
                    base = None
                    chosen = small
                    selected_model = "small"
                else:
                    base = self.generate_new_step(problem_prompt, steps_so_far, self.base_model_key)
                    step_time += float(base["wall_time"])
                    chosen = base
                    selected_model = "base"

            step_str = str(chosen["text"])
            if "</think>" in step_str and not any(marker in step_str for marker in FINISH_MARKERS):
                step_str = step_str.replace("</think>", "")
                warning_flag = True

            steps_so_far.append(step_str)
            final_usage = chosen["usage"]
            small_usage = small["usage"] if small is not None else None
            base_usage = base["usage"] if base is not None else None
            score_usage = score_payload["usage"] if score_payload is not None else None
            metadata = {
                "problem_id": problem_id,
                "repeat_id": repeat_id,
                "step_id": step_id,
                "step_str": step_str,
                "selected_model": selected_model,
                "small_model_step": small["text"] if small is not None else None,
                "num_output_tokens_small": small_usage["completion_tokens"] if small_usage is not None else None,
                "num_prompt_tokens_small": small_usage["prompt_tokens"] if small_usage is not None else None,
                "small_model_time": small["wall_time"] if small is not None else None,
                "score": score,
                "score_token": score_payload["score_token"] if score_payload is not None else None,
                "score_digit_top_logprobs": score_payload["digit_top_logprobs"] if score_payload is not None else None,
                "score_digit_probs": score_payload.get("digit_probs") if score_payload is not None else None,
                "eval_time": score_payload["wall_time"] if score_payload is not None else None,
                "num_score_prompt_tokens": score_usage["prompt_tokens"] if score_usage is not None else None,
                "num_score_output_tokens": score_usage["completion_tokens"] if score_usage is not None else None,
                "base_model_step": base["text"] if base is not None else None,
                "num_output_tokens_base": base_usage["completion_tokens"] if base_usage is not None else None,
                "num_prompt_tokens_base": base_usage["prompt_tokens"] if base_usage is not None else None,
                "base_model_time": base["wall_time"] if base is not None else None,
                "final_num_output_tokens": int(final_usage["completion_tokens"]),
                "step_time": step_time,
                "justification": score_payload["justification"] if score_payload is not None else None,
            }
            if warning_flag:
                metadata["warning"] = "step_str had a </think>"
            metadata_list.append(metadata)
            step_id += 1

            finished = bool(chosen["finished"])
            if len(steps_so_far) > 2:
                finished = finished or steps_so_far[-1] == steps_so_far[-2]
            generated_tokens = sum(int(item.get("final_num_output_tokens") or 0) for item in metadata_list)
            if finished or generated_tokens >= hp.token_budget:
                metadata_list[-1]["stop_reason"] = "budget" if generated_tokens >= hp.token_budget else "finished"
                break

        return metadata_list


def extract_answer_text(metadata_list: list[dict[str, Any]]) -> str | None:
    if not metadata_list:
        return None
    return str(metadata_list[-1].get("step_str") or "")


def route_stats(metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(item["score"]) for item in metadata_list if item.get("score") is not None]
    small_decode = sum(int(item.get("num_output_tokens_small") or 0) for item in metadata_list)
    small_prefill = sum(int(item.get("num_prompt_tokens_small") or 0) for item in metadata_list)
    score_decode = sum(int(item.get("num_score_output_tokens") or 0) for item in metadata_list)
    score_prefill = sum(int(item.get("num_score_prompt_tokens") or 0) for item in metadata_list)
    base_decode = sum(int(item.get("num_output_tokens_base") or 0) for item in metadata_list)
    base_prefill = sum(int(item.get("num_prompt_tokens_base") or 0) for item in metadata_list)
    stop_reason = None
    for item in reversed(metadata_list):
        if item.get("stop_reason"):
            stop_reason = item["stop_reason"]
            break
    return {
        "step_count": len(metadata_list),
        "small_accept_count": sum(1 for item in metadata_list if item.get("selected_model") == "small"),
        "base_fallback_count": sum(1 for item in metadata_list if item.get("selected_model") == "base"),
        "score_call_count": sum(1 for item in metadata_list if item.get("score") is not None),
        "avg_score": (sum(scores) / len(scores)) if scores else None,
        "max_score": max(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "small_decode_tokens": small_decode,
        "small_prefill_tokens": small_prefill,
        "score_decode_tokens": score_decode,
        "score_prefill_tokens": score_prefill,
        "base_decode_tokens": base_decode,
        "base_prefill_tokens": base_prefill,
        "total_decode_tokens": small_decode + score_decode + base_decode,
        "total_prefill_tokens": small_prefill + score_prefill + base_prefill,
        "route_sequence": ",".join(str(item.get("selected_model")) for item in metadata_list),
        "stop_reason": stop_reason or "finished",
    }
