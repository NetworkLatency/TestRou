from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

CLOSE_THINK_TAG = "</think>"
STEP_STOP_TOKEN = "\n\n"


@dataclass(frozen=True)
class EndpointConfig:
    model: str
    base_url: str
    api_key: str = "glimp_router"


@dataclass(frozen=True)
class GlimpRouterHyperparams:
    score_method: str = "first_token_entropy"
    score_threshold: float = 1.0
    token_budget: int = 14336
    step_max_tokens: int = 512
    answer_max_tokens: int = 2048
    step_stop_token: str = STEP_STOP_TOKEN
    generation_temperature: float = 0.6
    generation_top_p: float = 0.95
    score_temperature: float = 0.0
    top_logprobs: int = 20
    first_n_steps_base_model: int = 0

    def __post_init__(self) -> None:
        if self.score_method != "first_token_entropy":
            raise ValueError("Only score_method='first_token_entropy' is implemented for GlimpRouter.")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive.")
        if self.step_max_tokens < 1:
            raise ValueError("step_max_tokens must be positive.")
        if self.answer_max_tokens < 1:
            raise ValueError("answer_max_tokens must be positive.")
        if self.top_logprobs < 1:
            raise ValueError("top_logprobs must be positive.")


def parse_endpoints(data: dict[str, Any]) -> dict[str, EndpointConfig]:
    endpoints: dict[str, EndpointConfig] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Endpoint {name!r} must be an object.")
        endpoints[name] = EndpointConfig(
            model=str(raw["model"]),
            base_url=str(raw["base_url"]),
            api_key=str(raw.get("api_key") or "glimp_router"),
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


def first_token_top_logprobs(response: Any) -> tuple[str, dict[str, float]]:
    choices = list(getattr(response, "choices", []) or [])
    if not choices:
        raise ValueError("Scoring response has no choices.")
    logprobs_obj = getattr(choices[0], "logprobs", None)
    content = list(getattr(logprobs_obj, "content", []) or []) if logprobs_obj is not None else []
    if len(content) != 1:
        raise ValueError(f"Expected exactly one generated scoring token, got {len(content)}.")
    first = content[0]
    token = str(getattr(first, "token", "") or "")
    top = list(getattr(first, "top_logprobs", []) or [])
    token_logprobs = {
        str(getattr(item, "token", "") or ""): float(getattr(item, "logprob"))
        for item in top
        if getattr(item, "logprob", None) is not None
    }
    if not token_logprobs:
        raise ValueError("Scoring response did not return top_logprobs.")
    return token, token_logprobs


def entropy_from_logprobs(token_logprobs: dict[str, float]) -> float:
    probs = {token: math.exp(logprob) for token, logprob in token_logprobs.items()}
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Cannot normalize empty or zero-probability top_logprobs.")
    normalized = [prob / total for prob in probs.values()]
    return -sum(prob * math.log(prob) for prob in normalized)


def build_step_messages(problem_prompt: str, steps_so_far: list[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not steps_so_far:
        return [{"role": "user", "content": problem_prompt}], {"add_generation_prompt": True}
    steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
    return (
        [
            {"role": "user", "content": problem_prompt},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
        ],
        {"add_generation_prompt": False, "continue_final_message": True},
    )


def build_answer_messages(problem_prompt: str, steps_so_far: list[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    steps_so_far_str = "\n\n".join(steps_so_far)
    if CLOSE_THINK_TAG in steps_so_far_str:
        steps_so_far_str = steps_so_far_str.split(CLOSE_THINK_TAG, 1)[0]
    return (
        [
            {"role": "user", "content": problem_prompt},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}\n</think>\n\n"},
        ],
        {"add_generation_prompt": False, "continue_final_message": True},
    )


class GlimpRouter:
    def __init__(
        self,
        *,
        endpoints: dict[str, EndpointConfig],
        model_size: str = "32b",
        small_model_size: str = "4b",
        hyperparams: GlimpRouterHyperparams | None = None,
    ) -> None:
        self.endpoints = endpoints
        self.model_size = model_size
        self.small_model_size = small_model_size
        self.hyperparams = hyperparams or GlimpRouterHyperparams()
        for key in (model_size, small_model_size):
            if key not in endpoints:
                raise ValueError(f"Missing endpoint config for model key {key!r}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required to run GlimpRouter against OpenAI-compatible vLLM endpoints. "
                "Install the repository requirements on the remote Linux environment."
            ) from exc
        self.clients = {
            key: OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url)
            for key, endpoint in endpoints.items()
        }

    def model_name(self, model_key: str) -> str:
        return self.endpoints[model_key].model

    def generate_new_step(self, problem_prompt: str, steps_so_far: list[str], model_key: str) -> dict[str, Any]:
        hp = self.hyperparams
        messages, extra_body = build_step_messages(problem_prompt, steps_so_far)
        start = time.time()
        response = self.clients[model_key].chat.completions.create(
            model=self.model_name(model_key),
            messages=messages,
            temperature=hp.generation_temperature,
            top_p=hp.generation_top_p,
            max_tokens=hp.step_max_tokens,
            stop=[hp.step_stop_token],
            extra_body=extra_body,
        )
        wall_time = time.time() - start
        text = first_choice_text(response)
        tokens = usage_tokens(response)
        return {
            "text": text,
            "finished": CLOSE_THINK_TAG in text,
            "usage": tokens,
            "wall_time": wall_time,
        }

    def generate_answer(self, problem_prompt: str, steps_so_far: list[str], model_key: str) -> dict[str, Any]:
        hp = self.hyperparams
        messages, extra_body = build_answer_messages(problem_prompt, steps_so_far)
        start = time.time()
        response = self.clients[model_key].chat.completions.create(
            model=self.model_name(model_key),
            messages=messages,
            temperature=hp.generation_temperature,
            top_p=hp.generation_top_p,
            max_tokens=hp.answer_max_tokens,
            extra_body=extra_body,
        )
        wall_time = time.time() - start
        text = first_choice_text(response)
        tokens = usage_tokens(response)
        finished = any(marker in text for marker in ["boxed", "Answer:", "ANSWER:"])
        return {
            "text": text,
            "finished": finished,
            "usage": tokens,
            "wall_time": wall_time,
        }

    def score_first_token_entropy(self, problem_prompt: str, steps_so_far: list[str], model_key: str) -> dict[str, Any]:
        hp = self.hyperparams
        messages, extra_body = build_step_messages(problem_prompt, steps_so_far)
        start = time.time()
        response = self.clients[model_key].chat.completions.create(
            model=self.model_name(model_key),
            messages=messages,
            temperature=hp.score_temperature,
            max_tokens=1,
            logprobs=True,
            top_logprobs=hp.top_logprobs,
            extra_body=extra_body,
        )
        wall_time = time.time() - start
        token, token_logprobs = first_token_top_logprobs(response)
        entropy = entropy_from_logprobs(token_logprobs)
        return {
            "score": entropy,
            "first_token": token,
            "top_logprobs": token_logprobs,
            "usage": usage_tokens(response),
            "wall_time": wall_time,
        }

    def route(self, *, problem_prompt: str, dataset_name: str, problem_id: Any, repeat_id: int = 0) -> list[dict[str, Any]]:
        del dataset_name
        hp = self.hyperparams
        steps_so_far: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        step_id = 0

        while True:
            if step_id < hp.first_n_steps_base_model:
                base = self.generate_new_step(problem_prompt, steps_so_far, self.model_size)
                small = None
                score_payload = None
                selected_model = "base"
                step_str = base["text"]
                finished = bool(base["finished"])
            else:
                score_payload = self.score_first_token_entropy(
                    problem_prompt,
                    steps_so_far,
                    self.small_model_size,
                )
                if score_payload["score"] >= hp.score_threshold:
                    base = self.generate_new_step(problem_prompt, steps_so_far, self.model_size)
                    small = None
                    selected_model = "base"
                    step_str = base["text"]
                    finished = bool(base["finished"])
                else:
                    small = self.generate_new_step(problem_prompt, steps_so_far, self.small_model_size)
                    base = None
                    selected_model = "small"
                    step_str = small["text"]
                    finished = bool(small["finished"])

            steps_so_far.append(step_str)
            base_usage = base["usage"] if base is not None else None
            small_usage = small["usage"] if small is not None else None
            score_usage = score_payload["usage"] if score_payload is not None else None
            final_num_output_tokens = (
                base_usage["completion_tokens"]
                if base_usage is not None
                else small_usage["completion_tokens"]
                if small_usage is not None
                else 0
            )
            metadata = {
                "problem_id": problem_id,
                "repeat_id": repeat_id,
                "step_id": step_id,
                "step_str": step_str,
                "selected_model": selected_model,
                "small_model_step": small["text"] if small is not None else None,
                "num_output_tokens_small": small_usage["completion_tokens"] if small_usage is not None else None,
                "num_prompt_tokens_small": small_usage["prompt_tokens"] if small_usage is not None else None,
                "score": score_payload["score"] if score_payload is not None else None,
                "score_first_token": score_payload["first_token"] if score_payload is not None else None,
                "score_top_logprobs": score_payload["top_logprobs"] if score_payload is not None else None,
                "score_num_output_tokens": score_usage["completion_tokens"] if score_usage is not None else None,
                "score_num_prompt_tokens": score_usage["prompt_tokens"] if score_usage is not None else None,
                "score_wall_time": score_payload["wall_time"] if score_payload is not None else None,
                "base_model_step": base["text"] if base is not None else None,
                "num_output_tokens_base": base_usage["completion_tokens"] if base_usage is not None else None,
                "num_prompt_tokens_base": base_usage["prompt_tokens"] if base_usage is not None else None,
                "final_num_output_tokens": final_num_output_tokens,
                "generation_wall_time": (base or small)["wall_time"] if (base or small) is not None else None,
                "justification": score_payload["first_token"] if score_payload is not None else None,
            }
            metadata_list.append(metadata)
            step_id += 1

            if len(steps_so_far) > 2:
                finished = finished or steps_so_far[-1] == steps_so_far[-2]

            generated_tokens = sum(int(item.get("final_num_output_tokens") or 0) for item in metadata_list)
            if finished or generated_tokens >= hp.token_budget:
                metadata_list[-1]["stop_reason"] = "budget" if generated_tokens >= hp.token_budget else "finished"
                break

        answer = self.generate_answer(problem_prompt, steps_so_far, self.model_size)
        steps_so_far.append(answer["text"])
        usage = answer["usage"]
        metadata_list.append(
            {
                "problem_id": problem_id,
                "repeat_id": repeat_id,
                "step_id": step_id,
                "step_str": answer["text"],
                "selected_model": "base_final_answer",
                "small_model_step": None,
                "num_output_tokens_small": None,
                "num_prompt_tokens_small": None,
                "score": None,
                "score_first_token": None,
                "score_top_logprobs": None,
                "score_num_output_tokens": None,
                "score_num_prompt_tokens": None,
                "score_wall_time": None,
                "base_model_step": answer["text"],
                "num_output_tokens_base": usage["completion_tokens"],
                "num_prompt_tokens_base": usage["prompt_tokens"],
                "final_num_output_tokens": usage["completion_tokens"],
                "generation_wall_time": answer["wall_time"],
                "justification": None,
                "answer": answer["finished"],
            }
        )
        return metadata_list


def extract_answer_text(metadata_list: list[dict[str, Any]]) -> str | None:
    if not metadata_list:
        return None
    return str(metadata_list[-1].get("step_str") or "")


def route_stats(metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
    routing_steps = [item for item in metadata_list if item.get("selected_model") != "base_final_answer"]
    scores = [float(item["score"]) for item in routing_steps if item.get("score") is not None]
    small_generation_decode = sum(int(item.get("num_output_tokens_small") or 0) for item in metadata_list)
    small_generation_prefill = sum(int(item.get("num_prompt_tokens_small") or 0) for item in metadata_list)
    small_score_decode = sum(int(item.get("score_num_output_tokens") or 0) for item in metadata_list)
    small_score_prefill = sum(int(item.get("score_num_prompt_tokens") or 0) for item in metadata_list)
    base_decode = sum(int(item.get("num_output_tokens_base") or 0) for item in metadata_list)
    base_prefill = sum(int(item.get("num_prompt_tokens_base") or 0) for item in metadata_list)
    stop_reason = None
    for item in reversed(metadata_list):
        if item.get("stop_reason"):
            stop_reason = item.get("stop_reason")
            break
    return {
        "routing_step_count": len(routing_steps),
        "small_route_count": sum(1 for item in routing_steps if item.get("selected_model") == "small"),
        "base_route_count": sum(1 for item in routing_steps if item.get("selected_model") == "base"),
        "score_call_count": sum(1 for item in routing_steps if item.get("score") is not None),
        "avg_score": (sum(scores) / len(scores)) if scores else None,
        "max_score": max(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "small_generation_decode_tokens": small_generation_decode,
        "small_generation_prefill_tokens": small_generation_prefill,
        "small_score_decode_tokens": small_score_decode,
        "small_score_prefill_tokens": small_score_prefill,
        "small_total_decode_tokens": small_generation_decode + small_score_decode,
        "small_total_prefill_tokens": small_generation_prefill + small_score_prefill,
        "base_decode_tokens": base_decode,
        "base_prefill_tokens": base_prefill,
        "total_decode_tokens": small_generation_decode + small_score_decode + base_decode,
        "total_prefill_tokens": small_generation_prefill + small_score_prefill + base_prefill,
        "stop_reason": stop_reason or "finished",
        "route_sequence": ",".join(str(item.get("selected_model")) for item in routing_steps),
    }
