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
    timeout: float = 3600.0


@dataclass(frozen=True)
class RSDHyperparams:
    prm_threshold: float = 0.7
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens_per_call: int = 16384
    step_max_tokens: int = 512
    step_word: str = "\n\n"
    max_steps: int = 100
    patience: int = 5
    score_temperature: float = 0.0
    score_max_tokens: int = 1
    score_top_logprobs: int = 10
    enable_thinking: bool = True

    def __post_init__(self) -> None:
        if self.prm_threshold < 0:
            raise ValueError("prm_threshold must be non-negative.")
        if self.max_tokens_per_call < 1:
            raise ValueError("max_tokens_per_call must be positive.")
        if self.step_max_tokens < 1:
            raise ValueError("step_max_tokens must be positive.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if self.patience < 1:
            raise ValueError("patience must be positive.")
        if self.temperature == 0.0 and self.top_p != 1.0:
            raise ValueError("RSD follows vLLM greedy decoding: top_p must be 1 when temperature is 0.")


def parse_endpoints(data: dict[str, Any]) -> dict[str, EndpointConfig]:
    endpoints: dict[str, EndpointConfig] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Endpoint {name!r} must be an object.")
        model = str(raw.get("model") or raw.get("served_model_name") or "")
        if not model:
            raise ValueError(f"Endpoint {name!r} must define model.")
        endpoints[name] = EndpointConfig(
            model=model,
            base_url=str(raw["base_url"]),
            api_key=str(raw.get("api_key") or "EMPTY"),
            timeout=float(raw.get("timeout") or 3600.0),
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


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def chat_choice(response: Any) -> Any:
    choices = list(getattr(response, "choices", []) or [])
    if not choices:
        raise ValueError("Chat completion response has no choices.")
    return sorted(choices, key=lambda choice: int(getattr(choice, "index", 0) or 0))[0]


def choice_text(choice: Any) -> str:
    message = getattr(choice, "message", None)
    return str(getattr(message, "content", "") or "") if message is not None else ""


def choice_stopped_by_step(choice: Any) -> bool:
    stop_reason = getattr(choice, "stop_reason", None)
    finish_reason = getattr(choice, "finish_reason", None)
    return stop_reason is not None or finish_reason == "stop"


def build_step_messages(problem_prompt: str, accepted_response: str, hp: RSDHyperparams) -> tuple[list[dict[str, str]], dict[str, Any]]:
    extra_body: dict[str, Any] = {
        "add_generation_prompt": not accepted_response,
        "continue_final_message": bool(accepted_response),
        "chat_template_kwargs": {"enable_thinking": hp.enable_thinking},
    }
    if not accepted_response:
        return [{"role": "user", "content": problem_prompt}], extra_body
    return (
        [
            {"role": "user", "content": problem_prompt},
            {"role": "assistant", "content": f"<think>{accepted_response}"},
        ],
        extra_body,
    )


def build_score_messages(problem_prompt: str, accepted_response: str, draft_text: str, hp: RSDHyperparams) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidate_response = accepted_response + draft_text
    return (
        [
            {"role": "user", "content": problem_prompt},
            {"role": "assistant", "content": f"<think>{candidate_response}"},
            {
                "role": "user",
                "content": (
                    "Judge only the last reasoning step. Return a single digit from 0 to 9, "
                    "where 9 means the step is correct and useful, and 0 means it is wrong."
                ),
            },
            {"role": "assistant", "content": "Score: "},
        ],
        {
            "add_generation_prompt": False,
            "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": hp.enable_thinking},
        },
    )


def first_token_score(response: Any) -> tuple[float, dict[str, Any]]:
    choice = chat_choice(response)
    text = choice_text(choice).strip()
    token = text[:1]
    logprobs_obj = getattr(choice, "logprobs", None)
    content = list(getattr(logprobs_obj, "content", []) or []) if logprobs_obj is not None else []
    raw_top_logprobs: dict[str, float] = {}
    if content:
        first = content[0]
        token = str(getattr(first, "token", "") or token)
        for item in list(getattr(first, "top_logprobs", []) or []):
            item_token = str(getattr(item, "token", "") or "")
            logprob = getattr(item, "logprob", None)
            if logprob is not None:
                raw_top_logprobs[item_token] = float(logprob)
    digit = token.strip()[:1]
    score = float(int(digit)) if digit.isdigit() else 0.0
    return score / 9.0, {"score_token": token, "score_text": text, "raw_top_logprobs": raw_top_logprobs}


class RSDRouter:
    def __init__(
        self,
        *,
        endpoints: dict[str, EndpointConfig],
        hyperparams: RSDHyperparams | None = None,
    ) -> None:
        self.endpoints = endpoints
        self.hyperparams = hyperparams or RSDHyperparams()
        for key in ("draft", "target"):
            if key not in endpoints:
                raise ValueError(f"Missing endpoint config for {key!r}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required to run RSD against OpenAI-compatible vLLM endpoints."
            ) from exc
        self.clients = {
            key: OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url, timeout=endpoint.timeout)
            for key, endpoint in endpoints.items()
        }

    def model_name(self, key: str) -> str:
        return self.endpoints[key].model

    def generate_step(self, prompt: str, model_key: str) -> dict[str, Any]:
        hp = self.hyperparams
        messages, extra_body = build_step_messages(prompt, "", hp)
        start = time.perf_counter()
        response = self.clients[model_key].chat.completions.create(
            model=self.model_name(model_key),
            messages=messages,
            temperature=hp.temperature,
            top_p=hp.top_p,
            max_tokens=min(hp.step_max_tokens, hp.max_tokens_per_call),
            stop=[hp.step_word],
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - start
        choice = chat_choice(response)
        return {
            "text": choice_text(choice),
            "stopped_by_step": choice_stopped_by_step(choice),
            "finish_reason": getattr(choice, "finish_reason", None),
            "stop_reason": getattr(choice, "stop_reason", None),
            "usage": usage_tokens(response),
            "wall_time": elapsed,
        }

    def generate_continuation_step(self, *, problem_prompt: str, accepted_response: str, model_key: str, remaining_tokens: int) -> dict[str, Any]:
        hp = self.hyperparams
        messages, extra_body = build_step_messages(problem_prompt, accepted_response, hp)
        start = time.perf_counter()
        response = self.clients[model_key].chat.completions.create(
            model=self.model_name(model_key),
            messages=messages,
            temperature=hp.temperature,
            top_p=hp.top_p,
            max_tokens=max(1, min(hp.step_max_tokens, remaining_tokens)),
            stop=[hp.step_word],
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - start
        choice = chat_choice(response)
        return {
            "text": choice_text(choice),
            "stopped_by_step": choice_stopped_by_step(choice),
            "finish_reason": getattr(choice, "finish_reason", None),
            "stop_reason": getattr(choice, "stop_reason", None),
            "usage": usage_tokens(response),
            "wall_time": elapsed,
        }

    def score_draft_step(self, *, problem_prompt: str, accepted_response: str, draft_text: str) -> dict[str, Any]:
        hp = self.hyperparams
        messages, extra_body = build_score_messages(problem_prompt, accepted_response, draft_text, hp)
        start = time.perf_counter()
        response = self.clients["target"].chat.completions.create(
            model=self.model_name("target"),
            messages=messages,
            temperature=hp.score_temperature,
            max_tokens=hp.score_max_tokens,
            logprobs=True,
            top_logprobs=hp.score_top_logprobs,
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - start
        reward, score_payload = first_token_score(response)
        return {
            "reward": reward,
            "accepted": reward >= hp.prm_threshold,
            "all_step_rewards": [reward],
            "num_prm_input_ids": 0,
            "num_prm_steps": len((accepted_response + draft_text).split(hp.step_word)),
            "usage": usage_tokens(response),
            "wall_time": elapsed,
            **score_payload,
        }

    def should_stop(self, *, chosen: dict[str, Any], step_id: int, num_unchanged: int, total_decode_tokens: int) -> tuple[bool, str | None]:
        hp = self.hyperparams
        if not bool(chosen["stopped_by_step"]):
            return True, "generation_finished_or_length"
        if total_decode_tokens >= hp.max_tokens_per_call:
            return True, "completion_budget"
        if step_id >= hp.max_steps - 1:
            return True, "max_steps"
        if num_unchanged >= hp.patience - 1:
            return True, "patience"
        return False, None

    def run(
        self,
        *,
        problem_prompt: str,
        prm_problem: str,
        dataset_name: str,
        problem_id: Any,
        repeat_id: int = 0,
    ) -> list[dict[str, Any]]:
        del dataset_name
        hp = self.hyperparams
        accepted_steps: list[tuple[str, str]] = []
        metadata_list: list[dict[str, Any]] = []
        num_unchanged = 0

        for step_id in range(hp.max_steps):
            accepted_response = "".join(text for text, _ in accepted_steps)
            total_decode_tokens = sum(
                int(item.get("final_num_output_tokens") or 0) + int(item.get("num_discarded_output_tokens_draft") or 0)
                for item in metadata_list
            )
            remaining_tokens = hp.max_tokens_per_call - total_decode_tokens
            if remaining_tokens <= 0:
                break

            draft = self.generate_continuation_step(
                problem_prompt=problem_prompt,
                accepted_response=accepted_response,
                model_key="draft",
                remaining_tokens=remaining_tokens,
            )
            draft_text = str(draft["text"])
            score_payload = self.score_draft_step(
                problem_prompt=problem_prompt,
                accepted_response=accepted_response,
                draft_text=draft_text,
            )

            discarded_draft_tokens = 0
            target: dict[str, Any] | None = None
            if bool(score_payload["accepted"]):
                chosen = draft
                selected_model = "draft"
            else:
                draft_with_step = draft_text + hp.step_word
                discarded_draft_tokens = int(draft["usage"]["completion_tokens"])
                target = self.generate_continuation_step(
                    problem_prompt=problem_prompt,
                    accepted_response=accepted_response,
                    model_key="target",
                    remaining_tokens=max(1, remaining_tokens - discarded_draft_tokens),
                )
                chosen = target
                selected_model = "target"

            chosen_text = str(chosen["text"])
            response_text = chosen_text + hp.step_word
            accepted_steps.append((response_text, selected_model))
            full_response = "".join(text for text, _ in accepted_steps)
            answer_text = full_response[:-len(hp.step_word)] if full_response.endswith(hp.step_word) else full_response

            selected_decode_tokens = int(chosen["usage"]["completion_tokens"])
            draft_usage = draft["usage"]
            target_usage = target["usage"] if target is not None else None
            prm_usage = score_payload["usage"]
            should_stop, stop_reason = self.should_stop(
                chosen=chosen,
                step_id=step_id,
                num_unchanged=num_unchanged,
                total_decode_tokens=total_decode_tokens + selected_decode_tokens + discarded_draft_tokens,
            )

            metadata = {
                "problem_id": problem_id,
                "repeat_id": repeat_id,
                "step_id": step_id,
                "step_str": chosen_text,
                "response_text_with_step": response_text,
                "full_response_text": answer_text,
                "selected_model": selected_model,
                "draft_model_step": draft_text,
                "target_model_step": target["text"] if target is not None else None,
                "prm_reward": score_payload["reward"],
                "prm_threshold": hp.prm_threshold,
                "draft_accepted": bool(score_payload["accepted"]),
                "all_step_rewards": score_payload["all_step_rewards"],
                "score_token": score_payload.get("score_token"),
                "score_text": score_payload.get("score_text"),
                "score_top_logprobs": score_payload.get("raw_top_logprobs"),
                "num_output_tokens_draft": selected_decode_tokens if selected_model == "draft" else 0,
                "num_discarded_output_tokens_draft": discarded_draft_tokens,
                "num_output_tokens_target": selected_decode_tokens if selected_model == "target" else 0,
                "num_prompt_tokens_draft_api": draft_usage["prompt_tokens"],
                "num_output_tokens_draft_api": draft_usage["completion_tokens"],
                "num_prompt_tokens_target_api": target_usage["prompt_tokens"] if target_usage is not None else None,
                "num_output_tokens_target_api": target_usage["completion_tokens"] if target_usage is not None else None,
                "num_prm_input_ids": score_payload["num_prm_input_ids"],
                "num_prm_steps": score_payload["num_prm_steps"],
                "num_prm_prompt_tokens_api": prm_usage["prompt_tokens"],
                "num_prm_output_tokens_api": prm_usage["completion_tokens"],
                "draft_model_time": draft["wall_time"],
                "target_model_time": target["wall_time"] if target is not None else None,
                "prm_time": score_payload["wall_time"],
                "step_time": float(draft["wall_time"]) + float(score_payload["wall_time"]) + (float(target["wall_time"]) if target is not None else 0.0),
                "draft_finish_reason": draft["finish_reason"],
                "draft_stop_reason": str(draft["stop_reason"]) if draft["stop_reason"] is not None else None,
                "target_finish_reason": target["finish_reason"] if target is not None else None,
                "target_stop_reason": str(target["stop_reason"]) if target is not None and target["stop_reason"] is not None else None,
            }
            if should_stop:
                metadata["stop_reason"] = stop_reason
            metadata_list.append(metadata)
            if should_stop:
                break
            num_unchanged += 1

        return metadata_list


def extract_answer_text(metadata_list: list[dict[str, Any]]) -> str | None:
    if not metadata_list:
        return None
    return str(metadata_list[-1].get("full_response_text") or "")


def route_stats(metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(item["prm_reward"]) for item in metadata_list if item.get("prm_reward") is not None]
    draft_accepted_decode = sum(int(item.get("num_output_tokens_draft") or 0) for item in metadata_list)
    draft_discarded_decode = sum(int(item.get("num_discarded_output_tokens_draft") or 0) for item in metadata_list)
    target_decode = sum(int(item.get("num_output_tokens_target") or 0) for item in metadata_list)
    prm_input_ids = sum(int(item.get("num_prm_input_ids") or 0) for item in metadata_list)
    judge_prompt_tokens = sum(int(item.get("num_prm_prompt_tokens_api") or 0) for item in metadata_list)
    judge_decode_tokens = sum(int(item.get("num_prm_output_tokens_api") or 0) for item in metadata_list)
    draft_wall = sum(float(item.get("draft_model_time") or 0.0) for item in metadata_list)
    target_wall = sum(float(item.get("target_model_time") or 0.0) for item in metadata_list)
    prm_wall = sum(float(item.get("prm_time") or 0.0) for item in metadata_list)
    stop_reason = None
    for item in reversed(metadata_list):
        if item.get("stop_reason"):
            stop_reason = item["stop_reason"]
            break
    total_decode = draft_accepted_decode + draft_discarded_decode + target_decode
    return {
        "step_count": len(metadata_list),
        "draft_accept_count": sum(1 for item in metadata_list if item.get("selected_model") == "draft"),
        "target_fallback_count": sum(1 for item in metadata_list if item.get("selected_model") == "target"),
        "prm_call_count": sum(1 for item in metadata_list if item.get("prm_reward") is not None),
        "avg_prm_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "max_prm_reward": max(rewards) if rewards else None,
        "min_prm_reward": min(rewards) if rewards else None,
        "draft_accepted_decode_tokens": draft_accepted_decode,
        "draft_discarded_decode_tokens": draft_discarded_decode,
        "draft_decode_tokens": draft_accepted_decode + draft_discarded_decode,
        "target_decode_tokens": target_decode,
        "prm_input_ids": prm_input_ids,
        "excluded_judge_prompt_tokens": judge_prompt_tokens,
        "excluded_judge_decode_tokens": judge_decode_tokens,
        "excluded_judge_total_tokens": judge_prompt_tokens + judge_decode_tokens,
        "total_decode_tokens": total_decode,
        "draft_decode_share": ((draft_accepted_decode + draft_discarded_decode) / total_decode) if total_decode else None,
        "target_decode_share": (target_decode / total_decode) if total_decode else None,
        "draft_wall_time": draft_wall,
        "target_wall_time": target_wall,
        "prm_wall_time": prm_wall,
        "excluded_judge_wall_time": prm_wall,
        "route_sequence": ",".join(str(item.get("selected_model")) for item in metadata_list),
        "stop_reason": stop_reason or "finished",
    }
