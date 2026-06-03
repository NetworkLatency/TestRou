from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from steer_reliability import (
    fit_mixture_model,
    get_gmm_responsibility,
    get_step_reliability,
    route_prompts,
)


CLIENT_DRAFT = 1
CLIENT_TARGET = 2


@dataclass(frozen=True)
class ChatEndpointConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    timeout: float = 3600.0


@dataclass(frozen=True)
class SteerHyperparams:
    seed: int = 0
    temperature: float = 0.7
    top_p: float = 1.0
    n_sampling: int = 1
    max_tokens_per_call: int = 16384
    min_tokens: int = 2
    max_steps: int = 100
    patience: int = 5
    step_word: str = "\n\n"
    logprobs: int = 5
    reliability_metric: str = "R_eu"
    reliability_mode: str = "math_only_avg"
    reliability_k_top: int = 1
    reliability_target_usage: str = "gmm_responsibility"
    draft_gmm_threshold: float = 0.4
    target_gmm_threshold: float = 0.4

    def __post_init__(self) -> None:
        if self.reliability_metric != "R_eu":
            raise ValueError("This STEER adapter implements reliability_metric='R_eu'.")
        if self.reliability_target_usage != "gmm_responsibility":
            raise ValueError("This STEER adapter implements reliability_target_usage='gmm_responsibility'.")


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    stop_reason: Any
    token_strings: list[str]
    top_logits: list[list[float]]
    wall_time: float


@dataclass
class SteerState:
    item_id: int
    problem_id: Any
    repeat_id: int
    base_prompt: str
    responses: list[tuple[str, int]] = field(default_factory=list)
    output: str | None = None
    token_counts: list[int] = field(default_factory=lambda: [0, 0, 0])
    step_info: list[tuple[int, int]] = field(default_factory=list)
    reliabilities: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def prompt_text(self) -> str:
        return self.base_prompt + "".join(text for text, _ in self.responses)

    def solution_text(self) -> str:
        return "".join(text for text, _ in self.responses)


def reliability_from_generation(result: GenerationResult, hp: SteerHyperparams, *, target_model: bool) -> tuple[float, list[float]]:
    del target_model
    token_reliabilities = []
    for logprobs in result.top_logits:
        if not logprobs:
            continue
        probs = [math.exp(float(value)) for value in logprobs]
        total = sum(probs)
        if total <= 0:
            continue
        normalized = [prob / total for prob in probs]
        entropy = -sum(prob * math.log(prob) for prob in normalized if prob > 0)
        token_reliabilities.append(max(probs) / (1.0 + entropy))
    value = get_step_reliability(
        tokens=result.token_strings,
        token_reliabilities_list=token_reliabilities,
        mode=hp.reliability_mode,
    )
    return float(value), token_reliabilities


def parse_chat_endpoint(data: dict[str, Any]) -> ChatEndpointConfig:
    return ChatEndpointConfig(
        model=str(data["model"]),
        base_url=str(data["base_url"]),
        api_key=str(data.get("api_key") or "EMPTY"),
        timeout=float(data.get("timeout") or 3600.0),
    )


def usage_completion_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    return int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0


def build_chat_messages(problem: str, generated: str, hp: SteerHyperparams) -> tuple[list[dict[str, str]], dict[str, Any]]:
    extra_body = {
        "add_generation_prompt": not generated,
        "continue_final_message": bool(generated),
        "include_stop_str_in_output": True,
        "min_tokens": hp.min_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if not generated:
        return [{"role": "user", "content": problem}], extra_body
    return (
        [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"<think>{generated}"},
        ],
        extra_body,
    )


class ChatGenerator:
    def __init__(self, config: ChatEndpointConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for remote-chat STEER.") from exc
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout)

    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text) // 4)))

    def generate_continuations(self, items: list[tuple[str, str]], hp: SteerHyperparams) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for problem, generated in items:
            messages, extra_body = build_chat_messages(problem, generated, hp)
            top_p = 1.0 if hp.temperature == 0 else hp.top_p
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=hp.temperature,
                top_p=top_p,
                max_tokens=hp.max_tokens_per_call,
                stop=[hp.step_word],
                logprobs=True,
                top_logprobs=hp.logprobs,
                extra_body=extra_body,
            )
            wall_time = time.time() - start
            choice = list(getattr(response, "choices", []) or [None])[0]
            message = getattr(choice, "message", None) if choice is not None else None
            text = str(getattr(message, "content", "") or "") if message is not None else ""
            token_strings: list[str] = []
            top_logprobs: list[list[float]] = []
            logprobs_obj = getattr(choice, "logprobs", None) if choice is not None else None
            for step in list(getattr(logprobs_obj, "content", []) or []):
                token_strings.append(str(getattr(step, "token", "") or ""))
                top_logprobs.append(
                    [
                        float(getattr(item, "logprob"))
                        for item in list(getattr(step, "top_logprobs", []) or [])
                        if getattr(item, "logprob", None) is not None
                    ]
                )
            completion_tokens = usage_completion_tokens(response)
            if completion_tokens <= 0:
                completion_tokens = max(1, len(token_strings) or len(text) // 4)
            results.append(
                GenerationResult(
                    text=text,
                    token_ids=list(range(completion_tokens)),
                    stop_reason=getattr(choice, "finish_reason", None) if choice is not None else None,
                    token_strings=token_strings,
                    top_logits=top_logprobs,
                    wall_time=wall_time,
                )
            )
        return results


def mapped_stop_reason(stop_reason: Any, hp: SteerHyperparams) -> str:
    if stop_reason == hp.step_word:
        return "step_word"
    if stop_reason == "length":
        return "length"
    if stop_reason == "stop":
        return "stop_sequence"
    return "stop_sequence"


class SteerBatchRouter:
    def __init__(self, *, draft: Any, target: Any, hyperparams: SteerHyperparams) -> None:
        self.draft = draft
        self.target = target
        self.hp = hyperparams

    def run(self, items: list[tuple[Any, int, str]]) -> list[SteerState]:
        states = [
            SteerState(item_id=i, problem_id=problem_id, repeat_id=repeat_id, base_prompt=prompt)
            for i, (problem_id, repeat_id, prompt) in enumerate(items)
        ]
        active_ids = [state.item_id for state in states]
        idx_to_solve_with_draft: list[int] = []
        num_step = 0
        num_unchanged = 0
        stop_sequences_full_solve = ["<end_of_turn>", "<|im_end|>", "<|endoftext|>", "</s>"]

        while active_ids:
            state_by_id = {state.item_id: state for state in states}
            active = [state_by_id[idx] for idx in active_ids]
            prompts_for_draft: list[SteerState] = []
            prompts_for_target: list[SteerState] = []
            idx_value_pairs_draft: list[tuple[int, float]] = []
            idx_value_pairs_target: list[tuple[int, float]] = []

            for state in active:
                if num_step == 0:
                    prompts_for_draft.append(state)
                elif state.reliabilities:
                    last = state.reliabilities[-1]
                    pair = (state.item_id, float(last.get("decision_value", -float("inf"))))
                    if int(last.get("client_id")) == CLIENT_DRAFT:
                        idx_value_pairs_draft.append(pair)
                    else:
                        idx_value_pairs_target.append(pair)
                else:
                    prompts_for_target.append(state)

            if num_step != 0 and (idx_value_pairs_draft or idx_value_pairs_target):
                next_draft, next_target, target_rejection = route_prompts(
                    idx_value_pairs_draft,
                    idx_value_pairs_target,
                    draft_gmm_threshold=self.hp.draft_gmm_threshold,
                    target_gmm_threshold=self.hp.target_gmm_threshold,
                )
                for idx in target_rejection:
                    if idx not in idx_to_solve_with_draft:
                        idx_to_solve_with_draft.append(idx)
                next_draft = list(set(next_draft + idx_to_solve_with_draft))
                idx_to_solve_with_draft = [idx for idx in idx_to_solve_with_draft if idx in active_ids]
                prompts_for_draft.extend(state_by_id[idx] for idx in next_draft if idx in state_by_id)
                prompts_for_target.extend(
                    state_by_id[idx]
                    for idx in next_target
                    if idx in state_by_id and idx not in idx_to_solve_with_draft
                )

            current_results: dict[int, dict[str, Any]] = {}
            if prompts_for_draft:
                self._generate_for_states(
                    prompts_for_draft,
                    self.draft,
                    CLIENT_DRAFT,
                    current_results,
                    target_model=False,
                )

            if num_step == 0 and prompts_for_draft and self.hp.reliability_target_usage == "gmm_responsibility":
                idx_values = [
                    (state.item_id, float(current_results[state.item_id]["decision_value"]))
                    for state in prompts_for_draft
                    if state.item_id in current_results
                ]
                if idx_values:
                    _, _, _, gmm_model = fit_mixture_model([value for _, value in idx_values])
                    _, wrong_like, _ = get_gmm_responsibility(
                        idx_values,
                        gmm_model,
                        self.hp.draft_gmm_threshold,
                        target_gmm=False,
                    )
                    for idx in wrong_like or []:
                        state = state_by_id[idx]
                        state.token_counts[2] += len(current_results[idx]["token_ids"])
                        if state not in prompts_for_target:
                            prompts_for_target.append(state)

            if prompts_for_target:
                self._generate_for_states(
                    prompts_for_target,
                    self.target,
                    CLIENT_TARGET,
                    current_results,
                    target_model=True,
                )

            next_active: list[int] = []
            num_finished_this_step = 0
            for state in active:
                data = current_results.get(state.item_id)
                if data is None:
                    state.output = "ERROR_NO_GENERATION_RESULT_FOR_STEP"
                    num_finished_this_step += 1
                    continue

                text = str(data["text"])
                client_id = int(data["client_id"])
                num_tokens = len(data["token_ids"])
                if client_id == CLIENT_DRAFT:
                    state.token_counts[0] += num_tokens
                else:
                    state.token_counts[1] += num_tokens
                state.step_info.append((num_step, client_id))
                state.reliabilities.append(
                    {
                        "step": num_step,
                        "used_target": client_id == CLIENT_TARGET,
                        "client_id": client_id,
                        "decision_value": data["decision_value"],
                    }
                )

                processed = text
                stop_reason = mapped_stop_reason(data["stop_reason"], self.hp)
                if stop_reason == "step_word":
                    if not processed.endswith(self.hp.step_word):
                        processed += self.hp.step_word
                elif stop_reason != "length" and not any(processed.strip().endswith(sw) for sw in stop_sequences_full_solve):
                    if not processed.endswith(self.hp.step_word):
                        processed += self.hp.step_word

                state.responses.append((processed, client_id))
                state.steps.append(
                    {
                        "step": num_step,
                        "client_id": client_id,
                        "model": "draft" if client_id == CLIENT_DRAFT else "target",
                        "text": processed,
                        "raw_text": text,
                        "token_count": num_tokens,
                        "stop_reason_raw": data["stop_reason"],
                        "decision_value": data["decision_value"],
                        "token_reliabilities": data["token_reliabilities"],
                        "wall_time": data["wall_time"],
                    }
                )

                solution = state.solution_text()
                full_text_for_len_check = state.base_prompt + solution
                max_tokens_reached = (
                    len(self.draft.encode(full_text_for_len_check)) >= self.hp.max_tokens_per_call
                    or len(self.target.encode(full_text_for_len_check)) >= self.hp.max_tokens_per_call
                )
                is_finished = (
                    ("\\boxed" in processed and processed.endswith(self.hp.step_word))
                    or stop_reason == "length"
                    or max_tokens_reached
                    or num_step >= self.hp.max_steps - 1
                    or any(solution.strip().endswith(sw) for sw in stop_sequences_full_solve)
                )
                if is_finished:
                    num_finished_this_step += 1
                    output = solution.strip()
                    if output.endswith(self.hp.step_word):
                        output = output[: -len(self.hp.step_word)]
                    state.output = output
                else:
                    next_active.append(state.item_id)

            if num_finished_this_step > 0 or not next_active:
                if len(active_ids) == len(next_active) and num_finished_this_step == 0:
                    num_unchanged += 1
                else:
                    num_unchanged = 0
            else:
                num_unchanged += 1

            active_ids = next_active
            num_step += 1
            if num_unchanged >= self.hp.patience:
                for idx in active_ids:
                    state = state_by_id[idx]
                    output = state.solution_text().strip()
                    if output.endswith(self.hp.step_word):
                        output = output[: -len(self.hp.step_word)]
                    state.output = output
                break

        return states

    def _generate_for_states(
        self,
        states: list[SteerState],
        generator: Any,
        client_id: int,
        current_results: dict[int, dict[str, Any]],
        *,
        target_model: bool,
    ) -> None:
        prompts = []
        valid_states: list[SteerState] = []
        for state in states:
            generated = state.solution_text()
            prompt = state.base_prompt
            if len(generator.encode(prompt + generated)) > self.hp.max_tokens_per_call:
                continue
            prompts.append((prompt, generated))
            valid_states.append(state)
        if hasattr(generator, "generate_continuations"):
            generations = generator.generate_continuations(prompts, self.hp)
        else:
            generations = generator.generate([prompt + generated for prompt, generated in prompts], self.hp)
        for state, result in zip(valid_states, generations):
            decision_value, token_reliabilities = reliability_from_generation(
                result,
                self.hp,
                target_model=target_model,
            )
            current_results[state.item_id] = {
                "text": result.text,
                "token_ids": result.token_ids,
                "stop_reason": result.stop_reason,
                "client_id": client_id,
                "decision_value": decision_value,
                "token_reliabilities": token_reliabilities,
                "wall_time": result.wall_time,
            }
