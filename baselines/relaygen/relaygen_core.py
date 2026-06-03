from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


THINK_END = ["</think>"]
SENT_END = [".", "!", "?", "\n\n"]

QWEN3_SWITCH_CUES = [
    "Oh,", "another,", "Thus", "Now", "Alternatively", "alternatively,",
    "Thus,", "Therefore", "similarly", "similarly,", "now", "Again",
    "specifically,", "Again,", "Similarly,", "Now,", "Specifically,", "Hence",
    "Similarly", "Other", "now,", "hence", "Specifically", "So ",
    "Therefore,", "Wait,", "Also", "So,",
]

R1_SWITCH_CUES = [
    "Wait", "Thus", "thus", "similarly", "Again,", "Now",
    "Therefore", "hence", "Hence,", "Now,", "Thus,", "Oh,",
    "Similarly,", "Any", "Therefore,", "Alternatively,", "now,", "So,",
    "now", "verify", "Specifically,", "Alternatively", "Ah,", "wait",
    "So ",
]


@dataclass(frozen=True)
class EndpointConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    timeout: float = 3600.0


@dataclass(frozen=True)
class RelayGenHyperparams:
    budget: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = 20
    presence_penalty: float | None = None
    cue_family: str = "qwen3"
    answer_model: str = "small"
    min_tokens_large: int = 5
    enable_thinking: bool = True
    include_stop_str_in_output: bool = True

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be positive.")
        if self.cue_family not in {"qwen3", "r1"}:
            raise ValueError("cue_family must be 'qwen3' or 'r1'.")
        if self.answer_model not in {"small", "base"}:
            raise ValueError("answer_model must be 'small' or 'base'.")
        if self.min_tokens_large < 0:
            raise ValueError("min_tokens_large must be non-negative.")


def parse_endpoints(data: dict[str, Any]) -> dict[str, EndpointConfig]:
    endpoints: dict[str, EndpointConfig] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Endpoint {name!r} must be an object.")
        endpoints[name] = EndpointConfig(
            model=str(raw["model"]),
            base_url=str(raw["base_url"]),
            api_key=str(raw.get("api_key") or "EMPTY"),
            timeout=float(raw.get("timeout") or 3600.0),
        )
    return endpoints


def switch_cues_for_family(cue_family: str) -> list[str]:
    if cue_family == "qwen3":
        return list(QWEN3_SWITCH_CUES)
    if cue_family == "r1":
        return list(R1_SWITCH_CUES)
    raise ValueError(f"Unsupported cue_family: {cue_family}")


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


def first_choice(response: Any) -> Any:
    choices = list(getattr(response, "choices", []) or [])
    if not choices:
        raise ValueError("Chat completion response has no choices.")
    return choices[0]


def first_choice_text(response: Any) -> str:
    choice = first_choice(response)
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    return str(getattr(message, "content", "") or "")


def first_finish_reason(response: Any) -> str | None:
    return getattr(first_choice(response), "finish_reason", None)


def parse_reasoning_and_answer(text: str | None) -> tuple[str, str]:
    if not text:
        return "", ""
    if "</think>" in text:
        reasoning_str = text.split("</think>", 1)[0].replace("<think>", "").strip()
        answer_str = text.split("</think>", 1)[1].strip()
        return reasoning_str, answer_str
    return text, ""


def base_extra_body(hp: RelayGenHyperparams, *, add_generation_prompt: bool, continue_final_message: bool, min_tokens: int = 0) -> dict[str, Any]:
    extra_body: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "continue_final_message": continue_final_message,
        "include_stop_str_in_output": hp.include_stop_str_in_output,
        "chat_template_kwargs": {"enable_thinking": hp.enable_thinking},
    }
    if min_tokens > 0:
        extra_body["min_tokens"] = min_tokens
    if hp.top_k is not None:
        extra_body["top_k"] = hp.top_k
    if hp.presence_penalty is not None:
        extra_body["presence_penalty"] = hp.presence_penalty
    return extra_body


class RelayGenRouter:
    def __init__(
        self,
        *,
        endpoints: dict[str, EndpointConfig],
        hyperparams: RelayGenHyperparams | None = None,
    ) -> None:
        self.endpoints = endpoints
        self.hyperparams = hyperparams or RelayGenHyperparams()
        for key in ("base", "small"):
            if key not in endpoints:
                raise ValueError(f"Missing endpoint config for {key!r}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required to run RelayGen against OpenAI-compatible vLLM endpoints."
            ) from exc
        self.clients = {
            key: OpenAI(
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                timeout=endpoint.timeout,
            )
            for key, endpoint in endpoints.items()
        }

    def model_name(self, key: str) -> str:
        return self.endpoints[key].model

    def call_chat(
        self,
        *,
        model_key: str,
        problem: str,
        generated_text: str,
        max_tokens: int,
        stop_tokens: list[str] | None,
        mode: str,
    ) -> dict[str, Any]:
        hp = self.hyperparams
        min_tokens = min(hp.min_tokens_large, max_tokens) if mode == "L" else 0
        extra_body = base_extra_body(
            hp,
            add_generation_prompt=(generated_text == ""),
            continue_final_message=(generated_text != ""),
            min_tokens=min_tokens,
        )
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": generated_text},
        ]
        start = time.perf_counter()
        response = self.clients[model_key].chat.completions.create(
            model=self.model_name(model_key),
            messages=messages,
            temperature=hp.temperature,
            top_p=hp.top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - start
        text = first_choice_text(response)
        usage = usage_tokens(response)
        return {
            "text": text,
            "finish_reason": first_finish_reason(response),
            "usage": usage,
            "wall_time": elapsed,
            "extra_body": extra_body,
        }

    def run(
        self,
        *,
        problem_prompt: str,
        dataset_name: str,
        problem_id: Any,
        repeat_id: int = 0,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        del dataset_name
        hp = self.hyperparams
        switch_cues = switch_cues_for_family(hp.cue_family)
        generated_text = ""
        completion_tokens = 0
        base_tokens = 0
        small_tokens = 0
        base_wall_time = 0.0
        small_wall_time = 0.0
        mode = "L"
        sentence_id = 0
        in_think = True
        switch_log: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        final_finish_reason: str | None = None

        start = time.perf_counter()

        while in_think and completion_tokens < hp.budget:
            model_key = "base" if mode == "L" else "small"
            stop_tokens = THINK_END + (switch_cues if mode == "L" else SENT_END)
            remaining = hp.budget - completion_tokens
            if remaining <= 0:
                break
            chunk_payload = self.call_chat(
                model_key=model_key,
                problem=problem_prompt,
                generated_text=generated_text,
                max_tokens=remaining,
                stop_tokens=stop_tokens,
                mode=mode,
            )
            chunk = str(chunk_payload["text"] or "")
            chunk_tokens = int(chunk_payload["usage"]["completion_tokens"])
            finish_reason = chunk_payload["finish_reason"]
            final_finish_reason = finish_reason
            if not chunk:
                final_finish_reason = "empty_chunk"
                break

            if mode == "L":
                base_tokens += chunk_tokens
                base_wall_time += float(chunk_payload["wall_time"])
            else:
                small_tokens += chunk_tokens
                small_wall_time += float(chunk_payload["wall_time"])

            generated_text += chunk
            completion_tokens += chunk_tokens

            chunk_record = {
                "mode": mode,
                "model_key": model_key,
                "model": self.model_name(model_key),
                "text": chunk,
                "tokens": chunk_tokens,
                "finish_reason": finish_reason,
                "wall_time": chunk_payload["wall_time"],
                "tokens_at_end": completion_tokens,
            }
            chunks.append(chunk_record)

            if verbose:
                print(
                    f"[relaygen] mode={mode} tokens={chunk_tokens} total={completion_tokens}/{hp.budget} "
                    f"base={base_tokens} small={small_tokens}",
                    flush=True,
                )

            switched = False
            if mode == "L" and finish_reason == "stop":
                for cue in switch_cues:
                    if chunk.endswith(cue):
                        mode = "Ls"
                        sentence_id += 1
                        switch_entry = {
                            "sentence_id": sentence_id,
                            "switch_type": "L->Ls",
                            "trigger_cues": [cue],
                            "stop_token_triggered": True,
                            "text_len_at_switch": len(generated_text),
                            "tokens_at_switch": completion_tokens,
                        }
                        switch_log.append(switch_entry)
                        chunk_record["switch"] = switch_entry
                        switched = True
                        break
            elif mode == "Ls" and finish_reason == "stop":
                for sent_end in SENT_END:
                    if chunk.endswith(sent_end):
                        mode = "L"
                        sentence_id += 1
                        switch_entry = {
                            "sentence_id": sentence_id,
                            "switch_type": "Ls->L",
                            "trigger_cues": ["sentence_end"],
                            "stop_token_triggered": True,
                            "text_len_at_switch": len(generated_text),
                            "tokens_at_switch": completion_tokens,
                        }
                        switch_log.append(switch_entry)
                        chunk_record["switch"] = switch_entry
                        switched = True
                        break
            chunk_record["switched"] = switched

            if "</think>" in chunk:
                in_think = False
                break

        if "</think>" not in generated_text:
            generated_text += "\n</think>"
            completion_tokens += 1
            chunks.append(
                {
                    "mode": "synthetic",
                    "model_key": None,
                    "model": None,
                    "text": "\n</think>",
                    "tokens": 1,
                    "finish_reason": "inserted_think_end",
                    "wall_time": 0.0,
                    "tokens_at_end": completion_tokens,
                    "switched": False,
                }
            )

        answer_content = ""
        answer_tokens = 0
        answer_wall_time = 0.0
        if completion_tokens < hp.budget:
            answer_model_key = hp.answer_model
            remaining = hp.budget - completion_tokens
            answer_payload = self.call_chat(
                model_key=answer_model_key,
                problem=problem_prompt,
                generated_text=generated_text,
                max_tokens=remaining,
                stop_tokens=None,
                mode="answer",
            )
            answer_content = str(answer_payload["text"] or "")
            answer_tokens = int(answer_payload["usage"]["completion_tokens"])
            answer_wall_time = float(answer_payload["wall_time"])
            completion_tokens += answer_tokens
            if answer_model_key == "base":
                base_tokens += answer_tokens
                base_wall_time += answer_wall_time
            else:
                small_tokens += answer_tokens
                small_wall_time += answer_wall_time
            final_finish_reason = answer_payload["finish_reason"]
            chunks.append(
                {
                    "mode": "answer",
                    "model_key": answer_model_key,
                    "model": self.model_name(answer_model_key),
                    "text": answer_content,
                    "tokens": answer_tokens,
                    "finish_reason": answer_payload["finish_reason"],
                    "wall_time": answer_wall_time,
                    "tokens_at_end": completion_tokens,
                    "switched": False,
                }
            )
        else:
            final_finish_reason = "length"

        final_text = generated_text + answer_content
        generation_time = time.perf_counter() - start
        reasoning_str, answer_str = parse_reasoning_and_answer(final_text)

        l_to_ls_switches = sum(1 for item in switch_log if item["switch_type"] == "L->Ls")
        ls_to_l_switches = sum(1 for item in switch_log if item["switch_type"] == "Ls->L")
        total_switches = len(switch_log)
        large_percentage = (base_tokens / completion_tokens * 100.0) if completion_tokens > 0 else 0.0
        small_percentage = (small_tokens / completion_tokens * 100.0) if completion_tokens > 0 else 0.0
        avg_tokens_per_session = (completion_tokens / (total_switches + 1)) if total_switches > 0 else completion_tokens
        switch_rate = (total_switches / completion_tokens) if completion_tokens > 0 else 0.0

        metadata = {
            "problem_id": problem_id,
            "repeat_id": repeat_id,
            "base_model": self.model_name("base"),
            "small_model": self.model_name("small"),
            "budget": hp.budget,
            "temperature": hp.temperature,
            "top_p": hp.top_p,
            "top_k": hp.top_k,
            "presence_penalty": hp.presence_penalty,
            "cue_family": hp.cue_family,
            "answer_model": hp.answer_model,
            "generation_time": generation_time,
            "partial_text": final_text,
            "final_text": final_text,
            "reasoning_str": reasoning_str,
            "answer_str": answer_str,
            "total_tokens": completion_tokens,
            "base_tokens": base_tokens,
            "small_tokens": small_tokens,
            "base_wall_time": base_wall_time,
            "small_wall_time": small_wall_time,
            "answer_tokens": answer_tokens,
            "answer_wall_time": answer_wall_time,
            "finish_reason": final_finish_reason,
            "switch_log": switch_log,
            "chunks": chunks,
            "switching_stats": {
                "total_switches": total_switches,
                "l_to_ls_switches": l_to_ls_switches,
                "ls_to_l_switches": ls_to_l_switches,
                "avg_tokens_per_session": avg_tokens_per_session,
                "large_model_percentage": large_percentage,
                "small_model_percentage": small_percentage,
                "switch_rate": switch_rate,
            },
        }
        return [metadata]


def extract_answer_text(metadata_list: list[dict[str, Any]]) -> str | None:
    if not metadata_list:
        return None
    last = metadata_list[-1]
    return str(last.get("final_text") or last.get("partial_text") or "")


def route_stats(metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not metadata_list:
        return {}
    last = metadata_list[-1]
    switching = last.get("switching_stats") or {}
    total_tokens = int(last.get("total_tokens") or 0)
    base_tokens = int(last.get("base_tokens") or 0)
    small_tokens = int(last.get("small_tokens") or 0)
    base_wall = float(last.get("base_wall_time") or 0.0)
    small_wall = float(last.get("small_wall_time") or 0.0)
    total_wall = base_wall + small_wall
    switch_log = list(last.get("switch_log") or [])
    chunks = list(last.get("chunks") or [])
    large_sessions = sum(1 for chunk in chunks if chunk.get("mode") == "L")
    small_sessions = sum(1 for chunk in chunks if chunk.get("mode") in {"Ls", "answer"} and chunk.get("model_key") == "small")
    return {
        "step_count": int(switching.get("total_switches") or 0) + 1,
        "large_model_tokens": base_tokens,
        "small_model_tokens": small_tokens,
        "total_decode_tokens": total_tokens,
        "large_model_percentage": switching.get("large_model_percentage"),
        "small_model_percentage": switching.get("small_model_percentage"),
        "total_switches": switching.get("total_switches", 0),
        "l_to_ls_switches": switching.get("l_to_ls_switches", 0),
        "ls_to_l_switches": switching.get("ls_to_l_switches", 0),
        "avg_tokens_per_session": switching.get("avg_tokens_per_session"),
        "switch_rate": switching.get("switch_rate"),
        "large_sessions": large_sessions,
        "small_sessions": small_sessions,
        "slm_decode_tokens": small_tokens,
        "slm_prefill_tokens": 0,
        "llm_decode_tokens": base_tokens,
        "llm_prefill_tokens": 0,
        "slm_generate_calls": small_sessions,
        "llm_generate_calls": large_sessions,
        "llm_full_calls": large_sessions,
        "llm_scoring_calls": 0,
        "llm_token_share": (base_tokens / total_tokens) if total_tokens else None,
        "llm_decode_share": (base_tokens / total_tokens) if total_tokens else None,
        "llm_wall_time_share": (base_wall / total_wall) if total_wall else None,
        "slm_wall_time": small_wall,
        "llm_generation_wall_time": base_wall,
        "llm_scoring_wall_time": 0.0,
        "route_sequence": ",".join(str(item.get("switch_type")) for item in switch_log),
        "stop_reason": last.get("finish_reason") or "finished",
    }
