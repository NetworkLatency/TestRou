from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_MODEL_PAIRS: dict[str, dict[str, dict[str, Any]]] = {
    "qwen3_1p7b_qwen3_32b": {
        "small": {
            "model": "Qwen/Qwen3-1.7B",
            "base_url": "http://127.0.0.1:30002/v1",
            "api_key": "EMPTY",
        },
        "base": {
            "model": "Qwen/Qwen3-32B",
            "base_url": "http://127.0.0.1:30000/v1",
            "api_key": "EMPTY",
        },
    },
    "qwen3_1p7b_deepseek_qwen_32b": {
        "small": {
            "model": "Qwen/Qwen3-1.7B",
            "base_url": "http://127.0.0.1:30002/v1",
            "api_key": "EMPTY",
        },
        "base": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "base_url": "http://127.0.0.1:30001/v1",
            "api_key": "EMPTY",
        },
    },
    "deepseek_qwen_1p5b_qwen3_32b": {
        "small": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "base_url": "http://127.0.0.1:30003/v1",
            "api_key": "EMPTY",
        },
        "base": {
            "model": "Qwen/Qwen3-32B",
            "base_url": "http://127.0.0.1:30000/v1",
            "api_key": "EMPTY",
        },
    },
    "deepseek_qwen_1p5b_deepseek_qwen_32b": {
        "small": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "base_url": "http://127.0.0.1:30003/v1",
            "api_key": "EMPTY",
        },
        "base": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "base_url": "http://127.0.0.1:30001/v1",
            "api_key": "EMPTY",
        },
    },
}


def available_model_pairs(config: dict[str, Any] | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    pairs = deepcopy(DEFAULT_MODEL_PAIRS)
    if config:
        for name, pair in dict(config.get("model_pairs") or {}).items():
            pairs[name] = deepcopy(pair)
    return pairs


def apply_model_pair(
    config: dict[str, Any],
    model_pair: str | None,
    *,
    small_key: str,
    base_key: str,
    endpoints_key: str = "endpoints",
) -> dict[str, Any]:
    selected = model_pair or str(config.get("model_pair") or "qwen3_1p7b_qwen3_32b")
    pairs = available_model_pairs(config)
    if selected not in pairs:
        valid = ", ".join(sorted(pairs))
        raise ValueError(f"Unknown model_pair={selected!r}. Valid values: {valid}")
    pair = pairs[selected]
    if "small" not in pair or "base" not in pair:
        raise ValueError(f"model_pair={selected!r} must define both small and base endpoints.")

    endpoints = deepcopy(dict(config.get(endpoints_key) or {}))
    endpoints[small_key] = deepcopy(pair["small"])
    endpoints[base_key] = deepcopy(pair["base"])
    config[endpoints_key] = endpoints
    config["model_pair"] = selected
    return config


def model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "deepseek-r1-distill-qwen" in lowered:
        return "r1"
    if "qwen3" in lowered:
        return "qwen3"
    return "unknown"
