from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class PercentileNormalizer:
    def __init__(self, values: list[float] | np.ndarray):
        self.values = np.sort(np.asarray(values, dtype=np.float32))

    def transform(self, x: float) -> float:
        idx = np.searchsorted(self.values, np.float32(x), side="right")
        return float(idx / max(len(self.values), 1))

    def to_dict(self, *, topk_entropy: int) -> dict[str, Any]:
        return {
            "calibration_values": [float(v) for v in self.values.tolist()],
            "topk_entropy": int(topk_entropy),
            "num_values": int(len(self.values)),
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "PercentileNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        values = data.get("calibration_values", [])
        if not values:
            raise ValueError(f"Calibration file contains no calibration_values: {path}")
        return cls(values)


class IdentityNormalizer:
    def transform(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))


def smooth_confidence(c_norm_history: list[float], W: int = 2) -> float | None:
    if len(c_norm_history) < W:
        return None
    return float(np.mean(c_norm_history[-W:]))


def code_style_degeneration_event(prev_c_smooth: float | None, curr_c_smooth: float | None, delta: float = 0.55) -> int:
    if prev_c_smooth is None or curr_c_smooth is None:
        return 0
    return int((prev_c_smooth > curr_c_smooth) and (2.0 * curr_c_smooth - prev_c_smooth < delta))
