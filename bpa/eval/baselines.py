from __future__ import annotations

from bpa.config import BPAConfig
from bpa.pipeline import bpa_solve, solve_engine_only
from bpa.trace import BPAResult


def solve_variant(problem_text: str, variant: str, slm, llm, config: BPAConfig) -> BPAResult:
    if variant == "slm_only":
        return solve_engine_only(problem_text, slm, config, account="slm")
    if variant == "llm_only":
        return solve_engine_only(problem_text, llm, config, account="llm")
    if variant == "glimprouter_hinit":
        return bpa_solve(problem_text, slm, llm, config)
    raise ValueError(f"Unknown variant: {variant}")
