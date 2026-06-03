from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy.special import digamma


MIN_N_FOR_GMM = 10


def dirichlet_au_eu(logits: list[float] | np.ndarray, k_top_values: int | list[int], *, only_eu: bool = False) -> dict[str, Any]:
    original_k = k_top_values if isinstance(k_top_values, int) else None
    k_values = [int(k_top_values)] if isinstance(k_top_values, int) else [int(k) for k in k_top_values]
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0:
        result = {k: {"au": 0.0, "eu": 0.0, "alphas": []} for k in k_values}
        return result[original_k] if original_k is not None else result

    max_k = min(max(k_values), values.size)
    top_vals = np.sort(values)[-max_k:][::-1]
    result: dict[int, dict[str, Any]] = {}
    for raw_k in k_values:
        k = min(raw_k, max_k)
        if k <= 0:
            result[raw_k] = {"au": 0.0, "eu": 0.0, "alphas": []}
            continue
        alphas = top_vals[:k]
        alpha0 = float(np.sum(alphas))
        if alpha0 <= 1e-6:
            result[raw_k] = {"au": 0.0, "eu": 0.0, "alphas": []}
            continue
        eu = float(k / (alpha0 + k))
        if only_eu:
            result[raw_k] = {"eu": eu, "alphas": alphas.tolist()}
            continue
        psi_alpha_plus1 = np.nan_to_num(digamma(alphas + 1), nan=0.0, posinf=0.0, neginf=0.0)
        psi_alpha0_plus1 = float(np.nan_to_num(digamma(alpha0 + 1), nan=0.0, posinf=0.0, neginf=0.0))
        au = -float(np.sum((alphas / alpha0) * (psi_alpha_plus1 - psi_alpha0_plus1)))
        if not math.isfinite(au):
            au = 0.0
        if not math.isfinite(eu):
            eu = 0.0
        result[raw_k] = {"au": au, "eu": eu, "alphas": alphas.tolist()}

    return result[original_k] if original_k is not None else result


def token_reliability_from_logits(logits: list[float] | np.ndarray, k_top: int, *, only_eu: bool) -> float:
    metrics = dirichlet_au_eu(logits, k_top, only_eu=only_eu)
    return float(1.0 / (float(metrics["eu"]) + 1e-9))


def _math_mask(tokens: list[str]) -> list[bool]:
    latex_cmd_re = re.compile(r"\\[A-Za-z]+")
    begin_env_re = re.compile(r"\\begin\{([A-Za-z*]+)\}")
    end_env_re = re.compile(r"\\end\{([A-Za-z*]+)\}")
    inline_math = False
    next_is_math = False
    env_depth = 0
    mask = []
    for token in tokens:
        tk = str(token)
        if tk.count("$") % 2 == 1:
            inline_math = not inline_math
        if begin_env_re.search(tk):
            env_depth += 1
        if end_env_re.search(tk) and env_depth > 0:
            env_depth -= 1
        if "{" in tk:
            env_depth += 1
        if "}" in tk:
            env_depth -= 1
        if "\\(" in tk:
            env_depth += 1
        if "\\)" in tk:
            env_depth -= 1
        inside_math = inline_math or env_depth > 0
        has_digit = any(ch.isdigit() for ch in tk)
        is_operator = tk.strip() in {
            "+", "-", "*", "/", "=", "^", "_", "%", "(", ")", "[", "]",
            "{", "}", "\\times", "\\cdot", "\\pm", "\\frac", "\\sqrt",
        }
        is_latex_cmd = bool(latex_cmd_re.match(tk))
        is_math_token = inside_math or has_digit or is_operator or is_latex_cmd
        if next_is_math:
            if tk.startswith(" "):
                next_is_math = False
            else:
                is_math_token = True
                if " " in tk:
                    next_is_math = False
        if tk.endswith("\\"):
            next_is_math = True
        stripped = tk
        for item in ["(", ")", "[", "]", "{", "}", "\\"]:
            stripped = stripped.replace(item, "")
        if not stripped.strip():
            is_math_token = False
        mask.append(is_math_token)
    return mask


def get_step_reliability(
    *,
    tokens: list[str] | None,
    token_reliabilities_list: list[float],
    mode: str,
    k_val: int = 5,
) -> float:
    if not token_reliabilities_list:
        return -float("inf")

    active = list(token_reliabilities_list)
    processing_mode = mode
    if mode.startswith("number_only") or mode.startswith("math_only"):
        if tokens is None or len(tokens) != len(token_reliabilities_list):
            processing_mode = mode.split("_")[-1] if "_" in mode else "avg"
        elif mode.startswith("number_only"):
            processing_mode = mode.replace("number_only_", "")
            active = [rel for rel, token in zip(token_reliabilities_list, tokens) if str(token).isdigit()]
        else:
            processing_mode = mode.replace("math_only_", "")
            active = [rel for rel, keep in zip(token_reliabilities_list, _math_mask(tokens)) if keep]
        if not active:
            return -float("inf")

    if processing_mode == "avg":
        return float(np.mean(active))
    if processing_mode == "sum":
        return float(np.sum(active))
    if processing_mode == "min":
        return float(np.min(active))
    if processing_mode == "min_k_avg":
        if k_val <= 0:
            return -float("inf")
        sorted_values = sorted(active)
        k = min(k_val, len(sorted_values))
        return float(np.mean(sorted_values[:k])) if k else -float("inf")
    raise ValueError(f"Invalid reliability aggregation mode: {mode}")


def fit_mixture_model(values: list[float]):
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for STEER GMM routing.") from exc

    nan_arr = np.array([np.nan, np.nan])
    finite = [float(value) for value in values if value != -float("inf") and math.isfinite(float(value))]
    if len(finite) < MIN_N_FOR_GMM:
        return nan_arr, nan_arr, nan_arr, None

    q1 = np.percentile(finite, 25)
    q3 = np.percentile(finite, 75)
    iqr = q3 - q1
    lower_bound = q1 - 5 * iqr
    upper_bound = q3 + 5 * iqr
    filtered = [value for value in finite if lower_bound <= value <= upper_bound]
    if len(filtered) < 2:
        return nan_arr, nan_arr, nan_arr, None

    arr = np.array(filtered, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        init_params="kmeans",
        n_init=10,
        random_state=0,
        tol=1e-4,
        max_iter=100,
    )
    gmm.fit(arr)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    order = np.argsort(means)
    return means[order], stds[order], weights[order], gmm


def get_gmm_responsibility(
    idx_values: list[tuple[int, float]],
    gmm_model,
    threshold: float = 0.5,
    *,
    rejection_threshold: float = 0.9,
    target_gmm: bool = False,
    use_prior: bool = False,
    reverse_routing: bool = False,
) -> tuple[list[int] | None, list[int] | None, list[int] | None]:
    if not idx_values:
        return [], [], []
    if not use_prior and len(idx_values) < MIN_N_FOR_GMM:
        return [], [item[0] for item in idx_values], []
    raw_values = [float(value) for _, value in idx_values]
    finite_values = [value for value in raw_values if value != -float("inf") and math.isfinite(value)]
    if not finite_values:
        return [], [item[0] for item in idx_values], []
    min_value = min(finite_values)
    values = [value if value != -float("inf") and math.isfinite(value) else min_value for value in raw_values]
    if gmm_model is None:
        return None, None, None
    try:
        probs = gmm_model.predict_proba(np.array(values).reshape(-1, 1))
    except Exception:
        return None, None, None
    means = gmm_model.means_.flatten()
    if probs.shape[0] != len(idx_values) or len(means) != 2:
        return None, None, None

    smaller = int(np.argmin(means))
    correct_like: list[int] = []
    wrong_like: list[int] = []
    too_difficult_for_target: list[int] = []
    for idx, (original_idx, _) in enumerate(idx_values):
        prob_smaller = float(probs[idx, smaller])
        if not target_gmm:
            if (prob_smaller > threshold and not reverse_routing) or (prob_smaller < threshold and reverse_routing):
                wrong_like.append(original_idx)
            else:
                correct_like.append(original_idx)
        else:
            if prob_smaller > rejection_threshold:
                too_difficult_for_target.append(original_idx)
            elif prob_smaller > threshold:
                wrong_like.append(original_idx)
            else:
                correct_like.append(original_idx)
    return correct_like, wrong_like, too_difficult_for_target


def route_prompts(
    idx_value_pairs_draft: list[tuple[int, float]],
    idx_value_pairs_target: list[tuple[int, float]],
    *,
    draft_gmm_threshold: float = 0.5,
    target_gmm_threshold: float = 0.5,
) -> tuple[list[int], list[int], list[int]]:
    if (len(idx_value_pairs_draft) + len(idx_value_pairs_target)) < MIN_N_FOR_GMM:
        return [], [idx for idx, _ in [*idx_value_pairs_draft, *idx_value_pairs_target]], []

    draft_correct_like: list[int] = []
    draft_wrong_like: list[int] = []
    if idx_value_pairs_draft:
        _, _, weights, gmm_model = fit_mixture_model([value for _, value in idx_value_pairs_draft])
        if np.any(weights > 0.99):
            draft_wrong_like = [idx for idx, _ in idx_value_pairs_draft]
        elif gmm_model is not None:
            correct, wrong, _ = get_gmm_responsibility(
                idx_value_pairs_draft,
                gmm_model,
                draft_gmm_threshold,
                target_gmm=False,
                use_prior=False,
                reverse_routing=False,
            )
            draft_correct_like = correct or []
            draft_wrong_like = wrong or []
        else:
            draft_wrong_like = [idx for idx, _ in idx_value_pairs_draft]

    target_correct_like: list[int] = []
    target_wrong_like: list[int] = []
    too_difficult_for_target: list[int] = []
    if idx_value_pairs_target:
        _, _, weights, gmm_model = fit_mixture_model([value for _, value in idx_value_pairs_target])
        if np.any(weights > 0.99):
            target_wrong_like = [idx for idx, _ in idx_value_pairs_target]
        elif gmm_model is not None:
            correct, wrong, too_difficult = get_gmm_responsibility(
                idx_value_pairs_target,
                gmm_model,
                target_gmm_threshold,
                target_gmm=True,
                use_prior=False,
                reverse_routing=False,
            )
            target_correct_like = correct or []
            target_wrong_like = wrong or []
            too_difficult_for_target = too_difficult or []
        else:
            target_wrong_like = [idx for idx, _ in idx_value_pairs_target]

    next_draft = set(draft_correct_like + target_correct_like)
    next_target = set(draft_wrong_like + target_wrong_like) - next_draft
    target_rejection = set(too_difficult_for_target) - next_draft - next_target
    return list(next_draft), list(next_target), list(target_rejection)
