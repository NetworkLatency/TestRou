from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, pearsonr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bpa.config import BPAConfig
from bpa.eval.datasets import EvalProblem, load_eval_dataset
from bpa.render import render_for_continuation


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if value is True or str(value).strip().lower() == "true":
        return True
    if value is False or str(value).strip().lower() == "false":
        return False
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and math.isfinite(value)]
    return float(sum(present) / len(present)) if present else None


def _std(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and math.isfinite(value)]
    if len(present) < 2:
        return None
    return float(np.std(np.asarray(present, dtype=np.float64), ddof=0))


def _sha1_json(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _cache_key(row: dict[str, Any], rollout_idx: int, context_ids: list[int], target_ids: list[int]) -> str:
    payload = {
        "dataset": row.get("dataset"),
        "problem_id": row.get("problem_id"),
        "boundary_idx": row.get("boundary_idx"),
        "selected_rank": row.get("selected_rank"),
        "rollout_idx": rollout_idx,
        "context_ids_sha1": _sha1_json(context_ids),
        "target_ids_sha1": _sha1_json(target_ids),
    }
    return _sha1_json(payload)


class Scorer:
    def score_target(self, context_ids: list[int], target_ids: list[int]) -> float | None:
        raise NotImplementedError


@contextlib.contextmanager
def _cuda_visible_devices_scope(devices: str | None):
    if devices is None or str(devices).strip() == "":
        yield
        return
    devices = str(devices).removeprefix("cuda:")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


class VllmScorer(Scorer):
    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_path: str | None,
        trust_remote_code: bool,
        max_model_len: int,
        enable_prefix_caching: bool,
        engine_kwargs: dict[str, Any],
        cuda_visible_devices: str | None,
    ) -> None:
        from vllm import LLM, SamplingParams

        self._sampling_params_cls = SamplingParams
        kwargs = {
            "model": model_path,
            "tokenizer": tokenizer_path or model_path,
            "trust_remote_code": trust_remote_code,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
        }
        kwargs.update(engine_kwargs)
        with _cuda_visible_devices_scope(cuda_visible_devices):
            self.llm = LLM(**kwargs)

    def score_target(self, context_ids: list[int], target_ids: list[int]) -> float | None:
        if not target_ids:
            return None
        try:
            from vllm.inputs import TokensPrompt

            prompt = TokensPrompt(prompt_token_ids=context_ids + target_ids)
        except ImportError:
            prompt = {"prompt_token_ids": context_ids + target_ids}
        sampling = self._sampling_params_cls(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        out = self.llm.generate([prompt], sampling)[0]
        prompt_logprobs = list(getattr(out, "prompt_logprobs", []) or [])
        return _mean_prompt_logprobs(prompt_logprobs, context_ids, target_ids)


class OpenAICompletionScorer(Scorer):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        tokenizer: Any,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer
        self.api_key = api_key
        self.timeout = timeout

    def score_target(self, context_ids: list[int], target_ids: list[int]) -> float | None:
        if not target_ids:
            return None
        prompt_text = self.tokenizer.decode(
            context_ids + target_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            f"{self.base_url}/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible scoring failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI-compatible scoring failed: {exc}") from exc
        choices = response.get("choices") or []
        if not choices:
            return None
        token_logprobs = (((choices[0] or {}).get("logprobs") or {}).get("token_logprobs")) or []
        start = len(context_ids)
        end = start + len(target_ids)
        values = [_float_or_none(value) for value in token_logprobs[start:end]]
        return _mean(values)


def _logprob_value(record: Any) -> float | None:
    if record is None:
        return None
    if isinstance(record, dict):
        return _float_or_none(record.get("logprob"))
    return _float_or_none(getattr(record, "logprob", record))


def _mean_prompt_logprobs(
    prompt_logprobs: list[Any],
    context_ids: list[int],
    target_ids: list[int],
) -> float | None:
    values: list[float] = []
    start = len(context_ids)
    for offset, token_id in enumerate(target_ids):
        idx = start + offset
        if idx >= len(prompt_logprobs):
            continue
        step = prompt_logprobs[idx]
        if not isinstance(step, dict):
            continue
        record = step.get(token_id)
        if record is None:
            record = step.get(str(token_id))
        if record is None and len(step) == 1:
            record = next(iter(step.values()))
        value = _logprob_value(record)
        if value is not None:
            values.append(value)
    return float(sum(values) / len(values)) if values else None


def _load_problem_map(config: BPAConfig, dataset: str) -> dict[str, EvalProblem]:
    return {str(problem.problem_id): problem for problem in load_eval_dataset(dataset, config)}


def _find_label_files(roots: list[Path]) -> list[tuple[Path, str, Path]]:
    files: list[tuple[Path, str, Path]] = []
    for root in roots:
        for path in sorted((root / "boundary_continuation").glob("*/boundary_labels.jsonl")):
            files.append((root, path.parent.name, path))
    return files


def _build_context_and_target_ids(
    *,
    tokenizer: Any,
    problem_text: str,
    assistant_prefix_text: str,
    other_rollout_texts: list[str],
    target_text: str,
    rollout_separator: str,
) -> tuple[list[int], list[int]]:
    augmented_prefix = assistant_prefix_text + rollout_separator.join(other_rollout_texts)
    rendered_context = render_for_continuation(problem_text, augmented_prefix, tokenizer)
    context_ids = list(tokenizer.encode(rendered_context, add_special_tokens=False))
    target_ids = list(tokenizer.encode(target_text, add_special_tokens=False))
    return context_ids, target_ids


def _safe_auroc(rows: list[dict[str, Any]], score_key: str, target_key: str) -> dict[str, Any]:
    pairs: list[tuple[float, int]] = []
    for row in rows:
        score = _float_or_none(row.get(score_key))
        target = _parse_bool(row.get(target_key))
        if score is not None and target is not None:
            pairs.append((score, int(target)))
    positives = sum(target for _, target in pairs)
    out = {"n": len(pairs), "positives": positives, "auroc": None}
    if len({target for _, target in pairs}) == 2:
        out["auroc"] = float(roc_auc_score([target for _, target in pairs], [score for score, _ in pairs]))
    return out


def _safe_pearson(rows: list[dict[str, Any]], left_key: str, right_key: str) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = _float_or_none(row.get(left_key))
        y = _float_or_none(row.get(right_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if len(xs) < 3:
        return {"n": len(xs), "r": None, "p": None}
    result = pearsonr(xs, ys)
    return {"n": len(xs), "r": float(result.statistic), "p": float(result.pvalue)}


def _logistic_xrd_position(rows: list[dict[str, Any]], target_key: str, position_key: str) -> dict[str, Any]:
    y_values: list[int] = []
    xrd_values: list[float] = []
    pos_values: list[float] = []
    for row in rows:
        y = _parse_bool(row.get(target_key))
        xrd = _float_or_none(row.get("xrd"))
        pos = _float_or_none(row.get(position_key))
        if y is not None and xrd is not None and pos is not None:
            y_values.append(int(y))
            xrd_values.append(xrd)
            pos_values.append(pos)
    if len(y_values) < 10 or len(set(y_values)) < 2:
        return {"n": len(y_values), "positives": sum(y_values), "coef": None, "p": None, "position_key": position_key}

    y = np.asarray(y_values, dtype=np.float64)
    xrd = _zscore(np.asarray(xrd_values, dtype=np.float64))
    pos = _zscore(np.asarray(pos_values, dtype=np.float64))
    x = np.column_stack([np.ones_like(y), xrd, pos])

    def nll(beta: np.ndarray) -> float:
        logits = np.clip(x @ beta, -40.0, 40.0)
        return float(np.sum(np.logaddexp(0.0, logits) - y * logits))

    result = minimize(nll, np.zeros(x.shape[1], dtype=np.float64), method="BFGS")
    beta = np.asarray(result.x, dtype=np.float64)
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -40.0, 40.0)))
    w = np.clip(p_hat * (1.0 - p_hat), 1e-9, None)
    fisher = x.T @ (x * w[:, None])
    try:
        cov = np.linalg.inv(fisher)
        se = float(math.sqrt(cov[1, 1]))
        z = float(beta[1] / se) if se > 0 else None
        p_value = float(2.0 * norm.sf(abs(z))) if z is not None else None
    except np.linalg.LinAlgError:
        se = None
        z = None
        p_value = None
    return {
        "n": len(y_values),
        "positives": int(sum(y_values)),
        "coef": float(beta[1]),
        "se": se,
        "z": z,
        "p": p_value,
        "position_key": position_key,
        "converged": bool(result.success),
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0.0 or not math.isfinite(std):
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _summarize(rows: list[dict[str, Any]], dataset: str, root_name: str) -> dict[str, Any]:
    slm_wrong = [row for row in rows if _parse_bool(row.get("slm_final_correct")) is False]
    slm_wrong_oracle_correct = [row for row in slm_wrong if _parse_bool(row.get("llm_oracle_correct")) is True]
    return {
        "root": root_name,
        "dataset": dataset,
        "n_rows": len(rows),
        "n_xrd": sum(_float_or_none(row.get("xrd")) is not None for row in rows),
        "critical_auroc": _safe_auroc(rows, "xrd", "critical"),
        "recovery_auroc_slm_wrong": _safe_auroc(slm_wrong, "xrd", "llm_continuation_correct"),
        "recovery_auroc_slm_wrong_oracle_correct": _safe_auroc(
            slm_wrong_oracle_correct,
            "xrd",
            "llm_continuation_correct",
        ),
        "pearson_xrd_char_jaccard": _safe_pearson(rows, "xrd", "char_jaccard_disagreement"),
        "pearson_xrd_internal_confidence": _safe_pearson(rows, "xrd", "internal_confidence_mean_logprob"),
        "logit_critical_xrd_control_boundary_idx": _logistic_xrd_position(rows, "critical", "boundary_idx"),
        "logit_recovery_xrd_control_boundary_idx_slm_wrong": _logistic_xrd_position(
            slm_wrong,
            "llm_continuation_correct",
            "boundary_idx",
        ),
        "logit_critical_xrd_control_prefix_token_len": _logistic_xrd_position(rows, "critical", "prefix_token_len"),
    }


def compute_xrd_for_file(
    *,
    root: Path,
    dataset: str,
    labels_path: Path,
    config: BPAConfig,
    tokenizer: Any,
    scorer: Scorer,
    cache: dict[str, Any],
    rollout_separator: str,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    problem_map = _load_problem_map(config, dataset)
    rows = _read_jsonl(labels_path)
    output_rows: list[dict[str, Any]] = []
    desc = f"xrd:{root.name}:{dataset}"
    for row in tqdm(rows, desc=desc):
        probe = row.get("probe") or {}
        rollouts = list(probe.get("rollouts") or [])
        base_logprobs = [_float_or_none(rollout.get("mean_logprob")) for rollout in rollouts]
        row_out = {
            "source_root": str(root),
            "dataset": dataset,
            "problem_id": row.get("problem_id"),
            "question_id": row.get("question_id"),
            "boundary_idx": row.get("boundary_idx"),
            "selected_rank": row.get("selected_rank"),
            "prefix_char_len": row.get("prefix_char_len"),
            "prefix_token_len": row.get("prefix_token_len"),
            "critical": row.get("critical"),
            "slm_final_correct": row.get("slm_final_correct"),
            "llm_oracle_correct": row.get("llm_oracle_correct"),
            "llm_continuation_correct": row.get("llm_continuation_correct"),
            "label_reason": row.get("label_reason"),
            "char_jaccard_disagreement": row.get("char_jaccard_disagreement"),
            "structured_disagreement": row.get("structured_disagreement"),
            "score_variance": probe.get("score_variance", row.get("score_variance")),
            "internal_confidence_mean_logprob": _mean(base_logprobs),
            "internal_confidence_logprob_std": _std(base_logprobs),
        }
        if len(rollouts) != 4 or any(value is None for value in base_logprobs):
            row_out["xrd_error"] = "requires_four_rollouts_with_base_mean_logprob"
            output_rows.append(row_out)
            continue
        problem = problem_map.get(str(row.get("problem_id")))
        if problem is None:
            row_out["xrd_error"] = "missing_problem_text"
            output_rows.append(row_out)
            continue

        assistant_prefix = str(probe.get("assistant_prefix_text") or row.get("assistant_prefix_text") or "")
        rollout_texts = [str(rollout.get("text") or "") for rollout in rollouts]
        augmented_logprobs: list[float | None] = []
        deltas: list[float | None] = []
        for idx, target_text in enumerate(rollout_texts):
            other_texts = [text for j, text in enumerate(rollout_texts) if j != idx]
            context_ids, target_ids = _build_context_and_target_ids(
                tokenizer=tokenizer,
                problem_text=problem.problem_text,
                assistant_prefix_text=assistant_prefix,
                other_rollout_texts=other_texts,
                target_text=target_text,
                rollout_separator=rollout_separator,
            )
            key = _cache_key(row, idx, context_ids, target_ids)
            if key in cache:
                augmented = _float_or_none(cache[key].get("augmented_mean_logprob"))
            else:
                augmented = scorer.score_target(context_ids, target_ids)
                cache[key] = {
                    "augmented_mean_logprob": augmented,
                    "target_token_count": len(target_ids),
                    "context_token_count": len(context_ids),
                    "time": time.time(),
                }
                if sleep_seconds > 0.0:
                    time.sleep(sleep_seconds)
            augmented_logprobs.append(augmented)
            base = base_logprobs[idx]
            deltas.append((augmented - base) if augmented is not None and base is not None else None)
        row_out["augmented_mean_logprobs"] = json.dumps(augmented_logprobs, ensure_ascii=False)
        row_out["xrd_component_scores"] = json.dumps(deltas, ensure_ascii=False)
        row_out["xrd"] = _mean(deltas)
        row_out["xrd_error"] = "" if row_out["xrd"] is not None else "missing_augmented_score"
        output_rows.append(row_out)
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute augmented-context likelihood XRD for K=4 boundary rollouts.")
    parser.add_argument(
        "--diagnostics-root",
        action="append",
        required=True,
        help="Diagnostics root containing boundary_continuation/<dataset>/boundary_labels.jsonl. Repeatable.",
    )
    parser.add_argument("--config", default="configs/bpa_default.json")
    parser.add_argument("--output-dir", default="analysis/xrd_augmented_likelihood")
    parser.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    parser.add_argument("--model-path", default=None, help="Override config.slm_model_path for vLLM/tokenizer loading.")
    parser.add_argument("--tokenizer-path", default=None, help="Override config tokenizer path.")
    parser.add_argument("--api-base-url", default=None, help="OpenAI-compatible base URL, e.g. http://host:8000/v1.")
    parser.add_argument("--api-model", default=None, help="OpenAI-compatible served model name.")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--cuda-visible-devices",
        default=1,
        help="Override config.slm_device for local vLLM, e.g. 1 or 2,3. vLLM will see these as logical cuda:0,...",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--rollout-separator", default="")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    config = BPAConfig.from_json(args.config)
    roots = [Path(path) for path in args.diagnostics_root]
    out_dir = Path(args.output_dir)
    cache_path = Path(args.cache_path) if args.cache_path else out_dir / "xrd_forward_cache.json"
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    model_path = args.model_path or config.slm_model_path
    tokenizer_path = args.tokenizer_path or config.slm_tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=config.trust_remote_code, use_fast=True)

    if args.backend == "vllm":
        engine_kwargs = dict(config.slm_engine_kwargs)
        if args.gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        scorer: Scorer = VllmScorer(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            trust_remote_code=config.trust_remote_code,
            max_model_len=args.max_model_len or config.max_model_len,
            enable_prefix_caching=config.enable_prefix_caching,
            engine_kwargs=engine_kwargs,
            cuda_visible_devices=args.cuda_visible_devices or config.slm_device,
        )
    else:
        base_url = args.api_base_url or config.slm_api_base_url
        api_model = args.api_model or config.slm_api_model or model_path
        if not base_url:
            raise SystemExit("--api-base-url is required for --backend openai when config.slm_api_base_url is unset.")
        scorer = OpenAICompletionScorer(base_url=base_url, model=api_model, tokenizer=tokenizer, api_key=args.api_key)

    all_summary: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    try:
        for root, dataset, labels_path in _find_label_files(roots):
            rows = compute_xrd_for_file(
                root=root,
                dataset=dataset,
                labels_path=labels_path,
                config=config,
                tokenizer=tokenizer,
                scorer=scorer,
                cache=cache,
                rollout_separator=args.rollout_separator,
                sleep_seconds=args.sleep_seconds,
            )
            rel = Path(root.name) / dataset
            _write_jsonl(out_dir / rel / "xrd_boundary_scores.jsonl", rows)
            _write_csv(out_dir / rel / "xrd_boundary_scores.csv", rows)
            summary = _summarize(rows, dataset=dataset, root_name=root.name)
            all_summary.append(summary)
            all_rows.extend(rows)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    for dataset in sorted({row["dataset"] for row in all_rows}):
        pooled = [row for row in all_rows if row["dataset"] == dataset]
        all_summary.append(_summarize(pooled, dataset=dataset, root_name="POOLED"))
    (out_dir / "summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "summary_flat.csv", [_flatten_summary(row) for row in all_summary])
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


def _flatten_summary(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in _flatten_summary(value).items():
                out[f"{key}.{sub_key}"] = sub_value
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
