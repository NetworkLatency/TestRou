# %%
import os
import pprint
import logging
import time
import numpy as np
from openai import OpenAI
import statistics
from collections import Counter
import json
import re


ROUTING_MODES = ("vanilla", "fa_skip", "fa_strip")
MAX_FORMAT_SKIP = 3
DEFAULT_TOP_LOGPROBS = 50
KEEP_SLM = "KEEP_SLM"
HANDOFF = "HANDOFF"


def get_avg_score(scores):
    # Mean over non-null scores.
    return statistics.mean([x for x in scores if x is not None])


def get_frequency(scores):
    # Count frequency of score values.
    return dict(Counter(scores))


def get_model(model_size):
    return model_names[model_size]


# %%
model_names = {
    "32b": "YOUR_MODEL_NAME_FOR_32B",  # NOTE: change to the name of your 32b large model, e.g. "org/model-32b"
    "4b": "YOUR_MODEL_NAME_FOR_4B",  # NOTE: change to the name of your 4b small model
}

ports = {
    "32b": "YOUR_PORT_FOR_32B",  # NOTE: change to the port of your 32b large model, e.g. "11125"
    "4b": "YOUR_PORT_FOR_4B",  # NOTE: change to the port of your 4b small model, e.g. "11130"
}

clients = {}
for size, full_name in model_names.items():
    # OpenAI-compatible client; replace placeholders with your local endpoint.
    clients[size] = OpenAI(
        api_key="YOUR_API_KEY",  # NOTE: change to the api key of your model
        base_url="YOUR_BASE_URL",  # NOTE: change to the base url of your model, e.g. f"http://localhost:{ports[size]}/v1"
    )


def get_first_user_msg(problem, options=None):
    if options == "aime" or options == "math":
        system_prompt = "Solve the following math problem and return ONLY the final answer.\nPlease reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.\n\n"
        system_prompt += f"Problem: {problem['problem']}\n\n"
        return system_prompt
    elif options == "lcb":
        raw_prompt = problem["question_content"]
        starter = problem["starter_code"]
        system_prompt = "Write code to solve the following problem and return ONLY the code.\nYou will generate a correct Python program that matches the specification and passes all tests.\n\n"
        system_prompt += f"Question: {raw_prompt}\n\n"
        if starter:
            system_prompt += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            system_prompt += f"```python\n{starter}\n```\n\n"
        else:
            system_prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n"
            system_prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
        return system_prompt
    elif options == "gpqa":
        system_prompt = "What is the correct answer to the following problem? Please reason step by step.\nSeparate logical reasoning steps with two newline characters (\n\n).\nPut the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).\n\n**Example output:** \\boxed{{A}}\n\n"
        system_prompt += f"Problem: {problem['problem']}\n\n"
        return system_prompt
    else:
        raise NotImplementedError


def _get_attr(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_completion_tokens(response):
    usage = _get_attr(response, "usage")
    tokens = _get_attr(usage, "completion_tokens")
    return int(tokens) if tokens is not None else 0


def _extract_first_token_latency_s(response):
    """Best-effort vLLM/OpenAI-compatible first-token latency extraction."""
    metrics = _get_attr(response, "metrics")
    if metrics is not None:
        latency = _get_attr(metrics, "first_token_latency")
        if latency is not None:
            return float(latency)
        first_token_time = _get_attr(metrics, "first_token_time")
        arrival_time = _get_attr(metrics, "arrival_time")
        if first_token_time is not None and arrival_time is not None:
            return max(0.0, float(first_token_time) - float(arrival_time))
    return 0.0


def _reasoning_prefix(steps_so_far, prefix_text=""):
    steps_so_far_str = "\n\n".join(steps_so_far)
    if steps_so_far_str:
        steps_so_far_str += "\n\n"
    # Qwen3 needs an explicit thinking prefill; this also keeps step boundaries stable.
    return f"<think>{steps_so_far_str}{prefix_text}"


def _build_step_messages(problem, steps_so_far, options=None, prefix_text=""):
    return [
        {"role": "user", "content": get_first_user_msg(problem, options)},
        {"role": "assistant", "content": _reasoning_prefix(steps_so_far, prefix_text)},
    ]


def _step_extra_body():
    return {"add_generation_prompt": False, "continue_final_message": True}


# %%
def generate_new_step(problem, steps_so_far, model_size, options=None, stop_token="\n\n", prefix_text=""):
    client = clients[model_size]

    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=_build_step_messages(problem, steps_so_far, options, prefix_text),
        temperature=0.6,
        top_p=0.95,
        max_tokens=512,
        stop=[stop_token],
        extra_body=_step_extra_body(),
    )

    continuation = response.choices[0].message.content or ""
    step_str = prefix_text + continuation
    num_output_tokens = _get_completion_tokens(response)
    finished = "</think>" in step_str

    return step_str, finished, num_output_tokens, response


def generate_answer(problem, steps_so_far, model_size, options=None):
    client = clients[model_size]

    steps_so_far_str = "\n\n".join(steps_so_far)
    steps_so_far_str = steps_so_far_str.split("</think>")[0] if "</think>" in steps_so_far_str else steps_so_far_str

    # Always finalize with the large model to produce the answer.
    messages = [
        {"role": "user", "content": get_first_user_msg(problem, options)},
        {"role": "assistant", "content": f"<think>{steps_so_far_str}\n</think>\n\n"},
    ]
    extra_body = {"add_generation_prompt": False, "continue_final_message": True}

    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        extra_body=extra_body,
    )

    step_str = response.choices[0].message.content
    num_output_tokens = _get_completion_tokens(response)
    if options == "lcb":
        s = re.findall(r'```(?:python)?\n(.*?)```', step_str, re.DOTALL | re.IGNORECASE)
        finished = len(s) >= 1
    else:
        finished = any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]])

    return step_str, finished, num_output_tokens


def process_logprobs(response, method, temp=1.0):
    # Extract logprobs for the first generated token.
    assert len(response.choices[0].logprobs.content) == 1
    token = response.choices[0].logprobs.content[0].token
    token_logprobs = {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}
    token_logprobs = {k: v for k, v in token_logprobs.items() if k.isdigit()}  # filter out non-digit values

    if method == "greedy":
        # return the vanilla response
        if not token.isdigit():
            return 0
        return int(token)
    elif method == "average":
        # Convert log probabilities to probabilities and normalize each distribution.
        probs = {tok: np.exp(lp / temp) for tok, lp in token_logprobs.items()}
        total_probs = sum(probs.values())
        for tok in probs:
            probs[tok] /= total_probs
        for i in range(10):
            if i not in probs:
                probs[i] = 0
        return sum([int(t) * p for t, p in probs.items()])
    else:
        raise NotImplementedError


def entropy_from_logprobs(logprobs):
    if not logprobs:
        return 0.0
    finite_lps = [lp for lp in logprobs if np.isfinite(lp)]
    if not finite_lps:
        return 0.0
    max_lp = max(finite_lps)
    weights = np.array([np.exp(lp - max_lp) if np.isfinite(lp) else 0.0 for lp in logprobs], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    probs = weights / total
    return float(-np.sum(probs * np.log(probs + 1e-45)))


def _extract_token_logprobs(response):
    content_logprobs = response.choices[0].logprobs.content
    assert len(content_logprobs) == 1
    token_info = content_logprobs[0]
    token_str = _get_attr(token_info, "token", response.choices[0].message.content or "")
    token_id = _get_attr(token_info, "token_id")
    top_records = []

    for item in _get_attr(token_info, "top_logprobs", []) or []:
        top_records.append(
            {
                "token": _get_attr(item, "token", ""),
                "token_id": _get_attr(item, "token_id"),
                "logprob": float(_get_attr(item, "logprob", float("-inf"))),
            }
        )

    if not any(record["token"] == token_str for record in top_records):
        logprob = _get_attr(token_info, "logprob")
        if logprob is not None:
            top_records.append({"token": token_str, "token_id": token_id, "logprob": float(logprob)})

    return token_str, token_id, top_records


def _load_format_token_whitelist(format_tokens_path, required=False):
    if not format_tokens_path:
        if required:
            raise ValueError("format_tokens_path is required for FA routing modes.")
        return {"ids": set(), "strings": set(), "str_to_id": {}}

    if not os.path.exists(format_tokens_path):
        if required:
            raise FileNotFoundError(
                f"Format token whitelist not found: {format_tokens_path}. "
                "Run src/build_format_tokens.py first and inspect the decoded tokens."
            )
        return {"ids": set(), "strings": set(), "str_to_id": {}}

    with open(format_tokens_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = {int(tid) for tid in data.get("ids", [])}
    decoded = {}
    for key, value in data.get("decoded", {}).items():
        decoded[int(key)] = value
    strings = set(decoded.values())
    strings.update(data.get("strings", []))
    str_to_id = {value: tid for tid, value in decoded.items()}
    return {"ids": ids, "strings": strings, "str_to_id": str_to_id}


def _resolve_token_id(token_str, token_id, format_whitelist):
    if token_id is not None:
        return int(token_id)
    return format_whitelist.get("str_to_id", {}).get(token_str)


def _is_format_token(token_str, token_id, format_whitelist):
    if token_id is not None and int(token_id) in format_whitelist.get("ids", set()):
        return True
    return token_str in format_whitelist.get("strings", set())


def _top_non_format(top_records, format_whitelist):
    candidates = [
        record
        for record in top_records
        if not _is_format_token(record["token"], record.get("token_id"), format_whitelist)
    ]
    if not candidates:
        return None, []
    candidates = sorted(candidates, key=lambda record: record["logprob"], reverse=True)
    return candidates[0], candidates


def _decision_from_entropy(entropy, tau):
    if entropy is None:
        return None
    return HANDOFF if entropy > tau else KEEP_SLM


def glimpse_one_token(problem, steps_so_far, model_size="4b", options=None, prefix_text="", top_logprobs=DEFAULT_TOP_LOGPROBS):
    client = clients[model_size]

    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=_build_step_messages(problem, steps_so_far, options, prefix_text),
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
        extra_body=_step_extra_body(),
    )

    token_str, token_id, top_records = _extract_token_logprobs(response)
    entropy = entropy_from_logprobs([record["logprob"] for record in top_records])

    return {
        "entropy": entropy,
        "token": token_str,
        "token_id": token_id,
        "top_logprobs": top_records,
        "response": response,
    }


def get_score_first_token_entropy(problem, steps_so_far, model_size="4b", options=None, top_logprobs=DEFAULT_TOP_LOGPROBS):
    result = glimpse_one_token(
        problem,
        steps_so_far,
        model_size=model_size,
        options=options,
        prefix_text="",
        top_logprobs=top_logprobs,
    )
    return result["entropy"], result["token"], result["response"]


def get_score(score_method, problem, steps_so_far, model_size="32b", options=None, top_logprobs=DEFAULT_TOP_LOGPROBS):
    if score_method == "first_token_entropy":
        return get_score_first_token_entropy(
            problem,
            steps_so_far,
            model_size=model_size,
            options=options,
            top_logprobs=top_logprobs,
        )
    else:
        raise NotImplementedError


def _token_record_from_glimpse(glimpse, format_whitelist):
    token_str = glimpse["token"]
    token_id = _resolve_token_id(token_str, glimpse.get("token_id"), format_whitelist)
    return {
        "token": token_str,
        "token_id": token_id,
        "entropy": glimpse["entropy"],
    }


def _token_record_from_top_logprob(record, entropy, format_whitelist):
    token_str = record["token"]
    token_id = _resolve_token_id(token_str, record.get("token_id"), format_whitelist)
    return {
        "token": token_str,
        "token_id": token_id,
        "entropy": entropy,
    }


def glimpse_and_decide(
    problem,
    steps_so_far,
    tau,
    model_size="4b",
    options=None,
    routing_mode="vanilla",
    format_whitelist=None,
    max_format_skip=MAX_FORMAT_SKIP,
    top_logprobs=DEFAULT_TOP_LOGPROBS,
):
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"Unknown routing_mode={routing_mode!r}; expected one of {ROUTING_MODES}.")
    format_whitelist = format_whitelist or {"ids": set(), "strings": set(), "str_to_id": {}}

    skipped_tokens = []
    prefix_text = ""
    raw = glimpse_one_token(
        problem,
        steps_so_far,
        model_size=model_size,
        options=options,
        prefix_text=prefix_text,
        top_logprobs=top_logprobs,
    )
    raw_token = _token_record_from_glimpse(raw, format_whitelist)
    raw_H_init = raw["entropy"]
    content_token = raw_token
    H_content_init = raw_H_init

    if routing_mode == "fa_skip":
        current = raw
        for skip_idx in range(max_format_skip + 1):
            current_token = _token_record_from_glimpse(current, format_whitelist)
            is_format = _is_format_token(current_token["token"], current_token.get("token_id"), format_whitelist)
            if is_format and skip_idx < max_format_skip:
                skipped_tokens.append(current_token)
                prefix_text += current_token["token"]
                current = glimpse_one_token(
                    problem,
                    steps_so_far,
                    model_size=model_size,
                    options=options,
                    prefix_text=prefix_text,
                    top_logprobs=top_logprobs,
                )
                continue
            content_token = current_token
            H_content_init = current["entropy"]
            break
    elif routing_mode == "fa_strip":
        top_non_format, non_format_records = _top_non_format(raw["top_logprobs"], format_whitelist)
        if top_non_format is not None:
            H_content_init = entropy_from_logprobs([record["logprob"] for record in non_format_records])
            content_token = _token_record_from_top_logprob(top_non_format, H_content_init, format_whitelist)
        else:
            logging.warning("fa_strip could not find a non-format candidate in top_logprobs; falling back to raw token.")

    glimpse_tokens = skipped_tokens + [content_token]
    decision_raw = _decision_from_entropy(raw_H_init, tau)
    decision_fa = _decision_from_entropy(H_content_init, tau)
    actual_decision = decision_raw if routing_mode == "vanilla" else decision_fa

    log_entry = {
        "raw_H_init": raw_H_init,
        "H_content_init": H_content_init,
        "n_format_skipped": len(skipped_tokens),
        "skipped_token_ids": [token.get("token_id") for token in skipped_tokens],
        "skipped_token_strs": [token["token"] for token in skipped_tokens],
        "raw_token_id": raw_token.get("token_id"),
        "raw_token_str": raw_token["token"],
        "content_token_id": content_token.get("token_id"),
        "content_token_str": content_token["token"],
        "tau": tau,
        "decision_raw": decision_raw,
        "decision_fa": decision_fa,
        "actual_decision": actual_decision,
        "routing_mode": routing_mode,
        "glimpse_token_strs": [token["token"] for token in glimpse_tokens],
        "glimpse_token_ids": [token.get("token_id") for token in glimpse_tokens],
    }

    return actual_decision, glimpse_tokens, log_entry


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def glimprouter(
    problem,
    options=None,
    dataset_name="aime24",
    score_threshold=1.0,
    token_budget=8192,
    problem_id=0,
    repeat_id=0,
    score_method="first_token_entropy",
    output_dir="./results",
    first_n_steps_base_model=0,
    model_size="32b",
    small_model_size="4b",
    routing_mode="vanilla",
    format_tokens_path="format_tokens.json",
    max_format_skip=MAX_FORMAT_SKIP,
    top_logprobs=DEFAULT_TOP_LOGPROBS,
):
    problem_uid = f"{dataset_name}/{problem_id}"
    output_filename = os.path.join(output_dir, f"{problem_uid}/{repeat_id}")

    if os.path.exists(f"{output_filename}.json"):
        with open(f"{output_filename}.json", "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        return metadata_list

    format_whitelist = _load_format_token_whitelist(format_tokens_path, required=routing_mode != "vanilla")
    steps_so_far = []
    step_id = 0
    metadata_list = []
    step_logs = []
    prev_step_provenance = "START"
    total_slm_tokens = 0
    total_llm_tokens = 0
    n_handoffs = 0
    n_slm_steps = 0
    n_llm_steps = 0
    problem_start = time.perf_counter()

    try:
        while True:
            score = None
            justification = None
            base_model_step = None
            small_model_step = None
            num_output_tokens_base = None
            num_output_tokens_small = None
            t_glimpse_ms = 0.0
            t_step_generation_ms = 0.0
            t_handoff_prefill_ms = 0.0
            route_log = {
                "raw_H_init": None,
                "H_content_init": None,
                "n_format_skipped": 0,
                "skipped_token_ids": [],
                "skipped_token_strs": [],
                "content_token_id": None,
                "content_token_str": None,
                "tau": score_threshold,
                "decision_raw": None,
                "decision_fa": None,
                "actual_decision": HANDOFF,
                "routing_mode": routing_mode,
                "glimpse_token_strs": [],
                "glimpse_token_ids": [],
            }

            if step_id < first_n_steps_base_model:
                t1 = time.perf_counter()
                base_model_step, finished, num_output_tokens_base, response = generate_new_step(
                    problem,
                    steps_so_far,
                    model_size,
                    options=options,
                )
                t_step_generation_ms = (time.perf_counter() - t1) * 1000
                t_handoff_prefill_ms = _extract_first_token_latency_s(response) * 1000
                step_str = base_model_step
                step_n_tokens = num_output_tokens_base
                total_llm_tokens += step_n_tokens
                n_handoffs += 1
                n_llm_steps += 1
            elif score_method == "first_token_entropy":
                t0 = time.perf_counter()
                actual_decision, glimpse_tokens, route_log = glimpse_and_decide(
                    problem,
                    steps_so_far,
                    score_threshold,
                    model_size=small_model_size,
                    options=options,
                    routing_mode=routing_mode,
                    format_whitelist=format_whitelist,
                    max_format_skip=max_format_skip,
                    top_logprobs=top_logprobs,
                )
                t_glimpse_ms = (time.perf_counter() - t0) * 1000

                glimpse_text = "".join(token["token"] for token in glimpse_tokens)
                glimpse_n_tokens = len(glimpse_tokens)
                score = route_log["raw_H_init"] if routing_mode == "vanilla" else route_log["H_content_init"]
                justification = route_log["content_token_str"]

                t1 = time.perf_counter()
                if actual_decision == HANDOFF:
                    # large model continues after the SLM-glimpsed prefix
                    base_model_step, finished, num_output_tokens_base, response = generate_new_step(
                        problem,
                        steps_so_far,
                        model_size,
                        options=options,
                        prefix_text=glimpse_text,
                    )
                    t_handoff_prefill_ms = _extract_first_token_latency_s(response) * 1000
                    small_model_step = None
                    num_output_tokens_small = glimpse_n_tokens
                    step_str = base_model_step
                    total_slm_tokens += glimpse_n_tokens
                    total_llm_tokens += num_output_tokens_base
                    n_handoffs += 1
                    n_llm_steps += 1
                else:
                    # small model keeps generating after its own glimpsed prefix
                    small_model_step, finished, continuation_tokens, response = generate_new_step(
                        problem,
                        steps_so_far,
                        small_model_size,
                        options=options,
                        prefix_text=glimpse_text,
                    )
                    num_output_tokens_small = glimpse_n_tokens + continuation_tokens
                    base_model_step = None
                    num_output_tokens_base = None
                    step_str = small_model_step
                    total_slm_tokens += num_output_tokens_small
                    n_slm_steps += 1
                t_step_generation_ms = (time.perf_counter() - t1) * 1000
                step_n_tokens = (num_output_tokens_base or 0) + (num_output_tokens_small or 0)
            else:
                raise NotImplementedError

            steps_so_far.append(step_str)
            generator = "LLM" if base_model_step is not None else "SLM"

            # collect metadata
            metadata = {
                "step_id": step_id,
                "step_idx": step_id,
                "step_str": step_str,
                "small_model_step": small_model_step,
                "num_output_tokens_small": num_output_tokens_small,
                "score": score,
                "base_model_step": base_model_step,
                "num_output_tokens_base": num_output_tokens_base,
                "final_num_output_tokens": step_n_tokens,
                "justification": justification,
                "routing_mode": routing_mode,
                "routing_log": route_log,
                "generator": generator,
                "t_glimpse_ms": t_glimpse_ms,
                "t_step_generation_ms": t_step_generation_ms,
                "t_handoff_prefill_ms": t_handoff_prefill_ms,
            }
            metadata_list.append(metadata)

            step_log = {
                "problem_id": problem_uid,
                "repeat_id": repeat_id,
                "step_idx": step_id,
                "prev_step_provenance": prev_step_provenance,
                **route_log,
                "step_text": step_str,
                "step_n_tokens": step_n_tokens,
                "generator": generator,
                "t_glimpse_ms": t_glimpse_ms,
                "t_step_generation_ms": t_step_generation_ms,
                "t_handoff_prefill_ms": t_handoff_prefill_ms,
                "is_final_answer": False,
            }
            step_logs.append(step_log)
            prev_step_provenance = generator
            step_id += 1

            # Check if finished
            if len(steps_so_far) > 2:
                finished = finished or steps_so_far[-1] == steps_so_far[-2]

            if sum(m["final_num_output_tokens"] for m in metadata_list) >= token_budget:
                finished = True
                metadata_list[-1]["stop_reason"] = "budget"
                step_logs[-1]["stop_reason"] = "budget"
            elif finished:
                metadata_list[-1]["stop_reason"] = "finished"
                step_logs[-1]["stop_reason"] = "finished"

            if finished:
                break

        # Generation of Final Answer
        t1 = time.perf_counter()
        base_model_step, finished, num_output_tokens_base = generate_answer(
            problem, steps_so_far, model_size, options=options
        )
        t_step_generation_ms = (time.perf_counter() - t1) * 1000
        small_model_step, num_output_tokens_small = None, None
        score, justification = None, None
        step_str = base_model_step
        steps_so_far.append(step_str)
        total_llm_tokens += num_output_tokens_base

        metadata = {
            "step_id": step_id,
            "step_idx": step_id,
            "step_str": step_str,
            "small_model_step": small_model_step,
            "num_output_tokens_small": num_output_tokens_small,
            "score": score,
            "base_model_step": base_model_step,
            "num_output_tokens_base": num_output_tokens_base,
            "final_num_output_tokens": num_output_tokens_base,
            "justification": justification,
            "answer": finished,
            "routing_mode": routing_mode,
            "generator": "LLM",
            "t_glimpse_ms": 0.0,
            "t_step_generation_ms": t_step_generation_ms,
            "t_handoff_prefill_ms": 0.0,
        }
        metadata_list.append(metadata)

        step_logs.append(
            {
                "problem_id": problem_uid,
                "repeat_id": repeat_id,
                "step_idx": step_id,
                "prev_step_provenance": prev_step_provenance,
                "raw_H_init": None,
                "H_content_init": None,
                "n_format_skipped": 0,
                "skipped_token_ids": [],
                "skipped_token_strs": [],
                "content_token_id": None,
                "content_token_str": None,
                "tau": score_threshold,
                "decision_raw": None,
                "decision_fa": None,
                "actual_decision": "FINAL_ANSWER",
                "routing_mode": routing_mode,
                "step_text": step_str,
                "step_n_tokens": num_output_tokens_base,
                "generator": "LLM",
                "t_glimpse_ms": 0.0,
                "t_step_generation_ms": t_step_generation_ms,
                "t_handoff_prefill_ms": 0.0,
                "is_final_answer": True,
            }
        )

    except ValueError:
        logging.error("ValueError caught in chat template application, continuing")

    total_tokens = total_slm_tokens + total_llm_tokens
    problem_log = {
        "problem_id": problem_uid,
        "repeat_id": repeat_id,
        "dataset_name": dataset_name,
        "routing_mode": routing_mode,
        "tau": score_threshold,
        "total_wall_time_s": time.perf_counter() - problem_start,
        "total_slm_tokens": total_slm_tokens,
        "total_llm_tokens": total_llm_tokens,
        "llm_token_share": (total_llm_tokens / total_tokens) if total_tokens else 0.0,
        "n_handoffs": n_handoffs,
        "n_slm_steps": n_slm_steps,
        "n_llm_steps": n_llm_steps,
        "final_answer": metadata_list[-1]["step_str"] if metadata_list else "",
        "is_correct": None,
        "steps": step_logs,
    }

    # save results
    os.makedirs(os.path.dirname(f"{output_filename}.json"), exist_ok=True)

    with open(f"{output_filename}.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata_list), f, ensure_ascii=False, indent=4)

    with open(f"{output_filename}.problem.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(problem_log), f, ensure_ascii=False, indent=4)

    _write_jsonl(f"{output_filename}.steps.jsonl", step_logs)

    with open(f"{output_filename}.txt", "w", encoding="utf-8") as f:
        pprint.pprint(_json_safe(metadata_list), stream=f)

    return metadata_list
