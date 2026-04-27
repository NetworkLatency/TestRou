# BPA v2.1: Branch Probing & Arbitration — Implementation Specification

**前置约定**：vLLM 后端、PyTorch、两个独立 vLLM 实例、跨家族协同、benchmarks = MATH500 / AIME24+25 / GPQA-Diamond / HumanEval，主模型对 = DeepSeek-R1-Distill-Qwen-1.5B (SLM) + Qwen-32B (LLM)。

---

## 1. State Schema 与 Render Contract

### 1.1 主状态

```python
@dataclass
class GenerationState:
    # User problem，全程不变
    problem_text: str
    
    # Assistant 侧的 canonical text。这是跨模型的唯一可信状态。
    # 初始化为 "" — chat template 会自动加 <think>（thinking 模型）或 nothing（非-thinking）
    assistant_prefix_text: str
    
    # 状态机
    phase: Phase                          # THINKING | FINAL_ANSWER | DONE
    has_seen_close_think: bool = False    # 是否已经检测到 </think>
    
    # 进度统计
    step_count: int = 0
    
    # 成本统计（拆分）
    slm_decode_tokens: int = 0            # SLM 生成的 token（包括 shadow rollout）
    slm_prefill_tokens: int = 0           # SLM 的 prefill 等价 token（first call 真实 prefill, 后续 cache 命中部分按 0 算）
    llm_decode_tokens: int = 0            # LLM 生成 token（仅在 LLM_FULL 模式产生）
    llm_prefill_tokens: int = 0           # LLM scoring 的 prefill 等价 token（每次 scoring 算 len(token_ids)）
    llm_scoring_calls: int = 0            # LLM scoring 调用次数（每个 branch 一次）
    llm_full_calls: int = 0               # LLM full step generation 次数
    
    # diagnostic
    trace: list[TraceEvent] = field(default_factory=list)
    rejected_branches_log: list[RejectedBranch] = field(default_factory=list)


class Phase(Enum):
    THINKING = "thinking"            # 在 <think>...</think> 段内，BPA 全功能
    FINAL_ANSWER = "final_answer"    # 已检测到 </think>，最终答案由 LLM 一次性生成
    DONE = "done"
```

**强约束**：
- `assistant_prefix_text` **不**包含 user problem。它只包含 `<think>...` 段及之后的 assistant 内容。
- `assistant_prefix_text` 初始化为 `""`。chat template 在 assistant turn 开头自动加 `<think>` 标签（DeepSeek-R1 distill / Qwen3-thinking 系列），我们不手工拼。
- 任何模型要看 prefix 时，都用各自 tokenizer + chat template + `continue_final_message=True` 重新渲染。

### 1.2 Render Contract

```python
def render_for_continuation(
    problem_text: str,
    assistant_prefix_text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """
    返回：tokenizer 渲染后的字符串，可直接喂给 vLLM 作为 prompt。
    使用 continue_final_message=True 避免 EOS 被插入。
    """
    messages = [
        {"role": "user", "content": problem_text},
        {"role": "assistant", "content": assistant_prefix_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
        add_generation_prompt=False,
    )
```

`assistant_prefix_text=""` 时，模型实际看到的 assistant content 由 chat template 决定（R1-distill 会自动加 `<think>`）。

### 1.3 跨家族陷阱速查（v2 表保留）

| 陷阱 | 现象 | 处理 |
|---|---|---|
| chat template 自动加 EOS | LLM 看到 prefix 末尾 EOS 直接停 | `continue_final_message=True` |
| BOS 双重添加 | apply_chat_template 加了 BOS，vLLM 又加 | tokenize 后用 token_ids 输入，绕过 vLLM 的 add_special_tokens |
| `\n\n` 在不同 tokenizer 下不同 token 化 | 主状态文本侧一致，token 不必对齐 | canonical 是 text，scoring 走 char offset |
| `<think>` 在 SLM/LLM 上 special token 状态不同 | rendered 长度差异 | 不依赖 token 数对齐，只用字符 offset |
| `continue_final_message=True` 与 `add_generation_prompt=True` 互斥 | 报错 | 只设前者 |

### 1.4 Thinking / Final 状态机

```python
CLOSE_THINK_TAG = "</think>"

def detect_close_think(prev_assistant_prefix: str, newly_added_text: str) -> tuple[bool, int]:
    """
    检测新增内容是否包含 </think>。返回 (是否检测到, 在 newly_added_text 中的字符位置)。
    要包含跨段情形：</think> 可能跨在 prev 末尾和 new 开头。
    """
    combined = prev_assistant_prefix + newly_added_text
    idx = combined.find(CLOSE_THINK_TAG, max(0, len(prev_assistant_prefix) - len(CLOSE_THINK_TAG)))
    if idx < 0:
        return False, -1
    # 转换为 newly_added_text 内的位置
    rel_idx = max(0, idx - len(prev_assistant_prefix))
    return True, rel_idx
```

**状态切换规则**：
- 初始 `phase = THINKING`
- 每次 step 写入 `assistant_prefix_text` 之后扫描新增部分检测 `</think>`
- 检测到 `</think>`：
- 截断 `assistant_prefix_text` 到 `</think>` 结尾
  - 丢弃 `</think>` 之后已生成的内容（trace 记录）
  - 转 `phase = FINAL_ANSWER`
  - `FINAL_ANSWER` phase 一次性调用 `_llm_generate_final` 写到 EOS 或被 repetition guard 终止
- SLM 是非-thinking 模型时永远停在 `THINKING`，整道题在 step loop 内由 `eos` 自然终止


```python
def check_and_transition_phase(state: GenerationState, newly_added_text: str) -> None:
    if state.phase != Phase.THINKING:
        return
    
    found, abs_idx = detect_close_think(
        state.assistant_prefix_text[:-len(newly_added_text)],
        newly_added_text,
    )
    if not found:
        return
    
    state.has_seen_close_think = True
    
    # 截断 assistant_prefix_text 到 </think> 结束位置（含 </think> 标签本身）
    close_end = abs_idx + len(CLOSE_THINK_TAG)
    discarded_tail = state.assistant_prefix_text[close_end:]
    state.assistant_prefix_text = state.assistant_prefix_text[:close_end]
    
    if discarded_tail:
        state.trace.append(TraceEvent(
            state.step_count, "phase_to_final_answer_discard_tail",
            {"discarded_chars": len(discarded_tail), "discarded_preview": discarded_tail[:200]},
        ))
    else:
        state.trace.append(TraceEvent(state.step_count, "phase_to_final_answer", {}))
    
    state.phase = Phase.FINAL_ANSWER
```

---

## 2. Step Generation Loop

### 2.1 主循环

```python
def bpa_solve(problem_text: str, slm: ModelEngine, llm: ModelEngine,
              config: BPAConfig) -> BPAResult:
    state = GenerationState(
        problem_text=problem_text,
        assistant_prefix_text="",
        phase=Phase.THINKING,
    )
    rep = RepetitionState()
    start_time = time.time()
    
    while state.phase != Phase.DONE:
        # 全局预算
        if state.slm_decode_tokens + state.llm_decode_tokens >= config.max_total_tokens:
            state.trace.append(TraceEvent(state.step_count, "budget_exhausted_total", {}))
            break
        
        if state.phase == Phase.FINAL_ANSWER:
            # LLM 一次性生成最终答案，附带 repetition guard
            final_text, final_stop_reason = _llm_generate_final_with_rep_guard(state, llm, config)
            state.assistant_prefix_text += final_text
            state.phase = Phase.DONE
            if final_stop_reason == "repetition":
                state.trace.append(TraceEvent(state.step_count, "final_answer_stopped_by_repetition", {}))
            break
        
        # phase == THINKING
        intervention_disabled = state.llm_scoring_calls >= config.max_llm_interventions * 2
        
        cascade = run_cascade(state, slm, llm, config, disabled=intervention_disabled)
        step_text, finish_reason = generate_one_step(state, slm, llm, config, cascade)
        step_text_normalized = ensure_step_terminator(step_text, finish_reason)
        
        state.assistant_prefix_text += step_text_normalized
        state.step_count += 1
        
        # 1) 先检查 </think>
        check_and_transition_phase(state, step_text_normalized)
        
        # 2) 再检查重复（仅在仍是 THINKING 时）
        #    如果 phase 已经因为 </think> 切到 FINAL_ANSWER，本轮不做 thinking-rep 处理
        if state.phase == Phase.THINKING:
            rep_trigger = update_repetition(rep, step_text_normalized,
                                             config.repetition_ngram_size,
                                             config.repetition_ngram_threshold)
            if rep_trigger is not None:
                # THINKING 阶段重复：注入 </think>，转 FINAL_ANSWER
                state.assistant_prefix_text += CLOSE_THINK_TAG + "\n\n"
                state.has_seen_close_think = True
                state.phase = Phase.FINAL_ANSWER
                state.trace.append(TraceEvent(
                    state.step_count, "thinking_repetition_force_close",
                    {"trigger_reason": rep_trigger},
                ))
                rep = RepetitionState()  # 重置，让 final answer 重新开始计数
        
        # 3) 检查终止
        if finish_reason == "eos":
            state.phase = Phase.DONE
    
    answer = extract_answer(state.assistant_prefix_text)
    return BPAResult(
        answer=answer, state=state,
        total_wall_time=time.time() - start_time,
    )
```

vLLM 单次 `generate` 调用内部检测重复需要 streaming 或 logits processor。第一版用**分块生成**实现：每次让 LLM 生成 `chunk_size` token，写入 prefix，跑 repetition 检测，决定是否继续。

```python
def _llm_generate_final_with_rep_guard(state, llm, config) -> tuple[str, str]:
    """
    LLM 一次性写最终答案，但分块检测重复。
    返回 (生成的完整文本, stop_reason)。
    stop_reason ∈ {"eos", "max_tokens", "repetition"}
    """
    rep = RepetitionState()
    accumulated_text = ""
    accumulated_tokens = 0
    
    # 用 sentence-level 切分 chunk（用 . ! ? \n\n 作 stop string，比 fixed token chunk 更自然）
    # 但 stop strings 不要太多，避免每个 chunk 太短
    chunk_stops = [".\n", "!\n", "?\n", "\n\n"]
    
    while accumulated_tokens < config.final_answer_max_tokens:
        rendered = render_for_continuation(
            state.problem_text,
            state.assistant_prefix_text + accumulated_text,
            llm.tokenizer,
        )
        remaining = config.final_answer_max_tokens - accumulated_tokens
        sampling = SamplingParams(
            max_tokens=min(remaining, config.final_answer_chunk_tokens),
            temperature=0.0,
            stop=chunk_stops,
            include_stop_str_in_output=True,
        )
        out = llm.generate(rendered, sampling)
        chunk_text = out[0].outputs[0].text
        chunk_token_count = len(out[0].outputs[0].token_ids)
        chunk_finish = out[0].outputs[0].finish_reason
        
        state.llm_decode_tokens += chunk_token_count
        state.llm_prefill_tokens += len(llm.tokenizer.encode(rendered, add_special_tokens=False))
        state.llm_full_calls += 1
        
        accumulated_text += chunk_text
        accumulated_tokens += chunk_token_count
        
        # 自然终止
        if chunk_finish == "eos" or (chunk_finish == "stop" and not _ends_with_chunk_stop(chunk_text, chunk_stops)):
            return accumulated_text, "eos"
        
        # 重复检查（把这个 chunk 当作一个"step"）
        rep_trigger = update_repetition(rep, chunk_text,
                                         config.repetition_ngram_size,
                                         config.repetition_ngram_threshold)
        if rep_trigger is not None:
            return accumulated_text, "repetition"
    
    return accumulated_text, "max_tokens"
```

### 2.2 三种生成路径

```python
def generate_one_step(state, slm, llm, config, cascade) -> tuple[str, str]:
    if cascade.decision == Decision.SLM_DIRECT:
        return _slm_generate_step(state, slm, config)
    
    elif cascade.decision == Decision.LLM_ARBITRATE:
        winner = cascade.winner_branch  # BranchCandidate
        
        # 用 step_branch_was_truncated 判断是否已经写到 step 边界
        if winner.step_branch_was_truncated:
            # winner 的 step_branch_text 不含 \n\n，但 raw rollout 在第一个 \n\n 处被截
            # 完整 step 就是 step_branch_text + "\n\n"
            return winner.step_branch_text + "\n\n", "stop_in_branch"
        
        # 没遇到 \n\n，让 SLM 续写到下一个 step boundary
        suffix_text, finish = _slm_continue_step(
            state, slm, config, prefix_extension=winner.step_branch_text
        )
        return winner.step_branch_text + suffix_text, finish
    
    elif cascade.decision == Decision.LLM_FULL:
        return _llm_generate_step(state, llm, config)


def _slm_generate_step(state, slm, config) -> tuple[str, str]:
    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, slm.tokenizer)
    sampling = SamplingParams(
        max_tokens=config.max_step_tokens,
        temperature=0.0,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    out = slm.generate(rendered, sampling)
    text = out[0].outputs[0].text
    finish = out[0].outputs[0].finish_reason
    
    state.slm_decode_tokens += len(out[0].outputs[0].token_ids)
    # prefill 成本估计：第一次或者 prefix 改动后会真实 prefill；连续生成的中间 step 大部分能命中 prefix cache
    # 第一版保守估计：每次调用 prefill = len(rendered tokens)（高估，但确保不低估 cost）
    state.slm_prefill_tokens += len(slm.tokenizer.encode(rendered, add_special_tokens=False))
    
    return text, finish


def _slm_continue_step(state, slm, config, prefix_extension: str) -> tuple[str, str]:
    # 把 winner 的 step_branch_text 临时加到 prefix 上做续写
    extended_prefix = state.assistant_prefix_text + prefix_extension
    rendered = render_for_continuation(state.problem_text, extended_prefix, slm.tokenizer)
    sampling = SamplingParams(
        max_tokens=config.max_step_tokens,
        temperature=0.0,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    out = slm.generate(rendered, sampling)
    state.slm_decode_tokens += len(out[0].outputs[0].token_ids)
    state.slm_prefill_tokens += len(slm.tokenizer.encode(rendered, add_special_tokens=False))
    return out[0].outputs[0].text, out[0].outputs[0].finish_reason


def _llm_generate_step(state, llm, config) -> tuple[str, str]:
    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, llm.tokenizer)
    sampling = SamplingParams(
        max_tokens=config.max_step_tokens,
        temperature=0.0,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    out = llm.generate(rendered, sampling)
    state.llm_decode_tokens += len(out[0].outputs[0].token_ids)
    state.llm_prefill_tokens += len(llm.tokenizer.encode(rendered, add_special_tokens=False))
    state.llm_full_calls += 1
    return out[0].outputs[0].text, out[0].outputs[0].finish_reason


def _llm_generate_final(state, llm, config) -> str:
    """FINAL_ANSWER phase：LLM 一次性生成到 EOS。"""
    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, llm.tokenizer)
    sampling = SamplingParams(
        max_tokens=config.final_answer_max_tokens,  # 默认 1024
        temperature=0.0,
        # 不设 stop，让模型自然 EOS
    )
    out = llm.generate(rendered, sampling)
    state.llm_decode_tokens += len(out[0].outputs[0].token_ids)
    state.llm_prefill_tokens += len(llm.tokenizer.encode(rendered, add_special_tokens=False))
    state.llm_full_calls += 1
    return out[0].outputs[0].text
```

### 2.3 Step terminator 规范化

```python
def ensure_step_terminator(step_text: str, finish_reason: str) -> str:
    if finish_reason == "eos":
        return step_text
    if not step_text.endswith("\n\n"):
        return step_text + "\n\n"
    return step_text
```

---

## 3. Pre-step Cascade

### 3.1 Decision 枚举

```python
class Decision(Enum):
    SLM_DIRECT = "slm_direct"
    LLM_ARBITRATE = "llm_arbitrate"
    LLM_FULL = "llm_full"
```

### 3.2 Cascade 入口

```python
@dataclass
class CascadeResult:
    decision: Decision
    l0: L0Result
    l1: tuple[BranchCandidate, BranchCandidate] | None = None
    l2: L2Result | None = None
    arbitration: ArbitrationResult | None = None
    winner_branch: BranchCandidate | None = None  # 仅 LLM_ARBITRATE 时填


def run_cascade(state, slm, llm, config, disabled: bool = False) -> CascadeResult:
    # L0
    l0 = l0_filter(state, slm, config)
    state.trace.append(TraceEvent(state.step_count, "l0", l0.to_dict()))
    
    if disabled or not l0.passed:
        return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0)
    
    # L1
    branch1, branch2 = l1_shadow_rollout(state, slm, config, l0)
    state.trace.append(TraceEvent(state.step_count, "l1", {
        "b1_text": branch1.raw_rollout_text[:200],
        "b2_text": branch2.raw_rollout_text[:200],
        "b1_truncated": branch1.step_branch_was_truncated,
        "b2_truncated": branch2.step_branch_was_truncated,
    }))
    
    # L2 — 只产生 probing statistics，不做强决策
    l2 = l2_compute(branch1, branch2, config)
    state.trace.append(TraceEvent(state.step_count, "l2", l2.to_dict()))
    
    if not l2.triggered_arbitration:
        # 关键修正：返回 SLM_DIRECT，不返回 LLM_ARBITRATE+None
        return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0, l1=(branch1, branch2), l2=l2)
    
    # LLM Arbitration
    arb = llm_arbitrate(state, llm, branch1, branch2, config)
    state.trace.append(TraceEvent(state.step_count, "arbitration", arb.to_dict()))
    
    # Arbitration 失败 fallback
    if arb.is_invalid:
        if config.invalid_fallback == "llm_full":
            return CascadeResult(decision=Decision.LLM_FULL, l0=l0, l1=(branch1, branch2), l2=l2, arbitration=arb)
        else:  # "skip" — 忽略仲裁，按 SLM_DIRECT 走
            return CascadeResult(decision=Decision.SLM_DIRECT, l0=l0, l1=(branch1, branch2), l2=l2, arbitration=arb)
    
    winner = branch1 if arb.winner_idx == 0 else branch2
    loser = branch2 if arb.winner_idx == 0 else branch1
    
    state.rejected_branches_log.append(RejectedBranch(
        step_idx=state.step_count,
        loser_text=loser.step_branch_text,
        winner_text=winner.step_branch_text,
        l2=l2,
    ))
    
    return CascadeResult(
        decision=Decision.LLM_ARBITRATE,
        l0=l0, l1=(branch1, branch2), l2=l2, arbitration=arb,
        winner_branch=winner,
    )
```

### 3.3 L0 — first-token entropy / margin

```python
@dataclass
class L0Result:
    passed: bool
    h_init: float
    margin: float
    top_logprobs: dict[int, float]    # SLM token id → logprob
    top_token_strs: list[str]
    first_char_class: str

def l0_filter(state, slm, config) -> L0Result:
    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, slm.tokenizer)
    sampling = SamplingParams(
        max_tokens=1, temperature=0.0, logprobs=config.l0_topk,
    )
    out = slm.generate(rendered, sampling)
    state.slm_decode_tokens += 1
    state.slm_prefill_tokens += len(slm.tokenizer.encode(rendered, add_special_tokens=False))
    
    step_logprobs = out[0].outputs[0].logprobs[0]  # dict: tok_id -> Logprob
    sorted_lps = sorted(step_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
    
    probs = np.array([math.exp(lp.logprob) for _, lp in sorted_lps])
    probs_norm = probs / probs.sum()
    h = -np.sum(probs_norm * np.log(probs_norm + 1e-10)) / np.log(max(len(probs_norm), 2))
    margin = float(probs_norm[0] - probs_norm[1]) if len(probs_norm) > 1 else 1.0
    
    top_token_strs = [slm.tokenizer.decode([tok_id]) for tok_id, _ in sorted_lps]
    first_char_class = classify_first_char(top_token_strs[0])
    
    passed = (margin < config.l0_margin_thresh) or (h > config.l0_entropy_thresh)
    
    return L0Result(
        passed=passed, h_init=h, margin=margin,
        top_logprobs={tid: lp.logprob for tid, lp in sorted_lps},
        top_token_strs=top_token_strs,
        first_char_class=first_char_class,
    )
```

`classify_first_char` 见 §6.1。

### 3.4 L1 — top-2 shadow rollout（双字段）

```python
@dataclass
class BranchCandidate:
    first_token_id: int
    first_token_str: str
    raw_rollout_text: str
    raw_rollout_token_ids: list[int]      # 包含 first_token_id 在内的完整 SLM token 序列
    step_branch_text: str                 # 截到第一个 \n\n 之前；不含 \n\n
    step_branch_was_truncated: bool       # raw 是否包含了 \n\n
    rollout_logprobs: list[float]         # 长度 = len(raw_rollout_token_ids) - 1（first token 单独）
    first_token_logprob: float
    sum_logprob_raw: float
    sum_logprob_step: float


def l1_shadow_rollout(state, slm, config, l0: L0Result) -> tuple[BranchCandidate, BranchCandidate]:
    sorted_tokens = sorted(l0.top_logprobs.items(), key=lambda x: x[1], reverse=True)
    tok1, lp1 = sorted_tokens[0]
    tok2, lp2 = sorted_tokens[1]
    
    rendered = render_for_continuation(state.problem_text, state.assistant_prefix_text, slm.tokenizer)
    rendered_ids = slm.tokenizer.encode(rendered, add_special_tokens=False)
    
    sampling = SamplingParams(
        max_tokens=config.rollout_length,
        temperature=0.0,
        logprobs=1,
        # 不设 stop=["\n\n"]：固定长度 raw rollout
    )
    
    prompts = [
        TokensPrompt(prompt_token_ids=rendered_ids + [tok1]),
        TokensPrompt(prompt_token_ids=rendered_ids + [tok2]),
    ]
    outs = slm.generate(prompts, sampling)
    
    state.slm_decode_tokens += sum(len(o.outputs[0].token_ids) for o in outs)
    # 两个 branch share 同一个 rendered prefix，prefill 算一次
    state.slm_prefill_tokens += len(rendered_ids) + 1  # +1 for the differing first token
    
    branch1 = _build_branch(tok1, lp1, outs[0], slm.tokenizer)
    branch2 = _build_branch(tok2, lp2, outs[1], slm.tokenizer)
    return branch1, branch2


def _build_branch(first_tok_id, first_tok_lp, vllm_out, tokenizer) -> BranchCandidate:
    o = vllm_out.outputs[0]
    rollout_continuation_ids = list(o.token_ids)  # 不包含 first token
    rollout_logprobs = [
        list(lp.values())[0].logprob if lp else 0.0
        for lp in (o.logprobs or [])
    ]
    
    # 关键修正：raw rollout 用 [first_tok_id] + continuation_ids 统一 decode
    raw_ids = [first_tok_id] + rollout_continuation_ids
    raw_rollout_text = tokenizer.decode(
        raw_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    
    # 截到第一个 \n\n
    idx = raw_rollout_text.find("\n\n")
    if idx >= 0:
        step_branch_text = raw_rollout_text[:idx]
        step_branch_was_truncated = True
        # 找截止位置对应的 token 数（用于 sum_logprob_step）
        # 累积 decode 找出第一个使 cumulative len 超过 idx 的 token
        cumulative_text = ""
        cutoff_tok_count = 0
        for i, tid in enumerate(raw_ids):
            piece = tokenizer.decode(raw_ids[:i+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if len(piece) > idx:
                cutoff_tok_count = i  # 不包含触发 \n\n 的 token
                break
            cutoff_tok_count = i + 1
        # cutoff_tok_count 包含 first token 在内
        if cutoff_tok_count == 0:
            sum_logprob_step = 0.0
        elif cutoff_tok_count == 1:
            sum_logprob_step = first_tok_lp
        else:
            # cutoff_tok_count - 1 个 continuation token
            sum_logprob_step = first_tok_lp + sum(rollout_logprobs[:cutoff_tok_count - 1])
    else:
        step_branch_text = raw_rollout_text
        step_branch_was_truncated = False
        sum_logprob_step = first_tok_lp + sum(rollout_logprobs)
    
    return BranchCandidate(
        first_token_id=first_tok_id,
        first_token_str=tokenizer.decode([first_tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
        raw_rollout_text=raw_rollout_text,
        raw_rollout_token_ids=raw_ids,
        step_branch_text=step_branch_text,
        step_branch_was_truncated=step_branch_was_truncated,
        rollout_logprobs=rollout_logprobs,
        first_token_logprob=first_tok_lp,
        sum_logprob_raw=first_tok_lp + sum(rollout_logprobs),
        sum_logprob_step=sum_logprob_step,
    )
```

### 3.5 L2 — Probing Statistics（双统计量，方向待 D2 校准）

```python
@dataclass
class L2Result:
    avg_lp_raw_1: float
    avg_lp_raw_2: float
    delta_avg_lp_raw: float
    avg_lp_step_1: float
    avg_lp_step_2: float
    delta_avg_lp_step: float
    text_jaccard_3gram_raw: float
    text_jaccard_3gram_step: float
    branches_diverged_at_token: int
    triggered_arbitration: bool
    trigger_reason: str


def l2_compute(b1: BranchCandidate, b2: BranchCandidate, config) -> L2Result:
    n1_raw = max(len(b1.raw_rollout_token_ids), 1)
    n2_raw = max(len(b2.raw_rollout_token_ids), 1)
    avg_raw_1 = b1.sum_logprob_raw / n1_raw
    avg_raw_2 = b2.sum_logprob_raw / n2_raw
    delta_raw = abs(avg_raw_1 - avg_raw_2)
    
    n1_step = max(_count_step_tokens(b1), 1)
    n2_step = max(_count_step_tokens(b2), 1)
    avg_step_1 = b1.sum_logprob_step / n1_step
    avg_step_2 = b2.sum_logprob_step / n2_step
    delta_step = abs(avg_step_1 - avg_step_2)
    
    j_raw = char_ngram_jaccard(b1.raw_rollout_text, b2.raw_rollout_text, n=3)
    j_step = char_ngram_jaccard(b1.step_branch_text, b2.step_branch_text, n=3)
    
    diverge_pos = _first_diverge_pos(b1.raw_rollout_token_ids, b2.raw_rollout_token_ids)
    
    # ⚠️ 触发方向暂用启发式，必须由 Exp-D2 校准并锁定
    triggered = False
    reason = "no_trigger"
    if delta_raw < config.l2_divergence_thresh:
        triggered = True
        reason = "delta_raw_low"
    elif j_raw < config.l2_text_jaccard_thresh:
        triggered = True
        reason = "text_jaccard_low"
    
    return L2Result(
        avg_lp_raw_1=avg_raw_1, avg_lp_raw_2=avg_raw_2, delta_avg_lp_raw=delta_raw,
        avg_lp_step_1=avg_step_1, avg_lp_step_2=avg_step_2, delta_avg_lp_step=delta_step,
        text_jaccard_3gram_raw=j_raw, text_jaccard_3gram_step=j_step,
        branches_diverged_at_token=diverge_pos,
        triggered_arbitration=triggered, trigger_reason=reason,
    )
```

> **致 Claude Code**：L2 触发方向（"delta 小 → 触发" 还是反过来）**未确认**。第一版用上面这个启发式跑，但所有 L2 字段必须无条件 log（无论是否触发），等 Exp-D2 用 oracle 校准后固化方向。**不要把方向硬编码进 ablation 设计的"假设"。**

---

## 4. LLM Arbitration（跨家族 text-level scoring）

### 4.1 Span 定位：longest common prefix 优先

```python
@dataclass
class SpanLocateResult:
    token_ids: list[int]                  # rendered_full 的 LLM token 序列
    branch_start_token: int
    branch_end_token: int                 # exclusive
    span_method: str                      # "lcp" | "rfind_after_lcp_fail" | "prefix_len_fallback" | "invalid"
    has_boundary_crossing_token: bool
    char_start: int
    char_end: int
    is_invalid: bool
    invalid_reason: str | None


def locate_branch_token_span(
    problem_text: str,
    assistant_prefix_text: str,
    branch_text: str,
    llm_tokenizer: PreTrainedTokenizerBase,
) -> SpanLocateResult:
    rendered_prefix = render_for_continuation(problem_text, assistant_prefix_text, llm_tokenizer)
    rendered_full = render_for_continuation(
        problem_text, assistant_prefix_text + branch_text, llm_tokenizer,
    )
    
    # 主路径：longest common prefix
    lcp_len = _longest_common_prefix_len(rendered_prefix, rendered_full)
    char_start_lcp = lcp_len
    char_end_lcp = char_start_lcp + len(branch_text)
    
    # 验证：rendered_full[char_start:char_end] 应该等于 branch_text
    use_lcp = (
        char_end_lcp <= len(rendered_full) and
        rendered_full[char_start_lcp:char_end_lcp] == branch_text
    )
    
    if use_lcp:
        char_start, char_end, span_method = char_start_lcp, char_end_lcp, "lcp"
    else:
        # Fallback 1: rfind
        idx = rendered_full.rfind(branch_text)
        if idx >= 0:
            char_start, char_end = idx, idx + len(branch_text)
            span_method = "rfind_after_lcp_fail"
        else:
            # Fallback 2: 估计为 lcp 末尾到 rendered_full 末尾
            # 但这个 span 多半不准，标记为可疑
            char_start, char_end = lcp_len, len(rendered_full)
            span_method = "prefix_len_fallback"
    
    # Fast tokenizer offset_mapping
    if not isinstance(llm_tokenizer, PreTrainedTokenizerFast):
        return SpanLocateResult(
            token_ids=[], branch_start_token=0, branch_end_token=0,
            span_method="invalid", has_boundary_crossing_token=False,
            char_start=char_start, char_end=char_end,
            is_invalid=True, invalid_reason="tokenizer_not_fast",
        )
    
    encoding = llm_tokenizer(rendered_full, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    
    branch_token_idxs = []
    has_crossing = False
    for tok_idx, (s, e) in enumerate(offsets):
        if e <= char_start:
            continue
        if s >= char_end:
            break
        if s < char_start or e > char_end:
            has_crossing = True
        branch_token_idxs.append(tok_idx)
    
    if not branch_token_idxs:
        return SpanLocateResult(
            token_ids=token_ids, branch_start_token=0, branch_end_token=0,
            span_method=span_method, has_boundary_crossing_token=False,
            char_start=char_start, char_end=char_end,
            is_invalid=True, invalid_reason="no_tokens_in_span",
        )
    
    # Sanity check：解码这些 token 与 branch_text 的字符 overlap > 70%
    decoded_span = llm_tokenizer.decode(
        [token_ids[i] for i in branch_token_idxs],
        skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    overlap_ratio = _char_overlap_ratio(decoded_span, branch_text)
    is_invalid = overlap_ratio < 0.7
    invalid_reason = None if not is_invalid else f"low_decode_overlap_{overlap_ratio:.2f}"
    
    return SpanLocateResult(
        token_ids=token_ids,
        branch_start_token=branch_token_idxs[0],
        branch_end_token=branch_token_idxs[-1] + 1,
        span_method=span_method,
        has_boundary_crossing_token=has_crossing,
        char_start=char_start, char_end=char_end,
        is_invalid=is_invalid, invalid_reason=invalid_reason,
    )


def _longest_common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n
```

### 4.2 Scoring（actual token 缺失 → invalid）

```python
@dataclass
class BranchScore:
    mean_logprob: float | None        # None 时 is_invalid
    branch_token_count: int
    span_locate: SpanLocateResult
    is_invalid: bool
    invalid_reason: str | None
    # cost
    prefill_tokens: int


@dataclass
class ArbitrationResult:
    score1: BranchScore
    score2: BranchScore
    winner_idx: int                   # 0 or 1
    is_invalid: bool                  # 任一 score invalid → 整体 invalid
    invalid_reason: str | None


def llm_arbitrate(state, llm, b1: BranchCandidate, b2: BranchCandidate, config) -> ArbitrationResult:
    s1 = _score_branch(state, llm, b1.step_branch_text, config)
    s2 = _score_branch(state, llm, b2.step_branch_text, config)
    
    if s1.is_invalid or s2.is_invalid:
        # 关键修正：不用估计值做 winner selection
        return ArbitrationResult(
            score1=s1, score2=s2,
            winner_idx=0,                    # 占位，调用方应基于 is_invalid 走 fallback
            is_invalid=True,
            invalid_reason=f"s1={s1.invalid_reason} s2={s2.invalid_reason}",
        )
    
    winner_idx = 0 if s1.mean_logprob >= s2.mean_logprob else 1
    return ArbitrationResult(
        score1=s1, score2=s2,
        winner_idx=winner_idx,
        is_invalid=False,
        invalid_reason=None,
    )


def _score_branch(state, llm, branch_text: str, config) -> BranchScore:
    locate = locate_branch_token_span(
        state.problem_text, state.assistant_prefix_text, branch_text, llm.tokenizer,
    )
    
    if locate.is_invalid:
        return BranchScore(
            mean_logprob=None, branch_token_count=0,
            span_locate=locate, is_invalid=True, invalid_reason=locate.invalid_reason,
            prefill_tokens=0,
        )
    
    sampling = SamplingParams(
        max_tokens=1, temperature=0.0,
        prompt_logprobs=config.prompt_logprobs_topk,   # 默认 20
    )
    out = llm.generate(TokensPrompt(prompt_token_ids=locate.token_ids), sampling)
    
    # 关键修正：成本拆分
    state.llm_prefill_tokens += len(locate.token_ids)
    state.llm_decode_tokens += 1   # max_tokens=1 实际生成一个 token
    state.llm_scoring_calls += 1
    
    plp = out[0].prompt_logprobs   # list[dict | None]
    
    branch_lps = []
    missing_count = 0
    for i in range(locate.branch_start_token, locate.branch_end_token):
        if i >= len(plp) or plp[i] is None:
            missing_count += 1
            continue
        actual = plp[i].get(locate.token_ids[i])
        if actual is None:
            # actual token 不在 top-K：按你的决策标记 missing，不用估计值
            missing_count += 1
            continue
        branch_lps.append(actual.logprob)
    
    branch_token_count = locate.branch_end_token - locate.branch_start_token
    
    # 如果缺失比例过高，整个 branch score 视为 invalid
    if branch_token_count == 0:
        return BranchScore(
            mean_logprob=None, branch_token_count=0,
            span_locate=locate, is_invalid=True, invalid_reason="empty_span",
            prefill_tokens=len(locate.token_ids),
        )
    missing_ratio = missing_count / branch_token_count
    if missing_ratio > config.score_missing_ratio_thresh:    # 默认 0.2
        return BranchScore(
            mean_logprob=None, branch_token_count=branch_token_count,
            span_locate=locate, is_invalid=True,
            invalid_reason=f"missing_ratio_{missing_ratio:.2f}",
            prefill_tokens=len(locate.token_ids),
        )
    
    mean_lp = sum(branch_lps) / len(branch_lps)
    return BranchScore(
        mean_logprob=mean_lp, branch_token_count=branch_token_count,
        span_locate=locate, is_invalid=False, invalid_reason=None,
        prefill_tokens=len(locate.token_ids),
    )
```

### 4.3 Invalid 时的 fallback

`run_cascade` 中已经实现：

```python
if arb.is_invalid:
    if config.invalid_fallback == "llm_full":
        return CascadeResult(decision=Decision.LLM_FULL, ...)
    else:  # "skip"
        return CascadeResult(decision=Decision.SLM_DIRECT, ...)
```

第一版 `config.invalid_fallback = "skip"`（不浪费 LLM full 调用）。**不要用估计 logprob 做 winner selection**。

### 4.4 工程提醒

- `prompt_logprobs=20` 第一版默认。Smoke test 跑完前 5 题后报告每个 branch 的 missing_ratio 分布。如果 20 时仍有 >5% 的 branch 触发 invalid，下一版考虑用自定义 logits processor 强制返回 actual token 的 logprob。
- vLLM V1 下 `prompt_logprobs` 与 prefix cache 命中时仍会触发重计算，prefill 成本按 `len(token_ids)` 算，是诚实的高估。
- logprobs 数值不稳定：`mean_logprob` 比较留 0.05 nat 余量（即 `abs(s1 - s2) < 0.05` 时按 winner_idx=0 走，记为 tie）。

---

## 5. BPAResult 与成本计费

```python
@dataclass
class BPAResult:
    answer: str | None
    state: GenerationState
    total_wall_time: float
    correct: bool | None = None
    
    # 计费（从 state 复制，便于聚合）
    @property
    def slm_decode_tokens(self): return self.state.slm_decode_tokens
    @property
    def slm_prefill_tokens(self): return self.state.slm_prefill_tokens
    @property
    def llm_decode_tokens(self): return self.state.llm_decode_tokens
    @property
    def llm_prefill_tokens(self): return self.state.llm_prefill_tokens
    @property
    def llm_scoring_calls(self): return self.state.llm_scoring_calls
    @property
    def llm_full_calls(self): return self.state.llm_full_calls
    
    # 综合 cost：用模型相对算力比加权
    def equivalent_llm_tokens(self, slm_to_llm_flop_ratio: float) -> float:
        """
        slm_to_llm_flop_ratio: SLM forward / LLM forward 的算力比例（< 1）
        例如 1.5B vs 32B 大约 0.05
        """
        slm_total = self.slm_decode_tokens + self.slm_prefill_tokens
        llm_total = self.llm_decode_tokens + self.llm_prefill_tokens
        return slm_total * slm_to_llm_flop_ratio + llm_total
```

主对比表必须报告：
- `slm_decode_tokens`、`slm_prefill_tokens`
- `llm_decode_tokens`、`llm_prefill_tokens`
- `llm_scoring_calls`、`llm_full_calls`
- `equivalent_llm_tokens`（加权综合成本）
- `total_wall_time`

不能只报"LLM tokens"或"intervention rate"。

---

## 6. Diagnostic Suite

### 6.1 Exp-D0: 首 token 与特殊字符

```python
def classify_first_char(token_str: str) -> str:
    if not token_str:
        return "empty"
    s = token_str.lstrip()
    if not s:
        return "whitespace"
    c = s[0]
    if c.isalpha():
        return "alpha"
    if c.isdigit():
        return "digit"
    if c == "\\" and any(s.startswith(p) for p in [r"\frac", r"\sqrt", r"\sum", r"\int", r"\(", r"\["]):
        return "latex_command"
    if c in "*_#`>~|-":
        return "markdown"
    if c in "<":  # <think>, </think>
        return "special_tag"
    return "other_symbol"
```

记录字段（按你的第 3 点决策保留扩展统计）：

```python
@dataclass
class D0Record:
    step_idx: int
    boundary_pos_in_assistant_prefix: int    # 改名
    
    main_first_token_str: str
    main_first_char_class: str
    main_h_init: float
    main_margin: float
    main_l0_passed: bool
    
    # counterfactual_skip_structural
    skipped_first_token_str: str | None
    skipped_first_char_class: str | None
    skipped_h_init: float | None
    skipped_margin: float | None
    skipped_l0_passed: bool | None
    skip_changed_decision: bool
    
    # boundary_token_frequency
    top_k_token_strs: list[str]
    top_k_char_classes: list[str]
    
    # 整题维度（题级填充）
    arbitration_triggered: bool
    arbitration_changed_winner: bool
    final_correct: bool

# 全 run 维度
@dataclass
class D0RunMeta:
    model_name: str
    chat_template_hash: str
    system_prompt: str | None
    thinking_prefix: str
    rendered_initial_assistant_marker: str   # chat template 渲染后 assistant turn 头部的内容
```

`<think>` **不**默认归入跳过。仅跳过 `whitespace` / `latex_command` / `markdown`。

### 6.2 Exp-D1: Cascade funnel + rollout length sweep

跑 MATH500 200 题，rollout_length sweep {8, 16, 32}。

记录：每题 boundary_count、l0_pass、l1_invocation、l2_trigger、llm_arbitration_call、llm_full_call、各 rollout_length 下的 wall_time 分布。

### 6.3 Exp-D2: L2 oracle 校准（用 benchmark evaluator）

按你的第 6 点决策，**不依赖 GPT-4 judge**。流程：

1. 在 D1 跑过程中，对每个 L1 触发的 (b1, b2) 三元组，**记录两个 branch 的完整继续推理轨迹**：
   - `traj_1` = `assistant_prefix_text + b1.step_branch_text + (SLM 续写到 EOS)`
   - `traj_2` = `assistant_prefix_text + b2.step_branch_text + (SLM 续写到 EOS)`
2. 离线缓存 trajectory 文件，避免重跑
3. 用 benchmark evaluator 抽取每条 trajectory 的最终答案：
   - **MATH500 / AIME**: `extract_boxed_answer` + symbolic equivalence (sympy)
   - **GPQA**: 抽取最终选项字母 (A/B/C/D)，与 ground truth 字母对比
4. 每对 trajectory 得到 oracle label：`{both_correct, only_1, only_2, both_wrong}`
5. 分析：L2 各统计量与 "branches actually divergent (only_1 or only_2 correct)" 的 AUC

```python
def benchmark_eval_match(predicted: str, ground_truth: str, dataset: str) -> bool:
    if dataset in ("math500", "aime24", "aime25"):
        return _math_match(extract_boxed(predicted), ground_truth)
    elif dataset == "gpqa_diamond":
        return _gpqa_match(extract_choice_letter(predicted), ground_truth)
    else:
        raise ValueError(f"unknown dataset {dataset}")
```

GPT-4 judge **仅**作为 sanity check 子集（10% 样本上抽样验证 evaluator 与人类 / GPT-4 judge 的一致性），不作为主 oracle。

成本估算：MATH500 200 题 × 平均 30 step × L1 触发率 ~30%（先估）× 2 branch × ~6K SLM token = ~720M SLM token。1.5B 模型 ~50 token/s on H100，约 4000 GPU-小时 — **不可接受**。

**修订**：D2 缩减为：
- 题集：MATH500 50 题 + AIME24 30 题 + GPQA-Diamond 30 题 = 110 题
- 只对 L1 触发率最高的题取样：每题最多取前 5 个 L1 触发位置
- 每个位置只跑 SLM continuation 到 max_tokens=4096 或 EOS
- 估计成本：110 题 × 5 位置 × 2 branch × 4K token = 4.4M token，约 25 GPU-小时

### 6.4 Exp-D3: Commitment recurrence (diagnostic-only)

**记录**：
- 每次 LLM arbitration 的 `loser_step_branch_text`
- 后续每个 step 起点，SLM top-1/top-2/top-k 分支与历史 losers 的 char-3gram Jaccard
- recurrence rate, recurrence-then-error 相关性

### 6.5 Exp-D4: 在线 step-level 检查

参考 GlimpRouter 的简化实现：检测最近**两个 step 是否完全相同**（按 `\n\n` 切分后的 step 单元）。这个判据简单、误触发率低、实现成本低。补充一个 8-gram 计数作为兜底（避免 step 内部循环但 step 间不同的情况）。

```python
@dataclass
class RepetitionState:
    recent_steps: deque[str] = field(default_factory=lambda: deque(maxlen=4))
    ngram_counter: Counter = field(default_factory=Counter)
    triggered: bool = False
    trigger_reason: str | None = None    # "duplicate_step" | "ngram_repeat"


def update_repetition(rep: RepetitionState, new_step_text: str, ngram_size: int = 8,
                      ngram_threshold: int = 4) -> str | None:
    """
    在每次 step 写入 assistant_prefix_text 之后调用。
    返回 trigger_reason（None 表示未触发）。
    """
    # 标准化：去掉末尾的 \n\n 再比较，避免 terminator 影响
    normalized = new_step_text.rstrip("\n").rstrip()
    if len(normalized) < 10:    # 太短的 step 不参与判定，避免短分隔行误触
        rep.recent_steps.append(normalized)
        return None
    
    # 1) 连续重复 step
    if rep.recent_steps and rep.recent_steps[-1] == normalized:
        rep.triggered = True
        rep.trigger_reason = "duplicate_step"
        return "duplicate_step"
    
    # 2) 隔 1 个 step 重复（A B A B 模式）
    if len(rep.recent_steps) >= 2 and rep.recent_steps[-2] == normalized:
        rep.triggered = True
        rep.trigger_reason = "alternating_step"
        return "alternating_step"
    
    rep.recent_steps.append(normalized)
    
    # 3) 8-gram 累计计数（按字符）
    chars = normalized
    if len(chars) >= ngram_size:
        for i in range(len(chars) - ngram_size + 1):
            ng = chars[i:i+ngram_size]
            rep.ngram_counter[ng] += 1
            if rep.ngram_counter[ng] >= ngram_threshold:
                rep.triggered = True
                rep.trigger_reason = "ngram_repeat"
                return "ngram_repeat"
    
    return None
```
### 6.6: Exp-D5 — Prompt Logprobs Smoke Test**：

跑题集：MATH500 5 题。对每题完整跑一次 BPA logging-only（cascade 跑通但不实际用 arbitration 结果），同时**对每个 L1 触发位置**做一次完整 LLM scoring 试验：

- 对 `step_branch_text` 用 `prompt_logprobs ∈ {1, 5, 20, 50}` 各跑一次
- 记录每个 K 下每个 prompt token 的 actual token 是否在 top-K 内
- 输出统计：
  - `per_token_missing_rate[K]` = 整道题里 actual token 不在 top-K 的位置比例
  - `per_branch_invalid_rate[K]` = `missing_ratio > 0.2` 的 branch 比例
  - cost: K 增大对 vLLM scoring latency / GPU memory 的实际影响

**Smoke test 决策门**：

- 如果 `K=20` 下 `per_branch_invalid_rate <= 5%` → 用 K=20 跑后续实验
- 如果 `K=20` 下 `per_branch_invalid_rate ∈ (5%, 25%]` → 升 K=50
- 如果 `K=50` 下 `per_branch_invalid_rate > 10%` → **方案需要改**：要么实现自定义 logits processor 拿 actual token logprob，要么换成"短 continuation likelihood difference"作为打分（不依赖 prompt_logprobs lookup）。在动手前停下来讨论。


### THINKING / FINAL_ANSWER 下的 trigger 行为差异

按你的指示分两种行为：

**THINKING 阶段重复**：
- 不停整体生成
- 强制注入 `</think>\n\n` 进入 `assistant_prefix_text`，即"模拟模型自己写完了 thinking"
- 转 `phase = FINAL_ANSWER`
- 后续走 LLM 接管最终答案
- 重置 `RepetitionState`（避免 thinking 段累积的 ngram 计数影响 final answer）

**FINAL_ANSWER 阶段重复**：
- 直接停整道题
- `phase = DONE`
- 不让 LLM 重写

---

## 7. Baseline 与第一周计划

### 7.1 第一周 baseline（缩减后）

- **SLM only**：仅用 SLM 跑完整推理
- **LLM only**：仅用 LLM 跑完整推理
- **GlimpRouter-style H_init**：在 step 起点用 SLM first-token entropy/margin 决定是否让 LLM 接管整个 step（无 shadow rollout，无 LLM scoring）
- **BPA logging-only**：BPA 完整 cascade 跑通但 `Decision.LLM_ARBITRATE` 不实际生效，仅记录会触发的位置（用于 Exp-D2 数据收集）
- **BPA arbitration**：完整 BPA，含 LLM arbitration

第一周**不**实现的 baseline（推到主实验阶段）：Speculative Decoding、R-Stitch、Speculative Thinking。

### 7.2 第一周 milestone

| Day | 目标 |
|---|---|
| 1 | vLLM 双引擎跑通；render_for_continuation sanity（手工验证 R1-distill 自动加 `<think>`，传 `assistant_prefix_text=""` 时不双重 `<think>`） |
| 2 | §1-2 GenerationState + Phase 状态机 + Step Loop 实现；50 题 SLM only 跑通；验证 `</think>` 检测 |
| 3 | §3.3 L0 + classify_first_char + D0 收集器；50 题 SLM-only 上记录 D0 字段 |
| 4 | §3.4 L1 + raw rollout decode 修正 + step_branch_was_truncated 字段；§3.5 L2 双统计量 |
| 5 | §4.1 span 定位（lcp 优先 + rfind fallback + invalid 标记）+ Smoke test：跑 5 题，报告 prompt_logprobs missing_ratio 分布 |
| 6 | §4.2 scoring + invalid handling + §5 cost 拆分；BPA logging-only 跑 50 题 |
| 7 | BPA arbitration 跑 50 题；产出第一周报告：四个 baseline + BPA arbitration 在 MATH500 50 题上的 accuracy / cost / time |

第一周末必须有：
- (a) 五个变体在 MATH500 50 题上端到端跑通，结果在同一张表里
- (b) Exp-D0 报告（200 题 mix dataset，3 个 dataset 各取部分）
- (c) Exp-D1 漏斗图初版 + rollout_length sweep 初版
- (d) Smoke test 报告（prompt_logprobs missing_ratio）

如果 (a) 中 BPA arbitration 比 GlimpRouter-style H_init 没有任何提升（无论是 accuracy 还是 cost-accuracy Pareto），立即停下重新讨论方向。

### 7.3 主实验对比（推到第二周之后）

| 方法 | MATH500 | AIME24 | AIME25 | GPQA-D | wall time | equiv LLM tokens |
|---|---|---|---|---|---|---|
| SLM only | | | | | | |
| LLM only | | | | | | |
| GlimpRouter H_init | | | | | | |
| Speculative Decoding | | | | | | |
| R-Stitch | | | | | | |
| Speculative Thinking | | | | | | |
| **BPA (ours)** | | | | | | |

主实验的 BPA 设置由 D2 校准结果决定 trigger direction 与 thresholds。

---

## 8. 默认参数全表

| 参数 | 默认值 | 来源 |
|---|---|---|
| `l0_topk` | 10 | 不调 |
| `l0_margin_thresh` | 0.4 | D1 sweep |
| `l0_entropy_thresh` | 0.5 | D1 sweep |
| `rollout_length` | 16 | D1 sweep {8,16,32} |
| `l2_divergence_thresh` | 0.15 | **D2 校准 + 方向** |
| `l2_text_jaccard_thresh` | 0.4 | D2 校准 |
| `prompt_logprobs_topk` | 20 | smoke test 决定 |
| `score_missing_ratio_thresh` | 0.2 | 第一版固定，后续 sensitivity |
| `invalid_fallback` | "skip" | 第一版 |
| `max_step_tokens` | 1024 | 不调 |
| `max_total_tokens` | 16384 | = max_model_len |
| `max_llm_interventions` | 8 | sensitivity |
| `final_answer_max_tokens` | 1024 | 不调 |

---

## 9. 文件结构

```
bpa/
├── __init__.py
├── config.py                    # BPAConfig
├── state.py                     # GenerationState, BranchCandidate, Phase, ...
├── engines.py                   # ModelEngine wrapper, init_engines
├── render.py                    # render_for_continuation
├── phase_machine.py             # Phase 状态机, detect_close_think, _looks_like_final_answer
├── pipeline.py                  # bpa_solve 主循环 + generate_one_step
├── cascade/
│   ├── __init__.py
│   ├── l0.py                    # l0_filter, classify_first_char
│   ├── l1.py                    # l1_shadow_rollout, _build_branch
│   └── l2.py                    # l2_compute, char_ngram_jaccard
├── arbitration.py               # llm_arbitrate, locate_branch_token_span (lcp优先), _score_branch
├── trace.py                     # TraceEvent, BPAResult, RejectedBranch
├── safety.py                    # extract_answer, post_hoc 重复检测
└── eval/
    ├── benchmark_eval.py        # MATH/AIME/GPQA evaluator
    ├── exp_d0_first_token.py
    ├── exp_d1_cascade_funnel.py
    ├── exp_d2_oracle_l2.py      # 用 benchmark_eval, 不用 GPT-4
    ├── exp_d3_commitment_recurrence.py
    ├── exp_d4_repetition.py
    ├── baselines.py             # SLM-only, LLM-only, GlimpRouter-H_init
    └── main_benchmark.py
```



