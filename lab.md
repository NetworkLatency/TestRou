# 1. 方法名称与边界

建议方法名：

**SARR-CoDE: Stability-Aware Rollback Routing with Confidence-Degeneration Evidence**

更准确的完整名称：

**Aggressive Prefix-Centric SARR-CoDE**

方法边界必须写死：

1. 不使用 Hinit 或 initial-token entropy 作为主方法启动信号。
2. 不使用 final-answer probe。
3. 不解析 `\boxed{}`，不比较 answer identity。
4. 不让 LLM judge 或 verify SLM step。
5. LLM 只负责从 rollback anchor 处重新生成 recovery steps。
6. 在线信号只来自 SLM 对当前 prefix 的 next-token distribution。

这样可以避免和 GlimpRouter、SpecReason、Answer Convergence 混淆。

---

# 2. 背景概念

## 2.1 现有协同推理的问题

已有 step-level SLM-LLM collaboration 通常试图在每个 step 开始前判断该 step 是否困难。GlimpRouter 是代表性方法，它让 SLM 在 reasoning step 起点生成一个 “glimpse”，再用 initial token entropy 判断是否由 LLM 生成该 step。([arXiv][1])

这个方向的问题是：它只判断 **当前 step 的局部难度**，没有监控 SLM 已经生成出的 prefix 是否逐渐退化。你的实验现象说明，失败不一定发生在 step 起点，而可能发生在 SLM 已经沿着某条轨迹推进之后：它可能进入循环、反复修正、不稳定漂移，或者在错误 prefix 上继续生成低质量内容。

## 2.2 CoDE-Stop 的启发

CoDE-Stop 的核心观察是：单步 confidence 不够，错误轨迹往往表现为持续不稳定；有效 early stopping 应该建模 confidence dynamics，而不是只看某个局部 confidence。论文明确指出，低 confidence 可能只是正常探索，而错误轨迹更常表现为 persistent instability；因此需要累积 confidence 序列中的退化信号。([arXiv][2])

我们不使用 CoDE-Stop 的 answer-confidence prompt，但借鉴它的 trend-aware degeneration 结构。CoDE-Stop 也报告 trend-aware score 比 raw confidence、confidence drop、low confidence 更有效。([arXiv][2])

## 2.3 我们的核心转化

CoDE-Stop：

[
\text{answer confidence dynamics} \rightarrow \text{early stop}
]

SARR-CoDE：

[
\text{continuation confidence dynamics} \rightarrow \text{rollback-triggered LLM collaboration}
]

其中 continuation confidence 不问最终答案，只问：

**在当前 prefix 后，SLM 是否知道下一步如何继续生成。**

---

# 3. 基本对象定义

给定问题 (q)，小模型 (M_s)，大模型 (M_l)。

推理过程被切分为 step 序列：

[
S_{1:K} = {s_1, s_2, \ldots, s_K}
]

每个 step：

[
s_k = (y_{k,1}, y_{k,2}, \ldots, y_{k,L_k})
]

第 (k) 步后的 prefix：

[
C_k = q \oplus s_1 \oplus s_2 \oplus \cdots \oplus s_k
]

第 (0) 步 prefix：

[
C_0 = q
]

每个 step 的生成模型记为：

[
g_k \in {\text{SLM}, \text{LLM}}
]

Codex 实现时每个 step 必须记录 `generator` 字段。

---

# 4. Step 切分规则

默认 step delimiter：

```text
"\n\n"
```

如果模型输出中没有 `\n\n`，则 fallback 到 max tokens per step。

推荐配置：

```yaml
step_delimiters:
  - "\n\n"
max_new_tokens_per_step: 256
```

不要在第一版中使用 “Wait” 或 “Alternatively” 作为主 delimiter。CoDE-Stop 使用 “Wait” 这类 self-reflection token 作为 reasoning transition point，并报告 reasonable delimiter 下结果相对稳健。([arXiv][2]) 但你的方法面向 SLM-LLM step-level collaboration，工程上优先使用 `\n\n` 更稳定，也更接近 GlimpRouter 的 step-wise 设置。

Codex 需要实现：

```python
generate_step(model, context, stop_delimiters=["\n\n"], max_new_tokens=256)
```

返回：

```python
StepOutput(text, token_ids, finish_reason)
```

---

# 5. Continuation confidence 定义

在每个 step 生成完成后，无论该 step 来自 SLM 还是 LLM，都用 SLM 计算当前 prefix 后的 next-token distribution：

[
p_s(\cdot \mid C_k)
]

取 top-(K) logits，默认：

[
K=20
]

计算 top-(K) normalized entropy：

[
\widetilde{H}(p_s(\cdot\mid C_k))
=================================

\frac{-\sum_{i=1}^{K}p_i\log p_i}{\log K}
]

其中 (p_i) 是 top-(K) logits softmax 后的概率。

定义 raw continuation confidence：

[
c_k = 1-\widetilde{H}(p_s(\cdot\mid C_k))
]

因此：

[
c_k\in[0,1]
]

含义：

* (c_k) 越高：SLM 对下一步 continuation 越确定。
* (c_k) 越低：SLM 对后续推理方向越不确定。

注意：
(c_k) 不是 correctness confidence。
(c_k) 不是 answer confidence。
(c_k) 只是 continuation confidence。

Codex 实现：

```python
def slm_continuation_confidence(slm, tokenizer, context_text, topk=20):
    logits = slm.next_logits(context_text)  # shape [vocab]
    top_logits, top_ids = torch.topk(logits, k=topk)
    probs = torch.softmax(top_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    norm_entropy = entropy / math.log(topk)
    confidence = 1.0 - float(norm_entropy)
    return confidence, {
        "top_ids": top_ids.tolist(),
        "top_probs": probs.tolist(),
        "norm_entropy": float(norm_entropy),
    }
```

---

# 6. Percentile normalization

1.5B SLM 的 logits 尺度和 4B/8B 模型不同，不能直接使用固定 raw confidence 阈值。因此必须离线构建 calibration CDF。

给定 calibration traces 中所有 (c_k)，构建经验分布：

[
F_{\text{calib}}(c)
]

在线时将 raw confidence 归一化为 percentile：

[
\hat{c}*k = F*{\text{calib}}(c_k)
]

(\hat{c}_k\in[0,1])，表示当前 (c_k) 在 calibration distribution 中处于什么分位。

Codex 实现：

```python
class PercentileNormalizer:
    def __init__(self, values):
        self.values = np.sort(np.asarray(values, dtype=np.float32))

    def transform(self, x):
        # empirical CDF: fraction of calibration values <= x
        idx = np.searchsorted(self.values, x, side="right")
        return float(idx / max(len(self.values), 1))
```

保存文件：

```json
{
  "calibration_values": [...],
  "topk_entropy": 20,
  "num_values": 12345
}
```

---

# 7. 短窗口平滑

为了抑制单点噪声，对 normalized confidence 做短窗口平滑：

[
\tilde{c}*k = \frac{1}{W}\sum*{i=k-W+1}^{k}\hat{c}_i
]

默认：

[
W=2
]

这是工程平滑，不是核心理论。主实验固定 (W=2)，后续可以做 (W=1,2,3) 的 sensitivity。

Codex 实现：

```python
def smooth_confidence(c_norm_history, W=2):
    if len(c_norm_history) < W:
        return None
    return float(np.mean(c_norm_history[-W:]))
```

---

# 8. CoDE-style degeneration event

当至少有两个 smooth confidence 后，定义趋势型退化事件：

[
v_k =
\mathbf{1}[\tilde{c}*{k-1} > \tilde{c}*{k}]
\cdot
\mathbf{1}[2\tilde{c}*k-\tilde{c}*{k-1}<\delta]
]

默认：

[
\delta=0.55
]

解释：

[
2\tilde{c}*k-\tilde{c}*{k-1}
============================

\tilde{c}_k + (\tilde{c}*k-\tilde{c}*{k-1})
]

这是按当前下降趋势做的一步外推。如果当前 confidence 下降，并且外推后低于 (\delta)，则认为发生一次 degeneration event。

这个结构来自 CoDE-Stop 的 trend-aware instability 思想。CoDE-Stop 的论文说明，该设计能同时捕捉 low confidence 和 sudden confidence drop，比单纯 low-confidence rule 更有效。([arXiv][2])

在你的方法中，(v_k) 不表示答案错误，只表示当前 prefix 的 SLM continuation confidence 出现了趋势型退化。

Codex 实现：

```python
def code_style_degeneration_event(prev_c_smooth, curr_c_smooth, delta=0.55):
    if prev_c_smooth is None or curr_c_smooth is None:
        return 0
    return int((prev_c_smooth > curr_c_smooth) and
               (2.0 * curr_c_smooth - prev_c_smooth < delta))
```

---

# 9. 状态机

最终状态机：

```text
STARTUP
STABLE
DEGENERATED
```

不要使用 WARMUP。
不要使用 pre-step Hinit routing。
LLM 不通过 router 主动接管，只通过 rollback/recovery 启动。

## 9.1 STARTUP

初始状态。

含义：SLM 正在自主尝试生成初始 prefix，系统观察其 confidence dynamics。

在 STARTUP 中，每个 step 都由 SLM 生成，直到发生以下情况之一：

1. prefix 达到稳定状态；
2. prefix 发生 early degeneration；
3. prefix 在最大启动预算内仍未稳定。

## 9.2 STABLE

如果当前 smooth confidence 达到稳定阈值：

[
\tilde{c}_k \geq \theta_s
]

则进入 STABLE，并记录 stable anchor：

[
a = k
]

如果后续某一步仍满足：

[
\tilde{c}_k \geq \theta_s
]

则刷新 anchor：

[
a = k
]

并清空 post-stable degeneration score：

[
D_{\text{post}}=0
]

## 9.3 DEGENERATED

当 post-stable degeneration score 达到阈值：

[
D_{\text{post}} \geq \tau_D
]

则进入 DEGENERATED，并触发 rollback。

---

# 10. STARTUP 阶段逻辑

参数：

[
B_{\min}=2
]

[
B_{\max}=5
]

[
\tau_{\text{start}}=1
]

默认值建议先固定。之后可用 calibration 选择。

启动阶段每步更新：

[
D_{\text{start}} = D_{\text{start}} + v_k
]

如果：

[
\tilde{c}_k \geq \theta_s
]

则：

[
\text{state}=\text{STABLE},\quad a=k,\quad D_{\text{start}}=0
]

如果已经生成至少 (B_{\min}) 个 SLM steps，且：

[
D_{\text{start}}\geq \tau_{\text{start}}
]

则触发 startup rollback。

如果生成到：

[
k_{\text{startup}} = B_{\max}
]

仍未进入 STABLE，也触发 startup rollback。

这里的思想是：不要让 SLM 在一开始就长时间沿着不稳定 prefix 走下去。与 GlimpRouter 不同，这不是 step 前 Hinit dispatch，而是 SLM 真实生成短 prefix 后的动态诊断。

---

# 11. STARTUP 没有 stable anchor 时的 best-prefix anchor

如果 STARTUP 阶段触发 rollback，但尚未进入 STABLE，则没有 stable anchor。此时定义 best-prefix anchor：

[
a^\star = \arg\max_{j\in[0,k]} \tilde{c}_j
]

其中 (j=0) 表示原始 prompt，无 SLM step。

实现细节：

* 对 (j=0)，可以设置 (\tilde{c}_0=0)，但为了允许删除全部 SLM prefix，必须允许 anchor=0。
* 更稳的做法：在 startup rollback 时，如果所有 (\tilde{c}_j < \theta_s)，则选择 confidence 最高的 step；若最高值仍低于一个低阈值，例如 0.3，则 anchor=0。第一版可以更激进：直接 `argmax`，并允许结果为 0。
* 实现时需要在 `prefix_records` 中保存 step 0。

删除：

[
s_{a^\star+1:k}
]

如果：

[
a^\star=0
]

则删除全部 SLM startup prefix，让 LLM 从原始题目开始恢复。

---

# 12. POST-STABLE 阶段逻辑

在 STABLE 后，如果：

[
\tilde{c}_k \geq \theta_s
]

刷新 anchor：

[
a=k
]

并清空：

[
D_{\text{post}}=0
]

如果：

[
\tilde{c}_k < \theta_s
]

计算：

[
v_k =
\mathbf{1}[\tilde{c}*{k-1} > \tilde{c}*{k}]
\cdot
\mathbf{1}[2\tilde{c}*k-\tilde{c}*{k-1}<\delta]
]

累计：

[
D_{\text{post}} = D_{\text{post}} + v_k
]

如果：

[
D_{\text{post}}\geq \tau_D
]

触发 rollback 到当前 stable anchor (a)。

默认：

[
\tau_D=1 \text{ or } 2
]

建议第一版先用 calibration 选择，若没有 calibration，使用：

[
\tau_D=1
]

因为你当前策略偏激进：宁可误判，也不要放过坏 prefix。

---

# 13. Rollback 逻辑

假设当前触发 rollback 的 step 为 (k)，rollback anchor 为 (a)。

rollback span 长度：

[
m = k-a
]

如果：

[
m\leq 0
]

则不 rollback。

如果：

[
m > M_{\max}
]

则限制为 fallback：LLM 从当前上下文生成 1 step，或者直接把 anchor 设为 (k-M_{\max})。第一版建议 fallback，避免删除过长 prefix。

默认：

[
M_{\max}=5
]

如果合法，删除：

[
s_{a+1:k}
]

重建上下文：

[
C_a = q\oplus s_1\oplus \cdots \oplus s_a
]

注意：不要做 KV cache rollback。第一版直接用文本重建 context，稳定优先。

Codex 实现：

```python
def rollback_to_anchor(prompt, step_records, anchor_step_id, current_step_id):
    kept = [r for r in step_records if r.step_id <= anchor_step_id]
    removed = [r for r in step_records
               if anchor_step_id < r.step_id <= current_step_id]
    context = prompt + "".join(r.text for r in kept)
    m = current_step_id - anchor_step_id
    return context, kept, removed, m
```

---

# 14. LLM confidence-gated bounded recovery

Rollback 后，LLM 从 (C_a) 开始生成 recovery steps。

最大 recovery steps：

[
R_{\max}=m+1
]

其中 (m) 是删除的 span 长度。

每生成一个 LLM recovery step 后，都计算 SLM continuation confidence：

[
c^{rec}_r = 1-\widetilde{H}(p_s(\cdot\mid C^{rec}_r))
]

归一化：

[
\hat{c}^{rec}*r=F*{\text{calib}}(c^{rec}_r)
]

如果：

[
\hat{c}^{rec}_r \geq \theta_s
]

说明 LLM 已把 prefix 恢复到 SLM 可稳定续写的状态，停止 recovery。

因此：

[
R=\min{r\leq m+1: \hat{c}^{rec}_r\geq\theta_s}
]

若不存在，则：

[
R=m+1
]

并记录：

```python
recovery_exhausted = True
```

Recovery 结束后，强制 SLM 生成下一 step 一次：

```python
force_next_step_slm = True
```

强制一次后恢复正常 SARR-CoDE 主循环。

Codex 实现：

```python
def confidence_gated_recovery(
    context,
    llm,
    slm,
    normalizer,
    theta_s,
    max_recovery_steps,
    tokenizer,
):
    records = []
    for r in range(1, max_recovery_steps + 1):
        step = llm.generate_step(context)
        context = context + step.text

        c_raw, c_info = slm_continuation_confidence(
            slm, tokenizer, context
        )
        c_norm = normalizer.transform(c_raw)

        rec = {
            "recovery_step": r,
            "generator": "llm",
            "text": step.text,
            "token_ids": step.token_ids,
            "c_raw": c_raw,
            "c_norm": c_norm,
            "ready_for_slm": c_norm >= theta_s,
        }
        records.append(rec)

        if c_norm >= theta_s:
            return context, records, "SLM_READY"

    return context, records, "EXHAUSTED_FORCE_SLM"
```

---

# 15. Recovery 后状态重置

Rollback + recovery 后，不要继续沿用旧的 (D_{\text{start}})、(D_{\text{post}})。

重置：

```python
state = "STARTUP"
D_start = 0
D_post = 0
stable_anchor = None
force_next_step_slm = True
```

但不要清空全局日志。
不要删除 calibration normalizer。
不要清空已保留的 step records 和 recovery records。

由于 recovery 结束后强制 SLM 生成一步，系统会重新根据 continuation confidence 判断是否进入 STABLE 或再次触发 startup rollback。

---

# 16. 参数与 calibration

## 16.1 固定参数

建议先固定：

```yaml
topk_entropy: 20
smooth_window: 2
delta: 0.55
B_min: 2
B_max: 5
M_max: 5
force_slm_after_recovery: true
```

其中 (\delta=0.55) 来自 CoDE-style trend-aware degeneration 设计，但必须建立在 percentile-normalized confidence 上使用。CoDE-Stop 本身在其 trend-aware indicator 中使用固定阈值，并在实验中展示该类 threshold 有平滑 tradeoff。([arXiv][2])

## 16.2 需要 calibration 的参数

核心只需要：

[
\theta_s,\quad \tau_{\text{start}},\quad \tau_D
]

第一版可设：

[
\theta_s=0.70,\quad \tau_{\text{start}}=1,\quad \tau_D=1
]

更稳的方式是离线校准。

Calibration procedure：

1. 在 calibration split 上跑 SLM-only。
2. 收集所有 step 的 raw (c_k)，构建 percentile normalizer。
3. 用 candidate grid 模拟 SARR-CoDE 的 startup/post-stable rollback。
4. 选择满足 false rollback 风险约束的参数。

候选：

[
\theta_s \in {0.60,0.65,0.70,0.75,0.80}
]

[
\tau_{\text{start}}\in{1,2}
]

[
\tau_D\in{1,2}
]

如果你要激进：

[
\alpha_{\text{rollback}}=0.15
]

如果要保守：

[
\alpha_{\text{rollback}}=0.05
]

当前阶段建议先激进：

[
\alpha_{\text{rollback}}=0.15
]

因为目标是先通过日志发现可疑路径。

---

# 17. 完整在线算法伪代码

下面这段可以直接给 Codex。

```python
def run_sarr_code(
    problem_id,
    prompt,
    slm,
    llm,
    tokenizer,
    normalizer,
    cfg,
):
    context = prompt
    step_records = []
    rollback_events = []

    state = "STARTUP"
    D_start = 0
    D_post = 0
    stable_anchor = None

    c_norm_history = []
    c_smooth_history = []

    force_next_step_slm = False
    step_id = 0

    while step_id < cfg.max_steps:
        step_id += 1

        # ============================================================
        # 1. Decide generator
        # ============================================================
        # Main method: no Hinit routing. SLM generates by default.
        # LLM only appears inside rollback recovery.
        generator = "slm"

        if force_next_step_slm:
            generator = "slm"
            force_next_step_slm = False

        # ============================================================
        # 2. Generate one reasoning step
        # ============================================================
        step = slm.generate_step(
            context,
            stop_delimiters=cfg.step_delimiters,
            max_new_tokens=cfg.max_new_tokens_per_step,
        )

        context = context + step.text

        # ============================================================
        # 3. Compute SLM continuation confidence after this prefix
        # ============================================================
        c_raw, c_info = slm_continuation_confidence(
            slm=slm,
            tokenizer=tokenizer,
            context_text=context,
            topk=cfg.topk_entropy,
        )
        c_norm = normalizer.transform(c_raw)
        c_norm_history.append(c_norm)

        c_smooth = None
        if len(c_norm_history) >= cfg.smooth_window:
            c_smooth = mean(c_norm_history[-cfg.smooth_window:])
            c_smooth_history.append(c_smooth)

        # ============================================================
        # 4. Build record
        # ============================================================
        record = StepRecord(
            problem_id=problem_id,
            step_id=step_id,
            generator=generator,
            text=step.text,
            token_ids=step.token_ids,
            c_raw=c_raw,
            c_norm=c_norm,
            c_smooth=c_smooth,
            state_before=state,
            D_start=D_start,
            D_post=D_post,
            stable_anchor=stable_anchor,
            action="TRUST",
            extra=c_info,
        )
        step_records.append(record)

        # Not enough points to compute trend
        if c_smooth is None:
            record.state_after = state
            continue

        # ============================================================
        # 5. Stable detection
        # ============================================================
        if c_smooth >= cfg.theta_s:
            state = "STABLE"
            stable_anchor = step_id
            D_start = 0
            D_post = 0
            record.action = "REFRESH_STABLE_ANCHOR"
            record.state_after = state

            if is_finished(context):
                break
            continue

        # ============================================================
        # 6. Compute CoDE-style degeneration event
        # ============================================================
        v = 0
        if len(c_smooth_history) >= 2:
            prev_c = c_smooth_history[-2]
            curr_c = c_smooth_history[-1]
            v = int(
                (prev_c > curr_c)
                and (2.0 * curr_c - prev_c < cfg.delta)
            )

        record.degeneration_event = v

        # ============================================================
        # 7. STARTUP logic
        # ============================================================
        if state == "STARTUP":
            D_start += v
            record.D_start = D_start

            startup_steps = len(step_records)

            should_startup_rollback = False
            reason = None

            if startup_steps >= cfg.B_min and D_start >= cfg.tau_start:
                should_startup_rollback = True
                reason = "STARTUP_DEGENERATION"

            if startup_steps >= cfg.B_max:
                should_startup_rollback = True
                reason = "STARTUP_NOT_STABLE_WITHIN_BUDGET"

            if should_startup_rollback:
                anchor = choose_best_prefix_anchor(
                    step_records=step_records,
                    allow_zero=True,
                )

                context, kept, removed, m = rollback_to_anchor(
                    prompt=prompt,
                    step_records=step_records,
                    anchor_step_id=anchor,
                    current_step_id=step_id,
                )

                if m <= 0 or m > cfg.M_max:
                    # fallback: let LLM generate one step from current context
                    rec_context, rec_records, stop_reason = confidence_gated_recovery(
                        context=context,
                        llm=llm,
                        slm=slm,
                        normalizer=normalizer,
                        theta_s=cfg.theta_s,
                        max_recovery_steps=1,
                        tokenizer=tokenizer,
                    )
                    context = rec_context
                    recovery_steps = rec_records
                else:
                    rec_context, rec_records, stop_reason = confidence_gated_recovery(
                        context=context,
                        llm=llm,
                        slm=slm,
                        normalizer=normalizer,
                        theta_s=cfg.theta_s,
                        max_recovery_steps=m + 1,
                        tokenizer=tokenizer,
                    )
                    context = rec_context
                    recovery_steps = rec_records

                rollback_events.append({
                    "type": "STARTUP_ROLLBACK",
                    "reason": reason,
                    "trigger_step": step_id,
                    "anchor_step": anchor,
                    "rollback_span": m,
                    "removed_steps": serialize_steps(removed),
                    "recovery_steps": recovery_steps,
                    "stop_reason": stop_reason,
                })

                step_records = kept + convert_recovery_to_step_records(
                    recovery_steps,
                    start_step_id=anchor + 1,
                    problem_id=problem_id,
                )

                # reset local monitor after recovery
                context = prompt + "".join(r.text for r in step_records)
                state = "STARTUP"
                D_start = 0
                D_post = 0
                stable_anchor = None
                c_norm_history = []
                c_smooth_history = []
                force_next_step_slm = True
                step_id = len(step_records)
                continue

            record.state_after = state
            continue

        # ============================================================
        # 8. STABLE logic
        # ============================================================
        if state == "STABLE":
            D_post += v
            record.D_post = D_post

            if D_post >= cfg.tau_D:
                state = "DEGENERATED"
                record.action = "POST_STABLE_ROLLBACK"
                record.state_after = state

                anchor = stable_anchor

                context, kept, removed, m = rollback_to_anchor(
                    prompt=prompt,
                    step_records=step_records,
                    anchor_step_id=anchor,
                    current_step_id=step_id,
                )

                if m <= 0 or m > cfg.M_max:
                    max_recovery = 1
                else:
                    max_recovery = m + 1

                rec_context, rec_records, stop_reason = confidence_gated_recovery(
                    context=context,
                    llm=llm,
                    slm=slm,
                    normalizer=normalizer,
                    theta_s=cfg.theta_s,
                    max_recovery_steps=max_recovery,
                    tokenizer=tokenizer,
                )

                context = rec_context

                rollback_events.append({
                    "type": "POST_STABLE_ROLLBACK",
                    "trigger_step": step_id,
                    "anchor_step": anchor,
                    "rollback_span": m,
                    "removed_steps": serialize_steps(removed),
                    "recovery_steps": rec_records,
                    "stop_reason": stop_reason,
                })

                step_records = kept + convert_recovery_to_step_records(
                    rec_records,
                    start_step_id=anchor + 1,
                    problem_id=problem_id,
                )

                context = prompt + "".join(r.text for r in step_records)
                state = "STARTUP"
                D_start = 0
                D_post = 0
                stable_anchor = None
                c_norm_history = []
                c_smooth_history = []
                force_next_step_slm = True
                step_id = len(step_records)
                continue

            record.state_after = state
            continue

        if is_finished(context):
            break

    return {
        "problem_id": problem_id,
        "final_text": context,
        "step_records": step_records,
        "rollback_events": rollback_events,
    }
```

Codex 实现时可以简化部分细节，但上面的流程逻辑不能变。

---

# 18. 必要数据结构

## 18.1 StepRecord

```python
@dataclass
class StepRecord:
    problem_id: str
    step_id: int
    generator: str  # "slm" or "llm"
    text: str
    token_ids: list[int]

    c_raw: float | None = None
    c_norm: float | None = None
    c_smooth: float | None = None

    state_before: str | None = None
    state_after: str | None = None

    degeneration_event: int = 0
    D_start: int = 0
    D_post: int = 0
    stable_anchor: int | None = None

    action: str = "TRUST"
    extra: dict = field(default_factory=dict)
```

## 18.2 RollbackEvent

```python
@dataclass
class RollbackEvent:
    problem_id: str
    type: str  # STARTUP_ROLLBACK or POST_STABLE_ROLLBACK
    reason: str
    trigger_step: int
    anchor_step: int
    rollback_span: int

    removed_steps: list[dict]
    recovery_steps: list[dict]
    stop_reason: str  # SLM_READY or EXHAUSTED_FORCE_SLM

    force_next_step_slm: bool = True
```

---

# 19. 日志记录要求

必须保存 JSONL。否则后面无法分析方案是否有效。

## 19.1 每个 step 必须记录

```json
{
  "problem_id": "aime25_001",
  "step_id": 3,
  "generator": "slm",
  "text": "...",
  "token_count": 87,
  "c_raw": 0.421,
  "c_norm": 0.36,
  "c_smooth": 0.39,
  "theta_s": 0.70,
  "delta": 0.55,
  "degeneration_event": 1,
  "D_start": 1,
  "D_post": 0,
  "state_before": "STARTUP",
  "state_after": "STARTUP",
  "stable_anchor": null,
  "action": "STARTUP_ROLLBACK"
}
```

## 19.2 每次 rollback 必须记录

```json
{
  "problem_id": "aime25_001",
  "type": "STARTUP_ROLLBACK",
  "reason": "STARTUP_DEGENERATION",
  "trigger_step": 3,
  "anchor_step": 1,
  "rollback_span": 2,
  "removed_generators": ["slm", "slm"],
  "removed_text": ["...", "..."],
  "recovery_max_steps": 3,
  "recovery_actual_steps": 2,
  "recovery_c_norm": [0.51, 0.76],
  "stop_reason": "SLM_READY",
  "force_next_step_slm": true
}
```

## 19.3 额外 shadow 日志

虽然主方法不使用 boundary-safe，但建议记录 transition type：

```json
{
  "prev_generator": "llm",
  "curr_generator": "slm",
  "transition_type": "llm->slm",
  "delta_c_norm": -0.23
}
```

这样后面可以分析跨模型切换是否造成异常 confidence jump。

---

# 20. Codex 必须避免的实现错误

1. 不要实现 Hinit pre-step router。
2. 不要在 step 开始前根据 entropy 选择 LLM。
3. 不要实现 final-answer prompt。
4. 不要解析 `boxed`。
5. 不要把 LLM 用作 judge。
6. 不要把 recovery 后的 LLM step 当成最终答案。
7. 不要做 KV cache rollback。第一版用文本重建。
8. 不要在 rollback 后继续沿用旧的 degeneration score。
9. 不要让 LLM recovery 无限生成。最多 (m+1) steps。
10. Recovery 后必须强制 SLM 生成 1 step。

---

# 21. 推荐配置文件

```yaml
method: "sarr_code_aggressive_prefix"

generation:
  step_delimiters:
    - "\n\n"
  max_steps: 80
  max_new_tokens_per_step: 256

confidence:
  topk_entropy: 20
  percentile_normalization: true
  smooth_window: 2
  delta: 0.55

startup:
  B_min: 2
  B_max: 5
  tau_start: 1

stable:
  theta_s: 0.70   # replace by calibrated value if available
  tau_D: 1

rollback:
  M_max: 5
  recovery_max_policy: "m_plus_1"
  confidence_gated_recovery: true
  force_slm_after_recovery: true

logging:
  save_step_records: true
  save_rollback_events: true
  save_transition_stats: true
```

---

# 22. 实验检查项

实现后第一批日志要人工检查以下问题：

1. STARTUP 是否过于频繁 rollback？
2. `anchor_step=0` 的比例是否过高？
3. Recovery 后 `c_norm` 是否明显上升？
4. Recovery exhausted 的比例是否过高？
5. 强制 SLM 接管后是否立即再次 degeneration？
6. POST_STABLE rollback 是否删除了明显循环或不稳定 step？
7. 被删除 span 中是否包含 LLM step？如果有，删除后结果是否更好？
8. 错误样本中的 rollback 触发率是否高于正确样本？
9. 相比 SLM-only，是否提升 accuracy？
10. 相比 LLM-only，是否减少 LLM token？

---

# 23. 最终合理性检查

当前方案可以落地，理由如下。

第一，它只依赖 SLM next-token logits，工程上可直接由 transformers 本地模型得到。

第二，它不依赖 LLM logprob，不要求远程 LLM 返回概率，适合本地 SLM + vLLM/remote LLM 的混合架构。

第三，它没有 answer probe，不会偏离 SLM-LLM collaborative generation 的主题。

第四，它没有 Hinit，不会被视为 GlimpRouter 的简单变体。

第五，它使用 CoDE-style trend-aware degeneration，但把目标从 early stopping 改成 rollback-triggered collaboration，方法边界清楚。

第六，它使用 aggressive prefix rollback，允许删除 LLM 生成内容，符合你当前“前缀质量优先”的策略。

风险也要明确。

第一，continuation confidence 不一定总能反映 prefix quality。它是启发式信号，不是 correctness verifier。

第二，激进 rollback 可能误删正确 prefix。当前阶段通过日志评估，后续可用 calibration 收紧。

第三，1.5B SLM 的 confidence 可能噪声大。因此 percentile normalization 是必须项。

第四，step delimiter 会影响 (c_k) 序列。第一版固定 `\n\n`，后续再做 delimiter ablation。

第五，recovery 后强制 SLM 接管可能导致再次失败，但这有助于检验 LLM 是否真正修复了 prefix。

最终建议：**先实现 aggressive 版本，不加入 boundary-safe 或 Hinit。所有保守策略只作为日志对照，不进入主决策。**

[1]: https://arxiv.org/abs/2601.05110?utm_source=chatgpt.com "GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token of Thoughts"
[2]: https://arxiv.org/html/2604.04930v1 "Early Stopping for Large Reasoning Models via Confidence Dynamics"
