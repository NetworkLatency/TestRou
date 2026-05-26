# PDI Step-Window Controller 建模与实现规格

本文档用于指导实现新版 SARR-CoDE ownership controller。目标是把当前方法压缩成一个清晰、可审查、可消融的 **PDI-step-window controller**。

核心原则：

1. 单一原始观测量只使用 PDI。
2. 决策依据不是固定 PDI 阈值，而是当前问题内的 prior-calibrated percentile rank。
3. SLM 让位给 LLM 使用 upper-tail sequential evidence。
4. LLM 交还 SLM 使用 candidate-owner readiness test，即 SLM-side PDI。
5. rollback 后必须删除被回滚 token span 覆盖的 trusted PDI windows。
6. early stop 只在 answer-intent 之后检测 lower-tail plateau。

---

## 1. 记号定义

令当前生成序列为：

$$
x_{1:T}=(x_1,x_2,\ldots,x_T)
$$

其中 $x_i$ 是 token。

当前 owner 模型记为：

$$
o_t \in \{S,L\}
$$

其中：

- $S$ 表示 SLM；
- $L$ 表示 LLM。

token-level negative log-probability 定义为：

$$
\ell_i^{(o)}
=
-\log p_o(x_i \mid x_{<i})
$$

PDI 不是正确性指标，而是模型维持当前生成轨迹所需的自信息代价。

---

## 2. Step Boundary

生成过程按语义 step 结算。主边界为双换行：

```text
\n\n
```

一个 step 表示为：

$$
s_k=(x_{a_k},x_{a_k+1},\ldots,x_{b_k})
$$

其中 $a_k$ 和 $b_k$ 是该 step 覆盖的 token index。

step 关闭规则：

1. 如果生成遇到 `\n\n`，额外检查后续 16 token 中是否出现 EOS 或 `</think>` , 如没有则关闭当前 step。
2. 如果生成遇到 EOS 或 `</think>`，结束 think 部分，直接进入回答。
3. v0 可以不实现 long-step observation fallback；如果实现，只能作为 observation-only scoring，不能截断文本或强行切换语义 step。

每个 step 需要保存：

```python
class Step:
    step_id: int
    owner: str  # "SLM" or "LLM"
    text: str
    token_ids: list[int]
    logprobs: list[float]
    start_token_idx: int
    end_token_idx: int
    active: bool
```

---

## 3. Step-Window PDI

每当一个完整 step 生成后，尝试构造一个 PDI window。

PDI window 必须是 **episode-local** 且 **owner-homogeneous** 的。也就是说，窗口只能由当前 ownership episode 内、由同一个 owner 生成的 completed active steps 构成，不能跨越 SLM/LLM ownership boundary。

给定当前 ownership episode $e$ 内已完成的 active steps：

$$
s^{(e)}_1,s^{(e)}_2,\ldots,s^{(e)}_t
$$

定义窗口 $W_t$ 为最短 step 后缀，使其 token 数不少于 $T_{\min}$（超出部分保证是一个完整的 step 即可）：

$$
W_t
=
\operatorname{shortest\ suffix}
\left\{
s^{(e)}_j,\ldots,s^{(e)}_t:
\sum_{k=j}^{t}|s_k|\ge T_{\min}
\right\}
$$

其中 $T_{\min}$ 是 PDI window 的最小 token 数。

窗口内 token 数为：

$$
|W_t|
=
\sum_{s_k\in W_t}|s_k|
$$

PDI 定义为：

$$
y_t
=
PDI(W_t)
=
\frac{1}{|W_t|}
\sum_{x_i\in W_t}
-\log p_{o_t}(x_i\mid x_{<i})
$$

其中 $o_t$ 是生成该 window 的 owner。

每个 PDI window 必须绑定 token span：

```python
class PDIWindow:
    window_id: int
    owner: str  # "SLM" or "LLM"
    covered_step_ids: list[int]
    start_token_idx: int
    end_token_idx: int
    pdi: float
    status: str  # "trusted", "suspect", "failure", "invalid"
```

token span 绑定是必要逻辑，因为 rollback 时必须删除或失效化与 rollback span 重叠的 trusted windows。

---

## 4. Prior-Calibrated Effective Distribution

### 4.1 Pilot Prior

离线构造 SLM 的 PDI prior：

$$
B_0^S
=
\{y_i:
y_i \text{ is an accepted SLM active PDI window in pilot traces}\}
$$

accepted 的定义不能依赖最终答案是否正确，避免 oracle leakage。建议定义为：

```text
最终保留在 active trajectory 中，且没有被 rollback / discarded 的 SLM PDI window。
```

pilot prior 需要保存 raw distribution，而不只保存均值方差。

建议记录：

```text
median
p75
p90
p95
raw values
```

当前实验可作为初始参考：

$$
Q_{50}\approx 0.12,\quad
Q_{75}\approx 0.19,\quad
Q_{90}\approx 0.29,\quad
Q_{95}\approx 0.36
$$

这些数值只作为 cold-start prior，不应作为固定全局阈值。

### 4.2 Trusted Buffer

当前问题内的 trusted SLM buffer 定义为：

$$
B_t^S
=
\{y_j:
y_j \text{ is a trusted SLM PDI window on the active trajectory}\}
$$

必须区分：

```python
B_trusted: list[PDIWindow]
B_failure: list[PDIWindow]
```

在线控制只使用 `B_trusted`。`B_failure` 只用于日志、可视化和 ablation。

### 4.3 Effective Distribution

令 $F_0^S$ 是 prior distribution $B_0^S$ 的经验 CDF，$F_{B_t^S}$ 是当前问题 trusted buffer 的经验 CDF。

有效分布为：

$$
\hat F_t^S(y)
=
\frac{
\lambda_0 F_0^S(y)
+
n_t F_{B_t^S}(y)
}{
\lambda_0+n_t
}
$$

其中：

$$
n_t=|B_t^S|
$$

$\lambda_0$ 是 prior pseudo-count。推荐：

$$
\lambda_0 \in \{3,5\}
$$

prior influence 为：

$$
\frac{\lambda_0}{\lambda_0+n_t}
$$

因此不需要额外设计复杂衰减函数。

---

## 5. Percentile Rank

对当前 PDI window：

$$
y_t=PDI(W_t)
$$

计算其在有效分布中的 percentile rank：

$$
q_t
=
\hat F_t^S(y_t)
$$

含义：

```text
q_t 越接近 1，当前 PDI 越处于上尾；
q_t 越接近 0，当前 PDI 越处于下尾。
```

控制器后续不直接比较 PDI 绝对值，而比较 $q_t$。

---

## 6. COLD_START

定义：

$$
n_t<n_{\min}
\Rightarrow
\text{COLD\_START}
$$

COLD_START 不是 warmup，也不是无监控阶段。

COLD_START 期间：

1. 使用 prior-dominated $\hat F_t^S$；
2. 允许 upper-tail monitor；
3. 允许 rollback + LLM repair；
4. 禁用 early stop；
5. 只把明确保留在 active trajectory 中的窗口加入 trusted buffer。

退出 COLD_START：

$$
n_t\ge n_{\min}
$$

推荐：

$$
n_{\min}\in\{3,5\}
$$

---

## 7. Upper-Tail Evidence：SLM 让位给 LLM

upper-tail 用于检测 SLM 是否进入高 self-information cost regime。

注意：upper-tail PDI 不表示答案错误，只表示 SLM 维持当前轨迹的预测代价异常升高。

给定上尾起点：

$$
q_{high}\in(0,1)
$$

推荐：

$$
q_{high}=0.90
\quad\text{or}\quad
q_{high}=0.95
$$

定义归一化 upper-tail excess：

$$
a_t
=
\frac{
\max(0,q_t-q_{high})
}{
1-q_{high}
}
$$

因此：

$$
a_t\in[0,1]
$$

在最近 $r_u$ 个 PDI windows 上累积：

$$
A_t
=
\frac{1}{r_u}
\sum_{i=t-r_u+1}^{t}a_i
$$

触发条件：

$$
A_t>\eta_u
\Rightarrow
\text{rollback + LLM repair}
$$

其中：

- $q_{high}$：什么分位数以上视作 upper-tail；
- $r_u$：观察多少个 PDI windows；
- $\eta_u$：平均 upper-tail evidence 强度阈值。

不建议使用额外 soft/hard 双阈值体系。

### 7.1 Trusted 写入规则

upper-tail evidence 与 trusted baseline 更新必须解耦。

如果当前窗口满足：

$$
a_t > 0
$$

则该窗口不得立即加入 `B_trusted`。它应暂存为 pending/suspect evidence，用于 upper-tail evidence accumulation。

v0 中采用最简单的规则：

```text
if a_t == 0 and window remains on the active trajectory:
    add W_t to B_trusted
else:
    keep W_t out of B_trusted
---

## 8. Rollback

当 upper-tail evidence 触发时，回滚到本次 evidence 开始累积的位置。

定义 alarm 起点：

$$
\tau
=
\min\{i:
a_i>0
\text{ and } a_i \text{ contributes to current alarm}\}
$$

rollback span：

$$
R
=
[start(W_\tau),\ end(W_t)]
$$

动作：

$$
\text{rollback}(R),
\quad
o_{t+1}\leftarrow L
$$

同时处理 trusted buffer：

$$
B_t^S
\leftarrow
B_t^S
\setminus
\{W_j:
W_j\cap R\neq\emptyset\}
$$

被删除的窗口可以标记为：

```text
invalid
```

或移动到：

```text
B_failure
```

在 upper-tail evidence 开始累积时，应保存最近一次可信 SLM baseline snapshot：

$$
B_{\text{pre}}^S
=
B_{\text{last-trusted}}^S
$$

其中 $B_{\text{last-trusted}}^S$ 是进入 alarm region 前的 trusted SLM buffer，而不是被 SUSPECT / rollback 区间污染后的 buffer。

handoff test 使用冻结的 effective baseline：

$$
\hat F_{\text{pre}}^S(y)
=
\frac{
\lambda_0 F_0^S(y)
+
n_{\text{pre}} F_{B_{\text{pre}}^S}(y)
}{
\lambda_0+n_{\text{pre}}
}
$$

其中：

$$
n_{\text{pre}}=|B_{\text{pre}}^S|
$$

如果 $B_{\text{pre}}^S$ 很小或为空，handoff test 仍然可以退化为 prior-dominated baseline，而不会报错。

后续 LLM repair 期间不得更新 $B_{\text{pre}}^S$。

---

## 9. LLM Repair

rollback 后，由 LLM 从 rollback point 继续生成。

LLM repair 阶段仍然按 `\n\n` 形成 step。

LLM repair 的目标不是接管整个回答，而是生成一个 repaired prefix，使 SLM 有机会重新接续。

LLM repair 有三类出口：

1. SLM-side handoff readiness 成立，进入 SLM_PROBATION；
2. 达到局部 repair budget，进入 LLM finalize 或严格 handoff；
3. 如果实现 early stop for LLM，可以在 answer-intent 后低 PDI plateau 时进入 finalization。

v0 推荐只实现：

```text
handoff readiness
max_llm_repair_steps
```

---

## 10. SLM-Side Handoff Test

LLM 是否稳定不是 handback 的主要依据。

handoff 判断必须回答：

```text
SLM 是否能够在 repaired prefix 上重新进入可接续的 PDI regime？
```

因此，对 LLM 生成的 repair suffix 使用 SLM teacher-forced scoring。
实现时必须注意：SLM-side scoring 以文本为输入，并使用 SLM tokenizer 重新分词。不得复用 LLM token ids 或 LLM logprobs。

具体实现应为：

```text
prefix_text = active trajectory text before the scored LLM suffix
suffix_text = recent LLM repair suffix text

Run SLM teacher-forced scoring on:
    prefix_text + suffix_text

Compute NLL only over suffix_text tokens under the SLM tokenizer.

设 LLM repair suffix window 为：

$$
\tilde W_t^{L}
$$

定义 SLM-side PDI：

$$
\tilde y_t^{S\leftarrow L}
=
\frac{1}{|\tilde W_t^{L}|}
\sum_{x_i\in \tilde W_t^{L}}
-\log p_S(x_i^{L}\mid x_{<i})
$$

用 rollback 前冻结的 SLM baseline snapshot 形成经验分布：

$$
\hat F_{\text{pre}}^S
$$

计算：

$$
\tilde q_t^{S\leftarrow L}
=
\hat F_{\text{pre}}^S
\left(
\tilde y_t^{S\leftarrow L}
\right)
$$

handoff 条件：

$$
\tilde q_t^{S\leftarrow L}
\le
q_{handoff}
$$

连续满足 $r_h$ 个 handoff scoring windows：

$$
\tilde q_{t-r_h+1:t}^{S\leftarrow L}
\le
q_{handoff}
$$

则：

$$
\text{LLM\_REPAIR}
\rightarrow
\text{SLM\_PROBATION}
$$

默认建议：

$$
q_{handoff}=q_{high}
$$

例如当：

$$
q_{high}=0.90
$$

则：

$$
q_{handoff}=0.90
$$

含义：SLM-side PDI 不需要特别低，只要没有落入 SLM trusted baseline 的 upper-tail 区间，就允许进入 SLM_PROBATION。

更严格的值，例如：

$$
q_{handoff}=0.75
\quad\text{or}\quad
0.80
$$

可以作为 ablation 设置，但不建议作为 v0 默认值。因为 handoff test 只是进入 probation 的软门槛，真正的 handback 成功需要由 SLM_PROBATION 验证。

重要实现规则：

1. LLM-generated tokens 的 SLM-side PDI 只进入 handoff buffer；
2. 不得直接写入 SLM trusted baseline；
3. handoff buffer 只用于 readiness test；
4. handoff 失败时丢弃该临时 buffer。

---

## 11. SLM_PROBATION

teacher-forced scoring 不等于 SLM 自己继续生成稳定，因此 handoff 后必须进入 probation。

状态流：

```text
LLM_REPAIR
-> HANDOFF_TEST
-> SLM_PROBATION
-> SLM_NORMAL
```

SLM_PROBATION 从 handoff point 开始，由 SLM 继续生成自己的 step-window PDI。

probation 期间：

1. 不正常更新 `B_trusted`；
2. 不触发普通 rollback；
3. 只监控 severe upper-tail failure；
4. 如果失败，rollback 到 handoff point，由 LLM continue repair。

v0 中可以直接复用 upper-tail evidence：

$$
A_t>\eta_u
\Rightarrow
\text{rollback to handoff point + LLM continue repair}
$$

如果 SLM 在 probation 内稳定通过 $m$ 个 PDI windows：

$$
\text{stable\_count}\ge m
$$

则：

$$
\text{SLM\_PROBATION}
\rightarrow
\text{SLM\_NORMAL}
$$

probation 成功后，SLM 自己生成的 probation windows 才可以进入 trusted buffer。

推荐：

$$
m\in\{2,3\}
$$

---

## 12. Lower-Tail Plateau 与 Early Stop

early stop 不能只看低 PDI。低 PDI 可能只是正常确定性推导。

因此 v0 使用：

```text
lower-tail PDI plateau + answer-intent gate
```

定义 answer-intent：

$$
I_t=1
$$

当当前或历史 step 中出现以下信号之一：

```text
final answer
the answer is
therefore the answer
\boxed{
answer:
```

可根据数据集格式扩展，但不要做成复杂行为规则。
Answer-intent 不是 model-switching signal，也不是 correctness signal。它只是 finalization gate，用于防止 lower-tail early stop 在模型尚未尝试给出答案之前触发。真正的 stopping evidence 仍然来自 PDI lower-tail plateau。

lower-tail 条件：

$$
q_t\le q_{low}
$$

持续 $r_l$ 个 windows：

$$
q_{t-r_l+1:t}\le q_{low}
$$

同时：

$$
I_{\le t}=1
$$

且不处于 COLD_START：

$$
n_t\ge n_{\min}
$$

触发：

$$
\text{early stop reasoning}
\rightarrow
\text{finalization}
$$

论文表述建议：

```text
lower-tail plateau after answer-intent does not prove correctness;
it indicates low marginal gain of continued reasoning.
```

推荐：

$$
q_{low}=0.10
\quad\text{or}\quad
0.20
$$

---

## 13. Controller State

建议实现状态：

```python
class EpisodeState:
    owner: str  # "SLM" or "LLM"
    mode: str   # "COLD_START", "SLM_NORMAL", "LLM_REPAIR", "SLM_PROBATION"
    trusted_buffer: list[PDIWindow]
    failure_buffer: list[PDIWindow]
    pre_suspect_snapshot: list[float]
    upper_evidence_history: list[float]
    lower_tail_history: list[float]
    handoff_history: list[float]
    handoff_point_token_idx: int | None
```

建议实现 controller：

```python
class PDIController:
    prior_distribution: list[float]
    lambda0: float
    config: ControllerConfig

    def build_step_window(self, active_steps: list[Step]) -> PDIWindow:
        ...

    def effective_cdf(self, trusted_buffer: list[PDIWindow]) -> EmpiricalCDF:
        ...

    def percentile_rank(self, value: float, cdf: EmpiricalCDF) -> float:
        ...

    def update_upper_evidence(self, q: float) -> float:
        ...

    def should_repair(self, q: float) -> bool:
        ...

    def rollback(self, alarm_start_window: PDIWindow) -> None:
        ...

    def score_handoff_with_slm(self, llm_suffix: list[Step]) -> float:
        ...

    def should_handoff(self, q_slm_side: float) -> bool:
        ...

    def update_probation(self, q: float) -> str:
        ...

    def should_early_stop(self, q: float, answer_intent: bool) -> bool:
        ...
```

---

## 14. Minimal Hyperparameters

v0 推荐只保留以下配置项。

| Hyperparameter | Role | Necessity | Suggested Value |
| --- | --- | --- | --- |
| `T_min` | PDI window 最小 token 数 | 必要 | 32 或 64 |
| `lambda0` | pilot prior pseudo-count | 必要 | 3 或 5 |
| `n_min` | 退出 COLD_START 的 trusted window 数 | 必要 | 3 或 5 |
| `q_high` | upper-tail 起点 | 必要 | 0.90 或 0.95 |
| `r_upper` | upper-tail evidence 观察窗口数 | 必要 | 2 或 3 |
| `eta_upper` | 平均 upper-tail evidence 阈值 | 必要 | 0.40 到 0.60 |
| `q_handoff` | SLM-side handoff 容忍分位数 | 必要 | 默认等于 `q_high`，如 0.90；0.75/0.80 作为 ablation |
| `r_handoff` | handoff readiness 持续窗口数 | 必要 | 1 或 2 |
| `m_probation` | SLM probation 通过窗口数 | 必要 | 2 或 3 |
| `q_low` | lower-tail plateau 分位数 | 必要 | 0.10 或 0.20 |
| `r_low` | lower-tail 持续窗口数 | 必要 | 2 或 3 |
| `max_llm_repair_steps` | LLM repair 安全预算 | 工程保险 | 3 到 8 steps；耗尽后 v0 进入 LLM_FINALIZE |

建议不要加入：

```text
soft/hard 双阈值
EWMA 参数
CUSUM 参数
LLM 自身稳定性 handoff 主条件
复杂 behavior probe
```

---

## 15. End-to-End Algorithm

### 15.1 Main Loop

控制器必须优先按 `mode` 分派逻辑，而不是优先按 `owner` 分派。否则当 `owner == SLM` 且 `mode == SLM_PROBATION` 时，probation 分支会被普通 SLM 分支提前吞掉。

```text
Initialize:
    owner = SLM
    mode = COLD_START
    B_trusted = []
    B_failure = []
    prior = B0_S
    reset all evidence histories

While generation not finished:

    if mode in {COLD_START, SLM_NORMAL} and owner == SLM:
        SLM generates until step boundary
        close Step

        build episode-local PDIWindow W_t
        if no valid W_t because current episode has fewer than T_min completed tokens:
            continue

        compute F_eff from prior and B_trusted
        q_t = F_eff(PDI(W_t))

        compute upper-tail excess a_t
        update upper-tail evidence A_t

        if a_t == 0 and W_t remains active:
            add W_t to B_trusted
        else:
            keep W_t out of B_trusted as pending/suspect evidence

        if mode == COLD_START and len(B_trusted) >= n_min:
            mode = SLM_NORMAL

        if A_t > eta_upper:
            freeze B_pre_S = last trusted SLM baseline before alarm region
            mark current alarm windows as suspect/failure
            rollback to alarm start
            invalidate trusted windows overlapping rollback span
            reset evidence histories
            owner = LLM
            mode = LLM_REPAIR
            continue

        if mode != COLD_START:
            check answer-intent
            update lower-tail persistence
            if answer-intent and lower-tail persists:
                mode = FINALIZE
                break


    elif mode == LLM_REPAIR and owner == LLM:
        LLM generates until step boundary
        close Step

        if enough LLM repair suffix exists:
            retokenize recent LLM suffix with SLM tokenizer
            compute SLM-side PDI on suffix tokens
            q_h = F_pre_S(SLM_side_PDI)
            update handoff readiness history

            if handoff readiness holds for r_handoff:
                handoff_point = current token index
                reset evidence histories
                owner = SLM
                mode = SLM_PROBATION
                reset probation counters
                continue

        if max_llm_repair_steps reached and handoff readiness is not satisfied:
            mode = LLM_FINALIZE
            owner = LLM
            break


    elif mode == SLM_PROBATION and owner == SLM:
        SLM generates until step boundary
        close Step

        build episode-local PDIWindow W_t
        if no valid W_t because current episode has fewer than T_min completed tokens:
            continue

        compute q_t using the restored SLM effective distribution
        compute upper-tail excess a_t
        update upper-tail evidence A_t

        # Probation does not perform ordinary rollback or normal baseline update.
        # It only checks severe handback failure using the same upper-tail evidence.

        if A_t > eta_upper:
            rollback to handoff point
            invalidate steps/windows generated after handoff point
            reset evidence histories
            owner = LLM
            mode = LLM_REPAIR
            continue

        if probation has remained stable for m_probation valid PDI windows:
            add successful probation windows to B_trusted
            reset evidence histories
            mode = SLM_NORMAL
            continue


    elif mode in {FINALIZE, LLM_FINALIZE}:
        finish generation according to current owner/finalization policy
        break


    else:
        raise InvalidControllerState(owner, mode)


```

### 15.2 Rollback Invalidation

v0 只支持 step-boundary rollback。也就是说：

```text
rollback_start_token_idx must align with the start of a completed step.
partial-step rollback is not supported in v0.

rollback span：

rollback_span = [rollback_start_token_idx, current_end_token_idx]

trusted PDI windows invalidation：

for window in B_trusted:
    if overlaps(window.span, rollback_span):
        window.status = "invalid"
        remove window from B_trusted
        append window to B_failure

step invalidation：

for step in active_steps:
    if overlaps(step.span, rollback_span):
        step.active = False

```
Overlap condition:

$$
W_j\cap R\neq\emptyset
\iff
start(W_j)\le end(R)
\land
end(W_j)\ge start(R)
$$

Rollback 后必须 reset：

upper_evidence_history
lower_tail_history
handoff_history
probation counters

注意：rollback 删除的是 active trajectory 上的文本和窗口。被删除窗口可以进入 B_failure 用于日志和分析，但不得继续参与在线 threshold / percentile computation。
---

## 16. Required Logging

每个 PDI decision point 必须记录：

```text
problem_id
step_id
owner
mode
start_token_idx
end_token_idx
pdi
q_percentile
upper_excess
upper_evidence
lower_tail_count
answer_intent_seen
action
trusted_buffer_size
prior_weight
rollback_start_token_idx
handoff_q_slm_side
probation_status
```

这些日志用于后续 ablation 和论文分析。

---

## 17. Ablation Checklist

实现后至少需要以下消融：

```text
PDI controller full
no prior
fixed global PDI threshold
dynamic percentile without rollback invalidation
dynamic percentile without SLM-side handoff
dynamic percentile without probation
early stop with AnswerIntent
early stop without AnswerIntent
PDI vs entropy
PDI vs top-1/top-2 margin
```

主要指标：

```text
accuracy
total wall time
SLM wall time
LLM wall time
LLM decode tokens
SLM scoring overhead
LLM participation rate
rollback count
handoff success rate
probation failure rate
early-stop trigger count
early-stop correctness impact
```

---

## 18. Important Claims and Non-Claims

论文中可以主张：

```text
PDI measures trajectory self-information cost.
The controller monitors regime changes in the PDI flow.
The threshold is problem-adaptive through prior-calibrated percentile rank.
Rollback is triggered by sustained upper-tail evidence, not single-token uncertainty.
Handoff is decided from the candidate owner's perspective via SLM-side PDI.
Early stop targets low marginal gain after answer-intent, not correctness proof.
```

论文中不要主张：

```text
High PDI proves the answer is wrong.
Low PDI proves the answer is correct.
The controller finds the optimal intervention point.
SLM and LLM PDI are globally comparable.
LLM stability implies SLM readiness.
```

---

## 19. Minimal Implementation Target

v0 的闭环目标：

```text
SLM generates step by \n\n
-> build semantic PDI window with token lower bound
-> compute prior-calibrated percentile rank
-> accumulate upper-tail evidence
-> rollback + LLM repair
-> SLM-side handoff test
-> SLM probation
-> return to SLM normal
-> lower-tail plateau + answer-intent early stop
```

最终方法可以概括为：

$$
\boxed{
PDI
+
active\ trajectory\ trusted\ buffer
+
pilot-prior\ calibrated\ quantile\ rank
+
upper/lower-tail\ sequential\ evidence
}
$$

---

## 20. Implementation Clarifications

为了避免实现偏离方法意图，v0 必须遵守以下约束。

### 20.1 Mode-first Dispatch

controller main loop 必须优先根据 `mode` 分支，而不是优先根据 `owner` 分支。

合法状态包括：

```text
COLD_START
SLM_NORMAL
LLM_REPAIR
SLM_PROBATION
FINALIZE
LLM_FINALIZE
不允许出现：

owner == SLM and mode == LLM_REPAIR
owner == LLM and mode == SLM_PROBATION

如果出现，应直接抛出 InvalidControllerState。
