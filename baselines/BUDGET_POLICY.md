# Experiment Budget Policy

Current comparison experiments use a unified per-problem completion-token budget:

```text
B_total = 16384
```

For methods with an explicit reasoning/final-answer split:

```text
reasoning or think budget = 14336
answer budget = 2048
```

For methods without a separate answer phase:

```text
generation budget = 16384
```

Concrete defaults in this repository:

| Method | Budget setting |
| --- | --- |
| SARR-CoDE | `think_token_budget=14336`, `answer_token_budget=2048` |
| Single-model baseline | `max_new_tokens=16384` by default, derived from SARR config |
| GlimpRouter | `token_budget=14336`, `answer_max_tokens=2048` |
| STEER | `max_tokens_per_call=16384` |
| SpecReason | `token_budget=16384` |
| RSD | `max_tokens_per_call=16384` |
| RelayGen | `budget=16384` |

Segment-level limits such as `step_max_tokens=512` and method-specific scoring calls are kept unchanged because they are part of each method's routing mechanism.
