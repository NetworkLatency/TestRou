# Engineering Notes

This file preserves implementation ideas that were useful in earlier diagnostic experiments, even though the old sampling-disagreement and boundary-continuation experiments have been removed.

## Step Termination

- Keep strict step boundaries explicit. For stepwise reasoning, use `"\n\n"` as the normal stop string and append it when the backend stops for that boundary.
- Treat EOS separately from a step-boundary stop. EOS should not receive an extra artificial step terminator.
- Keep thinking and answer phases separate when reproducing GlimpRouter-style experiments: the thinking phase has its own budget, and the final answer is generated in a separate call with its own budget.

## Repetition Guards

- Track recent normalized steps and stop on exact duplicate or alternating duplicate steps.
- Avoid using internal n-gram repetition as a default stop condition for math reasoning unless it has been validated for the specific experiment. Repeated LaTeX fragments such as `\frac{...}` can otherwise create false positives.
- When repetition happens before a final answer phase, prefer recording a clear stop reason over silently continuing.

## Context And Budget Accounting

- Count prefill and decode tokens separately for SLM and LLM calls.
- Probe calls should be accounted for explicitly. If a probe is reused as generated text, mark that in the step log; if it is only a scoring glimpse, do not add it to the assistant prefix.
- Keep visible thinking tokens separate from total model tokens. Routing probes and discarded scoring glimpses affect cost but should not always advance the visible reasoning transcript.

## Trace Hygiene

- Per-step logs should avoid storing full prompt text or large raw rollouts in summary CSV files. Put large artifacts in JSONL or trace files instead.
- For resume support, write per-problem outputs and summary rows atomically where possible.
- Keep enough metadata to audit routing decisions: threshold, signal value, decision, generation source, token budget, and whether any probe text was reused.

## Failure Labels

Useful stop reasons to preserve across scripts:

- `context_budget`
- `context_budget_final_answer`
- `think_token_budget`
- `empty_step`
- `finished`
- `eos`
- `duplicate_step`
- `alternating_step`
