# SLM Sampling Disagreement Experiments

This repository now focuses on one research question:

> Can SLM self-sampling disagreement at reasoning boundaries reveal where the SLM reaches its reasoning limit?

The retained runnable paths are:

- `slm_only`
- `llm_only`
- `glimprouter_hinit`
- pure-SLM boundary sampling diagnostics
- LLM oracle / boundary continuation labeling
- disagreement analysis and top-quantile sanity routing

## 1. Setup

Install dependencies on the GPU host:

```bash
pip install -r requirements.txt
```

Edit `configs/bpa_default.json` before running experiments:

```json
{
  "slm_model_path": "/path/to/DeepSeek-R1-Distill-Qwen-1.5B",
  "llm_model_path": "/path/to/Qwen-or-DeepSeek-target-model",
  "slm_backend": "vllm",
  "llm_backend": "vllm",
  "dataset_paths": {
    "math500": "data/math500.jsonl",
    "aime24": "data/aime24.jsonl"
  }
}
```

By default both models are loaded in-process with `vllm.LLM`. To use a target model served by an OpenAI-compatible vLLM server, set the LLM backend fields:

```json
{
  "llm_model_path": "Qwen3-14B",
  "llm_tokenizer_path": "/path/to/local/Qwen3-14B-tokenizer-or-model",
  "llm_backend": "openai",
  "llm_api_base_url": "http://192.168.3.13:8080/v1",
  "llm_api_key": "EMPTY",
  "llm_api_model": "Qwen3-14B"
}
```

`llm_api_model` must match the model name exposed by the remote vLLM server. `llm_tokenizer_path` should point to a local tokenizer so the experiment host can render chat templates and estimate context length without network access. Setting `llm_api_base_url` is enough to switch the LLM engine to the OpenAI-compatible backend; leaving it `null` keeps local vLLM loading.

Supported local dataset formats: `.jsonl`, `.json`, `.csv`, `.tsv`, `.parquet`.

For MATH/AIME rows, provide `problem` and one of `answer`, `solution`, or `target`.

## 2. Test Commands

Run static compilation and unit tests:

```bash
python -m compileall bpa tests
python -m unittest tests.test_bpa_core
```

Check chat-template rendering:

```bash
python -m bpa.eval.render_sanity \
  --config configs/bpa_default.json
```

## 3. Baseline Runs

Run 50 MATH500 problems for the retained baselines:

```bash
for variant in slm_only llm_only glimprouter_hinit; do
  python -m bpa.eval.main_benchmark \
    --config configs/bpa_default.json \
    --variant "${variant}" \
    --dataset math500 \
    --max-problems 50
done
```

Outputs:

```text
bpa_results/math500/{variant}/summary.csv
bpa_results/math500/{variant}/summary_metrics.json
bpa_results/math500/{variant}/{problem_id}/
```

## 4. Boundary Sampling

Collect pure-SLM boundary-level K-rollout disagreement on MATH500:

```bash
python -m bpa.eval.exp_sampling_disagreement \
  --config configs/bpa_default.json \
  --dataset math500 \
  --max-problems 100 \
  --probe-k 4 \
  --probe-temperature 0.7 \
  --probe-max-tokens 32
```

For AIME24:

```bash
python -m bpa.eval.exp_sampling_disagreement \
  --config configs/bpa_default.json \
  --dataset aime24 \
  --max-problems 30 \
  --probe-k 4 \
  --probe-temperature 0.7 \
  --probe-max-tokens 32
```

Outputs:

```text
bpa_results/diagnostics/sampling_disagreement/{dataset}/probes.jsonl
bpa_results/diagnostics/sampling_disagreement/{dataset}/problem_summary.csv
```

Probe rows record sampled rollouts plus evidence channels used by the current router:

- `boxed_answer`
- `rhs_novel_number`
- `equation_claim`
- `novel_number_set`
- `operation_intent`

## 5. LLM Oracle

Run LLM-only oracle traces for the same problem slice:

```bash
python -m bpa.eval.exp_llm_oracle \
  --config configs/bpa_default.json \
  --dataset math500 \
  --max-problems 100
```

Outputs:

```text
bpa_results/diagnostics/llm_oracle/math500/oracle_summary.csv
bpa_results/diagnostics/llm_oracle/math500/oracle_traces.jsonl
```

## 6. Boundary Labels

For each problem, select evenly spaced SLM boundaries and let the LLM continue from each prefix:

```bash
python -m bpa.eval.exp_boundary_continuation \
  --config configs/bpa_default.json \
  --dataset math500 \
  --max-problems 100 \
  --boundaries-per-problem 5 \
  --continuation-max-tokens 2048
```

Outputs:

```text
bpa_results/diagnostics/boundary_continuation/math500/boundary_labels.csv
bpa_results/diagnostics/boundary_continuation/math500/boundary_labels.jsonl
```

Default `critical=True` definition:

```text
SLM final answer is wrong
AND LLM oracle is not failed
AND LLM continuation from this boundary answers correctly
```

## 7. Analysis and Plotting

Generate distribution plots, quantile plots, dip-test results, and AUROC for numeric probe fields:

```bash
python -m bpa.eval.analyze_sampling_disagreement \
  --config configs/bpa_default.json \
  --dataset math500 \
  --metrics prefix_consensus_support_count prefix_consensus_vote_fraction \
  --num-bins 10
```

Plot and analysis outputs:

```text
bpa_results/diagnostics/sampling_analysis/math500/analysis_summary.json
bpa_results/diagnostics/sampling_analysis/math500/distribution_summary.csv
bpa_results/diagnostics/sampling_analysis/math500/distribution_{metric}.png
bpa_results/diagnostics/sampling_analysis/math500/critical_by_quantile_{metric}.csv
bpa_results/diagnostics/sampling_analysis/math500/critical_by_quantile_{metric}.png
```

## 8. Evidence Consensus Routing

Route with the evidence-consensus rule. Reuse the best agreed SLM rollout instead of generating a fresh SLM step:

```bash
python -m bpa.eval.exp_disagreement_routing \
  --config configs/bpa_default.json \
  --dataset math500 \
  --max-problems 50 \
  --min-agreement-count 3 \
  --probe-k 4 \
  --probe-temperature 0.7 \
  --probe-max-tokens 32
```

With `--probe-k 4 --min-agreement-count 3`, a 3/4 or 4/4 signature majority stays on SLM and appends the rollout in that majority group with the highest `mean_logprob`; a 2/4 split routes the step to the LLM. The initial empty assistant prefix is still generated normally; this rule applies after a reasoning boundary exists.

Outputs:

```text
bpa_results/diagnostics/disagreement_routing/math500/summary.csv
bpa_results/diagnostics/disagreement_routing/math500/summary_metrics.json
bpa_results/diagnostics/disagreement_routing/math500/routing_boundaries.jsonl
```

Compare this with:

```text
bpa_results/math500/slm_only/summary.csv
bpa_results/math500/glimprouter_hinit/summary.csv
bpa_results/diagnostics/disagreement_routing/math500/summary.csv
```

## 9. Recommended First-Week Command Order

```bash
python -m compileall bpa tests
python -m unittest tests.test_bpa_core
python -m bpa.eval.render_sanity --config configs/bpa_default.json

for variant in slm_only llm_only glimprouter_hinit; do
  python -m bpa.eval.main_benchmark --config configs/bpa_default.json --variant "${variant}" --dataset math500 --max-problems 50
done

python -m bpa.eval.exp_sampling_disagreement --config configs/bpa_default.json --dataset math500 --max-problems 100 --probe-k 4 --probe-temperature 0.7 --probe-max-tokens 32
python -m bpa.eval.exp_llm_oracle --config configs/bpa_default.json --dataset math500 --max-problems 100
python -m bpa.eval.exp_boundary_continuation --config configs/bpa_default.json --dataset math500 --max-problems 100 --boundaries-per-problem 5 --continuation-max-tokens 2048
python -m bpa.eval.exp_disagreement_routing --config configs/bpa_default.json --dataset math500 --max-problems 100 --min-agreement-count 3 --probe-k 4 --probe-temperature 0.7 --probe-max-tokens 32
```
