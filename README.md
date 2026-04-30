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
  "dataset_paths": {
    "math500": "data/math500.jsonl",
    "aime24": "data/aime24.jsonl"
  }
}
```

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

Main disagreement metrics:

- `operation_vote_disagreement`
- `number_vote_disagreement`
- `self_bleu_disagreement`
- `char_jaccard_disagreement`
- `structured_disagreement`

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

Generate distribution plots, quantile plots, dip-test results, and AUROC:

```bash
python -m bpa.eval.analyze_sampling_disagreement \
  --config configs/bpa_default.json \
  --dataset math500 \
  --metrics operation_vote_disagreement number_vote_disagreement self_bleu_disagreement char_jaccard_disagreement structured_disagreement \
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

Continue this direction only if:

```text
dip test p < 0.05 or the distribution is visually bimodal
AUROC(disagreement -> critical) > 0.65
P(critical | disagreement quantile) is mostly monotonic increasing
```

## 8. Top-20% Routing Sanity Check

If the phenomenon holds, run a simple top-20% disagreement routing check:

```bash
python -m bpa.eval.exp_disagreement_routing \
  --config configs/bpa_default.json \
  --dataset math500 \
  --max-problems 50 \
  --metric number_vote_disagreement \
  --threshold-quantile 0.8 \
  --probe-k 4 \
  --probe-temperature 0.7 \
  --probe-max-tokens 32
```

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
python -m bpa.eval.analyze_sampling_disagreement --config configs/bpa_default.json --dataset math500 --metrics operation_vote_disagreement number_vote_disagreement self_bleu_disagreement char_jaccard_disagreement structured_disagreement --num-bins 10
```
