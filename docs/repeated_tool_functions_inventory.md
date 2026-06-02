# Repeated Tool Function Inventory

This note preserves the duplicated helper patterns found during cleanup. They are not required to run SARR, but may be useful if the analysis scripts are later consolidated.

## Common File Readers

Repeated helpers:

- `_read_jsonl`
- `_steps_path`
- `_problem_sort_key`
- `_read_summary`
- `write_csv`

Current locations:

- `scripts/analyze_probability_periodicity.py`
- `scripts/analyze_reflection_logits.py`
- `scripts/analyze_rpdi_entropy.py`
- `scripts/analyze_sarr_offline_signals.py`
- `scripts/visualize_sarr_entropy.py`

Potential consolidation target:

- `scripts/analysis_utils.py`

## Value Coercion

Repeated helpers:

- `_truthy`
- `_float`
- `_mean`
- `_pstdev`

Current locations:

- `scripts/analyze_probability_periodicity.py`
- `scripts/analyze_reflection_logits.py`
- `scripts/analyze_rpdi_entropy.py`
- `scripts/analyze_sarr_offline_signals.py`
- `scripts/visualize_sarr_entropy.py`
- `scripts/run_sarr_code.py`

Potential consolidation target:

- `scripts/analysis_utils.py` for analysis scripts
- Keep `scripts/run_sarr_code.py` local unless the runner grows shared dependencies.

## Plot Helpers

Repeated helpers:

- `plot_progress`
- `_margin`

Current locations:

- `scripts/analyze_reflection_logits.py`
- `scripts/analyze_rpdi_entropy.py`

Potential consolidation target:

- Keep local if the plots continue to diverge.
- Extract only if future changes require shared plotting behavior.

## Answer And Evaluation Helpers

Repeated patterns:

- math answer extraction
- choice-letter extraction
- CSV result writing

Current canonical SARR location:

- `sarr_code/safety.py`
- `sarr_code/eval/benchmark_eval.py`

Cleanup note:

- The old `src/` GlimpRouter/FA-routing experiment scripts had their own evaluator copies and have been removed from the active project tree.
