# BPA v2.1 Experiments

This repository now contains two runnable paths:

- **BPA v2.1** in `bpa/`: uses the vLLM Python offline API (`vllm.LLM`) directly. This is the default path for the new experiments.
- **OpenAI-compatible vLLM server** via `server/serve.sh`: useful for legacy `src/glimp_router.py` or for future HTTP-backend experiments. The current `bpa/` runner does not need these HTTP servers.

## Environment

Install the experiment environment on the GPU host:

```bash
pip install -r requirements.txt
```

Edit `configs/bpa_default.json` before a real run:

- `slm_model_path`: local path or HF id for `DeepSeek-R1-Distill-Qwen-1.5B`
- `llm_model_path`: local path or HF id for `Qwen-32B`
- `dataset_paths.math500`: local MATH500 file
- `dataset_paths.aime24`: local AIME24 file
- `dataset_paths.aime25`: local `aime25.parquet`
- `dataset_paths.gpqa`: local GPQA-Diamond json/jsonl
- `slm_engine_kwargs` / `llm_engine_kwargs`: vLLM engine settings such as tensor parallel size

Default offline BPA runs load the SLM and LLM as two vLLM engines in the same Python process. Do not leave both engines at vLLM's default `gpu_memory_utilization=0.9`, because the first engine can reserve almost all available GPU memory before the second engine starts. The default config uses:

```json
"slm_engine_kwargs": {
  "gpu_memory_utilization": 0.2
},
"llm_engine_kwargs": {
  "gpu_memory_utilization": 0.7
}
```

For larger LLMs, also set tensor parallelism, for example:

```json
"llm_engine_kwargs": {
  "gpu_memory_utilization": 0.75,
  "tensor_parallel_size": 2
}
```

## Local Dataset Files

Dataset loading is fully local. The experiment runner does not call HuggingFace Hub or `datasets.load_dataset`.

Supported local formats:

- `.jsonl`: one JSON object per line
- `.json`: either a list of objects or an object containing `data`, `train`, `test`, `examples`, or `rows`
- `.csv` / `.tsv`: header row required
- `.parquet`: requires `pandas` and `pyarrow`

Expected columns:

- MATH500 / AIME24 / AIME25: `problem` plus one of `answer`, `solution`, or `target`
- GPQA-Diamond: `problem`/`question` plus either `A`-`D` choices and `answer`, or original GPQA-style `Correct Answer`, `Incorrect Answer 1`, `Incorrect Answer 2`, `Incorrect Answer 3`
- HumanEval: `prompt`; execution evaluation is intentionally not implemented in the first BPA pass

Example config fragment:

```json
"dataset_paths": {
  "math500": "/data/benchmarks/math500.jsonl",
  "aime24": "/data/benchmarks/aime24.jsonl",
  "aime25": "/data/benchmarks/aime25.parquet",
  "gpqa": "/data/benchmarks/gpqa_diamond.jsonl",
  "gpqa_diamond": "/data/benchmarks/gpqa_diamond.jsonl",
  "humaneval": "/data/benchmarks/HumanEval.jsonl"
}
```

## BPA v2.1 Offline Runs

First verify tokenizer rendering. This checks whether the selected chat template inserts thinking markers as expected and avoids double `<think>`:

```bash
python -m bpa.eval.render_sanity --config configs/bpa_default.json
```

Run the prompt-logprobs smoke test before full BPA arbitration:

```bash
python -m bpa.eval.exp_d5_prompt_logprobs_smoke \
  --config configs/bpa_default.json \
  --dataset math500 \
  --max-problems 5
```

By default D5 sweeps `prompt_logprobs_sweep = [1, 5, 20]`. Many vLLM builds cap `prompt_logprobs` at 20 unless the engine is started with a larger `max_logprobs`, so `50` is not enabled by default. To test `50`, set both:

```json
"prompt_logprobs_sweep": [1, 5, 20, 50],
"llm_engine_kwargs": {
  "gpu_memory_utilization": 0.7,
  "max_logprobs": 50
}
```

Run the first-week baselines on a small MATH500 slice:

```bash
for variant in slm_only llm_only glimprouter_hinit bpa_logging_only bpa_arbitration; do
  python -m bpa.eval.main_benchmark \
    --config configs/bpa_default.json \
    --variant "${variant}" \
    --dataset math500 \
    --max-problems 50
done
```

Run diagnostics:

```bash
python -m bpa.eval.exp_d0_first_token --config configs/bpa_default.json --dataset math500 --max-problems 50
python -m bpa.eval.exp_d1_cascade_funnel --config configs/bpa_default.json --dataset math500 --max-problems 200
```

Outputs are written under `bpa_results/` by default:

- `*.problem.json`
- `*.steps.jsonl`
- `*.branches.jsonl`
- `summary.csv`
- diagnostic files under `bpa_results/diagnostics/`

## GPU Memory Troubleshooting

If you see an error like:

```text
Free memory on device cuda:0 (...) is less than desired GPU memory utilization (0.9, ...)
```

it means vLLM is trying to reserve more memory than is currently free. In BPA offline runs this often happens during D5 when the SLM has already been loaded and the LLM starts afterward.

Fixes, in order:

1. Check for unrelated GPU users:

```bash
nvidia-smi
```

2. Lower per-engine reservations in `configs/bpa_default.json`:

```json
"slm_engine_kwargs": {
  "gpu_memory_utilization": 0.15,
  "max_model_len": 8192
},
"llm_engine_kwargs": {
  "gpu_memory_utilization": 0.75,
  "max_model_len": 8192
}
```

3. If the LLM is a 32B FP16/BF16 model, a single 24 GiB GPU is not enough. Use a quantized checkpoint, reduce the LLM size, or set `tensor_parallel_size` across enough GPUs.

4. Reduce `max_model_len` if you do not need a 16k+ context for the smoke test.

## vLLM Server Script Review

The redesigned `server/serve.sh` is reasonable and safer than the old script:

- It validates `MODEL` and `PYTHON_BIN` before starting.
- It uses `/health` and `/v1/models` checks instead of assuming startup succeeded.
- It avoids killing an occupied port unless `FORCE_RESTART=1` is explicitly set.
- It logs each run to a timestamped file under `server/vllm_logs`.
- It keeps optional flags (`--chat-template`, `--trust-remote-code`, `--enforce-eager`, `--enable-prefix-caching`) explicit.

Recommended adjustments:

- For BPA-style long-prefix workloads, use `ENABLE_PREFIX_CACHING=1` unless you are debugging prefix-cache behavior.
- Keep `ENFORCE_EAGER=0` for normal throughput; set it to `1` only for debugging or CUDA graph issues.
- `MAX_MODEL_LEN=65536` is aggressive. Use it only if the model and GPU memory support it; otherwise start with `16384` or `32768`.
- Set `TENSOR_PARALLEL_SIZE` if a 32B model cannot fit on one GPU.
- Set `API_KEY` or bind `HOST=127.0.0.1` if the server is reachable by other users; `HOST=0.0.0.0` without auth exposes the OpenAI-compatible endpoint on the network.
- `TRUST_REMOTE_CODE=1` may be required for some local model repos, but leave it off when the model does not need custom code.

## OpenAI-Compatible vLLM Server Commands

These commands start HTTP servers. They are not required by `bpa.eval.main_benchmark`, which uses offline vLLM. Replace model paths with your local paths.

Start the SLM server:

```bash
PYTHON_BIN=/home/lhyang/anaconda3/envs/glimp_router/bin/python \
MODEL=/path/to/DeepSeek-R1-Distill-Qwen-1.5B \
SERVED_MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B \
PORT=8000 \
CUDA_DEVICE=0 \
GPU_MEMORY_UTILIZATION=0.75 \
MAX_MODEL_LEN=32768 \
DTYPE=auto \
CHAT_TEMPLATE="$(pwd)/server/template/deepseekr1.jinja" \
TRUST_REMOTE_CODE=1 \
ENABLE_PREFIX_CACHING=1 \
bash server/serve.sh
```

Start the LLM server:

```bash
PYTHON_BIN=/home/lhyang/anaconda3/envs/glimp_router/bin/python \
MODEL=/path/to/Qwen-32B \
SERVED_MODEL_NAME=Qwen-32B \
PORT=8001 \
CUDA_DEVICE=1 \
GPU_MEMORY_UTILIZATION=0.85 \
MAX_MODEL_LEN=32768 \
TENSOR_PARALLEL_SIZE=2 \
DTYPE=auto \
CHAT_TEMPLATE="$(pwd)/server/template/qwen3.jinja" \
TRUST_REMOTE_CODE=1 \
ENABLE_PREFIX_CACHING=1 \
bash server/serve.sh
```

If a port is occupied by an unhealthy process and you intentionally want to restart it:

```bash
FORCE_RESTART=1 PORT=8000 MODEL=/path/to/model bash server/serve.sh
```

Check server status:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models
```
