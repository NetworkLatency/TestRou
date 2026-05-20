#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/server/vllm_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/lhyang/anaconda3/envs/glimp_router/bin/python}"
MODEL="${MODEL:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
PORT="${PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20000}"
START_TIMEOUT="${START_TIMEOUT:-180}"
HOST="${HOST:-0.0.0.0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
API_KEY="${API_KEY:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
DTYPE="${DTYPE:-}"

mkdir -p "${LOG_DIR}"

if [[ -z "${MODEL}" ]]; then
  echo "MODEL is required" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -e "${MODEL}" ]]; then
  echo "MODEL path does not exist: ${MODEL}" >&2
  exit 1
fi

if [[ -z "${SERVED_MODEL_NAME}" ]]; then
  SERVED_MODEL_NAME="$(basename "${MODEL}")"
fi

HEALTH_URL="http://127.0.0.1:${PORT}/health"
MODELS_URL="http://127.0.0.1:${PORT}/v1/models"

check_healthy() {
  curl -fsS -m 5 "${HEALTH_URL}" >/dev/null 2>&1
}

existing_pid="$(lsof -ti tcp:${PORT} || true)"
if check_healthy; then
  echo "vLLM is already healthy on port ${PORT}"
  echo "health=${HEALTH_URL}"
  echo "models=${MODELS_URL}"
  exit 0
fi

if [[ -n "${existing_pid}" ]]; then
  if [[ "${FORCE_RESTART:-0}" != "1" ]]; then
    echo "Port ${PORT} is occupied by PID ${existing_pid}, but the health check failed." >&2
    echo "Set FORCE_RESTART=1 to kill that process and restart vLLM." >&2
    exit 1
  fi
  kill "${existing_pid}"
  sleep 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/vllm_${SERVED_MODEL_NAME}_cuda${CUDA_DEVICE}_port${PORT}_${timestamp}.log"

cmd=(
  "${PYTHON_BIN}" -u -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --port "${PORT}"
  --host "${HOST}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
)

if [[ -n "${API_KEY}" ]]; then
  cmd+=(--api-key "${API_KEY}")
fi

if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  cmd+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
fi

if [[ -n "${DTYPE}" ]]; then
  cmd+=(--dtype "${DTYPE}")
fi

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  cmd+=(--chat-template "${CHAT_TEMPLATE}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  cmd+=(--enable-prefix-caching)
fi

echo "Starting vLLM"
echo "model=${MODEL}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "cuda_device=${CUDA_DEVICE}"
echo "port=${PORT}"
echo "log=${log_file}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" nohup "${cmd[@]}" >"${log_file}" 2>&1 &
server_pid=$!

for ((i = 1; i <= START_TIMEOUT; i++)); do
  if check_healthy; then
    echo "vLLM started successfully on port ${PORT}"
    echo "pid=${server_pid}"
    echo "health=${HEALTH_URL}"
    echo "models=${MODELS_URL}"
    exit 0
  fi

  if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
    echo "vLLM exited before becoming healthy. Last log lines:" >&2
    tail -n 50 "${log_file}" >&2 || true
    exit 1
  fi

  sleep 1
done

echo "Timed out after ${START_TIMEOUT}s waiting for vLLM health check." >&2
tail -n 50 "${log_file}" >&2 || true
kill "${server_pid}" >/dev/null 2>&1 || true
exit 1
