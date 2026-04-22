#!/bin/bash
# Starts a vLLM server and captures logs; fill placeholders before running.
LOG_DIR="vllm_logs"  # or any directory you prefer

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

#################################################

CUDA_DEVICE=YOUR_CUDA_DEVICE_ID  # NOTE: change to your CUDA device id, e.g. 0/1/2/3
MODEL="YOUR_MODEL_NAME_OR_PATH"  # NOTE: change to your model path/name, e.g. org/model-name
API_KEY="YOUR_API_KEY"  # NOTE: change to the API key of your model (use a placeholder if auth is disabled)

LOG_FILE="${LOG_DIR}/CUDA${CUDA_DEVICE}_server_$(date +%Y%m%d_%H%M%S).out"
TEMPLATE_FILE="YOUR_TEMPLATE_FILE"  # NOTE: change to the template file of your model, e.g. "./template/your_model.jinja"
GPU_MEMORY_UTILIZATION=YOUR_GPU_MEMORY_UTILIZATION  # NOTE: change to the gpu memory utilization of your model, e.g. 0.60

PORT=YOUR_PORT  # NOTE: change to the port of your model, e.g. 11125

pid=$(lsof -ti tcp:$PORT)

if [ -n "$pid" ]; then
  # Free the port if another process is already listening.
  echo "Port $PORT is occupied, process PID: $pid"
  echo "Kill the process..."
  kill -9 $pid
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup vllm serve $MODEL \
  --dtype auto \
  --max-model-len 16384 \
  --chat-template $TEMPLATE_FILE \
  --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
  --api-key $API_KEY \
  --port $PORT \
  --host 0.0.0.0 \
  --enable-prefix-caching \
  > $LOG_FILE 2>&1 &
