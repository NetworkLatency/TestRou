#!/bin/bash
# Example runner; edit the parameters below for your experiment.
#################################################

CONFIG_PATH="config.json"
DATASET_NAME="lcbv5"
REPEAT_NUM=6
SCORE_METHOD="first_token_entropy"
ROUTING_MODE="fa_skip"
TOKEN_BUDGET=8192
MODEL_SIZE="32b"
SMALL_MODEL_SIZE="4b"
SCORE_TRESHOLD=0.8
FORMAT_TOKENS_PATH="./format_tokens.json"
MAX_FORMAT_SKIP=3
TOP_LOGPROBS=50
GENERATE_DASHBOARD=true
OUTPUT_DIR="./large${MODEL_SIZE}_small${SMALL_MODEL_SIZE}_${SCORE_METHOD}_${ROUTING_MODE}_${SCORE_TRESHOLD}_results"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/large${MODEL_SIZE}_small${SMALL_MODEL_SIZE}_${SCORE_METHOD}_${ROUTING_MODE}_${SCORE_TRESHOLD}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).out"

echo "Logging ${LOG_FILE}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Write the runtime config consumed by main.py.
cat <<EOF > "${CONFIG_PATH}"
{
  "dataset_name": "${DATASET_NAME}",
  "repeat_num": ${REPEAT_NUM},
  "score_method": "${SCORE_METHOD}",
  "routing_mode": "${ROUTING_MODE}",
  "token_budget": ${TOKEN_BUDGET},
  "output_dir": "${OUTPUT_DIR}",
  "model_size": "${MODEL_SIZE}",
  "small_model_size": "${SMALL_MODEL_SIZE}",
  "score_threshold": ${SCORE_TRESHOLD},
  "format_tokens_path": "${FORMAT_TOKENS_PATH}",
  "max_format_skip": ${MAX_FORMAT_SKIP},
  "top_logprobs": ${TOP_LOGPROBS},
  "generate_dashboard": ${GENERATE_DASHBOARD},
  "dashboard_path": "${OUTPUT_DIR}/fa_dashboard.png"
}
EOF

nohup python main.py > ${LOG_FILE} 2>&1 
