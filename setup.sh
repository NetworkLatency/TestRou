#!/bin/bash

set -e

# # 1. Create conda environment (replace with your preferred env name)
# conda create -n YOUR_CONDA_ENV_NAME python=3.12 -y

# # 2. Initialize conda environment and activate it
# conda activate YOUR_CONDA_ENV_NAME

# # 3. Install dependencies
# pip install -r requirements.txt

# 4. Download datasets in ./data (optional; update URLs if needed)
mkdir -p ./data/lcbv5
mkdir -p ./data/lcbv6

cd ./data/lcbv5
wget https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test5.jsonl

cd ../lcbv6
wget https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl

echo "Environment has been configured!"
