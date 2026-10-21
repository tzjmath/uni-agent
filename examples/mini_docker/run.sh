#!/usr/bin/env bash
# Smoke-test the RL rollout path on one GPU and keep the trajectory on disk.
# Run from the uni-agent repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PY="${PY:-/home/tianzhengjie/venvs/uni-agent/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/tianzhengjie/models/Qwen2.5-Coder-7B-Instruct}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
DATA_PATH="${DATA_PATH:-${HOME}/data/uni_agent/mini_docker.parquet}"
LOG_DIR="${LOG_DIR:-${HOME}/data/uni_agent/logs/mini_docker}"
# One GPU is enough for this 27B language-model-only smoke run.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="/home/tianzhengjie/venvs/uni-agent/bin:${PATH}"
# verl's Python package lives in verl/verl. The repository root must come after
# that path, otherwise `import verl` binds the git checkout instead of the package.
export PYTHONPATH="${ROOT}/verl:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PY}" examples/mini_docker/make_sample.py --output "${DATA_PATH}"

"${PY}" examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --task-config examples/mini_docker/task_config.yaml \
    --engine vllm \
    --tool-parser "${TOOL_PARSER}" \
    --tensor-parallel-size 1 \
    --n-gpus-per-node 1 \
    --gateway-count 1 \
    --concurrency 1 \
    --n 1 \
    --limit 1 \
    --temperature 0 \
    --top-p 1 \
    --response-length 8192 \
    --gpu-memory-utilization 0.85 \
    --allowed-request-sampling-param-keys temperature top_p top_k \
    --log-dir "${LOG_DIR}" \
    --result-path "${LOG_DIR}/result.json"
