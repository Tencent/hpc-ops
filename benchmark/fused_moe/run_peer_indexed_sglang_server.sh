#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-hpc}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the GLM-5.2-FP8 checkpoint}
SGLANG_ROOT=${SGLANG_ROOT:?set SGLANG_ROOT to the peer-indexed SGLang checkout}
PYTHON_BIN=${PYTHON_BIN:-python3}
PORT=${PORT:-30080}
SEED=${SEED:-4232}

HPCOPS_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${HPCOPS_ROOT}:${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

case ${MODE} in
  deepep)
    backend_args=(
      --moe-a2a-backend deepep
      --moe-runner-backend deep_gemm
      --deepep-mode auto
    )
    ;;
  megamoe)
    backend_args=(--moe-a2a-backend megamoe --moe-runner-backend auto)
    ;;
  hpc)
    backend_args=(--moe-a2a-backend hpc_ops --moe-runner-backend hpc_ops)
    ;;
  *)
    echo "MODE must be deepep, megamoe, or hpc" >&2
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.80 \
  --max-running-requests 256 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs \
  --schedule-conservativeness 0.3 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --attention-backend dsa \
  --dsa-prefill-backend flashmla_sparse \
  --dsa-decode-backend fa3 \
  --cuda-graph-max-bs-decode 32 \
  --cuda-graph-bs-decode 1 2 4 8 12 16 20 24 28 32 \
  --disable-prefill-cuda-graph \
  --random-seed "${SEED}" \
  --skip-server-warmup \
  --host 0.0.0.0 \
  --port "${PORT}" \
  "${backend_args[@]}"
