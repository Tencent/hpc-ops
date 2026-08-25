#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-hpc}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the GLM-5.2-FP8 checkpoint}
SGLANG_ROOT=${SGLANG_ROOT:?set SGLANG_ROOT to the peer-indexed SGLang checkout}
PYTHON_BIN=${PYTHON_BIN:-python3}
PORT=${PORT:-30080}
OUTPUT_ROOT=${OUTPUT_ROOT:-peer-indexed-serving-results}
BATCHES=${BATCHES:-"8 32"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-4232}
FORCED_TOKEN_ID=${FORCED_TOKEN_ID:-1000}
EXPECTED_TOKEN_TEXT=${EXPECTED_TOKEN_TEXT:-atus}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HPCOPS_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
VALIDATOR=${SCRIPT_DIR}/validate_peer_indexed_output.py
export PYTHONPATH="${HPCOPS_ROOT}:${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

mode_dir=${OUTPUT_ROOT}/${MODE}
mkdir -p "${mode_dir}"
curl --fail --silent --show-error "http://127.0.0.1:${PORT}/health" >/dev/null

run_case() {
  local batch=$1
  local output_len=$2
  local tag=$3
  local validate=${4:-0}
  local body
  body=$(printf '{"sampling_params":{"temperature":0.0,"max_new_tokens":%d,"ignore_eos":true,"logit_bias":{"%d":100.0}}}' "${output_len}" "${FORCED_TOKEN_ID}")

  "${PYTHON_BIN}" -m sglang.benchmark.serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dataset-name random-ids \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --random-input-len 8192 \
    --random-output-len "${output_len}" \
    --random-range-ratio 1 \
    --request-rate inf \
    --num-prompts "${batch}" \
    --max-concurrency "${batch}" \
    --seed "${SEED}" \
    --tokenize-prompt \
    --warmup-requests 0 \
    --flush-cache \
    --cache-report \
    --output-details \
    --disable-tqdm \
    --extra-request-body "${body}" \
    --output-file "${mode_dir}/${tag}.jsonl" \
    >"${mode_dir}/${tag}.stdout.log" 2>&1

  if [[ ${validate} == 1 ]]; then
    "${PYTHON_BIN}" "${VALIDATOR}" \
      --file "${mode_dir}/${tag}.jsonl" \
      --batch "${batch}" \
      --output-len "${output_len}" \
      --expected-token-text "${EXPECTED_TOKEN_TEXT}" \
      --expected-token-id "${FORCED_TOKEN_ID}" \
      >>"${mode_dir}/${tag}.stdout.log" 2>&1
  fi
}

for batch in ${BATCHES}; do
  run_case "${batch}" 64 "${MODE}_b${batch}_warmup"
  for repeat in $(seq 1 "${REPEATS}"); do
    run_case "${batch}" 1000 "${MODE}_b${batch}_r${repeat}" 1
  done
done
