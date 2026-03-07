#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <qwen|deepseek_r1|deepseek_llm> [start] [end]"
  exit 1
fi

MODEL="$1"
START="${2:-1}"
END="${3:-100}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/model_runs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${MODEL}_${TS}.log"
PID_FILE="${LOG_DIR}/${MODEL}_${TS}.pid"

cd "${PROJECT_ROOT}"

nohup python scripts/run_model_comparison_batch.py \
  --model "${MODEL}" \
  --start "${START}" \
  --end "${END}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started model run:"
echo "  model: ${MODEL}"
echo "  pid:   ${PID}"
echo "  log:   ${LOG_FILE}"
echo "  pidf:  ${PID_FILE}"
echo
echo "Monitor with:"
echo "  tail -f ${LOG_FILE}"
