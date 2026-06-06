#!/usr/bin/env bash
# =============================================================================
# Qwen2.5-VL Evaluation Pipeline
# Benchmarks: DocVQA (ANLS), ChartQA (Relaxed Acc), OCRBench v2 (EM)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_eval}"
TOOLS_DIR="${PROJECT_ROOT}/tools"

PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"

# --- Evaluation config ---
# 0 = full dataset; set to smaller number for quick test
NUM_SAMPLES="${NUM_SAMPLES:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BENCHMARKS="${BENCHMARKS:-docvqa chartqa ocrbench_v2}"

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_FILE="${OUTPUT_ROOT}/logs/eval_progress.log"
ts_human() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts_human)] $1" | tee -a "${LOG_FILE}"; }

log "=== Qwen2.5-VL Evaluation Pipeline ==="
log "Model  : ${MODEL_PATH}"
log "Data   : ${DATA_ROOT}"
log "Output : ${OUTPUT_ROOT}"
log "Benchmarks: ${BENCHMARKS}"
log "Samples   : ${NUM_SAMPLES} per benchmark"
log "Max tokens: ${MAX_NEW_TOKENS}"

# Convert space-separated benchmarks to --benchmarks args
BM_ARGS=""
for bm in ${BENCHMARKS}; do BM_ARGS="${BM_ARGS} ${bm}"; done

python3 -u "${TOOLS_DIR}/eval_qwen25vl.py" \
  --model-path   "${MODEL_PATH}" \
  --data-root    "${DATA_ROOT}" \
  --output-dir   "${OUTPUT_ROOT}" \
  --benchmarks   ${BM_ARGS} \
  --num-samples  "${NUM_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --precision    "${PRECISION}" \
  --seed         "${SEED}" \
  2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ "${EXIT_CODE}" -eq 0 ]; then
  log "=== Evaluation COMPLETE ==="
  log "Results: ${OUTPUT_ROOT}/eval_summary.json"
  if [ -f "${OUTPUT_ROOT}/eval_summary.json" ]; then
    log "--- Summary ---"
    python3 -c "
import json
d = json.load(open('${OUTPUT_ROOT}/eval_summary.json'))
print()
for bm, m in d.get('results',{}).items():
    print(f'  {bm:<15}: {m}')
print()
print(f'  Samples per benchmark: {d.get(\"num_samples\")}')
print(f'  Max new tokens       : {d.get(\"max_new_tokens\")}')
" 2>/dev/null || true
  fi
else
  log "=== Evaluation FAILED (exit ${EXIT_CODE}) ==="
fi

exit "${EXIT_CODE}"
