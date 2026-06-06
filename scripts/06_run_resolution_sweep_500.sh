#!/usr/bin/env bash
# =============================================================================
# Resolution Sweep — 500 data points per (benchmark × resolution)
# 4-GPU 병렬: GPU0(docvqa 448,896) GPU1(docvqa 1344,1792)
#             GPU2(chartqa 448,896) GPU3(chartqa 1344,1792)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/output/qwen25vl_baseline/manifests/benchmark_manifest_500_docker.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_sweep_500}"
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-baseline:latest}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
WARMUP="${WARMUP:-2}"
MEASURED_ITERS="${MEASURED_ITERS:-500}"

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_DIR="${OUTPUT_ROOT}/logs"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "${LOG_DIR}/sweep.log"; }

log "=== Resolution Sweep (500 data points/config) ==="
log "Model   : ${MODEL_PATH}"
log "Manifest: ${MANIFEST}"
log "Output  : ${OUTPUT_ROOT}"
log "Iters   : warmup=${WARMUP}  measured=${MEASURED_ITERS}"

run_config() {
  local gpu_id="$1"
  local benchmark="$2"
  local resolution="$3"
  local job="${benchmark}_res${resolution}"
  local log_file="${LOG_DIR}/gpu${gpu_id}_${job}.log"
  local out_dir="${OUTPUT_ROOT}/${job}"

  # host → container path 변환
  local c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
  local c_manifest="${MANIFEST/${PROJECT_ROOT}/\/workspace}"
  local c_out="${out_dir/${PROJECT_ROOT}/\/workspace}"

  mkdir -p "${out_dir}"
  log "  [GPU${gpu_id}] ${job} → ${out_dir}"

  docker run --rm \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --ipc=host \
    --ulimit memlock=-1 \
    -v "${PROJECT_ROOT}:/workspace" \
    -e "CUDA_VISIBLE_DEVICES=${gpu_id}" \
    -e "PYTHONUNBUFFERED=1" \
    "${IMAGE_NAME}" \
    -c "python3 -u /workspace/tools/profile_qwen25vl_baseline.py \
      --mode        app_timing \
      --model-path  ${c_model} \
      --manifest    ${c_manifest} \
      --benchmark   ${benchmark} \
      --num-samples 500 \
      --resolution  ${resolution} \
      --max-new-tokens 64 \
      --batch-size  1 \
      --precision   ${PRECISION} \
      --warmup      ${WARMUP} \
      --measured-iters ${MEASURED_ITERS} \
      --seed        ${SEED} \
      --output-dir  ${c_out}" \
    2>&1 | tee "${log_file}"
}

# ── GPU 0: docvqa 448, 896 (순차) ─────────────────────────────────────────────
( run_config 0 docvqa 448 && run_config 0 docvqa 896 ) &
PID0=$!

# ── GPU 1: docvqa 1344, 1792 (순차) ───────────────────────────────────────────
( run_config 1 docvqa 1344 && run_config 1 docvqa 1792 ) &
PID1=$!

# ── GPU 2: chartqa 448, 896 (순차) ────────────────────────────────────────────
( run_config 2 chartqa 448 && run_config 2 chartqa 896 ) &
PID2=$!

# ── GPU 3: chartqa 1344, 1792 (순차) ──────────────────────────────────────────
( run_config 3 chartqa 1344 && run_config 3 chartqa 1792 ) &
PID3=$!

log "4 GPU groups launched. Waiting..."
FAILED=0
wait $PID0; RC=$?; log "  GPU0 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID1; RC=$?; log "  GPU1 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID2; RC=$?; log "  GPU2 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID3; RC=$?; log "  GPU3 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1

# ── Collect & export CSV ──────────────────────────────────────────────────────
log "Collecting all timing_raw.jsonl → CSV..."
python3 - "${OUTPUT_ROOT}" "${PROJECT_ROOT}/results" << 'PYEOF'
import json, csv, re, sys
from pathlib import Path

out_root  = Path(sys.argv[1])
csv_out   = Path(sys.argv[2])
csv_out.mkdir(exist_ok=True)

FIELDS = [
    'benchmark','resolution','max_new_tokens','iteration_idx','run_id','sample_id',
    'num_input_text_tokens',
    'num_visual_tokens_before_merger','num_visual_tokens_after_merger',
    'actual_generated_tokens',
    'image_preprocess_ms','processor_ms',
    'visual_encoder_ms','visual_merger_ms','patch_embed_ms',
    'window_attn_total_ms','full_attn_total_ms',
    'llm_prefill_ms','ttft_ms','decode_loop_ms','tpot_ms','e2e_ms',
    'request_total_ms','request_output_tok_s','decode_only_tok_s',
    'f_vision','f_visual_prefill','f_decode',
]

all_rows = []
for d in sorted(out_root.iterdir()):
    if not d.is_dir(): continue
    raw = d / 'timing_raw.jsonl'
    if not raw.exists(): continue
    m = re.match(r'(\w+)_res(\d+)', d.name)
    if not m: continue
    bm, res = m.group(1), int(m.group(2))
    records = [json.loads(l) for l in raw.read_text().strip().split('\n') if l.strip()]
    for i, r in enumerate(records):
        row = {'benchmark': bm, 'resolution': res,
               'max_new_tokens': 64, 'iteration_idx': i}
        for f in FIELDS[4:]: row[f] = r.get(f, '')
        all_rows.append(row)

# Count per config
from collections import Counter
counts = Counter((r['benchmark'], r['resolution']) for r in all_rows)
print(f"\n  총 {len(all_rows)} rows")
for (bm, res), n in sorted(counts.items()):
    print(f"    {bm} res={res}: {n} data points")

out_path = csv_out / 'fig8a_resolution_scaling_500dp.csv'
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader(); w.writerows(all_rows)
print(f"\n  Written: {out_path}")
PYEOF

if [ "${FAILED}" -eq 0 ]; then
  log "=== Sweep COMPLETE ==="
  log "CSV: ${PROJECT_ROOT}/results/fig8a_resolution_scaling_500dp.csv"
else
  log "=== Sweep done with failures ==="
fi
exit "${FAILED}"
