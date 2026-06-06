#!/usr/bin/env bash
# =============================================================================
# Phase 2: FA2 vs GUIDE — 500 samples × 4 resolutions × 2 benchmarks
#
# GPU 배분 전략 (FA2 & GUIDE 동시 병렬 실행):
#   GPU 0 → FA2  : docvqa  (res 448→896→1344→1792 순차)
#   GPU 1 → FA2  : chartqa (res 448→896→1344→1792 순차)
#   GPU 2 → GUIDE: docvqa  (res 448→896→1344→1792 순차)
#   GPU 3 → GUIDE: chartqa (res 448→896→1344→1792 순차)
#
# 이미지를 plan의 4개 고정 해상도로 강제 리사이즈 → merged grid 항상 divisible
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-qwen25vl-guide:latest}"    # FA2+GUIDE 모두 이 이미지 사용
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/output/qwen25vl_baseline/manifests/benchmark_manifest_500_docker.jsonl}"
FA2_ROOT="${FA2_ROOT:-${PROJECT_ROOT}/output/qwen25vl_fa2_500}"
GUIDE_ROOT="${GUIDE_ROOT:-${PROJECT_ROOT}/output/qwen25vl_guide_500}"
NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2"
PRECISION="${PRECISION:-bf16}"
N_SAMPLES="${N_SAMPLES:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
WARMUP="${WARMUP:-2}"
SEED="${SEED:-42}"

c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
c_manifest="${MANIFEST/${PROJECT_ROOT}/\/workspace}"
c_fa2="${FA2_ROOT/${PROJECT_ROOT}/\/workspace}"
c_guide="${GUIDE_ROOT/${PROJECT_ROOT}/\/workspace}"

# ── 로깅 ─────────────────────────────────────────────────────────────────────
mkdir -p "${FA2_ROOT}/logs" "${GUIDE_ROOT}/logs"
FA2_LOG="${FA2_ROOT}/logs/progress.log"
GUIDE_LOG="${GUIDE_ROOT}/logs/progress.log"
ts() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "${FA2_ROOT}/logs/master.log" "${GUIDE_ROOT}/logs/master.log"; }

log "================================================================"
log " Phase 2: FA2 vs GUIDE  500-Sample Comparative Run"
log " Image   : ${IMAGE}"
log " Manifest: ${MANIFEST}"
log " N       : ${N_SAMPLES} samples per config"
log " Res     : 448 / 896 / 1344 / 1792"
log " Tok     : ${MAX_NEW_TOKENS}"
log "================================================================"

# ── Docker 실행 함수 ──────────────────────────────────────────────────────────
run_one() {
  local gpu="$1" attn="$2" bm="$3" res="$4"
  local out_root; out_root=$([ "$attn" = "fa2" ] && echo "${FA2_ROOT}" || echo "${GUIDE_ROOT}")
  local out_dir="${out_root}/${bm}_res${res}_tok${MAX_NEW_TOKENS}"
  local c_out="${out_dir/${PROJECT_ROOT}/\/workspace}"
  local log_file="${out_dir}/docker.log"
  mkdir -p "${out_dir}"

  local mounts=("-v" "${PROJECT_ROOT}:/workspace")
  [ -d "${NSYS_HOST_DIR}" ] && mounts+=("-v" "${NSYS_HOST_DIR}:/opt/nsight-systems:ro")

  docker run --rm --gpus "device=${gpu}" --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" "${mounts[@]}" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
    -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton -e XDG_CACHE_HOME=/tmp/.cache \
    -e PATH="/opt/nsight-systems/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    -w /workspace "${IMAGE}" -c "
python3 -u /workspace/tools/profile_qwen25vl_baseline.py \
  --mode          app_timing \
  --attn-impl     ${attn} \
  --model-path    '${c_model}' \
  --manifest      '${c_manifest}' \
  --benchmark     ${bm} \
  --num-samples   ${N_SAMPLES} \
  --resolution    ${res} \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --batch-size    1 \
  --precision     ${PRECISION} \
  --warmup        ${WARMUP} \
  --measured-iters ${N_SAMPLES} \
  --seed          ${SEED} \
  --output-dir    '${c_out}'
" 2>&1 | tee "${log_file}"
}

# ── 4 GPU 병렬 실행 ────────────────────────────────────────────────────────────
log "Launching 4 GPU workers..."

# GPU 0: FA2 docvqa 4 resolutions
( for res in 448 896 1344 1792; do
    log "[gpu0] FA2 docvqa res=${res} START"
    run_one 0 fa2 docvqa ${res} && log "[gpu0] FA2 docvqa res=${res} DONE" \
                                || log "[gpu0] FA2 docvqa res=${res} FAIL"
  done ) &
PID0=$!

# GPU 1: FA2 chartqa 4 resolutions
( for res in 448 896 1344 1792; do
    log "[gpu1] FA2 chartqa res=${res} START"
    run_one 1 fa2 chartqa ${res} && log "[gpu1] FA2 chartqa res=${res} DONE" \
                                  || log "[gpu1] FA2 chartqa res=${res} FAIL"
  done ) &
PID1=$!

# GPU 2: GUIDE docvqa 4 resolutions
( for res in 448 896 1344 1792; do
    log "[gpu2] GUIDE docvqa res=${res} START"
    run_one 2 guide docvqa ${res} && log "[gpu2] GUIDE docvqa res=${res} DONE" \
                                   || log "[gpu2] GUIDE docvqa res=${res} FAIL"
  done ) &
PID2=$!

# GPU 3: GUIDE chartqa 4 resolutions
( for res in 448 896 1344 1792; do
    log "[gpu3] GUIDE chartqa res=${res} START"
    run_one 3 guide chartqa ${res} && log "[gpu3] GUIDE chartqa res=${res} DONE" \
                                    || log "[gpu3] GUIDE chartqa res=${res} FAIL"
  done ) &
PID3=$!

log "Waiting for all 4 workers (PIDs: $PID0 $PID1 $PID2 $PID3)..."
wait $PID0; E0=$?
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?

log "All workers done: exits=$E0 $E1 $E2 $E3"

# ── 결과 병합 ──────────────────────────────────────────────────────────────────
log "Merging results..."
python3 - "${FA2_ROOT}" "${GUIDE_ROOT}" "${PROJECT_ROOT}/results" "${MAX_NEW_TOKENS}" <<'PY'
import sys, json, csv, os
from pathlib import Path

fa2_root  = Path(sys.argv[1])
guide_root= Path(sys.argv[2])
res_dir   = Path(sys.argv[3]); res_dir.mkdir(parents=True, exist_ok=True)
tok       = sys.argv[4]

benchmarks = ["docvqa", "chartqa"]
resolutions = [448, 896, 1344, 1792]

all_rows = []
for attn, root in [("fa2", fa2_root), ("guide", guide_root)]:
    for bm in benchmarks:
        for res in resolutions:
            p = root / f"{bm}_res{res}_tok{tok}" / "timing_raw.jsonl"
            if not p.exists():
                print(f"  MISSING: {p}")
                continue
            rows = [json.loads(l) for l in open(p)]
            # ensure attn_impl/benchmark/resolution columns
            for r in rows:
                r.setdefault("attn_impl", attn)
                r.setdefault("benchmark", bm)
                r.setdefault("resolution", res)
                r.setdefault("max_new_tokens", int(tok))
            all_rows += rows
            print(f"  {attn} {bm} res={res}: {len(rows)} rows")

if not all_rows:
    print("No rows collected."); sys.exit(1)

# write merged CSV
out_csv = res_dir / f"fa2_vs_guide_500dp_tok{tok}.csv"
keys = list(all_rows[0].keys())
# ensure key columns are first
priority = ["attn_impl","benchmark","resolution","max_new_tokens",
            "sample_id","run_id","iteration_idx",
            "visual_encoder_ms","ttft_ms","tpot_ms","e2e_ms",
            "window_attn_total_ms","full_attn_total_ms","f_vision"]
keys = [k for k in priority if k in keys] + [k for k in keys if k not in priority]

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(all_rows)

print(f"\nMerged: {len(all_rows)} rows -> {out_csv}")
PY

log "================================================================"
log " Phase 2 COMPLETE"
log " FA2 results : ${FA2_ROOT}"
log " GUIDE results: ${GUIDE_ROOT}"
log " Merged CSV  : ${PROJECT_ROOT}/results/fa2_vs_guide_500dp_tok${MAX_NEW_TOKENS}.csv"
log "================================================================"
