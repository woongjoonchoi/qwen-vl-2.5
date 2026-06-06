#!/usr/bin/env bash
# =============================================================================
# Phase 3: Nsys Profiling, Segment Comparison & Scientific Reporting (Stage 11~13)
# All profiling runs INSIDE Docker (qwen25vl-guide:latest) with nsys bind-mounted.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_guide_kernel_level}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
WARMUP="${WARMUP:-10}"
PROFILE_ITERS="${PROFILE_ITERS:-50}"
RESIZE_PX="${RESIZE_PX:-896}"          # 64x64 patches -> 32x32 merged -> divisible
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-guide:latest}"

NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
PROGRESS_LOG="${LOG_DIR}/progress.log"
STATUS_JSONL="${LOG_DIR}/status.jsonl"
CURRENT_STAGE="${LOG_DIR}/current_stage.txt"
touch "${PROGRESS_LOG}" "${STATUS_JSONL}"

ts_human() { date +"%Y-%m-%d %H:%M:%S"; }
log_progress() { echo "[$(ts_human)] [$1] $2" | tee -a "${PROGRESS_LOG}"; }
write_status() {
  python3 - "${STATUS_JSONL}" "$1" "$2" "${3:-}" <<'PY'
import json, sys, datetime
path, stage, status, message = sys.argv[1:5]
with open(path,"a") as f:
    f.write(json.dumps({"time":datetime.datetime.now().isoformat(timespec="seconds"),
                        "stage":stage,"status":status,"message":message})+"\n")
PY
}

run_in_docker() {
  local cmd="$1"
  local mounts=( "-v" "${PROJECT_ROOT}:/workspace" )
  [ -d "${NSYS_HOST_DIR}" ] && mounts+=("-v" "${NSYS_HOST_DIR}:/opt/nsight-systems:ro")
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" \
    "${mounts[@]}" \
    -e "CUDA_VISIBLE_DEVICES=0" -e "PYTHONUNBUFFERED=1" \
    -e "HOME=/tmp" -e "TRITON_CACHE_DIR=/tmp/.triton" -e "XDG_CACHE_HOME=/tmp/.cache" \
    -e "PATH=/opt/nsight-systems/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    -w /workspace "${IMAGE_NAME}" -c "${cmd}"
}

c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
c_out="${OUTPUT_ROOT/${PROJECT_ROOT}/\/workspace}"

# ── Stage 11: block-size sweep ───────────────────────────────────────────────
stage11_block_sweep() {
  local stage="stage11_block_sweep"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/nsys_block_sweep/stage11_block_sweep"
  mkdir -p "${out}"
  local c_sout="${out/${PROJECT_ROOT}/\/workspace}"
  local base="qwen25vl-guide-kernel-block-sweep-${PRECISION}"
  log_progress "START" "Stage 11: nsys block-size sweep"
  write_status "${stage}" "START" ""

  run_in_docker "set -e
nsys profile -t cuda,nvtx,osrt,cublas -s none --force-overwrite=true \
  -o '${c_sout}/${base}' \
  python3 -u /workspace/tools/bench_qwen25vl_guide_kernel_nsys.py \
    --model-path '${c_model}' --mode captured --target-layer 8 \
    --resize-px ${RESIZE_PX} --block-sizes '16,32,64' --num-warps '4,8' \
    --num-stages '3,4' --profile-iters ${PROFILE_ITERS} --warmup ${WARMUP} \
    --precision '${PRECISION}' --output-dir '${c_sout}/${base}_run' --seed ${SEED}
nsys stats -r nvtx_gpu_proj_trace --format csv --output '${c_sout}/${base}' \
  '${c_sout}/${base}.nsys-rep'
python3 /workspace/tools/parse_qwen_kernel_nsys_sweep.py \
  --input '${c_sout}/${base}_nvtx_gpu_proj_trace.csv' \
  --output '${c_sout}/${base}_block_sweep_summary.csv' \
  --info '${c_sout}/${base}_run/bench_info.json' \
  --correctness '${c_sout}/${base}_run/candidates_correctness.json' \
  --best '${c_sout}/best_block_config.json'
" 2>&1 | tee "${out}/${base}_stdout.log"

  if [ -f "${out}/best_block_config.json" ]; then
    write_status "${stage}" "DONE" "block sweep complete"
    log_progress "DONE" "Stage 11: block sweep done"
  else
    write_status "${stage}" "FAIL" "no best config produced"
    log_progress "FAIL" "Stage 11: block sweep FAILED"
    return 1
  fi
}

# ── Stage 12: segment comparison ─────────────────────────────────────────────
stage12_segment_compare() {
  local stage="stage12_segment_compare"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/segment_compare/stage12_segment_compare"
  mkdir -p "${out}"
  local c_sout="${out/${PROJECT_ROOT}/\/workspace}"
  local best="${c_out}/nsys_block_sweep/stage11_block_sweep/best_block_config.json"
  local base="qwen25vl-guide-kernel-segment-compare-${PRECISION}"
  log_progress "START" "Stage 12: segment comparison"
  write_status "${stage}" "START" ""

  run_in_docker "set -e
nsys profile -t cuda,nvtx,osrt,cublas -s none --force-overwrite=true \
  -o '${c_sout}/${base}' \
  python3 -u /workspace/tools/compare_qwen_baseline_vs_guide_segments.py \
    --model-path '${c_model}' --target-layers '0,1,8,16,24' \
    --best-config '${best}' --resize-px ${RESIZE_PX} \
    --profile-iters ${PROFILE_ITERS} --warmup ${WARMUP} \
    --precision '${PRECISION}' --output-dir '${c_sout}/${base}_run' --seed ${SEED}
nsys stats -r nvtx_gpu_proj_trace --format csv --output '${c_sout}/${base}' \
  '${c_sout}/${base}.nsys-rep'
python3 /workspace/tools/parse_qwen_segment_compare_nsys.py \
  --input '${c_sout}/${base}_nvtx_gpu_proj_trace.csv' \
  --output '${c_sout}/${base}_segment_compare_summary.csv' \
  --info '${c_sout}/${base}_run/segment_info.json' \
  --derived '${c_sout}/segment_compare_derived.json'
" 2>&1 | tee "${out}/${base}_stdout.log"

  if [ -f "${out}/segment_compare_derived.json" ]; then
    write_status "${stage}" "DONE" "segment compare complete"
    log_progress "DONE" "Stage 12: segment compare done"
  else
    write_status "${stage}" "FAIL" "no derived metrics produced"
    log_progress "FAIL" "Stage 12: segment compare FAILED"
    return 1
  fi
}

# ── Stage 13: final report ───────────────────────────────────────────────────
stage13_report() {
  local stage="stage13_final_report"
  echo "${stage}" > "${CURRENT_STAGE}"
  log_progress "START" "Stage 13: final report"
  write_status "${stage}" "START" ""
  run_in_docker "python3 /workspace/tools/generate_qwen_guide_final_report.py --root '${c_out}'" \
    2>&1 | tee "${OUTPUT_ROOT}/summaries/phase3_report_stdout.log"
  write_status "${stage}" "DONE" ""
  log_progress "DONE" "Stage 13: final report done"
}

# ── Scientific briefing ──────────────────────────────────────────────────────
brief() {
  echo ""
  echo "=================================================================="
  echo " PHASE 3 SCIENTIFIC BRIEFING (kernel-level only)"
  echo "=================================================================="
  python3 - "${OUTPUT_ROOT}" <<'PY'
import json, sys, csv
from pathlib import Path
R = Path(sys.argv[1])
best = {}
bp = R/"nsys_block_sweep/stage11_block_sweep/best_block_config.json"
if bp.exists(): best = json.loads(bp.read_text())
der = {}
dp = R/"segment_compare/stage12_segment_compare/segment_compare_derived.json"
if dp.exists(): der = json.loads(dp.read_text())

def ratio(rows, key="status", val="PASS"):
    return sum(1 for r in rows if r.get(key)==val), len(rows)
def rd(p):
    p=Path(p); return list(csv.DictReader(open(p))) if p.exists() else []
s9=rd(R/"triton_correctness/stage9_synthetic/triton_synthetic_correctness.csv")
s10=rd(R/"captured_activation/stage10_qwen_activation_correctness/captured_activation_correctness.csv")

print("\nQ1. Fastest Triton block-size config (correctness-passing only):")
if best:
    print(f"   BLOCK_SIZE={best.get('block_size')} num_warps={best.get('num_warps')} "
          f"num_stages={best.get('num_stages')}")
    print(f"   mean = {best.get('mean_ms_per_iter',0)*1e3:.2f} us/iter  "
          f"correctness={best.get('correctness_status','PASS')} "
          f"(rel_l2={best.get('rel_l2_error')})")
    p9,t9=ratio(s9); p10,t10=ratio(s10)
    print(f"   accuracy gates: synthetic {p9}/{t9} PASS, captured {p10}/{t10} PASS")
else:
    print("   (no winner — Stage 11 incomplete)")

print("\nQ2. Comparison A (Attention-Core Only):")
acs = der.get("attention_core_speedup")
print(f"   baseline_attn = {der.get('baseline_attention_core_mean_ms')} ms/iter")
print(f"   guide_attn    = {der.get('guide_attention_core_mean_ms')} ms/iter")
if acs is not None:
    verb = "faster" if acs>=1 else "slower"
    print(f"   => GUIDE attention core is {acs}x ({verb}) than baseline window attention core")

print("\nQ3. Comparison B (Replacement-Equivalent, layout amortized):")
print(f"   one-time layout_total = {der.get('layout_total_ms')} ms "
      f"(amortized/window-layer = {der.get('layout_amortized_per_window_layer_ms')} ms over "
      f"{der.get('num_window_layers')} layers)")
print(f"   baseline_equivalent_per_layer = {der.get('baseline_equivalent_per_layer_ms')} ms")
print(f"   guide_equivalent_per_layer    = {der.get('guide_equivalent_per_layer_ms')} ms")
res = der.get("replacement_equivalent_speedup")
if res is not None:
    verb = "gain" if res>=1 else "loss"
    print(f"   => reorder-free GUIDE segment shows {res}x kernel-level {verb}")
print("\n(Non-goals: no E2E/TTFT/TPOT/accuracy claims.)")
PY
}

main() {
  log_progress "START" "=== Phase 3: Nsys Profiling & Report (Stage 11~13) ==="
  stage11_block_sweep || { log_progress "FAIL" "Phase 3 aborted at Stage 11"; exit 1; }
  stage12_segment_compare || { log_progress "FAIL" "Phase 3 aborted at Stage 12"; exit 1; }
  stage13_report
  log_progress "DONE" "=== Phase 3 COMPLETE ==="
  brief
}

main "$@"
