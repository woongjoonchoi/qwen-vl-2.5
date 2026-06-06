#!/usr/bin/env bash
# =============================================================================
# Phase 2: Descriptor Design, Triton Implementation & Correctness (Stage 5~10)
# All Python runs INSIDE Docker (qwen25vl-guide:latest)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_guide_kernel_level}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-guide:latest}"

NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2"
NCU_HOST_DIR="/opt/nvidia/nsight-compute/2025.3.1"
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
  local gpu_id="${1:-0}"; shift
  local cmd="$1"
  local mounts=(
    "-v" "${PROJECT_ROOT}:/workspace"
  )
  [ -d "${NSYS_HOST_DIR}" ] && mounts+=("-v" "${NSYS_HOST_DIR}:/opt/nsight-systems:ro")
  [ -d "${NCU_HOST_DIR}"  ] && mounts+=("-v" "${NCU_HOST_DIR}:/opt/nsight-compute:ro")

  docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" \
    "${mounts[@]}" \
    -e "CUDA_VISIBLE_DEVICES=${gpu_id}" \
    -e "PYTHONUNBUFFERED=1" \
    -e "HOME=/tmp" \
    -e "TRITON_CACHE_DIR=/tmp/.triton" \
    -e "XDG_CACHE_HOME=/tmp/.cache" \
    -e "PATH=/opt/nsight-systems/bin:/opt/nsight-compute:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    -w /workspace \
    "${IMAGE_NAME}" \
    -c "${cmd}"
}

c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
c_data="${DATA_ROOT/${PROJECT_ROOT}/\/workspace}"
c_out="${OUTPUT_ROOT/${PROJECT_ROOT}/\/workspace}"

# ── Stage 5: Descriptor Generator Validation (Python) ────────────────────────
stage5_descriptor_gen() {
  local stage="stage5_descriptor_generator"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/descriptor_specs"
  mkdir -p "${out}"
  log_progress "START" "Stage 5: Qwen descriptor generator"
  write_status "${stage}" "START" ""

  run_in_docker 0 "python3 - << 'PY'
import sys
sys.path.insert(0, '/workspace/tools')
from qwen_descriptor import QwenDescriptor, get_per_batch_blocks
import json, torch
from pathlib import Path

cfg_path = '/workspace/output/qwen25vl_guide_kernel_level/stage4_qwen_structure/qwen_vision_config.json'
cfg = json.loads(Path(cfg_path).read_text())
nh = cfg['num_heads']; rws = cfg['raw_window_size']; sms = cfg['spatial_merge_size']
gen = QwenDescriptor(num_heads=nh, raw_window_size=rws, spatial_merge_size=sms)

examples = []
for res, H, W in [(448, 32, 32), (896, 64, 64), (1344, 96, 96)]:
    for BS in [16, 32, 64]:
        meta = gen.build(H, W, BS, device='cpu')
        pbk  = get_per_batch_blocks(H, W, nh, rws, sms, BS)
        win_groups = gen.emit_window_groups(H, W, BS)
        ex = {
            'resolution': res, 'H': H, 'W': W, 'BLOCK_SIZE': BS,
            'meta_len': len(meta), 'per_batch_blocks': pbk,
            'n_windows': len(win_groups),
            'first_window': win_groups[0][:8] if win_groups else [],
            'meta_prefix': meta[:10].tolist(),
        }
        examples.append(ex)
        print(f'H={H} W={W} BS={BS}: meta_len={len(meta)} pbk={pbk} n_win={len(win_groups)}')

Path('/workspace/output/qwen25vl_guide_kernel_level/descriptor_specs/qwen_descriptor_examples.json').write_text(
    json.dumps(examples, indent=2))
print('Stage5: DONE')
PY
" 2>&1 | tee "${out}/stage5_stdout.log"

  python3 -c "import json; open('${OUTPUT_ROOT}/stage_status/stage5_status.json','w').write(json.dumps({'status':'DONE'}))"
  write_status "${stage}" "DONE" ""
  log_progress "DONE" "Stage 5: descriptor generator done"
}

# ── Stage 6: Offset Equivalence Tests ────────────────────────────────────────
stage6_offset_equivalence() {
  local stage="stage6_offset_equivalence"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/synthetic_tests/stage6_offset_equivalence"
  mkdir -p "${out}"
  log_progress "START" "Stage 6: offset/length/RoPE equivalence tests"
  write_status "${stage}" "START" ""

  local c_sout="${out/${PROJECT_ROOT}/\/workspace}"

  run_in_docker 0 "python3 -u /workspace/tools/test_qwen_descriptor_offsets.py \
    --model-path '${c_model}' \
    --output-dir '${c_sout}' \
    --cases 'divisible' \
    --seed ${SEED}" 2>&1 | tee "${out}/run_stdout.log"

  local exit_code=${PIPESTATUS[0]}
  if [ "${exit_code}" -ne 0 ]; then
    log_progress "FAIL" "Stage 6: offset equivalence FAILED — cannot proceed to Triton"
    write_status "${stage}" "FAIL" "offset not equivalent"
    exit 1
  fi

  write_status "${stage}" "DONE" "offset equivalence PASS"
  log_progress "DONE" "Stage 6: offset equivalence done"
}

# ── Stage 7: Python Reference Correctness ─────────────────────────────────────
stage7_python_reference() {
  local stage="stage7_python_reference"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/python_reference/stage7_python_reference"
  mkdir -p "${out}"
  log_progress "START" "Stage 7: Python reference attention correctness"
  write_status "${stage}" "START" ""

  local c_sout="${out/${PROJECT_ROOT}/\/workspace}"

  run_in_docker 0 "python3 -u /workspace/tools/test_qwen_python_descriptor_attention.py \
    --model-path '${c_model}' \
    --output-dir '${c_sout}' \
    --dtype 'fp32,bf16' \
    --seed ${SEED}" 2>&1 | tee "${out}/run_stdout.log"

  local exit_code=${PIPESTATUS[0]}
  if [ "${exit_code}" -ne 0 ]; then
    log_progress "WARN" "Stage 7: Python reference had failures (check log)"
  fi

  write_status "${stage}" "DONE" ""
  log_progress "DONE" "Stage 7: Python reference done"
}

# ── Stage 9: Triton Synthetic Correctness ─────────────────────────────────────
stage9_triton_synthetic() {
  local stage="stage9_triton_synthetic"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/triton_correctness/stage9_synthetic"
  mkdir -p "${out}"
  log_progress "START" "Stage 9: Triton synthetic correctness"
  write_status "${stage}" "START" ""

  local c_sout="${out/${PROJECT_ROOT}/\/workspace}"

  run_in_docker 0 "python3 -u /workspace/tools/test_qwen_triton_descriptor_attention.py \
    --model-path '${c_model}' \
    --output-dir '${c_sout}' \
    --cases '8x8,16x16,32x32,32x48' \
    --block-sizes '16,32,64' \
    --dtypes 'fp16,bf16' \
    --seed ${SEED}" 2>&1 | tee "${out}/run_stdout.log"

  # Show result CSV
  if [ -f "${out}/triton_synthetic_correctness.csv" ]; then
    log_progress "INFO" "Triton correctness results:"
    python3 - "${out}/triton_synthetic_correctness.csv" <<'PY'
import csv, sys
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        status = row.get('status','?')
        err = row.get('max_abs_error','?')
        print(f"  {row.get('case'):<12} BS={row.get('BLOCK_SIZE'):<4} {row.get('dtype'):<6} "
              f"max_err={err:<12} {status}")
PY
  fi

  write_status "${stage}" "DONE" ""
  log_progress "DONE" "Stage 9: Triton synthetic correctness done"
}

# ── Stage 10: Captured Qwen Activation Correctness ───────────────────────────
stage10_captured_activation() {
  local stage="stage10_captured_activation"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/captured_activation/stage10_qwen_activation_correctness"
  mkdir -p "${out}"
  log_progress "START" "Stage 10: captured Qwen activation correctness"
  write_status "${stage}" "START" ""

  local c_sout="${out/${PROJECT_ROOT}/\/workspace}"

  # Check if manifest exists
  local manifest="${OUTPUT_ROOT}/stage0_50_data_manifest/benchmark_manifest.jsonl" 2>/dev/null
  [ -f "${manifest}" ] || manifest="${PROJECT_ROOT}/output/qwen25vl_baseline/stage0_50_data_manifest/benchmark_manifest.jsonl"

  local c_manifest="${manifest/${PROJECT_ROOT}/\/workspace}"

  run_in_docker 0 "python3 -u /workspace/tools/test_qwen_captured_activation_correctness.py \
    --model-path '${c_model}' \
    --manifest '${c_manifest}' \
    --target-layers '0,1,8' \
    --block-sizes '16,32,64' \
    --precision '${PRECISION}' \
    --output-dir '${c_sout}' \
    --seed ${SEED}" 2>&1 | tee "${out}/run_stdout.log"

  write_status "${stage}" "DONE" ""
  log_progress "DONE" "Stage 10: captured activation done"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  log_progress "START" "=== Phase 2: Kernel Dev & Test (Stage 5~10) ==="

  stage5_descriptor_gen
  stage6_offset_equivalence
  stage7_python_reference
  stage9_triton_synthetic
  stage10_captured_activation

  log_progress "DONE" "=== Phase 2 COMPLETE ==="

  echo ""
  echo "=================================================================="
  echo " PHASE 2 COMPLETE"
  echo "=================================================================="
  # Summary
  python3 - "${OUTPUT_ROOT}" <<'PY'
import json, csv
from pathlib import Path
root = Path(sys.argv[1]) if len(__import__('sys').argv)>1 else Path('.')
import sys; root = Path(sys.argv[1])

print("\nStage Results:")
stages = {
    "5_descriptor_gen": root/"stage_status/stage5_status.json",
    "6_offset_equiv":   root/"synthetic_tests/stage6_offset_equivalence/stage_status.json",
    "7_python_ref":     root/"python_reference/stage7_python_reference/stage_status.json",
    "9_triton_synth":   root/"triton_correctness/stage9_synthetic/stage_status.json",
    "10_captured":      root/"captured_activation/stage10_qwen_activation_correctness/stage_status.json",
}
for name, path in stages.items():
    try:
        d = json.loads(path.read_text())
        status = d.get("status","?")
    except:
        status = "MISSING"
    print(f"  {name}: {status}")

# Print Triton correctness summary
csv_path = root/"triton_correctness/stage9_synthetic/triton_synthetic_correctness.csv"
if csv_path.exists():
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    pass_count = sum(1 for r in rows if r.get("status")=="PASS")
    print(f"\nTriton correctness: {pass_count}/{len(rows)} PASS")
    best_bs = None
    best_err = float("inf")
    for r in rows:
        if r.get("status")=="PASS":
            err = float(r.get("max_abs_error", 999))
            if err < best_err:
                best_err = err
                best_bs = r.get("BLOCK_SIZE")
    if best_bs:
        print(f"Best BLOCK_SIZE for correctness: {best_bs}")
PY
}

main "$@"
