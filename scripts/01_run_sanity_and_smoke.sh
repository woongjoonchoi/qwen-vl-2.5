#!/usr/bin/env bash
# =============================================================================
# Phase 1: Sanity & Smoke Test Pipeline  (Stage 0 ~ Stage 4)
# Qwen2.5-VL Baseline Measurement
# =============================================================================
set -uo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_baseline}"
TOOLS_DIR="${PROJECT_ROOT}/tools"

PRECISION="${PRECISION:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-2}"
MEASURED_ITERS="${MEASURED_ITERS:-5}"
PROFILE_ITERS="${PROFILE_ITERS:-3}"
SEED="${SEED:-42}"

# ─── Logging infrastructure ───────────────────────────────────────────────────
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}" \
         "${OUTPUT_ROOT}/runs" \
         "${OUTPUT_ROOT}/nsys" \
         "${OUTPUT_ROOT}/ncu" \
         "${OUTPUT_ROOT}/stage_status" \
         "${OUTPUT_ROOT}/manifests" \
         "${OUTPUT_ROOT}/summaries"

PROGRESS_LOG="${LOG_DIR}/progress.log"
STATUS_JSONL="${LOG_DIR}/status.jsonl"
ERROR_LOG="${LOG_DIR}/error.log"
CURRENT_STAGE="${LOG_DIR}/current_stage.txt"
LATEST_LOG="${LOG_DIR}/latest.log"

touch "${PROGRESS_LOG}" "${STATUS_JSONL}" "${ERROR_LOG}"

NSYS_AVAILABLE="false"
NCU_AVAILABLE="false"
SMOKE_RESULT_FILE="${OUTPUT_ROOT}/stage_status/profiler_smoke.json"

timestamp() { date +"%Y-%m-%dT%H:%M:%S"; }
ts_human() { date +"%Y-%m-%d %H:%M:%S"; }

log_progress() {
  local level="$1"; local msg="$2"
  local line="[$(ts_human)] [${level}] ${msg}"
  echo "${line}" | tee -a "${PROGRESS_LOG}" | tee -a "${LATEST_LOG}"
}

write_status() {
  local stage="$1"; local status="$2"; local message="${3:-}"
  python3 - "${STATUS_JSONL}" "${stage}" "${status}" "${message}" <<'PY'
import json, sys, datetime
path, stage, status, message = sys.argv[1:5]
row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
       "stage": stage, "status": status, "message": message}
with open(path, "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

fail_hard() {
  local stage="$1"; local msg="$2"
  log_progress "FAIL" "${stage}: ${msg}"
  write_status "${stage}" "FAIL" "${msg}"
  echo "${msg}" >> "${ERROR_LOG}"
  echo "HARD FAIL at ${stage}: ${msg}"
  exit 1
}

# ─── Stage 0: Path / Environment Inspection ──────────────────────────────────
stage0_path_inspection() {
  local stage="stage0_path_inspection"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 0: path/environment inspection"
  write_status "${stage}" "START" "checking DATA_ROOT and MODEL_PATH"

  {
    echo "=== PATH INSPECTION ==="
    echo "PROJECT_ROOT: ${PROJECT_ROOT}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "MODEL_ROOT: ${MODEL_ROOT}"
    echo "MODEL_PATH: ${MODEL_PATH}"
    echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"

    echo ""
    echo "=== DIRECTORY EXISTENCE ==="
    test -d "${DATA_ROOT}"   && echo "DATA_ROOT: OK"   || echo "DATA_ROOT: MISSING"
    test -d "${MODEL_ROOT}"  && echo "MODEL_ROOT: OK"  || echo "MODEL_ROOT: MISSING"
    test -d "${MODEL_PATH}"  && echo "MODEL_PATH: OK"  || echo "MODEL_PATH: MISSING"

    echo ""
    echo "=== MODEL FILES ==="
    if [ -d "${MODEL_PATH}" ]; then
      ls "${MODEL_PATH}"
    else
      echo "MODEL_PATH does not exist, searching..."
      find "${MODEL_ROOT}" -maxdepth 3 -name "config.json" 2>/dev/null
    fi

    echo ""
    echo "=== DATA ROOT CONTENTS ==="
    ls "${DATA_ROOT}" 2>/dev/null || echo "(empty or missing)"
  } 2>&1 | tee "${out}/path_check.txt"

  {
    echo "=== NVIDIA-SMI ==="
    nvidia-smi 2>&1 || echo "nvidia-smi not available"

    echo ""
    echo "=== GPU DETAILS ==="
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used \
      --format=csv,noheader 2>/dev/null || true

    echo ""
    echo "=== NSYS / NCU PATHS ==="
    which nsys 2>/dev/null && echo "nsys: $(which nsys)" || echo "nsys: not in PATH"
    which ncu 2>/dev/null  && echo "ncu: $(which ncu)"   || echo "ncu: not in PATH"
    ls /usr/local/bin/nsys 2>/dev/null || true
    ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1 || true

    echo ""
    echo "=== PYTHON PACKAGES ==="
    python3 -c "
import sys
print('python:', sys.version[:20])
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('gpu_count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'  gpu_{i}:', torch.cuda.get_device_name(i),
                  'mem:', round(torch.cuda.get_device_properties(i).total_memory/1e9, 1), 'GB')
except ImportError as e:
    print('torch: MISSING -', e)

try:
    import transformers
    print('transformers:', transformers.__version__)
except ImportError as e:
    print('transformers: MISSING -', e)

try:
    import qwen_vl_utils
    print('qwen_vl_utils: OK')
except ImportError as e:
    print('qwen_vl_utils: MISSING -', e)

try:
    import PIL
    print('Pillow:', PIL.__version__)
except ImportError as e:
    print('Pillow: MISSING -', e)

try:
    import pandas
    print('pandas:', pandas.__version__)
except ImportError as e:
    print('pandas: MISSING -', e)
" 2>&1
  } 2>&1 | tee "${out}/environment.txt"

  # Check hard requirements
  local fail=0
  if [ ! -d "${DATA_ROOT}" ]; then
    log_progress "FAIL" "DATA_ROOT does not exist: ${DATA_ROOT}"
    fail=1
  fi
  if [ ! -d "${MODEL_ROOT}" ]; then
    log_progress "FAIL" "MODEL_ROOT does not exist: ${MODEL_ROOT}"
    fail=1
  fi
  if [ ! -d "${MODEL_PATH}" ]; then
    log_progress "WARN" "MODEL_PATH not found: ${MODEL_PATH}, searching..."
    FOUND=$(find "${MODEL_ROOT}" -maxdepth 3 -name "config.json" 2>/dev/null | head -1)
    if [ -n "${FOUND}" ]; then
      MODEL_PATH="$(dirname ${FOUND})"
      log_progress "INFO" "Found model at: ${MODEL_PATH}"
      export MODEL_PATH
    else
      log_progress "FAIL" "No model found under ${MODEL_ROOT}"
      fail=1
    fi
  fi

  if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log_progress "INFO" "PyTorch CUDA: OK"
  else
    log_progress "FAIL" "PyTorch CUDA not available"
    fail=1
  fi

  if [ "${fail}" -eq 1 ]; then
    fail_hard "${stage}" "Hard requirements not met (see ${out}/environment.txt)"
  fi

  python3 - "${out}" "${MODEL_PATH}" "${DATA_ROOT}" <<'PY'
import json, sys
from pathlib import Path
out, model_path, data_root = sys.argv[1:4]
result = {
    "model_path": model_path,
    "data_root": data_root,
    "model_exists": Path(model_path).exists(),
    "data_exists": Path(data_root).exists(),
    "benchmarks_found": [d.name for d in Path(data_root).iterdir() if d.is_dir()],
}
(Path(out) / "detected_model_paths.json").write_text(json.dumps(result, indent=2))
(Path(out) / "stage_status.json").write_text(json.dumps({"status": "PASS"}, indent=2))
PY

  write_status "${stage}" "DONE" "path inspection complete"
  log_progress "DONE" "Stage 0: path inspection OK"
}

# ─── Stage 0.25: Docker/Profiler Smoke Test ───────────────────────────────────
stage0_25_profiler_smoke() {
  local stage="stage0_25_profiler_smoke"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 0.25: CUDA + nsys + ncu smoke test"
  write_status "${stage}" "START" "testing CUDA, nsys, ncu"

  local cuda_ok="FAIL"
  local nsys_ok="FAIL"
  local nsys_stats_ok="FAIL"
  local ncu_ok="FAIL"
  local ncu_fail_reason=""

  # --- CUDA smoke ---
  if python3 "${TOOLS_DIR}/smoke_cuda_profiler.py" \
     2>&1 | tee "${out}/cuda_smoke_stdout.log" | grep -q "SMOKE_CUDA_PROFILER_OK"; then
    cuda_ok="PASS"
    log_progress "INFO" "  CUDA smoke: PASS"
  else
    log_progress "FAIL" "  CUDA smoke: FAIL"
    write_status "${stage}" "FAIL" "CUDA smoke failed"
    echo "CUDA smoke FAIL" >> "${ERROR_LOG}"
    # Write status and abort everything
    python3 - "${out}" "${cuda_ok}" "${nsys_ok}" "${ncu_ok}" <<'PY'
import json, sys
out, cuda, nsys, ncu = sys.argv[1:5]
s = {"cuda_smoke": cuda, "nsys_smoke": nsys, "nsys_stats": "FAIL",
     "ncu_smoke": ncu, "NSYS_AVAILABLE": False, "NCU_AVAILABLE": False}
open(f"{out}/profiler_smoke_status.json", "w").write(json.dumps(s, indent=2))
open(f"{out}/stage_status.json", "w").write(json.dumps({"status":"FAIL","message":"CUDA smoke fail"}, indent=2))
PY
    fail_hard "${stage}" "CUDA smoke test failed — cannot proceed"
  fi

  # --- nsys smoke ---
  NSYS_BIN=$(which nsys 2>/dev/null || echo "")
  if [ -z "${NSYS_BIN}" ]; then
    log_progress "WARN" "  nsys: not found in PATH"
    nsys_ok="NOT_FOUND"
  else
    log_progress "INFO" "  nsys binary: ${NSYS_BIN}"
    "${NSYS_BIN}" profile \
       -t cuda,nvtx \
       --force-overwrite=true \
       -o "${out}/nsys_smoke" \
       python3 "${TOOLS_DIR}/smoke_cuda_profiler.py" \
       2>&1 | tee "${out}/nsys_smoke_stdout.log" || true
    # Check success by file existence (nsys may return non-zero even on success)
    if [ -f "${out}/nsys_smoke.nsys-rep" ] && \
       grep -q "SMOKE_CUDA_PROFILER_OK" "${out}/nsys_smoke_stdout.log" 2>/dev/null; then
      nsys_ok="PASS"
      log_progress "INFO" "  nsys profile: PASS (nsys-rep created)"
    elif [ -f "${out}/nsys_smoke.nsys-rep" ]; then
      nsys_ok="PASS"
      log_progress "INFO" "  nsys profile: PASS (nsys-rep created, stdout may differ)"
    else
      nsys_ok="FAIL"
      log_progress "WARN" "  nsys profile: FAIL (no nsys-rep created)"
    fi

    # nsys stats
    if [ "${nsys_ok}" = "PASS" ] && [ -f "${out}/nsys_smoke.nsys-rep" ]; then
      "${NSYS_BIN}" stats \
        -r nvtx_gpu_proj_trace \
        --format csv \
        --output "${out}/nsys_smoke" \
        "${out}/nsys_smoke.nsys-rep" \
        2>&1 | tee "${out}/nsys_stats_stdout.log" || true

      if [ -f "${out}/nsys_smoke_nvtx_gpu_proj_trace.csv" ]; then
        nsys_stats_ok="PASS"
        log_progress "INFO" "  nsys stats: PASS"
      else
        nsys_stats_ok="FAIL"
        log_progress "WARN" "  nsys stats CSV not generated"
      fi
    fi
  fi

  # --- ncu smoke ---
  NCU_BIN=$(which ncu 2>/dev/null || echo "")
  if [ -z "${NCU_BIN}" ]; then
    log_progress "WARN" "  ncu: not found in PATH"
    ncu_ok="NOT_FOUND"
    ncu_fail_reason="ncu not in PATH"
  else
    log_progress "INFO" "  ncu binary: ${NCU_BIN}"
    local ncu_out
    ncu_out=$("${NCU_BIN}" \
      --set launchstats \
      --target-processes all \
      --force-overwrite \
      -o "${out}/ncu_smoke" \
      python3 "${TOOLS_DIR}/smoke_cuda_profiler.py" \
      2>&1 | tee "${out}/ncu_smoke_stdout.log"; true)

    if grep -qiE "ERR_NVGPUCTRPERM|permission|counter" "${out}/ncu_smoke_stdout.log" 2>/dev/null; then
      ncu_ok="FAIL_PERMISSION"
      ncu_fail_reason="ERR_NVGPUCTRPERM or permission/counter error"
      log_progress "WARN" "  ncu smoke: FAIL_PERMISSION (${ncu_fail_reason})"
    elif [ -f "${out}/ncu_smoke.ncu-rep" ]; then
      ncu_ok="PASS"
      log_progress "INFO" "  ncu smoke: PASS"
    else
      ncu_ok="FAIL"
      ncu_fail_reason="ncu report not generated"
      log_progress "WARN" "  ncu smoke: FAIL (${ncu_fail_reason})"
    fi
  fi

  # Set global availability flags
  if [ "${nsys_ok}" = "PASS" ] && [ "${nsys_stats_ok}" = "PASS" ]; then
    NSYS_AVAILABLE="true"
  fi
  if [ "${ncu_ok}" = "PASS" ]; then
    NCU_AVAILABLE="true"
  fi

  # Save status
  python3 - "${out}" "${cuda_ok}" "${nsys_ok}" "${nsys_stats_ok}" "${ncu_ok}" \
            "${ncu_fail_reason}" "${NSYS_AVAILABLE}" "${NCU_AVAILABLE}" \
            "${SMOKE_RESULT_FILE}" <<'PY'
import json, sys
from pathlib import Path
out, cuda, nsys, nsys_stats, ncu, ncu_reason, nsys_av, ncu_av, smoke_file = sys.argv[1:10]
s = {
    "cuda_smoke": cuda, "nsys_smoke": nsys, "nsys_stats": nsys_stats,
    "ncu_smoke": ncu, "ncu_fail_reason": ncu_reason,
    "NSYS_AVAILABLE": nsys_av == "true",
    "NCU_AVAILABLE": ncu_av == "true",
}
Path(out).mkdir(parents=True, exist_ok=True)
Path(f"{out}/profiler_smoke_status.json").write_text(json.dumps(s, indent=2))
Path(f"{out}/stage_status.json").write_text(json.dumps({"status": "DONE", **s}, indent=2))
Path(smoke_file).parent.mkdir(parents=True, exist_ok=True)
Path(smoke_file).write_text(json.dumps(s, indent=2))
PY

  local case_result
  if [ "${nsys_ok}" = "PASS" ] && [ "${ncu_ok}" = "PASS" ]; then
    case_result="Case A: nsys+ncu PASS"
  elif [ "${nsys_ok}" = "PASS" ] && [[ "${ncu_ok}" == *PERMISSION* || "${ncu_ok}" == "FAIL_PERMISSION" ]]; then
    case_result="Case B: nsys PASS, ncu PERMISSION FAIL"
  elif [ "${nsys_ok}" != "PASS" ]; then
    case_result="Case C: nsys FAIL"
  else
    case_result="Case B: nsys PASS, ncu FAIL"
  fi

  write_status "${stage}" "DONE" \
    "cuda=${cuda_ok} nsys=${nsys_ok} ncu=${ncu_ok} NSYS_AVAILABLE=${NSYS_AVAILABLE} NCU_AVAILABLE=${NCU_AVAILABLE}"
  log_progress "DONE" "Stage 0.25 profiler smoke: ${case_result} | nsys=${nsys_ok} ncu=${ncu_ok}"
}

# ─── Stage 0.50: Data Manifest Smoke ─────────────────────────────────────────
stage0_50_data_manifest_smoke() {
  local stage="stage0_50_data_manifest"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 0.50: benchmark data manifest smoke"
  write_status "${stage}" "START" "scanning data/raw"

  python3 -u "${TOOLS_DIR}/build_qwen_benchmark_manifest.py" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${out}" \
    --max-samples-per-benchmark 5 \
    2>&1 | tee "${out}/run_stdout.log"

  local exit_code=${PIPESTATUS[0]}
  if [ "${exit_code}" -ne 0 ]; then
    fail_hard "${stage}" "manifest build failed (exit ${exit_code})"
  fi

  # Verify manifest exists and has content
  if [ ! -f "${out}/benchmark_manifest.jsonl" ]; then
    fail_hard "${stage}" "benchmark_manifest.jsonl not created"
  fi
  local n_samples
  n_samples=$(wc -l < "${out}/benchmark_manifest.jsonl")
  if [ "${n_samples}" -eq 0 ]; then
    fail_hard "${stage}" "benchmark_manifest.jsonl is empty"
  fi

  # Export manifest path for downstream stages
  export MANIFEST_PATH="${out}/benchmark_manifest.jsonl"

  write_status "${stage}" "DONE" "manifest: ${n_samples} samples"
  log_progress "DONE" "Stage 0.50: manifest with ${n_samples} samples"
}

# ─── Stage 0.75: Model Load + Inference Smoke ────────────────────────────────
stage0_75_model_inference_smoke() {
  local stage="stage0_75_model_data_smoke"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 0.75: model load + benchmark inference smoke"
  write_status "${stage}" "START" "loading model and running 2 samples"

  echo "python3 -u tools/profile_qwen25vl_baseline.py --mode inference_smoke ..." \
    > "${out}/run_command.txt"

  python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
    --mode inference_smoke \
    --model-path "${MODEL_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --benchmark "any" \
    --num-samples 2 \
    --resolution 896 \
    --max-new-tokens 32 \
    --batch-size 1 \
    --precision "${PRECISION}" \
    --seed "${SEED}" \
    --output-dir "${out}" \
    2>&1 | tee "${out}/run_stdout.log"

  local exit_code=${PIPESTATUS[0]}

  if [ "${exit_code}" -ne 0 ]; then
    fail_hard "${stage}" "model inference smoke failed (exit ${exit_code})"
  fi

  local status_file="${out}/stage_status.json"
  if [ -f "${status_file}" ]; then
    local status
    status=$(python3 -c "import json; d=json.load(open('${status_file}')); print(d.get('status','UNKNOWN'))")
    if [ "${status}" != "PASS" ]; then
      fail_hard "${stage}" "inference smoke status=${status}"
    fi
  fi

  write_status "${stage}" "DONE" "model inference smoke PASS"
  log_progress "DONE" "Stage 0.75: model inference smoke PASS"
}

# ─── Stage 1: Generation Sanity ──────────────────────────────────────────────
stage1_generation_sanity() {
  local stage="stage1_generation_sanity"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 1: generation sanity (5 samples)"
  write_status "${stage}" "START" "docvqa 5 samples res=896 tok=64"

  python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
    --mode inference_smoke \
    --model-path "${MODEL_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --benchmark "docvqa" \
    --num-samples 5 \
    --resolution 896 \
    --max-new-tokens 64 \
    --batch-size 1 \
    --precision "${PRECISION}" \
    --seed "${SEED}" \
    --output-dir "${out}" \
    2>&1 | tee "${out}/run_stdout.log"

  local exit_code=${PIPESTATUS[0]}

  # If docvqa has <5 samples, try any
  if [ "${exit_code}" -ne 0 ]; then
    log_progress "WARN" "  Stage 1: docvqa failed, trying any benchmark"
    python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
      --mode inference_smoke \
      --model-path "${MODEL_PATH}" \
      --manifest "${MANIFEST_PATH}" \
      --benchmark "any" \
      --num-samples 5 \
      --resolution 896 \
      --max-new-tokens 64 \
      --batch-size 1 \
      --precision "${PRECISION}" \
      --seed "${SEED}" \
      --output-dir "${out}" \
      2>&1 | tee "${out}/run_stdout.log"
    exit_code=${PIPESTATUS[0]}
  fi

  # Generate report
  python3 - "${out}" <<'PY'
import json, csv
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
import sys
out = Path(sys.argv[1])

records = []
jsonl = out / "smoke_generated_outputs.jsonl"
if jsonl.exists():
    with open(jsonl) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

with open(out / "generation_sanity_report.md", "w") as f:
    f.write("# Generation Sanity Report\n\n")
    f.write(f"- num_samples: {len(records)}\n")
    for r in records[:5]:
        f.write(f"- {r.get('sample_id','?')}: gen={r.get('actual_generated_tokens',0)} tok, "
                f"e2e={r.get('e2e_ms',0):.1f}ms\n")
        f.write(f"  > {r.get('generated_text','')[:80]!r}\n")

passed = len(records) >= 1 and all(r.get("actual_generated_tokens",0) > 0 for r in records)
status = {"status": "PASS" if passed else "FAIL", "num_successful": len(records)}
(out / "stage_status.json").write_text(json.dumps(status, indent=2))
PY

  write_status "${stage}" "DONE" "generation sanity complete"
  log_progress "DONE" "Stage 1: generation sanity done"
}

# ─── Stage 2: App-Level Timing Sanity ────────────────────────────────────────
stage2_app_timing_sanity() {
  local stage="stage2_app_timing_sanity"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 2: app-level timing sanity (docvqa res=896 tok=64)"
  write_status "${stage}" "START" "app timing docvqa res=896 tok=64"

  python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
    --mode app_timing \
    --model-path "${MODEL_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --benchmark "docvqa" \
    --num-samples 5 \
    --resolution 896 \
    --max-new-tokens 64 \
    --batch-size 1 \
    --precision "${PRECISION}" \
    --warmup "${WARMUP}" \
    --measured-iters "${MEASURED_ITERS}" \
    --seed "${SEED}" \
    --output-dir "${out}" \
    2>&1 | tee "${out}/run_stdout.log"

  local exit_code=${PIPESTATUS[0]}

  if [ "${exit_code}" -ne 0 ]; then
    log_progress "WARN" "  Stage 2: docvqa failed, trying any benchmark"
    python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
      --mode app_timing \
      --model-path "${MODEL_PATH}" \
      --manifest "${MANIFEST_PATH}" \
      --benchmark "any" \
      --num-samples 5 \
      --resolution 896 \
      --max-new-tokens 64 \
      --batch-size 1 \
      --precision "${PRECISION}" \
      --warmup "${WARMUP}" \
      --measured-iters "${MEASURED_ITERS}" \
      --seed "${SEED}" \
      --output-dir "${out}" \
      2>&1 >> "${out}/run_stdout.log"
  fi

  if [ -f "${out}/timing_summary.csv" ]; then
    log_progress "INFO" "  timing_summary.csv created"
    python3 - "${out}/timing_summary.csv" <<'PY'
import csv, sys
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    for row in reader:
        for k in ["mean_e2e_ms", "mean_visual_encoder_ms", "mean_ttft_ms",
                  "mean_f_vision", "mean_decode_only_tok_s"]:
            if row.get(k):
                print(f"  {k}: {row[k]}")
PY
  fi

  write_status "${stage}" "DONE" "app timing sanity complete"
  log_progress "DONE" "Stage 2: app timing sanity done"
}

# ─── Stage 3: Visual Encoder Structure ───────────────────────────────────────
stage3_visual_encoder_structure() {
  local stage="stage3_visual_encoder_structure"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 3: visual encoder structure inspection"
  write_status "${stage}" "START" "inspecting ViT structure"

  python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
    --mode structure_inspect \
    --model-path "${MODEL_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --output-dir "${out}" \
    2>&1 | tee "${out}/run_stdout.log"

  write_status "${stage}" "DONE" "structure inspection complete"
  log_progress "DONE" "Stage 3: visual encoder structure done"
}

# ─── Stage 4: Nsight Systems Sanity ──────────────────────────────────────────
stage4_nsys_sanity() {
  local stage="stage4_nsys_sanity"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 4: Nsight Systems sanity run"
  write_status "${stage}" "START" "nsys sanity docvqa res=896 tok=64"

  if [ "${NSYS_AVAILABLE}" != "true" ]; then
    log_progress "WARN" "  Stage 4: NSYS_AVAILABLE=false, skipping"
    python3 -c "import json; open('${out}/stage_status.json','w').write(json.dumps({'status':'SKIPPED_NSYS_UNAVAILABLE'}))"
    write_status "${stage}" "SKIPPED" "NSYS_AVAILABLE=false"
    return 0
  fi

  local base="qwen25vl3b-docvqa-res896-tok64-bs1-${PRECISION}"
  local NSYS_BIN
  NSYS_BIN=$(which nsys)

  "${NSYS_BIN}" profile \
    -w true \
    -t cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -o "${out}/${base}" \
    python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
      --mode nsys_profile \
      --model-path "${MODEL_PATH}" \
      --manifest "${MANIFEST_PATH}" \
      --benchmark "docvqa" \
      --num-samples 1 \
      --resolution 896 \
      --max-new-tokens 64 \
      --batch-size 1 \
      --precision "${PRECISION}" \
      --warmup "${WARMUP}" \
      --profile-iters "${PROFILE_ITERS}" \
      --seed "${SEED}" \
      --output-dir "${out}/${base}_run" \
    2>&1 | tee "${out}/${base}_nsys_stdout.log" || {
    log_progress "WARN" "  Stage 4: nsys profile command failed, continuing..."
  }

  # Export stats if .nsys-rep was created
  if [ -f "${out}/${base}.nsys-rep" ]; then
    log_progress "INFO" "  nsys-rep created, running stats export..."
    "${NSYS_BIN}" stats \
      -r nvtx_gpu_proj_trace \
      --format csv \
      --output "${out}/${base}" \
      "${out}/${base}.nsys-rep" \
      2>&1 | tee "${out}/${base}_nsys_stats_stdout.log" || true

    # Parse nsys CSV if available
    local nsys_csv="${out}/${base}_nvtx_gpu_proj_trace.csv"
    if [ -f "${nsys_csv}" ]; then
      python3 -u "${TOOLS_DIR}/stats_qwen_nsys.py" \
        --input "${nsys_csv}" \
        --output "${out}/${base}_nsys_summary.csv" \
        2>&1 | tee "${out}/${base}_stats_stdout.log" || true
      log_progress "INFO" "  nsys CSV parsed"
    else
      log_progress "WARN" "  nsys CSV not found: ${nsys_csv}"
    fi
  else
    log_progress "WARN" "  nsys-rep not created"
  fi

  # Write stage status
  python3 - "${out}" "${base}" <<'PY'
import json, sys
from pathlib import Path
out, base = sys.argv[1], sys.argv[2]
out = Path(out)
rep = (out / f"{base}.nsys-rep").exists()
csv_f = (out / f"{base}_nvtx_gpu_proj_trace.csv").exists()
summ = (out / f"{base}_nsys_summary.csv").exists()
status = "PASS" if rep else "PARTIAL"
(out / "stage_status.json").write_text(json.dumps({
    "status": status, "nsys_rep": rep, "nvtx_csv": csv_f, "summary_csv": summ
}, indent=2))
PY

  write_status "${stage}" "DONE" "nsys sanity complete"
  log_progress "DONE" "Stage 4: nsys sanity done"
}

# ─── Summary helper ───────────────────────────────────────────────────────────
write_phase1_summary() {
  local summary_file="${OUTPUT_ROOT}/summaries/phase1_smoke_summary.md"
  mkdir -p "${OUTPUT_ROOT}/summaries"

  python3 - "${OUTPUT_ROOT}" "${NSYS_AVAILABLE}" "${NCU_AVAILABLE}" "${summary_file}" <<'PY'
import json, sys
from pathlib import Path

out_root, nsys_av, ncu_av, summary_file = sys.argv[1:5]
root = Path(out_root)

def read_status(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}

stages = {
    "stage0": root / "stage0_path_inspection" / "stage_status.json",
    "stage0_25": root / "stage0_25_profiler_smoke" / "stage_status.json",
    "stage0_50": root / "stage0_50_data_manifest" / "stage_status.json",
    "stage0_75": root / "stage0_75_model_data_smoke" / "stage_status.json",
    "stage1": root / "stage1_generation_sanity" / "stage_status.json",
    "stage2": root / "stage2_app_timing_sanity" / "stage_status.json",
    "stage3": root / "stage3_visual_encoder_structure" / "stage_status.json",
    "stage4": root / "stage4_nsys_sanity" / "stage_status.json",
}

lines = ["# Phase 1 Smoke Test Summary\n"]
lines.append(f"- NSYS_AVAILABLE: {nsys_av}\n")
lines.append(f"- NCU_AVAILABLE: {ncu_av}\n\n")
lines.append("## Stage Results\n\n")
for name, path in stages.items():
    s = read_status(path)
    status = s.get("status", "UNKNOWN")
    lines.append(f"- **{name}**: {status}\n")

Path(summary_file).write_text("".join(lines))
print(f"Written: {summary_file}")
PY

  log_progress "INFO" "Phase 1 summary: ${summary_file}"
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
  log_progress "START" "=== Phase 1: Sanity & Smoke Test Pipeline ==="
  log_progress "INFO" "Project root: ${PROJECT_ROOT}"
  log_progress "INFO" "Output root: ${OUTPUT_ROOT}"
  log_progress "INFO" "Model: ${MODEL_PATH}"
  log_progress "INFO" "Data: ${DATA_ROOT}"

  # Save global config
  python3 - "${OUTPUT_ROOT}" "${MODEL_PATH}" "${DATA_ROOT}" "${PRECISION}" "${SEED}" <<'PY'
import json, sys, datetime
from pathlib import Path
out, model, data, prec, seed = sys.argv[1:6]
cfg = {
    "model_path": model, "data_root": data,
    "precision": prec, "seed": seed,
    "timestamp": datetime.datetime.now().isoformat(),
    "phase": "phase1_sanity_smoke",
}
Path(out).mkdir(parents=True, exist_ok=True)
(Path(out) / "global_config.json").write_text(json.dumps(cfg, indent=2))
PY

  stage0_path_inspection
  stage0_25_profiler_smoke

  # Load profiler flags from smoke result
  if [ -f "${SMOKE_RESULT_FILE}" ]; then
    NSYS_AVAILABLE=$(python3 -c "import json; d=json.load(open('${SMOKE_RESULT_FILE}')); print(str(d.get('NSYS_AVAILABLE',False)).lower())")
    NCU_AVAILABLE=$(python3 -c "import json; d=json.load(open('${SMOKE_RESULT_FILE}')); print(str(d.get('NCU_AVAILABLE',False)).lower())")
  fi
  log_progress "INFO" "Profiler flags: NSYS_AVAILABLE=${NSYS_AVAILABLE} NCU_AVAILABLE=${NCU_AVAILABLE}"
  export NSYS_AVAILABLE NCU_AVAILABLE

  stage0_50_data_manifest_smoke
  stage0_75_model_inference_smoke
  stage1_generation_sanity
  stage2_app_timing_sanity
  stage3_visual_encoder_structure
  stage4_nsys_sanity

  write_phase1_summary

  log_progress "DONE" "=== Phase 1 COMPLETE ==="
  echo ""
  echo "=================================================================="
  echo " PHASE 1 COMPLETE"
  echo "=================================================================="
  echo " NSYS_AVAILABLE : ${NSYS_AVAILABLE}"
  echo " NCU_AVAILABLE  : ${NCU_AVAILABLE}"
  echo " Logs           : ${LOG_DIR}"
  echo " Progress       : ${PROGRESS_LOG}"
  echo " Status         : ${STATUS_JSONL}"
  echo "=================================================================="

  # Determine Case
  local profiler_smoke="${OUTPUT_ROOT}/stage0_25_profiler_smoke/profiler_smoke_status.json"
  python3 - "${profiler_smoke}" "${NSYS_AVAILABLE}" "${NCU_AVAILABLE}" <<'PY'
import json, sys
from pathlib import Path
smoke_file, nsys_av, ncu_av = sys.argv[1:4]
try:
    s = json.loads(Path(smoke_file).read_text())
except:
    s = {}
nsys = s.get("nsys_smoke","?")
ncu = s.get("ncu_smoke","?")
print(f"\n  Smoke Test Result:")
print(f"  - cuda_smoke  : {s.get('cuda_smoke','?')}")
print(f"  - nsys_smoke  : {nsys}")
print(f"  - nsys_stats  : {s.get('nsys_stats','?')}")
print(f"  - ncu_smoke   : {ncu}")
ncu_fail = s.get("ncu_fail_reason","")
if ncu_fail:
    print(f"  - ncu_reason  : {ncu_fail}")
print()
if nsys == "PASS" and ncu == "PASS":
    print("  ► Case A: nsys + ncu PASS → All stages including Stage 9 NCU will run")
elif nsys == "PASS" and ("PERMISSION" in ncu or "FAIL" in ncu):
    print("  ► Case B: nsys PASS, ncu FAIL_PERMISSION → Stage 9 NCU will be SKIPPED")
elif "FAIL" in nsys:
    print("  ► Case C: nsys FAIL → Only app-level timing stages will run in Phase 2")
else:
    print(f"  ► Status: nsys={nsys} ncu={ncu}")
PY
}

main "$@"
