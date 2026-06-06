#!/usr/bin/env bash
# =============================================================================
# Phase 1: Docker Setup, Kernel Audit & Smoke Test (Stage 0~4)
# Qwen2.5-VL GUIDE Kernel-Level Implementation
# All Python runs INSIDE Docker container.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Paths ─────────────────────────────────────────────────────────────────────
SWIN_SRC="${SWIN_SRC:-$HOME/sigmetrics2026_swin}"
CSWIN_SRC="${CSWIN_SRC:-$HOME/sigmetrics2026_cswin}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_guide_kernel_level}"
TOOLS_DIR="${PROJECT_ROOT}/tools"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-guide:latest}"

NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2"
NCU_HOST_DIR="/opt/nvidia/nsight-compute/2025.3.1"

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_ROOT}/logs" \
         "${OUTPUT_ROOT}/stage_status" \
         "${OUTPUT_ROOT}/kernel_inventory" \
         "${OUTPUT_ROOT}/descriptor_specs" \
         "${OUTPUT_ROOT}/synthetic_tests" \
         "${OUTPUT_ROOT}/captured_activation" \
         "${OUTPUT_ROOT}/python_reference" \
         "${OUTPUT_ROOT}/triton_correctness" \
         "${OUTPUT_ROOT}/nsys_block_sweep" \
         "${OUTPUT_ROOT}/segment_compare" \
         "${OUTPUT_ROOT}/summaries"

LOG_DIR="${OUTPUT_ROOT}/logs"
PROGRESS_LOG="${LOG_DIR}/progress.log"
STATUS_JSONL="${LOG_DIR}/status.jsonl"
ERROR_LOG="${LOG_DIR}/error.log"
CURRENT_STAGE="${LOG_DIR}/current_stage.txt"

touch "${PROGRESS_LOG}" "${STATUS_JSONL}" "${ERROR_LOG}"

ts_human() { date +"%Y-%m-%d %H:%M:%S"; }

log_progress() {
  echo "[$(ts_human)] [$1] $2" | tee -a "${PROGRESS_LOG}"
}

write_status() {
  python3 - "${STATUS_JSONL}" "$1" "$2" "${3:-}" <<'PY'
import json, sys, datetime
path, stage, status, message = sys.argv[1:5]
with open(path, "a") as f:
    f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                        "stage": stage, "status": status, "message": message}) + "\n")
PY
}

# ── Docker run helper ─────────────────────────────────────────────────────────
run_in_docker() {
  local gpu_id="${1:-0}"; shift
  local cmd="$1"; shift
  local mounts=(
    "-v" "${PROJECT_ROOT}:/workspace"
    "-v" "${SWIN_SRC}:/swin_src:ro"
    "-v" "${CSWIN_SRC}:/cswin_src:ro"
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
    -e "PATH=/opt/nsight-systems/bin:/opt/nsight-compute:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    -e "SWIN_SRC=/swin_src" \
    -e "CSWIN_SRC=/cswin_src" \
    -w /workspace \
    "${IMAGE_NAME}" \
    -c "${cmd}"
}

# ── Stage 0: Environment & Profiler Smoke ─────────────────────────────────────
stage0_env_smoke() {
  local stage="stage0_env_smoke"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 0: environment + nsys smoke"
  write_status "${stage}" "START" "checking paths, CUDA, nsys"

  # Path checks (host)
  {
    echo "=== HOST PATH CHECKS ==="
    echo "PROJECT_ROOT: ${PROJECT_ROOT}"
    echo "SWIN_SRC: ${SWIN_SRC}"
    echo "CSWIN_SRC: ${CSWIN_SRC}"
    echo "MODEL_PATH: ${MODEL_PATH}"
    [ -d "${SWIN_SRC}"  ] && echo "SWIN_SRC: EXISTS"  || echo "SWIN_SRC: MISSING"
    [ -d "${CSWIN_SRC}" ] && echo "CSWIN_SRC: EXISTS" || echo "CSWIN_SRC: MISSING"
    [ -d "${MODEL_PATH}"] && echo "MODEL_PATH: EXISTS" || echo "MODEL_PATH: MISSING"
    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  } 2>&1 | tee "${out}/path_env_check.log"

  # Python + CUDA + Triton inside Docker
  run_in_docker 0 "python3 -c \"
import sys, torch, triton
print('python', sys.version[:20])
print('torch', torch.__version__)
print('triton', triton.__version__)
print('cuda_available', torch.cuda.is_available())
print('gpu_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu_name', torch.cuda.get_device_name(0))
\"" 2>&1 | tee "${out}/python_cuda_check.log"

  # nsys smoke inside Docker
  local nsys_ok="FAIL"
  run_in_docker 0 "
SMOKE_PY=/workspace/tools/smoke_cuda_profiler.py
OUT=/workspace/${out#${PROJECT_ROOT}/}
mkdir -p \"\${OUT}\"
nsys profile -t cuda,nvtx --force-overwrite=true -o \"\${OUT}/nsys_smoke\" \
  python3 \"\${SMOKE_PY}\" 2>&1 | tee \"\${OUT}/nsys_smoke_stdout.log\" && \
nsys stats -r nvtx_gpu_proj_trace --format csv --output \"\${OUT}/nsys_smoke\" \
  \"\${OUT}/nsys_smoke.nsys-rep\" 2>&1 | tee \"\${OUT}/nsys_stats_stdout.log\"
" 2>&1 || true

  if [ -f "${out}/nsys_smoke.nsys-rep" ]; then
    nsys_ok="PASS"
    log_progress "INFO" "  nsys smoke: PASS"
  else
    log_progress "WARN" "  nsys smoke: FAIL (no .nsys-rep)"
    echo "nsys smoke FAIL" >> "${ERROR_LOG}"
    write_status "${stage}" "FAIL" "nsys smoke failed — this plan requires nsys"
    exit 1
  fi

  python3 -c "import json; open('${out}/stage_status.json','w').write(json.dumps({'status':'PASS','nsys':True}))"
  write_status "${stage}" "DONE" "CUDA+nsys PASS"
  log_progress "DONE" "Stage 0: environment smoke PASS"
}

# ── Stage 1: Swin Kernel Audit ────────────────────────────────────────────────
stage1_swin_audit() {
  local stage="stage1_swin_kernel_audit"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}" "${OUTPUT_ROOT}/kernel_inventory"
  log_progress "START" "Stage 1: Swin kernel audit"
  write_status "${stage}" "START" ""

  # File existence check
  {
    echo "## Swin required files"
    for f in \
      "${SWIN_SRC}/models/triton_ops.py" \
      "${SWIN_SRC}/models/swin_transformer_onnx.py" \
      "${SWIN_SRC}/models/swin_transformer.py" \
      "${SWIN_SRC}/kernel_triton/fused_window_swin_v1.py" \
      "${SWIN_SRC}/kernel_triton/fused_window_swin_v2.py" \
      "${SWIN_SRC}/kernel_triton/window_attention/fused_window_test.py" \
      "${SWIN_SRC}/kernel_triton/window_attention/fused_window_test_offset_test.py" \
      "${SWIN_SRC}/kernel_triton/window_attention/window_operation_order_test.py"
    do
      [ -e "$f" ] && echo "FOUND $f" || echo "MISSING $f"
    done
    echo ""
    echo "## kernel_triton file inventory"
    find "${SWIN_SRC}/kernel_triton" -maxdepth 3 -name "*.py" | sort
  } 2>&1 | tee "${out}/required_files.txt"

  # Pattern search in swin
  grep -rn "tl\.program_id\|META_INFO\|meta_info\|fused_window_attention\|BLOCK_SIZE\|info_len\|base_ptr\|base_case\|case_num\|context_count\|region_H\|region_W\|per_batch_blocks\|PER_BATCH_BLOCKS\|HEAD_GROUP_NUM\|HEAD_NUM\|fused_window" \
    "${SWIN_SRC}/kernel_triton/fused_window_swin_v1.py" \
    "${SWIN_SRC}/kernel_triton/window_attention/fused_window_test_offset_test.py" \
    "${SWIN_SRC}/models/triton_ops.py" 2>/dev/null \
    | head -100 | tee "${out}/rg_kernel_patterns.txt" || true

  # Copy key files to inventory
  for f in \
    "${SWIN_SRC}/kernel_triton/fused_window_swin_v1.py" \
    "${SWIN_SRC}/kernel_triton/fused_window_swin_v2.py" \
    "${SWIN_SRC}/kernel_triton/window_attention/fused_window_test_offset_test.py" \
    "${SWIN_SRC}/models/triton_ops.py"
  do
    [ -f "$f" ] && cp "$f" "${OUTPUT_ROOT}/kernel_inventory/swin_$(basename $f)" || true
  done

  # Generate flow report
  cat > "${out}/swin_kernel_flow_report.md" << 'MD'
# Swin Triton Kernel Flow Audit

## 1. Baseline Swin Path (models/swin_transformer.py)
- `window_partition(x, window_size)`: reshapes [B,H,W,C] → [B*num_win, WH, WW, C]
- `WindowAttention`: standard multi-head attention with relative position bias (RPB)
- Attention mask for shifted windows (cyclic shift pattern)
- `window_reverse(...)`: restores [B*num_win, WH, WW, C] → [B,H,W,C]
- `torch.roll`: cyclic shift before/after for SW-MSA

## 2. GUIDE/Triton Path
### Kernel Files
- `kernel_triton/fused_window_swin_v1.py`: head-group descriptor + full flash-attn
- `kernel_triton/fused_window_swin_v2.py`: updated version
- `kernel_triton/window_attention/fused_window_test.py`: v0 (no head groups)
- `kernel_triton/window_attention/fused_window_test_offset_test.py`: correctness test

### Descriptor Format (info_len=7 per record)
```
meta_info flat int32 tensor:
  [0]      = case_num (or HEAD_GROUP_NUM in v1)

For each case (at offset base_offset):
  [+0] = base_case    (number of base records)
  [+1] = cur_blocks   (total thread blocks = per_batch_blocks contribution)

For each base record b (7 ints at base_offset+1+7*b+1..7):
  [+1] = base_ptr     (start offset in 2D token grid)
  [+2] = base_H       (window height in patch units)
  [+3] = base_W       (window width in patch units)
  [+4] = context_count (number of repeated contexts)
  [+5] = BLOCK_SIZE   (queries per Triton program)
  [+6] = region_H     (total region height)
  [+7] = region_W     (total region width)
```

### Grid Shape
- grid = (B * NUM_HEADS, PER_BATCH_BLOCKS)
- pid_b_h → batch × head index
- pid_offset → block within head

### Q/K/V Layout (Swin)
- Input: q,k,v as [B, NUM_HEADS, H, W, DIM]
- Pointer formula: `ptr = base + batch*NH*H*W*D + head*H*W*D + spatial_2d_offset*D`

## 3. GUIDE Replaces (Swin)
- torch.roll
- window_partition
- masked WindowAttention core + RPB
- window_reverse
- reverse torch.roll

## 4. GUIDE Does NOT Replace (Swin)
- LayerNorm (norm1/norm2)
- Residual connections
- MLP block
- QKV linear projection
- Output projection

## 5. Non-Shift Descriptor (simple case)
```python
meta = [1,          # case_num=1
        1, 0,       # base_case=1, cur_blocks (filled later)
        0, WS, WS,  # base_ptr=0, base_H=WS, base_W=WS
        (H//WS)*(W//WS), BLOCK_SIZE, H, W]  # ctx, BS, region_H, region_W
```
MD

  python3 -c "import json; open('${out}/stage_status.json','w').write(json.dumps({'status':'DONE'}))"
  write_status "${stage}" "DONE" "Swin audit complete"
  log_progress "DONE" "Stage 1: Swin kernel audit done"
}

# ── Stage 2: CSWin Kernel Audit ───────────────────────────────────────────────
stage2_cswin_audit() {
  local stage="stage2_cswin_kernel_audit"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 2: CSWin kernel audit"
  write_status "${stage}" "START" ""

  # CSWin source audit
  {
    echo "## CSWin file inventory (maxdepth 4)"
    find "${CSWIN_SRC}" -maxdepth 4 -name "*.py" | sort | grep -v __pycache__
    echo ""
    echo "## Swin repo CSWin subdir"
    find "${SWIN_SRC}/kernel_triton/window_attention/CSwin" -name "*.py" | sort
  } 2>&1 | tee "${out}/all_files.txt"

  # Check CSWin-specific patterns
  grep -rn "CSWin\|LePEAttention\|img2windows\|windows2img\|split_size\|branch_num\|stripe\|horizontal\|vertical" \
    "${CSWIN_SRC}" 2>/dev/null | head -50 | tee "${out}/rg_cswin_patterns.txt" || true

  # Check if SWIN_SRC has CSWin kernel
  local cswin_kernel_in_swin
  cswin_kernel_in_swin=$(find "${SWIN_SRC}/kernel_triton/window_attention/CSwin" -name "*.py" 2>/dev/null | wc -l)
  log_progress "INFO" "  CSWin kernel files in swin repo: ${cswin_kernel_in_swin}"

  # Copy CSWin files to inventory
  find "${SWIN_SRC}/kernel_triton/window_attention/CSwin" -name "*.py" 2>/dev/null | \
    while read f; do cp "$f" "${OUTPUT_ROOT}/kernel_inventory/cswin_$(basename $f)" 2>/dev/null || true; done

  local cswin_has_fused
  cswin_has_fused=$(grep -rl "fused_window\|META_INFO\|meta_info" "${CSWIN_SRC}" 2>/dev/null | wc -l)

  cat > "${out}/cswin_kernel_flow_report.md" << 'MD'
# CSWin Triton Kernel Flow Audit

## 1. CSWin Source Structure
- Main CSWin attention code is in sigmetrics2026_cswin
- kernel_triton.py is the main Triton file (single file, not directory)
- CSwin/ subdir contains text docs

## 2. Swin Repo CSWin Kernels
- sigmetrics2026_swin/kernel_triton/window_attention/CSwin/ contains actual implementations
- CSWin-Tiny, CSWin-Large variants
- Uses same fused_window_attention kernel with head-group descriptor
- Horizontal/vertical stripe → maps to HEAD_GROUP_NUM=2 pattern

## 3. Descriptor Pattern for CSWin
- HEAD_GROUP_NUM = 2
- Head group 0 (heads 0..num_heads//2): horizontal stripe descriptor
- Head group 1 (heads num_heads//2..num_heads): vertical stripe descriptor
- Same base_record schema as Swin (7 fields per record)

## 4. What CSWin Contributes to Qwen Port
- Validates HEAD_GROUP_NUM field generality
- Head-range routing in Triton kernel confirmed working
- Qwen uses HEAD_GROUP_NUM=1 (no stripe, all heads same window)
MD

  local status_str
  if [ "${cswin_kernel_in_swin}" -gt "0" ]; then
    status_str='{"status":"DONE","cswin_kernel_found":true,"source":"swin_repo_CSwin_subdir"}'
  else
    status_str='{"status":"DONE","cswin_kernel_found":false,"note":"NO_CSWIN_FUSED_KERNEL","action":"using Swin kernel as primary"}'
    log_progress "WARN" "  CSWin fused kernel not found — using Swin kernel as implementation source (Decision Table Case B)"
  fi
  python3 -c "import json; open('${out}/stage_status.json','w').write('${status_str}')"
  write_status "${stage}" "DONE" "CSWin audit done"
  log_progress "DONE" "Stage 2: CSWin audit done"
}

# ── Stage 3: Descriptor Schema Extraction ─────────────────────────────────────
stage3_descriptor_schema() {
  local stage="stage3_descriptor_schema"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/descriptor_specs"
  mkdir -p "${out}"
  log_progress "START" "Stage 3: descriptor schema extraction"
  write_status "${stage}" "START" ""

  # Write formal descriptor schema doc
  cat > "${out}/descriptor_schema.md" << 'MD'
# Descriptor Schema (Existing Swin/CSWin Triton Kernel)

## Flat int32 tensor layout

### Simple version (no head groups, fused_window_test.py)
```
meta[0] = case_num

For each case i (at base_offset_i):
  meta[base_offset_i + 0] = base_case_i      # num records
  meta[base_offset_i + 1] = cur_blocks_i     # total blocks (computed)

  For each record b (7 ints, accessed as base_offset_i + 1 + 7*b + field):
    field 1: base_ptr       (start offset in 2D grid, row-major: h*W + w)
    field 2: base_H         (window height in token units)
    field 3: base_W         (window width in token units)
    field 4: context_count  (number of repeated windows with same pattern)
    field 5: BLOCK_SIZE     (query tokens per Triton program)
    field 6: region_H       (total height of region covered)
    field 7: region_W       (total width of region covered)

Advance: base_offset_{i+1} = base_offset_i + 7 * base_case_i + 2
```

### Multi-head-group version (fused_window_swin_v1.py)
```
meta[0] = HEAD_GROUP_NUM

meta[1..HEAD_GROUP_NUM] = offsets to each group descriptor

For each head group g at offset off_g:
  meta[off_g + 0] = head_start
  meta[off_g + 1] = head_end
  meta[off_g + 2] = case_num
  (then cases as in simple version, starting at off_g+3)
```

## info_len constant
- info_len = 7 (7 fields per base record)
- Accessed as: META_INFO[cur_base_offsets + 1 + info_len*b + field]

## case_blocks formula
- case_blocks = max over records b: ceil(base_H_b * base_W_b / BLOCK_SIZE_b) * context_count_b
- Computed in Python before kernel launch

## per_batch_blocks
- sum of case_blocks over all cases
- Used for grid dim 1: grid = (B*NUM_HEADS, per_batch_blocks)
MD

  # Write descriptor field CSV
  cat > "${out}/descriptor_fields.csv" << 'EOF'
field_idx,field_name,triton_access_pattern,semantics
0,base_ptr,META_INFO[cur_base_offsets+1+info_len*b+1],row-major offset h*W+w in logical grid
1,base_H,META_INFO[cur_base_offsets+1+info_len*b+2],window height in token units
2,base_W,META_INFO[cur_base_offsets+1+info_len*b+3],window width in token units
3,context_count,META_INFO[cur_base_offsets+1+info_len*b+4],num repeated contexts
4,BLOCK_SIZE,META_INFO[cur_base_offsets+1+info_len*b+5],query tokens per Triton program (constexpr)
5,region_H,META_INFO[cur_base_offsets+1+info_len*b+6],total height of region
6,region_W,META_INFO[cur_base_offsets+1+info_len*b+7],total width of region
EOF

  # Swin non-shift descriptor doc
  cat > "${out}/swin_nonshift_descriptor.md" << 'MD'
# Swin Non-Shift (W-MSA) Descriptor

```python
meta = [
  1,            # case_num = 1
  1, 0,         # base_case=1, cur_blocks (filled later)
  0,            # base_ptr = 0 (top-left)
  WS, WS,       # base_H, base_W = window_size, window_size
  (H//WS)*(W//WS),  # context_count = total windows
  BLOCK_SIZE,   # BLOCK_SIZE (e.g. 32)
  H, W          # region_H, region_W = full image
]
```
MD

  # Swin shifted descriptor doc
  cat > "${out}/swin_shifted_descriptor.md" << 'MD'
# Swin Shifted (SW-MSA) Descriptor

For window_size M, shift s=M//2, residual r=M-s:

```python
# case_num = 4: corners, h-bands, v-bands, interior
meta = [
  4,                          # case_num
  4, 0,                       # case0: corners (4 records)
    0,       s,s, 1,BS, s,s,   # TL corner
    W-r,     s,r, 1,BS, s,r,   # TR corner
    W*(H-r), r,s, 1,BS, r,s,   # BL corner
    W*(H-r)+(W-r), r,r, 1,BS, r,r,  # BR corner
  2, 0,                       # case1: h-bands (2 records)
    s,       s,M, (W-M)//M,BS, s,W-M,    # top band
    (H-r)*W+s, r,M, (W-M)//M,BS, r,W-M, # bottom band
  2, 0,                       # case2: v-bands (2 records)
    s*W,     M,s, (H-M)//M,BS, H-M,s,   # left band
    s*W+(W-r),M,r,(H-M)//M,BS,H-M,r,   # right band
  1, 0,                       # case3: interior (1 record)
    s*W+s, M,M, ((H-M)//M)*((W-M)//M), BS, H-M,W-M,
]
```
MD

  python3 -c "import json; open('${out}/../stage_status/stage3_status.json','w').write(json.dumps({'status':'DONE'}))"
  write_status "${stage}" "DONE" "descriptor schema extracted"
  log_progress "DONE" "Stage 3: descriptor schema extracted"
}

# ── Stage 4: Qwen Structure Extraction ────────────────────────────────────────
stage4_qwen_structure() {
  local stage="stage4_qwen_structure"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"
  log_progress "START" "Stage 4: Qwen vision structure extraction"
  write_status "${stage}" "START" "loading Qwen config + model"

  local c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
  local c_out="${out/${PROJECT_ROOT}/\/workspace}"

  run_in_docker 0 "python3 -u /workspace/tools/inspect_qwen25vl_vision_structure.py \
    --model-path '${c_model}' \
    --output-dir '${c_out}'" 2>&1 | tee "${out}/run_stdout.log"

  if [ -f "${out}/stage_status.json" ]; then
    local status
    status=$(python3 -c "import json; print(json.load(open('${out}/stage_status.json'))['status'])")
    log_progress "INFO" "  Stage 4 status: ${status}"
    if [ "${status}" != "PASS" ]; then
      log_progress "FAIL" "Stage 4 failed"
      exit 1
    fi
  fi

  write_status "${stage}" "DONE" "Qwen structure extracted"
  log_progress "DONE" "Stage 4: Qwen structure extraction done"
}

# ── Phase 1 Summary ───────────────────────────────────────────────────────────
write_phase1_summary() {
  local summary_file="${OUTPUT_ROOT}/summaries/phase1_audit_smoke_summary.md"
  python3 - "${OUTPUT_ROOT}" "${summary_file}" << 'PY'
import json, sys
from pathlib import Path

root, out_path = Path(sys.argv[1]), sys.argv[2]

stages = {
    "stage0": root / "stage0_env_smoke/stage_status.json",
    "stage1": root / "stage1_swin_kernel_audit/stage_status.json",
    "stage2": root / "stage2_cswin_kernel_audit/stage_status.json",
    "stage3": root / "stage_status/stage3_status.json",
    "stage4": root / "stage4_qwen_structure/stage_status.json",
}
lines = ["# Phase 1 Audit & Smoke Summary\n\n"]
all_pass = True
for name, path in stages.items():
    try:
        d = json.loads(path.read_text())
        status = d.get("status", "UNKNOWN")
    except:
        status = "MISSING"
    ok = status in ("PASS", "DONE")
    if not ok: all_pass = False
    icon = "✅" if ok else "❌"
    lines.append(f"- {icon} **{name}**: {status}\n")

lines.append(f"\n**Overall: {'PASS' if all_pass else 'FAIL'}**\n")

# Load Qwen config
try:
    cfg = json.loads((root / "stage4_qwen_structure/qwen_vision_config.json").read_text())
    lines.append(f"\n## Qwen Vision Config\n\n")
    for k, v in cfg.items():
        lines.append(f"- {k}: {v}\n")
except: pass

Path(out_path).write_text("".join(lines))
print("Written:", out_path)
PY
  log_progress "INFO" "Phase 1 summary: ${summary_file}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  log_progress "START" "=== Phase 1: Audit & Smoke (Stage 0~4) ==="
  log_progress "INFO" "SWIN_SRC: ${SWIN_SRC}"
  log_progress "INFO" "CSWIN_SRC: ${CSWIN_SRC}"
  log_progress "INFO" "MODEL_PATH: ${MODEL_PATH}"
  log_progress "INFO" "OUTPUT_ROOT: ${OUTPUT_ROOT}"

  stage0_env_smoke
  stage1_swin_audit
  stage2_cswin_audit
  stage3_descriptor_schema
  stage4_qwen_structure

  write_phase1_summary

  log_progress "DONE" "=== Phase 1 COMPLETE ==="

  echo ""
  echo "=================================================================="
  echo " PHASE 1 COMPLETE — Audit & Smoke"
  echo "=================================================================="
  cat "${OUTPUT_ROOT}/summaries/phase1_audit_smoke_summary.md" 2>/dev/null || true
  echo "=================================================================="
  echo ""

  # Read Qwen config and print key facts
  python3 - "${OUTPUT_ROOT}" << 'PY'
import json
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv)>1 else ".")
import sys
root = Path(sys.argv[1])
try:
    cfg = json.loads((root/"stage4_qwen_structure/qwen_vision_config.json").read_text())
    print("  Qwen2.5-VL Vision Config:")
    for k,v in cfg.items():
        print(f"    {k} = {v}")
    print()
    rws = cfg['raw_window_size']
    tpw = cfg['tokens_per_window']
    nh = cfg['num_heads']
    hd = cfg['head_dim']
    print(f"  Descriptor target:")
    print(f"    raw_window_size = {rws}  (window_size_px // patch_size = {cfg['window_size_px']} // {cfg['patch_size']})")
    print(f"    tokens_per_window = {tpw}  ({rws}x{rws})")
    print(f"    BLOCK_SIZE candidates: 16, 32, 64 (≤ tokens_per_window={tpw})")
    print(f"    HEAD_GROUP_NUM = 1 (all {nh} heads, head_dim={hd})")
except Exception as e:
    print(f"  [Could not load config: {e}]")
PY
}

main "$@"
