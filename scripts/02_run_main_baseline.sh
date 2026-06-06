#!/usr/bin/env bash
# =============================================================================
# Phase 2: Main Baseline Measurement  (Stage 5 ~ Stage 9)
# Qwen2.5-VL Baseline Measurement
# Requires Phase 1 to have completed successfully.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_baseline}"
TOOLS_DIR="${PROJECT_ROOT}/tools"

PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
WARMUP="${WARMUP:-2}"
MEASURED_ITERS="${MEASURED_ITERS:-5}"
PROFILE_ITERS="${PROFILE_ITERS:-3}"

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PROGRESS_LOG="${LOG_DIR}/progress.log"
STATUS_JSONL="${LOG_DIR}/status.jsonl"
ERROR_LOG="${LOG_DIR}/error.log"
CURRENT_STAGE="${LOG_DIR}/current_stage.txt"

touch "${PROGRESS_LOG}" "${STATUS_JSONL}" "${ERROR_LOG}"

ts_human() { date +"%Y-%m-%d %H:%M:%S"; }

log_progress() {
  local level="$1"; local msg="$2"
  echo "[$(ts_human)] [${level}] ${msg}" | tee -a "${PROGRESS_LOG}"
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

fail_soft() {
  local stage="$1"; local msg="$2"
  log_progress "WARN" "${stage}: ${msg} (continuing)"
  write_status "${stage}" "WARN" "${msg}"
  echo "${msg}" >> "${ERROR_LOG}"
}

# ─── Load Phase 1 state ───────────────────────────────────────────────────────
SMOKE_RESULT_FILE="${OUTPUT_ROOT}/stage_status/profiler_smoke.json"
MANIFEST_PATH="${OUTPUT_ROOT}/stage0_50_data_manifest/benchmark_manifest.jsonl"

NSYS_AVAILABLE="false"
NCU_AVAILABLE="false"
if [ -f "${SMOKE_RESULT_FILE}" ]; then
  NSYS_AVAILABLE=$(python3 -c "import json; d=json.load(open('${SMOKE_RESULT_FILE}')); print(str(d.get('NSYS_AVAILABLE',False)).lower())" 2>/dev/null || echo "false")
  NCU_AVAILABLE=$(python3 -c "import json; d=json.load(open('${SMOKE_RESULT_FILE}')); print(str(d.get('NCU_AVAILABLE',False)).lower())" 2>/dev/null || echo "false")
fi

if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "ERROR: manifest not found at ${MANIFEST_PATH}"
  echo "  Run Phase 1 first: bash scripts/01_run_sanity_and_smoke.sh"
  exit 1
fi

log_progress "INFO" "Phase 2 starting: NSYS=${NSYS_AVAILABLE} NCU=${NCU_AVAILABLE}"
export NSYS_AVAILABLE NCU_AVAILABLE

# ─── Detect available benchmarks ─────────────────────────────────────────────
get_available_benchmarks() {
  python3 -c "
import json
bms = set()
with open('${MANIFEST_PATH}') as f:
    for line in f:
        d = json.loads(line)
        bms.add(d.get('benchmark',''))
print(' '.join(sorted(bms)))
" 2>/dev/null
}

AVAILABLE_BENCHMARKS=$(get_available_benchmarks)
log_progress "INFO" "Available benchmarks: ${AVAILABLE_BENCHMARKS}"

HAS_DOCVQA=false; HAS_CHARTQA=false
echo "${AVAILABLE_BENCHMARKS}" | grep -q "docvqa" && HAS_DOCVQA=true
echo "${AVAILABLE_BENCHMARKS}" | grep -q "chartqa" && HAS_CHARTQA=true

# ─── Run one config and save results ─────────────────────────────────────────
run_config() {
  local benchmark="$1"
  local resolution="$2"
  local max_new_tokens="$3"
  local out_dir="$4"
  local use_nsys="${5:-false}"
  local nsys_out="${6:-}"

  mkdir -p "${out_dir}"

  log_progress "INFO" "  run: bm=${benchmark} res=${resolution} tok=${max_new_tokens} nsys=${use_nsys}"

  python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
    --mode app_timing \
    --model-path "${MODEL_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --benchmark "${benchmark}" \
    --num-samples 5 \
    --resolution "${resolution}" \
    --max-new-tokens "${max_new_tokens}" \
    --batch-size 1 \
    --precision "${PRECISION}" \
    --warmup "${WARMUP}" \
    --measured-iters "${MEASURED_ITERS}" \
    --seed "${SEED}" \
    --output-dir "${out_dir}" \
    2>&1 | tee "${out_dir}/run_stdout.log" || {
    fail_soft "run_config" "failed: bm=${benchmark} res=${resolution} tok=${max_new_tokens}"
    return 0
  }

  # Optional nsys trace
  if [ "${use_nsys}" = "true" ] && [ "${NSYS_AVAILABLE}" = "true" ] && [ -n "${nsys_out}" ]; then
    local NSYS_BIN
    NSYS_BIN=$(which nsys 2>/dev/null || echo "")
    if [ -n "${NSYS_BIN}" ]; then
      local base="qwen25vl3b-${benchmark}-res${resolution}-tok${max_new_tokens}-${PRECISION}"
      mkdir -p "${nsys_out}"
      log_progress "INFO" "    nsys trace: ${base}"
      "${NSYS_BIN}" profile \
        -w true \
        -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        -o "${nsys_out}/${base}" \
        python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
          --mode nsys_profile \
          --model-path "${MODEL_PATH}" \
          --manifest "${MANIFEST_PATH}" \
          --benchmark "${benchmark}" \
          --num-samples 1 \
          --resolution "${resolution}" \
          --max-new-tokens "${max_new_tokens}" \
          --batch-size 1 \
          --precision "${PRECISION}" \
          --warmup "${WARMUP}" \
          --profile-iters "${PROFILE_ITERS}" \
          --seed "${SEED}" \
          --output-dir "${nsys_out}/${base}_run" \
        2>&1 | tee "${nsys_out}/${base}_nsys_stdout.log" || true

      if [ -f "${nsys_out}/${base}.nsys-rep" ]; then
        "${NSYS_BIN}" stats \
          -r nvtx_gpu_proj_trace \
          --format csv \
          --output "${nsys_out}/${base}" \
          "${nsys_out}/${base}.nsys-rep" \
          2>&1 | tee "${nsys_out}/${base}_stats.log" || true

        local nsys_csv="${nsys_out}/${base}_nvtx_gpu_proj_trace.csv"
        if [ -f "${nsys_csv}" ]; then
          python3 -u "${TOOLS_DIR}/stats_qwen_nsys.py" \
            --input "${nsys_csv}" \
            --output "${nsys_out}/${base}_nsys_summary.csv" \
            2>/dev/null || true
        fi
      fi
    fi
  fi
}

# ─── Aggregate stage results ──────────────────────────────────────────────────
aggregate_stage_results() {
  local stage_dir="$1"
  local out_jsonl="$2"
  local out_csv="$3"
  local stage_label="$4"

  python3 - "${stage_dir}" "${out_jsonl}" "${out_csv}" "${stage_label}" <<'PY'
import json, csv, sys
from pathlib import Path

stage_dir, out_jsonl, out_csv, stage_label = sys.argv[1:5]

all_rows = []
for timing_file in sorted(Path(stage_dir).rglob("timing_raw.jsonl")):
    try:
        with open(timing_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["stage"] = stage_label
                    # Infer resolution/max_new_tokens from path if not in row
                    import re
                    path_str = str(timing_file)
                    if not r.get("resolution"):
                        m = re.search(r"res(\d+)", path_str)
                        if m: r["resolution"] = int(m.group(1))
                    if not r.get("max_new_tokens"):
                        m = re.search(r"tok(\d+)", path_str)
                        if m: r["max_new_tokens"] = int(m.group(1))
                    all_rows.append(r)
    except Exception as e:
        print(f"  Warning: {e}")

if not all_rows:
    print(f"  No rows found in {stage_dir}")
    Path(out_jsonl).write_text("")
    Path(out_csv).write_text("stage,benchmark,resolution,max_new_tokens,mean_e2e_ms,mean_f_vision\n")
    return

# Write raw
with open(out_jsonl, "w") as f:
    for r in all_rows:
        f.write(json.dumps({k: v for k, v in r.items() if k != "per_block_ms"}) + "\n")

# Compute per-config summary
from collections import defaultdict
groups = defaultdict(list)
for r in all_rows:
    key = (r.get("benchmark",""), str(r.get("resolution",0)), str(r.get("max_new_tokens",0)))
    groups[key].append(r)

def mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else 0

def get_float(r, k):
    v = r.get(k, 0)
    try: return float(v)
    except: return 0.0

summary_rows = []
metric_keys = ["e2e_ms", "visual_encoder_ms", "visual_merger_ms", "llm_prefill_ms",
               "ttft_ms", "decode_loop_ms", "tpot_ms", "request_output_tok_s",
               "decode_only_tok_s", "f_vision", "f_visual_prefill", "f_decode",
               "num_visual_tokens_before_merger", "num_visual_tokens_after_merger",
               "actual_generated_tokens"]
for (bm, res, tok), rows in sorted(groups.items()):
    row = {"stage": stage_label, "benchmark": bm, "resolution": res, "max_new_tokens": tok,
           "n": len(rows)}
    for mk in metric_keys:
        vals = [get_float(r, mk) for r in rows if get_float(r, mk) > 0]
        row[f"mean_{mk}"] = mean(vals)
    summary_rows.append(row)

if summary_rows:
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  {stage_label}: {len(all_rows)} raw rows, {len(summary_rows)} configs")
PY
}

# ─── Stage 5: Resolution Sweep ───────────────────────────────────────────────
stage5_resolution_sweep() {
  local stage="stage5_resolution_sweep"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  local nsys_out="${out}/selected_nsys"
  mkdir -p "${out}" "${nsys_out}"

  log_progress "START" "Stage 5: resolution sweep"
  write_status "${stage}" "START" "sweeping resolutions 448/896/1344/1792"

  local resolutions=(448 896 1344 1792)
  local max_tok=64
  # nsys only for these specific configs
  local nsys_configs=("docvqa:896" "docvqa:1792" "chartqa:896" "chartqa:1792")

  for bm in docvqa chartqa; do
    for res in "${resolutions[@]}"; do
      local config_out="${out}/${bm}_res${res}_tok${max_tok}"
      local do_nsys="false"
      for nc in "${nsys_configs[@]}"; do
        if [ "${nc}" = "${bm}:${res}" ]; then do_nsys="true"; break; fi
      done
      run_config "${bm}" "${res}" "${max_tok}" "${config_out}" "${do_nsys}" "${nsys_out}"
    done
  done

  # Aggregate results
  aggregate_stage_results "${out}" \
    "${out}/resolution_sweep_raw.jsonl" \
    "${out}/resolution_sweep_summary.csv" \
    "${stage}"

  # Generate report
  python3 - "${out}/resolution_sweep_summary.csv" "${out}/resolution_sweep_report.md" <<'PY'
import csv, sys
from pathlib import Path

csv_path, report_path = sys.argv[1], sys.argv[2]
try:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
except Exception:
    rows = []

with open(report_path, "w") as f:
    f.write("# Stage 5: Resolution Sweep Report\n\n")
    f.write("| benchmark | resolution | tokens_after | mean_e2e_ms | mean_ve_ms | mean_f_vision | mean_tok_s |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r.get('benchmark','')} | {r.get('resolution','')} "
                f"| {r.get('mean_num_visual_tokens_after_merger','')} "
                f"| {r.get('mean_e2e_ms','')} | {r.get('mean_visual_encoder_ms','')} "
                f"| {r.get('mean_f_vision','')} | {r.get('mean_decode_only_tok_s','')} |\n")
PY

  python3 -c "import json; open('${out}/stage_status.json','w').write(json.dumps({'status':'DONE'}))"
  write_status "${stage}" "DONE" "resolution sweep complete"
  log_progress "DONE" "Stage 5: resolution sweep done"
}

# ─── Stage 6: Output Length Sweep ────────────────────────────────────────────
stage6_output_length_sweep() {
  local stage="stage6_output_length_sweep"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  local nsys_out="${out}/selected_nsys"
  mkdir -p "${out}" "${nsys_out}"

  log_progress "START" "Stage 6: output length sweep"
  write_status "${stage}" "START" "sweeping output lengths 32/64/256/1024"

  local res=1792
  local max_toks=(32 64 256 1024)
  local nsys_toks=(64 1024)  # nsys only for these

  for bm in docvqa chartqa; do
    for tok in "${max_toks[@]}"; do
      local config_out="${out}/${bm}_res${res}_tok${tok}"
      local do_nsys="false"
      for nt in "${nsys_toks[@]}"; do
        if [ "${nt}" = "${tok}" ] && [ "${bm}" = "docvqa" ]; then
          do_nsys="true"; break
        fi
      done
      run_config "${bm}" "${res}" "${tok}" "${config_out}" "${do_nsys}" "${nsys_out}"
    done
  done

  aggregate_stage_results "${out}" \
    "${out}/output_length_raw.jsonl" \
    "${out}/output_length_summary.csv" \
    "${stage}"

  python3 - "${out}/output_length_summary.csv" "${out}/output_length_sweep_report.md" <<'PY'
import csv, sys
from pathlib import Path
csv_path, report_path = sys.argv[1], sys.argv[2]
try:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
except:
    rows = []
with open(report_path, "w") as f:
    f.write("# Stage 6: Output Length Sweep Report\n\n")
    f.write("| benchmark | resolution | max_new_tokens | mean_e2e_ms | mean_ttft_ms | mean_tpot_ms | mean_f_decode | mean_tok_s |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r.get('benchmark','')} | {r.get('resolution','')} | {r.get('max_new_tokens','')} "
                f"| {r.get('mean_e2e_ms','')} | {r.get('mean_ttft_ms','')} "
                f"| {r.get('mean_tpot_ms','')} | {r.get('mean_f_decode','')} "
                f"| {r.get('mean_decode_only_tok_s','')} |\n")
PY

  python3 -c "import json; open('${out}/stage_status.json','w').write(json.dumps({'status':'DONE'}))"
  write_status "${stage}" "DONE" "output length sweep complete"
  log_progress "DONE" "Stage 6: output length sweep done"
}

# ─── Stage 7: Five-Workload Baseline Table ────────────────────────────────────
stage7_workload_table() {
  local stage="stage7_workload_table"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"

  log_progress "START" "Stage 7: five-workload baseline table"
  write_status "${stage}" "START" "running all available workloads"

  local res=1344
  local max_tok=64
  local missing_workloads=()

  # Run available benchmarks
  local all_bms=("docvqa" "chartqa" "ocr" "grounding" "visual_chat")

  for bm in "${all_bms[@]}"; do
    local bm_count
    bm_count=$(python3 -c "
import json
n=0
with open('${MANIFEST_PATH}') as f:
    for line in f:
        d=json.loads(line)
        if d.get('benchmark','') == '${bm}': n+=1
print(n)
" 2>/dev/null || echo "0")

    if [ "${bm_count}" -gt 0 ]; then
      log_progress "INFO" "  workload: ${bm} (${bm_count} samples available)"
      run_config "${bm}" "${res}" "${max_tok}" "${out}/${bm}_res${res}_tok${max_tok}" "false" ""
    else
      log_progress "WARN" "  workload: ${bm} MISSING_DATA"
      missing_workloads+=("${bm}")
    fi
  done

  # Save missing workloads
  python3 - "${out}" "${missing_workloads[@]+"${missing_workloads[@]}"}" <<'PY'
import json, sys
out = sys.argv[1]
missing = sys.argv[2:] if len(sys.argv) > 2 else []
with open(f"{out}/missing_workloads.json", "w") as f:
    json.dump(missing, f, indent=2)
PY

  aggregate_stage_results "${out}" \
    "${out}/workload_raw.jsonl" \
    "${out}/workload_summary.csv" \
    "${stage}"

  python3 - "${out}/workload_summary.csv" "${out}/workload_table_report.md" <<'PY'
import csv, sys
from pathlib import Path
csv_path, report_path = sys.argv[1], sys.argv[2]
try:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
except:
    rows = []
with open(report_path, "w") as f:
    f.write("# Stage 7: Five-Workload Baseline Table\n\n")
    f.write("| benchmark | n | mean_ve_ms | mean_ttft_ms | mean_e2e_ms | mean_tpot_ms | mean_f_vision | tok_s |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r.get('benchmark','')} | {r.get('n','')} "
                f"| {r.get('mean_visual_encoder_ms','')} | {r.get('mean_ttft_ms','')} "
                f"| {r.get('mean_e2e_ms','')} | {r.get('mean_tpot_ms','')} "
                f"| {r.get('mean_f_vision','')} | {r.get('mean_decode_only_tok_s','')} |\n")
PY

  python3 -c "import json; open('${out}/stage_status.json','w').write(json.dumps({'status':'DONE'}))"
  write_status "${stage}" "DONE" "workload table complete"
  log_progress "DONE" "Stage 7: workload table done"
}

# ─── Stage 8: Visual Encoder Breakdown ───────────────────────────────────────
stage8_visual_encoder_breakdown() {
  local stage="stage8_visual_encoder_breakdown"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"

  log_progress "START" "Stage 8: visual encoder breakdown (docvqa res=1792 tok=64)"
  write_status "${stage}" "START" "visual encoder breakdown"

  local config_out="${out}/docvqa_res1792_tok64"
  mkdir -p "${config_out}"

  python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
    --mode app_timing \
    --model-path "${MODEL_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --benchmark "docvqa" \
    --num-samples 5 \
    --resolution 1792 \
    --max-new-tokens 64 \
    --batch-size 1 \
    --precision "${PRECISION}" \
    --warmup "${WARMUP}" \
    --measured-iters "${MEASURED_ITERS}" \
    --seed "${SEED}" \
    --output-dir "${config_out}" \
    2>&1 | tee "${config_out}/run_stdout.log" || {
    fail_soft "${stage}" "visual encoder breakdown run failed"
    return 0
  }

  # Extract per-block timing from timing_raw.jsonl
  python3 - "${config_out}" "${out}" <<'PY'
import json, csv, sys
from pathlib import Path
from collections import defaultdict

config_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
all_blocks = defaultdict(list)
all_top = defaultdict(list)

for line in open(config_dir / "timing_raw.jsonl"):
    r = json.loads(line.strip())
    for bname, bms in r.get("per_block_ms", {}).items():
        if bms and bms > 0:
            all_blocks[bname].append(bms)
    for key in ["visual_encoder_ms", "visual_merger_ms", "patch_embed_ms",
                "window_attn_total_ms", "full_attn_total_ms", "llm_prefill_ms",
                "decode_loop_ms", "e2e_ms"]:
        v = r.get(key, -1)
        if v and v > 0:
            all_top[key].append(v)

def mean(vs): return round(sum(vs)/len(vs), 4) if vs else -1

# Write breakdown CSV
rows = []
for key, vals in all_top.items():
    rows.append({"component": key, "mean_ms": mean(vals), "n": len(vals)})
with open(out_dir / "visual_encoder_breakdown_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["component", "mean_ms", "n"])
    w.writeheader()
    w.writerows(rows)

# Per-block CSV
block_rows = []
for bname, vals in sorted(all_blocks.items()):
    block_rows.append({"block": bname, "mean_ms": mean(vals), "n": len(vals)})
if block_rows:
    with open(out_dir / "per_block_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "mean_ms", "n"])
        w.writeheader()
        w.writerows(block_rows)

# Window vs full attention summary
wa_keys = [k for k in all_blocks if "win" in k]
fa_keys = [k for k in all_blocks if "full" in k]
wa_ms = mean([v for k in wa_keys for v in all_blocks[k]])
fa_ms = mean([v for k in fa_keys for v in all_blocks[k]])
with open(out_dir / "window_vs_full_attention_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["type", "mean_ms_per_block", "num_blocks"])
    w.writeheader()
    w.writerows([
        {"type": "window_attention", "mean_ms_per_block": wa_ms, "num_blocks": len(wa_keys)},
        {"type": "full_attention", "mean_ms_per_block": fa_ms, "num_blocks": len(fa_keys)},
    ])

# Write breakdown raw (copy from config_dir)
import shutil
raw_src = config_dir / "timing_raw.jsonl"
raw_dst = out_dir / "visual_encoder_breakdown_raw.jsonl"
if raw_src.exists():
    shutil.copy(str(raw_src), str(raw_dst))

# Report
ve_ms = mean(all_top.get("visual_encoder_ms", []))
vm_ms = mean(all_top.get("visual_merger_ms", []))
wa_total = mean(all_top.get("window_attn_total_ms", []))
fa_total = mean(all_top.get("full_attn_total_ms", []))
e2e_ms = mean(all_top.get("e2e_ms", []))

with open(out_dir / "stage_status.json", "w") as f:
    json.dump({"status": "DONE",
               "visual_encoder_ms": ve_ms, "visual_merger_ms": vm_ms,
               "window_attn_total_ms": wa_total, "full_attn_total_ms": fa_total}, f, indent=2)

print(f"  visual_encoder_ms: {ve_ms}")
print(f"  visual_merger_ms:  {vm_ms}")
print(f"  window_attn_total: {wa_total}")
print(f"  full_attn_total:   {fa_total}")
print(f"  e2e_ms:            {e2e_ms}")
PY

  write_status "${stage}" "DONE" "visual encoder breakdown complete"
  log_progress "DONE" "Stage 8: visual encoder breakdown done"
}

# ─── Stage 9: NCU Analysis ────────────────────────────────────────────────────
stage9_ncu_analysis() {
  local stage="stage9_ncu_analysis"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"

  log_progress "START" "Stage 9: Nsight Compute analysis"
  write_status "${stage}" "START" "ncu analysis"

  if [ "${NCU_AVAILABLE}" != "true" ]; then
    log_progress "WARN" "  Stage 9: NCU_AVAILABLE=false, skipping"
    python3 -c "import json; open('${out}/stage_status.json','w').write(
      json.dumps({'status':'SKIPPED_NCU_UNAVAILABLE','reason':'NCU_AVAILABLE=false'}))"
    write_status "${stage}" "SKIPPED" "NCU_AVAILABLE=false"
    return 0
  fi

  local NCU_BIN
  NCU_BIN=$(which ncu 2>/dev/null || echo "")
  if [ -z "${NCU_BIN}" ]; then
    log_progress "WARN" "  Stage 9: ncu not in PATH, skipping"
    python3 -c "import json; open('${out}/stage_status.json','w').write(
      json.dumps({'status':'SKIPPED_NCU_NOT_FOUND'}))"
    write_status "${stage}" "SKIPPED" "ncu not in PATH"
    return 0
  fi

  # Run ncu for representative target: window_attn at block 8
  local layer=8
  local base="qwen25vl3b-docvqa-res1792-layer${layer}-window_attn"
  local nvtx_path="regex:.*vit\.blocks\.08.*"

  local ncu_out
  ncu_out=$("${NCU_BIN}" \
    --set full \
    --nvtx \
    --nvtx-include "${nvtx_path}" \
    --target-processes all \
    --force-overwrite \
    -o "${out}/${base}" \
    python3 -u "${TOOLS_DIR}/profile_qwen25vl_baseline.py" \
      --mode nsys_profile \
      --model-path "${MODEL_PATH}" \
      --manifest "${MANIFEST_PATH}" \
      --benchmark "docvqa" \
      --num-samples 1 \
      --resolution 1792 \
      --max-new-tokens 64 \
      --batch-size 1 \
      --precision "${PRECISION}" \
      --warmup "${WARMUP}" \
      --profile-iters 1 \
      --seed "${SEED}" \
      --output-dir "${out}/${base}_run" \
    2>&1 | tee "${out}/${base}_ncu_stdout.log"; true)

  if grep -qiE "ERR_NVGPUCTRPERM|permission" "${out}/${base}_ncu_stdout.log" 2>/dev/null; then
    python3 -c "import json; open('${out}/stage_status.json','w').write(
      json.dumps({'status':'SKIPPED_NCU_PERMISSION','reason':'ERR_NVGPUCTRPERM'}))"
    write_status "${stage}" "SKIPPED" "NCU permission error"
    log_progress "WARN" "  Stage 9: NCU permission error → SKIPPED_NCU_PERMISSION"
    return 0
  fi

  if [ -f "${out}/${base}.ncu-rep" ]; then
    log_progress "INFO" "  ncu-rep created: ${base}.ncu-rep"
    # Export CSV summary
    "${NCU_BIN}" --csv \
      --page details \
      "${out}/${base}.ncu-rep" \
      > "${out}/${base}_ncu_details.csv" 2>/dev/null || true
    python3 -c "import json; open('${out}/stage_status.json','w').write(
      json.dumps({'status':'DONE','ncu_rep':'${base}.ncu-rep'}))"
  else
    python3 -c "import json; open('${out}/stage_status.json','w').write(
      json.dumps({'status':'PARTIAL','message':'ncu-rep not created'}))"
  fi

  write_status "${stage}" "DONE" "ncu analysis complete"
  log_progress "DONE" "Stage 9: ncu analysis done"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  log_progress "START" "=== Phase 2: Main Baseline Measurement (Stage 5-9) ==="
  log_progress "INFO" "NSYS_AVAILABLE=${NSYS_AVAILABLE} NCU_AVAILABLE=${NCU_AVAILABLE}"

  stage5_resolution_sweep
  stage6_output_length_sweep
  stage7_workload_table
  stage8_visual_encoder_breakdown
  stage9_ncu_analysis

  log_progress "DONE" "=== Phase 2 COMPLETE ==="
  echo ""
  echo "=================================================================="
  echo " PHASE 2 COMPLETE"
  echo " Results: ${OUTPUT_ROOT}"
  echo " Stages: 5-9"
  echo "=================================================================="
}

main "$@"
