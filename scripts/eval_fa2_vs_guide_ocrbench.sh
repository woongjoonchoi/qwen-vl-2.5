#!/usr/bin/env bash
# FA2 vs GUIDE — OCRBench v2 full evaluation (10,000 samples)
# GPU 0: FA2   shard 0/2 (5,000 samples)
# GPU 1: FA2   shard 1/2 (5,000 samples)
# GPU 2: GUIDE shard 0/2 (5,000 samples)
# GPU 3: GUIDE shard 1/2 (5,000 samples)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-qwen25vl-guide:latest}"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
DATA_ROOT="${PROJECT_ROOT}/data/raw"
FA2_OUT="${PROJECT_ROOT}/output/qwen25vl_fa2_eval_full"
GUIDE_OUT="${PROJECT_ROOT}/output/qwen25vl_guide_eval_full"
c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
c_data="${DATA_ROOT/${PROJECT_ROOT}/\/workspace}"
c_fa2="${FA2_OUT/${PROJECT_ROOT}/\/workspace}"
c_guide="${GUIDE_OUT/${PROJECT_ROOT}/\/workspace}"
mkdir -p "${FA2_OUT}/ocr_s0" "${FA2_OUT}/ocr_s1" \
         "${GUIDE_OUT}/ocr_s0" "${GUIDE_OUT}/ocr_s1"

ts() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "${FA2_OUT}/ocr_master.log"; }

log "================================================================"
log " OCRBench v2 Full Eval: FA2 vs GUIDE (10,000 samples, 2-way shard)"
log "================================================================"

run_shard() {
  local gpu="$1" attn="$2" c_out="$3" shard_idx="$4"
  local mounts=("-v" "${PROJECT_ROOT}:/workspace")
  docker run --rm --gpus "device=${gpu}" --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" "${mounts[@]}" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
    -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton -e XDG_CACHE_HOME=/tmp/.cache \
    -w /workspace "${IMAGE}" -c "
python3 -u /workspace/tools/eval_qwen25vl.py \
  --model-path '${c_model}' --data-root '${c_data}' \
  --output-dir '${c_out}' --precision bf16 \
  --attn-impl ${attn} --max-new-tokens 32 --num-samples 0 \
  --benchmarks ocrbench_v2 \
  --shard-idx ${shard_idx} --num-shards 2" \
  2>&1 | grep -v "Warning\|deprecat\|fast processor\|Loading checkpoint\|UserWarning"
}

( log "[gpu0] FA2   shard 0/2 START"
  run_shard 0 fa2   "${c_fa2}/ocr_s0"   0 | tee "${FA2_OUT}/ocr_s0.log"
  log "[gpu0] FA2   shard 0/2 DONE" ) &
PID0=$!

( log "[gpu1] FA2   shard 1/2 START"
  run_shard 1 fa2   "${c_fa2}/ocr_s1"   1 | tee "${FA2_OUT}/ocr_s1.log"
  log "[gpu1] FA2   shard 1/2 DONE" ) &
PID1=$!

( log "[gpu2] GUIDE shard 0/2 START"
  run_shard 2 guide "${c_guide}/ocr_s0" 0 | tee "${GUIDE_OUT}/ocr_s0.log"
  log "[gpu2] GUIDE shard 0/2 DONE" ) &
PID2=$!

( log "[gpu3] GUIDE shard 1/2 START"
  run_shard 3 guide "${c_guide}/ocr_s1" 1 | tee "${GUIDE_OUT}/ocr_s1.log"
  log "[gpu3] GUIDE shard 1/2 DONE" ) &
PID3=$!

log "Waiting for all 4 workers..."
wait $PID0; E0=$?
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
log "All done: $E0 $E1 $E2 $E3"

# ── Merge & report ────────────────────────────────────────────────────────────
python3 - "${FA2_OUT}" "${GUIDE_OUT}" \
  "${PROJECT_ROOT}/output/qwen25vl_baseline/eval_summary.json" \
  "${PROJECT_ROOT}/results/fa2_vs_guide_accuracy_full.json" <<'PY'
import sys, json
from pathlib import Path

fa2_root   = Path(sys.argv[1])
guide_root = Path(sys.argv[2])
base = json.loads(Path(sys.argv[3]).read_text())
prev = json.loads(Path(sys.argv[4]).read_text())

def merge_ocr(root):
    preds = []
    for shard in ["ocr_s0", "ocr_s1"]:
        for p in (root / shard).rglob("*predictions.jsonl"):
            preds += [json.loads(l) for l in open(p)]
    if not preds: return None, None, 0
    em  = round(sum(r.get("exact_match", 0) for r in preds) / len(preds), 4)
    anls= round(sum(r.get("anls", 0)        for r in preds) / len(preds), 4)
    return em, anls, len(preds)

fa2_em,  fa2_anls,  fa2_n  = merge_ocr(fa2_root)
gui_em,  gui_anls,  gui_n  = merge_ocr(guide_root)
base_em  = base["results"].get("ocrbench_v2", {}).get("exact_match")
base_n   = base["num_samples"]

print()
print("="*72)
print("  OCRBench v2 Full Eval: SDPA-baseline vs FA2 vs GUIDE")
print("="*72)
print(f"  {'metric':<20}  {'SDPA(n=500)':>12}  {'FA2(full)':>12}  {'GUIDE(full)':>12}  diff")
print("  " + "-"*65)
for name, b, f, g in [
    ("exact_match", base_em, fa2_em, gui_em),
    ("ANLS",        None,    fa2_anls, gui_anls),
]:
    bs = f"{b:.4f}" if b else "N/A"
    fs = f"{f:.4f}(n={fa2_n})" if f is not None else "N/A"
    gs = f"{g:.4f}(n={gui_n})" if g is not None else "N/A"
    diff = f"{g-f:+.4f}" if (f is not None and g is not None) else "N/A"
    print(f"  {name:<20}  {bs:>12}  {fs:>12}  {gs:>12}  {diff}")
print("="*72)

# Update result JSON
prev["fa2"]["ocrbench_em"]    = fa2_em;  prev["fa2"]["ocrbench_anls"]   = fa2_anls;  prev["fa2"]["n_ocrbench"]   = fa2_n
prev["guide"]["ocrbench_em"]  = gui_em;  prev["guide"]["ocrbench_anls"] = gui_anls;  prev["guide"]["n_ocrbench"] = gui_n
prev["sdpa_baseline"]["ocrbench_em"] = base_em
Path(sys.argv[4]).write_text(json.dumps(prev, indent=2))

# Full summary
print()
print("="*72)
print("  COMPLETE ACCURACY SUMMARY: FA2 vs GUIDE (full val sets)")
print("="*72)
print(f"  {'benchmark':<28}  {'FA2':>10}  {'GUIDE':>10}  {'diff':>8}")
print("  " + "-"*58)
for name, fk, gk in [
    ("DocVQA ANLS     (n=5,349)", "docvqa_anls",         "docvqa_anls"),
    ("ChartQA Acc     (n=1,920)", "chartqa_relaxed_acc", "chartqa_relaxed_acc"),
    ("OCRBench EM     (n=10K)",   "ocrbench_em",         "ocrbench_em"),
]:
    f = prev["fa2"].get(fk); g = prev["guide"].get(gk)
    diff = f"{g-f:+.4f}" if (f and g) else "N/A"
    print(f"  {name:<28}  {f or 'N/A':>10}  {g or 'N/A':>10}  {diff:>8}")
print("="*72)
PY
