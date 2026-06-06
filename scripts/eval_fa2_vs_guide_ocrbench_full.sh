#!/usr/bin/env bash
# FA2 vs GUIDE — OCRBench v2 full eval with task-specific max_new_tokens
# GPU 0: FA2   shard 0/2
# GPU 1: FA2   shard 1/2
# GPU 2: GUIDE shard 0/2
# GPU 3: GUIDE shard 1/2
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-qwen25vl-guide:latest}"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
DATA_ROOT="${PROJECT_ROOT}/data/raw"
FA2_OUT="${PROJECT_ROOT}/output/qwen25vl_fa2_ocrbench"
GUIDE_OUT="${PROJECT_ROOT}/output/qwen25vl_guide_ocrbench"
c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
c_data="${DATA_ROOT/${PROJECT_ROOT}/\/workspace}"
c_fa2="${FA2_OUT/${PROJECT_ROOT}/\/workspace}"
c_guide="${GUIDE_OUT/${PROJECT_ROOT}/\/workspace}"
mkdir -p "${FA2_OUT}" "${GUIDE_OUT}"

ts() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "${FA2_OUT}/master.log"; }

log "================================================================"
log " OCRBench v2 Full Eval (task-specific max_tokens)"
log " FA2 vs GUIDE — 10,000 samples, 2-way shard per mode"
log "================================================================"

run_shard() {
  local gpu="$1" attn="$2" c_out="$3" shard_idx="$4"
  local mounts=("-v" "${PROJECT_ROOT}:/workspace")
  docker run --rm --gpus "device=${gpu}" --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" "${mounts[@]}" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
    -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton -e XDG_CACHE_HOME=/tmp/.cache \
    -w /workspace "${IMAGE}" -c "
python3 -u /workspace/tools/eval_ocrbench_full.py \
  --model-path  '${c_model}' \
  --data-root   '${c_data}' \
  --output-dir  '${c_out}' \
  --precision   bf16 \
  --attn-impl   ${attn} \
  --shard-idx   ${shard_idx} \
  --num-shards  2 \
  --num-samples 0" \
  2>&1 | grep -v "Warning\|deprecat\|fast processor\|Loading checkpoint\|UserWarning"
}

( log "[gpu0] FA2   shard 0/2 START"
  run_shard 0 fa2   "${c_fa2}/s0"   0 | tee "${FA2_OUT}/s0.log"
  log "[gpu0] FA2   shard 0/2 DONE" ) &  PID0=$!

( log "[gpu1] FA2   shard 1/2 START"
  run_shard 1 fa2   "${c_fa2}/s1"   1 | tee "${FA2_OUT}/s1.log"
  log "[gpu1] FA2   shard 1/2 DONE" ) &  PID1=$!

( log "[gpu2] GUIDE shard 0/2 START"
  run_shard 2 guide "${c_guide}/s0" 0 | tee "${GUIDE_OUT}/s0.log"
  log "[gpu2] GUIDE shard 0/2 DONE" ) &  PID2=$!

( log "[gpu3] GUIDE shard 1/2 START"
  run_shard 3 guide "${c_guide}/s1" 1 | tee "${GUIDE_OUT}/s1.log"
  log "[gpu3] GUIDE shard 1/2 DONE" ) &  PID3=$!

log "Waiting for all 4 workers..."
wait $PID0; E0=$?; wait $PID1; E1=$?
wait $PID2; E2=$?; wait $PID3; E3=$?
log "All done: $E0 $E1 $E2 $E3"

# ── Merge & report ────────────────────────────────────────────────────────────
python3 - "${FA2_OUT}" "${GUIDE_OUT}" \
  "${PROJECT_ROOT}/output/qwen25vl_baseline/ocrbench_v2/ocrbench_v2_summary.json" \
  "${PROJECT_ROOT}/results/fa2_vs_guide_accuracy_full.json" <<'PY'
import sys, json
from pathlib import Path
from collections import defaultdict

fa2_root   = Path(sys.argv[1])
guide_root = Path(sys.argv[2])
base_path  = Path(sys.argv[3])
full_path  = Path(sys.argv[4])

base = json.loads(base_path.read_text()) if base_path.exists() else {}

def merge(root):
    preds = []
    for p in sorted(root.rglob("*predictions.jsonl")):
        preds += [json.loads(l) for l in open(p)]
    if not preds: return None, None, 0, {}
    n = len(preds)
    em = round(sum(1 if r.get("correct") else 0 for r in preds) / n, 4)
    by_type = defaultdict(lambda: [0, 0])
    for r in preds:
        by_type[r.get("type","?")][1] += 1
        if r.get("correct"): by_type[r.get("type","?")][0] += 1
    by_type_acc = {t: round(v[0]/v[1], 4) for t, v in by_type.items() if v[1] > 0}
    return em, None, n, by_type_acc

fa2_em,   _, fa2_n,  fa2_by   = merge(fa2_root)
guide_em, _, guide_n, guide_by = merge(guide_root)
base_em = base.get("exact_match")
base_n  = base.get("num_samples", 500)

print()
print("="*72)
print("  OCRBench v2 (task-specific max_tokens): SDPA vs FA2 vs GUIDE")
print("="*72)
print(f"  {'metric':<16}  {'SDPA(n=500)':>12}  {'FA2(n=10K)':>12}  {'GUIDE(n=10K)':>13}  diff")
print("  " + "-"*65)
diff = f"{guide_em-fa2_em:+.4f}" if (fa2_em and guide_em) else "N/A"
print(f"  {'exact_match':<16}  {base_em or 'N/A':>12}  {fa2_em or 'N/A':>12}  {guide_em or 'N/A':>13}  {diff}")
print()

# By-type comparison
all_types = sorted(set(fa2_by) | set(guide_by))
print(f"  {'task type':<42}  {'FA2':>6}  {'GUIDE':>6}  {'diff':>6}")
print("  " + "-"*60)
for t in all_types:
    f = fa2_by.get(t); g = guide_by.get(t)
    if f is None or g is None: continue
    d = f"{g-f:+.3f}"
    print(f"  {t:<42}  {f:>6.3f}  {g:>6.3f}  {d:>6}")
print("="*72)

# Update full result JSON
full = json.loads(full_path.read_text()) if full_path.exists() else {}
full.setdefault("fa2",   {})["ocrbench_em_taskspecific"]   = fa2_em
full.setdefault("guide", {})["ocrbench_em_taskspecific"]   = guide_em
full.setdefault("sdpa_baseline", {})["ocrbench_em_sdpa"]   = base_em
full["fa2"]["n_ocrbench_taskspec"]   = fa2_n
full["guide"]["n_ocrbench_taskspec"] = guide_n
full_path.write_text(json.dumps(full, indent=2))

print()
print("="*72)
print("  COMPLETE ACCURACY SUMMARY: FA2 vs GUIDE (full val sets)")
print("="*72)
print(f"  {'benchmark':<35}  {'FA2':>8}  {'GUIDE':>8}  {'diff':>8}")
print("  " + "-"*62)
for name, fk, gk in [
    ("DocVQA ANLS          (n=5,349)", "docvqa_anls",                "docvqa_anls"),
    ("ChartQA Acc          (n=1,920)", "chartqa_relaxed_acc",        "chartqa_relaxed_acc"),
    ("OCRBench EM/32tok    (n=10K)",   "ocrbench_em",                "ocrbench_em"),
    ("OCRBench EM/task-tok (n=10K)",   "ocrbench_em_taskspecific",   "ocrbench_em_taskspecific"),
]:
    f = full.get("fa2",{}).get(fk); g = full.get("guide",{}).get(gk)
    diff2 = f"{g-f:+.4f}" if (f and g) else "N/A"
    print(f"  {name:<35}  {str(f or 'N/A'):>8}  {str(g or 'N/A'):>8}  {diff2:>8}")
print("="*72)
print(f"\nSaved: {full_path}")
PY
