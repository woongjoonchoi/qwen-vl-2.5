#!/usr/bin/env bash
# FA2 vs GUIDE — FULL validation set accuracy
# DocVQA val: 5,349 samples  (2-way sharded per mode)
# ChartQA val: full set
# 4 GPUs 동시 실행:
#   GPU 0 → FA2   DocVQA shard 0/2
#   GPU 1 → FA2   DocVQA shard 1/2, then FA2 ChartQA full
#   GPU 2 → GUIDE DocVQA shard 0/2
#   GPU 3 → GUIDE DocVQA shard 1/2, then GUIDE ChartQA full
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
mkdir -p "${FA2_OUT}" "${GUIDE_OUT}"

ts() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "${FA2_OUT}/master.log" "${GUIDE_OUT}/master.log"; }

log "================================================================"
log " Full Validation: FA2 vs GUIDE"
log " DocVQA: 5,349 samples (2-way shard)  ChartQA: full"
log "================================================================"

run_docker() {
  local gpu="$1" attn="$2" c_out="$3" extra="$4"
  local mounts=("-v" "${PROJECT_ROOT}:/workspace")
  docker run --rm --gpus "device=${gpu}" --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" "${mounts[@]}" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
    -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton -e XDG_CACHE_HOME=/tmp/.cache \
    -w /workspace "${IMAGE}" -c "
python3 -u /workspace/tools/eval_qwen25vl.py \
  --model-path '${c_model}' --data-root '${c_data}' \
  --output-dir '${c_out}' --precision bf16 \
  --attn-impl ${attn} --max-new-tokens 32 --num-samples 0 ${extra}" \
  2>&1 | grep -v "Warning\|deprecat\|fast processor\|Loading checkpoint\|UserWarning"
}

# GPU 0: FA2 DocVQA shard 0/2
(
  log "[gpu0] FA2 DocVQA shard 0/2 START"
  run_docker 0 fa2 "${c_fa2}/docvqa_s0" \
    "--benchmarks docvqa --shard-idx 0 --num-shards 2" \
    | tee "${FA2_OUT}/docvqa_s0.log"
  log "[gpu0] FA2 DocVQA shard 0/2 DONE"
) &
PID0=$!

# GPU 1: FA2 DocVQA shard 1/2, then ChartQA
(
  log "[gpu1] FA2 DocVQA shard 1/2 START"
  run_docker 1 fa2 "${c_fa2}/docvqa_s1" \
    "--benchmarks docvqa --shard-idx 1 --num-shards 2" \
    | tee "${FA2_OUT}/docvqa_s1.log"
  log "[gpu1] FA2 DocVQA shard 1/2 DONE → ChartQA START"
  run_docker 1 fa2 "${c_fa2}/chartqa" \
    "--benchmarks chartqa" \
    | tee "${FA2_OUT}/chartqa.log"
  log "[gpu1] FA2 ChartQA DONE"
) &
PID1=$!

# GPU 2: GUIDE DocVQA shard 0/2
(
  log "[gpu2] GUIDE DocVQA shard 0/2 START"
  run_docker 2 guide "${c_guide}/docvqa_s0" \
    "--benchmarks docvqa --shard-idx 0 --num-shards 2" \
    | tee "${GUIDE_OUT}/docvqa_s0.log"
  log "[gpu2] GUIDE DocVQA shard 0/2 DONE"
) &
PID2=$!

# GPU 3: GUIDE DocVQA shard 1/2, then ChartQA
(
  log "[gpu3] GUIDE DocVQA shard 1/2 START"
  run_docker 3 guide "${c_guide}/docvqa_s1" \
    "--benchmarks docvqa --shard-idx 1 --num-shards 2" \
    | tee "${GUIDE_OUT}/docvqa_s1.log"
  log "[gpu3] GUIDE DocVQA shard 1/2 DONE → ChartQA START"
  run_docker 3 guide "${c_guide}/chartqa" \
    "--benchmarks chartqa" \
    | tee "${GUIDE_OUT}/chartqa.log"
  log "[gpu3] GUIDE ChartQA DONE"
) &
PID3=$!

log "Waiting for all 4 workers..."
wait $PID0; E0=$?
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?
log "All done: $E0 $E1 $E2 $E3"

# ── Merge shards & compute final accuracy ─────────────────────────────────────
python3 - "${FA2_OUT}" "${GUIDE_OUT}" \
  "${PROJECT_ROOT}/output/qwen25vl_baseline/eval_summary.json" <<'PY'
import sys, json, csv
from pathlib import Path

fa2_root   = Path(sys.argv[1])
guide_root = Path(sys.argv[2])
baseline_p = Path(sys.argv[3])
base = json.loads(baseline_p.read_text()) if baseline_p.exists() else {}

def merge_docvqa_shards(root):
    """ANLS average across shards (prediction-level merge)."""
    preds = []
    for shard in ["docvqa_s0", "docvqa_s1"]:
        p = root / shard / "docvqa_predictions.jsonl"
        if not p.exists():
            print(f"  MISSING shard: {p}"); continue
        preds += [json.loads(l) for l in open(p)]
    if not preds: return None, 0
    total = sum(r.get("anls", 0) for r in preds)   # field name is "anls"
    return round(total / len(preds), 4), len(preds)

def load_chartqa(root):
    p = root / "chartqa" / "chartqa_predictions.jsonl"
    if not p.exists(): return None, 0
    rows = [json.loads(l) for l in open(p)]
    total = sum(1 if r.get("correct") else 0 for r in rows)  # field is "correct" bool
    return round(total / len(rows), 4), len(rows)

fa2_docvqa,   fa2_n_doc  = merge_docvqa_shards(fa2_root)
fa2_chartqa,  fa2_n_cha  = load_chartqa(fa2_root)
guide_docvqa, gui_n_doc  = merge_docvqa_shards(guide_root)
guide_chartqa,gui_n_cha  = load_chartqa(guide_root)

base_docvqa  = base.get("results",{}).get("docvqa",{}).get("ANLS")
base_chartqa = base.get("results",{}).get("chartqa",{}).get("relaxed_acc")
base_n       = base.get("num_samples", 500)

print()
print("="*70)
print("  Full Validation Accuracy: SDPA-baseline vs FA2 vs GUIDE")
print("="*70)
print(f"  {'metric':<28}  {'SDPA(n=500)':>12}  {'FA2':>10}  {'GUIDE':>10}  diff(FA2→GUIDE)")
print("  " + "-"*68)

rows = [
    ("DocVQA ANLS",    base_docvqa,  fa2_docvqa,  guide_docvqa, fa2_n_doc,  gui_n_doc),
    ("ChartQA relaxed_acc", base_chartqa, fa2_chartqa, guide_chartqa, fa2_n_cha, gui_n_cha),
]
for name, b, f, g, fn, gn in rows:
    bs = f"{b:.4f}(n={base_n})" if b else "N/A"
    fs = f"{f:.4f}(n={fn})" if f else "N/A"
    gs = f"{g:.4f}(n={gn})" if g else "N/A"
    diff = f"{g-f:+.4f}" if (f and g) else "N/A"
    print(f"  {name:<28}  {bs:>12}  {fs:>10}  {gs:>10}  {diff}")
print("="*70)

# Save
result = {
    "sdpa_baseline": {"docvqa_anls": base_docvqa, "chartqa_relaxed_acc": base_chartqa, "n": base_n},
    "fa2":   {"docvqa_anls": fa2_docvqa,   "chartqa_relaxed_acc": fa2_chartqa,   "n_docvqa": fa2_n_doc,  "n_chartqa": fa2_n_cha},
    "guide": {"docvqa_anls": guide_docvqa, "chartqa_relaxed_acc": guide_chartqa, "n_docvqa": gui_n_doc, "n_chartqa": gui_n_cha},
}
out_p = fa2_root.parent / "fa2_vs_guide_accuracy_full.json"
out_p.write_text(json.dumps(result, indent=2))
print(f"\nSaved: {out_p}")
PY
