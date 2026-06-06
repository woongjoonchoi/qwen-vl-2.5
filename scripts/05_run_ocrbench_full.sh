#!/usr/bin/env bash
# =============================================================================
# OCRBench v2 Full Evaluation — 4-GPU 병렬
# task-specific max_new_tokens + prompts 적용
# GPU 0~3: 각 2,500샘플 (10,000 / 4)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_eval_ocrbench}"
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-baseline:latest}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
NSYS_DIR="/opt/nvidia/nsight-systems/2025.3.2"
NCU_DIR="/opt/nvidia/nsight-compute/2025.3.1"

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_DIR="${OUTPUT_ROOT}/logs"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "${LOG_DIR}/ocrbench_full.log"; }

log "=== OCRBench v2 Full Evaluation (4-GPU, task-specific tokens) ==="
log "Model  : ${MODEL_PATH}"
log "Output : ${OUTPUT_ROOT}"

run_shard() {
  local gpu_id="$1"
  local shard_idx="$2"
  local num_shards=4
  local log_file="${LOG_DIR}/gpu${gpu_id}_shard${shard_idx}.log"

  local c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
  local c_data="${DATA_ROOT/${PROJECT_ROOT}/\/workspace}"
  local c_out="${OUTPUT_ROOT/${PROJECT_ROOT}/\/workspace}"

  local mounts=(-v "${PROJECT_ROOT}:/workspace")
  [ -d "${NSYS_DIR}" ] && mounts+=(-v "${NSYS_DIR}:/opt/nsight-systems:ro")
  [ -d "${NCU_DIR}"  ] && mounts+=(-v "${NCU_DIR}:/opt/nsight-compute:ro")

  log "  [GPU ${gpu_id}] shard ${shard_idx}/4 starting..."
  docker run \
    --rm \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --ipc=host \
    --ulimit memlock=-1 \
    "${mounts[@]}" \
    -e "CUDA_VISIBLE_DEVICES=${gpu_id}" \
    -e "PYTHONUNBUFFERED=1" \
    "${IMAGE_NAME}" \
    -c "python3 -u /workspace/tools/eval_ocrbench_full.py \
      --model-path  ${c_model} \
      --data-root   ${c_data} \
      --output-dir  ${c_out} \
      --num-samples 0 \
      --precision   ${PRECISION} \
      --device      cuda:0 \
      --shard-idx   ${shard_idx} \
      --num-shards  ${num_shards} \
      --seed        ${SEED}" \
    2>&1 | tee "${log_file}"
}

# 4 shards in parallel
run_shard 0 0 & PID0=$!
run_shard 1 1 & PID1=$!
run_shard 2 2 & PID2=$!
run_shard 3 3 & PID3=$!

log "All 4 GPU shards launched. Waiting..."
FAILED=0
wait $PID0; RC=$?; log "  GPU0 shard0 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID1; RC=$?; log "  GPU1 shard1 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID2; RC=$?; log "  GPU2 shard2 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID3; RC=$?; log "  GPU3 shard3 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1

# ── Merge 4 shards ────────────────────────────────────────────────────────────
log "Merging 4 shards..."
python3 - "${OUTPUT_ROOT}" << 'PYEOF'
import json, sys
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])
out_dir = root / "ocrbench_v2"
out_dir.mkdir(parents=True, exist_ok=True)

shard_files = sorted(root.glob("ocrbench_v2_shard*_predictions.jsonl"))
if not shard_files:
    print("  ERROR: no shard files found")
    sys.exit(1)

all_preds = []
for sf in shard_files:
    with open(sf) as f:
        for line in f:
            if line.strip(): all_preds.append(json.loads(line))

all_preds.sort(key=lambda x: x.get("sample_id",""))
n = len(all_preds)

type_stats = defaultdict(lambda: {"c":0,"t":0})
for r in all_preds:
    t = r.get("type","unknown")
    type_stats[t]["t"] += 1
    if r.get("correct"): type_stats[t]["c"] += 1

total_c = sum(v["c"] for v in type_stats.values())
em_acc = total_c / n if n else 0

# category grouping
CAT = {
    "OCR Core": ["text recognition en","fine-grained text recognition en",
                  "handwritten answer extraction cn","full-page OCR en","full-page OCR cn",
                  "text counting en","text spotting en","text grounding en"],
    "Cognition & Reasoning": ["cognition VQA en","cognition VQA cn","reasoning VQA en",
                               "reasoning VQA cn","VQA with position en","math QA en","science QA en"],
    "Document & Structured": ["document parsing en","document parsing cn","document classification en",
                               "key information extraction en","key information extraction cn",
                               "key information mapping en","table parsing en","table parsing cn",
                               "formula recognition en","formula recognition cn"],
    "Scene & Chart": ["chart parsing en","APP agent en","diagram QA en"],
    "Multilingual": ["text translation cn"],
    "Misc": ["ASCII art classification en"],
}

print(f"\n  Merged: {n} samples  EM={em_acc:.4f}\n")
print("="*65)
print("  OCRBench v2 Final Results (task-specific metrics)")
print("="*65)
for cat, tasks in CAT.items():
    cat_c = sum(type_stats[t]["c"] for t in tasks if t in type_stats)
    cat_t = sum(type_stats[t]["t"] for t in tasks if t in type_stats)
    if cat_t == 0: continue
    bar = '█' * int(cat_c/cat_t*20)
    print(f"\n  [{cat}]  {cat_c}/{cat_t} = {cat_c/cat_t*100:.1f}%  {bar}")
    for t in tasks:
        v = type_stats.get(t)
        if not v or v["t"]==0: continue
        acc = v["c"]/v["t"]
        print(f"    ├ {t:<42} {v['c']:>4}/{v['t']:<4} = {acc*100:5.1f}%")

print(f"\n  ★ Overall EM: {em_acc:.4f}  ({total_c}/{n})")

# Save
with open(out_dir / "ocrbench_v2_predictions.jsonl","w") as f:
    for r in all_preds: f.write(json.dumps(r, ensure_ascii=False)+"\n")

summary = {
    "num_samples": n, "exact_match": round(em_acc,4),
    "by_type": {t: round(v["c"]/v["t"],4) for t,v in type_stats.items() if v["t"]>0},
    "by_category": {
        cat: round(sum(type_stats[t]["c"] for t in tasks if t in type_stats) /
                   max(sum(type_stats[t]["t"] for t in tasks if t in type_stats),1), 4)
        for cat, tasks in CAT.items()
    },
}
(out_dir / "ocrbench_v2_summary.json").write_text(json.dumps(summary,indent=2))
print(f"\n  Written: {out_dir}/ocrbench_v2_summary.json")
PYEOF

if [ "${FAILED}" -eq 0 ]; then
  log "=== OCRBench v2 Full Evaluation COMPLETE ==="
else
  log "=== OCRBench v2 done with failures (check logs) ==="
fi
exit "${FAILED}"
