#!/usr/bin/env bash
# =============================================================================
# Qwen2.5-VL Full Evaluation — 4-GPU 병렬 실행
#
# GPU 배분:
#   GPU 0  : DocVQA        (5,349 samples) ~ 153 min
#   GPU 1  : ChartQA       (1,920 samples) ~  55 min
#   GPU 2  : OCRBench v2 shard 0/2 (5,000) ~ 144 min
#   GPU 3  : OCRBench v2 shard 1/2 (5,000) ~ 144 min
#
# 예상 총 소요시간: ~153 min (DocVQA 기준)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/raw}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_eval_full}"
TOOLS_DIR="${PROJECT_ROOT}/tools"

PRECISION="${PRECISION:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SEED="${SEED:-42}"
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-baseline:latest}"

NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2"
NCU_HOST_DIR="/opt/nvidia/nsight-compute/2025.3.1"

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_DIR="${OUTPUT_ROOT}/logs"
ts_human() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts_human)] $1" | tee -a "${LOG_DIR}/eval_full.log"; }

log "=== Qwen2.5-VL Full Evaluation (4-GPU Parallel) ==="
log "Model  : ${MODEL_PATH}"
log "Output : ${OUTPUT_ROOT}"

# ── helper: run one eval job in a Docker container ─────────────────────────
run_eval_job() {
  local gpu_id="$1"
  local benchmarks="$2"
  local shard_idx="${3:-0}"
  local num_shards="${4:-1}"
  local job_name="eval_gpu${gpu_id}_${benchmarks// /_}"
  local log_file="${LOG_DIR}/${job_name}.log"

  # Convert host paths → container-internal paths (/workspace/...)
  local c_model_path="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
  local c_data_root="${DATA_ROOT/${PROJECT_ROOT}/\/workspace}"
  local c_output_root="${OUTPUT_ROOT/${PROJECT_ROOT}/\/workspace}"

  log "  [GPU ${gpu_id}] Starting: ${benchmarks} (shard ${shard_idx}/${num_shards})"

  local mounts=(-v "${PROJECT_ROOT}:/workspace")
  [ -d "${NSYS_HOST_DIR}" ] && mounts+=(-v "${NSYS_HOST_DIR}:/opt/nsight-systems:ro")
  [ -d "${NCU_HOST_DIR}"  ] && mounts+=(-v "${NCU_HOST_DIR}:/opt/nsight-compute:ro")

  docker run \
    --rm \
    --gpus "device=${gpu_id}" \
    --privileged \
    --user "$(id -u):$(id -g)" \
    --ipc=host \
    --ulimit memlock=-1 \
    "${mounts[@]}" \
    -e "PATH=/opt/nsight-systems/bin:/opt/nsight-compute:/workspace/tools:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    -e "PYTHONUNBUFFERED=1" \
    "${IMAGE_NAME}" \
    -c "python3 -u /workspace/tools/eval_qwen25vl.py \
      --model-path   ${c_model_path} \
      --data-root    ${c_data_root} \
      --output-dir   ${c_output_root} \
      --benchmarks   ${benchmarks} \
      --num-samples  0 \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      --precision    ${PRECISION} \
      --device       cuda:0 \
      --shard-idx    ${shard_idx} \
      --num-shards   ${num_shards} \
      --seed         ${SEED}" \
    2>&1 | tee "${log_file}"
}

# ── Launch 4 jobs in parallel ───────────────────────────────────────────────

log "Launching 4 parallel GPU jobs..."

# GPU 0: DocVQA full
run_eval_job 0 "docvqa" 0 1 &
PID_DOCVQA=$!
log "  [GPU 0] DocVQA PID=${PID_DOCVQA}"

# GPU 1: ChartQA full
run_eval_job 1 "chartqa" 0 1 &
PID_CHARTQA=$!
log "  [GPU 1] ChartQA PID=${PID_CHARTQA}"

# GPU 2: OCRBench shard 0/2
run_eval_job 2 "ocrbench_v2" 0 2 &
PID_OCR0=$!
log "  [GPU 2] OCRBench shard 0/2 PID=${PID_OCR0}"

# GPU 3: OCRBench shard 1/2
run_eval_job 3 "ocrbench_v2" 1 2 &
PID_OCR1=$!
log "  [GPU 3] OCRBench shard 1/2 PID=${PID_OCR1}"

log "All jobs launched. Waiting for completion..."

# ── Wait and collect exit codes ─────────────────────────────────────────────

FAILED=0
wait $PID_DOCVQA;  RC=$?; log "  [GPU 0] DocVQA done (rc=${RC})";  [ $RC -ne 0 ] && FAILED=1
wait $PID_CHARTQA; RC=$?; log "  [GPU 1] ChartQA done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID_OCR0;    RC=$?; log "  [GPU 2] OCRBench shard 0 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID_OCR1;    RC=$?; log "  [GPU 3] OCRBench shard 1 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1

# ── Merge OCRBench shards ────────────────────────────────────────────────────

log "Merging OCRBench shards..."
python3 - "${OUTPUT_ROOT}" << 'PYEOF'
import json, csv, sys
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])
ocr_dir = root / "ocrbench_v2"

# Merge prediction jsonl files
shard_files = sorted(ocr_dir.glob("ocrbench_v2_shard*of2_predictions.jsonl"))
if not shard_files:
    print("  No shard files found to merge, skipping.")
    sys.exit(0)

all_preds = []
for sf in shard_files:
    with open(sf) as f:
        for line in f:
            if line.strip():
                all_preds.append(json.loads(line))

# Sort by sample_id for reproducibility
all_preds.sort(key=lambda x: x.get("sample_id", ""))

# Compute merged metrics
em_total  = sum(1 for r in all_preds if r.get("exact_match"))
n = len(all_preds)
em_acc = em_total / n if n else 0

by_type = defaultdict(lambda: {"c": 0, "t": 0})
for r in all_preds:
    t = r.get("type", "unknown")
    by_type[t]["t"] += 1
    if r.get("exact_match"):
        by_type[t]["c"] += 1

print(f"  OCRBench merged: {n} samples, EM={em_acc:.4f}")

# Write merged predictions
with open(ocr_dir / "ocrbench_v2_predictions.jsonl", "w") as f:
    for r in all_preds:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Write merged summary
summary = {
    "benchmark": "ocrbench_v2", "split": "test", "num_samples": n,
    "exact_match": round(em_acc, 4),
    "by_type": {t: round(v["c"]/v["t"], 4) for t, v in by_type.items() if v["t"] > 0},
}
(ocr_dir / "ocrbench_v2_summary.json").write_text(json.dumps(summary, indent=2))
print(f"  Written merged summary: {ocr_dir / 'ocrbench_v2_summary.json'}")
for t, v in sorted(by_type.items(), key=lambda x: -x[1]["t"]):
    print(f"    {t}: {v['c']}/{v['t']} = {v['c']/v['t']:.3f}")
PYEOF

# ── Final consolidated report ────────────────────────────────────────────────

log "Generating final report..."
python3 - "${OUTPUT_ROOT}" "${MAX_NEW_TOKENS}" << 'PYEOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
max_tok = int(sys.argv[2])

results = {}
for bm, fname in [("docvqa", "docvqa/docvqa_summary.json"),
                   ("chartqa", "chartqa/chartqa_summary.json"),
                   ("ocrbench_v2", "ocrbench_v2/ocrbench_v2_summary.json")]:
    p = root / fname
    if p.exists():
        d = json.loads(p.read_text())
        results[bm] = d
    else:
        results[bm] = {"status": "missing"}

# Write consolidated summary
final = {
    "model": "Qwen2.5-VL-3B-Instruct",
    "max_new_tokens": max_tok,
    "full_dataset": True,
    "results": {}
}

print()
print("=" * 65)
print("  FULL EVALUATION RESULTS — Qwen2.5-VL-3B-Instruct")
print("=" * 65)

for bm, d in results.items():
    if "mean_anls" in d:
        metric_val = d["mean_anls"]
        metric_name = "ANLS"
        final["results"][bm] = {"ANLS": metric_val, "n": d.get("num_samples")}
    elif "overall_relaxed_acc" in d:
        metric_val = d["overall_relaxed_acc"]
        metric_name = "Relaxed Acc"
        final["results"][bm] = {
            "relaxed_acc": metric_val,
            "human_acc": d.get("human_relaxed_acc"),
            "machine_acc": d.get("machine_relaxed_acc"),
            "n": d.get("num_samples"),
        }
    elif "exact_match" in d:
        metric_val = d["exact_match"]
        metric_name = "Exact Match"
        final["results"][bm] = {
            "exact_match": metric_val,
            "by_type": d.get("by_type", {}),
            "n": d.get("num_samples"),
        }
    else:
        metric_val = None
        metric_name = "?"
        final["results"][bm] = d

    n = d.get("num_samples", "?")
    print(f"  {bm:<15}: {metric_name} = {metric_val}  (n={n})")

print()
(root / "eval_full_summary.json").write_text(json.dumps(final, indent=2))
print(f"  Written: {root / 'eval_full_summary.json'}")
PYEOF

if [ "${FAILED}" -eq 0 ]; then
  log "=== Full Evaluation COMPLETE (all GPUs succeeded) ==="
else
  log "=== Full Evaluation DONE with some failures (check GPU logs) ==="
fi

exit "${FAILED}"
