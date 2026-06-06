#!/usr/bin/env bash
# FA2 vs GUIDE — benchmark accuracy evaluation
# GPU 0: FA2  (docvqa + chartqa)
# GPU 1: GUIDE (docvqa + chartqa)
# 기존 baseline eval과 동일 조건: num_samples=500, max_new_tokens=32, short-answer prompt
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-qwen25vl-guide:latest}"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
DATA_ROOT="${PROJECT_ROOT}/data/raw"
FA2_OUT="${PROJECT_ROOT}/output/qwen25vl_fa2_eval"
GUIDE_OUT="${PROJECT_ROOT}/output/qwen25vl_guide_eval"
c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
c_data="${DATA_ROOT/${PROJECT_ROOT}/\/workspace}"
c_fa2="${FA2_OUT/${PROJECT_ROOT}/\/workspace}"
c_guide="${GUIDE_OUT/${PROJECT_ROOT}/\/workspace}"
mkdir -p "${FA2_OUT}" "${GUIDE_OUT}"

run_eval() {
  local gpu="$1" attn="$2" c_out="$3"
  local mounts=("-v" "${PROJECT_ROOT}:/workspace")
  docker run --rm --gpus "device=${gpu}" --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" "${mounts[@]}" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
    -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton -e XDG_CACHE_HOME=/tmp/.cache \
    -w /workspace "${IMAGE}" -c "
python3 -u /workspace/tools/eval_qwen25vl.py \
  --model-path    '${c_model}' \
  --data-root     '${c_data}' \
  --output-dir    '${c_out}' \
  --benchmarks    docvqa chartqa \
  --num-samples   500 \
  --max-new-tokens 32 \
  --precision     bf16 \
  --attn-impl     ${attn} \
  --seed          42"
}

echo "=== FA2 eval (GPU 0) + GUIDE eval (GPU 1) 동시 실행 ==="
run_eval 0 fa2   "${c_fa2}"   2>&1 | grep -v "Warning\|deprecat\|fast processor\|Loading checkpoint" | tee "${FA2_OUT}/eval.log" &
PID0=$!
run_eval 1 guide "${c_guide}" 2>&1 | grep -v "Warning\|deprecat\|fast processor\|Loading checkpoint" | tee "${GUIDE_OUT}/eval.log" &
PID1=$!

wait $PID0; E0=$?
wait $PID1; E1=$?
echo "Done: FA2=$E0 GUIDE=$E1"

# 결과 비교
python3 - "${FA2_OUT}" "${GUIDE_OUT}" \
  "${PROJECT_ROOT}/output/qwen25vl_baseline/eval_summary.json" <<'PY'
import sys, json
from pathlib import Path
fa2_dir, guide_dir, baseline_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

def load_summary(d):
    for name in ["eval_summary.json","summary.json"]:
        p = d / name
        if p.exists(): return json.loads(p.read_text())
    return {}

base  = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
fa2   = load_summary(fa2_dir)
guide = load_summary(guide_dir)

print("\n" + "="*65)
print("  Benchmark Accuracy: SDPA(baseline) vs FA2 vs GUIDE")
print("="*65)
print(f"  {'metric':<30} {'SDPA':>8}  {'FA2':>8}  {'GUIDE':>8}")
print("  " + "-"*55)

def get_score(d, bm, metric):
    r = d.get("results", {})
    return r.get(bm, {}).get(metric)

metrics = [("docvqa","ANLS"), ("chartqa","relaxed_acc")]
for bm, m in metrics:
    b = get_score(base,  bm, m)
    f = get_score(fa2,   bm, m)
    g = get_score(guide, bm, m)
    bs = f"{b:.4f}" if b else "N/A"
    fs = f"{f:.4f}" if f else "N/A"
    gs = f"{g:.4f}" if g else "N/A"
    diff = f"{g-f:+.4f}" if (f and g) else ""
    print(f"  {bm} {m:<24} {bs:>8}  {fs:>8}  {gs:>8}  FA2→GUIDE: {diff}")
print("="*65)
print("  (SDPA baseline: num_samples=500, max_new_tokens=32)")
PY
