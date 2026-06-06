#!/usr/bin/env bash
# Phase 1: Comparative smoke test — FA2 vs GUIDE, 2 samples each
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-qwen25vl-guide:latest}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/output/qwen25vl_baseline/manifests/benchmark_manifest_500_docker.jsonl}"
NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2"

c_root="/workspace"
c_model="${MODEL_PATH/${PROJECT_ROOT}/${c_root}}"
c_manifest="${MANIFEST/${PROJECT_ROOT}/${c_root}}"

run_docker() {
  local gpu="$1"; shift; local cmd="$1"
  local mounts=("-v" "${PROJECT_ROOT}:/workspace")
  [ -d "${NSYS_HOST_DIR}" ] && mounts+=("-v" "${NSYS_HOST_DIR}:/opt/nsight-systems:ro")
  docker run --rm --gpus "device=${gpu}" --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" "${mounts[@]}" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
    -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton -e XDG_CACHE_HOME=/tmp/.cache \
    -e PATH="/opt/nsight-systems/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    -w /workspace "${IMAGE}" -c "${cmd}"
}

echo "=================================================================="
echo " Phase 1: Comparative Smoke Test"
echo " Image  : ${IMAGE}"
echo " Manifest: ${MANIFEST}"
echo "=================================================================="

# ── 1. GPU status ────────────────────────────────────────────────────────────
echo "[1/5] GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# ── 2. Manifest check ────────────────────────────────────────────────────────
echo "[2/5] Manifest:"
n=$(wc -l < "${MANIFEST}" 2>/dev/null || echo 0)
echo "  total lines: ${n}"
python3 -c "
import json,collections
rows=[json.loads(l) for l in open('${MANIFEST}')]
bk=collections.Counter(r.get('benchmark') for r in rows)
print('  benchmarks:', dict(bk))
"

# ── 3. nsys smoke ────────────────────────────────────────────────────────────
echo "[3/5] nsys smoke (GPU 0):"
run_docker 0 "nsys profile -t cuda,nvtx --force-overwrite=true \
  -o /tmp/nsys_smoke python3 /workspace/tools/smoke_cuda_profiler.py \
  2>&1 | tail -3" && echo "  nsys PASS" || echo "  nsys FAIL (check manually)"

# ── 4. FA2 smoke (2 samples) ─────────────────────────────────────────────────
FA2_SMOKE="${PROJECT_ROOT}/output/qwen25vl_fa2_500/smoke"
mkdir -p "${FA2_SMOKE}"
c_fa2="${FA2_SMOKE/${PROJECT_ROOT}/${c_root}}"

echo "[4/5] FA2 smoke (2 samples):"
run_docker 0 "python3 -u /workspace/tools/run_comparative_inference.py \
  --mode fa2 \
  --manifest '${c_manifest}' \
  --output-dir '${c_fa2}' \
  --gpu-id 0 --start-idx 0 --end-idx 2 \
  --max-new-tokens 32 --model-path '${c_model}'" \
  2>&1 | grep -v "Warning\|deprecat\|fast processor\|Loading checkpoint" | tail -15

echo "  FA2 smoke results:"
python3 -c "
import json
from pathlib import Path
p=Path('${FA2_SMOKE}/results_gpu0.jsonl')
if not p.exists(): print('  NO RESULTS'); exit()
rows=[json.loads(l) for l in open(p)]
for r in rows:
    print(f\"  [{r.get('benchmark')}] ve={r.get('ve_ms',-1):.1f}ms ttft={r.get('ttft_ms',-1):.1f}ms pred='{r.get('pred_text','')[:40]}' score={r.get('anls',r.get('relaxed_acc','?'))}\")
" 2>/dev/null

# ── 5. GUIDE smoke (2 samples) ───────────────────────────────────────────────
GUIDE_SMOKE="${PROJECT_ROOT}/output/qwen25vl_guide_500/smoke"
mkdir -p "${GUIDE_SMOKE}"
c_guide="${GUIDE_SMOKE/${PROJECT_ROOT}/${c_root}}"

echo "[5/5] GUIDE smoke (2 samples):"
run_docker 0 "python3 -u /workspace/tools/run_comparative_inference.py \
  --mode guide \
  --manifest '${c_manifest}' \
  --output-dir '${c_guide}' \
  --gpu-id 0 --start-idx 0 --end-idx 2 \
  --max-new-tokens 32 --model-path '${c_model}'" \
  2>&1 | grep -v "Warning\|deprecat\|fast processor\|Loading checkpoint" | tail -15

echo "  GUIDE smoke results:"
python3 -c "
import json
from pathlib import Path
p=Path('${GUIDE_SMOKE}/results_gpu0.jsonl')
if not p.exists(): print('  NO RESULTS'); exit()
rows=[json.loads(l) for l in open(p)]
for r in rows:
    print(f\"  [{r.get('benchmark')}] ve={r.get('ve_ms',-1):.1f}ms ttft={r.get('ttft_ms',-1):.1f}ms pred='{r.get('pred_text','')[:40]}' score={r.get('anls',r.get('relaxed_acc','?'))}\")
" 2>/dev/null

echo
echo "=================================================================="
echo " Phase 1 COMPLETE — check results above before running Phase 2"
echo "=================================================================="
