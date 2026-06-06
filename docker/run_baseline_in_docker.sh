#!/usr/bin/env bash
# Run Qwen2.5-VL baseline scripts inside Docker with GPU + nsys/ncu access
# Usage:
#   bash docker/run_baseline_in_docker.sh                     # Phase 1 (default)
#   bash docker/run_baseline_in_docker.sh phase1              # Phase 1
#   bash docker/run_baseline_in_docker.sh phase2              # Phase 2
#   bash docker/run_baseline_in_docker.sh phase3              # Phase 3
#   bash docker/run_baseline_in_docker.sh all                 # All phases
#   bash docker/run_baseline_in_docker.sh bash                # Interactive shell
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-baseline:latest}"
PHASE="${1:-phase1}"

# ─── Profiler paths (from host) ────────────────────────────────────────────────
NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2"
NCU_HOST_DIR="/opt/nvidia/nsight-compute/2025.3.1"

# Verify Docker image exists
if ! docker image inspect "${IMAGE_NAME}" &>/dev/null; then
  echo "ERROR: Docker image '${IMAGE_NAME}' not found."
  echo "  Build it first with: bash docker/build_baseline_image.sh"
  exit 1
fi

# ─── Volume mounts ─────────────────────────────────────────────────────────────
MOUNTS=(
  "-v" "${PROJECT_ROOT}:/workspace"
)

EXTRA_PATH="/workspace/tools"

if [ -d "${NSYS_HOST_DIR}" ]; then
  MOUNTS+=("-v" "${NSYS_HOST_DIR}:/opt/nsight-systems:ro")
  EXTRA_PATH="/opt/nsight-systems/bin:${EXTRA_PATH}"
  echo "[docker] Mounting nsys from: ${NSYS_HOST_DIR}"
fi

if [ -d "${NCU_HOST_DIR}" ]; then
  MOUNTS+=("-v" "${NCU_HOST_DIR}:/opt/nsight-compute:ro")
  EXTRA_PATH="/opt/nsight-compute:${EXTRA_PATH}"
  echo "[docker] Mounting ncu from: ${NCU_HOST_DIR}"
fi

# ─── Determine command to run ──────────────────────────────────────────────────
case "${PHASE}" in
  phase1)
    CMD="/workspace/scripts/01_run_sanity_and_smoke.sh"
    ;;
  phase2)
    CMD="/workspace/scripts/02_run_main_baseline.sh"
    ;;
  phase3)
    CMD="/workspace/scripts/03_run_scientific_analysis.sh"
    ;;
  eval)
    CMD="/workspace/scripts/04_run_evaluation.sh"
    ;;
  all)
    CMD="-c bash /workspace/scripts/01_run_sanity_and_smoke.sh && bash /workspace/scripts/02_run_main_baseline.sh && bash /workspace/scripts/03_run_scientific_analysis.sh"
    ;;
  bash|shell)
    CMD=""
    ;;
  *)
    CMD="-c ${PHASE}"
    ;;
esac

echo "[docker] Running: ${CMD}"
echo "[docker] Image: ${IMAGE_NAME}"
echo "[docker] Project: ${PROJECT_ROOT} → /workspace"
echo ""

# ─── Docker run ────────────────────────────────────────────────────────────────
docker run \
  --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  "${MOUNTS[@]}" \
  -e "PROJECT_ROOT=/workspace" \
  -e "DATA_ROOT=/workspace/data/raw" \
  -e "MODEL_ROOT=/workspace/models" \
  -e "MODEL_PATH=/workspace/models/Qwen2.5-VL-3B-Instruct" \
  -e "OUTPUT_ROOT=/workspace/output/qwen25vl_baseline" \
  -e "TOOLS_DIR=/workspace/tools" \
  -e "PATH=${EXTRA_PATH}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
  -e "PRECISION=${PRECISION:-bf16}" \
  -e "SEED=${SEED:-42}" \
  -e "WARMUP=${WARMUP:-2}" \
  -e "MEASURED_ITERS=${MEASURED_ITERS:-5}" \
  -e "PROFILE_ITERS=${PROFILE_ITERS:-3}" \
  -e "BENCHMARKS=${BENCHMARKS:-docvqa chartqa ocrbench_v2}" \
  -e "NUM_SAMPLES=${NUM_SAMPLES:-500}" \
  -e "MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}" \
  -w /workspace \
  "${IMAGE_NAME}" \
  ${CMD:-}
