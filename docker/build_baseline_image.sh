#!/usr/bin/env bash
# Build qwen25vl-baseline Docker image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-baseline:latest}"

echo "[build] Building Docker image: ${IMAGE_NAME}"
echo "[build] Dockerfile: ${SCRIPT_DIR}/Dockerfile-qwen25vl-baseline"
echo "[build] Build context: $(dirname ${SCRIPT_DIR})"

docker build \
  -f "${SCRIPT_DIR}/Dockerfile-qwen25vl-baseline" \
  -t "${IMAGE_NAME}" \
  "$(dirname ${SCRIPT_DIR})"

echo "[build] DONE: ${IMAGE_NAME}"
docker images "${IMAGE_NAME%:*}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
