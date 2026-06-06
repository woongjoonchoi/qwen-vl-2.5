#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"

MODEL_ROOT="${PROJECT_ROOT}/models"
MODEL_DIR="${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct"
REPO_URL="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"

mkdir -p "${MODEL_ROOT}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Model dir:    ${MODEL_DIR}"
echo "[INFO] Repo URL:     ${REPO_URL}"

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git is not installed or not in PATH."
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  echo "[ERROR] git-lfs is not installed or not in PATH."
  exit 1
fi

git --version
git lfs version

if [[ -d "${MODEL_DIR}" && ! -d "${MODEL_DIR}/.git" ]]; then
  echo "[ERROR] ${MODEL_DIR} exists but is not a git repo."
  echo "[ERROR] I will not overwrite it automatically."
  echo "[ERROR] Inspect it with:"
  echo "  find ${MODEL_DIR} -maxdepth 2 -type f | head -50"
  exit 1
fi

if [[ ! -d "${MODEL_DIR}/.git" ]]; then
  echo "[INFO] Cloning model metadata first with LFS smudge disabled..."
  GIT_LFS_SKIP_SMUDGE=1 git clone "${REPO_URL}" "${MODEL_DIR}"
else
  echo "[INFO] Existing model git repo found. Pulling metadata..."
  git -C "${MODEL_DIR}" pull --ff-only || true
fi

echo "[INFO] Listing LFS files..."
git -C "${MODEL_DIR}" lfs ls-files | head -50 || true

echo "[INFO] Pulling model weights with git-lfs..."
git -C "${MODEL_DIR}" lfs pull

echo "[DONE] Qwen2.5-VL-3B-Instruct downloaded."

echo "[INFO] Model size:"
du -sh "${MODEL_DIR}" || true

echo "[INFO] Top-level model files:"
find "${MODEL_DIR}" -maxdepth 2 -type f | sed "s#${MODEL_DIR}/##" | sort | head -100

echo "[INFO] Safetensors files:"
find "${MODEL_DIR}" -maxdepth 2 -type f -name "*.safetensors" -lh | sort || true
