#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
RAW_DIR="${PROJECT_ROOT}/data/raw"

mkdir -p "${RAW_DIR}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Raw dir: ${RAW_DIR}"

check_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1"
    exit 1
  fi
}

check_cmd git

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  echo "[ERROR] git-lfs is not available."
  exit 1
fi

git lfs version

clone_or_pull_dataset() {
  local repo="$1"
  local out="$2"
  local name="$3"

  echo ""
  echo "============================================================"
  echo "[INFO] Dataset: ${name}"
  echo "[INFO] Repo:    ${repo}"
  echo "[INFO] Out:     ${out}"
  echo "============================================================"

  # If directory exists only with .gitkeep, remove it so git clone can use it.
  if [[ -d "${out}" && ! -d "${out}/.git" ]]; then
    find "${out}" -maxdepth 1 -name ".gitkeep" -type f -delete

    if [[ -n "$(find "${out}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "[WARN] ${out} exists and is not empty, but not a git repo."
      echo "[WARN] Skipping clone to avoid overwriting files."
      echo "[WARN] Move or remove this directory if you want to re-clone."
      return 0
    fi
  fi

  if [[ ! -d "${out}/.git" ]]; then
    echo "[INFO] Cloning metadata first with LFS smudge disabled..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${repo}" "${out}"
  else
    echo "[INFO] Existing git repo found. Fetching latest metadata..."
    git -C "${out}" fetch --depth 1 origin main || git -C "${out}" fetch --depth 1 origin master || true
  fi

  echo "[INFO] Pulling LFS files..."
  git -C "${out}" lfs pull

  echo "[DONE] ${name}"
  du -sh "${out}" || true
  find "${out}" -maxdepth 3 -type f | head -30 || true
}

clone_or_pull_dataset \
  "https://huggingface.co/datasets/HuggingFaceM4/ChartQA" \
  "${RAW_DIR}/chartqa" \
  "ChartQA"

clone_or_pull_dataset \
  "https://huggingface.co/datasets/lmms-lab/OCRBench-v2" \
  "${RAW_DIR}/ocrbench_v2" \
  "OCRBench_v2"

clone_or_pull_dataset \
  "https://huggingface.co/datasets/opendatalab/OmniDocBench" \
  "${RAW_DIR}/omnidocbench" \
  "OmniDocBench"

clone_or_pull_dataset \
  "https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1" \
  "${RAW_DIR}/visionarena_bench" \
  "VisionArena-Bench v0.1"

echo ""
echo "============================================================"
echo "[DONE] All selected Hugging Face datasets downloaded via git-lfs."
echo "============================================================"

echo "[INFO] Size summary:"
du -sh "${RAW_DIR}/chartqa" || true
du -sh "${RAW_DIR}/ocrbench_v2" || true
du -sh "${RAW_DIR}/omnidocbench" || true
du -sh "${RAW_DIR}/visionarena_bench" || true
