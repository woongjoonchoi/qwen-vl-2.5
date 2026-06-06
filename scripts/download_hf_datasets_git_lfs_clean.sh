#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
RAW_DIR="${PROJECT_ROOT}/data/raw"

mkdir -p "${RAW_DIR}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Raw dir: ${RAW_DIR}"

safe_prepare_dir() {
  local dir="$1"

  if [[ ! -d "${dir}" ]]; then
    return 0
  fi

  if [[ -d "${dir}/.git" ]]; then
    echo "[INFO] Existing git repo found: ${dir}"
    return 0
  fi

  # Count files except .gitkeep
  local non_gitkeep_count
  non_gitkeep_count="$(find "${dir}" -mindepth 1 -maxdepth 1 ! -name ".gitkeep" | wc -l)"

  if [[ "${non_gitkeep_count}" -eq 0 ]]; then
    echo "[INFO] Removing empty/.gitkeep-only directory: ${dir}"
    rm -rf "${dir}"
  else
    echo "[ERROR] ${dir} exists, is not a git repo, and contains files."
    echo "[ERROR] I will not delete it automatically."
    echo "[ERROR] Inspect with:"
    echo "  find ${dir} -maxdepth 2 -type f | head -50"
    exit 1
  fi
}

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

  safe_prepare_dir "${out}"

  if [[ ! -d "${out}/.git" ]]; then
    echo "[INFO] Cloning metadata first with LFS smudge disabled..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${repo}" "${out}"
  else
    echo "[INFO] Existing git repo found. Pulling latest metadata..."
    git -C "${out}" pull --ff-only || true
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
