#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
TEST_DIR="${PROJECT_ROOT}/data/raw/_git_lfs_check"

mkdir -p "${TEST_DIR}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Test dir: ${TEST_DIR}"
echo ""

echo "============================================================"
echo "[CHECK] git"
echo "============================================================"
if command -v git >/dev/null 2>&1; then
  git --version
else
  echo "[FAIL] git not found"
  exit 1
fi

echo ""
echo "============================================================"
echo "[CHECK] git-lfs"
echo "============================================================"
if command -v git-lfs >/dev/null 2>&1; then
  git-lfs --version
elif git lfs version >/dev/null 2>&1; then
  git lfs version
else
  echo "[FAIL] git-lfs not found"
  echo "[INFO] git exists, but git-lfs is not installed or not in PATH."
  exit 1
fi

echo ""
echo "============================================================"
echo "[CHECK] git lfs env"
echo "============================================================"
git lfs env || true

echo ""
echo "============================================================"
echo "[CHECK] Hugging Face repo access via git ls-remote"
echo "============================================================"

REPOS=(
  "https://huggingface.co/datasets/HuggingFaceM4/ChartQA"
  "https://huggingface.co/datasets/lmms-lab/OCRBench-v2"
  "https://huggingface.co/datasets/opendatalab/OmniDocBench"
  "https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1"
)

for repo in "${REPOS[@]}"; do
  echo ""
  echo "[TEST] ${repo}"
  if git ls-remote "${repo}" HEAD >/dev/null 2>&1; then
    echo "[OK] git ls-remote works"
  else
    echo "[FAIL] git ls-remote failed: ${repo}"
  fi
done

echo ""
echo "============================================================"
echo "[CHECK] Metadata-only clone with GIT_LFS_SKIP_SMUDGE=1"
echo "============================================================"
echo "[INFO] This does NOT download large LFS files."
echo ""

clone_metadata_only() {
  local repo="$1"
  local name="$2"
  local out="${TEST_DIR}/${name}"

  echo ""
  echo "------------------------------------------------------------"
  echo "[TEST] ${name}"
  echo "[REPO] ${repo}"
  echo "[OUT]  ${out}"
  echo "------------------------------------------------------------"

  rm -rf "${out}"

  if GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${repo}" "${out}"; then
    echo "[OK] metadata clone succeeded: ${name}"
  else
    echo "[FAIL] metadata clone failed: ${name}"
    return 0
  fi

  echo "[INFO] LFS tracked files:"
  if git -C "${out}" lfs ls-files | head -30; then
    local count
    count="$(git -C "${out}" lfs ls-files | wc -l || echo 0)"
    echo "[INFO] LFS file count: ${count}"
  else
    echo "[WARN] Could not list LFS files."
  fi

  echo "[INFO] Top-level files:"
  find "${out}" -maxdepth 2 -type f | sed "s#${out}/##" | head -30 || true
}

clone_metadata_only "https://huggingface.co/datasets/HuggingFaceM4/ChartQA" "chartqa"
clone_metadata_only "https://huggingface.co/datasets/lmms-lab/OCRBench-v2" "ocrbench_v2"
clone_metadata_only "https://huggingface.co/datasets/opendatalab/OmniDocBench" "omnidocbench"
clone_metadata_only "https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1" "visionarena_bench"

echo ""
echo "============================================================"
echo "[DONE] git-lfs check complete."
echo "============================================================"
echo ""
echo "[INTERPRETATION]"
echo "- If git-lfs is missing: you cannot use git-lfs without installing it or using another machine."
echo "- If ls-remote works but metadata clone fails: network/proxy/certificate issue."
echo "- If metadata clone works and lfs ls-files shows files: git-lfs path is usable."
echo "- This script does not download the full datasets."
