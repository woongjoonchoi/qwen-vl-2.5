#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"
RAW_DIR="${PROJECT_ROOT}/data/raw"
MODEL_DIR="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"

FAIL=0
WARN=0

print_header() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

pass() {
  echo "[PASS] $1"
}

warn() {
  echo "[WARN] $1"
  WARN=$((WARN + 1))
}

fail() {
  echo "[FAIL] $1"
  FAIL=$((FAIL + 1))
}

exists_file() {
  local f="$1"
  local name="$2"
  if [[ -f "$f" ]]; then
    pass "${name}: exists"
    ls -lh "$f" || true
  else
    fail "${name}: missing: $f"
  fi
}

exists_dir() {
  local d="$1"
  local name="$2"
  if [[ -d "$d" ]]; then
    pass "${name}: exists"
    du -sh "$d" || true
  else
    fail "${name}: missing: $d"
  fi
}

is_lfs_pointer() {
  local f="$1"
  if [[ -f "$f" ]] && head -n 1 "$f" 2>/dev/null | grep -q "version https://git-lfs.github.com/spec/v1"; then
    return 0
  fi
  return 1
}

check_not_lfs_pointer() {
  local f="$1"
  local name="$2"

  if [[ ! -f "$f" ]]; then
    fail "${name}: missing: $f"
    return
  fi

  if is_lfs_pointer "$f"; then
    fail "${name}: still a Git LFS pointer, real content not downloaded: $f"
  else
    pass "${name}: real file, not LFS pointer"
  fi
}

check_archive_type() {
  local f="$1"
  local pattern="$2"
  local name="$3"

  if [[ ! -f "$f" ]]; then
    fail "${name}: missing: $f"
    return
  fi

  echo "[INFO] file type for ${name}:"
  file "$f" || true

  if file "$f" | grep -Eiq "$pattern"; then
    pass "${name}: archive type looks valid"
  else
    fail "${name}: archive type invalid or unexpected"
    echo "[INFO] First 300 bytes:"
    head -c 300 "$f" || true
    echo ""
  fi
}

check_git_lfs_repo() {
  local d="$1"
  local name="$2"

  print_header "Git LFS check: ${name}"

  if [[ ! -d "$d" ]]; then
    fail "${name}: directory missing: $d"
    return
  fi

  du -sh "$d" || true

  if [[ ! -d "$d/.git" ]]; then
    fail "${name}: not a git repo: $d"
    return
  fi

  if ! command -v git >/dev/null 2>&1; then
    fail "git command not found"
    return
  fi

  if ! git -C "$d" lfs version >/dev/null 2>&1; then
    fail "${name}: git-lfs not available in this repo"
    return
  fi

  pass "${name}: git repo and git-lfs available"

  local lfs_count
  lfs_count="$(git -C "$d" lfs ls-files --name-only | wc -l | tr -d ' ')"
  echo "[INFO] ${name}: LFS tracked file count = ${lfs_count}"

  if [[ "$lfs_count" -eq 0 ]]; then
    warn "${name}: no LFS files listed"
    return
  fi

  local missing=0
  local pointer=0
  local checked=0

  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    checked=$((checked + 1))
    local f="${d}/${rel}"

    if [[ ! -f "$f" ]]; then
      echo "[MISSING] $rel"
      missing=$((missing + 1))
      continue
    fi

    if is_lfs_pointer "$f"; then
      echo "[POINTER] $rel"
      pointer=$((pointer + 1))
      continue
    fi
  done < <(git -C "$d" lfs ls-files --name-only)

  echo "[INFO] ${name}: checked LFS files = ${checked}"
  echo "[INFO] ${name}: missing LFS files = ${missing}"
  echo "[INFO] ${name}: pointer-only files = ${pointer}"

  if [[ "$missing" -eq 0 && "$pointer" -eq 0 ]]; then
    pass "${name}: all LFS files appear downloaded"
  else
    fail "${name}: some LFS files are missing or still pointer files"
  fi

  echo "[INFO] Top-level files:"
  find "$d" -maxdepth 3 -type f | sed "s#${d}/##" | head -40 || true
}

check_parquet_count() {
  local d="$1"
  local min_count="$2"
  local name="$3"

  local count
  count="$(find "$d" -type f -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')"

  echo "[INFO] ${name}: parquet count = ${count}"

  if [[ "$count" -ge "$min_count" ]]; then
    pass "${name}: parquet files found"
  else
    fail "${name}: expected at least ${min_count} parquet files, got ${count}"
  fi

  find "$d" -type f -name "*.parquet" -lh 2>/dev/null | head -30 || true
}

check_image_count() {
  local d="$1"
  local min_count="$2"
  local name="$3"

  local count
  count="$(find "$d" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" \) 2>/dev/null | wc -l | tr -d ' ')"

  echo "[INFO] ${name}: image count = ${count}"

  if [[ "$count" -ge "$min_count" ]]; then
    pass "${name}: images found"
  else
    warn "${name}: expected at least ${min_count} images, got ${count}"
  fi
}

print_header "Environment"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Raw dir:      ${RAW_DIR}"
echo "[INFO] Date:         $(date)"
echo "[INFO] Host:         $(hostname)"
echo ""

if command -v git >/dev/null 2>&1; then
  git --version
else
  warn "git not found"
fi

if command -v git-lfs >/dev/null 2>&1; then
  git-lfs --version
elif git lfs version >/dev/null 2>&1; then
  git lfs version
else
  warn "git-lfs not found"
fi

# ------------------------------------------------------------
# DocVQA
# ------------------------------------------------------------
print_header "DocVQA Task 1"

DOCVQA_DIR="${RAW_DIR}/docvqa"
DOCVQA_DOWNLOADS="${DOCVQA_DIR}/downloads"
DOCVQA_ANN_ARCHIVE="${DOCVQA_DOWNLOADS}/spdocvqa_qas.zip"
DOCVQA_IMG_ARCHIVE="${DOCVQA_DOWNLOADS}/spdocvqa_images.tar.gz"
DOCVQA_ANN_DIR="${DOCVQA_DIR}/annotations"
DOCVQA_IMG_DIR="${DOCVQA_DIR}/images"

exists_dir "$DOCVQA_DIR" "DocVQA root"
exists_dir "$DOCVQA_DOWNLOADS" "DocVQA downloads"

if [[ -f "$DOCVQA_ANN_ARCHIVE" ]]; then
  check_archive_type "$DOCVQA_ANN_ARCHIVE" "Zip archive" "DocVQA annotations archive"
else
  warn "DocVQA annotation archive not found: $DOCVQA_ANN_ARCHIVE"
fi

if [[ -f "$DOCVQA_IMG_ARCHIVE" ]]; then
  check_archive_type "$DOCVQA_IMG_ARCHIVE" "gzip compressed|tar archive" "DocVQA images archive"
else
  warn "DocVQA image archive not found: $DOCVQA_IMG_ARCHIVE"
fi

if [[ -d "$DOCVQA_ANN_DIR" ]]; then
  local_json_count="$(find "$DOCVQA_ANN_DIR" -type f -name "*.json" 2>/dev/null | wc -l | tr -d ' ')"
  echo "[INFO] DocVQA annotation json count = ${local_json_count}"
  if [[ "$local_json_count" -gt 0 ]]; then
    pass "DocVQA annotations extracted"
    find "$DOCVQA_ANN_DIR" -type f -name "*.json" -lh | head -20 || true
  else
    warn "DocVQA annotations directory exists, but no json files found"
  fi
else
  warn "DocVQA annotations directory not found"
fi

if [[ -d "$DOCVQA_IMG_DIR" ]]; then
  check_image_count "$DOCVQA_IMG_DIR" 100 "DocVQA extracted images"
else
  warn "DocVQA images directory not found"
fi

# ------------------------------------------------------------
# ChartQA
# ------------------------------------------------------------
check_git_lfs_repo "${RAW_DIR}/chartqa" "ChartQA"
check_parquet_count "${RAW_DIR}/chartqa" 5 "ChartQA"

# ------------------------------------------------------------
# OCRBench_v2
# ------------------------------------------------------------
check_git_lfs_repo "${RAW_DIR}/ocrbench_v2" "OCRBench_v2"
check_parquet_count "${RAW_DIR}/ocrbench_v2" 11 "OCRBench_v2"

# ------------------------------------------------------------
# OmniDocBench
# ------------------------------------------------------------
check_git_lfs_repo "${RAW_DIR}/omnidocbench" "OmniDocBench"
exists_file "${RAW_DIR}/omnidocbench/OmniDocBench.json" "OmniDocBench.json"
check_not_lfs_pointer "${RAW_DIR}/omnidocbench/OmniDocBench.json" "OmniDocBench.json"
check_image_count "${RAW_DIR}/omnidocbench/images" 100 "OmniDocBench images"

# ------------------------------------------------------------
# VisionArena-Bench
# ------------------------------------------------------------
check_git_lfs_repo "${RAW_DIR}/visionarena_bench" "VisionArena-Bench"
check_parquet_count "${RAW_DIR}/visionarena_bench" 1 "VisionArena-Bench"

# ------------------------------------------------------------
# Optional: Qwen2.5-VL-3B-Instruct model
# ------------------------------------------------------------
if [[ -d "$MODEL_DIR" ]]; then
  check_git_lfs_repo "$MODEL_DIR" "Qwen2.5-VL-3B-Instruct model"

  print_header "Model file sanity"
  exists_file "${MODEL_DIR}/config.json" "model config.json"
  exists_file "${MODEL_DIR}/tokenizer.json" "model tokenizer.json"

  safetensor_count="$(find "$MODEL_DIR" -maxdepth 2 -type f -name "*.safetensors" 2>/dev/null | wc -l | tr -d ' ')"
  echo "[INFO] model safetensors count = ${safetensor_count}"

  if [[ "$safetensor_count" -gt 0 ]]; then
    pass "model safetensors found"
    find "$MODEL_DIR" -maxdepth 2 -type f -name "*.safetensors" -lh | sort || true
  else
    fail "model safetensors not found"
  fi
else
  warn "Qwen2.5-VL-3B-Instruct model directory not found, skipping model check: ${MODEL_DIR}"
fi

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
print_header "Summary"

echo "[INFO] Warnings: ${WARN}"
echo "[INFO] Failures: ${FAIL}"

echo ""
echo "[INFO] Raw size summary:"
du -sh "${RAW_DIR}"/* 2>/dev/null || true

if [[ "$FAIL" -eq 0 ]]; then
  echo ""
  echo "[PASS] Raw dataset check completed with no hard failures."
  if [[ "$WARN" -gt 0 ]]; then
    echo "[WARN] There are warnings. Check optional/missing datasets above."
  fi
  exit 0
else
  echo ""
  echo "[FAIL] Raw dataset check found ${FAIL} hard failure(s)."
  exit 1
fi
