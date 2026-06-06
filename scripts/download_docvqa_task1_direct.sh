#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"

DOCVQA_DIR="${PROJECT_ROOT}/data/raw/docvqa"
DOWNLOAD_DIR="${DOCVQA_DIR}/downloads"
ANNOTATION_DIR="${DOCVQA_DIR}/annotations"
IMAGE_DIR="${DOCVQA_DIR}/images"

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${ANNOTATION_DIR}"
mkdir -p "${IMAGE_DIR}"

ANNOTATION_URL="https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_qas.zip"
IMAGE_URL="https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_images.tar.gz"

ANNOTATION_FILE="${DOWNLOAD_DIR}/spdocvqa_qas.zip"
IMAGE_FILE="${DOWNLOAD_DIR}/spdocvqa_images.tar.gz"

echo "[INFO] Removing invalid tiny files if they exist..."
if [[ -f "${ANNOTATION_FILE}" ]] && [[ $(stat -c%s "${ANNOTATION_FILE}") -lt 100000 ]]; then
  echo "[WARN] Removing tiny annotation file: ${ANNOTATION_FILE}"
  rm -f "${ANNOTATION_FILE}"
fi

if [[ -f "${IMAGE_FILE}" ]] && [[ $(stat -c%s "${IMAGE_FILE}") -lt 100000 ]]; then
  echo "[WARN] Removing tiny image file: ${IMAGE_FILE}"
  rm -f "${IMAGE_FILE}"
fi

download_file() {
  local url="$1"
  local out="$2"
  local name="$3"

  echo "[INFO] Downloading ${name}"
  echo "[INFO] URL: ${url}"
  echo "[INFO] OUT: ${out}"

  wget \
    --no-check-certificate \
    --continue \
    --tries=20 \
    --timeout=60 \
    --waitretry=10 \
    --output-document="${out}" \
    "${url}"

  echo "[INFO] Downloaded ${name}:"
  ls -lh "${out}"
  file "${out}"
}

download_file "${ANNOTATION_URL}" "${ANNOTATION_FILE}" "DocVQA annotations"
download_file "${IMAGE_URL}" "${IMAGE_FILE}" "DocVQA images"

echo "[INFO] Verifying archives..."

if ! file "${ANNOTATION_FILE}" | grep -qi "Zip archive"; then
  echo "[ERROR] Annotation file is not a valid zip archive."
  echo "[ERROR] Content preview:"
  head -c 500 "${ANNOTATION_FILE}" || true
  echo
  exit 1
fi

if ! file "${IMAGE_FILE}" | grep -Eqi "gzip compressed|tar archive"; then
  echo "[ERROR] Image file is not a valid tar.gz archive."
  echo "[ERROR] Content preview:"
  head -c 500 "${IMAGE_FILE}" || true
  echo
  exit 1
fi

echo "[INFO] Extracting annotations..."
unzip -o "${ANNOTATION_FILE}" -d "${ANNOTATION_DIR}"

echo "[INFO] Extracting images..."
tar -xzf "${IMAGE_FILE}" -C "${IMAGE_DIR}"

echo "[DONE] DocVQA Task 1 download and extraction complete."

echo "[INFO] Size summary:"
du -sh "${DOCVQA_DIR}" || true
du -sh "${DOWNLOAD_DIR}" || true
du -sh "${ANNOTATION_DIR}" || true
du -sh "${IMAGE_DIR}" || true
