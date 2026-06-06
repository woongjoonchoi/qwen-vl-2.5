#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$(pwd)}"

DOCVQA_DIR="${PROJECT_ROOT}/data/raw/docvqa"
DOWNLOAD_DIR="${DOCVQA_DIR}/downloads"
ANNOTATION_DIR="${DOCVQA_DIR}/annotations"
IMAGE_DIR="${DOCVQA_DIR}/images"

ANNOTATION_FILE="${DOWNLOAD_DIR}/spdocvqa_qas.zip"
IMAGE_FILE="${DOWNLOAD_DIR}/spdocvqa_images.tar.gz"

mkdir -p "${ANNOTATION_DIR}"
mkdir -p "${IMAGE_DIR}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] DocVQA dir: ${DOCVQA_DIR}"

echo "[INFO] Checking archives..."
file "${ANNOTATION_FILE}"
file "${IMAGE_FILE}"

echo "[INFO] Listing annotation zip contents..."
unzip -l "${ANNOTATION_FILE}" | head -50

echo "[INFO] Listing image tar.gz contents..."
tar -tzf "${IMAGE_FILE}" | head -50

echo "[INFO] Extracting annotations..."
unzip -o "${ANNOTATION_FILE}" -d "${ANNOTATION_DIR}"

echo "[INFO] Extracting images..."
tar -xzf "${IMAGE_FILE}" -C "${IMAGE_DIR}"

echo "[INFO] Checking extracted annotations..."
find "${ANNOTATION_DIR}" -type f | head -50
echo "[INFO] JSON count:"
find "${ANNOTATION_DIR}" -type f -name "*.json" | wc -l

echo "[INFO] Checking extracted images..."
find "${IMAGE_DIR}" -type f | head -50
echo "[INFO] Image count:"
find "${IMAGE_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l

echo "[INFO] Size summary:"
du -sh "${DOCVQA_DIR}"
du -sh "${ANNOTATION_DIR}"
du -sh "${IMAGE_DIR}"

echo "[DONE] DocVQA extraction complete."
