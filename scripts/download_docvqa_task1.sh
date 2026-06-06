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

ANNOTATION_URL="https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazEvc3Bkb2N2cWFfcWFzLnppcA=="
IMAGE_URL="https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazEvc3Bkb2N2cWFfaW1hZ2VzLnRhci5neg=="

ANNOTATION_FILE="${DOWNLOAD_DIR}/spdocvqa_qas.zip"
IMAGE_FILE="${DOWNLOAD_DIR}/spdocvqa_images.tar.gz"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] DocVQA dir: ${DOCVQA_DIR}"
echo "[INFO] Download dir: ${DOWNLOAD_DIR}"

download_with_check() {
  local url="$1"
  local out="$2"
  local name="$3"

  echo "[INFO] Preparing download: ${name}"
  echo "[INFO] Output: ${out}"

  # 이전에 HTML 로그인 페이지 같은 잘못된 파일이 받아진 경우 제거
  if [[ -f "${out}" ]]; then
    if file "${out}" | grep -qi "HTML"; then
      echo "[WARN] Existing file looks like HTML, removing: ${out}"
      rm -f "${out}"
    fi
  fi

  wget \
    --no-check-certificate \
    --continue \
    --tries=20 \
    --timeout=30 \
    --waitretry=10 \
    --output-document="${out}" \
    "${url}"

  echo "[INFO] Downloaded ${name}:"
  ls -lh "${out}"
  file "${out}"
}

echo "[INFO] Downloading DocVQA Task 1 annotations..."
download_with_check "${ANNOTATION_URL}" "${ANNOTATION_FILE}" "annotations"

echo "[INFO] Downloading DocVQA Task 1 images..."
download_with_check "${IMAGE_URL}" "${IMAGE_FILE}" "images"

echo "[INFO] Verifying downloaded file types..."

if ! file "${ANNOTATION_FILE}" | grep -qi "Zip archive"; then
  echo "[ERROR] Annotation file is not a valid zip archive."
  echo "[ERROR] It may be an HTML login page or failed download."
  echo "[ERROR] Check with: file ${ANNOTATION_FILE}"
  exit 1
fi

if ! file "${IMAGE_FILE}" | grep -Eqi "gzip compressed|tar archive"; then
  echo "[ERROR] Image file is not a valid tar.gz archive."
  echo "[ERROR] It may be an HTML login page or failed download."
  echo "[ERROR] Check with: file ${IMAGE_FILE}"
  exit 1
fi

echo "[INFO] Extracting annotations..."
unzip -o "${ANNOTATION_FILE}" -d "${ANNOTATION_DIR}"

echo "[INFO] Extracting images..."
tar -xzf "${IMAGE_FILE}" -C "${IMAGE_DIR}"

echo "[INFO] Extraction complete."

echo "[INFO] Final structure:"
find "${DOCVQA_DIR}" -maxdepth 3 -type d | sort

echo "[INFO] File size summary:"
du -sh "${DOCVQA_DIR}" || true
du -sh "${DOWNLOAD_DIR}" || true
du -sh "${ANNOTATION_DIR}" || true
du -sh "${IMAGE_DIR}" || true

echo "[DONE] DocVQA Task 1 annotations and images are ready."
