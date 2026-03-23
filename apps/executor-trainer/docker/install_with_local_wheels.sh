#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  install_with_local_wheels.sh <python_bin> --requirements <file> [wheel_dir]"
  echo "  install_with_local_wheels.sh <python_bin> --package <spec_or_requirements_file> [wheel_dir]"
  exit 1
fi

PYTHON_BIN="$1"
MODE="$2"
TARGET="${3:-}"
WHEEL_DIR="${4:-/trainer/packages/pip}"

if [ -z "${TARGET}" ]; then
  echo "ERROR: missing target for ${MODE}"
  exit 1
fi

mkdir -p "${WHEEL_DIR}"

echo "==> offline install"
echo "==> python: ${PYTHON_BIN}"
echo "==> wheel dir: ${WHEEL_DIR}"
find "${WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' | sort || true

if ! find "${WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' | grep -q .; then
  echo "ERROR: no local wheels found in ${WHEEL_DIR}"
  exit 1
fi

if [ "${MODE}" = "--requirements" ]; then
  echo "==> installing from requirements file: ${TARGET}"
  "${PYTHON_BIN}" -m pip install \
    --no-index \
    --find-links "${WHEEL_DIR}" \
    --prefer-binary \
    -r "${TARGET}" \
    --break-system-packages
elif [ "${MODE}" = "--package" ]; then
  echo "==> installing package/spec: ${TARGET}"
  "${PYTHON_BIN}" -m pip install \
    --no-index \
    --find-links "${WHEEL_DIR}" \
    --prefer-binary \
    "${TARGET}" \
    --break-system-packages
else
  echo "ERROR: unsupported mode ${MODE}"
  exit 1
fi