#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${ROOT_DIR}/packages/pip"

# Можно переопределить:
#   PREFETCH_IMAGE=python:3.12-slim
#   PREFETCH_PLATFORM=linux/amd64
PREFETCH_IMAGE="${PREFETCH_IMAGE:-python:3.12-slim}"
PREFETCH_PLATFORM="${PREFETCH_PLATFORM:-linux/amd64}"
CLEAN_WHEELS="${CLEAN_WHEELS:-1}"

mkdir -p "${WHEEL_DIR}"

if [ "${CLEAN_WHEELS}" = "1" ]; then
  echo "==> cleaning old wheels in ${WHEEL_DIR}"
  find "${WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -delete
fi

echo "==> prefetch image: ${PREFETCH_IMAGE}"
echo "==> prefetch platform: ${PREFETCH_PLATFORM}"
echo "==> wheel dir: ${WHEEL_DIR}"

docker run --rm \
  --platform "${PREFETCH_PLATFORM}" \
  -v "${ROOT_DIR}:/work" \
  -w /work \
  "${PREFETCH_IMAGE}" \
  bash -lc '
    set -euo pipefail
    python -m pip install -U pip setuptools wheel

    echo "==> downloading train env wheels"
    python -m pip download \
      --dest /work/packages/pip \
      --only-binary=:all: \
      --prefer-binary \
      -r /work/requirements.txt

    echo "==> downloading vllm env wheels"
    python -m pip download \
      --dest /work/packages/pip \
      --only-binary=:all: \
      --prefer-binary \
      -r /work/requirements.vllm.txt

    echo "==> wheels downloaded:"
    find /work/packages/pip -maxdepth 1 -type f -name "*.whl" | sort
  '

echo "==> prefetch complete"