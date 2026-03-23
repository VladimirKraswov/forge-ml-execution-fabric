#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  install_with_local_wheels.sh <python_bin> --requirements <file> [wheel_dir]"
  echo "  install_with_local_wheels.sh <python_bin> --package <spec> [wheel_dir]"
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

find_local_wheels() {
  "${PYTHON_BIN}" - "$MODE" "$TARGET" "$WHEEL_DIR" <<'PY'
import glob
import json
import os
import sys

mode = sys.argv[1]
target = sys.argv[2]
wheel_dir = sys.argv[3]

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename
except Exception:
    print("[]")
    raise SystemExit(0)

def extract_req_names_from_file(path: str):
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r ") or line.startswith("--requirement "):
                continue
            if line.startswith("--"):
                continue
            if "://" in line:
                continue
            if line.startswith(".") or line.startswith("/"):
                continue
            try:
                req = Requirement(line)
            except Exception:
                continue
            names.append(canonicalize_name(req.name))
    return names

def extract_req_names(mode_value: str, target_value: str):
    if mode_value == "--requirements":
        return extract_req_names_from_file(target_value)
    if mode_value == "--package":
        try:
            req = Requirement(target_value)
            return [canonicalize_name(req.name)]
        except Exception:
            return []
    return []

req_names = set(extract_req_names(mode, target))
if not req_names:
    print("[]")
    raise SystemExit(0)

matched = []
for wheel_path in sorted(glob.glob(os.path.join(wheel_dir, "*.whl"))):
    try:
        name, version, build, tags = parse_wheel_filename(os.path.basename(wheel_path))
        norm_name = canonicalize_name(name)
    except Exception:
        continue
    if norm_name in req_names:
        matched.append(os.path.abspath(wheel_path))

print(json.dumps(matched, ensure_ascii=False))
PY
}

LOCAL_WHEELS_JSON="$(find_local_wheels)"
LOCAL_WHEELS=$("${PYTHON_BIN}" - <<'PY' "$LOCAL_WHEELS_JSON"
import json
import sys

items = json.loads(sys.argv[1])
for item in items:
    print(item)
PY
)

if [ -n "${LOCAL_WHEELS}" ]; then
  echo "==> installing local wheels first from ${WHEEL_DIR}"
  while IFS= read -r wheel_path; do
    [ -z "${wheel_path}" ] && continue
    echo "==> local wheel: ${wheel_path}"
    "${PYTHON_BIN}" -m pip install \
      --find-links "${WHEEL_DIR}" \
      "${wheel_path}" \
      --break-system-packages
  done <<< "${LOCAL_WHEELS}"
else
  echo "==> no matching local wheels found in ${WHEEL_DIR}"
fi

if [ "${MODE}" = "--requirements" ]; then
  echo "==> installing requirements from ${TARGET} with local wheel fallback"
  "${PYTHON_BIN}" -m pip install \
    --find-links "${WHEEL_DIR}" \
    -r "${TARGET}" \
    --break-system-packages
elif [ "${MODE}" = "--package" ]; then
  echo "==> installing package ${TARGET} with local wheel fallback"
  "${PYTHON_BIN}" -m pip install \
    --find-links "${WHEEL_DIR}" \
    "${TARGET}" \
    --break-system-packages
else
  echo "ERROR: unsupported mode ${MODE}"
  exit 1
fi