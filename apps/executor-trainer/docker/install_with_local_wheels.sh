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

echo "==> wheel dir: ${WHEEL_DIR}"
find "${WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' | sort || true

find_local_wheels() {
  "${PYTHON_BIN}" - "$MODE" "$TARGET" "$WHEEL_DIR" <<'PY'
import glob
import json
import os
import re
import sys

mode = sys.argv[1]
target = sys.argv[2]
wheel_dir = sys.argv[3]

def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

def wheel_name_from_filename(path: str):
    base = os.path.basename(path)
    if not base.endswith(".whl"):
        return None
    # wheel format: {dist}-{version}(-{build})?-{py}-{abi}-{platform}.whl
    parts = base[:-4].split("-")
    if len(parts) < 5:
        return None
    return normalize_name(parts[0])

def parse_req_name(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("-r ") or line.startswith("--requirement "):
        return None
    if line.startswith("--"):
        return None
    if "://" in line:
        return None
    if line.startswith(".") or line.startswith("/"):
        return None

    # remove env markers
    line = line.split(";", 1)[0].strip()

    for sep in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
        if sep in line:
            line = line.split(sep, 1)[0].strip()
            break

    if "[" in line:
        line = line.split("[", 1)[0].strip()

    if not line:
        return None

    return normalize_name(line)

def extract_req_names(mode_value: str, target_value: str):
    if mode_value == "--requirements":
        names = []
        with open(target_value, "r", encoding="utf-8") as f:
            for raw_line in f:
                name = parse_req_name(raw_line)
                if name:
                    names.append(name)
        return names

    if mode_value == "--package":
        name = parse_req_name(target_value)
        return [name] if name else []

    return []

req_names = set(extract_req_names(mode, target))
if not req_names:
    print("[]")
    raise SystemExit(0)

matched = []
for wheel_path in sorted(glob.glob(os.path.join(wheel_dir, "*.whl"))):
    wheel_name = wheel_name_from_filename(wheel_path)
    if wheel_name and wheel_name in req_names:
        matched.append(os.path.abspath(wheel_path))

print(json.dumps(matched, ensure_ascii=False))
PY
}

LOCAL_WHEELS_JSON="$(find_local_wheels)"
LOCAL_WHEELS=$("${PYTHON_BIN}" - <<'PY' "$LOCAL_WHEELS_JSON"
import json
import sys

for item in json.loads(sys.argv[1]):
    print(item)
PY
)

if [ -n "${LOCAL_WHEELS}" ]; then
  echo "==> installing matching local wheels first"
  while IFS= read -r wheel_path; do
    [ -z "${wheel_path}" ] && continue
    echo "==> local wheel: ${wheel_path}"
    "${PYTHON_BIN}" -m pip install \
      --no-index \
      "${wheel_path}" \
      --break-system-packages
  done <<< "${LOCAL_WHEELS}"
else
  echo "==> no matching top-level local wheels found"
fi

if [ "${MODE}" = "--requirements" ]; then
  echo "==> installing requirements with local-wheel preference"
  "${PYTHON_BIN}" -m pip install \
    --prefer-binary \
    --find-links "${WHEEL_DIR}" \
    -r "${TARGET}" \
    --break-system-packages
elif [ "${MODE}" = "--package" ]; then
  echo "==> installing package ${TARGET} with local-wheel preference"
  "${PYTHON_BIN}" -m pip install \
    --prefer-binary \
    --find-links "${WHEEL_DIR}" \
    "${TARGET}" \
    --break-system-packages
else
  echo "ERROR: unsupported mode ${MODE}"
  exit 1
fi