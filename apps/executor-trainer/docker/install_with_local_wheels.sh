#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
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

echo "==> install mode: local-first, internet fallback"
echo "==> python: ${PYTHON_BIN}"
echo "==> wheel dir: ${WHEEL_DIR}"
find "${WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' | sort || true

TMP_WHEEL_LIST="$(mktemp)"
trap 'rm -f "${TMP_WHEEL_LIST}"' EXIT

# Выбираем только совместимые с текущим Python/ABI/platform колёса.
# Если для одного пакета есть несколько версий, берём самую новую локальную.
"${PYTHON_BIN}" - "${WHEEL_DIR}" > "${TMP_WHEEL_LIST}" <<'PY'
import sys
from pathlib import Path

from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
from pip._vendor.packaging.version import Version

wheel_dir = Path(sys.argv[1])
supported_tags = set(sys_tags())
best = {}

for wheel_path in sorted(wheel_dir.glob("*.whl")):
    try:
        name, version, build_tag, tags = parse_wheel_filename(wheel_path.name)
    except Exception:
        continue

    if not (set(tags) & supported_tags):
        continue

    key = canonicalize_name(str(name))
    version_obj = Version(str(version))

    prev = best.get(key)
    if prev is None or version_obj > prev[0]:
        best[key] = (version_obj, wheel_path)

for key in sorted(best):
    print(str(best[key][1]))
PY

mapfile -t LOCAL_WHEELS < "${TMP_WHEEL_LIST}"

echo "==> compatible local wheels selected: ${#LOCAL_WHEELS[@]}"
for wheel in "${LOCAL_WHEELS[@]:0:200}"; do
  echo "==> selected: ${wheel}"
done

# Шаг 1. Предустанавливаем всё, что нашли локально.
# Без зависимостей, чтобы сначала максимально насытить env локальными wheel-файлами.
if [ "${#LOCAL_WHEELS[@]}" -gt 0 ]; then
  "${PYTHON_BIN}" -m pip install \
    --no-index \
    --no-deps \
    --prefer-binary \
    "${LOCAL_WHEELS[@]}" \
    --break-system-packages
else
  echo "==> no compatible local wheels found, will use internet only"
fi

COMMON_ARGS=(
  --find-links "${WHEEL_DIR}"
  --prefer-binary
  --upgrade-strategy only-if-needed
  --break-system-packages
)

# Шаг 2. Доставляем недостающее.
# Pip сначала использует уже установленные пакеты, потом wheelhouse, и только потом интернет.
if [ "${MODE}" = "--requirements" ]; then
  echo "==> resolving requirements with local-first fallback: ${TARGET}"
  "${PYTHON_BIN}" -m pip install \
    "${COMMON_ARGS[@]}" \
    -r "${TARGET}"
elif [ "${MODE}" = "--package" ]; then
  echo "==> resolving package with local-first fallback: ${TARGET}"
  "${PYTHON_BIN}" -m pip install \
    "${COMMON_ARGS[@]}" \
    "${TARGET}"
else
  echo "ERROR: unsupported mode ${MODE}"
  exit 1
fi