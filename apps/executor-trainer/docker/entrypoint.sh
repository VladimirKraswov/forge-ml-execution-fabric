#!/usr/bin/env bash
set -euo pipefail

DEFAULT_LOCAL_CONFIG="/configs/job.local.json"

echo "==> trainer-service starting"

if [ "$#" -gt 0 ]; then
  echo "==> launching with explicit CLI args: $*"
  exec python /trainer/app/runner.py "$@"
fi

if [ -n "${JOB_CONFIG_URL:-}" ]; then
  echo "==> remote bootstrap via JOB_CONFIG_URL"
  echo "==> job config url: ${JOB_CONFIG_URL}"
  exec python /trainer/app/runner.py --job-config-url "${JOB_CONFIG_URL}"
fi

CONFIG_SOURCE="${CONFIG_SOURCE:-local}"
CONFIG_REF="${CONFIG_REF:-}"

if [ -n "${CONFIG_PATH:-}" ]; then
  CONFIG_REF="$CONFIG_PATH"
  CONFIG_SOURCE="local"
fi

echo "==> config source: ${CONFIG_SOURCE}"
echo "==> config ref: ${CONFIG_REF:-<empty>}"
echo "==> model path in image assumed at: /app"

if [ "${CONFIG_SOURCE}" = "remote" ]; then
  if [ -z "${CONFIG_REF}" ]; then
    echo "ERROR: CONFIG_REF is required when CONFIG_SOURCE=remote"
    exit 1
  fi

  exec python /trainer/app/runner.py --config "${CONFIG_REF}"
fi

if [ -z "${CONFIG_REF}" ]; then
  CONFIG_REF="${DEFAULT_LOCAL_CONFIG}"
fi

if [ ! -f "${CONFIG_REF}" ]; then
  echo "ERROR: local config file not found: ${CONFIG_REF}"
  exit 1
fi

exec python /trainer/app/runner.py --config "${CONFIG_REF}"