#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${ROOT_DIR}/.venv/bin/activate"

if [[ ! -f "${VENV_PATH}" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}" >&2
  exit 1
fi

source "${VENV_PATH}"

PIPELINE_PID=""
ADMIN_PID=""

cleanup() {
  if [[ -n "${PIPELINE_PID}" ]] && ps -p "${PIPELINE_PID}" > /dev/null 2>&1; then
    kill "${PIPELINE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ADMIN_PID}" ]] && ps -p "${ADMIN_PID}" > /dev/null 2>&1; then
    kill "${ADMIN_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

SHOULD_RUN_PIPELINE="$(python - <<'PY'
import json, os, sys
from pathlib import Path

config_path = Path(os.environ.get("PIPELINE_CONFIG_PATH", "config.json")).expanduser()
try:
    data = json.loads(config_path.read_text(encoding="utf-8"))
except Exception:
    data = {}
auto_process = bool(data.get("pipeline", {}).get("auto_process_inbox", False))
sys.stdout.write("true" if auto_process else "false")
PY
)"

if [[ "${SHOULD_RUN_PIPELINE}" == "true" ]]; then
  echo "Starting pipeline (python -m onedrive_ollama_pipeline.cli)"
  python -m onedrive_ollama_pipeline.cli &
  PIPELINE_PID=$!
else
  echo "Auto-processing disabled; skipping pipeline startup."
fi

echo "Starting admin UI (uvicorn onedrive_ollama_pipeline.admin_app:app)"
uvicorn onedrive_ollama_pipeline.admin_app:app --host 127.0.0.1 --port 8000 &
ADMIN_PID=$!

exit_code=0
stopped_process=""

capitalize() {
  local input="${1:-}"
  if [[ -z "${input}" ]]; then
    printf 'process'
    return
  fi
  local first="${input:0:1}"
  local rest="${input:1}"
  first="$(printf '%s' "${first}" | tr '[:lower:]' '[:upper:]')"
  printf '%s%s' "${first}" "${rest}"
}

while true; do
  if [[ -n "${PIPELINE_PID}" ]]; then
    if ! ps -p "${PIPELINE_PID}" > /dev/null 2>&1; then
      wait "${PIPELINE_PID}" 2>/dev/null || exit_code=$?
      echo "Pipeline process exited (code ${exit_code}). Admin UI will continue."
      PIPELINE_PID=""
    fi
  fi
  if ! ps -p "${ADMIN_PID}" > /dev/null 2>&1; then
    wait "${ADMIN_PID}" 2>/dev/null || exit_code=$?
    stopped_process="admin"
    break
  fi
  sleep 1
done

echo "$(capitalize "${stopped_process}") process exited (code ${exit_code}); shutting down remaining services."
cleanup

# Ensure background processes are fully reaped before exiting.
if [[ -n "${PIPELINE_PID}" ]]; then
  wait "${PIPELINE_PID}" 2>/dev/null || true
fi
wait "${ADMIN_PID}" 2>/dev/null || true

exit ${exit_code}
