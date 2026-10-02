#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_MODE="${1:-cpu}"
case "$TASK_MODE" in cpu|gpu|engine) ;; *) echo 'Usage: bootstrap.sh cpu|gpu|engine' >&2; exit 2;; esac
export UV_CACHE_DIR="$TASK_ROOT/.bootstrap/cache"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/.bootstrap/python"
export UV_PYTHON_BIN_DIR="$TASK_ROOT/.bootstrap/python-bin"
export PIP_CACHE_DIR="$TASK_ROOT/.bootstrap/pip-cache"

# The bootstrap is isolated from the caller's Python and global packages.
TASK_UV="${TOOLSLACK_UV:-$TASK_ROOT/.bootstrap/bin/uv}"
if [ ! -x "$TASK_UV" ]; then
  TASK_BASE_PYTHON="${TOOLSLACK_BOOTSTRAP_PYTHON:-python3}"
  "$TASK_BASE_PYTHON" "$TASK_ROOT/scripts/bootstrap_uv.py" --root "$TASK_ROOT"
fi
case "$("$TASK_UV" --version)" in
  'uv 0.12.22'|'uv 0.12.22 '*) ;;
  *) echo 'The artifact requires uv 0.12.22; use a clean extraction or the pinned installer.' >&2; exit 2;;
esac
if [ "$TASK_MODE" = engine ]; then
  if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    echo 'The frozen CUDA engine requires Linux x86_64. CPU validation runs on other platforms.' >&2
    exit 2
  fi
  TASK_VERSION=3.11.13
  TASK_ENV="$TASK_ROOT/.engine-venv"
  TASK_LOCK="$TASK_ROOT/requirements/engine.lock"
  TASK_PYTHON="${TOOLSLACK_ENGINE_PYTHON:-}"
else
  TASK_VERSION=3.12.14
  TASK_ENV="$TASK_ROOT/.venv"
  TASK_LOCK="$TASK_ROOT/requirements/cpu.lock"
  [ "$TASK_MODE" != gpu ] || TASK_LOCK="$TASK_ROOT/requirements/agent.lock"
  TASK_PYTHON="${TOOLSLACK_PYTHON:-}"
fi
if [ ! -f "$TASK_LOCK" ]; then echo 'A required dependency lock file is missing.' >&2; exit 2; fi
if [ -z "$TASK_PYTHON" ]; then
  TASK_PYTHON="$("$TASK_UV" python find "$TASK_VERSION" 2>/dev/null || true)"
  if [ -z "$TASK_PYTHON" ]; then
    "$TASK_UV" python install --no-bin "$TASK_VERSION"
    TASK_PYTHON="$("$TASK_UV" python find "$TASK_VERSION")"
  fi
fi
if [ ! -x "$TASK_ENV/bin/python" ]; then "$TASK_UV" venv --python "$TASK_PYTHON" "$TASK_ENV"; fi
"$TASK_ENV/bin/python" -c 'import platform, sys; actual=platform.python_version(); expected=sys.argv[1];
if actual != expected: raise SystemExit("Existing artifact environment uses Python " + actual + "; expected " + expected + ". Use a clean extraction or the pinned interpreter.")' "$TASK_VERSION"
"$TASK_UV" pip sync --python "$TASK_ENV/bin/python" --require-hashes "$TASK_LOCK"
if [ "$TASK_MODE" = engine ]; then
  "$TASK_UV" pip install --python "$TASK_ENV/bin/python" --no-deps --no-build-isolation \
    --editable "$TASK_ROOT/vendor/native_engine"
fi
"$TASK_ENV/bin/python" -c 'import sys; print("ToolSlack environment ready: Python " + sys.version.split()[0])'
