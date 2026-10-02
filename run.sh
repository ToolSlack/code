#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
artifact_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$artifact_root"
if [[ -x "$artifact_root/.venv/bin/python" ]]; then
  artifact_python="$artifact_root/.venv/bin/python"
elif [[ -n "${TOOLSLACK_BOOTSTRAP_PYTHON:-}" ]]; then
  artifact_python="$TOOLSLACK_BOOTSTRAP_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  artifact_python="$(command -v python3.12)"
elif command -v python3 >/dev/null 2>&1; then
  artifact_python="$(command -v python3)"
else
  printf '%s\n' 'Python 3.9 or newer is required to bootstrap the pinned Python 3.12 environment.' >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then set -- cpu; fi
exec "$artifact_python" -m artifact.runner "$@"
