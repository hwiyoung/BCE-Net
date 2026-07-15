#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${BCENET_VENV:-$REPO_ROOT/.venv-bcenet-geo}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: BCE-Net environment is missing." >&2
  echo "Run: ./scripts/setup_env.sh" >&2
  exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/DCNv2${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -eq 0 ]]; then
  exec "$SHELL"
fi

exec "$@"
