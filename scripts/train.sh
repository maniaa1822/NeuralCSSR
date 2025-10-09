#!/usr/bin/env bash

# Wrapper to run nanoGPT/train.py from the repo root without manual
# environment fiddling.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_or_args...>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv_cache}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}/nanoGPT"
uv run python train.py "$@"
