#!/usr/bin/env bash

# Convenience wrapper for running the multitask OOD evaluator without
# manually exporting PYTHONPATH/UV_CACHE_DIR. Accepts the same arguments as
# nanoGPT/mtl/eval_ood.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# ensure uv caches into a writable directory when invoked inside the repo
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv_cache}"

# ensure local packages (machines, nanoGPT, etc.) are importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

uv run "${REPO_ROOT}/nanoGPT/mtl/eval_ood.py" "$@"
