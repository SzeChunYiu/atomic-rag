#!/bin/bash
# Push local repo to LUNARC and pull runs back. Idempotent rsync, no git on cluster needed.
set -euo pipefail

LOCAL="$(cd "$(dirname "$0")"/../.. && pwd)"
REMOTE="lunarc:/projects/hep/fs10/shared/nnbar/billy/RAG"

case "${1:-push}" in
  push)
    rsync -avz --delete \
      --exclude='/runs/' \
      --exclude='/data/' \
      --exclude='/.venv/' \
      --exclude='__pycache__/' \
      --exclude='*.egg-info/' \
      --exclude='/.pytest_cache/' \
      "${LOCAL}/" "${REMOTE}/"
    ;;
  pull-runs)
    mkdir -p "${LOCAL}/runs"
    rsync -avz "${REMOTE}/runs/" "${LOCAL}/runs/"
    ;;
  push-data)
    rsync -avz "${LOCAL}/data/" "${REMOTE}/data/"
    ;;
  *)
    echo "usage: $0 [push|pull-runs|push-data]" >&2
    exit 2
    ;;
esac
