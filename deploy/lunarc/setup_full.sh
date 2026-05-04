#!/bin/bash
# Full LUNARC setup: venv + hf extras + ollama + base model + tiny smoke.
set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"
OLLAMA_DIR="/projects/hep/fs10/shared/nnbar/billy/ollama"
export OLLAMA_MODELS="${PROJECT_DIR}/models/ollama"
export PIP_CACHE_DIR="/projects/hep/fs10/shared/nnbar/billy/pip_cache"

module purge
module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
set +u
source "${VENV}/bin/activate"
set -u

cd "${PROJECT_DIR}"
echo "[acsrag] env: $(which python) — $(python -V)"
pip install -e ".[dev,hf]" -q

# Install Ollama into project space (rootless tarball install).
mkdir -p "${OLLAMA_DIR}/bin" "${OLLAMA_DIR}/lib" "${OLLAMA_MODELS}"
if [[ ! -x "${OLLAMA_DIR}/bin/ollama" ]]; then
  echo "[acsrag] downloading ollama tarball..."
  TMPDIR_OLLAMA=$(mktemp -d)
  curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz \
      -o "${TMPDIR_OLLAMA}/ollama.tgz"
  tar -xzf "${TMPDIR_OLLAMA}/ollama.tgz" -C "${OLLAMA_DIR}"
  rm -rf "${TMPDIR_OLLAMA}"
fi
export PATH="${OLLAMA_DIR}/bin:${PATH}"
ollama --version || true

# Start ollama serve as a background daemon if not already running.
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  nohup "${OLLAMA_DIR}/bin/ollama" serve > "${PROJECT_DIR}/logs/ollama_serve.log" 2>&1 &
  sleep 6
fi

# Pull the canonical 8B generator (~4.9 GB).
ollama pull llama3.1:8b-instruct-q4_K_M

# Optional: also pull Qwen 2.5 7B for robustness comparisons (commented to save disk).
# ollama pull qwen2.5:7b-instruct

# Smoke test: end-to-end on the tiny corpus.
mkdir -p "${PROJECT_DIR}/data/tiny"
[ -f "${PROJECT_DIR}/data/tiny/corpus.jsonl" ] && echo "[acsrag] tiny data already present" || echo "[acsrag] WARNING: tiny data missing — sync data/tiny first"
rag-run-benchmark --config configs/benchmarks/tiny_stub.yaml | tail -1

echo "[acsrag] setup complete"
