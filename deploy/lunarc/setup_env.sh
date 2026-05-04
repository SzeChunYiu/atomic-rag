#!/bin/bash
# One-time conda env creation on LUNARC for the Astro-CS-RAG project.
# Idempotent: re-running updates the env in place.
set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
CONDA_ENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag"
PYVER="3.11"

module purge
module load Anaconda3
module load CUDA/12.1.1

mkdir -p "$(dirname "${CONDA_ENV}")"

if [[ ! -d "${CONDA_ENV}" ]]; then
  conda create -y -p "${CONDA_ENV}" "python=${PYVER}"
fi

set +u
conda activate "${CONDA_ENV}"
set -u

# Project deps + dev + hf extras.
pip install --upgrade pip
pip install -e "${PROJECT_DIR}[dev,hf]"

# Torch with CUDA wheels matching the loaded CUDA module.
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121

# Ollama (linux installer).
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Pull the canonical generators once (cached under OLLAMA_MODELS at run time).
export OLLAMA_MODELS="${PROJECT_DIR}/models/ollama"
mkdir -p "${OLLAMA_MODELS}"
nohup ollama serve >/dev/null 2>&1 &
sleep 5
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct
# 70B sanity, requires the 80 GB A100 — only pull on demand:
# ollama pull llama3.1:70b-instruct-q4_K_M

echo "[acsrag] env ready at ${CONDA_ENV}"
