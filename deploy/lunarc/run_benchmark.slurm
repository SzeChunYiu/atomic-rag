#!/bin/bash
#SBATCH --job-name=acsrag-bench
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- environment ---------------------------------------------------------
# Modules: pin versions so reruns are byte-stable.
module purge
module load Anaconda3
module load CUDA/12.1.1

CONDA_ENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag"
PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
RUN_ROOT="${PROJECT_DIR}/runs"

set +u
conda activate "${CONDA_ENV}"
set -u

cd "${PROJECT_DIR}"

# ---- inputs --------------------------------------------------------------
# CONFIG : path (relative to repo) to the benchmark YAML.
# RUN_TAG: optional human label; affects log naming only.
CONFIG="${CONFIG:-configs/benchmarks/hotpotqa_1k_dense_rerank.yaml}"
RUN_TAG="${RUN_TAG:-default}"

echo "[acsrag] node     = $(hostname)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "[acsrag] config   = ${CONFIG}"
echo "[acsrag] run_tag  = ${RUN_TAG}"
echo "[acsrag] commit   = $(git -C "${PROJECT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'no-git')"

# ---- ensure ollama (for generation) -------------------------------------
# Ollama is started as a subprocess on the compute node; the model digest
# is pinned by the OLLAMA_MODELS variable to the project's models cache.
export OLLAMA_MODELS="${PROJECT_DIR}/models/ollama"
mkdir -p "${OLLAMA_MODELS}"
nohup ollama serve >/dev/null 2>&1 &
OLLAMA_PID=$!
trap "kill ${OLLAMA_PID} 2>/dev/null || true" EXIT
sleep 5
ollama list || true

# ---- run ---------------------------------------------------------------
rag-run-benchmark --config "${CONFIG}"
