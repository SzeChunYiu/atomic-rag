#!/bin/bash
#SBATCH --job-name=acsrag-matrix
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-53

set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"

module purge
module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
set +u
source "${VENV}/bin/activate"
set -u

cd "${PROJECT_DIR}"

# Map array index → config file (alphabetical order over configs/benchmarks/matrix/).
mapfile -t CONFIGS < <(ls configs/benchmarks/matrix/*.yaml | sort)
N=${#CONFIGS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= N )); then
  echo "[acsrag] array index $IDX >= $N — exiting"
  exit 0
fi
CFG="${CONFIGS[$IDX]}"

echo "[acsrag] node     = $(hostname)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "[acsrag] config   = ${CFG}"
echo "[acsrag] N_total  = ${N}"

# Optional: start Ollama in the background if any config in this array uses it.
if grep -q 'provider: ollama' "${CFG}"; then
  export OLLAMA_MODELS="${PROJECT_DIR}/models/ollama"
  mkdir -p "${OLLAMA_MODELS}"
  if ! command -v ollama >/dev/null 2>&1; then
    echo "[acsrag] WARNING: ollama not installed — falling back to stub" >&2
    sed -i "s/provider: ollama/provider: stub/" "${CFG}"
  else
    nohup ollama serve >"${PROJECT_DIR}/logs/ollama_${SLURM_JOB_ID}_${IDX}.log" 2>&1 &
    OLLAMA_PID=$!
    trap "kill ${OLLAMA_PID} 2>/dev/null || true" EXIT
    sleep 5
  fi
fi

rag-run-benchmark --config "${CFG}"
