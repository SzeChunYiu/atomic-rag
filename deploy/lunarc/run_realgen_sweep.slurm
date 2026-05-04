#!/bin/bash
#SBATCH --job-name=acsrag-realgen
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"
export HF_HOME=/projects/hep/fs10/shared/nnbar/billy/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets

module purge
module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
set +u
source "${VENV}/bin/activate"
set -u

cd "${PROJECT_DIR}"

echo "[acsrag] node     = $(hostname -s)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Real-generator sweep at cs=384: greedy / v4 α=0.7 / MMR. Each ~30 min.
CONFIGS=(
  "configs/benchmarks/realgen/hotpotqa_1k_cs384_greedy_qwen7b.yaml"
  "configs/benchmarks/realgen/hotpotqa_1k_cs384_v4a0_7_qwen7b.yaml"
  "configs/benchmarks/realgen/hotpotqa_1k_cs384_mmr_qwen7b.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "==== running ${cfg} ===="
  rag-run-benchmark --config "${cfg}" || echo "FAILED: ${cfg}"
done

echo "[acsrag] real-gen sweep done"
