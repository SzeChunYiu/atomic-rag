#!/bin/bash
#SBATCH --job-name=acsrag-v4-sweep
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

# v4 sweep: 8 HotpotQA configs (cs ∈ {256, 384} × alpha ∈ {0.3, 0.5, 0.7, 0.9}).
CONFIGS=(
  "configs/benchmarks/chunksize_sweep_v4/hotpotqa_1k_cs256_v4a0_3.yaml"
  "configs/benchmarks/chunksize_sweep_v4/hotpotqa_1k_cs256_v4a0_5.yaml"
  "configs/benchmarks/chunksize_sweep_v4/hotpotqa_1k_cs256_v4a0_7.yaml"
  "configs/benchmarks/chunksize_sweep_v4/hotpotqa_1k_cs256_v4a0_9.yaml"
  "configs/benchmarks/chunksize_sweep_v4/hotpotqa_1k_cs384_v4a0_3.yaml"
  "configs/benchmarks/chunksize_sweep_v4/hotpotqa_1k_cs384_v4a0_5.yaml"
  "configs/benchmarks/chunksize_sweep_v4/hotpotqa_1k_cs384_v4a0_7.yaml"
  "configs/benchmarks/chunksize_sweep_v4/hotpotqa_1k_cs384_v4a0_9.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "==== running ${cfg} ===="
  rag-run-benchmark --config "${cfg}" || echo "FAILED: ${cfg}"
done

echo "[acsrag] v4 hotpotqa sweep done"

# Synthetic IRC v4 sweep — verifies C2a (preserve +0.21).
echo "==== synthetic v4 sweep ===="
python scripts/synthetic_irc_v4_sweep.py --n-seeds 3 --chunk-steps 8

echo "[acsrag] v4 synthetic sweep done"
