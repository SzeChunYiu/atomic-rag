#!/bin/bash
#SBATCH --job-name=acsrag-cs-sweep
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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

# Linchpin experiment: HotpotQA × chunk_size × selector.
# Each chunk_size triplet (greedy, anti_kt, mmr) ≈ 1.5h on A100; budgeted 6h covers up to 4 sizes.
# Order from smallest (most likely to expose IRC advantage) to largest (control).
CONFIGS=(
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs64_greedy.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs64_anti_kt.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs64_mmr.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs128_greedy.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs128_anti_kt.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs128_mmr.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs256_greedy.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs256_anti_kt.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs256_mmr.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs384_greedy.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs384_anti_kt.yaml"
  "configs/benchmarks/chunksize_sweep/hotpotqa_1k_cs384_mmr.yaml"
)

echo "[acsrag] node     = $(hostname)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"

for CFG in "${CONFIGS[@]}"; do
  echo "==== running ${CFG} ===="
  rag-run-benchmark --config "${CFG}" || echo "FAILED: ${CFG}"
done

echo "[acsrag] chunksize sweep done"
