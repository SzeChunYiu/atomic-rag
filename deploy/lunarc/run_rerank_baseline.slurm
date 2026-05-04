#!/bin/bash
#SBATCH --job-name=acsrag-rerank-baseline
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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

# Cross-encoder rerank baseline: dense retrieval → BGE-reranker-v2-m3 → greedy → Qwen 7B.
# This is the gold-standard strong baseline RAG papers compare against.
echo "==== running cross-encoder rerank baseline ===="
rag-run-benchmark --config configs/benchmarks/realgen/hotpotqa_1k_cs384_greedy_rerank_qwen7b.yaml || echo "FAILED"

echo "[acsrag] rerank baseline done"
