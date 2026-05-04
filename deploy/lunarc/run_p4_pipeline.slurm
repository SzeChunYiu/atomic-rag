#!/bin/bash
#SBATCH --job-name=acsrag-p4-pipeline
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

echo "[acsrag] node     = $(hostname -s)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Stage 1 — paraphrase cache (Qwen 2.5-7B, n=4 paraphrases per query, ~30 min for 1000 queries).
echo "==== Stage 1: paraphrase caching ===="
PARA_CACHE=data/hotpotqa_1k/paraphrases_qwen7b_n4.jsonl
if [ ! -f "${PARA_CACHE}" ] || [ "$(wc -l < "${PARA_CACHE}")" -lt 1000 ]; then
  python scripts/cache_paraphrases.py \
    --queries data/hotpotqa_1k/queries.jsonl \
    --out "${PARA_CACHE}" \
    --n 4 \
    --provider transformers \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --temperature 0.7
else
  echo "  cache exists with $(wc -l < "${PARA_CACHE}") lines, skipping"
fi
echo "  cache size: $(wc -l < "${PARA_CACHE}") lines"

# Stage 2 — lock-in benchmark vs fusion_rrf baseline (stub generator first; cheap).
echo "==== Stage 2: lock-in retrieval comparison ===="
CONFIGS=(
  "configs/benchmarks/p4_lockin/hotpotqa_1k_cs384_fusion_rrf_baseline.yaml"
  "configs/benchmarks/p4_lockin/hotpotqa_1k_cs384_lockin_n4.yaml"
)
for cfg in "${CONFIGS[@]}"; do
  echo "  running ${cfg}"
  rag-run-benchmark --config "${cfg}" || echo "FAILED: ${cfg}"
done

echo "[acsrag] P4 pipeline done"
