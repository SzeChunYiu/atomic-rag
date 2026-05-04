#!/bin/bash
#SBATCH --job-name=acsrag-d04
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
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

echo "[d04] node = $(hostname -s)"

# Reuse the existing index from the greedy/dense HotpotQA run
INDEX_DIR=runs/realgen_terse/hotpotqa_1k_cs384_greedy_qwen7b/index_bundle
ATOMS_DIR=${INDEX_DIR}/atoms

# Stage 1: build atoms (one-time per index)
if [ ! -f "${ATOMS_DIR}/atoms.jsonl" ]; then
    echo "==== building atoms ===="
    python3 scripts/build_atoms.py "${INDEX_DIR}" --out_dir "${ATOMS_DIR}" --batch_size 64
else
    echo "atoms already built at ${ATOMS_DIR}"
fi
ls -la "${ATOMS_DIR}/"
cat "${ATOMS_DIR}/atom_meta.json"

# Stage 2: bench D04-bare (lambda_type=0, no typed bonus)
mkdir -p runs/realgen_d04
for variant in bare typed; do
    out=runs/realgen_d04/hotpotqa_1k_cs384_d04_${variant}
    rm -rf "${out}"
    if [ "${variant}" = "bare" ]; then
        lt=0.0
    else
        lt=0.05
    fi
    echo "==== running D04-${variant} (lambda_type=${lt}) ===="
    python3 scripts/atom_retrieve_bench.py \
        --atoms_dir "${ATOMS_DIR}" \
        --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl \
        --out_dir "${out}" \
        --lambda_type "${lt}" \
        --top_k 50 \
        --token_budget 1024 \
        --max_new_tokens 32
done

echo "[d04] sweep done"
