#!/bin/bash
#SBATCH --job-name=acsrag-full-stack
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

INDEX_DIR=runs/realgen_terse/hotpotqa_1k_cs384_greedy_qwen7b/index_bundle
ATOMS_DIR=${INDEX_DIR}/atoms

mkdir -p runs/realgen_full_stack

# Single-experiment full stack: RRF retrieval + MaxEnt selection
# (best beta/lambda_f from sweep — using sane default 2.0/0.5 here)
# + CoT prompting. This is the candidate paper-headline stack.
out=runs/realgen_full_stack/rrf_maxent_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_maxent_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --beta 2.0 --lambda_f 0.5 --mu 0.001 \
        --hybrid rrf \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

# Ablation: dense-only retrieval, same selector + prompt
out=runs/realgen_full_stack/dense_maxent_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_maxent_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --beta 2.0 --lambda_f 0.5 --mu 0.001 \
        --hybrid dense \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

# Ablation: linear fusion alpha=0.7
out=runs/realgen_full_stack/linear_a0.7_maxent_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_maxent_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --beta 2.0 --lambda_f 0.5 --mu 0.001 \
        --hybrid linear --hybrid_alpha 0.7 \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

echo "[full-stack] done"
