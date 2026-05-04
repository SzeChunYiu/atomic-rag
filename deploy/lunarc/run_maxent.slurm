#!/bin/bash
#SBATCH --job-name=acsrag-maxent
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

mkdir -p runs/realgen_maxent

# MaxEnt sweep over (beta, lambda_f) — Lagrange multiplier ratios
# beta=score weight, lambda_f=coverage weight, mu=token-cost penalty
for beta in 0.5 1.0 2.0 5.0; do
    for lf in 0.1 0.5 1.0; do
        out=runs/realgen_maxent/b${beta}_lf${lf}
        [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
        rm -rf "${out}"
        echo "==== beta=${beta} lambda_f=${lf} ===="
        python3 scripts/atom_maxent_bench.py \
            --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
            --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
            --lambda_type 0.05 --top_k 50 --token_budget 1024 \
            --beta "${beta}" --lambda_f "${lf}" --mu 0.001 \
            --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
    done
done

# Best-MaxEnt + CoT (will be re-tuned post-hoc)
out=runs/realgen_maxent/b2_lf0.5_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_maxent_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --beta 2.0 --lambda_f 0.5 --mu 0.001 \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

echo "[maxent] done"
