#!/bin/bash
#SBATCH --job-name=acsrag-replicas
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
mkdir -p runs/realgen_replicas

# K=3 replicas with maxent + CoT (default best-stack candidate)
out=runs/realgen_replicas/k3_maxent_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_replica_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --selector maxent --beta 2.0 --lambda_f 0.5 \
        --n_replicas 3 \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

# K=5 replicas (more averaging, more cost)
out=runs/realgen_replicas/k5_maxent_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_replica_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --selector maxent --beta 2.0 --lambda_f 0.5 \
        --n_replicas 5 \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

# K=3 with score-greedy selector (control: rules out selector interaction)
out=runs/realgen_replicas/k3_greedy_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_replica_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --selector greedy \
        --n_replicas 3 \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

echo "[replicas] done"
