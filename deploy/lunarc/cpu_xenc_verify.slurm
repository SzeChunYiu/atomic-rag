#!/bin/bash
#SBATCH --job-name=acsrag-xenc-verify
#SBATCH --account=lu2026-2-51
#SBATCH --partition=lu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"
export HF_HOME=/projects/hep/fs10/shared/nnbar/billy/hf_cache
module purge
module load GCC/12.3.0 Python/3.11.3
set +u
source "${VENV}/bin/activate"
set -u
cd "${PROJECT_DIR}"

# Verify citations on D04+D06 CoT (best-known atom-level run)
python3 scripts/verify_citations_xenc.py \
    --run_dir runs/realgen_d06/d04_d06_cot \
    --queries data/hotpotqa_1k/queries.jsonl \
    --device cpu --batch_size 8

# Also on D04 bare (greedy, no submodular) for comparison
python3 scripts/verify_citations_xenc.py \
    --run_dir runs/realgen_d04/hotpotqa_1k_cs384_d04_bare \
    --queries data/hotpotqa_1k/queries.jsonl \
    --device cpu --batch_size 8

echo "[xenc-verify] done"
