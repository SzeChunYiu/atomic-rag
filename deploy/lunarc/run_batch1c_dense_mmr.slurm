#!/bin/bash
#SBATCH --job-name=acsrag-b1c-mmr
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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

# dense__mmr was killed mid-run by the 2h timeout in Batch 1 — finish it.
# Clean stale partial run dir first to force a fresh run.
rm -rf runs/matrix/hotpotqa_1k__dense__mmr

CFG="configs/benchmarks/matrix/hotpotqa_1k__dense__mmr.yaml"
python - <<PYEOF
import yaml
cfg = yaml.safe_load(open("$CFG"))
cfg["generator"]["provider"] = "stub"
cfg["generator"]["enabled"] = True
yaml.safe_dump(cfg, open("$CFG", "w"), sort_keys=False)
PYEOF

echo "[acsrag] node     = $(hostname)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "==== running ${CFG} ===="
rag-run-benchmark --config "${CFG}"

python scripts/collect_matrix_results.py
echo "[acsrag] batch 1c done"
