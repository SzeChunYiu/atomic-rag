#!/bin/bash
#SBATCH --job-name=acsrag-b1b-v2
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

# Anti-kT v2: re-run dense + anti_kt with the fixed selector (n_jets=-1, member SNR sort).
# Also re-run fusion_rrf × {greedy, anti_kt, mmr} that batch 1 will skip when it times out.
CONFIGS=(
  "configs/benchmarks/matrix/hotpotqa_1k__dense__anti_kt.yaml"
  "configs/benchmarks/matrix/hotpotqa_1k__fusion_rrf__greedy.yaml"
  "configs/benchmarks/matrix/hotpotqa_1k__fusion_rrf__anti_kt.yaml"
  "configs/benchmarks/matrix/hotpotqa_1k__fusion_rrf__mmr.yaml"
)

# Force stub generator + tag the v2 anti_kt outputs to a separate run dir.
for CFG in "${CONFIGS[@]}"; do
  python - <<PYEOF
import yaml
cfg = yaml.safe_load(open("$CFG"))
cfg["generator"]["provider"] = "stub"
cfg["generator"]["enabled"] = True
# Override n_jets so this batch is unambiguously v2 even if the YAML has a stale 1.
cfg.setdefault("selector", {})["anti_kt_n_jets"] = -1
yaml.safe_dump(cfg, open("$CFG", "w"), sort_keys=False)
PYEOF
done

echo "[acsrag] node     = $(hostname)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"

for CFG in "${CONFIGS[@]}"; do
  echo "==== running ${CFG} ===="
  rag-run-benchmark --config "${CFG}" || echo "FAILED: ${CFG}"
done

python scripts/collect_matrix_results.py
echo "[acsrag] batch 1b v2 done"
