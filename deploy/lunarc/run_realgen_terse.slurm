#!/bin/bash
#SBATCH --job-name=acsrag-realgen-terse
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

echo "[acsrag] node     = $(hostname -s)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Re-run HotpotQA × {greedy, v4, MMR} with TERSE generation prompt (new
# CITATION_SYSTEM forces short answers — should unlock selector signal that
# was masked by Qwen's verbosity).
mkdir -p runs/realgen_terse
CONFIGS=(
  "configs/benchmarks/realgen/hotpotqa_1k_cs384_greedy_qwen7b.yaml"
  "configs/benchmarks/realgen/hotpotqa_1k_cs384_v4a0_7_qwen7b.yaml"
  "configs/benchmarks/realgen/hotpotqa_1k_cs384_mmr_qwen7b.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  # Override output_dir to runs/realgen_terse so old runs are preserved.
  base=$(basename "${cfg}" .yaml)
  override="runs/realgen_terse/${base}"
  rm -rf "${override}"
  mkdir -p "${override}"
  # Stash a config with overridden output_dir.
  python3 -c "
import yaml, sys
from pathlib import Path
cfg_path = Path('${cfg}')
data = yaml.safe_load(cfg_path.read_text())
data['paths']['output_dir'] = '${override}'
data['dataset'] = '${base}_terse'
out_cfg = Path('${override}/cfg.yaml')
out_cfg.write_text(yaml.safe_dump(data))
print(out_cfg)
"
  echo "==== running ${override}/cfg.yaml ===="
  rag-run-benchmark --config "${override}/cfg.yaml" || echo "FAILED: ${cfg}"
done

echo "[acsrag] realgen-terse sweep done"
