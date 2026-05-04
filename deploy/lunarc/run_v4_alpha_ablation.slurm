#!/bin/bash
#SBATCH --job-name=acsrag-v4-alpha
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

# v4 anti-kT α-ablation: characterize the partner score-gate sensitivity.
# alpha = required ratio of partner SNR to primary SNR for cluster pull-in.
# alpha=0   → no gating (v3 behavior)
# alpha=0.3 → loose gate, more partner pull-ins
# alpha=0.7 → current setting (best on validation)
# alpha=0.9 → strict gate, partner must be near-equal to primary
# alpha=1.0 → only equal-strength partners (degenerate)
mkdir -p runs/realgen_terse
for alpha in 0.0 0.3 0.5 0.9; do
  base=hotpotqa_1k_cs384_v4a${alpha}_qwen7b
  override=runs/realgen_terse/${base}
  rm -rf "${override}"
  mkdir -p "${override}"
  cfg="${override}/cfg.yaml"
  python3 -c "
import yaml
from pathlib import Path
src = Path('configs/benchmarks/realgen/hotpotqa_1k_cs384_v4a0_7_qwen7b.yaml')
data = yaml.safe_load(src.read_text())
data['paths']['output_dir'] = '${override}'
data['dataset'] = '${base}_terse'
data['selector']['partner_score_gate_alpha'] = ${alpha}
Path('${cfg}').write_text(yaml.safe_dump(data))
print('${cfg}')
"
  echo "==== running alpha=${alpha} ===="
  rag-run-benchmark --config "${cfg}" || echo "FAILED alpha=${alpha}"
done

echo "[acsrag] v4 alpha ablation done"
