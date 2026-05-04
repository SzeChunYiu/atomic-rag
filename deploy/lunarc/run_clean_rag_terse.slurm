#!/bin/bash
#SBATCH --job-name=acsrag-cleanrag-terse
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

echo "[acsrag] node     = $(hostname -s)"

# CLEAN-RAG with TERSE prompt to compare apples-to-apples vs greedy/v4/MMR/rerank.
# This is P5: residual-aware deconvolution selector (Högbom CLEAN analog).
mkdir -p runs/realgen_terse
base=hotpotqa_1k_cs384_clean_rag_qwen7b
override=runs/realgen_terse/${base}
rm -rf "${override}"
mkdir -p "${override}"
python3 -c "
import yaml
from pathlib import Path
data = yaml.safe_load(Path('configs/benchmarks/realgen/${base}.yaml').read_text())
data['paths']['output_dir'] = '${override}'
data['dataset'] = '${base}_terse'
Path('${override}/cfg.yaml').write_text(yaml.safe_dump(data))
print('${override}/cfg.yaml')
"
rag-run-benchmark --config "${override}/cfg.yaml"

echo "[acsrag] clean-rag terse done"
