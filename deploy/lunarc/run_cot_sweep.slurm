#!/bin/bash
#SBATCH --job-name=acsrag-cot-sweep
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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

# Multi-hop CoT prompting on HotpotQA-1k cs=384.
# Compares: greedy/dense+CoT vs greedy/dense+terse and greedy+rerank+CoT vs greedy+rerank+terse.
# Generation max_tokens raised to 200 to allow reasoning steps.
mkdir -p runs/realgen_cot configs/benchmarks/cot

for variant in greedy greedy_rerank; do
  base=hotpotqa_1k_cs384_${variant}_qwen7b
  cfg="configs/benchmarks/cot/${base}_cot.yaml"
  override="runs/realgen_cot/${base}"
  rm -rf "${override}"
  mkdir -p "${override}"
  python3 -c "
import yaml
from pathlib import Path
src = Path('configs/benchmarks/realgen/${base}.yaml')
data = yaml.safe_load(src.read_text())
data['paths']['output_dir'] = '${override}'
data['dataset'] = '${base}_cot'
data['generator']['prompt_style'] = 'cot'
data['generator']['max_tokens'] = 200
Path('${cfg}').write_text(yaml.safe_dump(data))
print('${cfg}')
"
  echo "==== running ${cfg} ===="
  rag-run-benchmark --config "${cfg}" || echo "FAILED ${variant}"
done

echo "[acsrag] cot sweep done"

# Post-process: extract final answer + recompute F1
for variant in greedy greedy_rerank; do
  rd=$(ls -d runs/realgen_cot/hotpotqa_1k_cs384_${variant}_qwen7b/*/ 2>/dev/null | grep -v index_bundle | head -1)
  echo "=== post ${variant} ==="
  python3 scripts/extract_cot_final.py "${rd}" --queries data/hotpotqa_1k/queries.jsonl
done
