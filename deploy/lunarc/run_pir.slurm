#!/bin/bash
#SBATCH --job-name=acsrag-pir
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

echo "[acsrag] node = $(hostname -s)"

# PIR benchmark on HotpotQA-1k cs=384 terse:
#   1. retrieve top-50 (dense) — run via standard pipeline with reranker disabled
#   2. apply PIR rerank to candidates.jsonl (edge_thresh=0.5, max_path=3)
#   3. continue with greedy selector + Qwen 7B terse generation
mkdir -p runs/realgen_pir configs/benchmarks/pir
base=hotpotqa_1k_cs384_pir_qwen7b
override=runs/realgen_pir/${base}
rm -rf "${override}"
mkdir -p "${override}"
cfg="${override}/cfg.yaml"
# clone the greedy config (no reranker)
python3 -c "
import yaml
from pathlib import Path
src = Path('configs/benchmarks/realgen/hotpotqa_1k_cs384_greedy_qwen7b.yaml')
data = yaml.safe_load(src.read_text())
data['paths']['output_dir'] = '${override}'
data['dataset'] = '${base}_terse'
data['reranker'] = {'enabled': False}
Path('${cfg}').write_text(yaml.safe_dump(data))
print('${cfg}')
"

# Run partial pipeline: retrieve only, then PIR, then continue.
# rag-run-benchmark runs end-to-end, so we hook in a custom flow:
#   stage A: retrieve (writes candidates.jsonl)
#   stage B: PIR rewrite candidates.jsonl
#   stage C: detect + select + generate
# We use rag-retrieve / rag-detect / rag-select / rag-generate sub-commands
# if they exist; otherwise run the full pipeline twice.

# Easiest: run the full pipeline (which builds the index, retrieves,
# detects, selects, generates) — then re-do retrieve + PIR + downstream.
# To save time, use the same run_dir but invoke the PIR step then
# rerun selection + generation.
#
# Concretely: run end-to-end with a sentinel "pir_pending" marker, then
# overwrite candidates.jsonl with PIR output and rerun stages 4..6.
#
# Cleanest: use a small Python orchestrator that does:
#   from astro_cs_rag.pipeline.benchmark import retrieve, detect, select, generate, evaluate
# and inserts PIR between retrieve and detect.

python3 << PYEOF
import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, "/projects/hep/fs10/shared/nnbar/billy/RAG/src")

import yaml
from astro_cs_rag.config.schema import (
    BenchmarkConfig, DetectConfig, EvaluateConfig, GenerateConfig,
    IndexConfig, PathsConfig, RetrieveConfig, SelectConfig,
)
from astro_cs_rag.pipeline.detect_run import detect_run
from astro_cs_rag.pipeline.evaluate_run import evaluate_run
from astro_cs_rag.pipeline.generate_run import generate_run
from astro_cs_rag.pipeline.index_build import build_index_bundle
from astro_cs_rag.pipeline.retrieve_run import retrieve_run
from astro_cs_rag.pipeline.select_run import select_run

cfg_path = Path("${cfg}")
data = yaml.safe_load(cfg_path.read_text())
cfg = BenchmarkConfig.model_validate(data)
index_dir = cfg.paths.output_dir / "index_bundle"

idx = IndexConfig(
    dataset=cfg.dataset, seed=cfg.seed,
    paths=PathsConfig(corpus_path=cfg.paths.corpus_path, output_dir=index_dir),
    chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
    retriever=cfg.retriever, embedding=cfg.embedding, side_indices=cfg.side_indices,
)
build_index_bundle(idx)

retr = RetrieveConfig(
    dataset=cfg.dataset, seed=cfg.seed,
    paths=PathsConfig(corpus_path=cfg.paths.corpus_path, queries_path=cfg.paths.queries_path, output_dir=cfg.paths.output_dir),
    index_dir=index_dir, retriever=cfg.retriever, embedding=cfg.embedding,
)
run_dir = retrieve_run(retr)
print("retrieved candidates at", run_dir)

# PIR stage
subprocess.run([
    sys.executable, "scripts/pir_rerank.py", str(run_dir),
    "--top_n_in", "50", "--top_n_out", "20",
    "--edge_threshold", "0.5", "--max_path_length", "3",
], cwd="/projects/hep/fs10/shared/nnbar/billy/RAG", check=True)
print("PIR rerank applied")

det = DetectConfig(dataset=cfg.dataset, seed=cfg.seed, run_dir=run_dir, index_dir=index_dir, detector=cfg.detector)
detect_run(det)
sel = SelectConfig(dataset=cfg.dataset, seed=cfg.seed, run_dir=run_dir, index_dir=index_dir, queries_path=cfg.paths.queries_path, selector=cfg.selector)
select_run(sel)
gen_run = generate_run(GenerateConfig(
    dataset=cfg.dataset, seed=cfg.seed, run_dir=run_dir, index_dir=index_dir,
    queries_path=cfg.paths.queries_path, generator=cfg.generator,
))
ev = EvaluateConfig(
    dataset=cfg.dataset, seed=cfg.seed,
    paths=PathsConfig(corpus_path=cfg.paths.corpus_path, queries_path=cfg.paths.queries_path, gold_path=cfg.paths.gold_path, output_dir=cfg.paths.output_dir),
    run_dir=run_dir, index_dir=index_dir, metrics=cfg.metrics,
)
evaluate_run(ev)
print("DONE  run_dir =", run_dir)
PYEOF

echo "[acsrag] PIR run done"
