#!/usr/bin/env bash
# Automated D04 evidence-crowding study.
#
# Sweeps the phase-1 grid with one (--hash-only) or two embedders
# (hash + sbert) and emits a single REPORT.md auto-summarising the
# findings (C*, AUC, smoking-gun nd=0 selection failures, cross-embedder
# diff if both ran).
#
# Usage:
#   scripts/run_d04_study.sh                # hash only (CPU, ~2s)
#   scripts/run_d04_study.sh --with-sbert   # adds sentence-transformers
#   scripts/run_d04_study.sh --grid smoke   # quick smoke variant

set -euo pipefail

GRID="phase1"
N_QUERIES=50
WITH_SBERT=0
SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-sbert) WITH_SBERT=1; shift ;;
        --grid) GRID="$2"; shift 2 ;;
        --n_queries) N_QUERIES="$2"; shift 2 ;;
        --sbert_model) SBERT_MODEL="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
STUDY_DIR="runs/d04_crowding/study_${TS}"
mkdir -p "${STUDY_DIR}"

echo "[d04-study] grid=${GRID} n_queries=${N_QUERIES} dir=${STUDY_DIR}"

HASH_DIR="${STUDY_DIR}/hash"
python3 -m astro_cs_rag.cli.run_crowding_sweep \
    --grid "${GRID}" --n_queries "${N_QUERIES}" \
    --systems atom_dense chunk_dense \
    --embedder hash \
    --out_dir "${HASH_DIR}"

ANALYZE_ARGS=("--sweep_dir" "${HASH_DIR}")

if [[ "${WITH_SBERT}" -eq 1 ]]; then
    SBERT_DIR="${STUDY_DIR}/sbert"
    python3 -m astro_cs_rag.cli.run_crowding_sweep \
        --grid "${GRID}" --n_queries "${N_QUERIES}" \
        --systems atom_dense chunk_dense \
        --embedder sbert --sbert_model "${SBERT_MODEL}" \
        --out_dir "${SBERT_DIR}"
    ANALYZE_ARGS+=("--sweep_dir" "${SBERT_DIR}")
fi

REPORT="${STUDY_DIR}/REPORT.md"
python3 scripts/analyze_crowding_sweep.py "${ANALYZE_ARGS[@]}" --out "${REPORT}"

echo "[d04-study] done"
echo "[d04-study] report: ${REPORT}"
