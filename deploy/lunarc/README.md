# LUNARC deployment

Cluster: cosmos.lunarc.lu.se
Account: lu2025-2-51
Partition: gpua100 (1× A100 80 GB)
Project root: `/projects/hep/fs10/shared/nnbar/billy/RAG`
Conda env:    `/projects/hep/fs10/shared/nnbar/billy/packages/acsrag`

## Initial setup (once per cluster)

```bash
ssh lunarc
mkdir -p /projects/hep/fs10/shared/nnbar/billy/RAG
exit

# Push code to cluster.
deploy/lunarc/sync.sh push

# Build conda env + install ollama + pull base models.
ssh lunarc "bash /projects/hep/fs10/shared/nnbar/billy/RAG/deploy/lunarc/setup_env.sh"
```

## Submitting a benchmark

```bash
deploy/lunarc/sync.sh push   # or push-data if data/ changed
ssh lunarc "cd /projects/hep/fs10/shared/nnbar/billy/RAG && \
  mkdir -p logs && \
  CONFIG=configs/benchmarks/hotpotqa_1k_dense_rerank.yaml \
  RUN_TAG=hotpot_dense_rerank \
  sbatch deploy/lunarc/run_benchmark.slurm"

# Watch progress.
ssh lunarc "squeue -u scyiu -o '%.10i %.18j %.8T %.10M %.12l %.20R'"
ssh lunarc "tail -f /projects/hep/fs10/shared/nnbar/billy/RAG/logs/acsrag-bench_*.out"

# Pull artifacts back when done.
deploy/lunarc/sync.sh pull-runs
```

## Notes

- The 70B-Llama sanity model is **not** pulled by `setup_env.sh` to save disk;
  uncomment the line and re-run when needed.
- gpua100 max wall-time is 7 days; default in the SLURM template is 12 h to
  encourage staged ablations rather than monolithic sweeps.
- We do not use git pull on the cluster — `deploy/lunarc/sync.sh push` is the
  single source of truth, so there's no two-way merge to worry about.
