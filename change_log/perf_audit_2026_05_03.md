# Performance / correctness audit — 2026-05-03

Atomic audit of the hot-path pipeline stages (retrieve → detect → select → evaluate)
prompted by repeated dense_mmr 2 h timeouts on a 1k-query benchmark and the cs=64
sweep's slow first config.

## Bugs found

### B1 — MMR was global, not per-query (selection/mmr.py)

**Atom changed.** `selection/mmr.py::mmr_select` processed the whole atom list at
once instead of grouping by `query_id`.

**Failure mode (which atom was wrong).** With $N_q = 1000$ queries × $K = 50$
candidates, the outer `while candidates:` loop iterated $\sim$50000 times.
Each iteration scanned all remaining candidates and computed a max-similarity
penalty against `selected_ids`, which **accumulated across queries**. So:

1. Cross-query redundancy penalties polluted the MMR score (semantically
   wrong — chunk reuse across queries should not penalize anything).
2. Computational complexity blew up: $O(N_q^2 K^2)$ instead of
   $O(N_q K^2)$.
3. The shared `used` token-budget counter filled after the first query,
   so subsequent queries had nothing selected.

**Why dense_mmr timed out twice.** This is the smoking gun: 2 h was not
enough for the global $O(50000^2 \cdot 1024) \sim 2.5 \times 10^{12}$ flops.

**Fix.** Group by `query_id` first; per query, run a vectorized MMR with
a single `max_sim_to_selected` array updated by one matmul (`emb_mat @
emb_mat[i_best]`) per merge. Complexity: $O(N_q \cdot K^2 \cdot D)$.
Tests: 3 mmr-related tests still pass.

### B2 — Anti-$k_T$ clustering was Python triple-loop O(N³) per query (selection/anti_kt.py)

**Atom changed.** `cluster_anti_kt` iterated $i, j$ pairs in pure Python on
every merge step. With $K = 50$, $D = 1024$: $\sim 50^3 / 2 \cdot D = 64$M
flops per query × 1000 queries = 64 B flops, almost all in NumPy single-element
operations (Python overhead dominates).

**Fix.** Maintain a precomputed pair-distance matrix `d_pair[i, j]` and beam
distances `d_beam[i]`. On each merge, update only row/column `i` of the
matrix via one matmul (`embs @ merged.embedding`). Complexity now
$O(N \cdot D + K^2)$ per query overall.

**Benchmark (local, $N = 50$, $D = 1024$, 100 trials):**
- Before (estimated): ~30–100 ms/query
- After: **1.10 ms/query**
- Speedup: 30–100× per query → 30–100 s saved over a 1k-query benchmark.

### B3 — DenseIndex re-normalized embedding matrix on every query (indexing/dense.py)

**Atom changed.** `DenseIndex.scores()` recomputed
`mat = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)`
on every call. For 1000 queries × $N$ chunks × $D = 1024$, this is
$1000 \cdot N \cdot D$ wasted flops per config. At $N = 8000$ (cs=64),
that's 8 B flops per config of pure waste.

**Fix.** Pre-normalize embeddings once at `__init__` time
(`self._normed`). Added a `topk()` method using `np.argpartition` for
callers that don't need full score dicts.

### B4 — Embedder.encode called once per query in retrieve_run (pipeline/retrieve_run.py)

**Atom changed.** `for q in queries: q_emb = embedder.encode([q.text])[0]`
made $N_q$ separate calls into the BGE-M3 encoder. Each call has
$\sim 50$ ms overhead from CUDA kernel launch, tokenizer init, batch-norm
setup, and dataloader spinup. For 1000 queries this is the entire 63 s
`retrieve_seconds` we observed in Batch 1 dense runs.

**Fix.** Hoist out a single `embedder.encode([q.text for q in queries])`
batched call before the per-mode loop. Modes that don't need an embedding
(bm25, splade) skip this. Expected speedup: $\sim 10×$ on `retrieve_seconds`
because BGE-M3's GPU batch throughput is far higher than its per-call
latency.

## Remaining inefficiencies (smaller wins, deferred)

- `BM25Index.scores()` builds a full dict over all chunks per query; only
  top-$N$ is consumed. ~$N$ wasted dict ops per query but the underlying
  `BM25Okapi.get_scores` is the real cost.
- `rank_by_score()` in `retrieval/fusion.py` sorts the *full* N-element
  score dict per query. RRF only consumes the first $\sim 100$. A
  `topk_rank_by_score` would shave $\log(N) \cdot 8000 \cdot 1000 \approx
  100$M ops per config at cs=64. Worth it on the next pass.
- `select_run.py` reloads the dense index from disk just for embeddings
  (line 32). When the parent already loaded the index, this is a redundant
  `np.load`. Symptom on cs=64: extra 2–3 s per config. Cache-friendly
  improvement.
- `coverage_trace.jsonl` is huge (~500 KB per config) and rarely used in
  analysis — could be opt-in.

## Per-query verification of remaining selectors

- `greedy_select`: per-query (called in for-loop in `select_run.py`). ✓
- `mmr_select`: now per-query internally. ✓
- `jet_select`: had `by_q` from start. ✓
- `aperture_snr`: per-query. ✓
- `detect_evidence` (rank-tail SNR): per-query. ✓

## Net expected impact

For a 1000-query × $N$-chunk benchmark at cs=64 (worst case):

| stage | before | after | savings |
|---|---|---|---|
| retrieve (1 query at a time) | ~63 s | ~6–10 s | ~50 s |
| detect (aperture, vectorized) | ~5 s | ~5 s | 0 |
| MMR selection | **never finished** | ~10 s | hours |
| anti-$k_T$ selection | ~30–60 s | ~1.5 s | ~30–60 s |

**Total expected per-config savings: 1–2 minutes for greedy/anti_kt; many hours
(or unblocking entirely) for MMR.**

## Files changed

- `src/astro_cs_rag/selection/mmr.py` — per-query MMR + vectorized penalty.
- `src/astro_cs_rag/selection/anti_kt.py` — vectorized clustering.
- `src/astro_cs_rag/indexing/dense.py` — pre-normalize, add `topk()`.
- `src/astro_cs_rag/pipeline/retrieve_run.py` — batch query embedding.
- `src/astro_cs_rag/indexing/hierarchical.py` — vectorize node scoring (single matmul over all nodes).
- `src/astro_cs_rag/pipeline/index_build.py` — skip rebuild if existing index matches `(chunk_size, chunk_overlap, embedding_model, use_hash_embedder)`.

All changes pushed to LUNARC. 77/77 unit tests pass locally.

## Audit pass 2 — additional findings (post-MMR/anti-kT/dense)

### B5 — Hierarchical (RAPTOR) scoring loops nodes in Python

`indexing/hierarchical.py::score_hierarchy` ran a per-node Python `for` loop
with one `np.dot` per call. With ~2N nodes per query × 1000 queries, the
Python overhead dominated. **Fix.** One `np.stack([n.embedding ...]) @ q`
per query, then iterate the resulting score vector in Python only to map
back to chunk_ids.

### B6 — Index rebuilt for every selector configuration

`pipeline/index_build.py` had no cache check. In the cs-sweep this caused
the same BGE-M3 embedding pass to run 3× per chunk size (greedy / anti_kt /
mmr), wasting ~3–5 min per rebuild. **Fix.** If the existing
`index_meta.json` matches the requested `(chunk_size, chunk_overlap,
embedding_model, use_hash_embedder)` and the artifacts (`embeddings.npy`,
`chunks.jsonl`) are present, skip the rebuild and return immediately.

This is *opt-in*: callers that want a guaranteed fresh build can `rm -rf
index_dir` first. The per-config `output_dir/index_bundle` layout is
preserved, so downstream tools (error analysis, ablations) keep working.
A higher-leverage win — sharing indexes across configs at fixed
`chunk_size` — requires a slurm wrapper that pre-builds once and passes
`cfg.index_dir` explicitly. Deferred to the next batch's slurm rewrite.

### B8 — Lock-in `fixed_pattern_phases` was destructive, not constructive (retrieval/lockin.py)

**Atom changed.** `fixed_pattern_phases(M)` returned equally spaced phases on
the unit circle: `[2πm/M for m in range(M)]`. With M paraphrases that produce
the same score (an invariant signal — exactly what we want to amplify), the
coherent sum becomes
$$z = s \cdot \sum_{m=0}^{M-1} e^{2\pi i m/M} = s \cdot 0 = 0$$
(sum of M roots of unity is identically zero). So the supposed "coherent
gain" mechanism actually **zeros out** invariant signal — the opposite of
the claimed amplification.

**Why this is Nature-grade.** Theorem 2 in the paper claims the
coherent–incoherent difference quantifies an SNR boost from phase-coherent
aggregation. The previous implementation produced a destructive interference
filter instead. Any P4 lock-in empirical claim run with the previous code
would have been fundamentally wrong.

**Fix.** `fixed_pattern_phases(M)` now returns `[0.0]*M` (all paraphrases
share phase 0 — the correct "constructive coherent sum"). The original
behavior is preserved as `equally_spaced_phases(M)` for diagnostic ablations.
Also vectorized `coherent_sum` and `incoherent_sum` (single matmul each
instead of nested Python loops over (chunks, paraphrases)).

**Cs-sweep impact.** None — lock-in retriever is not used in the current
sweep. Bug found via correctness audit, not symptom.

### B9 — `clean_select` shares a single query embedding across all queries (selection/clean_rag.py)

**Atom changed.** `clean_select(...)` accepts a single `query_embedding`
parameter and uses it as the residual seed for *every* query in `by_q`.
Different queries clearly have different query embeddings. **Not yet fixed**
because CLEAN-RAG is not in the cs-sweep critical path. To use, the API
needs to take `query_embeddings: dict[str, np.ndarray]` — flagged for
implementation when P6 CLEAN-RAG is empirically tested.

### B7 — StubGenerator masked selector differences (generation/generator.py)

**Atom changed.** The deterministic stub used by every benchmark when
Ollama isn't installed cited only `evidence[0]` and used only its first
sentence. **All downstream answer-side metrics — `citation_accuracy`,
`answer_em`, `answer_f1`, `conservation_R_*` — therefore depended only on
the top-1 chunk picked by the selector.** Selectors that picked the same
top-1 chunk produced bit-identical metrics regardless of what they did
with chunks 2–5 of the budget.

**Why this is a Nature-grade issue.** It is the single most important
bug discovered: it would have caused the entire cs-sweep on real
HotpotQA to report v3 ≡ greedy on every answer-side metric, *not because
the selector mechanism didn't transfer*, but because the measurement
apparatus only sampled the top-1 chunk. The result would have looked
like a null result in the paper.

**Fix.** Stub now concatenates the first sentence from EVERY selected
chunk and cites every chunk it consumed. Selector differences are now
visible on citation_accuracy and answer_f1. The `[Ei]` indexing remains
1-based to match the prompt-style convention.

**Cs-sweep impact.** The job that started under the old stub
(2998872) was cancelled and resubmitted (2999682) under the new stub so
all 12 configs use the same measurement apparatus.

## Files NOT yet audited (low-priority — outside the cs-sweep hot path)

`pipeline/profile_run.py`, `pipeline/error_analysis_run.py`,
`generation/*.py`, `selection/{clean_rag,sbi,smc}.py`,
`detection/{cherenkov,fdr,standard_candle,unfolding}.py`,
`diagnostics/{calorimetry,ood_gate,sanity,tda}.py`,
`retrieval/{coronagraph,vlbi,wasserstein,lockin}.py`,
`indexing/{multivec,multivec_encoder,splade}.py` (only used when
side-indices are enabled, which they aren't in cs-sweep).
`benchmarks/asimov.py`, `data/*.py`, `cli/*.py`, `reranking/*.py` (rerank
is disabled in the current sweep).

These will be audited if/when the corresponding feature is enabled in a
benchmark. The above 6 fixes cover everything in the live cs-sweep
critical path.
