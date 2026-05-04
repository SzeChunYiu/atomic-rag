# Atomic decomposition v3 — physics-mapped (2026-05-03 evening)

## Trigger
User asked for sharper atomic decomposition with physics methods at each
bottleneck. New data point arrived: D04+D06 submodular regressed F1 to
0.4375, below D04-alone (0.482) and chunk baseline (~0.58).

## Bottleneck table with empirical loss

| # | Pipeline step | Atomic loss | Status |
|---|---|---|---|
| 1 | Doc → chunks | 0pp | Saturated |
| 2 | Chunk emb (BGE-M3) | 0pp | Frozen |
| 3 | Query emb | 0pp | Frozen |
| 4 | Top-K dense retrieval | ~10pp | 10.6% bridge queries miss 2nd gold |
| 5 | Rerank | recovers ~3-4pp | Not universal |
| 6 | **Selection** | **~10pp + ACTIVE REGRESSION** | submodular dropped F1 to 0.44 |
| 7 | Prompt assembly | 0pp | |
| 8 | Generation (Qwen 7B greedy) | ~8pp | Model ceiling |
| 9 | CoT extraction | <1pp | |
| 10 | Citation extraction | ~30pp on cit_acc | Cosmetic on F1 |

## Physics method per bottleneck

### B1: Selection objective imbalance — Maximum Entropy (Jaynes 1957)

**Diagnosis.** Submodular gain = coverage_marginal + score * 0.05.
Coverage gain ∈ [0, 2], score bonus ∈ [0, 0.05] → coverage dominates 40×.
Submodular ignores relevance; selects diverse but weakly-supportive atoms.

**Fix.** Gibbs distribution over selections:
P(S) ∝ exp(β·score(S) + Σ_f λ_f·cov_f(S) − μ·tok(S))

Multipliers (β, λ_f, μ) come from constraint dual on dev set, not hand-set.
Greedy at T→0 from this Gibbs distribution gives a principled balance.

**Status.** Implemented in src/astro_cs_rag/selection/maxent.py.
Sweep queued (job 3004947): β ∈ {0.5, 1, 2, 5} × λ_f ∈ {0.1, 0.5, 1.0}.

### B2: Bridge-recall failure — Renormalization Group multi-scale retrieval

**Diagnosis.** 10.6% of bridge queries miss the second gold doc at top-50.
Both gold docs share the bridge entity; at coarse scale they cluster
together. Single-scale top-K misses this clustering.

**Fix.** RG-style multi-scale retrieval:
- Scale 1: doc-centroid retrieval, top-K_1 docs
- Scale 2: chunk retrieval within retained docs, top-K_2 chunks
- Scale 3: atom retrieval within retained chunks, top-K_3 atoms

The RG twist: similarity metric transforms with scale. Coarse → soft topic
kernel; fine → sharp entity match. Like running coupling constants in
RG flow.

**Status.** Spec only. To implement after MaxEnt sweep results.

### B3: Citation cleanup — Kalman-filter track reconstruction (Frühwirth 1987)

**Diagnosis.** Generator emits [E_i] confidently but inconsistently when
selected atoms don't support the answer. cit_acc dropped 0.70 → 0.52
under submodular.

**Fix.** Each cited atom = "hit". True answer chain = "track". Use
Kalman-filter to find most-likely track:
- State: candidate answer entity (from greedy decode)
- Observations: generator's per-token attention weights over selected atoms
- Track-finding: which atoms most likely contributed?
- Update: cross-check entity match in atom text

Output: cleaned citation set, grounded in supporting evidence.

**Status.** Spec only. After MaxEnt + RG.

### B4: Generator ceiling — structural; no physics fix

Qwen 7B at greedy decode has fundamental ceiling on factoid F1. Honest:
swap to Qwen 14B / 32B, or fine-tune. Out of scope for physics methods.

## Sequencing

1. Wait for MaxEnt sweep (3004947) to land — tests B1 fix
2. If MaxEnt > submodular: implement RG retrieval to address B2
3. After RG: implement Kalman citation cleanup to address B3
4. B4 only if needed for paper-tier numbers

## What this means for the paper

- **Anti-kT** stays: real physics method, real adaptation, real ablation
- **Submodular set-cover** stays only as the FAILED-balance baseline that
  motivates MaxEnt — instructive failure showing why principled
  Lagrange-multiplier balance matters
- **MaxEnt** becomes a headline contribution: Jaynes (1957) applied to
  RAG selection, with the multipliers tuned by constraint dual
- **RG retrieval** if it works: another headline, multi-scale with
  scale-dependent metric (the actual RG idea, not just hierarchy)
- **Kalman citation** if it works: HEP track-finding for citation cleanup

This is a paper with 3-4 cross-disciplinary methods, each addressing a
diagnosed bottleneck, each with vanilla → adapted → ablation evidence.
That's Nature MI shape if the numbers land.
