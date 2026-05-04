# Publication strategy — what would a Nature MI paper look like, what we have, what we need

**Date.** 2026-05-03
**Purpose.** Honest assessment so research direction matches the journal target.

## What a Nature Machine Intelligence RAG paper looks like

Recent NMI / top-tier ML venues for retrieval-augmented generation papers
typically have:

1. **A novel, theoretically motivated mechanism** with a clear physical /
   mathematical principle (not just engineering tricks).
2. **Multi-benchmark gains** — at minimum HotpotQA + 1–2 of {MuSiQue,
   2WikiMultiHopQA, NaturalQuestions, TriviaQA, FEVER}. Single-benchmark
   wins read as benchmark-fit.
3. **Substantial effect size** — typical thresholds:
   - 5–10pp on EM/F1 over the strongest baseline
   - or a smaller win combined with significant efficiency improvement
   (e.g., 5pp + 50% latency).
4. **Strong baselines** — at minimum: BM25, dense (BGE / E5), RRF, MMR,
   ColBERT or another late-interaction baseline, RankGPT/BM25+rerank.
5. **Real generator** — GPT-4, Llama-3-70B, Qwen-72B, or similar. Stub
   answers do not count.
6. **Ablations** isolating the mechanism's contribution.
7. **Mechanistic interpretability** — atomic explanation of *why* the
   mechanism works, with quantitative diagnostics.
8. **Failure-mode analysis** — honest exposition of when it fails.

## What we currently have

| dimension | status |
|---|---|
| novel mechanism | anti-kT atomic-unit (P3) — **not winning baselines** |
| novel mechanism | lock-in coherent paraphrase (P4) — **untested** |
| novel mechanism | CLEAN-RAG / SBI / SMC (P5/P6) — partial, untested |
| benchmark coverage | HotpotQA-1k only (1 dataset) |
| effect size | v4 = greedy on HotpotQA cit_acc (no win) |
| baselines | greedy, MMR, fusion_rrf — others not implemented |
| real generator | Qwen 2.5-7B wired up; first sweep RUNNING |
| ablations | v3 vs v4 done; partner-gate analysis solid |
| mechanistic understanding | excellent on anti-kT failure mode |
| failure-mode analysis | rigorous (1.5% partner gold rate vs 5.3% baseline) |

## Brutally honest gap analysis

**To get to NMI from current state we need:**

1. At least ONE mechanism that wins meaningfully on HotpotQA with real
   generator. Right now we have ZERO. Lock-in is the most promising
   remaining shot.

2. That mechanism must also win on at least one other benchmark
   (MuSiQue or 2Wiki). Otherwise it's benchmark-fit.

3. Strong baselines — we currently have 3 (greedy, MMR, RRF). NMI
   reviewers would expect ColBERT, RankGPT, possibly a cross-encoder
   reranker, possibly a learned-sparse method like SPLADE. Several of
   these are partially implemented.

4. Effect size of ≥5pp F1 or ≥10pp recall@k. Smaller is fine if combined
   with an efficiency gain or a unique theoretical contribution.

5. Time budget: roughly 4–6 weeks of focused work assuming each
   mechanism takes 3–7 days to implement, validate, and ablate.

## Realistic placement spectrum

| outcome | venue likely |
|---|---|
| Lock-in wins strongly + reproducible across benchmarks | NMI / Nature Communications |
| Lock-in wins on 1 benchmark + mechanistic story | TMLR / NeurIPS / ICLR |
| Lock-in modestly wins + anti-kT failure analysis | EMNLP / ACL |
| Nothing wins, but careful negative results | EMNLP Findings / workshop |

**The honest expected outcome at current trajectory is around the third
row.** We have rigorous failure-mode analysis but no positive finding yet.
The lock-in experiment in flight could change this.

## Decision tree based on lock-in result (running now)

**If lock-in beats fusion_rrf on HotpotQA recall@5 by ≥2pp:**
→ We have a real positive direction. Next: replicate on MuSiQue / 2Wiki,
add ColBERT baseline, scale paraphrases to M ∈ {2, 4, 8, 16}, do
mechanistic analysis. **Aim NMI**.

**If lock-in ties or underperforms on HotpotQA:**
→ Two interpretations:
  (a) Theorem 2 √M boost requires invariant-evidence regime that HotpotQA
      lacks (all paraphrases retrieve the bridge entity → no diversity to
      coherently sum). Need a different benchmark or a different
      mechanism.
  (b) Mechanism is fundamentally weak.
→ Decision: try MuSiQue (more diverse evidence) before declaring P4
dead. If MuSiQue also ties, pivot to P5 (SBI / CLEAN-RAG residual-aware)
or P6 (calibration).

## Highest-EV next experiments (ranked)

1. **Lock-in HotpotQA result** (running) — definitive on whether
   √M-coherent-aggregation is real on bridge queries.
2. **Real-generator HotpotQA** (running) — definitive on whether
   selector signal exists at non-floor F1.
3. **Anti-kT R sweep** at cs=256 (configs prepared, queued for next
   slot) — closes the door on R=1.0 being pessimal. Cheap.
4. **MuSiQue or 2Wiki download + 1 baseline run** — gets us to 2
   datasets immediately. Needed regardless of which mechanism wins.
5. **CLEAN-RAG residual-aware selection (P5)** — orthogonal mechanism,
   theoretically motivated by signal subtraction.

## Working assumption for the next 3 hours

Both jobs RUNNING. By 12:30 UTC we will have:
- Real-gen results on greedy / v4 / MMR (3000315)
- Paraphrase cache + lock-in vs fusion_rrf (3000316)

If either gives a meaningful positive result, we pivot to multi-benchmark
replication immediately. If both flat, we shelve P3/P4 and move to P5
(CLEAN-RAG residual-aware selection) which has a different theoretical
basis (residual = signal − selected, pick chunks that maximize residual
information).

## Two publication archetypes — pick by mid-day

After today's results, the likely path is one of:

### Path A — SOTA paper

- One mechanism wins ≥5pp F1 over the strongest baseline on multiple
  benchmarks (HotpotQA + NQ-open + maybe MuSiQue).
- We optimize hard, do extensive ablations, defend SOTA claims.
- Target: NMI / Nature Comm / NeurIPS Datasets&Benchmarks.
- Probability today: low (selector-stage already dead, lock-in unlikely
  to give >5pp without leakage).

### Path B — Framework paper

- Unified physics-inspired RAG framework with multiple mechanisms:
  anti-kT / lock-in / CLEAN-RAG / aperture / Cherenkov.
- Each mechanism has theoretical motivation (a "Theorem N" for SNR boost
  or selection optimality).
- Experiments show:
  - Each mechanism's regime of validity (where it wins, where it fails).
  - Atomic mechanistic interpretability (what we did with anti-kT).
  - Modest empirical wins (~1–3pp) but consistent across mechanisms.
- Failure-mode analyses are part of the contribution, not bugs.
- Target: NMI / TMLR (NMI does publish framework papers).
- Probability today: moderate. We already have a solid negative-result
  story for anti-kT + dilution mechanism; lock-in + CLEAN-RAG add 2 more.

**Bias toward Path B** unless lock-in shows a clear ≥5pp win. Path B is
honest, achievable, and aligned with the physics-inspired framing.

## Mechanisms we have / want for Path B

| mechanism | status | theory needed |
|---|---|---|
| anti-kT atomic-unit | done; v4 = greedy | Theorem on partner-pull failure mode |
| lock-in coherent paraphrase | running | Theorem 2 (√M SNR boost) — exists; need empirical confirmation |
| CLEAN-RAG residual-aware | wired, ready | Theorem on convergence; need to articulate |
| Aperture photometry (already in detector) | done | small contribution |
| Cherenkov threshold (already in code) | partial | need experiment |
| VLBI multi-baseline retrieval | not implemented | non-trivial |
| Coronagraph dominant-signal suppression | partial | needs theory |

For Path B, we need 3–4 mechanisms with theory + experiments, plus
unifying framework. Realistic 4–6 weeks.
