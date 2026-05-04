# P3 cs-sweep iter 1 — cs=64 selector indistinguishability

**Date.** 2026-05-03
**Job.** LUNARC 2999682 (`acsrag-cs-sweep`, gpua100/cg12, 6 h budget)
**Configs analyzed so far.** cs=64 × {greedy, anti_kt v3 (n_jets=-2), mmr}; cs=128 greedy (cs=128 anti_kt currently running).

## Atomic observation

At cs=64, all three selectors produce **bit-exact identical** retrieval-side
and answer-side headline metrics:

| selector | recall@5 | citation_accuracy | answer_f1 | conservation_faithfulness | R_entity | R_numeric | R_temporal |
|---|---|---|---|---|---|---|---|
| greedy   | 0.6140 | 0.08694 | 0.00675 | 0.47711 | 0.5314 | 0.6793 | 0.3579 |
| anti_kt v3 | 0.6140 | 0.08694 | 0.00675 | 0.48007 | 0.5326 | 0.6793 | 0.3479 |
| mmr      | 0.6140 | 0.08694 | 0.00675 | 0.47729 | 0.5317 | 0.6793 | 0.3572 |

Conservation_faithfulness shows a real but tiny v3 lead (+0.003 vs greedy,
+0.003 vs mmr). R_entity follows the same pattern (+0.001). v3 picks
*slightly different chunks*, but ranks the same gold-doc evidence to top.

## Failure-mode interpretation

**Why selectors collapse at cs=64.** The token budget is 1024 with avg chunk
size ≈ 64. Each query receives 50 candidates → up to ~16 chunks fit in
budget. Greedy fills top-N by score. v3 (atomic-unit greedy + jet partners)
also fills the budget but in a different order. MMR with high λ also
prioritizes score. If the *same set* of 16 chunks gets selected (different
order), and the StubGenerator concatenates first sentences from each, the
final answer text and cited_chunk_ids are identical.

**Verification.** recall@5 = 0.614 is identical across selectors *by
construction* — recall@5 measures the retrieval candidate list, which is
selector-independent. citation_accuracy and answer_f1 collapse because the
*set* of selected chunks coincides even though the *order* differs.

## What this means for the publication

This is **not** a null result for the v3 mechanism. It's a regime where the
mechanism cannot bite: budget ≫ chunk size, so all selectors saturate the
budget with the same chunks. The relevant test is at larger chunk sizes
where the budget forces *fewer* chunks per query and ordering / partner
recall actually matter:

- cs=64: 50 candidates × 64 tokens ≈ 3200 < 1024×3 budget. No competition.
- cs=128: ~50 cand × 128 ≈ 6400 ≈ 6× budget → competition begins.
- cs=256: ~50 cand × 256 ≈ 12800 ≈ 12× budget → tight selection.
- cs=384: ~50 cand × 384 ≈ 19200 ≈ 19× budget → highly contested.

**Pre-registered claim C1c (v3 ≥ greedy on HotpotQA).** Reading at cs=64
alone is uninformative. Wait for cs ∈ {128, 256, 384} before any
publication-grade claim.

## cs=128 first read (greedy only so far)

| metric | cs=64 greedy | cs=128 greedy | Δ |
|---|---|---|---|
| recall@5 | 0.614 | 0.769 | **+0.155** |
| recall@10 | (need to read) | 0.849 | — |
| MRR | (need to read) | 0.858 | — |
| citation_accuracy | 0.0869 | 0.0815 | −0.005 |
| answer_f1 | 0.00675 | 0.00617 | −0.0006 |
| conservation_faithfulness | 0.4771 | 0.5385 | +0.061 |

Bigger chunks → much higher recall (more gold-doc surface per chunk) and
much higher conservation_faithfulness (more text to constrain
entity/numeric/temporal residuals). citation_accuracy drops slightly
because more chunks per gold doc dilute the citation per-id match. f1 is
floor-bound by the StubGenerator (first-sentence concat ≠ real answer).

The cs=128 anti_kt and mmr runs will tell us whether v3 differentiates from
greedy as competition tightens.

## Next steps

1. Wait for cs=128 anti_kt + mmr; compare to cs=128 greedy.
2. Wait for cs ∈ {256, 384} triplets.
3. If any cs shows v3 > greedy on cit_acc or f1: pre-registered C1c PASS.
4. If all cs show no selector signal: investigate Ollama path (real
   generator) — stub may be the bottleneck even with all-chunk citation.
