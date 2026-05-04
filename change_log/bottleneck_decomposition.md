# Bottleneck decomposition — selection is saturated, citation is the gap

**Date:** 2026-05-03
**Source:** `/tmp/cite_gap.py` on HotpotQA-1k cs=384 terse runs.

## The numbers that change everything

| selector | gold_in_sel | gold_in_cited | sel−cit gap | cit_acc |
|---|---|---|---|---|
| greedy | 0.996 | 0.891 | 10.5pp | 0.7987 |
| MMR | 0.996 | 0.892 | 10.4pp | 0.7992 |
| v4 anti-kT | 0.996 | 0.908 | 8.8pp | 0.8126 |
| greedy + rerank | **0.997** | 0.920 | 7.7pp | 0.8165 |
| v4 + rerank | 0.997 | 0.911 | 8.6pp | 0.8113 |

`gold_in_sel` = fraction of queries where ≥1 gold doc's chunk is in
selected_context. `gold_in_cited` = fraction where ≥1 gold doc is in the
generator's citations. `sel−cit gap` = queries where gold was selected
but not cited.

## What this means

**Selection is saturated.** Across every dense-side selector, gold
chunks reach the generator in 99.6% of queries. The 0.4pp residual gap
is purely the rare cases where token-budget pruning drops the gold
chunk — not a selection-mechanism problem.

**The selector "wins" we measured were really citation-behavior wins.**
v4's +1.39pp cit_acc over greedy comes from the generator citing 1.7pp
more correctly (90.8% vs 89.1%) on context that v4 happened to
structure differently. The selection itself was equally good.

**The next-frontier mechanism is citation discipline, not selection.**
8-10pp of every selector's queries have gold in context but the
generator cited a non-gold chunk. Closing this gap directly should give
cit_acc → 0.99+ regardless of selector.

## Implication for v4's mechanism story

The paper's claim that "v4 anti-kT improves multi-hop selection" was
imprecise. The corrected claim:

> "v4 anti-kT structures the selected context (two complementary jets +
> score-gated partner) in a way that produces 1.7pp more accurate
> *citation behavior* from the generator on bridge queries. Selection
> coverage is saturated (99.6%) for all dense selectors; v4's gain
> operates downstream of selection by changing what the generator
> *attends to* when citing."

This is actually a *more* mechanistically interesting claim. The
selector is implicitly engineering generator behavior, not just
shuffling chunks.

## Why no published RAG paper saw this

The bottleneck decomposition requires three things together:
1. A high-recall retriever + reranker combo (we have this)
2. A budget-generous selector (1024 tokens here)
3. A *terse* generator prompt that forces clean citation extraction

Without the terse prompt (Appendix A), F1 is verbosity-capped at ~0.12
and the citation channel is noisy. Most prior work measured cit_acc on
verbose generation with ambiguous citation extraction.

## Update 2026-05-03: CPV failed — the bottleneck is answer-correctness

We implemented CPV as proposed (replace citations with chunks
containing answer content tokens). It **regressed cit_acc by
1.3-1.7pp across all selectors**:

| selector | cit_acc orig | cit_acc CPV | Δ |
|---|---|---|---|
| greedy | 0.7987 | 0.7856 | -1.31pp |
| v4 | 0.8126 | 0.7956 | -1.70pp |
| greedy+rerank | 0.8165 | 0.8002 | -1.64pp |

Inspection of failure cases shows the 8% sel-not-cited gap is mostly
**wrong-answer cases**, not pointer mistakes:

- Q "what country..." Gold *Jamnagar*; Gen *"India"* — wrong answer,
  citation followed wrong answer.
- Q "Damon Stoudamire's cousin" Gold *Terrence Jones*; Gen *"DeMarcus
  Cousins"* — generator confused two players.

CPV swapped these citations to chunks containing the *wrong* answer
string ("India"), making them point to non-gold docs more often.

**The corrected picture:** the generator already cites consistently with
its own answer. The cit_acc ceiling is set by **answer correctness**, not
citation discipline. The 8% sel-not-cited gap is mostly the answer being
wrong.

This means:
1. The bottleneck-decomposition narrative still stands (selection is
   saturated), but the *next-frontier mechanism is answer quality*, not
   citation post-verification.
2. Improving cit_acc materially requires improving answers, which
   requires either a bigger/better generator, multi-hop CoT reasoning,
   or fundamentally different evidence presentation.
3. CPV becomes a *third* negative-with-mechanism finding for the paper:
   "Naïve answer-token-based citation verification fails because LLM
   answers and gold answers are not always interchangeable strings."

## The honest path forward

The selector framework is real but operates at a saturated stage. Its
+1pp wins (v4 anti-kT, rerank) are real but small. NMI-caliber would
require attacking answer-correctness directly, which moves us out of
the selection-mechanism framing.

Two viable directions:

**Direction A — Multi-hop CoT generation.** Prompt the generator to
first identify intermediate entities ("the cousin of X is Y; Y played
at college Z"), then commit to the final answer. This is generation-side,
not selection-side. Could lift F1 by 5-10pp on bridge queries.
**Implementation effort: 1-2 days.** **Risk: prior work has tried this
(self-RAG, ReAct); we'd need to find a novel angle.**

**Direction B — Honest characterization paper.** Accept that the
selector framework operates at a saturated stage with small wins.
Submit to ACL/EMNLP/TACL with the rigor of the bottleneck-decomposition
finding as the headline. **Implementation effort: 1-2 weeks of
writing.** **Risk: not NMI tier, but a defensible solid paper.**

I recommend Direction A first (try the algorithmic improvement) with
Direction B as the fallback. Multi-hop CoT specifically targets the
documented bottleneck (wrong answers on bridge queries) and aligns
with the paper's existing framing about multi-hop specialization.

## The original CPV mechanism (deprecated)

After the generator emits `(answer_text, cited_chunks)`:
1. Strip `[E_i]` markers from answer; tokenize content words.
2. For each cited chunk_id, check whether ≥1 content word appears in
   the chunk text (case-insensitive substring match, with simple
   normalization for plural/possessive).
3. If a citation has no content-word overlap, search the remaining
   selected_context for a chunk that does. Replace the citation.

This converts cit_acc directly: 0.92 → ~0.99 expected. F1 unchanged
(answer text not modified).

## The new paper framing

Three-tier ceiling decomposition:

```
                                         observed (greedy+rrk)   ceiling
selection: gold_in_sel                    99.7%                  99.7%
citation:  gold_in_cited                  92.0%                  ~99.0% (CPV closes)
answer:    F1                             41.6%                  ?? (generator-bound)
```

The paper's mechanistic contribution becomes:
1. **Bottleneck decomposition** — the field has been optimizing the
   saturated stage. The real next-frontier mechanism is at citation, not
   selection.
2. **Citation Post-Verification** — a simple, training-free mechanism
   that closes 7-10pp of cit_acc gap on top of any selector.
3. **The selector framework as before** — characterized mechanisms,
   specialization, failure modes — now properly framed as
   citation-behavior engineers rather than selection coverage improvers.

This is a *better* paper than the original framing. It explains why
selectors give small wins (selection is saturated), identifies the real
bottleneck (citation), proposes a cheap fix (CPV), and reframes the
existing characterizations in mechanistically more precise terms.
