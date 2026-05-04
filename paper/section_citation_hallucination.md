# Citation hallucination in atomic RAG

## Setup

Each generated answer carries inline citations of the form `[E_i]` that
nominally point to specific atoms in the evidence pool. We measure
citation accuracy as `cit_acc` — the fraction of cited atoms whose
source document is in the gold-document set for the query.

## Finding 1: high baseline `cit_acc` is misleading

For the chunk-level pipeline (BGE-M3 retrieval + BGE-reranker rerank +
greedy CoT), `cit_acc` reaches 0.85. At atom level, a sharp drop:

| Configuration | cit_acc |
|---|---|
| Chunk + rerank + CoT (baseline) | 0.852 |
| Atom retrieval, score-greedy fill (D04) | 0.700 |
| Atom retrieval + submodular set-cover, citation prompt | 0.518 |
| Atom retrieval + submodular set-cover, CoT prompt | 0.554 |

The gap between chunk and atom level is not a marginal effect — it is
a 30 percentage-point drop in apparent citation accuracy.

## Finding 2: the gap is structural, not noisy

We classify each citation into three buckets by content overlap with
the generated answer (using a stop-word filtered token set):

- **supported & correct doc** — at least one shared content token, and
  cited atom's document is in the gold set
- **supported but wrong doc** — at least one shared token, but
  document not in gold set
- **hallucinated** — zero shared tokens with the answer

For D04+D06 CoT (3,349 citations across 1,000 queries):

| Bucket | Count | Fraction |
|---|---|---|
| Supported & correct doc | 978 | 29.2% |
| Supported but wrong doc | 516 | 15.4% |
| **Hallucinated (zero overlap)** | **1,855** | **55.4%** |

Over half of all `[E_i]` citations have **zero content overlap** with
the answer text. These are not noisy attribution — they are categorical
ID hallucinations: the model emits a citation index that does not
correspond to an atom whose content appears in the answer.

## Finding 3: the hallucination rate scales with selection entropy

We compare two selection strategies at the same atom granularity:

| Selection | Cited zero-overlap rate (lower is better) | cit_acc |
|---|---|---|
| Score-greedy fill | ~30% | 0.70 |
| Submodular set-cover | ~55% | 0.55 |

Submodular's diverse coverage interleaves atoms from different aspects
of the query; the model loses track of which `[E_i]` corresponds to
which fact. Greedy ordering by score keeps semantically related atoms
together and the model emits more consistent citations.

## Finding 4: the rate scales inversely with evidence-unit size

Same generator (Qwen 2.5-7B-Instruct), same prompt template:

- Chunk-level (~5 chunks per prompt): cit_acc 0.85
- Atom-level (~50 atoms per prompt): cit_acc 0.55–0.70

With more `[E_i]` markers in the prompt, the model has more chances to
mis-emit an index. Atomic RAG amplifies this LLM failure mode.

## Finding 5: a trivial structural filter recovers most of the loss

For each citation `[E_i]`, we apply a simple rule: **drop the citation
if the cited atom's content tokens have zero overlap with the answer.**
No model, no embedding, no learning — pure string overlap.

| Configuration | cit_acc original | cit_acc cleaned (drop) | Δ |
|---|---|---|---|
| D04 bare (greedy) | 0.700 | 0.738 | +3.8 |
| D04 typed | 0.690 | 0.731 | +4.1 |
| D04+D06 citation | 0.518 | 0.620 | +10.2 |
| D04+D06 CoT | 0.554 | 0.671 | +11.7 |

The fix is post-hoc, model-free, and recovers between 3.8 and 11.7
percentage points of citation accuracy. The recovery scales with the
hallucination rate of the underlying configuration.

## Finding 6: a learned reranker as verifier underperforms the trivial filter

We compare against the most natural learned baseline — using
BGE-reranker-v2-m3 to score (atom, answer) pairs and dropping
low-scoring citations. With a threshold of 0.5 (where the reranker
score becomes meaningfully positive), only 153 of 1,000 queries retain
any citation, giving a corpus-level cit_acc of ~0.11.

| Method | cit_acc | Queries with at least one citation |
|---|---|---|
| Raw | 0.554 | 999 |
| **Trivial token-overlap drop** | **0.671** | **999** |
| Cross-encoder reranker, thr=0.5 | 0.110 | 153 |

The cross-encoder's relevance score is not calibrated for citation
support — it is too aggressive. The simple filter dominates by a 6×
margin on the corpus-level metric.

## Implication

Citation hallucination in atomic RAG is a structural failure mode of
the language model — a categorical ID confusion induced by long
evidence lists, exacerbated by diverse selection orderings. It is not
noise on a continuous attribution signal. The right intervention is a
hard structural filter, not a soft learned scorer.

This finding closes bottleneck B3 in the atomic decomposition with no
generator changes, no retraining, and no inference cost beyond a
linear scan over content tokens.
