# Root Cause Analysis — Performance Ceiling (2026-05-04)

## TL;DR

Two bugs masked all progress. After fixing them, the corrected numbers reveal
a clear, dataset-specific bottleneck decomposition. One breakthrough (CoT) was
already in production but invisible.

---

## Bug 1: Citation markers in EM/F1 computation (FIXED)

**Symptom**: `answer_em_mean = 0.000` for every single run across all datasets.
`answer_f1_mean` underreported by ~25-40%.

**Cause**: The normalizer `normalize_answer()` strips punctuation but leaves
`[` and `]` as spaces — so `[E2]` becomes `e2` as a token. Every answer with
a citation marker would fail EM: "american e2" ≠ "american".

**Fix**: Strip `[E\d+]` from prediction before computing EM/F1 in
`evaluate_run.py::_score_answers`.

**Corrected numbers** (HotpotQA terse greedy):
- EM: 0.000 → **0.436**
- F1: 0.397 → **0.553**

---

## Bug 2: CoT outputs not extracting "Final answer:" (FIXED)

**Symptom**: CoT runs reported F1 ≈ 0.14, far below terse F1 = 0.55.
CoT appeared to be worse, and was not considered for the paper.

**Cause**: `_score_answers` computed EM/F1 on the full CoT output
(`"Reasoning: ... Final answer: X"`), not just the final answer.
The reasoning tokens dilute F1 and prevent any EM match.

**Fix**: Extract `Final answer:` portion with regex before scoring.
Added `_extract_answer()` to `evaluate_run.py`.

**Corrected numbers** (HotpotQA CoT greedy):
- F1: 0.142 → **0.604** (+5.1pp over terse greedy at similar cost)
- EM: 0.000 → **0.471**

The CoT gain was real all along. We just couldn't measure it.

---

## Corrected Pareto Frontier (HotpotQA-1k, Qwen 2.5-7B)

| Method | EM | F1 | Latency |
|--------|----|----|---------|
| Terse greedy | 0.436 | 0.553 | 590ms |
| Terse v4 (anti-kT) | 0.443 | 0.553 | 590ms |
| Terse rerank | 0.460 | 0.582 | 960ms |
| **CoT greedy** | **0.471** | **0.604** | ~800ms |
| **CoT rerank** | **0.498** | **0.633** | ~1.4s |

CoT greedy beats terse rerank at lower compute cost.

---

## Failure Mode Decomposition (diagnose_chunk_runs.py)

### HotpotQA-1k greedy (terse, cs=384)

| Failure type | Rate | Interpretation |
|---|---|---|
| Retrieval miss | 0.4% | Gold doc never retrieved — not a bottleneck |
| Selection miss | 13.8% | Gold retrieved but wrong chunks selected |
| **Generation miss** | **28.0%** | Gold text in context, model still wrong |
| Bridge coverage | 85.6% | Both gold docs in selected pool |
| Success (F1≥0.5) | 57.8% | |

**Primary bottleneck: the generator** (28%), not the retriever.
IRCoT (which targets retrieval) is the WRONG method for HotpotQA.

### HotpotQA-1k greedy+rerank (terse, cs=384)

| Failure type | Rate |
|---|---|
| Retrieval miss | 0.3% |
| Selection miss | 10.2% |
| Generation miss | 28.2% |
| Bridge coverage | 89.3% (+3.7pp) |
| Success | 61.3% (+3.5pp) |

Rerank improves bridge coverage and selection, but generation miss is identical.
The generator is the hard ceiling.

### 2WikiMultiHopQA-1k greedy (terse, cs=384)

| Failure type | Rate | Interpretation |
|---|---|---|
| Retrieval miss | 0.0% | Gold always in top-50 |
| **Selection miss** | **30.2%** | Wrong chunks selected — dominant failure |
| Generation miss | 40.9% | Gold text present but answer wrong |
| **Bridge coverage** | **47.8%** | Only 48% have BOTH gold docs selected |
| Success | 28.9% | |

**Primary bottleneck: multi-hop selection** — 52% of queries are missing
at least one gold document from the selected context. This is the IRCoT target.

### Key contrast

- HotpotQA: retrieval fine, selection mostly fine, **generator is the ceiling**
- 2Wiki: retrieval fine, **selection is the bottleneck** (missing second hop)

Different datasets → different methods needed. Physics-inspired selectors
(anti-kT, MMR) don't address either bottleneck directly.

---

## Breakthrough Plan

### Tier 1 (running now — job 3007853)

5 CoT runs to complete the metric table:
1. 2Wiki greedy + CoT
2. 2Wiki rerank + CoT
3. HotpotQA v4 + CoT
4. HotpotQA v4 + rerank + CoT
5. 2Wiki v4 + CoT

### Tier 2 (also in job 3007853)

Iterative retrieval for 2Wiki (`iterative_retrieval_bench.py`):
- Step 1: retrieve + CoT → extract intermediate claim
- Step 2: embed (query + intermediate) → retrieve second hop → augment context
- Step 3: final generation with both hops

Expected: bridge coverage 47.8% → 70%+, F1 0.285 → 0.40+

---

## Paper Implications

1. **Anti-kT v4 claim holds**: +1.4pp cit_acc on HotpotQA, Pareto-optimal
   at zero extra compute for citation accuracy. Still valid.

2. **CoT + anti-kT**: Whether v4 stacks with CoT is now measurable.
   Running now. If v4+CoT > CoT alone, that's an additional paper contribution.

3. **Failure mode taxonomy**: The retrieval/selection/generation decomposition
   is a novel diagnostic contribution independent of the method claims.
   28% generation bottleneck on HotpotQA, 52% bridge miss on 2Wiki — these
   numbers characterize the atomic RAG failure landscape.

4. **Iterative retrieval**: If the 2-step retrieval improves 2Wiki bridge
   coverage and F1, that's the headline result ("physics-inspired selection
   + iterative retrieval closes the multi-hop gap").
