# Atomic decomposition v4 — per-query joint failure analysis (2026-05-03)

## New diagnostic: per-query bucketing of (F1, cit_acc, gold-in-pool, answer-text-in-pool)

Run on D04+D06 CoT (n=1000):

| Bucket (F1+/-, cit+/-, gold+/-, ans-text+/-) | Count | % |
|---|---|---|
| F1+ cit+ gold+ text+ (everything works) | 310 | 31% |
| **F1- cit+ gold+ text+ (B4 generation: gold present, cites right, answer wrong)** | **139** | **14%** |
| F1- cit- gold+ text+ (selection picked gold doc but failed to use it) | 124 | 12% |
| F1- cit- gold+ text- (gold doc but answer text not in atoms) | 119 | 12% |
| F1+ cit- gold+ text+ (right answer, hallucinated cites) | 107 | 11% |
| F1- cit+ gold+ text- (cites right, gold doc but no answer text in pool) | 82 | 8% |
| F1+ cit+ gold+ text- (right answer, hallucinated answer location) | 79 | 8% |
| F1+ cit- gold+ text- (right answer despite no support — model knowledge) | 39 | 4% |
| F1- cit- gold- text- (full retrieval miss) | 1 | 0.1% |

## Key findings

1. **F1 vs cit_acc correlation = 0.054 (essentially zero).** Citations and
   answer correctness are statistically independent. Confirms that
   citation cleanup (B3 fix) won't move F1 — the metrics are decoupled.

2. **B4 generation is the BIGGEST single bottleneck**, not B2 retrieval.
   - 26.3% of queries have gold answer text *in the selected pool*
     but F1 < 0.5
   - This is much larger than my earlier "8pp generation ceiling" estimate
   - The LLM is failing to extract the answer despite having it.

3. **B2 bridge recall**: 85.3% of queries have BOTH gold docs in the
   selected pool. So 14.7% miss at least one — that's the B2 quantitative.

4. **Span-selection failure**: 12% of queries have gold doc selected but
   answer text not in any selected atom (gold+ text-). This is a
   *within-doc selection* failure — different from B2 which is
   *cross-doc retrieval* failure.

## Updated bottleneck table

| Bottleneck | % of queries affected | Mechanism | Method |
|---|---|---|---|
| B4 generation: answer present but unused | 26% | LLM position/distraction bias | (NEW) prompt-ordering replicas |
| B2 bridge recall: 2nd gold doc missing | 15% | retrieval can't follow bridge entity | PRF + multi-scale RG (queued) |
| B-span: gold doc but wrong sentences | 12% | atom-level selection misses span | sharper selection (MaxEnt queued) |
| B-misc: full miss | 1% | catastrophic | none feasible |
| Saturated (everything works) | 31% | — | — |

The remaining ~15% are partial-credit cases (F1+ cit- gold+ text+ etc).

## Implication for paper-quality contributions

I had been undercounting B4 by 18pp. The biggest single intervention
opportunity is on the LLM-generation side, not retrieval/selection.

## NEW physics-inspired method for B4: prompt-ordering replicas

**Diagnosis.** 26% of failures are "gold answer text in pool but LLM
misses it." Documented LLM failure mode: position bias (Liu et al. 2023,
"Lost in the Middle"). The answer atom is buried mid-context and gets
low attention.

**Why physics methods help.** Multi-measurement averaging:
- Run the LLM K times with the SAME atoms in K different orderings
- Each ordering puts the answer atom at a different position
- Position-bias errors are uncorrelated across orderings
- Majority-vote final answer averages out the noise

**Distinct from self-consistency CoT** (which we already tried, was null):
- SC perturbs *decoding* (sampling temperature)
- This perturbs *prompt structure* (atom ordering)
- For Qwen 7B at greedy T=0, SC gives null because greedy is
  deterministic. Prompt-ordering replicas perturb at a layer where
  Qwen's bias actually lives.

**Cost.** K-x generation compute. K=3 is the cheapest meaningful test.
Selection is unchanged — only the assemble() function shuffles the
[E_i] indices for each replica.

**Falsifier.** Predicted to help the F1- cit+ gold+ text+ bucket (139
queries). If it fails on this bucket but helps elsewhere, mechanism
isn't position bias — possibly distraction by similar atoms.

**Status.** Not yet implemented. Will implement after current chain
clears so it sits as the headline B4 method.

## Sequencing

The 5-job chain (param sweep → research dirs → MaxEnt → PRF → multi-scale)
is GPU-blocked. Once it lands:
1. If MaxEnt fixes B1 selection imbalance — move on to B2 confirmation
2. If PRF or multi-scale fix B2 — measure remaining gap to ceiling
3. The remaining gap will be ~26% B4, addressable by prompt-ordering replicas
4. With B1, B2, B3 closed and B4 mitigated, headline F1 should move from
   0.633 (current best) to ≥0.65, with each contribution measurable
   independently

This is the path to publication-tier numbers if the methods work as
mechanism-predicted. The most-uncertain piece is whether prompt-ordering
replicas actually beat self-consistency null on Qwen 7B — it's a clean
prediction either way.
