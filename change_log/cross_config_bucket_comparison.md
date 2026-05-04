# Cross-configuration bucket comparison (2026-05-03)

## Setup
Compared per-query failure buckets across two atom-level configurations
to measure how D06 (submodular set-cover) + CoT prompting shift the
failure-mode distribution.

## Bucket distributions

| Bucket (F1+/-, cit+/-, gold+/-, text+/-) | D04 bare (greedy, citation prompt) | D04+D06 (submodular, CoT) | Δ |
|---|---|---|---|
| F1+ cit+ gold+ text+ (all good) | 37.2% | 31.0% | -6.2 |
| **F1- cit+ gold+ text+ (B4: gen fails with support)** | **19.2%** | **13.9%** | **-5.3** |
| F1- cit- gold+ text- (B-span) | 10.6% | 11.9% | +1.3 |
| F1- cit+ gold+ text- (no atom-text support) | 10.0% | 8.2% | -1.8 |
| F1- cit- gold+ text+ (selection failure) | 9.7% | 12.4% | +2.7 |
| F1+ cit+ gold+ text- (closed-book hit) | 9.2% | 7.9% | -1.3 |
| **F1+ cit- gold+ text+ (right answer, halluc cite)** | **2.9%** | **10.7%** | **+7.8** |
| F1+ cit- gold+ text- | 1.1% | 3.9% | +2.8 |
| Catastrophic | 0.1% | 0.1% | 0 |

## Aggregates

| Metric | D04 bare | D04+D06 CoT | Δ |
|---|---|---|---|
| F1 (mean) | 0.482 | 0.503 | +2.1 |
| EM | 0.380 | 0.385 | +0.5 |
| cit_acc raw | 0.700 | 0.554 | -14.6 |
| cit_acc cleaned | 0.738 | 0.671 | -6.7 |
| F1 vs cit correlation | 0.171 | 0.054 | -0.117 |
| bridge_in_pool | 86.2% | 85.3% | -0.9 |

## Three structural findings

**1. D04+D06+CoT genuinely improves F1 by addressing B4.**
The "F1- cit+ gold+ text+" bucket — answer support is in the pool, but
the generator failed to extract it — drops from 19.2% to 13.9%. That's
53 queries flipped from generation-failure to correct. Combined with
the other shifts, +2.1pp F1 mean.

**2. D04+D06+CoT degrades cit_acc by creating "right answer, halluc cite"
cases.** The "F1+ cit- gold+ text+" bucket grows from 2.9% to 10.7% —
78 queries where the model gets the right answer but emits hallucinated
citation IDs. CoT lets the model reason its way to the correct answer
without needing accurate citation tracking, so it stops bothering with
correct cites.

**3. F1 and cit_acc are independent under CoT.** Correlation drops from
0.171 to 0.054. With CoT, improving F1 and improving cit_acc are nearly
orthogonal — each must be tackled separately. This is good news for
the paper: it justifies separate contributions for each metric.

## Implications for the paper

The atomic decomposition is empirically falsified-or-confirmed at this
level. Specifically:

- **B4 is real and large**: -5.3pp on bucket, +2.1pp on aggregate F1
  even with the modest CoT-only intervention. A targeted B4 method
  (prompt-ordering replicas) should add more.
- **B3 fix (token-overlap drop) becomes more impactful as CoT is
  enabled**: the new "right-answer, halluc-cite" bucket is exactly
  what the overlap filter targets. Predicted: +11.7pp cit_acc when
  applied to D04+D06 CoT (already verified — that's where the figure
  came from).
- **B1 and B2 fixes are still pending**: bridge_in_pool stayed at ~85%
  across configurations, so neither selection nor CoT moved B2. RRF
  + multiscale + PRF will be measured against this 85% baseline.
- **The methods have orthogonal targets**: B3 (cit), B4 (F1 from
  generation), B1 (F1 from selection), B2 (F1 from retrieval).
  Combined gains should approximately add — falsifier: if combined gain
  is much less than sum of individual gains, methods interact.

## Predicted final stack (post-data)

If each method adds its predicted gain independently:

| Component | F1 contribution | cit_acc contribution |
|---|---|---|
| D04+D06+CoT baseline | 0.503 | 0.554 |
| + RRF retrieval | +0.02 (B2) | 0 |
| + MaxEnt selection | +0.02 (B1) | 0 |
| + replicas (B4) | +0.03 (B4) | 0 |
| + token-overlap cite cleanup (B3) | 0 | +0.117 |
| **Stack F1 / cit_acc** | **0.573** | **0.671** |

This would still be below the current best chunk-pipeline F1 (0.633).
For atomic-RAG to beat chunk-pipeline, we'd need the gains to be
super-additive (interactions favorable) or to find a 5th independent
intervention. Most likely path to F1 > 0.633 at atom level: improve
the tagger to give claim-type confidences (fixes the ANY=0 problem,
recovers some retrieval signal that's currently being thrown away).
