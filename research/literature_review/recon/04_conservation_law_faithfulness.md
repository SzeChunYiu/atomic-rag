# Conservation-law faithfulness verification

**Status:** stub.

## 1. Method one-liner
Define explicit conservation laws over selected evidence: entity-count
balance, numerical claim consistency, time-ordering DAG. The generated answer
must "balance" against the evidence's books; large residuals → unsupported
claim or missing evidence → abstain or re-retrieve.

## 2. Physics analog (operator-level)
Energy-momentum and quantum-number conservation in HEP event reconstruction.
Detector data must satisfy $\sum p^\mu = 0$; violations indicate missing
particles. Same operator structure here.

## 3. Closest prior art (preliminary)
- RAGAS (Es et al. 2023): LLM-judge faithfulness.
- FActScore (Min et al. 2023): atomic fact decomposition + verification.
- Self-RAG (Asai et al. 2023): explicit retrieve-and-critique.
- Q^2 (Honovich et al. 2021): question-generation faithfulness.
All use *learned* judges. Symbolic conservation residuals are new.

## 4. Novelty estimate
- algorithmic: high (symbolic, computable, no LLM judge required).
- theoretical: medium-high (residual is a falsifiable signal).
- empirical: high if residual correlates with human faithfulness judgments.

## 5. Why publishable
Attacks F7 (multi-source contradiction) and F9 (generator drift) with a
*computable, symbolic* signal. Reviewer-friendly because the residual is
algorithmically defined; no LLM-as-judge concerns.

## 6. Falsification protocol
If the residual does not correlate with human or RAGAS faithfulness above
$\rho>0.5$ on a held-out set, drop the claim that residual = faithfulness.
Salvage as an abstention signal if precision is high enough.

## 7. Status
- [ ] prior-art search done
- [ ] minimal implementation sketch
