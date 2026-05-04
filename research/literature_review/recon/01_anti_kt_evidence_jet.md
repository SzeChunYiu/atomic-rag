# Anti-$k_T$ evidence-jet clustering for RAG

## 1. Method one-liner
Cluster candidate evidence atoms using the anti-$k_T$ jet algorithm with a
distance metric in (relevance, semantic-distance) space; the leading jet is
the selected evidence set. Inherits IRC-safety from the algorithm: stable
under chunk re-splitting and low-relevance distractors.

## 2. Physics analog (operator-level)
Anti-$k_T$ algorithm (Cacciari, Salam, Soyez 2008, JHEP 04:063):
$$d_{ij} = \min(p_{T,i}^{-2}, p_{T,j}^{-2}) \frac{\Delta R^2_{ij}}{R^2}, \quad d_{iB} = p_{T,i}^{-2}.$$
We substitute $p_{T,i} \to s_i$ (relevance score, dense or fused) and
$\Delta R_{ij} \to$ semantic distance in the embedding space (geodesic on the
unit sphere). The same operator; we are not borrowing a metaphor, we are
borrowing the algorithm.

## 3. Closest prior art (preliminary — to be validated by full search)
1. (2008) Cacciari, Salam, Soyez — *anti-kT* original. Not RAG.
2. (1998) Carbonell & Goldstein — *MMR* — diversity selector, no IRC properties.
3. (2018) Wilhelm et al. — *Practical Diversified Recommendations on YouTube* — DPP
   selector. Determinantal point process; quite different mathematics.
4. (2019) Yu et al. — *Submodular RAG context selection*. Greedy marginal
   submodular; no chunk-boundary safety claim.
5. (2023) Sarthi et al. — *RAPTOR* — hierarchical clustering of chunks via
   k-means. Different objective; no algorithmic IRC theorem.

**Distinct contribution.** No prior IR/RAG work imports a jet-clustering
algorithm; no prior selector proves IRC-safety; no prior selector demonstrates
robustness to chunk-size perturbation as a *property of the algorithm* rather
than an empirical observation.

## 4. Novelty estimate
| dimension | grade | justification |
|---|---|---|
| algorithmic | high | importing an unrelated-domain algorithm with new metric |
| theoretical | high | IRC-safety theorem restated for retrieval; *novel* invariance |
| empirical | medium | ablations against MMR, DPP, submodular; chunk-perturbation curve |

## 5. Why publishable
- NMI: cross-domain methodological transfer (HEP → IR), with a theorem.
- NeurIPS / SIGIR: clean novel selector with a non-trivial invariance property.
- Attacks F4 (split evidence) and F2 (distractor swarm) directly; secondary
  on F8 (multi-hop) via cluster-mediated entity bridging.
- Falsifiable, reproducible, single-file algorithm.

## 6. Falsification protocol
- **Kill condition A:** under matched compute and matched candidate pool,
  anti-$k_T$ does **not** beat MMR + cross-encoder on any of HotpotQA-1k,
  NQ-open-1k, MuSiQue-1k by ≥ 1 pp recall@10 *and* ≥ 1 pp answer F1. We then
  pivot the selector to a diagnostic role and remove the headline claim.
- **Kill condition B:** chunk-perturbation robustness curve does not show a
  statistically significant gap (paired bootstrap, $p<0.01$, 1000 resamples)
  between anti-$k_T$ and MMR. If the gap vanishes, we drop the IRC-safety
  *claim* even if average gain holds.

## 7. Dependencies and risks
- Embedding for semantic distance: BGE-M3 already in stack.
- O(N log N) implementation: implementable in 200 lines using fastjet-style
  $k_T$ recombination over $N \le 50$ candidates. No external dep needed.
- Risk: in low-N regime (small candidate pool) jet algorithm reduces to
  trivial clustering; gain may vanish. Mitigation: sweep candidate-pool sizes
  in ablations.

## 8. Status
- [x] physics-analog mapping written
- [ ] prior-art search done (need ACL/SIGIR/EMNLP 2020–2026 + arXiv)
- [ ] minimal implementation sketch
- [ ] ablation plan written
- [ ] first benchmark numbers
- [ ] falsification run
