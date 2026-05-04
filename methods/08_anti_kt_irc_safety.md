# Anti-$k_T$ Evidence-Jet Clustering — IRC-Safety in RAG

## 1. Setting

Let $\mathcal{A} = \{a_i\}_{i=1}^N$ be a set of evidence atoms with relevance
scores $s_i > 0$ and unit-norm semantic embeddings $\hat e_i \in S^{d-1}$.
Define the *retrieval anti-$k_T$ metric*

$$
d_{ij} \;=\; \min\!\bigl(s_i^{-2},\, s_j^{-2}\bigr) \cdot \frac{\Delta_{ij}^2}{R^2},
\qquad
d_{iB} \;=\; s_i^{-2},
$$

with $\Delta_{ij} = 1 - \langle \hat e_i, \hat e_j \rangle$ and jet radius
$R \in (0, 2]$. Run the standard sequential recombination: at each step take
the smallest $d_{ij}$ or $d_{iB}$. If $d_{ij}$ wins, merge $i$ and $j$ into a
proto-jet of relevance $s_i + s_j$ and direction
$(s_i \hat e_i + s_j \hat e_j)/\|\cdot\|$; if $d_{iB}$ wins, finalize $i$ as a
jet and remove it from the pool.

This is the operator from Cacciari, Salam, Soyez (JHEP 04:063, 2008), with
$p_T \to s$ and azimuth-rapidity distance $\to$ cosine distance.

## 2. Two invariances we claim

Define the *leading-jet relevance* $S_J = \sum_{i \in J} s_i$ and the
*hard support set* $H_J(\tau) = \{ i \in J : s_i > \tau \}$ for threshold $\tau$.

**Theorem (Collinear safety).** Replace any single atom $i$ with two co-located
atoms $i_1, i_2$ at the same embedding $\hat e_i$ and relevances
$s_{i_1} + s_{i_2} = s_i$. Then the leading-jet relevance $S_J$ and centroid
direction are unchanged, and the hard support set $H_J(\tau)$ for any
$\tau < \min(s_{i_1}, s_{i_2})$ is unchanged after the substitution
$i \mapsto \{i_1, i_2\}$.

**Theorem (Infrared safety).** Add an atom $i_0$ with $s_{i_0} \to 0^+$. For
sufficiently small $s_{i_0}$, the leading-jet's centroid direction and total
relevance change by an amount $O(s_{i_0})$, the hard support set is
unchanged, and no hard atom is dropped from the leading jet.

**Sketch.** Both follow from the standard anti-$k_T$ proofs: the metric has a
$\min(s_i^{-2}, s_j^{-2})$ prefactor that makes hard–hard pairs have strictly
larger distance than soft–hard pairs, so soft atoms cluster *into* hard ones
rather than perturbing the hard structure. Co-located splitting preserves the
sum of $p_T$ contributions because the linear combination coefficient
$s_{i_1} + s_{i_2}$ enters exactly as $s_i$ would. See §2 of Cacciari–Salam–
Soyez for the original kinematic argument; the substitution to $(s, \cos\Delta)$
preserves the algebra because both the $\min$ prefactor and the $\Delta^2/R^2$
factor are positive-monotone.

## 3. Why these invariances matter for RAG

| Invariance | Atomic failure mode it neutralizes |
|---|---|
| collinear | F4 (split evidence): chunking-boundary perturbations leave the leading jet's hard members unchanged |
| infrared | F2 (distractor swarm), F5 (popular-but-empty): low-relevance atoms cannot push hard answer-bearing atoms out |

Empirically, this means anti-$k_T$ should *out-perform MMR / DPP / submodular
greedy on chunk-size sweeps and noisy-distractor injections*, without any
hyper-parameter tuning specifically for those perturbations.

## 4. Falsification protocol (numerical IRC test)

For each of HotpotQA-1k, NQ-open-1k, MuSiQue-1k:

1. Run the dense+rerank+anti-$k_T$ pipeline at chunk sizes
   $\{256, 384, 512, 640, 768\}$ tokens.
2. Run the same pipeline with MMR, DPP, plain greedy at the same chunk sizes.
3. Plot answer-F1 vs chunk size for each selector.
4. **Pass condition.** The variance of anti-$k_T$'s F1 across chunk sizes is
   strictly smaller than the variance of any other selector's F1, with a
   paired bootstrap p-value $p < 0.01$ (1000 resamples).
5. Inject distractor pools of size $\{0, 5, 10, 25, 50\}$ low-relevance atoms.
   Same pass condition on F1 vs distractor count.

Failing the chunk-size variance test invalidates the IRC-safety *empirical*
claim even if average performance is competitive — and we publish the failure
as a null result.

## 5. Implementation

`src/astro_cs_rag/selection/anti_kt.py` (175 lines, pure Python) +
`src/astro_cs_rag/selection/jet_select.py` (89 lines, pipeline adapter).

Tests covering correctness, collinear safety, infrared safety, and
$R$-dependence: `tests/test_anti_kt.py`, `tests/test_jet_select_pipeline.py`.

## 6. References

- Cacciari, Salam, Soyez. *The anti-$k_T$ jet clustering algorithm*. JHEP 04:063 (2008).
- Cacciari, Salam, Soyez. *FastJet user manual*. Eur. Phys. J. C 72:1896 (2012).
- Carbonell & Goldstein. *MMR diversity selector*. SIGIR 1998 (the baseline).
- Wilhelm et al. *Practical Diversified Recommendations on YouTube*. CIKM 2018 (DPP baseline).
