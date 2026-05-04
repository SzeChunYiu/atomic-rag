# D06 — Submodular sparse reconstruction algorithm

## Setup
Inputs:
- $q$ — the query.
- $\mathcal{A}$ — candidate evidence atoms (from D04 retrieval).
- $B$ — token budget.
- Each atom $a$ has: text, embedding $v_a$, claim type $t_a$, source
  doc $d_a$, score $s_a$.

Output: subset $S \subseteq \mathcal{A}$ with $\sum_{a \in S}
\mathrm{tok}(a) \le B$.

## Coverage utility (submodular)

Define **facets** as (claim_type × distinct entity mention). For HotpotQA
bridge queries, gold facets are typically two: the bridge entity and
the answer entity, each in its own claim_type.

$$F(S) = \sum_{f \in \mathcal{F}_q} \left[1 - \prod_{a \in S} (1 -
\mathrm{cov}(a, f))\right]$$

where:
- $\mathcal{F}_q$ — query facets, derived from query NER + intent type.
- $\mathrm{cov}(a, f) \in [0, 1]$ — atom $a$'s coverage of facet $f$.

This $F$ is submodular: adding an atom that already covers a facet
yields diminishing returns.

## Per-atom coverage definition

```
cov(a, f):
    if claim_type(a) != f.type: return 0.0
    if f.entity not in a.text:   return 0.0
    return min(1.0, score(a))
```

## Greedy submodular maximization

```
S = ∅; budget_used = 0
while budget_used < B:
    best_atom = argmax_{a ∈ A \ S, tok(a) ≤ B - budget_used} (F(S ∪ {a}) - F(S)) / tok(a)
    if best_atom is None or F(S ∪ {best}) - F(S) ≤ 0: break
    S = S ∪ {best_atom}
    budget_used += tok(best_atom)
return S
```

(1 − 1/e) approximation guarantee for monotone submodular maximization.

## Computational cost
- $O(B / \mathrm{avg\_tok}(a) \cdot |\mathcal{A}|)$ per query.
- For HotpotQA-1k: 21 atoms × 50 candidates × |F| ≈ 1000 ops/query.

## Falsification
- If $F1$ with submodular ≤ greedy-by-score $F1$ on the SAME atoms
  (D04 baseline), submodular is not adding value.
- If $F1$ improves by < 1pp, contribution is marginal; document but
  do not headline.
