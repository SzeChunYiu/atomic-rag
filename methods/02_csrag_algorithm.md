# CS-RAG Algorithm

Purpose:
Select the smallest sufficient evidence set.

Pipeline:
1. Receive detected evidence atoms.
2. Infer query facets.
3. Estimate each atom's coverage of each facet.
4. Greedily select atoms with highest marginal utility.
5. Penalize redundancy and token cost.
6. Guard against missing required facets.
7. Emit selected evidence set.

Start with greedy selection.
Move to optimization only after the greedy baseline is understood.
