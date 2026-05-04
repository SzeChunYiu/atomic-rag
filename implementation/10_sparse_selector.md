# Sparse Selector Module

Inputs:
- detected evidence candidates
- query facets
- token budget

Steps:
1. estimate coverage gain
2. estimate redundancy
3. estimate contradiction if available
4. greedily select atoms
5. stop by coverage or budget

Outputs:
- selected evidence set
- dropped candidates with reasons
- coverage trace
- token count
