# Sparse Objective Math

Greedy utility:

```text
utility(atom) =
  new_coverage
+ support_strength
+ reliability
- redundancy
- contradiction
- token_cost
- uncertainty
```

Stop when:
- all required facets covered, or
- token budget reached, or
- marginal gain falls below threshold.

Guardrail:
If evidence recall is low, do not over-compress.

The selector must explain every atom it keeps or drops.
