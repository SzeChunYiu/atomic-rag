# Crowding Benchmark — Controlled Synthetic Sweeps

## Purpose
Build a closed-form benchmark whose ground truth is constructed, not annotated, so we can sweep distractor density, similarity, and bridge-hop distance and see whether support-chain recovery has a phase transition.

## Why a synthetic benchmark first
Real datasets confound failure modes. HotpotQA gold-doc lists make us guess where the answer atom lives. The audit (`gold_atom_audit.py`) reveals where signal is lost on real data, but we cannot *control* crowding on real data — we can only measure it. To prove a phase transition exists we need to vary one knob and watch P(success) collapse.

## Generator

Two-hop bridge template:

```text
Atom A (hop1):  "{film} was directed by {director}."
Atom B (hop2):  "{director} was born in {country}."
Query:          "What country was the director of {film} born in?"
Answer:         {country}
```

Distractors are sampled from typed pools:
- entity-overlap distractors: same `{director}` or `{film}`, wrong relation
- type-overlap distractors: same claim_type (`WHERE`), unrelated entity
- semantic-similar distractors: paraphrases of the answer relation pointing to wrong country
- noise distractors: random typed atoms

## Controlled axes

| Axis | Values |
|------|--------|
| `n_distractors_per_gold` | 0, 2, 5, 10, 20, 50 |
| `semantic_similarity` | low, medium, high (cosine bins against gold atom) |
| `entity_overlap` | none, partial, high |
| `answer_type_overlap` | false, true |
| `chunk_size` | 80, 160, 384, 768 |
| `chunk_mixing` | gold_isolated, gold_with_distractors, bridge_buried |
| `hop_count` | 1, 2, 3 |
| `token_budget` | 128, 256, 512, 1024 |

Phase-1 run keeps `hop_count=2`, `chunk_size=384`, varies `n_distractors`, `semantic_similarity`, `token_budget` only — 6 × 3 × 4 = 72 cells, 50 queries each.

## Oracle metric (no LLM in loop)

```text
support_chain_complete(selected_atoms, gold_atoms) =
    all(g in selected_atoms for g in gold_atoms)
answer_oracle_success = support_chain_complete
```

This avoids paying for LLM generation while we are still studying retrieval/selection. Generation-stage metrics are added in Phase 2.

## Output schema
`crowding_sweep_results.jsonl` — one row per (system, query, cell):
- system_name, query_id, n_distractors, semantic_similarity_bin, entity_overlap_bin, chunk_size, hop_count, token_budget
- gold_doc_recall_at_k, gold_atom_recall_at_k, gold_atom_selected, support_chain_complete
- selected_tokens, latency_ms

`phase_diagram_summary.json` — per-system, per-axis summary of P(success) and the estimated threshold C* (50% crossover).

## Falsification

If every system degrades smoothly without a sharp threshold, soften the claim from "phase transition" to "crowding sensitivity." Keep the benchmark either way — sensitivity curves are still publishable.
