# D04 — Bottleneck analysis (root-cause traceback)

## Observed bottleneck (from prior atomic decomposition)

End-to-end metric: F1 on HotpotQA-1k bridge with terse generation.

```
F1 = 0.397 (greedy/dense)
F1 = 0.633 (greedy+rerank+CoT, current best)
```

## Recursive trace

`answer error → context error → evidence selection error →
candidate retrieval error → atom extraction error → corpus/index error
→ theory assumption error.`

For the queries that fail F1 at greedy/dense:

**Layer 1 — corpus/index:** all_gold@50 = 0.856. The 14% with
incomplete gold coverage have F1 = 0.19 (vs 0.61 with full gold).

**Layer 2 — candidate retrieval:** GWR (graph propagation) raises
all_gold@50 to 0.916 (+6pp). Confirms Layer 1 has headroom and
graph-walk recovers it.

**Layer 3 — atom extraction:** GWR's gain dies here. all_gold_in_sel
remains at 0.858, identical to greedy. Reason: bridge chunks have
LOW cos(query, chunk_emb), so their atoms compute LOW SNR and the
selector skips them. **THIS is the bottleneck D04 attacks.**

**Layer 4 — selection:** Token budget filling 21 atoms; bridge atoms
never reach this stage to be selected.

**Layer 5 — generation:** CoT helps when gold is in context (+5pp F1
absolute). Independent stage; D04 should compose.

## The atom-extraction failure mode

Currently:
- `chunk_emb = encode(384_token_window)`
- `atom = (chunk_id, snr=cos(query, chunk_emb))`
- An atom inherits the WHOLE chunk's score.

Problem: a chunk can be relevant for the query (high cos) AND contain
unrelated answer atoms. Or a chunk can be marginally relevant (low
cos) AND contain the bridge answer atom. Either way, atom-level signal
is washed out by chunk-level scoring.

D04's fix: split chunk into sentence-level atoms with their OWN
embeddings. A bridge sentence ("X's mother was Y, born in 1932")
gets its own embedding and its own claim-type. Even if the chunk
overall has cos=0.3, this specific sentence might be the answer-bearing
unit.

## Predicted improvement

If the bottleneck is correctly identified:
- all_gold_in_sel should rise from 0.858 to >0.92 (matching GWR's
  candidate-level gain).
- F1 should rise from 0.397 to ~0.55 (matching the
  `0.92 × F1_with_all_gold + 0.08 × F1_without`).
- Combined with greedy+rerank+CoT, projected F1 ≈ 0.65-0.68.

## What would falsify this analysis

- D04 raises all_gold_in_sel by >0.04 but F1 stays flat → atom-level
  signal IS reaching the generator but the generator can't use it
  (bottleneck is downstream of atom extraction).
- D04 raises F1 but all_gold_in_sel doesn't move → mechanism is not
  what we claimed; some other feature of atom-level retrieval is
  driving the gain (worth investigating).
- D04 produces no measurable effect → atom-level granularity is too
  fine-grained for the LLM to use; chunks were already at the right
  scale.
