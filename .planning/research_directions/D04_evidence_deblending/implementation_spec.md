# D04 — Implementation spec (Phase 1, minimal viable)

## File layout
```
src/astro_cs_rag/
  atoms/
    deblend.py            # NEW — sentence split + claim-type tagger
    schemas.py            # extend EvidenceAtom with `claim_type` field
  retrieval/
    atom_dense.py         # NEW — atom-level dense retrieval
  pipeline/
    deblend_run.py        # NEW — pipeline stage between index and retrieve
scripts/
  build_atoms.py          # CLI to materialize atoms.jsonl per index
```

## Data contract

```python
class EvidenceAtomV2:
    atom_id: str          # f"{chunk_id}::s{sentence_idx}"
    chunk_id: str
    doc_id: str
    text: str             # the sentence
    claim_type: Literal["WHO", "WHEN", "WHERE", "WHAT_NUM",
                        "WHAT_OBJ", "ANY"]
    embedding: np.ndarray # (d,)
    span_start: int
    span_end: int
    source_doc_id: str
```

## API surface
```python
def deblend_chunks(chunks: list[Chunk], embedder) -> list[EvidenceAtomV2]: ...
def type_tag(text: str) -> str: ...
def atom_retrieve(query: str, atoms: list[EvidenceAtomV2],
                  k: int = 50,
                  lambda_type: float = 0.05) -> list[Candidate]: ...
```

## Test plan (must pass before benchmark)
- Unit: `type_tag("When did WWII end?") == "WHEN"`.
- Unit: `type_tag("1945")` returns `WHEN`.
- Unit: `deblend_chunks` produces ≥1 atom per non-empty chunk.
- Integration: HotpotQA-1k corpus produces 200k–400k atoms.
- Sanity: atom-level retrieval recovers gold doc in top-50 with
  recall@50 ≥ chunk-level retrieval (no recall regression).

## Benchmark spec
- Dataset: HotpotQA-1k cs=384 terse.
- Baselines:
  - greedy/dense (chunk-level), F1=0.397.
  - greedy+rerank/dense (chunk-level), F1=0.416.
  - greedy+rerank+CoT, F1=0.633.
- Variants:
  - D04-bare (atom-retrieval, lambda_type=0).
  - D04-typed (atom-retrieval, lambda_type=0.05).
  - D04-typed + CoT.
- Metrics: F1, EM, cit_acc, all_gold_in_sel, latency.
- Significance: paired bootstrap n=10000 vs greedy baseline.

## Stop conditions
- Phase 1 produces F1 ≥ 0.50 on D04-bare → continue to typed-bonus
  ablation.
- Phase 1 produces F1 < 0.50 on D04-bare → diagnose by gold-atom
  presence in top-50; if all_gold_in_atom_retrieve < 0.92, atom
  extraction itself is lossy. Iterate the deblender; do not proceed
  to type bonus.
- Phase 1 produces no measurable improvement after 2 deblender
  iterations → write decision.md as "deferred — atom-level
  granularity insufficient signal in this codebase".

## Compute budget
- Build: 1 SLURM job, ≤ 30 min on A100.
- Bench: 3 SLURM jobs (D04-bare, D04-typed, D04-typed+CoT), ≤ 30 min each.
- Total: ≤ 2 hours of GPU.
