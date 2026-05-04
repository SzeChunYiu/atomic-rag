# D04 — Algorithm specification

## Inputs
- Corpus: list of documents.
- Query: text.

## Output
- Ranked list of evidence atoms with typed slots, suitable for the
  selector and downstream generator.

## Pipeline

### Stage A — Atom extraction (one-time, offline)
For each chunk (cs=384):
1. Sentence-split (regex on `.!?` + abbreviation guard).
2. For each sentence, compute a claim-type tag using lightweight
   patterns:
   - `WHEN`: contains a 4-digit year, month name, or `DATE` regex.
   - `WHO`: capitalized 2+ token entity at subject position (heuristic).
   - `WHERE`: contains a known location-token suffix (`,
     <COUNTRY>`, "City of", etc.) or a country name from a small list.
   - `WHAT_NUM`: contains a number > 100 or unit (km, kg, %, etc.).
   - `WHAT_OBJ`: anything else; default.
3. Embed the sentence with the same encoder as the chunk index
   (BAAI/bge-m3).
4. Store: `(chunk_id, atom_idx, sentence, type_tag, sentence_emb,
   span_start, span_end, source_doc_id)`.

### Stage B — Query intent typing (per query)
1. Apply a regex/keyword tagger to the query:
   - "when", "year", "date" → `WHEN`
   - "who", "person", "name" → `WHO`
   - "where", "located", "city", "country" → `WHERE`
   - "how many", "how much", number-question patterns → `WHAT_NUM`
   - else → `ANY`

### Stage C — Atom retrieval
Score each atom $a$ for query $q$:

$$\mathrm{score}(a) = \cos(q_\mathrm{emb}, a_\mathrm{emb}) +
\lambda_\mathrm{type} \cdot \mathbb{1}[\mathrm{type}(a) =
\mathrm{intent}(q)]$$

with $\lambda_\mathrm{type} = 0.05$ (tunable).

Take top-K atoms by score (K=50).

### Stage D — Selection
Pass atoms (instead of chunks) to the existing greedy/v4 selector.
Token budget: 1024 tokens of atom-text concatenated.

### Stage E — Generation
Same as before (terse or CoT prompt). Each evidence block in the
prompt is now ONE atom (not a 384-token chunk).

## Pseudocode
```python
def deblend_corpus(chunks, embedder):
    atoms = []
    for ch in chunks:
        for sent in split_sentences(ch.text):
            t = type_tag(sent)
            a = Atom(
                chunk_id=ch.chunk_id,
                doc_id=ch.doc_id,
                text=sent,
                type=t,
                emb=embedder.encode(sent),
                span=(ch.span_start, ch.span_end),
            )
            atoms.append(a)
    return atoms

def retrieve_atoms(query, atoms, lambda_type=0.05, k=50):
    q_emb = embedder.encode(query)
    q_type = type_tag(query)
    scores = []
    for a in atoms:
        s = cos(q_emb, a.emb)
        if a.type == q_type:
            s += lambda_type
        scores.append(s)
    return top_k(atoms, scores, k)
```

## Complexity
- Offline: O(N_chunks × avg_sentences_per_chunk × embed_cost). For
  HotpotQA-1k that's ~50000 chunks × ~5 sentences = 250k atoms × 1 ms
  per embedding ≈ 4 minutes.
- Per query: O(N_atoms) dot product = 250k × 1024-d float = 250M ops ≈
  0.5s. Use FAISS index for sub-millisecond lookups.

## Falsification (re-stated for the algorithm)
- Atom-level F1 on HotpotQA-1k must exceed greedy-chunk-level F1 by
  ≥3pp absolute.
- The typed-bonus contribution (lambda_type=0 vs 0.05 vs 0.1 sweep)
  must show a non-flat ablation curve. If flat, type-channel is
  inactive — drop the type tagger.
