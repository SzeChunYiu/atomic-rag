# D04 — Prior work

## Atom-level / sentence-level retrieval

- **Sentence-BERT** (Reimers & Gurevych 2019): embeds at sentence
  level, used as a default for fine-grained similarity. We use this
  pattern but with claim-type metadata.
- **DRUID** (Roller et al. 2024, https://arxiv.org/html/2410.22508v1):
  decomposes documents into discourse-relation units. Operates on
  rhetorical structure rather than answer-bearing atoms.
- **Sparse retrievers (SPLADE)** (Formal et al. 2021): score over
  vocabulary terms; adjacent to atom decomposition but lexical, not
  semantic-claim.

## Question typing / intent classification

- **Wh-question taxonomies** (Li & Roth 2002): classic 6-way taxonomy
  (entity/numeric/date/location/manner/description). We use this as
  the lightweight claim-type tagger.
- **DPR** (Karpukhin et al. 2020): treats queries as flat strings;
  relies on dense embedder to learn intent. We add an explicit type
  channel on top.

## Multi-hop QA via decomposition

- **Self-Ask** (Press et al. 2023): decompose multi-hop into
  sub-questions, retrieve per sub-question. Operates on the QUERY side;
  D04 operates on the EVIDENCE side.
- **DSP** (Khattab et al. 2022): programmatic
  decomposition+search+predict. We do not need full programmability —
  one-pass deblending should suffice.
- **HippoRAG** (Gutiérrez et al. 2024,
  https://arxiv.org/abs/2405.14831): builds a knowledge graph of
  extracted entities and uses PageRank for retrieval. Closest related
  work; D04 differs by working at the atom level (claim-typed) rather
  than entity-level only.

## Open-claim extraction

- **OpenIE** (Banko et al. 2007, Stanford OpenIE): extracts (subject,
  relation, object) triples. Our minimal deblender uses a far cheaper
  approach (sentence + claim-type) but the heavy version converges to
  OpenIE-style triples.

## Why this is RAG-native (per directions doc)

We do **not** copy a telescope/HEP "atomic detection" algorithm. We
adapt the *concept* of "atomic unit of evidence" to the QA setting,
where atoms are answer-bearing claims with typed slots. The decoder
benefits because it sees pre-extracted claims with explicit types,
reducing the burden of parse-extract-verify on the LLM.
