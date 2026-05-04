# D04 — Novelty assessment

## Known territory
- Sentence-level retrieval is well-studied (SBERT, ColBERT-late-pool).
- Wh-question typing is decades old (Li & Roth 2002).
- Multi-hop decomposition (Self-Ask, DSP, HippoRAG) is established
  but operates query-side or entity-graph-side.

## Partially novel territory
- Combining (a) sentence-level dense retrieval with (b) explicit
  claim-type typing of evidence atoms with (c) a typed-bonus in the
  retrieval score is a compact arrangement we have not seen presented
  as a single method.
- The motivation — *closing the chunk-coarseness gap exposed by an
  atomic bottleneck-decomposition diagnostic* — is itself a novel
  framing. Prior work proposes deblending without first **proving**
  via measurement that chunk-level scoring is the bottleneck.

## Potentially novel territory
- **Type-anchored atom-SNR.** Compute SNR not over the whole atom
  embedding population but only over atoms of the *same predicted
  claim type*. A WHEN-question ranks DATE atoms against the DATE-atom
  background, not all atoms. This is a per-type calibration.
- **Atom-level diversity selection.** Instead of greedy by SNR,
  enforce that the selected set covers the diverse set of typed atoms
  needed by the query (one entity atom, one date atom, etc.). This
  bridges D04 with D06.

## Falsification

D04 is **not novel** if a vanilla sentence-level dense retriever
(SBERT) on HotpotQA-1k achieves the same F1 with the same selector.
We will run that as a head-to-head ablation.

D04 is **only useful** if the typed-bonus or per-type-SNR
gives a measurable F1 / cit_acc gain over plain sentence retrieval.
If the typed bonus is null, the contribution shrinks to "sentence
retrieval, applied", which is not novel.
