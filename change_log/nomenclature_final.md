# Nomenclature: final decision (2026-05-03)

## Rule

**Use plain accessible names for data units; reserve physics names for
mechanisms where they do real mathematical work.**

This avoids both extremes:
- jargon overhead (hadron / baryon / flavor quantum number — fails the
  Nature MI accessibility test)
- fake-physics labels (polished names that don't bring math content —
  the earlier "Path-Integral Retrieval", "Graph-Walk Retrieval" failure
  pattern)

## Data units (plain names)

| Pipeline unit | Name | Note |
|---|---|---|
| single-claim sentence | **atom** | "atomic decomposition" already standard in math/physics for "decomposition into smallest units"; everyday English meaning of "atomic" = irreducible |
| chunk (~3 atoms) | **chunk** or **molecule** | use "chunk" in technical sections, "molecule" only when emphasizing bound-atoms structure |
| document | **document** | no analogy needed |
| corpus | corpus / system | — |
| claim_type | **claim type** | not flavor quantum number |

Keep code identifiers (`atoms_dir`, `atom_embs.npy`, `atom_id`,
`claim_type`) — already consistent with this naming and avoids
breaking the queued SLURM jobs.

## Method-level names (physics where it earns its keep)

These get physics names because the math is genuinely the physics:

| Method | Physics name | Why it earns the name |
|---|---|---|
| score-gated agglomerative selection | **anti-kT clustering** | literally uses Cacciari-Salam-Soyez 2008 distance kernel + score-gated α |
| 3-scale retrieval with kernel flow | **renormalization-group retrieval** | the metric transforms with scale; that IS RG |
| selection with score+coverage Lagrange multipliers | **maximum-entropy selection** | Jaynes 1957; multipliers from constraint dual |
| K-replica prompt-ordering | **multi-event averaging** | accessible physics intuition |
| similarity score | **scattering amplitude** (informal) | direct analog, accessible |

Methods that did NOT earn a physics name (failed methods that we
called physics-inspired without the math content):

- "Path-Integral Retrieval" → just truncated random walk / Katz centrality
- "Graph-Walk Retrieval" → 1-hop expansion / max-product
- "CLEAN-RAG" → kept the name only because it's a literal port of
  Hogbom 1974 deconvolution (failed — used as cautionary transfer)
- "Lock-in coherent paraphrase" → kept the name for similar reasons
  (failed — instructive transfer)
- "Evidence-SNR atoms" → just z-scoring, drop the SNR framing

## Title plan

*"Atomic decomposition of retrieval-augmented generation: a
cross-disciplinary methods framework for multi-hop reasoning"*

- "Atomic decomposition" reads as "decomposition into smallest units"
  to a non-physicist (which is correct math/physics meaning), AND
  matches our code, AND signals our methodology.
- "cross-disciplinary methods framework" tells the reader they're
  getting physics + IR + OR methods, each at the right pipeline stage.
- "multi-hop reasoning" anchors the dataset and problem.

## Falsifier of the framing itself

If on the final results, no physics-named method beats a non-physics
baseline (PRF — IR; submodular set-cover — OR), then the physics
framing was decorative. The cross-disciplinary frame still holds, but
"physics-inspired" disappears from the title.

We'll know once the queued sweeps land.
