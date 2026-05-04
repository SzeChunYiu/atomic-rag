# Local-Background Atom Detectability

## Idea
Raw `cos(q, atom)` is calibrated against the *global* corpus distribution. A bridge atom can have low absolute cosine yet be the brightest atom *within its local neighborhood* (same claim type, same entity context, same embedding annulus). Score atoms against *comparable* backgrounds, not against the whole corpus.

## Backgrounds

| Symbol | Atom set | Question it answers |
|--------|----------|---------------------|
| B_global | all candidate atoms | how does this atom compare to a random atom? |
| B_type | atoms with same claim_type | given we are asking a WHERE question, is this a strong WHERE atom? |
| B_entity | atoms sharing query or candidate entities | among atoms about the same entities, is this the most query-relevant? |
| B_annulus | atoms within an embedding-distance shell of candidate | locally calibrated cosine — does this atom stand out from neighbors? |
| B_te_ann | weighted union of B_type, B_entity, B_annulus | full local-background model |

## Detectability score

```
z_X = (raw_score(q, atom) - mean(raw_score(q, B_X))) / (std(raw_score(q, B_X)) + ε)

detectability = w0 * raw_score
              + w1 * z_type
              + w2 * z_entity
              + w3 * z_annulus
              + w4 * type_match_bonus
              + w5 * path_bridge_bonus
```

Weights set by validation grid on synthetic crowding benchmark, frozen before evaluating on real datasets.

## Ablations
- raw atom cosine only (baseline)
- raw + type bonus
- raw + global SNR
- raw + type SNR
- raw + entity SNR
- raw + annulus SNR
- raw + type/entity/annulus SNR (full)

## Falsification
If raw atom cosine performs as well as `B_te_ann` detectability on the synthetic benchmark, the detector is not novel enough — we drop the local-background framing and the contribution shrinks to plain sentence retrieval, which is already known prior art.

## Implementation
- `src/astro_cs_rag/detection/atom_background.py` — compute B_global, B_type, B_entity, B_annulus statistics.
- `src/astro_cs_rag/detection/atom_detectability.py` — combine into z-scores and final detectability.
- `src/astro_cs_rag/retrieval/deblend_retrieve.py` — atom-level retrieval that uses detectability as the rank score.

Implement only after the gold-atom audit and crowding benchmark prove the failure exists. Order matters; don't ship a detector before the diagnostic.
