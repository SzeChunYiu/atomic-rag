# Calorimetric query archetype routing

**Status:** stub (P1 deliverable — profiler implemented, archetypes identified).

## 1. Method one-liner
Classify each query by the morphology of its score field over the corpus
(compact / plateau / bimodal / diffuse / noisy), then route different
retrieval recipes to different archetypes.

## 2. Physics analog (operator-level)
Calorimeter shower-shape classification. Electromagnetic showers are compact
and longitudinal; hadronic showers are extended and irregular. The shape —
not the total energy — distinguishes particle types.

## 3. Closest prior art
- Adaptive retrieval (Self-RAG, FLARE): retrieves on demand, but routes by
  generation uncertainty, not by score-field morphology.
- Query difficulty prediction (Carmel & Yom-Tov 2010): predicts hard/easy via
  query features, not score-field shape.
- Mixture-of-retrievers / router-style architectures: route by query type,
  but do not formalize the routing signal as a calorimetric shape.

## 4. Novelty estimate
- algorithmic: medium-high (score-field morphology is a new signal).
- theoretical: medium.
- empirical: high — the per-archetype baseline gap (P1 product) is itself an
  artifact the field would adopt regardless of our method.

## 5. Why publishable
The archetype map is a diagnostic with independent value. Routing is an
additive method on top of the diagnostic.

## 6. Falsification protocol
If archetype labels do not predict baseline failure modes (mutual information
< some threshold against F1–F9), we drop routing and keep only the diagnostic.

## 7. Status
- [x] profiler implemented (`src/astro_cs_rag/diagnostics/calorimetry.py`)
- [ ] archetype-to-failure mapping verified empirically
- [ ] per-archetype routing recipe written
