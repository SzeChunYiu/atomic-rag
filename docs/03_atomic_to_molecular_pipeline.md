# Atoms to Molecules to Polymers

This project should grow in layers.

Atom level:
- query facet
- evidence atom
- score component
- single sanity check

Molecule level:
- candidate generator
- background estimator
- SNR detector
- sparse selector
- reranker
- evaluator

Polymer level:
- Astro-RAG pipeline
- CS-RAG pipeline
- Astro-CS-RAG combined system
- benchmark harness
- final search algorithm

Engineering rule:
Never build polymer logic before atomic outputs are visible and testable.
