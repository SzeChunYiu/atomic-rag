# Sparse RAG and Context Selection

Known direction:
Reducing retrieved context can lower latency and improve focus.

Closest overlap:
Sparse RAG and context-compression methods already select or drop documents.

Our distinction should be:
- evidence atoms, not only documents
- source-detection score before selection
- sparse reconstruction objective
- root-cause traces for every selected atom

Claude task:
Find all papers that use sparse context, document dropping, or evidence selection.
