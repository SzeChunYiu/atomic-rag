# Sanity Check Protocol

Before trusting a result, check:

1. Dataset leakage
2. Duplicate documents
3. Gold evidence present in corpus
4. Query-answer mismatch
5. Score distribution shape
6. Top selected chunks are readable
7. SNR is not only length bias
8. Sparse selector is not selecting empty/generic text
9. Generator did not ignore evidence
10. Improvement appears across seeds

If a benchmark improves, recursively find the root cause.
