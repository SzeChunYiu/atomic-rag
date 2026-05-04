# Run Protocol

Standard run stages:
1. load config
2. load dataset
3. build or load index
4. run candidate retrieval
5. run detector
6. run sparse selector
7. optionally rerank
8. generate answer
9. evaluate
10. write artifacts
11. run sanity checks

Every stage must emit inspectable outputs.
If a stage cannot be inspected, it is not trusted.
