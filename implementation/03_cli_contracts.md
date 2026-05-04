# CLI Contracts

Required commands:

```text
rag-build-index --config configs/index.yaml
rag-retrieve --config configs/retrieve.yaml
rag-detect --config configs/detect.yaml
rag-select --config configs/select.yaml
rag-evaluate --config configs/eval.yaml
rag-run-benchmark --config configs/benchmark.yaml
rag-sanity-check --run runs/<id>
```

Every CLI should:
- load config
- validate paths
- write artifacts
- fail loudly on schema mismatch
