# Package Layout

```text
src/astro_cs_rag/
  atoms/schemas.py
  data/loaders.py
  chunking/splitters.py
  indexing/dense.py
  indexing/bm25.py
  retrieval/candidates.py
  detection/background.py
  detection/snr.py
  selection/coverage.py
  selection/greedy.py
  evaluation/metrics.py
  diagnostics/sanity.py
  reporting/reports.py
```

Split further if any file nears 220 lines.
