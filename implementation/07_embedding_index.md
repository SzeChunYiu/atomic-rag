# Embedding and Index Module

Responsibilities:
- create embeddings
- save vectors
- build ANN index
- load existing index
- query top-N candidates

Keep index code separate from detector code.

Required outputs:
- vector file
- index file
- embedding manifest
- retrieval latency stats

Baseline first:
Use simple dense retrieval before ANN optimization.
