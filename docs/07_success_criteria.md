# Success Criteria

Minimum success:
- reproduce BM25 and dense retrieval baselines
- produce reliable logs and metrics
- show at least one dataset with better context precision

Strong success:
- better faithfulness with fewer tokens
- lower p95 latency at same answer quality
- better Recall@k or NDCG after SNR reranking

Breakthrough-level success:
- consistently beats dense+reranker or strong hybrid baselines
- improves quality-cost Pareto frontier
- survives ablations and sanity checks
- works beyond one dataset

Never optimize only one metric.
