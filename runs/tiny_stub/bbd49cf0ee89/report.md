# Evaluation tiny_stub

## Metrics

- answer_count: 3.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.000000
- citation_accuracy_mean: 1.000000
- evaluate_seconds: 0.000033
- mean_reciprocal_rank_doc: 1.000000
- ndcg@1_chunk_mean: 1.000000
- ndcg@3_chunk_mean: 1.000000
- precision@1_chunk_mean: 1.000000
- precision@3_chunk_mean: 0.333333
- recall@1_doc_mean: 1.000000
- recall@3_doc_mean: 1.000000
- retrieve_seconds: 0.001098

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to bbd49cf0ee89`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
