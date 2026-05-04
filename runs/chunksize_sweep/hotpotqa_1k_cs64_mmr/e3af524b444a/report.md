# Evaluation hotpotqa_1k_cs64_mmr

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.006752
- citation_accuracy_mean: 0.086940
- conservation_R_entity_mean: 0.531674
- conservation_R_numeric_mean: 0.679305
- conservation_R_temporal_mean: 0.357162
- conservation_faithfulness_mean: 0.477286
- evaluate_seconds: 0.024987
- mean_reciprocal_rank_doc: 0.693652
- ndcg@10_chunk_mean: 0.719280
- ndcg@1_chunk_mean: 0.555000
- ndcg@3_chunk_mean: 0.690798
- ndcg@5_chunk_mean: 0.717648
- precision@10_chunk_mean: 0.247600
- precision@1_chunk_mean: 0.555000
- precision@3_chunk_mean: 0.412000
- precision@5_chunk_mean: 0.339400
- recall@10_doc_mean: 0.753000
- recall@1_doc_mean: 0.277500
- recall@3_doc_mean: 0.508000
- recall@5_doc_mean: 0.614000
- retrieve_seconds: 90.518782

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to e3af524b444a`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
