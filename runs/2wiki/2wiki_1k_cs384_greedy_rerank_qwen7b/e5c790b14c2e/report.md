# Evaluation 2wiki_1k_cs384_greedy_rerank_terse

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.207085
- citation_accuracy_mean: 0.645323
- conservation_R_entity_mean: 0.753759
- conservation_R_numeric_mean: 0.507008
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.579744
- evaluate_seconds: 0.023390
- mean_reciprocal_rank_doc: 0.989033
- ndcg@10_chunk_mean: 0.964270
- ndcg@1_chunk_mean: 0.980000
- ndcg@3_chunk_mean: 0.988699
- ndcg@5_chunk_mean: 0.979254
- precision@10_chunk_mean: 0.217000
- precision@1_chunk_mean: 0.980000
- precision@3_chunk_mean: 0.589000
- precision@5_chunk_mean: 0.392800
- recall@10_doc_mean: 0.729750
- recall@1_doc_mean: 0.430000
- recall@3_doc_mean: 0.670750
- recall@5_doc_mean: 0.699500
- retrieve_seconds: 7.239866

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to e5c790b14c2e`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
