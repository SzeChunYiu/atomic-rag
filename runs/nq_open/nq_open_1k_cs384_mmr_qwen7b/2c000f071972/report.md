# Evaluation nq_open_1k_cs384_mmr

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.714354
- citation_accuracy_mean: 0.960220
- conservation_R_entity_mean: 0.789276
- conservation_R_numeric_mean: 0.416007
- conservation_R_temporal_mean: 0.001333
- conservation_faithfulness_mean: 0.597795
- evaluate_seconds: 0.020564
- mean_reciprocal_rank_doc: 0.976543
- ndcg@10_chunk_mean: 0.978012
- ndcg@1_chunk_mean: 0.974000
- ndcg@3_chunk_mean: 0.975131
- ndcg@5_chunk_mean: 0.976335
- precision@10_chunk_mean: 0.098400
- precision@1_chunk_mean: 0.974000
- precision@3_chunk_mean: 0.325333
- precision@5_chunk_mean: 0.195800
- recall@10_doc_mean: 0.984000
- recall@1_doc_mean: 0.974000
- recall@3_doc_mean: 0.976000
- recall@5_doc_mean: 0.979000
- retrieve_seconds: 1.897409

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 2c000f071972`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
