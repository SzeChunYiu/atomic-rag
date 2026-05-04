# Evaluation hotpotqa_1k_cs128_anti_kt

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.006171
- citation_accuracy_mean: 0.081401
- conservation_R_entity_mean: 0.418468
- conservation_R_numeric_mean: 0.548213
- conservation_R_temporal_mean: 0.434335
- conservation_faithfulness_mean: 0.532994
- evaluate_seconds: 0.024337
- mean_reciprocal_rank_doc: 0.858189
- ndcg@10_chunk_mean: 0.845119
- ndcg@1_chunk_mean: 0.772000
- ndcg@3_chunk_mean: 0.865341
- ndcg@5_chunk_mean: 0.864580
- precision@10_chunk_mean: 0.276400
- precision@1_chunk_mean: 0.772000
- precision@3_chunk_mean: 0.558000
- precision@5_chunk_mean: 0.429200
- recall@10_doc_mean: 0.849000
- recall@1_doc_mean: 0.386000
- recall@3_doc_mean: 0.678500
- recall@5_doc_mean: 0.769000
- retrieve_seconds: 35.948725

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 07dabb9d8c67`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
