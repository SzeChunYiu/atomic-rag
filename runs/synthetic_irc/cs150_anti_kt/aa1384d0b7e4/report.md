# Evaluation synthetic_irc_150_anti_kt

## Metrics

- answer_count: 30.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.000000
- citation_accuracy_mean: 0.033333
- conservation_R_entity_mean: 0.488889
- conservation_R_numeric_mean: 1.000000
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.503704
- evaluate_seconds: 0.000196
- mean_reciprocal_rank_doc: 0.128434
- ndcg@1_chunk_mean: 0.033333
- ndcg@3_chunk_mean: 0.050000
- ndcg@5_chunk_mean: 0.064356
- precision@1_chunk_mean: 0.033333
- precision@3_chunk_mean: 0.022222
- precision@5_chunk_mean: 0.020000
- recall@1_doc_mean: 0.033333
- recall@3_doc_mean: 0.066667
- recall@5_doc_mean: 0.100000
- retrieve_seconds: 0.010010

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to aa1384d0b7e4`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
