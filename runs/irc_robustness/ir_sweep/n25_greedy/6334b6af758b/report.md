# Evaluation irc_120_greedy

## Metrics

- answer_count: 3.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.000000
- citation_accuracy_mean: 0.666667
- conservation_R_entity_mean: 0.333333
- conservation_R_numeric_mean: 0.000000
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.888889
- evaluate_seconds: 0.000032
- mean_reciprocal_rank_doc: 0.833333
- ndcg@1_chunk_mean: 0.666667
- ndcg@3_chunk_mean: 0.876977
- ndcg@5_chunk_mean: 0.876977
- precision@1_chunk_mean: 0.666667
- precision@3_chunk_mean: 0.333333
- precision@5_chunk_mean: 0.200000
- recall@1_doc_mean: 0.666667
- recall@3_doc_mean: 1.000000
- recall@5_doc_mean: 1.000000
- retrieve_seconds: 0.001286

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 6334b6af758b`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
