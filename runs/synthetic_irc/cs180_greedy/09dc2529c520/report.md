# Evaluation synthetic_irc_180_greedy

## Metrics

- answer_count: 30.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.007407
- citation_accuracy_mean: 0.066667
- conservation_R_entity_mean: 0.500000
- conservation_R_numeric_mean: 1.000000
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.500000
- evaluate_seconds: 0.000193
- mean_reciprocal_rank_doc: 0.169479
- ndcg@1_chunk_mean: 0.066667
- ndcg@3_chunk_mean: 0.083333
- ndcg@5_chunk_mean: 0.137835
- precision@1_chunk_mean: 0.066667
- precision@3_chunk_mean: 0.033333
- precision@5_chunk_mean: 0.046667
- recall@1_doc_mean: 0.066667
- recall@3_doc_mean: 0.100000
- recall@5_doc_mean: 0.233333
- retrieve_seconds: 0.007219

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 09dc2529c520`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
