# Evaluation synthetic_irc_90_mmr

## Metrics

- answer_count: 30.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.028148
- citation_accuracy_mean: 0.066667
- conservation_R_entity_mean: 0.983333
- conservation_R_numeric_mean: 0.350000
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.555556
- evaluate_seconds: 0.000204
- mean_reciprocal_rank_doc: 0.339305
- ndcg@1_chunk_mean: 0.166667
- ndcg@3_chunk_mean: 0.293750
- ndcg@5_chunk_mean: 0.349436
- precision@1_chunk_mean: 0.166667
- precision@3_chunk_mean: 0.144444
- precision@5_chunk_mean: 0.126667
- recall@1_doc_mean: 0.166667
- recall@3_doc_mean: 0.400000
- recall@5_doc_mean: 0.533333
- retrieve_seconds: 0.009806

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 2e5ef53cd092`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
