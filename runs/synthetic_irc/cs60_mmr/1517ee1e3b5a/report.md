# Evaluation synthetic_irc_60_mmr

## Metrics

- answer_count: 30.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.005556
- citation_accuracy_mean: 0.033333
- conservation_R_entity_mean: 0.950000
- conservation_R_numeric_mean: 0.333333
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.572222
- evaluate_seconds: 0.000200
- mean_reciprocal_rank_doc: 0.266202
- ndcg@1_chunk_mean: 0.100000
- ndcg@3_chunk_mean: 0.200593
- ndcg@5_chunk_mean: 0.261541
- precision@1_chunk_mean: 0.100000
- precision@3_chunk_mean: 0.111111
- precision@5_chunk_mean: 0.106667
- recall@1_doc_mean: 0.100000
- recall@3_doc_mean: 0.266667
- recall@5_doc_mean: 0.433333
- retrieve_seconds: 0.012893

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 1517ee1e3b5a`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
