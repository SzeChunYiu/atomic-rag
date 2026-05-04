# Evaluation synthetic_irc_120_greedy

## Metrics

- answer_count: 30.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.020000
- citation_accuracy_mean: 0.166667
- conservation_R_entity_mean: 0.833333
- conservation_R_numeric_mean: 0.983333
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.394444
- evaluate_seconds: 0.000203
- mean_reciprocal_rank_doc: 0.244182
- ndcg@1_chunk_mean: 0.166667
- ndcg@3_chunk_mean: 0.166667
- ndcg@5_chunk_mean: 0.200084
- precision@1_chunk_mean: 0.166667
- precision@3_chunk_mean: 0.066667
- precision@5_chunk_mean: 0.066667
- recall@1_doc_mean: 0.166667
- recall@3_doc_mean: 0.166667
- recall@5_doc_mean: 0.233333
- retrieve_seconds: 0.010086

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to eb65579ae64c`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
