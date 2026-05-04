# Evaluation hotpotqa_1k__dense__anti_kt

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.096752
- citation_accuracy_mean: 0.886000
- conservation_R_entity_mean: 0.471472
- conservation_R_numeric_mean: 0.438167
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.696787
- evaluate_seconds: 0.024652
- mean_reciprocal_rank_doc: 0.931554
- ndcg@10_chunk_mean: 0.908572
- ndcg@1_chunk_mean: 0.886000
- ndcg@3_chunk_mean: 0.933744
- ndcg@5_chunk_mean: 0.927435
- precision@10_chunk_mean: 0.202500
- precision@1_chunk_mean: 0.886000
- precision@3_chunk_mean: 0.547667
- precision@5_chunk_mean: 0.363200
- recall@10_doc_mean: 0.895500
- recall@1_doc_mean: 0.443000
- recall@3_doc_mean: 0.779000
- recall@5_doc_mean: 0.842000
- retrieve_seconds: 63.621976

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to fda10de23135`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
