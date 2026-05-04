# Evaluation hotpotqa_1k_cs256_v4a0_5

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.009475
- citation_accuracy_mean: 0.098468
- conservation_R_entity_mean: 0.393181
- conservation_R_numeric_mean: 0.425638
- conservation_R_temporal_mean: 0.382057
- conservation_faithfulness_mean: 0.599708
- evaluate_seconds: 0.024929
- mean_reciprocal_rank_doc: 0.925337
- ndcg@10_chunk_mean: 0.897421
- ndcg@1_chunk_mean: 0.876000
- ndcg@3_chunk_mean: 0.925906
- ndcg@5_chunk_mean: 0.919950
- precision@10_chunk_mean: 0.248100
- precision@1_chunk_mean: 0.876000
- precision@3_chunk_mean: 0.587333
- precision@5_chunk_mean: 0.419000
- recall@10_doc_mean: 0.876500
- recall@1_doc_mean: 0.438000
- recall@3_doc_mean: 0.757500
- recall@5_doc_mean: 0.817000
- retrieve_seconds: 18.432201

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 027a41ebdb55`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
