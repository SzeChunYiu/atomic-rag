# Evaluation hotpotqa_1k_cs384_fusion_rrf_baseline

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.015766
- citation_accuracy_mean: 0.115782
- conservation_R_entity_mean: 0.431564
- conservation_R_numeric_mean: 0.521124
- conservation_R_temporal_mean: 0.247747
- conservation_faithfulness_mean: 0.599855
- evaluate_seconds: 0.023933
- mean_reciprocal_rank_doc: 0.889560
- ndcg@10_chunk_mean: 0.869156
- ndcg@1_chunk_mean: 0.825000
- ndcg@3_chunk_mean: 0.887090
- ndcg@5_chunk_mean: 0.886088
- precision@10_chunk_mean: 0.200500
- precision@1_chunk_mean: 0.825000
- precision@3_chunk_mean: 0.494333
- precision@5_chunk_mean: 0.344400
- recall@10_doc_mean: 0.857500
- recall@1_doc_mean: 0.412500
- recall@3_doc_mean: 0.691500
- recall@5_doc_mean: 0.774000
- retrieve_seconds: 90.990647

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to c34f072d9a3a`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
