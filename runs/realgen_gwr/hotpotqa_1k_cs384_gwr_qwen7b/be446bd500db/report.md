# Evaluation hotpotqa_1k_cs384_gwr_qwen7b_terse

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.398226
- citation_accuracy_mean: 0.792936
- conservation_R_entity_mean: 0.691354
- conservation_R_numeric_mean: 0.441568
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.622360
- evaluate_seconds: 0.024224
- mean_reciprocal_rank_doc: 0.929562
- ndcg@10_chunk_mean: 0.905856
- ndcg@1_chunk_mean: 0.886000
- ndcg@3_chunk_mean: 0.927522
- ndcg@5_chunk_mean: 0.925278
- precision@10_chunk_mean: 0.221300
- precision@1_chunk_mean: 0.886000
- precision@3_chunk_mean: 0.572667
- precision@5_chunk_mean: 0.391400
- recall@10_doc_mean: 0.889000
- recall@1_doc_mean: 0.443000
- recall@3_doc_mean: 0.772000
- recall@5_doc_mean: 0.836500
- retrieve_seconds: 13.231101

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to be446bd500db`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
