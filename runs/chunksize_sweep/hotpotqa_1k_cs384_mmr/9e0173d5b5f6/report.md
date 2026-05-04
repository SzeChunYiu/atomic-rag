# Evaluation hotpotqa_1k_cs384_mmr

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.013597
- citation_accuracy_mean: 0.114397
- conservation_R_entity_mean: 0.411328
- conservation_R_numeric_mean: 0.472103
- conservation_R_temporal_mean: 0.288383
- conservation_faithfulness_mean: 0.609395
- evaluate_seconds: 0.024400
- mean_reciprocal_rank_doc: 0.929614
- ndcg@10_chunk_mean: 0.906255
- ndcg@1_chunk_mean: 0.886000
- ndcg@3_chunk_mean: 0.927603
- ndcg@5_chunk_mean: 0.925785
- precision@10_chunk_mean: 0.220500
- precision@1_chunk_mean: 0.886000
- precision@3_chunk_mean: 0.572333
- precision@5_chunk_mean: 0.390000
- recall@10_doc_mean: 0.889000
- recall@1_doc_mean: 0.443000
- recall@3_doc_mean: 0.771500
- recall@5_doc_mean: 0.835500
- retrieve_seconds: 13.793756

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 9e0173d5b5f6`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
