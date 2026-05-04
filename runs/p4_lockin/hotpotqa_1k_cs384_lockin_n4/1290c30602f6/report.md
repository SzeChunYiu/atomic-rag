# Evaluation hotpotqa_1k_cs384_lockin_n4

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.015071
- citation_accuracy_mean: 0.112424
- conservation_R_entity_mean: 0.425056
- conservation_R_numeric_mean: 0.516589
- conservation_R_temporal_mean: 0.259930
- conservation_faithfulness_mean: 0.599475
- evaluate_seconds: 0.029326
- mean_reciprocal_rank_doc: 0.855301
- ndcg@10_chunk_mean: 0.836507
- ndcg@1_chunk_mean: 0.782000
- ndcg@3_chunk_mean: 0.853378
- ndcg@5_chunk_mean: 0.852939
- precision@10_chunk_mean: 0.198900
- precision@1_chunk_mean: 0.782000
- precision@3_chunk_mean: 0.472000
- precision@5_chunk_mean: 0.332800
- recall@10_doc_mean: 0.838500
- recall@1_doc_mean: 0.391000
- recall@3_doc_mean: 0.656000
- recall@5_doc_mean: 0.740000
- retrieve_seconds: 389.850131

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 1290c30602f6`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
