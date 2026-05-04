# Evaluation hotpotqa_1k__fusion_rrf__anti_kt

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.096444
- citation_accuracy_mean: 0.834000
- conservation_R_entity_mean: 0.465944
- conservation_R_numeric_mean: 0.446417
- conservation_R_temporal_mean: 0.001000
- conservation_faithfulness_mean: 0.695546
- evaluate_seconds: 0.024317
- mean_reciprocal_rank_doc: 0.889341
- ndcg@10_chunk_mean: 0.869694
- ndcg@1_chunk_mean: 0.828000
- ndcg@3_chunk_mean: 0.885272
- ndcg@5_chunk_mean: 0.885612
- precision@10_chunk_mean: 0.188500
- precision@1_chunk_mean: 0.828000
- precision@3_chunk_mean: 0.483000
- precision@5_chunk_mean: 0.326400
- recall@10_doc_mean: 0.863000
- recall@1_doc_mean: 0.414000
- recall@3_doc_mean: 0.698500
- recall@5_doc_mean: 0.772500
- retrieve_seconds: 137.507048

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 3a779599bae7`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
