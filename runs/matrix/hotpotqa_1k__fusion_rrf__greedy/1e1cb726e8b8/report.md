# Evaluation hotpotqa_1k__fusion_rrf__greedy

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.099473
- citation_accuracy_mean: 0.828000
- conservation_R_entity_mean: 0.464765
- conservation_R_numeric_mean: 0.446417
- conservation_R_temporal_mean: 0.001000
- conservation_faithfulness_mean: 0.695939
- evaluate_seconds: 0.023490
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
- retrieve_seconds: 139.834206

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 1e1cb726e8b8`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
