# Evaluation 2wiki_1k_cs384_v4a0_7_terse

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.217074
- citation_accuracy_mean: 0.649698
- conservation_R_entity_mean: 0.756474
- conservation_R_numeric_mean: 0.501156
- conservation_R_temporal_mean: 0.000000
- conservation_faithfulness_mean: 0.580790
- evaluate_seconds: 0.023692
- mean_reciprocal_rank_doc: 0.980783
- ndcg@10_chunk_mean: 0.953918
- ndcg@1_chunk_mean: 0.965000
- ndcg@3_chunk_mean: 0.979223
- ndcg@5_chunk_mean: 0.969956
- precision@10_chunk_mean: 0.216700
- precision@1_chunk_mean: 0.965000
- precision@3_chunk_mean: 0.580667
- precision@5_chunk_mean: 0.388600
- recall@10_doc_mean: 0.729500
- recall@1_doc_mean: 0.423250
- recall@3_doc_mean: 0.666250
- recall@5_doc_mean: 0.699750
- retrieve_seconds: 7.295742

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to 4d74b6f373e1`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
