# Evaluation hotpotqa_1k_cs384_greedy_rerank_qwen7b_cot

## Metrics

- answer_count: 1000.000000
- answer_em_mean: 0.000000
- answer_f1_mean: 0.128481
- citation_accuracy_mean: 0.852200
- conservation_R_entity_mean: 0.569077
- conservation_R_numeric_mean: 0.387441
- conservation_R_temporal_mean: 0.046700
- conservation_faithfulness_mean: 0.665594
- evaluate_seconds: 0.023336
- mean_reciprocal_rank_doc: 0.961938
- ndcg@10_chunk_mean: 0.940014
- ndcg@1_chunk_mean: 0.936000
- ndcg@3_chunk_mean: 0.960292
- ndcg@5_chunk_mean: 0.952913
- precision@10_chunk_mean: 0.227400
- precision@1_chunk_mean: 0.936000
- precision@3_chunk_mean: 0.627000
- precision@5_chunk_mean: 0.418600
- recall@10_doc_mean: 0.930500
- recall@1_doc_mean: 0.468000
- recall@3_doc_mean: 0.867000
- recall@5_doc_mean: 0.907000
- retrieve_seconds: 12.810613

## Reproduction

- `python -m pip install -e '.[dev]'`
- `rag-run-benchmark --config configs/benchmark.yaml`
- `rag-evaluate --config configs/eval.yaml  # set run_dir to d5656d4127c9`

## Notes

- Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.
- Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.
