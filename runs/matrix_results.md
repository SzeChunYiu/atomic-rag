# Experiment-matrix results

| dataset | retriever | selector | recall@1_doc_mean | recall@5_doc_mean | recall@10_doc_mean | answer_em_mean | answer_f1_mean | citation_accuracy_mean |
|---|---|---|---|---|---|---|---|---|
| hotpotqa_1k | dense | anti_kt | 0.443 | 0.842 | 0.895 | 0.000 | 0.097 | 0.886 |
| hotpotqa_1k | dense | greedy | 0.443 | 0.842 | 0.895 | 0.000 | 0.097 | 0.886 |
| hotpotqa_1k | fusion_rrf | anti_kt | 0.414 | 0.772 | 0.863 | 0.000 | 0.096 | 0.834 |
| hotpotqa_1k | fusion_rrf | greedy | 0.414 | 0.772 | 0.863 | 0.000 | 0.099 | 0.828 |
| hotpotqa_1k | fusion_rrf | mmr | 0.414 | 0.772 | 0.863 | 0.000 | 0.022 | 0.129 |
| tiny | bm25 | anti_kt | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
