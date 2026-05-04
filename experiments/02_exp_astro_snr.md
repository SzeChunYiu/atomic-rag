# Experiment: Astro-SNR Reranking

Goal:
Test whether evidence-SNR beats raw similarity ranking.

Run:
1. retrieve top-100 dense candidates
2. compute local background
3. rerank by SNR
4. compare top-k against raw top-k

Measure:
- context precision
- gold evidence hit rate
- false positive rate
- score distribution

Sanity:
Check that SNR is not only rewarding short chunks.
