# Experiment: Large Scale

Goal:
Test speed and memory under larger corpora.

Only run after small experiments are stable.

Measure:
- p50 latency
- p95 latency
- index size
- memory use
- candidates processed
- reranker calls
- tokens sent to LLM

Check:
Does SNR computation become bottleneck?
If yes, optimize background estimation.
